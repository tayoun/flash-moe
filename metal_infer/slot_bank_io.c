/**
 * slot_bank_io.c
 *
 * Slot bank + async pread I/O for Flash-MOE expert loading.
 * Ported from anemll-flash-mlx (https://github.com/Anemll/anemll-flash-mlx)
 * Adapted for Qwen3.5-122B-A10B-4bit.
 *
 * Architecture:
 * - 8-thread pthread pool for async pread operations
 * - Slot bank: expert_id -> slot_id mapping with LRU victim selection
 * - Per-layer files: layer_00.bin, layer_01.bin, ...
 * - Page-aligned pre-allocated buffers (not mmap)
 *
 * Key invariant: slot bank ownership is layer-scoped.
 * Each layer has its own set of slot_bank_size slots.
 */

#include "slot_bank_io.h"
#include <errno.h>
#include <fcntl.h>
#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

/* Thread pool configuration */
#define IO_THREADS 8
#define MAX_CACHE_IO_SPLIT 8
#define MAX_LAYERS 512
#define PAGE_BYTES (16 * 1024)  /* 16 KB pages */

/* ─────────────────────────────────────────────────────────── */
#pragma mark - Pread Task

typedef struct FlashMoePreadTask {
    int fd;
    void* dst;
    off_t offset;
    size_t size;
    ssize_t result;
} FlashMoePreadTask;

#pragma mark - Thread Pool

typedef struct FlashMoeIOThreadPool {
    pthread_t threads[IO_THREADS];
    pthread_mutex_t mutex;
    pthread_cond_t work_ready;
    pthread_cond_t work_done;
    FlashMoePreadTask* tasks;
    int num_tasks;
    int tasks_completed;
    int generation;
    int completed_generation;
    int shutdown;
    int initialized;
} FlashMoeIOThreadPool;

typedef struct FlashMoeExpertLoader FlashMoeExpertLoader;

typedef struct {
    FlashMoeIOThreadPool* pool;
    int thread_id;
} FlashMoeWorkerArg;

static void flash_moe_io_pool_shutdown(FlashMoeIOThreadPool* pool);

static void* flash_moe_io_worker_with_pool(void* arg) {
    FlashMoeWorkerArg* worker = (FlashMoeWorkerArg*)arg;
    FlashMoeIOThreadPool* pool = worker->pool;
    int tid = worker->thread_id;
    int my_generation = 0;

    pthread_mutex_lock(&pool->mutex);
    while (1) {
        while (pool->generation == my_generation && !pool->shutdown) {
            pthread_cond_wait(&pool->work_ready, &pool->mutex);
        }
        if (pool->shutdown) {
            break;
        }

        my_generation = pool->generation;
        int num_tasks = pool->num_tasks;
        FlashMoePreadTask* tasks = pool->tasks;
        pthread_mutex_unlock(&pool->mutex);

        for (int i = tid; i < num_tasks; i += IO_THREADS) {
            FlashMoePreadTask* task = &tasks[i];
            task->result = pread(task->fd, task->dst, task->size, task->offset);
        }

        pthread_mutex_lock(&pool->mutex);
        pool->tasks_completed++;
        if (pool->tasks_completed == IO_THREADS) {
            pool->completed_generation = my_generation;
            pthread_cond_signal(&pool->work_done);
        }
    }
    pthread_mutex_unlock(&pool->mutex);
    return NULL;
}

static int flash_moe_io_pool_init(FlashMoeIOThreadPool* pool) {
    if (pool->initialized) {
        return 0;
    }
    memset(pool, 0, sizeof(*pool));
    if (pthread_mutex_init(&pool->mutex, NULL) != 0) {
        return -1;
    }
    if (pthread_cond_init(&pool->work_ready, NULL) != 0) {
        pthread_mutex_destroy(&pool->mutex);
        return -1;
    }
    if (pthread_cond_init(&pool->work_done, NULL) != 0) {
        pthread_cond_destroy(&pool->work_ready);
        pthread_mutex_destroy(&pool->mutex);
        return -1;
    }

    FlashMoeWorkerArg* worker_args =
        (FlashMoeWorkerArg*)calloc(IO_THREADS, sizeof(FlashMoeWorkerArg));
    if (worker_args == NULL) {
        pthread_cond_destroy(&pool->work_done);
        pthread_cond_destroy(&pool->work_ready);
        pthread_mutex_destroy(&pool->mutex);
        return -1;
    }

    for (int i = 0; i < IO_THREADS; ++i) {
        worker_args[i].pool = pool;
        worker_args[i].thread_id = i;
        if (pthread_create(
                &pool->threads[i],
                NULL,
                flash_moe_io_worker_with_pool,
                &worker_args[i]) != 0) {
            pool->shutdown = 1;
            for (int j = 0; j < i; ++j) {
                pthread_join(pool->threads[j], NULL);
            }
            free(worker_args);
            pthread_cond_destroy(&pool->work_done);
            pthread_cond_destroy(&pool->work_ready);
            pthread_mutex_destroy(&pool->mutex);
            return -1;
        }
    }

    pool->initialized = 1;
    return 0;
}

static int flash_moe_io_pool_start(
    FlashMoeIOThreadPool* pool,
    FlashMoePreadTask* tasks,
    int num_tasks
) {
    if (num_tasks == 0) {
        return 0;
    }
    pthread_mutex_lock(&pool->mutex);
    pool->tasks = tasks;
    pool->num_tasks = num_tasks;
    pool->tasks_completed = 0;
    pool->generation++;
    int my_generation = pool->generation;
    pthread_cond_broadcast(&pool->work_ready);
    pthread_mutex_unlock(&pool->mutex);
    return my_generation;
}

static void flash_moe_io_pool_wait_generation(
    FlashMoeIOThreadPool* pool,
    int generation
) {
    if (generation <= 0) {
        return;
    }
    pthread_mutex_lock(&pool->mutex);
    while (pool->completed_generation < generation) {
        pthread_cond_wait(&pool->work_done, &pool->mutex);
    }
    pthread_mutex_unlock(&pool->mutex);
}

static void flash_moe_io_pool_shutdown(FlashMoeIOThreadPool* pool) {
    if (!pool->initialized) {
        return;
    }
    pthread_mutex_lock(&pool->mutex);
    pool->shutdown = 1;
    pthread_cond_broadcast(&pool->work_ready);
    pthread_mutex_unlock(&pool->mutex);

    for (int i = 0; i < IO_THREADS; ++i) {
        pthread_join(pool->threads[i], NULL);
    }

    pthread_cond_destroy(&pool->work_done);
    pthread_cond_destroy(&pool->work_ready);
    pthread_mutex_destroy(&pool->mutex);
    pool->initialized = 0;
}

#pragma mark - Expert Loader

struct FlashMoeExpertLoader {
    int num_layers;
    size_t expert_size;
    int max_k;
    int cache_io_split;
    int layer_fds[MAX_LAYERS];
    void** slot_buffers;
    FlashMoeIOThreadPool pool;

    /* Slot bank state (enabled via slot_bank_set_size) */
    int slot_bank_size;
    int* slot_bank_owner;          /* [num_layers * slot_bank_size] */
    uint64_t* slot_bank_last_used; /* [num_layers * slot_bank_size] */
    uint64_t* slot_bank_access_counter; /* [num_layers] */

    /* Pending prefetch generation (for async prefetch) */
    int prefetch_generation;
    FlashMoePreadTask* prefetch_tasks;
    int prefetch_task_count;
};

static int flash_moe_active_cache_io_split(
    size_t expert_size,
    int requested_split
) {
    int chunks = requested_split;
    if (chunks < 1) chunks = 1;
    if (chunks > MAX_CACHE_IO_SPLIT) chunks = MAX_CACHE_IO_SPLIT;
    if (expert_size == 0 || (expert_size % PAGE_BYTES) != 0) {
        return 1;
    }
    size_t pages = expert_size / PAGE_BYTES;
    if ((size_t)chunks > pages) {
        chunks = (int)pages;
    }
    if (chunks < 1) chunks = 1;
    return chunks;
}

static void flash_moe_close_all_layer_fds(FlashMoeExpertLoader* loader) {
    for (int i = 0; i < loader->num_layers; ++i) {
        if (loader->layer_fds[i] >= 0) {
            close(loader->layer_fds[i]);
            loader->layer_fds[i] = -1;
        }
    }
}

static void flash_moe_free_slot_buffers(FlashMoeExpertLoader* loader) {
    if (loader->slot_buffers == NULL) {
        return;
    }
    for (int i = 0; i < loader->max_k; ++i) {
        free(loader->slot_buffers[i]);
    }
    free(loader->slot_buffers);
    loader->slot_buffers = NULL;
}

static void flash_moe_free_slot_bank(FlashMoeExpertLoader* loader) {
    free(loader->slot_bank_owner);
    free(loader->slot_bank_last_used);
    free(loader->slot_bank_access_counter);
    free(loader->prefetch_tasks);
    loader->slot_bank_owner = NULL;
    loader->slot_bank_last_used = NULL;
    loader->slot_bank_access_counter = NULL;
    loader->prefetch_tasks = NULL;
    loader->slot_bank_size = 0;
}

static int flash_moe_open_layer_files(
    FlashMoeExpertLoader* loader,
    const char* experts_dir
) {
    for (int i = 0; i < loader->num_layers; ++i) {
        loader->layer_fds[i] = -1;
    }

    for (int layer = 0; layer < loader->num_layers; ++layer) {
        char path[4096];
        snprintf(path, sizeof(path), "%s/layer_%02d.bin", experts_dir, layer);
        int fd = open(path, O_RDONLY);
        if (fd < 0) {
            return -1;
        }
        loader->layer_fds[layer] = fd;
    }
    return 0;
}

static int flash_moe_allocate_slot_buffers(FlashMoeExpertLoader* loader) {
    loader->slot_buffers = (void**)calloc((size_t)loader->max_k, sizeof(void*));
    if (loader->slot_buffers == NULL) {
        return -1;
    }
    for (int i = 0; i < loader->max_k; ++i) {
        void* slot = NULL;
        if (posix_memalign(&slot, PAGE_BYTES, loader->expert_size) != 0) {
            return -1;
        }
        memset(slot, 0, loader->expert_size);
        loader->slot_buffers[i] = slot;
    }
    return 0;
}

/* ─────────────────────────────────────────────────────────── */
#pragma mark - Public API

void* slot_bank_create(
    const char* experts_dir,
    int num_layers,
    size_t expert_size,
    int max_k,
    int cache_io_split
) {
    if (experts_dir == NULL || num_layers <= 0 || num_layers > MAX_LAYERS ||
        expert_size == 0 || max_k <= 0) {
        return NULL;
    }

    FlashMoeExpertLoader* loader =
        (FlashMoeExpertLoader*)calloc(1, sizeof(FlashMoeExpertLoader));
    if (loader == NULL) {
        return NULL;
    }

    loader->num_layers = num_layers;
    loader->expert_size = expert_size;
    loader->max_k = max_k;
    loader->cache_io_split = cache_io_split;
    loader->prefetch_generation = 0;
    loader->prefetch_tasks = NULL;
    loader->prefetch_task_count = 0;

    if (flash_moe_open_layer_files(loader, experts_dir) != 0) {
        flash_moe_close_all_layer_fds(loader);
        free(loader);
        return NULL;
    }
    if (flash_moe_allocate_slot_buffers(loader) != 0) {
        flash_moe_free_slot_buffers(loader);
        flash_moe_close_all_layer_fds(loader);
        free(loader);
        return NULL;
    }
    if (flash_moe_io_pool_init(&loader->pool) != 0) {
        flash_moe_free_slot_buffers(loader);
        flash_moe_close_all_layer_fds(loader);
        free(loader);
        return NULL;
    }

    return loader;
}

void slot_bank_destroy(void* handle) {
    FlashMoeExpertLoader* loader = (FlashMoeExpertLoader*)handle;
    if (loader == NULL) {
        return;
    }
    flash_moe_io_pool_shutdown(&loader->pool);
    flash_moe_free_slot_bank(loader);
    flash_moe_free_slot_buffers(loader);
    flash_moe_close_all_layer_fds(loader);
    free(loader);
}

size_t slot_bank_expert_size(void* handle) {
    FlashMoeExpertLoader* loader = (FlashMoeExpertLoader*)handle;
    if (loader == NULL) {
        return 0;
    }
    return loader->expert_size;
}

void* slot_bank_get_buffer(void* handle, int slot_idx) {
    FlashMoeExpertLoader* loader = (FlashMoeExpertLoader*)handle;
    if (loader == NULL || slot_idx < 0 || slot_idx >= loader->max_k) {
        return NULL;
    }
    return loader->slot_buffers[slot_idx];
}

int slot_bank_set_size(void* handle, int slot_bank_size) {
    FlashMoeExpertLoader* loader = (FlashMoeExpertLoader*)handle;
    if (loader == NULL || slot_bank_size <= 0 || slot_bank_size > loader->max_k) {
        return -1;
    }

    flash_moe_free_slot_bank(loader);

    size_t total_slots = (size_t)loader->num_layers * (size_t)slot_bank_size;
    loader->slot_bank_owner = (int*)malloc(total_slots * sizeof(int));
    loader->slot_bank_last_used = (uint64_t*)calloc(total_slots, sizeof(uint64_t));
    loader->slot_bank_access_counter =
        (uint64_t*)calloc((size_t)loader->num_layers, sizeof(uint64_t));
    if (loader->slot_bank_owner == NULL ||
        loader->slot_bank_last_used == NULL ||
        loader->slot_bank_access_counter == NULL) {
        flash_moe_free_slot_bank(loader);
        return -1;
    }
    for (size_t i = 0; i < total_slots; ++i) {
        loader->slot_bank_owner[i] = -1;
    }
    loader->slot_bank_size = slot_bank_size;
    return 0;
}

void slot_bank_flush(void* handle) {
    FlashMoeExpertLoader* loader = (FlashMoeExpertLoader*)handle;
    if (loader == NULL) {
        return;
    }
    if (loader->slot_bank_owner) {
        size_t total = (size_t)loader->num_layers * (size_t)loader->slot_bank_size;
        for (size_t i = 0; i < total; ++i) {
            loader->slot_bank_owner[i] = -1;
        }
    }
    if (loader->slot_bank_access_counter) {
        for (int i = 0; i < loader->num_layers; ++i) {
            loader->slot_bank_access_counter[i] = 0;
        }
    }
}

#pragma mark - Slot Bank LRU Logic

static int flash_moe_slot_bank_find_owner(
    FlashMoeExpertLoader* loader,
    int layer_index,
    int expert_id
) {
    int base = layer_index * loader->slot_bank_size;
    for (int slot = 0; slot < loader->slot_bank_size; ++slot) {
        if (loader->slot_bank_owner[base + slot] == expert_id) {
            return slot;
        }
    }
    return -1;
}

static void flash_moe_slot_bank_touch(
    FlashMoeExpertLoader* loader,
    int layer_index,
    int slot_id
) {
    uint64_t access = ++loader->slot_bank_access_counter[layer_index];
    loader->slot_bank_last_used[layer_index * loader->slot_bank_size + slot_id] = access;
}

static int flash_moe_slot_bank_choose_victim(
    FlashMoeExpertLoader* loader,
    int layer_index,
    const uint8_t* protected_slots,
    const uint8_t* reserved_slots
) {
    int base = layer_index * loader->slot_bank_size;

    /* First: find an empty slot */
    for (int slot = 0; slot < loader->slot_bank_size; ++slot) {
        if (!reserved_slots[slot] && loader->slot_bank_owner[base + slot] == -1) {
            return slot;
        }
    }

    /* Second: LRU among unprotected, unreserved slots */
    uint64_t best_age = UINT64_MAX;
    int best_slot = -1;
    for (int slot = 0; slot < loader->slot_bank_size; ++slot) {
        if (protected_slots[slot] || reserved_slots[slot]) {
            continue;
        }
        uint64_t age = loader->slot_bank_last_used[base + slot];
        if (age < best_age) {
            best_age = age;
            best_slot = slot;
        }
    }
    return best_slot;
}

#pragma mark - Low-Level Load

static int flash_moe_do_load(
    FlashMoeExpertLoader* loader,
    int layer_index,
    const int* expert_ids,
    int k,
    int* slot_indices,
    void** slot_buffers_out,
    int* miss_count_out
) {
    if (miss_count_out) *miss_count_out = 0;

    int chunks = flash_moe_active_cache_io_split(
        loader->expert_size,
        loader->cache_io_split);
    int num_tasks = k * chunks;
    FlashMoePreadTask* tasks =
        (FlashMoePreadTask*)calloc((size_t)num_tasks, sizeof(FlashMoePreadTask));
    if (tasks == NULL) {
        return -1;
    }

    for (int slot = 0; slot < k; ++slot) {
        size_t total_pages =
            chunks > 1 ? (loader->expert_size / PAGE_BYTES) : 0;
        size_t page_cursor = 0;
        char* dst_base = (char*)loader->slot_buffers[slot];
        off_t expert_offset = (off_t)expert_ids[slot] * (off_t)loader->expert_size;

        for (int chunk = 0; chunk < chunks; ++chunk) {
            size_t chunk_offset = 0;
            size_t chunk_size = loader->expert_size;
            if (chunks > 1) {
                size_t pages_this_chunk = total_pages / (size_t)chunks;
                if ((size_t)chunk < (total_pages % (size_t)chunks)) {
                    pages_this_chunk++;
                }
                chunk_offset = page_cursor * PAGE_BYTES;
                chunk_size = pages_this_chunk * PAGE_BYTES;
                page_cursor += pages_this_chunk;
            }

            int task_index = slot * chunks + chunk;
            tasks[task_index].fd = loader->layer_fds[layer_index];
            tasks[task_index].dst = dst_base + chunk_offset;
            tasks[task_index].offset = expert_offset + (off_t)chunk_offset;
            tasks[task_index].size = chunk_size;
            tasks[task_index].result = 0;
        }
    }

    int generation = flash_moe_io_pool_start(&loader->pool, tasks, num_tasks);
    flash_moe_io_pool_wait_generation(&loader->pool, generation);

    int miss_count = 0;
    for (int slot = 0; slot < k; ++slot) {
        ssize_t total = 0;
        for (int chunk = 0; chunk < chunks; ++chunk) {
            FlashMoePreadTask* task = &tasks[slot * chunks + chunk];
            if (task->result > 0) {
                total += task->result;
            }
        }
        if (total != (ssize_t)loader->expert_size) {
            slot_indices[slot] = -1;
        } else if (slot_buffers_out) {
            slot_buffers_out[slot] = loader->slot_buffers[slot];
        }
    }

    free(tasks);
    return 0;
}

#pragma mark - Single Load

int slot_bank_load_single(void* handle, int layer_idx, int expert_id) {
    FlashMoeExpertLoader* loader = (FlashMoeExpertLoader*)handle;
    if (loader == NULL) {
        return -1;
    }
    if (layer_idx < 0 || layer_idx >= loader->num_layers) {
        return -1;
    }

    int slot_idx = 0;
    int expert_ids[1] = { expert_id };
    int slot_indices[1];
    int miss_count = 0;

    int result = flash_moe_do_load(
        loader, layer_idx, expert_ids, 1, slot_indices, NULL, &miss_count);

    return (result == 0 && slot_indices[0] >= 0) ? 0 : -1;
}

#pragma mark - Slot Bank Load (full algorithm)

int slot_bank_load(
    void* handle,
    int layer_idx,
    const int* expert_ids,
    int k,
    int* slot_ids_out,
    int* miss_slots_out,
    int* miss_expert_ids_out,
    int* miss_count_out
) {
    FlashMoeExpertLoader* loader = (FlashMoeExpertLoader*)handle;
    if (loader == NULL || expert_ids == NULL || slot_ids_out == NULL ||
        miss_slots_out == NULL || miss_expert_ids_out == NULL ||
        miss_count_out == NULL) {
        return -1;
    }
    if (loader->slot_bank_size <= 0 ||
        layer_idx < 0 || layer_idx >= loader->num_layers ||
        k <= 0 || k > loader->slot_bank_size) {
        return -1;
    }

    /* Phase 1: Find hits, collect misses */
    uint8_t* protected_slots =
        (uint8_t*)calloc((size_t)loader->slot_bank_size, sizeof(uint8_t));
    uint8_t* reserved_slots =
        (uint8_t*)calloc((size_t)loader->slot_bank_size, sizeof(uint8_t));
    int* missing_positions = (int*)malloc((size_t)k * sizeof(int));
    int* missing_experts = (int*)malloc((size_t)k * sizeof(int));
    int* missing_slots = (int*)malloc((size_t)k * sizeof(int));
    if (protected_slots == NULL || reserved_slots == NULL ||
        missing_positions == NULL || missing_experts == NULL ||
        missing_slots == NULL) {
        free(protected_slots);
        free(reserved_slots);
        free(missing_positions);
        free(missing_experts);
        free(missing_slots);
        return -1;
    }

    int miss_count = 0;
    for (int i = 0; i < k; ++i) {
        miss_slots_out[i] = -1;
        int slot = flash_moe_slot_bank_find_owner(loader, layer_idx, expert_ids[i]);
        if (slot >= 0) {
            slot_ids_out[i] = slot;
            protected_slots[slot] = 1;
            flash_moe_slot_bank_touch(loader, layer_idx, slot);
        } else {
            slot_ids_out[i] = -1;
            miss_slots_out[i] = -1;
            missing_positions[miss_count] = i;
            missing_experts[miss_count] = expert_ids[i];
            miss_count++;
        }
    }

    /* Phase 2: Choose victim slots for misses */
    for (int m = 0; m < miss_count; ++m) {
        int slot = flash_moe_slot_bank_choose_victim(
            loader, layer_idx, protected_slots, reserved_slots);
        if (slot < 0) {
            free(protected_slots);
            free(reserved_slots);
            free(missing_positions);
            free(missing_experts);
            free(missing_slots);
            return -1;
        }
        reserved_slots[slot] = 1;
        missing_slots[m] = slot;
        slot_ids_out[missing_positions[m]] = slot;
        miss_slots_out[missing_positions[m]] = slot;
        miss_expert_ids_out[m] = missing_experts[m];
    }

    *miss_count_out = miss_count;

    /* Phase 3: Async load misses */
    if (miss_count > 0) {
        int chunks = flash_moe_active_cache_io_split(
            loader->expert_size,
            loader->cache_io_split);
        int num_tasks = miss_count * chunks;
        FlashMoePreadTask* tasks =
            (FlashMoePreadTask*)calloc((size_t)num_tasks, sizeof(FlashMoePreadTask));
        if (tasks == NULL) {
            free(protected_slots);
            free(reserved_slots);
            free(missing_positions);
            free(missing_experts);
            free(missing_slots);
            return -1;
        }

        for (int m = 0; m < miss_count; ++m) {
            size_t total_pages =
                chunks > 1 ? (loader->expert_size / PAGE_BYTES) : 0;
            size_t page_cursor = 0;
            char* dst_base = (char*)loader->slot_buffers[missing_slots[m]];
            off_t expert_offset =
                (off_t)missing_experts[m] * (off_t)loader->expert_size;

            for (int chunk = 0; chunk < chunks; ++chunk) {
                size_t chunk_offset = 0;
                size_t chunk_size = loader->expert_size;
                if (chunks > 1) {
                    size_t pages_this_chunk = total_pages / (size_t)chunks;
                    if ((size_t)chunk < (total_pages % (size_t)chunks)) {
                        pages_this_chunk++;
                    }
                    chunk_offset = page_cursor * PAGE_BYTES;
                    chunk_size = pages_this_chunk * PAGE_BYTES;
                    page_cursor += pages_this_chunk;
                }

                int task_index = m * chunks + chunk;
                tasks[task_index].fd = loader->layer_fds[layer_idx];
                tasks[task_index].dst = dst_base + chunk_offset;
                tasks[task_index].offset =
                    expert_offset + (off_t)chunk_offset;
                tasks[task_index].size = chunk_size;
                tasks[task_index].result = 0;
            }
        }

        int generation =
            flash_moe_io_pool_start(&loader->pool, tasks, num_tasks);
        flash_moe_io_pool_wait_generation(&loader->pool, generation);

        /* Phase 4: Update slot bank ownership on successful loads */
        int base = layer_idx * loader->slot_bank_size;
        for (int m = 0; m < miss_count; ++m) {
            ssize_t total = 0;
            for (int chunk = 0; chunk < chunks; ++chunk) {
                FlashMoePreadTask* task = &tasks[m * chunks + chunk];
                if (task->result > 0) {
                    total += task->result;
                }
            }
            if (total == (ssize_t)loader->expert_size) {
                loader->slot_bank_owner[base + missing_slots[m]] =
                    missing_experts[m];
                flash_moe_slot_bank_touch(loader, layer_idx, missing_slots[m]);
            } else {
                int pos = missing_positions[m];
                slot_ids_out[pos] = -1;
                miss_slots_out[pos] = -1;
            }
        }

        free(tasks);
    }

    free(protected_slots);
    free(reserved_slots);
    free(missing_positions);
    free(missing_experts);
    free(missing_slots);
    return miss_count;
}

#pragma mark - Prefetch (async)

int slot_bank_prefetch(
    void* handle,
    int layer_idx,
    const int* expert_ids,
    int k
) {
    FlashMoeExpertLoader* loader = (FlashMoeExpertLoader*)handle;
    if (loader == NULL || expert_ids == NULL) {
        return -1;
    }
    if (layer_idx < 0 || layer_idx >= loader->num_layers ||
        k <= 0 || k > loader->max_k) {
        return -1;
    }

    /* Check which experts are already in slot bank (hits) */
    int* hit_slots = (int*)malloc((size_t)k * sizeof(int));
    int* miss_expert_ids = (int*)malloc((size_t)k * sizeof(int));
    int miss_count = 0;
    if (!hit_slots || !miss_expert_ids) {
        free(hit_slots);
        free(miss_expert_ids);
        return -1;
    }

    for (int i = 0; i < k; ++i) {
        int slot = (loader->slot_bank_size > 0)
            ? flash_moe_slot_bank_find_owner(loader, layer_idx, expert_ids[i])
            : -1;
        if (slot < 0) {
            miss_expert_ids[miss_count++] = expert_ids[i];
        }
    }

    if (miss_count == 0) {
        free(hit_slots);
        free(miss_expert_ids);
        return 0;
    }

    int chunks = flash_moe_active_cache_io_split(
        loader->expert_size,
        loader->cache_io_split);
    int num_tasks = miss_count * chunks;

    /* We reuse existing slots for prefetch — pick first miss_count slots */
    FlashMoePreadTask* tasks =
        (FlashMoePreadTask*)calloc((size_t)num_tasks, sizeof(FlashMoePreadTask));
    if (tasks == NULL) {
        free(hit_slots);
        free(miss_expert_ids);
        return -1;
    }

    for (int m = 0; m < miss_count; ++m) {
        int slot = m;  /* Reuse first miss_count slots */
        size_t total_pages =
            chunks > 1 ? (loader->expert_size / PAGE_BYTES) : 0;
        size_t page_cursor = 0;
        char* dst_base = (char*)loader->slot_buffers[slot];
        off_t expert_offset =
            (off_t)miss_expert_ids[m] * (off_t)loader->expert_size;

        for (int chunk = 0; chunk < chunks; ++chunk) {
            size_t chunk_offset = 0;
            size_t chunk_size = loader->expert_size;
            if (chunks > 1) {
                size_t pages_this_chunk = total_pages / (size_t)chunks;
                if ((size_t)chunk < (total_pages % (size_t)chunks)) {
                    pages_this_chunk++;
                }
                chunk_offset = page_cursor * PAGE_BYTES;
                chunk_size = pages_this_chunk * PAGE_BYTES;
                page_cursor += pages_this_chunk;
            }

            int task_index = m * chunks + chunk;
            tasks[task_index].fd = loader->layer_fds[layer_idx];
            tasks[task_index].dst = dst_base + chunk_offset;
            tasks[task_index].offset =
                expert_offset + (off_t)chunk_offset;
            tasks[task_index].size = chunk_size;
            tasks[task_index].result = 0;
        }
    }

    free(hit_slots);
    free(miss_expert_ids);

    loader->prefetch_generation =
        flash_moe_io_pool_start(&loader->pool, tasks, num_tasks);
    loader->prefetch_tasks = tasks;
    loader->prefetch_task_count = num_tasks;

    return 0;
}

void slot_bank_wait_prefetch(void* handle) {
    FlashMoeExpertLoader* loader = (FlashMoeExpertLoader*)handle;
    if (loader == NULL || loader->prefetch_generation <= 0) {
        return;
    }
    flash_moe_io_pool_wait_generation(&loader->pool, loader->prefetch_generation);
    free(loader->prefetch_tasks);
    loader->prefetch_tasks = NULL;
    loader->prefetch_task_count = 0;
    loader->prefetch_generation = 0;
}
