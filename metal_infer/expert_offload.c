/*
 * expert_offload.c — SSD expert offload staging buffer
 *
 * A background thread pre-loads experts from SSD (via the existing layer files
 * or a packed SSD file) into a staging buffer. When the main thread needs an
 * expert, the lookup order is:
 *   1. GPU weights (not managed here)
 *   2. Metal buffer LRU cache (not managed here)
 *   3. Offload staging buffer (this module) — O(1) lookup
 *   4. Synchronous SSD fallback
 *
 * If --offload-ssd is not specified, falls back to sequential readahead mode
 * using the existing layer files (no packed file needed).
 *
 * Packed SSD file layout (if packed file is used):
 *   [16 bytes header]
 *     - uint32 num_layers
 *     - uint32 num_experts_per_layer
 *     - uint64 expert_size
 *   [data: (num_layers * num_experts) * expert_size bytes]
 *   Expert offset: 16 + (layer * num_experts + expert) * expert_size
 */

#include "expert_offload.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <unistd.h>
#include <fcntl.h>
#include <errno.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <mach/mach.h>

// ============================================================================
// Config
// ============================================================================

#define STAGING_ENTRIES EXPERT_OFFLOAD_STAGING_ENTRIES
#define PREFETCH_QUEUE  EXPERT_OFFLOAD_PREFETCH_QUEUE_SIZE
#define MAX_LAYER_FDS   64  // maximum layer files

// ============================================================================
// Globals
// ============================================================================

ExpertOffloadCtx *g_offload = NULL;

// ============================================================================
// Utility
// ============================================================================

static inline int staging_slot(int layer_idx, int expert_idx) {
    if (!g_offload) return -1;
    uint64_t key = (uint64_t)layer_idx * g_offload->num_experts_per_layer + (uint64_t)expert_idx;
    return (int)(key % STAGING_ENTRIES);
}

static inline off_t expert_ssd_offset(ExpertOffloadCtx *ctx, int layer_idx, int expert_idx) {
    if (ctx->use_packed_file) {
        return (off_t)16 + ((off_t)layer_idx * ctx->num_experts_per_layer + (off_t)expert_idx) * (off_t)ctx->expert_size;
    } else {
        // Sequential layout within each layer file
        return (off_t)expert_idx * (off_t)ctx->expert_size;
    }
}

// Get fd for a layer (opens on demand, caches in array)
static int get_layer_fd(ExpertOffloadCtx *ctx, int layer_idx) {
    if (ctx->use_packed_file) {
        return ctx->ssd_fd;  // single packed file fd
    }
    if (layer_idx < 0 || (uint32_t)layer_idx >= ctx->num_layers) return -1;
    if (ctx->layer_fds[layer_idx] < 0) {
        // Open on demand
        char path[512];
        snprintf(path, sizeof(path), "%s/layer_%02d.bin", ctx->layer_dir, layer_idx);
        ctx->layer_fds[layer_idx] = open(path, O_RDONLY | O_CLOEXEC);
    }
    return ctx->layer_fds[layer_idx];
}

// ============================================================================
// Background thread
// ============================================================================

static void *offload_thread_fn(void *arg) {
    ExpertOffloadCtx *ctx = (ExpertOffloadCtx *)arg;

    while (1) {
        pthread_mutex_lock(&ctx->mutex);

        // Wait for work or shutdown
        while (ctx->prefetch_count == 0 && !ctx->shutdown) {
            pthread_cond_wait(&ctx->cond, &ctx->mutex);
        }

        if (ctx->shutdown) {
            pthread_mutex_unlock(&ctx->mutex);
            break;
        }

        // Dequeue one prediction
        int slot = ctx->prefetch_tail;
        int layer_idx = ctx->prefetch_queue[slot].layer_idx;
        int expert_idx = ctx->prefetch_queue[slot].expert_idx;
        ctx->prefetch_tail = (slot + 1) % PREFETCH_QUEUE;
        ctx->prefetch_count--;
        ctx->prefetch_issued++;

        pthread_mutex_unlock(&ctx->mutex);

        // Find staging slot
        int entry_idx = staging_slot(layer_idx, expert_idx);
        OffloadStagingEntry *entry = &ctx->entries[entry_idx];

        // Check if entry already has this expert (no-op if already loaded)
        int do_load = 0;
        pthread_mutex_lock(&ctx->mutex);
        if (!entry->valid ||
            entry->layer_idx != layer_idx ||
            entry->expert_idx != expert_idx) {
            do_load = 1;
        }
        pthread_mutex_unlock(&ctx->mutex);

        if (!do_load) continue;

        // Load from SSD
        int fd = get_layer_fd(ctx, layer_idx);
        if (fd < 0) continue;

        off_t offset = expert_ssd_offset(ctx, layer_idx, expert_idx);
        ssize_t r = pread(fd, entry->data, ctx->expert_size, offset);

        // Update entry metadata (atomically via mutex)
        pthread_mutex_lock(&ctx->mutex);
        if (r == (ssize_t)ctx->expert_size) {
            entry->valid = 1;
            entry->layer_idx = layer_idx;
            entry->expert_idx = expert_idx;
            entry->last_used = ++ctx->prefetch_completed;
        } else {
            // Mark invalid on read error
            entry->valid = 0;
            entry->layer_idx = -1;
            entry->expert_idx = -1;
            if (r < 0) {
                fprintf(stderr, "[offload] pread error for L%d,E%d: %s\n",
                        layer_idx, expert_idx, strerror(errno));
            } else {
                fprintf(stderr, "[offload] short read %zd/%llu for L%d,E%d\n",
                        r, (unsigned long long)ctx->expert_size, layer_idx, expert_idx);
            }
        }
        pthread_mutex_unlock(&ctx->mutex);
    }

    return NULL;
}

// ============================================================================
// Public API
// ============================================================================

int expert_offload_init(const char *packed_ssd_path_or_dir) {
    if (g_offload) {
        fprintf(stderr, "[offload] already initialized\n");
        return 0;
    }

    int use_packed = 0;
    char layer_dir[512] = {0};

    // Check if the path is a packed SSD file or a directory of layer files
    if (packed_ssd_path_or_dir) {
        struct stat st;
        if (stat(packed_ssd_path_or_dir, &st) == 0) {
            if (S_ISREG(st.st_mode)) {
                // Regular file — treat as packed SSD file
                use_packed = 1;
            } else if (S_ISDIR(st.st_mode)) {
                // Directory — use layer files directly
                use_packed = 0;
                strncpy(layer_dir, packed_ssd_path_or_dir, sizeof(layer_dir) - 1);
            }
        }
    }

    int packed_fd = -1;
    uint32_t num_layers = 48;
    uint32_t num_experts = 256;
    uint64_t expert_size = 5308416;

    if (use_packed) {
        // Open packed SSD file
        packed_fd = open(packed_ssd_path_or_dir, O_RDONLY | O_CLOEXEC);
        if (packed_fd < 0) {
            fprintf(stderr, "[offload] cannot open %s: %s\n", packed_ssd_path_or_dir, strerror(errno));
            return -1;
        }

        // Read header
        uint8_t header[16];
        ssize_t hr = read(packed_fd, header, sizeof(header));
        if (hr != sizeof(header)) {
            fprintf(stderr, "[offload] cannot read header from %s: %zd\n", packed_ssd_path_or_dir, hr);
            close(packed_fd);
            return -1;
        }
        num_layers = ((uint32_t)header[0] << 24) | ((uint32_t)header[1] << 16) |
                     ((uint32_t)header[2] << 8)  | ((uint32_t)header[3]);
        num_experts = ((uint32_t)header[4] << 24) | ((uint32_t)header[5] << 16) |
                      ((uint32_t)header[6] << 8)  | ((uint32_t)header[7]);
        expert_size = ((uint64_t)header[8] << 56) | ((uint64_t)header[9] << 48) |
                      ((uint64_t)header[10] << 40) | ((uint64_t)header[11] << 32) |
                      ((uint64_t)header[12] << 24) | ((uint64_t)header[13] << 16) |
                      ((uint64_t)header[14] << 8)  | ((uint64_t)header[15]);

        fprintf(stderr, "[offload] packed SSD: %u layers x %u experts, %llu bytes/expert\n",
                num_layers, num_experts, (unsigned long long)expert_size);
    } else {
        // Use layer files directly from directory
        // Try to detect layer_dir from the path
        if (!layer_dir[0] && packed_ssd_path_or_dir) {
            strncpy(layer_dir, packed_ssd_path_or_dir, sizeof(layer_dir) - 1);
        }
        // If no directory specified, use packed_experts in out_122b
        if (!layer_dir[0]) {
            snprintf(layer_dir, sizeof(layer_dir), "out_122b/packed_experts");
        }

        // Try to detect expert_size from first layer file
        char probe[512];
        snprintf(probe, sizeof(probe), "%s/layer_00.bin", layer_dir[0] ? layer_dir : ".");
        int probe_fd = open(probe, O_RDONLY | O_CLOEXEC);
        if (probe_fd >= 0) {
            struct stat st;
            if (fstat(probe_fd, &st) == 0) {
                // 1.3589GB / 256 = 5.3MB per expert
                if (st.st_size > 0) {
                    expert_size = (uint64_t)(st.st_size / num_experts);
                }
            }
            close(probe_fd);
        }

        fprintf(stderr, "[offload] layer-dir mode: %s (experts: %u layers x %u, %llu bytes)\n",
                layer_dir, num_layers, num_experts, (unsigned long long)expert_size);
    }

    // Allocate context
    g_offload = calloc(1, sizeof(ExpertOffloadCtx));
    g_offload->ssd_fd = packed_fd;
    g_offload->num_layers = num_layers;
    g_offload->num_experts_per_layer = num_experts;
    g_offload->expert_size = expert_size;
    g_offload->use_packed_file = use_packed;
    g_offload->running = 0;
    g_offload->staging_hits = 0;
    g_offload->staging_misses = 0;
    g_offload->prefetch_issued = 0;
    g_offload->prefetch_completed = 0;
    strncpy(g_offload->layer_dir, layer_dir, sizeof(g_offload->layer_dir) - 1);

    // Initialize layer fds to -1 (lazy open)
    for (int i = 0; i < MAX_LAYER_FDS; i++) {
        g_offload->layer_fds[i] = -1;
    }

    pthread_mutex_init(&g_offload->mutex, NULL);
    pthread_cond_init(&g_offload->cond, NULL);

    // Allocate staging buffer entries (page-aligned memory)
    size_t page_size = (size_t)getpagesize();
    size_t aligned_size = (expert_size + page_size - 1) & ~(page_size - 1);

    int actual_entries = 0;
    for (int i = 0; i < STAGING_ENTRIES; i++) {
        g_offload->entries[i].valid = 0;
        g_offload->entries[i].layer_idx = -1;
        g_offload->entries[i].expert_idx = -1;
        g_offload->entries[i].last_used = 0;
        g_offload->entries[i].data = NULL;

        void *buf = NULL;
        if (posix_memalign(&buf, page_size, aligned_size) == 0 && buf) {
            memset(buf, 0, aligned_size);
            g_offload->entries[i].data = buf;
            actual_entries++;
        }
    }

    if (actual_entries == 0) {
        fprintf(stderr, "[offload] WARNING: could not allocate any staging entries\n");
    }

    fprintf(stderr, "[offload] staging buffer: %d entries (%.1f GB)\n",
            actual_entries,
            (double)actual_entries * aligned_size / 1e9);

    // Start background thread
    g_offload->shutdown = 0;
    if (pthread_create(&g_offload->thread, NULL, offload_thread_fn, g_offload) != 0) {
        fprintf(stderr, "[offload] pthread_create failed: %s\n", strerror(errno));
        if (packed_fd >= 0) close(packed_fd);
        free(g_offload);
        g_offload = NULL;
        return -1;
    }
    g_offload->running = 1;
    fprintf(stderr, "[offload] background thread started\n");

    return 0;
}

void expert_offload_shutdown(void) {
    if (!g_offload) return;

    pthread_mutex_lock(&g_offload->mutex);
    g_offload->shutdown = 1;
    pthread_cond_signal(&g_offload->cond);
    pthread_mutex_unlock(&g_offload->mutex);

    pthread_join(g_offload->thread, NULL);

    if (g_offload->ssd_fd >= 0) close(g_offload->ssd_fd);
    for (int i = 0; i < MAX_LAYER_FDS; i++) {
        if (g_offload->layer_fds[i] >= 0) close(g_offload->layer_fds[i]);
    }
    free(g_offload);
    g_offload = NULL;

    fprintf(stderr, "[offload] shutdown complete\n");
}

void expert_offload_prefetch(int layer_idx, int expert_idx) {
    if (!g_offload || !g_offload->running) return;

    pthread_mutex_lock(&g_offload->mutex);

    if (g_offload->prefetch_count >= PREFETCH_QUEUE) {
        pthread_mutex_unlock(&g_offload->mutex);
        return;
    }

    // Deduplicate: check if already in queue
    int found = 0;
    for (int i = 0; i < g_offload->prefetch_count; i++) {
        int idx = (g_offload->prefetch_tail + i) % PREFETCH_QUEUE;
        if (g_offload->prefetch_queue[idx].layer_idx == layer_idx &&
            g_offload->prefetch_queue[idx].expert_idx == expert_idx) {
            found = 1;
            break;
        }
    }

    if (!found) {
        int slot = (g_offload->prefetch_tail + g_offload->prefetch_count) % PREFETCH_QUEUE;
        g_offload->prefetch_queue[slot].layer_idx = layer_idx;
        g_offload->prefetch_queue[slot].expert_idx = expert_idx;
        g_offload->prefetch_count++;
        pthread_cond_signal(&g_offload->cond);
    }

    pthread_mutex_unlock(&g_offload->mutex);
}

// Fast O(1) lookup in staging buffer
int expert_offload_lookup(int layer_idx, int expert_idx, void **out_data) {
    if (!g_offload || !g_offload->running) return 0;

    int entry_idx = staging_slot(layer_idx, expert_idx);
    OffloadStagingEntry *entry = &g_offload->entries[entry_idx];

    pthread_mutex_lock(&g_offload->mutex);
    int valid = entry->valid;
    int match = valid && entry->layer_idx == layer_idx && entry->expert_idx == expert_idx;
    if (match && out_data) {
        *out_data = entry->data;
        entry->last_used = ++g_offload->staging_hits;
    } else {
        g_offload->staging_misses++;
    }
    pthread_mutex_unlock(&g_offload->mutex);

    return match;
}

// Copy from staging buffer to destination (caller's buffer)
int expert_offload_copy(int layer_idx, int expert_idx, void *dst, size_t size) {
    if (!g_offload || !g_offload->running) return 0;

    int entry_idx = staging_slot(layer_idx, expert_idx);
    OffloadStagingEntry *entry = &g_offload->entries[entry_idx];

    pthread_mutex_lock(&g_offload->mutex);
    int match = entry->valid &&
               entry->layer_idx == layer_idx &&
               entry->expert_idx == expert_idx;
    if (match) {
        size_t copy_size = size < g_offload->expert_size ? size : (size_t)g_offload->expert_size;
        memcpy(dst, entry->data, copy_size);
        entry->last_used = ++g_offload->staging_hits;
    } else {
        g_offload->staging_misses++;
    }
    pthread_mutex_unlock(&g_offload->mutex);

    return match;
}

void expert_offload_promote(int layer_idx, int expert_idx) {
    if (!g_offload || !g_offload->running) return;

    int entry_idx = staging_slot(layer_idx, expert_idx);
    OffloadStagingEntry *entry = &g_offload->entries[entry_idx];

    pthread_mutex_lock(&g_offload->mutex);
    if (entry->valid &&
        entry->layer_idx == layer_idx &&
        entry->expert_idx == expert_idx) {
        entry->last_used = ++g_offload->staging_hits;
    }
    pthread_mutex_unlock(&g_offload->mutex);
}

void expert_offload_print_stats(void) {
    if (!g_offload) return;

    pthread_mutex_lock(&g_offload->mutex);
    uint64_t hits = g_offload->staging_hits;
    uint64_t misses = g_offload->staging_misses;
    uint64_t total = hits + misses;
    uint64_t prefetch_done = g_offload->prefetch_completed;
    double hit_rate = total > 0 ? 100.0 * hits / total : 0.0;
    fprintf(stderr, "[offload] staging: hits=%llu misses=%llu (%.1f%% hit rate), "
                    "prefetch_done=%llu\n",
            (unsigned long long)hits, (unsigned long long)misses,
            hit_rate, (unsigned long long)prefetch_done);
    pthread_mutex_unlock(&g_offload->mutex);
}
