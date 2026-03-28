#ifndef METAL_IO_POOL_H
#define METAL_IO_POOL_H

#include <pthread.h>
#include <stdint.h>

#define MAX_K 8

typedef struct {
    int gen;   // generation this entry belongs to
    int done;  // 1 if this expert pread completed
} async_pread_entry_t;

typedef struct {
    InferMetal *metal;              // GPU object
    int num_valid;                   // valid entries in the packed file
    async_pread_entry_t entries[MAX_K];  // per-slot: gen+done [num_valid]
    int generation;                  // incremented each dispatch
    size_t offsets[MAX_K];           // byte offsets within packed file
} async_pread_valid_t;

typedef struct {
    int fd;
    void *dst;
    size_t offset;
    size_t size;
    int result;
} InferPreadTask;

typedef struct {
    InferPreadTask tasks[64];
    int num_tasks;
    int tasks_completed;
    int generation;
    pthread_mutex_t mutex;
    pthread_cond_t work_ready;
    pthread_cond_t work_done;
    pthread_t threads[16];
    int num_threads;
} InferIOPool;

void io_pool_init(InferIOPool *pool, int num_threads);
void io_pool_wait_all(InferIOPool *pool);
void io_pool_dispatch(InferIOPool *pool, InferPreadTask *tasks, int num_tasks);

#endif
