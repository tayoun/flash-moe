/**
 * slot_bank_io.h
 *
 * Slot bank + async pread I/O for Flash-MOE expert loading.
 * Ported from anemll-flash-mlx (https://github.com/Anemll/anemll-flash-mlx)
 * Adapted for Qwen3.5-122B-A10B-4bit (expert_size = 5,308,416 bytes).
 *
 * Key differences from anemll:
 * - Uses per-layer files (layer_00.bin … layer_47.bin)
 * - Expert size is 5,308,416 bytes (vs 35B model's ~1.69 MB)
 * - All function signatures preserved from original
 */

#ifndef SLOT_BANK_IO_H
#define SLOT_BANK_IO_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Opaque handle to the expert loader.
 * Created by slot_bank_create(), destroyed by slot_bank_destroy().
 */
typedef void* SlotBankHandle;

/**
 * Create a new slot bank expert loader.
 *
 * @param experts_dir  Path to directory containing layer_00.bin, layer_01.bin, ...
 * @param num_layers   Number of layers (48 for 122B)
 * @param expert_size  Size of each expert in bytes (5,308,416 for 122B)
 * @param max_k        Maximum number of experts that can be loaded (slot buffer count)
 * @param cache_io_split  Chunk count for I/O splitting (1-8, 0=default 1)
 * @return             Handle to use for subsequent calls, or NULL on error.
 */
void* slot_bank_create(
    const char* experts_dir,
    int num_layers,
    size_t expert_size,
    int max_k,
    int cache_io_split
);

/**
 * Destroy a slot bank expert loader.
 * Closes all file handles, frees slot buffers, shuts down thread pool.
 *
 * @param handle  Handle returned by slot_bank_create()
 */
void slot_bank_destroy(void* handle);

/**
 * Load a single expert into slot 0 (for initial/testing use).
 * Uses async I/O via thread pool.
 *
 * @param handle       Handle from slot_bank_create()
 * @param layer_idx    Layer index (0 to num_layers-1)
 * @param expert_id    Expert index within the layer (0 to 255)
 * @return             0 on success, -1 on error
 */
int slot_bank_load_single(void* handle, int layer_idx, int expert_id);

/**
 * Load K experts into the slot bank using slot bank algorithm.
 * On miss: chooses victim slot via LRU, loads expert async.
 * On hit: marks slot as protected.
 *
 * @param handle              Handle from slot_bank_create()
 * @param layer_idx           Layer index
 * @param expert_ids          Array of K expert IDs to load
 * @param k                   Number of experts (must be <= slot_bank_size)
 * @param slot_ids_out        Output: slot index for each expert_ids[i] (-1 on error)
 * @param miss_slots_out      Output: victim slot chosen for each miss
 * @param miss_expert_ids_out Output: expert ID for each miss
 * @param miss_count_out      Output: number of misses
 * @return                    Number of misses, or -1 on error
 */
int slot_bank_load(
    void* handle,
    int layer_idx,
    const int* expert_ids,
    int k,
    int* slot_ids_out,
    int* miss_slots_out,
    int* miss_expert_ids_out,
    int* miss_count_out
);

/**
 * Enable and configure the slot bank.
 * Must be called after slot_bank_create() and before slot_bank_load().
 *
 * @param handle          Handle from slot_bank_create()
 * @param slot_bank_size  Number of slots per layer (e.g., 32, 64, 128)
 * @return                0 on success, -1 on error
 */
int slot_bank_set_size(void* handle, int slot_bank_size);

/**
 * Get the pointer to a slot's buffer.
 * The buffer is page-aligned and pre-allocated.
 *
 * @param handle      Handle from slot_bank_create()
 * @param slot_idx    Slot index (0 to max_k-1)
 * @return            Pointer to slot buffer, or NULL on error
 */
void* slot_bank_get_buffer(void* handle, int slot_idx);

/**
 * Flush all slot bank state (invalidate all slots).
 * Does NOT free buffers — just resets ownership.
 *
 * @param handle  Handle from slot_bank_create()
 */
void slot_bank_flush(void* handle);

/**
 * Get the expert size.
 *
 * @param handle  Handle from slot_bank_create()
 * @return        Expert size in bytes, or 0 on error
 */
size_t slot_bank_expert_size(void* handle);

/**
 * Prefetch experts asynchronously (for temporal prefetch).
 * Loads experts into slot bank without blocking — uses thread pool.
 * Call slot_bank_wait_prefetch() to wait for completion.
 *
 * @param handle       Handle from slot_bank_create()
 * @param layer_idx    Layer index
 * @param expert_ids   Array of expert IDs to prefetch
 * @param k            Number of experts
 * @return             0 on success, -1 on error
 */
int slot_bank_prefetch(
    void* handle,
    int layer_idx,
    const int* expert_ids,
    int k
);

/**
 * Wait for any pending prefetch operations to complete.
 *
 * @param handle  Handle from slot_bank_create()
 */
void slot_bank_wait_prefetch(void* handle);

#ifdef __cplusplus
}
#endif

#endif /* SLOT_BANK_IO_H */
