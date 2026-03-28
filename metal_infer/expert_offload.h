/*
 * expert_offload.h — SSD expert offload staging buffer
 *
 * A background thread pre-loads experts from SSD into a staging buffer.
 * When the main thread needs an expert, the lookup order is:
 *   1. GPU weights (not managed here)
 *   2. Metal buffer LRU cache (not managed here)
 *   3. Offload staging buffer (this module) — O(1) lookup
 *   4. Synchronous SSD fallback
 *
 * Supports two modes:
 *   - Packed SSD file mode: single file with all experts sequentially
 *   - Layer-dir mode: reads from existing layer_XX.bin files
 *
 * Packed SSD file layout:
 *   [16 bytes header]
 *     - uint32 num_layers
 *     - uint32 num_experts_per_layer
 *     - uint64 expert_size
 *   [data: (num_layers * num_experts) * expert_size bytes]
 *
 * Expert offset in packed file: 16 + (layer * num_experts + expert) * expert_size
 */

#ifndef EXPERT_OFFLOAD_H
#define EXPERT_OFFLOAD_H

#include <stdint.h>
#include <pthread.h>

// Tunables — adjust based on available RAM
#ifndef EXPERT_OFFLOAD_STAGING_ENTRIES
#define EXPERT_OFFLOAD_STAGING_ENTRIES 256  // entries in the staging buffer (~1.3 GB)
#endif

#ifndef EXPERT_OFFLOAD_PREFETCH_QUEUE_SIZE
#define EXPERT_OFFLOAD_PREFETCH_QUEUE_SIZE 64  // predicted expert queue depth
#endif

#ifndef MAX_LAYER_FDS
#define MAX_LAYER_FDS 64  // max layer files (for layer-dir mode)
#endif

// An entry in the offload staging buffer
typedef struct {
    int valid;           // 1 = entry contains valid data
    int layer_idx;       // layer index, or -1 for empty
    int expert_idx;      // expert index, or -1 for empty
    uint64_t last_used;  // monotonic counter for LRU tracking
    void *data;          // pointer to expert data (page-aligned)
} OffloadStagingEntry;

// The offload context (one per process, owned by the background thread)
typedef struct {
    pthread_t thread;

    // Mode: packed file vs layer files
    int use_packed_file;      // 1 = packed SSD file, 0 = layer files
    int ssd_fd;              // fd for packed file, -1 if using layer files
    int layer_fds[MAX_LAYER_FDS];  // fds for layer files (lazy open)
    char layer_dir[512];      // directory containing layer_XX.bin files

    uint32_t num_layers;
    uint32_t num_experts_per_layer;
    uint64_t expert_size;    // bytes per expert

    // Staging buffer (direct-mapped cache)
    OffloadStagingEntry entries[EXPERT_OFFLOAD_STAGING_ENTRIES];

    // Prediction queue: experts predicted to be needed soon
    struct {
        int layer_idx;
        int expert_idx;
    } prefetch_queue[EXPERT_OFFLOAD_PREFETCH_QUEUE_SIZE];
    int prefetch_head;
    int prefetch_tail;
    int prefetch_count;

    // Control
    pthread_mutex_t mutex;
    pthread_cond_t cond;
    int shutdown;
    int running;

    // Statistics
    uint64_t staging_hits;
    uint64_t staging_misses;
    uint64_t prefetch_issued;
    uint64_t prefetch_completed;
} ExpertOffloadCtx;

// Global singleton (declared extern in the .m file)
extern ExpertOffloadCtx *g_offload;

// Init: path can be a packed SSD file OR a directory of layer_XX.bin files.
// If NULL, uses "out_122b/packed_experts" directory.
int expert_offload_init(const char *packed_ssd_path_or_dir);
void expert_offload_shutdown(void);

// Submit a predicted expert to the prefetch queue
void expert_offload_prefetch(int layer_idx, int expert_idx);

// Check if an expert is in the staging buffer and return its pointer.
// If found: fills *out_data and returns 1.
// If not found: returns 0 (caller should do sync SSD read).
int expert_offload_lookup(int layer_idx, int expert_idx, void **out_data);

// Copy expert data from the staging buffer to a destination buffer.
// Returns 1 on success, 0 on miss (caller must do sync SSD read).
int expert_offload_copy(int layer_idx, int expert_idx, void *dst, size_t size);

// Promote an entry (mark as recently used — called after expert is used)
void expert_offload_promote(int layer_idx, int expert_idx);

// Print statistics
void expert_offload_print_stats(void);

#endif // EXPERT_OFFLOAD_H
