/**
 * SlotBank.h
 *
 * Objective-C wrapper around slot_bank_io.c.
 * Provides slot bank + async pread I/O for Flash-MOE expert loading.
 *
 * Usage:
 *   SlotBank *bank = [[SlotBank alloc] initWithExpertsDir:@"/path/to/experts"
 *                              numLayers:48
 *                            expertSize:5308416
 *                               maxK:256
 *                         cacheIOSplit:0];
 *   [bank setSlotBankSize:32];
 *   ...
 *   SlotBankResult *r = [bank loadLayer:0 expertIds:expertIds count:k];
 *   void *buf = [bank bufferPointerForSlot:r.slotIds[0]];
 *   // pass buf to Metal kernel
 */

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

NS_ASSUME_NONNULL_BEGIN

/**
 * Result of a slot bank load operation.
 */
typedef struct {
    int slotIds[256];   // slot index for each requested expert
    int missSlots[256]; // victim slot chosen for each miss
    int missExpertIds[256]; // expert ID for each miss
    int missCount;      // number of misses
} SlotBankResult;

/**
 * SlotBank — slot bank + async pread I/O for expert loading.
 *
 * Wraps the C slot_bank_io API and provides Metal buffer integration.
 * The slot bank uses an LRU policy to choose victim slots when there are misses.
 *
 * Thread-safe for concurrent loadLayer: calls from multiple threads — the underlying
 * C code uses a mutex to serialize access to shared data structures.
 */
@interface SlotBank : NSObject

/**
 * Initialize a slot bank expert loader.
 *
 * @param expertsDir  Path to directory containing layer_00.bin, layer_01.bin, ...
 * @param numLayers   Number of layers (48 for 122B)
 * @param expertSize  Size of each expert in bytes (5,308,416 for 122B)
 * @param maxK        Maximum number of slots (should equal num_experts_per_tok or larger)
 * @param cacheIOSplit  I/O splitting factor (0=default, 1=no split, 2-8=chunk count)
 * @return            Initialized instance, or nil on error
 */
- (nullable instancetype)initWithExpertsDir:(NSString *)expertsDir
                                 numLayers:(int)numLayers
                                 expertSize:(size_t)expertSize
                                      maxK:(int)maxK
                               cacheIOSplit:(int)cacheIOSplit;

/**
 * Enable and configure the slot bank with the given size.
 *
 * @param slotBankSize  Number of slots per layer (e.g., 32, 64, 128).
 *                      Must be <= maxK passed at init.
 * @return             0 on success, -1 on error
 */
- (int)setSlotBankSize:(int)slotBankSize;

/**
 * Load K experts for a layer using the slot bank algorithm.
 *
 * On hit: expert is already in a slot — returned immediately (no I/O).
 * On miss: chooses victim slot via LRU, loads expert async via thread pool.
 *
 * @param layerIdx   Layer index (0 to numLayers-1)
 * @param expertIds  Array of K expert IDs to load
 * @param count      Number of experts (must be <= slotBankSize)
 * @return           SlotBankResult with slot IDs and miss info
 */
- (SlotBankResult)loadLayer:(int)layerIdx
                  expertIds:(const int *)expertIds
                      count:(int)count;

/**
 * Get a pointer to the raw bytes in a slot buffer.
 * The buffer is page-aligned and pre-allocated.
 *
 * @param slotIdx  Slot index (0 to maxK-1)
 * @return         Pointer to slot buffer bytes, or NULL on error
 */
- (nullable void *)bufferPointerForSlot:(int)slotIdx;

/**
 * Get the MTLBuffer for a slot (if using with Metal).
 *
 * Note: The SlotBank does NOT manage MTLBuffer lifecycle — it only
 * wraps the raw slot buffers. You can create an MTLBuffer from
 * the bytes using [device newBufferWithBytesNoCopy:length:options:deallocator:].
 *
 * @param slotIdx  Slot index
 * @return         MTLBuffer wrapper, or nil on error
 */
- (nullable id<MTLBuffer>)metalBufferForSlot:(int)slotIdx
                                      device:(id<MTLDevice>)device;

/**
 * Flush all slot bank state (invalidate all slots).
 * Does NOT free buffers — just resets ownership so next load will reload.
 */
- (void)flush;

/**
 * Prefetch experts asynchronously (temporal prefetch).
 * Loads experts into slot bank without blocking — uses thread pool.
 * Call [self waitPrefetch] to wait for completion.
 *
 * @param layerIdx   Layer index
 * @param expertIds  Array of expert IDs to prefetch
 * @param count      Number of experts
 * @return           0 on success, -1 on error
 */
- (int)prefetchLayer:(int)layerIdx
           expertIds:(const int *)expertIds
               count:(int)count;

/**
 * Wait for any pending prefetch operations to complete.
 */
- (void)waitPrefetch;

/**
 * Get the configured slot bank size (0 if not enabled).
 */
- (int)slotBankSize;

/**
 * Expert size in bytes.
 */
- (size_t)expertSize;

@end

NS_ASSUME_NONNULL_END
