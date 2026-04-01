#import "SlotBank.h"
#import "slot_bank_io.h"

@interface SlotBank () {
    SlotBankHandle _handle;
    int _slotBankSize;
    size_t _expertSize;
    int _maxK;
    NSString *_expertsDir;
}
@end

@implementation SlotBank

- (nullable instancetype)initWithExpertsDir:(NSString *)expertsDir
                                 numLayers:(int)numLayers
                                 expertSize:(size_t)expertSize
                                      maxK:(int)maxK
                               cacheIOSplit:(int)cacheIOSplit {
    self = [super init];
    if (self) {
        _expertsDir = expertsDir;
        _expertSize = expertSize;
        _maxK = maxK;
        _slotBankSize = 0;

        _handle = slot_bank_create(
            [expertsDir fileSystemRepresentation],
            numLayers,
            expertSize,
            maxK,
            cacheIOSplit);

        if (_handle == NULL) {
            return nil;
        }
    }
    return self;
}

- (void)dealloc {
    if (_handle != NULL) {
        slot_bank_destroy(_handle);
        _handle = NULL;
    }
}

- (int)setSlotBankSize:(int)slotBankSize {
    int result = slot_bank_set_size(_handle, slotBankSize);
    if (result == 0) {
        _slotBankSize = slotBankSize;
    }
    return result;
}

- (SlotBankResult)loadLayer:(int)layerIdx
                  expertIds:(const int *)expertIds
                      count:(int)count {
    SlotBankResult result;
    memset(&result, 0, sizeof(result));

    if (_handle == NULL || count > 256) {
        return result;
    }

    int missCount = slot_bank_load(
        _handle,
        layerIdx,
        expertIds,
        count,
        result.slotIds,
        result.missSlots,
        result.missExpertIds,
        &result.missCount);

    if (missCount < 0) {
        memset(&result, 0, sizeof(result));
    }

    return result;
}

- (nullable void *)bufferPointerForSlot:(int)slotIdx {
    if (_handle == NULL) {
        return NULL;
    }
    return slot_bank_get_buffer(_handle, slotIdx);
}

- (nullable id<MTLBuffer>)metalBufferForSlot:(int)slotIdx
                                     device:(id<MTLDevice>)device {
    void *ptr = [self bufferPointerForSlot:slotIdx];
    if (ptr == NULL) {
        return nil;
    }

    // Create an MTLBuffer that wraps the existing slot buffer.
    // IMPORTANT: We pass NULL deallocator — the SlotBank owns the buffer memory.
    // When the SlotBank is deallocated, it frees the underlying buffers.
    id<MTLBuffer> buf = [device newBufferWithBytesNoCopy:ptr
                                                  length:_expertSize
                                                 options:MTLResourceStorageModeShared
                                             deallocator:nil];
    return buf;
}

- (void)flush {
    if (_handle != NULL) {
        slot_bank_flush(_handle);
    }
}

- (int)prefetchLayer:(int)layerIdx
           expertIds:(const int *)expertIds
               count:(int)count {
    if (_handle == NULL) {
        return -1;
    }
    return slot_bank_prefetch(_handle, layerIdx, expertIds, count);
}

- (void)waitPrefetch {
    if (_handle != NULL) {
        slot_bank_wait_prefetch(_handle);
    }
}

- (int)slotBankSize {
    return _slotBankSize;
}

- (size_t)expertSize {
    return _expertSize;
}

@end
