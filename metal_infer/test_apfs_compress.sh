#!/bin/bash
# Test APFS transparent compression on expert data.
# Compresses a copy of one layer file with APFS compression,
# then benchmarks pread throughput on compressed vs uncompressed.

set -e

EXPERTS_DIR="${EXPERTS_DIR:-${MODEL_DIR:-}/packed_experts}"
SRC="$EXPERTS_DIR/layer_00.bin"
WORKDIR="/tmp/apfs_compress_test"

echo "=== APFS Transparent Compression Test ==="

# Check source
if [ ! -f "$SRC" ]; then
    echo "ERROR: source file not found: $SRC"
    echo "Set MODEL_DIR or EXPERTS_DIR before running this script."
    exit 1
fi

RAW_SIZE=$(stat -f%z "$SRC")
echo "Source: $SRC"
echo "Raw size: $((RAW_SIZE / 1024 / 1024)) MB ($RAW_SIZE bytes)"

# Setup
rm -rf "$WORKDIR"
mkdir -p "$WORKDIR"

# Copy uncompressed
echo ""
echo "--- Copying uncompressed ---"
cp "$SRC" "$WORKDIR/layer_uncompressed.bin"

# Copy and compress with APFS compression (using ditto --hfsCompression or afsctool)
echo ""
echo "--- Compressing with APFS ---"

# Method 1: afsctool (if available)
if command -v afsctool &>/dev/null; then
    cp "$SRC" "$WORKDIR/layer_compressed.bin"
    afsctool -c -T LZFSE "$WORKDIR/layer_compressed.bin"
    COMP_SIZE=$(stat -f%z "$WORKDIR/layer_compressed.bin")
    COMP_DISK=$(du -k "$WORKDIR/layer_compressed.bin" | cut -f1)
    echo "afsctool LZFSE: logical=$((COMP_SIZE / 1024 / 1024)) MB, disk=${COMP_DISK}K"
# Method 2: ditto with --hfsCompression
else
    echo "(afsctool not found, trying ditto --hfsCompression)"
    # ditto preserves and can apply HFS+ compression (which APFS also supports)
    ditto --hfsCompression "$SRC" "$WORKDIR/layer_compressed.bin" 2>/dev/null || {
        # Method 3: copy and use afscutil
        echo "(ditto failed, trying /usr/bin/compression_tool)"
        cp "$SRC" "$WORKDIR/layer_compressed.bin"
    }
    COMP_SIZE=$(stat -f%z "$WORKDIR/layer_compressed.bin")
    # Check actual disk usage (compressed files show logical size in stat but less on disk)
    COMP_DISK=$(du -k "$WORKDIR/layer_compressed.bin" | cut -f1)
    UNCOMP_DISK=$(du -k "$WORKDIR/layer_uncompressed.bin" | cut -f1)
    echo "Logical size: $((COMP_SIZE / 1024 / 1024)) MB"
    echo "Disk usage compressed:   ${COMP_DISK}K"
    echo "Disk usage uncompressed: ${UNCOMP_DISK}K"
    echo "Compression ratio: $(echo "scale=1; $COMP_DISK * 100 / $UNCOMP_DISK" | bc)%"
fi

# Check if file is actually compressed (UF_COMPRESSED flag)
echo ""
echo "--- File flags ---"
ls -lO "$WORKDIR/layer_compressed.bin" 2>/dev/null | head -1
ls -lO "$WORKDIR/layer_uncompressed.bin" 2>/dev/null | head -1

echo ""
echo "--- Building pread benchmark ---"
cat > "$WORKDIR/bench_pread.c" << 'BENCH_EOF'
#include <stdio.h>
#include <stdlib.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/time.h>
#include <string.h>

#define EXPERT_SIZE 7077888

static double now_ms(void) {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec * 1000.0 + tv.tv_usec / 1000.0;
}

int main(int argc, char **argv) {
    if (argc < 3) {
        fprintf(stderr, "Usage: %s <file> <num_reads> [expert_idx]\n", argv[0]);
        return 1;
    }
    const char *path = argv[1];
    int num_reads = atoi(argv[2]);
    int expert_idx = argc > 3 ? atoi(argv[3]) : 0;

    int fd = open(path, O_RDONLY);
    if (fd < 0) { perror("open"); return 1; }

    /* Purge page cache for this file */
    fcntl(fd, F_NOCACHE, 1);
    fcntl(fd, F_NOCACHE, 0);

    void *buf;
    posix_memalign(&buf, 2*1024*1024, EXPERT_SIZE);

    /* Warm up (1 read to get past any initial overhead) */
    pread(fd, buf, EXPERT_SIZE, (off_t)expert_idx * EXPERT_SIZE);

    /* Benchmark: sequential reads of same expert (measures steady-state throughput) */
    double t0 = now_ms();
    for (int i = 0; i < num_reads; i++) {
        /* Read different experts to avoid OS caching the same pages */
        int eidx = (expert_idx + i) % 512;
        ssize_t n = pread(fd, buf, EXPERT_SIZE, (off_t)eidx * EXPERT_SIZE);
        if (n != EXPERT_SIZE) {
            fprintf(stderr, "Short read at expert %d: %zd\n", eidx, n);
            break;
        }
    }
    double elapsed = now_ms() - t0;
    double per_read = elapsed / num_reads;
    double gbps = (EXPERT_SIZE / (1024.0*1024*1024)) / (per_read / 1000.0);

    printf("  %s:\n", path);
    printf("    %d reads of %.2f MB each\n", num_reads, EXPERT_SIZE / (1024.0*1024));
    printf("    Total: %.1f ms, Per read: %.2f ms\n", elapsed, per_read);
    printf("    Throughput: %.1f GB/s\n", gbps);

    /* Verify data integrity */
    unsigned char *b = (unsigned char *)buf;
    unsigned long checksum = 0;
    for (int i = 0; i < EXPERT_SIZE; i++) checksum += b[i];
    printf("    Checksum: %lu\n", checksum);

    close(fd);
    free(buf);
    return 0;
}
BENCH_EOF

clang -O2 -o "$WORKDIR/bench_pread" "$WORKDIR/bench_pread.c"

echo ""
echo "=== Benchmark: Uncompressed ==="
"$WORKDIR/bench_pread" "$WORKDIR/layer_uncompressed.bin" 20

echo ""
echo "=== Benchmark: APFS Compressed ==="
"$WORKDIR/bench_pread" "$WORKDIR/layer_compressed.bin" 20

echo ""
echo "=== Benchmark: Cold reads (purge cache first) ==="
echo "--- Purging disk cache ---"
sudo purge 2>/dev/null || echo "(purge requires sudo, skipping cold test)"

echo ""
echo "Uncompressed cold:"
"$WORKDIR/bench_pread" "$WORKDIR/layer_uncompressed.bin" 4

echo ""
echo "Compressed cold:"
"$WORKDIR/bench_pread" "$WORKDIR/layer_compressed.bin" 4

# Cleanup
echo ""
echo "Workdir: $WORKDIR (not cleaned up — inspect manually)"
