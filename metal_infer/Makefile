# Makefile for metal_infer — Pure C/Metal MoE inference engine
#
# Targets:
#   make          — build metal_infer binary
#   make run      — single expert forward pass
#   make verify   — verify Metal vs CPU reference
#   make bench    — benchmark single expert (10 iterations)
#   make moe      — full MoE forward pass (K experts, single layer)
#   make moebench — benchmark MoE (10 iterations)
#   make full     — full 60-layer MoE forward pass (K=4)
#   make fullbench — benchmark full 60-layer forward (3 iterations)
#   make chat     — build interactive chat TUI
#   make cachebench — warm page-cache pread microbenchmark for packed experts
#   make clean    — remove build artifacts
#
# Note: Metal shaders are compiled from source at runtime via
# MTLDevice newLibraryWithSource:, so no offline metal compiler needed.

CC = clang
CFLAGS = -O2 -Wall -Wextra -fobjc-arc -DACCELERATE_NEW_LAPACK
FRAMEWORKS = -framework Metal -framework Foundation -framework Accelerate
LDFLAGS = -lpthread -lcompression

TARGET = metal_infer
MAIN_SRC = main.m

# Optional: offline shader compilation (faster startup, but not required)
METALC = xcrun -sdk macosx metal
METALLIB_TOOL = xcrun -sdk macosx metallib
SHADER_SRC = shaders.metal
SHADER_AIR = shaders.air
SHADER_LIB = shaders.metallib

# Inference engine (complete forward pass)
INFER_TARGET = infer
INFER_SRC = infer.m

# Chat TUI (interactive multi-turn)
CHAT_TARGET = chat
CHAT_SRC = chat.m

# Warm page-cache pread microbenchmark
CACHE_BENCH_TARGET = cachebench
CACHE_BENCH_SRC = cache_pread_bench.c

.PHONY: all clean run verify bench moe moebench full fullbench fast metallib infer infer-run chat-run build-chat cachebench cachebench-run gemma-one-layer-check

all: $(TARGET) $(INFER_TARGET)

# Build the binary (shaders compiled at runtime from source)
$(TARGET): $(MAIN_SRC) $(SHADER_SRC)
	$(CC) $(CFLAGS) $(FRAMEWORKS) $(LDFLAGS) $(MAIN_SRC) -o $(TARGET)

# Optional: pre-compile shaders (not required — runtime compilation is the default)
metallib: $(SHADER_LIB)

$(SHADER_AIR): $(SHADER_SRC)
	$(METALC) -c $(SHADER_SRC) -o $(SHADER_AIR)

$(SHADER_LIB): $(SHADER_AIR)
	$(METALLIB_TOOL) $(SHADER_AIR) -o $(SHADER_LIB)

# Build the inference engine
$(INFER_TARGET): $(INFER_SRC)
	$(CC) $(CFLAGS) $(FRAMEWORKS) $(LDFLAGS) $(INFER_SRC) -o $(INFER_TARGET)

# Build the chat client (thin HTTP/SSE client + linenoise line editor)
$(CHAT_TARGET): $(CHAT_SRC) linenoise.c linenoise.h
	$(CC) -O2 -Wall -fobjc-arc -framework Foundation $(CHAT_SRC) linenoise.c -o $(CHAT_TARGET)

# Build the cache pread microbenchmark
$(CACHE_BENCH_TARGET): $(CACHE_BENCH_SRC)
	$(CC) -O3 -Wall -Wextra -std=c11 $(CACHE_BENCH_SRC) -lpthread -o $(CACHE_BENCH_TARGET)

clean:
	rm -f $(TARGET) $(INFER_TARGET) $(CHAT_TARGET) $(CACHE_BENCH_TARGET) $(SHADER_AIR) $(SHADER_LIB)

# Run targets
run: $(TARGET)
	./$(TARGET) --layer 0 --expert 0

verify: $(TARGET)
	./$(TARGET) --layer 0 --expert 0 --verify

fast: $(TARGET)
	./$(TARGET) --layer 0 --expert 0 --fast --verify

bench: $(TARGET)
	./$(TARGET) --layer 0 --expert 0 --fast --benchmark

moe: $(TARGET)
	./$(TARGET) --layer 0 --fast --moe

moebench: $(TARGET)
	./$(TARGET) --layer 0 --fast --moe --benchmark

full: $(TARGET)
	./$(TARGET) --fast --full --k 4

fullbench: $(TARGET)
	./$(TARGET) --fast --full --k 4 --benchmark

# Inference engine targets
build-infer: $(INFER_TARGET)

infer-run: $(INFER_TARGET)
	./$(INFER_TARGET) --prompt "Hello, what is" --tokens 20 --k 4

# Chat TUI targets (use: make chat)

chat-run: $(CHAT_TARGET)
	./$(CHAT_TARGET) --k 4

cachebench-run: $(CACHE_BENCH_TARGET)
	./$(CACHE_BENCH_TARGET) --q3-experts --experts 128 --threads 8 --split 4 --passes 5

gemma-one-layer-check: $(INFER_TARGET)
	@echo "Usage:"
	@echo "  ./infer --model <gemma_runtime_dir> --prompt \"Hello\" --tokens 16 --k 8 \\"
	@echo "      --gemma-one-layer-dump /tmp/gemma_one_layer.json"
	@echo "  python3 ../tools/gemma4/check_one_layer.py --checkpoint <gemma_ckpt_dir> \\"
	@echo "      --runtime-dump /tmp/gemma_one_layer.json"
