# Build with:  make            (default ARCH=sm_75 = Tesla T4)
# Other GPUs:  make ARCH=sm_80  (A100)   make ARCH=sm_89 (L4 / RTX 40)   make ARCH=sm_90 (H100)
ARCH ?= sm_75
NVCC ?= nvcc

all: build/gemm_ladder

build:
	mkdir -p build results

build/gemm_ladder: src/main.cu src/kernels.cuh | build
	$(NVCC) -O3 -std=c++17 -arch=$(ARCH) -lineinfo src/main.cu -lcublas -o $@

# CPU emulation of every kernel (no GPU needed): checks indexing and synchronization logic
emu: build
	g++ -std=c++20 -O2 -pthread tests/emu_test.cpp -o build/emu_test
	./build/emu_test 256

# quick correctness pass on the GPU, then the full benchmark
check: build/gemm_ladder
	./build/gemm_ladder --size 512 --reps 3

run: build/gemm_ladder
	./build/gemm_ladder --size 4096 --reps 10 --csv results/gemm_ladder.csv

# Nsight Compute on one rung, e.g.  make ncu K=k5_regtile2d
K ?= k5_regtile2d
ncu: build/gemm_ladder
	ncu --set full -k regex:$(K) -c 1 -o results/ncu_$(K) ./build/gemm_ladder --size 2048 --reps 1

clean:
	rm -rf build

.PHONY: all emu check run ncu clean
