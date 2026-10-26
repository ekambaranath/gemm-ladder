# gemm-ladder: from a naive CUDA GEMM to tensor cores, one idea at a time

Seven CUDA C++ matrix-multiply kernels, each changing exactly **one** idea, every rung checked against
cuBLAS and timed against it. The point is not the final number. It is being able to say *why* each step is
faster, with the arithmetic to back it up. (Inspired by the "build it up from scratch" style of
`micrograd`/`nanoGPT`/`llm.c`, and by Simon Boehm's CUDA matmul worklog.)

```mermaid
flowchart LR
    k1["k1 naive<br/>one thread per output"] --> k2["k2 coalesced<br/>adjacent threads, adjacent addresses"]
    k2 --> k3["k3 shared-memory tiling<br/>load each value once per tile"]
    k3 --> k4["k4 1D register tile<br/>8 outputs per thread"]
    k4 --> k5["k5 2D register tile<br/>8x8 outer products"]
    k5 --> k6["k6 vectorized<br/>float4 loads, transposed A"]
    k6 --> k7["k7 tensor cores<br/>WMMA fp16 in, fp32 accumulate"]
```

## Run it (Colab T4, about 5 minutes)

```bash
git clone https://github.com/ekambaranath/gemm-ladder && cd gemm-ladder
make check                       # builds, checks all 7 rungs vs cuBLAS at 512^3
make run                         # M=N=K=4096, writes results/gemm_ladder.csv
python scripts/plot_ladder.py    # writes docs/gemm_ladder.png
```

Other GPUs: `make ARCH=sm_80` (A100), `sm_89` (L4/RTX 40), `sm_90` (H100). No GPU? `make emu` runs every kernel
on the CPU with real threads and barriers and compares to a double-precision reference. It verifies indexing and
synchronization only, and says nothing about speed. I confirmed it is a real test: deliberately breaking two
kernels (a transposed index in k3, a wrong component in k6) makes it fail both.

## What each rung does, and why it should help

Notation: `C[M,N] = A[M,K] x B[K,N]`, `2*M*N*K` flops (137 GFLOP at M=N=K=4096). Without reuse every FMA needs two
4-byte loads, so the whole game is raising the number of FMAs per byte moved at each level of the memory hierarchy
(global, shared, registers).

**k1 naive.** One thread per C element, loop over K. Threads in a warp differ in the *row* index, so the 32
lanes read `A[x*K+k]` 4096 floats apart and write `C[x*N+y]` 4096 floats apart: each lane's access is its own
memory transaction. Memory-system efficiency is the problem, not arithmetic.

**k2 coalesced.** Same math, remap threads so the 32 lanes of a warp own 32 consecutive *columns* of one row.
`B[k*N+col]` and `C[row*N+col]` are now contiguous (one transaction per warp) and `A[row*K+k]` is a broadcast.
Only an index change; the memory-access pattern is what moves.

**k3 shared-memory tiling.** A block owns a 32x32 tile of C and walks K in steps of 32. Each step loads one
32x32 tile of A and B into shared memory, then does 32 FMAs per thread out of shared memory. Each global value is
now fetched once per tile instead of once per use, so global traffic drops by about 32x.

**k4 1D register tiling.** Each thread computes 8 outputs of one column. For each k it loads one B value into a
register and reuses it 8 times. Per k step that is 9 shared-memory loads (1 B + 8 A) for 8 FMAs, about 0.9 FMA
per load, versus 0.5 in k3 (2 loads per FMA). Fewer loads per flop means the FMA units wait less.

**k5 2D register tiling.** Each thread owns an 8x8 patch (64 accumulators). Per k it loads 8 A values and 8 B
values and does an 8x8 outer product: 64 FMAs for 16 shared loads, 4 FMAs per load, up from 0.5 in k3. This is
the structure real SGEMM kernels use.

**k6 vectorized.** `float4` (128-bit) global loads and stores, plus storing A *transposed* in shared memory so
the inner loop reads 8 consecutive A values instead of striding by BK. Fewer, wider memory instructions and a
cleaner shared-memory access pattern.

**k7 tensor cores (WMMA).** fp16 inputs, fp32 accumulation, 16x16x16 matrix-multiply-accumulate per warp
instruction. 8 warps per block, each owning a 64x32 patch (4x2 fragments). Shared tiles are padded by 8 halves
so rows land in different banks and every fragment pointer stays 32-byte aligned (a WMMA requirement). This rung
changes *precision*, so it is compared with cuBLAS running on tensor cores, not with fp32 SGEMM.

## Correctness first

Every rung is checked before it is timed, two ways:

1. **Against cuBLAS** (relative Frobenius error over the whole matrix).
2. **Against an independent fp64 reference** computed on the host: 128 sampled output elements recomputed in
   double precision from the same fp16-rounded inputs. This does not trust cuBLAS, so it catches the case where a
   kernel and cuBLAS agree with each other but are both wrong.

Output is pre-filled with a NaN pattern, so an unwritten element fails. A rung that fails either check is reported
FAIL and its timing is not to be trusted. Tolerance: `< 1e-4` for fp32 rungs, `< 2e-3` for the fp16 tensor-core rung.

Note on the "0.00e+00" errors: the fp32 rungs match cuBLAS fp32 *bit for bit*. That is plausible rather than
suspicious, because each of these kernels accumulates every output element sequentially over k with fused
multiply-add, and a non-split-K cuBLAS SGEMM does the same, so the rounding sequence is identical. The tensor-core
rung accumulates differently and shows a small non-zero error (about 7e-6). The fp64 check above is what makes the
claim independent of that explanation.

## Results

Measured on a Tesla T4 (Colab, sm_75, 40 SMs), M=N=K=4096, CUDA events, median of 10 runs after warm-up, one
session. Raw data: `results/gemm_ladder.csv`.

![GEMM ladder](docs/gemm_ladder.png)

| kernel | ms | TFLOP/s | % of matching cuBLAS | step speedup | rel err vs cuBLAS |
|---|---|---|---|---|---|
| cuBLAS fp32 (reference) | 29.60 | 4.64 | 100 | | |
| cuBLAS fp16 tensor cores | 3.30 | 41.68 | 100 | | |
| k1 naive | 2221.04 | 0.06 | 1.3 | | 0 |
| k2 coalesced | 231.61 | 0.59 | 12.8 | 9.6x | 0 |
| k3 shared-memory tiling | 146.57 | 0.94 | 20.2 | 1.58x | 0 |
| k4 1D register tile | 69.59 | 1.98 | 42.5 | 2.11x | 0 |
| k5 2D register tile | 39.49 | 3.48 | 75.0 | 1.76x | 0 |
| k6 vectorized | 35.15 | 3.91 | 84.2 | 1.12x | 0 |
| k7 WMMA tensor cores (fp16) | 6.75 | 20.37 | 48.9 (of tensor-core cuBLAS) | 5.2x vs k6 | 6.7e-6 |

k1 to k6 is a 63x speedup. k7 is not an improvement on k6 in the same units, because it changes precision.

### What the data shows

- **Memory access pattern dominates first.** k1 to k2 is 9.6x from an index remapping alone, with identical
  arithmetic.
- **Reuse is the next lever, with diminishing returns.** Shared-memory tiling gave 1.58x, register tiling 2.1x then
  1.76x. FMAs per shared-memory load rise from 0.5 (k3) to about 0.9 (k4) to 4 (k5), but k3 to k5 is a 3.7x speedup,
  not 8x, so something else caps the gain.
- **Vectorization is a small win here (12%).** k6 changes two things at once (float4 loads and a transposed A tile),
  so the data cannot say which one helped; that needs an ablation.
- **At 512^3 the ladder flattens at the top.** k5 and k4 take the same time (0.278 ms vs 0.281 ms). With 128x128
  tiles a 512x512 output is only 16 blocks for 40 SMs, so the GPU cannot be filled. The grid arithmetic is fact; that
  it is the cause is my inference, not something I profiled.
- **Tensor cores: big absolute gain, half of cuBLAS.** k7 is 5.2x faster than the best fp32 rung and reaches 49%
  of cuBLAS tensor-core GEMM (about 31% of the T4's 65 TFLOP/s fp16 spec).

### What it does not show (yet)

- **Why k7 stops at 49%.** Likely missing pieces: `cp.async` software pipelining, an XOR-swizzled shared layout
  instead of padding, and per-shape tuned tile sizes. These are hypotheses until profiled.
- **Percent of hardware peak.** The harness prints a spec-sheet fp32 peak (8.14 TFLOP/s at the 1590 MHz boost
  clock). T4s are power-capped and typically run below boost under sustained load, so percent-of-spec flatters the
  hardware; percent-of-cuBLAS is the more reliable number and is what the table uses. The actual clock was not measured.
- **Run-to-run variance.** One session, median only. No Nsight Compute profile yet (occupancy, bank conflicts, stall
  reasons); that is the evidence that would turn these hypotheses into findings.

## Known limits (deliberate)

- Sizes must be multiples of 128 (K a multiple of 32). No edge-tile handling, to keep each kernel readable.
- Tile shapes are fixed, not autotuned; cuBLAS selects per-shape.
- k7 uses WMMA, not raw `mma.sync` / `cp.async` pipelines, so it is not expected to match cuBLAS tensor-core speed.
- Single square shape only; skinny shapes (LLM decode) behave very differently and are the next step.

## Next steps

- Raw `mma.sync` with `cp.async` double buffering and an XOR-swizzled shared layout.
- Skinny-M int8/int4 weight GEMM for LLM decode (the follow-up to the Triton kernel in `inferlab-pytorch`).
- Warp-tile level tuning and a small autotuner.

## Layout

```
src/kernels.cuh       the seven kernels, heavily commented
src/main.cu           harness: correctness vs cuBLAS, CUDA-event timing, CSV output
tests/emu_shim.h      CPU emulation of the CUDA execution model (threads, barriers, shared memory, WMMA)
tests/emu_test.cpp    runs every kernel on the CPU against a double-precision reference
scripts/plot_ladder.py  bar chart of the ladder
Makefile              build / check / run / ncu / emu
```
