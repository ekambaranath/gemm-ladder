// kernels.cuh : a ladder of GEMM kernels, C = A * B, all row-major.
//
//   A : [M x K]   B : [K x N]   C : [M x N]
//
// Every rung changes ONE idea. Read them in order. The arithmetic behind each step is in README.md.
//
//   k1  naive            one thread per output, but adjacent threads walk DOWN a column (bad access)
//   k2  coalesced        same math, remap threads so adjacent threads touch adjacent addresses
//   k3  smem tiling      stage 32x32 tiles in shared memory so each global value is loaded once per tile
//   k4  1D register tile each thread computes 8 outputs, reusing one B value from a register
//   k5  2D register tile each thread computes an 8x8 patch with outer products (highest reuse)
//   k6  vectorized       float4 global loads/stores + A stored transposed in shared memory
//   k7  tensor cores     WMMA fp16 inputs, fp32 accumulate (different precision: compare to cuBLAS TC)
//
// Size requirements (kept simple on purpose): M, N multiples of 128 and K multiple of 32.

#pragma once

#ifndef GEMM_EMU
#include <cuda_fp16.h>
#include <mma.h>
using namespace nvcuda;
#endif

// ---------------------------------------------------------------------------------------------
// k1: naive.  Thread (x, y) -> C[x][y].  Warp lanes differ in x, i.e. in the ROW index, so the
// 32 lanes of a warp read A[x*K + k] with a stride of K floats and write C[x*N + y] with a stride
// of N floats: every access becomes its own memory transaction.
// ---------------------------------------------------------------------------------------------
__global__ void k1_naive(int M, int N, int K, const float* __restrict__ A,
                         const float* __restrict__ B, float* __restrict__ C) {
  const int x = blockIdx.x * blockDim.x + threadIdx.x;  // row
  const int y = blockIdx.y * blockDim.y + threadIdx.y;  // col
  if (x < M && y < N) {
    float acc = 0.f;
    for (int k = 0; k < K; ++k) acc += A[x * K + k] * B[k * N + y];
    C[x * N + y] = acc;
  }
}

// ---------------------------------------------------------------------------------------------
// k2: coalesced.  Same arithmetic, different thread -> output mapping. Consecutive lanes now own
// consecutive COLUMNS of one row: B and C accesses are contiguous (one transaction per warp) and
// the A element is the same for the whole warp (a broadcast).
// ---------------------------------------------------------------------------------------------
__global__ void k2_coalesced(int M, int N, int K, const float* __restrict__ A,
                             const float* __restrict__ B, float* __restrict__ C) {
  const int row = blockIdx.x * 32 + (threadIdx.x / 32);
  const int col = blockIdx.y * 32 + (threadIdx.x % 32);
  if (row < M && col < N) {
    float acc = 0.f;
    for (int k = 0; k < K; ++k) acc += A[row * K + k] * B[k * N + col];
    C[row * N + col] = acc;
  }
}

// ---------------------------------------------------------------------------------------------
// k3: shared-memory tiling.  A block owns a BS x BS tile of C. It marches along K in steps of BS,
// each step cooperatively loading one BS x BS tile of A and one of B into shared memory, then
// every thread does BS multiply-adds out of fast on-chip memory.
// Global traffic drops by a factor of BS compared with k2.
// ---------------------------------------------------------------------------------------------
template <int BS>
__global__ void k3_smem(int M, int N, int K, const float* __restrict__ A,
                        const float* __restrict__ B, float* __restrict__ C) {
  __shared__ float As[BS * BS];
  __shared__ float Bs[BS * BS];

  const int cRow = blockIdx.y, cCol = blockIdx.x;
  const int tx = threadIdx.x % BS, ty = threadIdx.x / BS;

  A += cRow * BS * K;
  B += cCol * BS;
  C += cRow * BS * N + cCol * BS;

  float acc = 0.f;
  for (int bk = 0; bk < K; bk += BS) {
    As[ty * BS + tx] = A[ty * K + tx];
    Bs[ty * BS + tx] = B[ty * N + tx];
    __syncthreads();
    A += BS;
    B += BS * N;
    for (int d = 0; d < BS; ++d) acc += As[ty * BS + d] * Bs[d * BS + tx];
    __syncthreads();
  }
  C[ty * N + tx] = acc;
}

// ---------------------------------------------------------------------------------------------
// k4: 1D register tiling.  Each thread owns TM outputs in one column. For each k it loads ONE B
// value into a register and reuses it TM times. Fewer shared-memory loads per FMA.
// Block tile BM x BN, step BK; threads = BM*BN/TM.
// ---------------------------------------------------------------------------------------------
template <int BM, int BN, int BK, int TM>
__global__ void k4_regtile1d(int M, int N, int K, const float* __restrict__ A,
                             const float* __restrict__ B, float* __restrict__ C) {
  static_assert(BM * BK == (BM * BN) / TM, "one A element per thread per step");
  static_assert(BK * BN == (BM * BN) / TM, "one B element per thread per step");

  const int cRow = blockIdx.y, cCol = blockIdx.x;
  const int threadCol = threadIdx.x % BN;
  const int threadRow = threadIdx.x / BN;

  __shared__ float As[BM * BK];
  __shared__ float Bs[BK * BN];

  A += cRow * BM * K;
  B += cCol * BN;
  C += cRow * BM * N + cCol * BN;

  const int innerColA = threadIdx.x % BK, innerRowA = threadIdx.x / BK;
  const int innerColB = threadIdx.x % BN, innerRowB = threadIdx.x / BN;

  float acc[TM];
  for (int i = 0; i < TM; ++i) acc[i] = 0.f;

  for (int bk = 0; bk < K; bk += BK) {
    As[innerRowA * BK + innerColA] = A[innerRowA * K + innerColA];
    Bs[innerRowB * BN + innerColB] = B[innerRowB * N + innerColB];
    __syncthreads();
    A += BK;
    B += BK * N;

    for (int d = 0; d < BK; ++d) {
      const float b = Bs[d * BN + threadCol];
      for (int r = 0; r < TM; ++r) acc[r] += As[(threadRow * TM + r) * BK + d] * b;
    }
    __syncthreads();
  }
  for (int r = 0; r < TM; ++r) C[(threadRow * TM + r) * N + threadCol] = acc[r];
}

// ---------------------------------------------------------------------------------------------
// k5: 2D register tiling.  Each thread owns a TM x TN patch of C. Per k step it loads TM values of
// A and TN values of B into registers and does TM*TN FMAs (an outer product): 2*TM*TN flops for
// TM+TN shared-memory loads. With 8x8 that is 128 flops per 16 loads.
// ---------------------------------------------------------------------------------------------
template <int BM, int BN, int BK, int TM, int TN>
__global__ void __launch_bounds__((BM * BN) / (TM * TN))
    k5_regtile2d(int M, int N, int K, const float* __restrict__ A, const float* __restrict__ B,
                 float* __restrict__ C) {
  constexpr int NT = (BM * BN) / (TM * TN);
  const int cRow = blockIdx.y, cCol = blockIdx.x;
  const int threadCol = threadIdx.x % (BN / TN);
  const int threadRow = threadIdx.x / (BN / TN);

  __shared__ float As[BM * BK];
  __shared__ float Bs[BK * BN];

  A += cRow * BM * K;
  B += cCol * BN;
  C += cRow * BM * N + cCol * BN;

  const int innerRowA = threadIdx.x / BK, innerColA = threadIdx.x % BK;
  constexpr int strideA = NT / BK;
  const int innerRowB = threadIdx.x / BN, innerColB = threadIdx.x % BN;
  constexpr int strideB = NT / BN;

  float acc[TM * TN];
  for (int i = 0; i < TM * TN; ++i) acc[i] = 0.f;
  float regM[TM], regN[TN];

  for (int bk = 0; bk < K; bk += BK) {
    for (int off = 0; off < BM; off += strideA)
      As[(innerRowA + off) * BK + innerColA] = A[(innerRowA + off) * K + innerColA];
    for (int off = 0; off < BK; off += strideB)
      Bs[(innerRowB + off) * BN + innerColB] = B[(innerRowB + off) * N + innerColB];
    __syncthreads();
    A += BK;
    B += BK * N;

    for (int d = 0; d < BK; ++d) {
      for (int i = 0; i < TM; ++i) regM[i] = As[(threadRow * TM + i) * BK + d];
      for (int j = 0; j < TN; ++j) regN[j] = Bs[d * BN + threadCol * TN + j];
      for (int i = 0; i < TM; ++i)
        for (int j = 0; j < TN; ++j) acc[i * TN + j] += regM[i] * regN[j];
    }
    __syncthreads();
  }
  for (int i = 0; i < TM; ++i)
    for (int j = 0; j < TN; ++j) C[(threadRow * TM + i) * N + threadCol * TN + j] = acc[i * TN + j];
}

// ---------------------------------------------------------------------------------------------
// k6: vectorized.  (1) 128-bit float4 global loads and stores: fewer, wider memory instructions.
// (2) A is written to shared memory TRANSPOSED (As[k][m]) so the inner loop reads TM consecutive
// values of A as a contiguous run instead of striding by BK.
// ---------------------------------------------------------------------------------------------
template <int BM, int BN, int BK, int TM, int TN>
__global__ void __launch_bounds__((BM * BN) / (TM * TN))
    k6_vectorized(int M, int N, int K, const float* __restrict__ A, const float* __restrict__ B,
                  float* __restrict__ C) {
  static_assert(BM * BK / 4 == (BM * BN) / (TM * TN), "one float4 of A per thread");
  static_assert(BK * BN / 4 == (BM * BN) / (TM * TN), "one float4 of B per thread");
  static_assert(TN % 4 == 0, "float4 stores");

  const int cRow = blockIdx.y, cCol = blockIdx.x;
  const int threadCol = threadIdx.x % (BN / TN);
  const int threadRow = threadIdx.x / (BN / TN);

  __shared__ float As[BK * BM];  // transposed: As[k * BM + m]
  __shared__ float Bs[BK * BN];

  A += cRow * BM * K;
  B += cCol * BN;
  C += cRow * BM * N + cCol * BN;

  const int innerRowA = threadIdx.x / (BK / 4), innerColA = threadIdx.x % (BK / 4);
  const int innerRowB = threadIdx.x / (BN / 4), innerColB = threadIdx.x % (BN / 4);

  float acc[TM * TN];
  for (int i = 0; i < TM * TN; ++i) acc[i] = 0.f;
  float regM[TM], regN[TN];

  for (int bk = 0; bk < K; bk += BK) {
    const float4 a4 = reinterpret_cast<const float4*>(&A[innerRowA * K + innerColA * 4])[0];
    As[(innerColA * 4 + 0) * BM + innerRowA] = a4.x;
    As[(innerColA * 4 + 1) * BM + innerRowA] = a4.y;
    As[(innerColA * 4 + 2) * BM + innerRowA] = a4.z;
    As[(innerColA * 4 + 3) * BM + innerRowA] = a4.w;
    reinterpret_cast<float4*>(&Bs[innerRowB * BN + innerColB * 4])[0] =
        reinterpret_cast<const float4*>(&B[innerRowB * N + innerColB * 4])[0];
    __syncthreads();
    A += BK;
    B += BK * N;

    for (int d = 0; d < BK; ++d) {
      for (int i = 0; i < TM; ++i) regM[i] = As[d * BM + threadRow * TM + i];
      for (int j = 0; j < TN; ++j) regN[j] = Bs[d * BN + threadCol * TN + j];
      for (int i = 0; i < TM; ++i)
        for (int j = 0; j < TN; ++j) acc[i * TN + j] += regM[i] * regN[j];
    }
    __syncthreads();
  }
  for (int i = 0; i < TM; ++i) {
    for (int j = 0; j < TN; j += 4) {
      float4 v;
      v.x = acc[i * TN + j + 0];
      v.y = acc[i * TN + j + 1];
      v.z = acc[i * TN + j + 2];
      v.w = acc[i * TN + j + 3];
      reinterpret_cast<float4*>(&C[(threadRow * TM + i) * N + threadCol * TN + j])[0] = v;
    }
  }
}

// ---------------------------------------------------------------------------------------------
// k7: tensor cores via WMMA.  fp16 inputs, fp32 accumulation.  This is a different precision, so it
// is compared with cuBLAS running on tensor cores, not with fp32 SGEMM.
// Block tile 128x128, step BK=32, 8 warps arranged 2 (rows) x 4 (cols); each warp owns a 64x32
// patch = 4x2 WMMA 16x16x16 fragments. Shared tiles are padded by 8 halves to avoid bank conflicts
// and to keep every fragment pointer 32-byte aligned (WMMA requires it).
// ---------------------------------------------------------------------------------------------
template <int BM, int BN, int BK>
__global__ void __launch_bounds__(256)
    k7_wmma(int M, int N, int K, const half* __restrict__ A, const half* __restrict__ B,
            float* __restrict__ C) {
  constexpr int WM = 64, WN = 32;
  constexpr int FM = WM / 16, FN = WN / 16;
  constexpr int LDA = BK + 8, LDB = BN + 8;
  constexpr int WARPS_N = BN / WN;
  static_assert((BM / WM) * (BN / WN) == 8, "8 warps");

  __shared__ __align__(128) half As[BM * LDA];
  __shared__ __align__(128) half Bs[BK * LDB];

  const int warp = threadIdx.x / 32;
  const int warpRow = warp / WARPS_N;
  const int warpCol = warp % WARPS_N;

  A += blockIdx.y * BM * K;
  B += blockIdx.x * BN;
  C += (blockIdx.y * BM) * N + blockIdx.x * BN;

  wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc[FM][FN];
  for (int i = 0; i < FM; ++i)
    for (int j = 0; j < FN; ++j) wmma::fill_fragment(acc[i][j], 0.0f);

  for (int bk = 0; bk < K; bk += BK) {
    // 128-bit (8 x half) cooperative loads
    for (int i = threadIdx.x; i < BM * BK / 8; i += 256) {
      const int r = i / (BK / 8), c = (i % (BK / 8)) * 8;
      *reinterpret_cast<int4*>(&As[r * LDA + c]) = *reinterpret_cast<const int4*>(&A[r * K + c]);
    }
    for (int i = threadIdx.x; i < BK * BN / 8; i += 256) {
      const int r = i / (BN / 8), c = (i % (BN / 8)) * 8;
      *reinterpret_cast<int4*>(&Bs[r * LDB + c]) = *reinterpret_cast<const int4*>(&B[r * N + c]);
    }
    __syncthreads();
    A += BK;
    B += BK * N;

    for (int kk = 0; kk < BK; kk += 16) {
      wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a[FM];
      wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> b[FN];
      for (int i = 0; i < FM; ++i)
        wmma::load_matrix_sync(a[i], &As[(warpRow * WM + i * 16) * LDA + kk], LDA);
      for (int j = 0; j < FN; ++j)
        wmma::load_matrix_sync(b[j], &Bs[kk * LDB + warpCol * WN + j * 16], LDB);
      for (int i = 0; i < FM; ++i)
        for (int j = 0; j < FN; ++j) wmma::mma_sync(acc[i][j], a[i], b[j], acc[i][j]);
    }
    __syncthreads();
  }

  for (int i = 0; i < FM; ++i)
    for (int j = 0; j < FN; ++j)
      wmma::store_matrix_sync(&C[(warpRow * WM + i * 16) * N + warpCol * WN + j * 16], acc[i][j], N,
                              wmma::mem_row_major);
}
