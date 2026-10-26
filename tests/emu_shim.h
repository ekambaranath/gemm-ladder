// emu_shim.h : run the CUDA kernels in kernels.cuh on the CPU, with real threads.
//
// Purpose: verify indexing / synchronization logic WITHOUT a GPU, so a bug shows up as a failed
// test on a laptop instead of as a mystery on the device. It is a correctness tool only: it says
// nothing about performance, memory coalescing, bank conflicts or alignment.
//
// How: each CUDA thread becomes a std::thread, __syncthreads() becomes a std::barrier, __shared__
// becomes a static array (blocks run one after another, so one copy is enough), and the WMMA
// fragments become plain 16x16 float tiles that lane 0 of each warp computes on behalf of the warp.
#pragma once
#include <barrier>
#include <cstdint>
#include <thread>
#include <vector>

#define GEMM_EMU
#define __global__
#define __device__
#define __launch_bounds__(...)
#define __shared__ static
#define __align__(x)

struct dim3 {
  unsigned x, y, z;
  dim3(unsigned x_ = 1, unsigned y_ = 1, unsigned z_ = 1) : x(x_), y(y_), z(z_) {}
};
struct float4 { float x, y, z, w; };
struct int4 { int x, y, z, w; };
typedef _Float16 half;

inline thread_local dim3 threadIdx, blockIdx, blockDim, gridDim;
inline thread_local std::barrier<>* tl_barrier = nullptr;
#define __syncthreads() tl_barrier->arrive_and_wait()

namespace nvcuda {
namespace wmma {
struct matrix_a {};
struct matrix_b {};
struct accumulator {};
struct row_major {};
enum layout_t { mem_row_major, mem_col_major };

template <class Use, int M, int N, int K, class T = void, class L = void>
struct fragment { float v[16 * 16]; };

inline bool lane0() { return threadIdx.x % 32 == 0; }

template <class F> void fill_fragment(F& f, float x) {
  if (lane0()) for (int i = 0; i < 256; ++i) f.v[i] = x;
}
template <class F, class T> void load_matrix_sync(F& f, const T* p, unsigned ldm) {
  if (lane0())
    for (int r = 0; r < 16; ++r)
      for (int c = 0; c < 16; ++c) f.v[r * 16 + c] = (float)p[r * ldm + c];
}
template <class FA, class FB, class FC> void mma_sync(FC& d, const FA& a, const FB& b, const FC& c) {
  if (lane0()) {
    float t[256];
    for (int i = 0; i < 16; ++i)
      for (int j = 0; j < 16; ++j) {
        float s = c.v[i * 16 + j];
        for (int k = 0; k < 16; ++k) s += a.v[i * 16 + k] * b.v[k * 16 + j];
        t[i * 16 + j] = s;
      }
    for (int i = 0; i < 256; ++i) d.v[i] = t[i];
  }
}
template <class F> void store_matrix_sync(float* p, const F& f, unsigned ldm, layout_t) {
  if (lane0())
    for (int r = 0; r < 16; ++r)
      for (int c = 0; c < 16; ++c) p[r * ldm + c] = f.v[r * 16 + c];
}
}  // namespace wmma
}  // namespace nvcuda
using namespace nvcuda;

// Launch `kernel<<<grid, block>>>(args...)` on the CPU.
template <class K, class... Args>
void emu_launch(dim3 grid, dim3 block, K kernel, Args... args) {
  const unsigned n = block.x * block.y * block.z;
  for (unsigned bz = 0; bz < grid.z; ++bz)
    for (unsigned by = 0; by < grid.y; ++by)
      for (unsigned bx = 0; bx < grid.x; ++bx) {
        std::barrier<> bar(n);
        std::vector<std::thread> ts;
        ts.reserve(n);
        for (unsigned t = 0; t < n; ++t)
          ts.emplace_back([&, t, bx, by, bz] {
            tl_barrier = &bar;
            threadIdx = dim3(t % block.x, (t / block.x) % block.y, t / (block.x * block.y));
            blockIdx = dim3(bx, by, bz);
            blockDim = block;
            gridDim = grid;
            kernel(args...);
          });
        for (auto& th : ts) th.join();
      }
}
