// CPU emulation test: every kernel in the ladder must reproduce a plain triple-loop GEMM.
//   g++ -std=c++20 -O2 -pthread tests/emu_test.cpp -o build/emu_test && ./build/emu_test
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include "emu_shim.h"
#include "../src/kernels.cuh"

static int g_fail = 0;

static void check(const char* name, const std::vector<float>& got, const std::vector<double>& ref,
                  double tol) {
  double num = 0, den = 0, maxabs = 0;
  for (size_t i = 0; i < ref.size(); ++i) {
    const double d = got[i] - ref[i];
    num += d * d;
    den += ref[i] * ref[i];
    if (std::fabs(d) > maxabs) maxabs = std::fabs(d);
  }
  const double rel = std::sqrt(num / den);
  const bool ok = rel < tol;
  std::printf("%-26s rel_err=%.2e max_abs=%.2e  %s\n", name, rel, maxabs, ok ? "PASS" : "FAIL");
  if (!ok) ++g_fail;
}

int main(int argc, char** argv) {
  const int S = argc > 1 ? std::atoi(argv[1]) : 256;  // M = N = K
  const int M = S, N = S, K = S;
  std::vector<float> A(M * K), B(K * N);
  std::vector<half> Ah(M * K), Bh(K * N);
  std::srand(1);
  for (size_t i = 0; i < A.size(); ++i) {
    Ah[i] = (half)((std::rand() / (float)RAND_MAX) * 2 - 1);
    A[i] = (float)Ah[i];  // fp32 kernels see exactly the fp16-rounded values
  }
  for (size_t i = 0; i < B.size(); ++i) {
    Bh[i] = (half)((std::rand() / (float)RAND_MAX) * 2 - 1);
    B[i] = (float)Bh[i];
  }
  std::vector<double> ref(M * N, 0.0);
  for (int i = 0; i < M; ++i)
    for (int k = 0; k < K; ++k) {
      const double a = A[i * K + k];
      for (int j = 0; j < N; ++j) ref[i * N + j] += a * B[k * N + j];
    }

  std::vector<float> C(M * N);
  auto reset = [&] { std::fill(C.begin(), C.end(), -777.f); };  // sentinel catches unwritten outputs

  reset(); emu_launch(dim3((M + 31) / 32, (N + 31) / 32), dim3(32, 32), k1_naive, M, N, K, A.data(), B.data(), C.data());
  check("k1_naive", C, ref, 1e-5);

  reset(); emu_launch(dim3(M / 32, N / 32), dim3(1024), k2_coalesced, M, N, K, A.data(), B.data(), C.data());
  check("k2_coalesced", C, ref, 1e-5);

  reset(); emu_launch(dim3(N / 32, M / 32), dim3(1024), k3_smem<32>, M, N, K, A.data(), B.data(), C.data());
  check("k3_smem", C, ref, 1e-5);

  reset(); emu_launch(dim3(N / 64, M / 64), dim3(512), k4_regtile1d<64, 64, 8, 8>, M, N, K, A.data(), B.data(), C.data());
  check("k4_regtile1d", C, ref, 1e-5);

  reset(); emu_launch(dim3(N / 128, M / 128), dim3(256), k5_regtile2d<128, 128, 8, 8, 8>, M, N, K, A.data(), B.data(), C.data());
  check("k5_regtile2d", C, ref, 1e-5);

  reset(); emu_launch(dim3(N / 128, M / 128), dim3(256), k6_vectorized<128, 128, 8, 8, 8>, M, N, K, A.data(), B.data(), C.data());
  check("k6_vectorized", C, ref, 1e-5);

  reset(); emu_launch(dim3(N / 128, M / 128), dim3(256), k7_wmma<128, 128, 32>, M, N, K, Ah.data(), Bh.data(), C.data());
  check("k7_wmma (fp16 in)", C, ref, 1e-4);

  std::printf(g_fail ? "\n%d kernel(s) FAILED\n" : "\nall kernels match the reference\n", g_fail);
  return g_fail ? 1 : 0;
}
