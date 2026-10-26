// main.cu : correctness + timing harness for the GEMM ladder.
//
//   ./build/gemm_ladder                 # M=N=K=4096, 10 timed reps
//   ./build/gemm_ladder --size 2048 --reps 20 --csv results/gemm_ladder.csv
//
// For every rung we (1) check the result against cuBLAS, (2) time it with CUDA events (median of
// --reps runs after warm-up), (3) report TFLOP/s and the fraction of cuBLAS achieved.
// A rung that fails the correctness check is reported as FAIL and its speed is not trusted.

#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <string>
#include <vector>

#include "kernels.cuh"

#define CUDA_CHECK(x)                                                                   \
  do {                                                                                  \
    cudaError_t e_ = (x);                                                               \
    if (e_ != cudaSuccess) {                                                            \
      std::fprintf(stderr, "CUDA error %s at %s:%d: %s\n", #x, __FILE__, __LINE__,      \
                   cudaGetErrorString(e_));                                             \
      std::exit(1);                                                                     \
    }                                                                                   \
  } while (0)

#define CUBLAS_CHECK(x)                                                                 \
  do {                                                                                  \
    cublasStatus_t s_ = (x);                                                            \
    if (s_ != CUBLAS_STATUS_SUCCESS) {                                                  \
      std::fprintf(stderr, "cuBLAS error %s at %s:%d: status %d\n", #x, __FILE__,       \
                   __LINE__, (int)s_);                                                  \
      std::exit(1);                                                                     \
    }                                                                                   \
  } while (0)

struct Rung {
  std::string name;
  bool tensor_core;                 // compared against the cuBLAS tensor-core baseline
  std::function<void()> launch;     // enqueue one GEMM
  double ms = 0, tflops = 0, rel_err = 0, max_abs = 0, spot = 0;
  bool ok = false;
};

static double median(std::vector<float> v) {
  std::sort(v.begin(), v.end());
  return v[v.size() / 2];
}

// Time fn() `reps` times with CUDA events; returns the median in milliseconds.
static double time_ms(const std::function<void()>& fn, int reps, int warmup = 3) {
  for (int i = 0; i < warmup; ++i) fn();
  CUDA_CHECK(cudaDeviceSynchronize());
  cudaEvent_t s, e;
  CUDA_CHECK(cudaEventCreate(&s));
  CUDA_CHECK(cudaEventCreate(&e));
  std::vector<float> t;
  for (int i = 0; i < reps; ++i) {
    CUDA_CHECK(cudaEventRecord(s));
    fn();
    CUDA_CHECK(cudaEventRecord(e));
    CUDA_CHECK(cudaEventSynchronize(e));
    float ms;
    CUDA_CHECK(cudaEventElapsedTime(&ms, s, e));
    t.push_back(ms);
  }
  CUDA_CHECK(cudaEventDestroy(s));
  CUDA_CHECK(cudaEventDestroy(e));
  return median(t);
}

// Relative Frobenius error and max abs error of `got` vs `ref` (both device pointers, M*N floats).
static void compare(const float* d_got, const float* d_ref, size_t n, double* rel, double* maxabs) {
  std::vector<float> g(n), r(n);
  CUDA_CHECK(cudaMemcpy(g.data(), d_got, n * sizeof(float), cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(r.data(), d_ref, n * sizeof(float), cudaMemcpyDeviceToHost));
  double num = 0, den = 0, mx = 0;
  for (size_t i = 0; i < n; ++i) {
    const double d = (double)g[i] - r[i];
    num += d * d;
    den += (double)r[i] * r[i];
    mx = std::max(mx, std::fabs(d));
  }
  *rel = std::sqrt(num / (den > 0 ? den : 1));
  *maxabs = mx;
}

static int cores_per_sm(int major, int minor) {
  if (major == 7) return 64;                      // Volta / Turing
  if (major == 8) return minor == 0 ? 64 : 128;   // A100 : Ampere GA10x / Ada
  if (major == 9) return 128;                     // Hopper
  return 128;
}

// Fill with a deterministic pseudo-random sequence in [-1, 1].
static void fill(std::vector<float>& v, unsigned seed) {
  unsigned s = seed;
  for (auto& x : v) {
    s = s * 1664525u + 1013904223u;
    x = ((s >> 8) & 0xFFFF) / 32768.0f - 1.0f;
  }
}

int main(int argc, char** argv) {
  int size = 4096, reps = 10;
  std::string csv_path;
  for (int i = 1; i < argc; ++i) {
    if (!std::strcmp(argv[i], "--size") && i + 1 < argc) size = std::atoi(argv[++i]);
    else if (!std::strcmp(argv[i], "--reps") && i + 1 < argc) reps = std::atoi(argv[++i]);
    else if (!std::strcmp(argv[i], "--csv") && i + 1 < argc) csv_path = argv[++i];
    else { std::fprintf(stderr, "usage: %s [--size S] [--reps R] [--csv path]\n", argv[0]); return 2; }
  }
  if (size % 128 != 0) { std::fprintf(stderr, "--size must be a multiple of 128\n"); return 2; }
  const int M = size, N = size, K = size;
  const double flops = 2.0 * M * N * K;

  cudaDeviceProp prop;
  CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
  int clock_khz = 0;
  CUDA_CHECK(cudaDeviceGetAttribute(&clock_khz, cudaDevAttrClockRate, 0));
  const double peak_fp32 = 2.0 * prop.multiProcessorCount * cores_per_sm(prop.major, prop.minor) *
                           (clock_khz * 1e3) / 1e12;
  std::printf("device : %s (sm_%d%d, %d SMs)\n", prop.name, prop.major, prop.minor,
              prop.multiProcessorCount);
  std::printf("problem: M=N=K=%d   fp32 peak ~ %.2f TFLOP/s at %.0f MHz (spec estimate)\n\n", size,
              peak_fp32, clock_khz / 1e3);

  // ---- data: fp16-rounded values so every rung (fp32 and tensor core) sees identical inputs ----
  std::vector<float> hA(M * K), hB(K * N);
  fill(hA, 1);
  fill(hB, 2);
  std::vector<__half> hAh(M * K), hBh(K * N);
  for (size_t i = 0; i < hA.size(); ++i) { hAh[i] = __float2half(hA[i]); hA[i] = __half2float(hAh[i]); }
  for (size_t i = 0; i < hB.size(); ++i) { hBh[i] = __float2half(hB[i]); hB[i] = __half2float(hBh[i]); }

  float *dA, *dB, *dC, *dRef;
  __half *dAh, *dBh;
  CUDA_CHECK(cudaMalloc(&dA, sizeof(float) * M * K));
  CUDA_CHECK(cudaMalloc(&dB, sizeof(float) * K * N));
  CUDA_CHECK(cudaMalloc(&dC, sizeof(float) * M * N));
  CUDA_CHECK(cudaMalloc(&dRef, sizeof(float) * M * N));
  CUDA_CHECK(cudaMalloc(&dAh, sizeof(__half) * M * K));
  CUDA_CHECK(cudaMalloc(&dBh, sizeof(__half) * K * N));
  CUDA_CHECK(cudaMemcpy(dA, hA.data(), sizeof(float) * M * K, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(dB, hB.data(), sizeof(float) * K * N, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(dAh, hAh.data(), sizeof(__half) * M * K, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(dBh, hBh.data(), sizeof(__half) * K * N, cudaMemcpyHostToDevice));

  cublasHandle_t handle;
  CUBLAS_CHECK(cublasCreate(&handle));
  const float alpha = 1.f, beta = 0.f;

  // cuBLAS is column-major. A row-major C = A*B is the column-major C^T = B^T * A^T, so we swap
  // the operands and the m/n dimensions.
  auto cublas_fp32 = [&] {
    CUBLAS_CHECK(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha, dB, N, dA, K, &beta, dC, N));
  };
  auto cublas_tc = [&] {
    CUBLAS_CHECK(cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha, dBh, CUDA_R_16F, N,
                              dAh, CUDA_R_16F, K, &beta, dC, CUDA_R_32F, N, CUBLAS_COMPUTE_32F,
                              CUBLAS_GEMM_DEFAULT_TENSOR_OP));
  };

  // Reference result: cuBLAS fp32 (no tensor cores on this path for fp32 inputs).
  CUBLAS_CHECK(cublasSetMathMode(handle, CUBLAS_DEFAULT_MATH));
  cublas_fp32();
  CUDA_CHECK(cudaDeviceSynchronize());
  CUDA_CHECK(cudaMemcpy(dRef, dC, sizeof(float) * M * N, cudaMemcpyDeviceToDevice));

  // Independent check that does not trust cuBLAS: 128 sampled outputs recomputed in fp64 on the host from
  // the same fp16-rounded inputs. Catches the case where a kernel and cuBLAS agree but are both wrong.
  const int NS = 128;
  std::vector<int> si(NS), sj(NS);
  std::vector<double> sref(NS);
  {
    unsigned st = 12345u;
    for (int t = 0; t < NS; ++t) {
      st = st * 1664525u + 1013904223u; si[t] = (int)((st >> 8) % (unsigned)M);
      st = st * 1664525u + 1013904223u; sj[t] = (int)((st >> 8) % (unsigned)N);
      double acc = 0;
      for (int k = 0; k < K; ++k) acc += (double)hA[(size_t)si[t] * K + k] * (double)hB[(size_t)k * N + sj[t]];
      sref[t] = acc;
    }
  }
  // worst scaled error |got - ref| / (|ref| + 1); infinity if any sampled value is not finite
  auto spot_err = [&](const float* d) {
    double worst = 0;
    for (int t = 0; t < NS; ++t) {
      float v;
      CUDA_CHECK(cudaMemcpy(&v, d + (size_t)si[t] * N + sj[t], sizeof(float), cudaMemcpyDeviceToHost));
      if (!std::isfinite(v)) return (double)INFINITY;
      worst = std::max(worst, std::fabs((double)v - sref[t]) / (std::fabs(sref[t]) + 1.0));
    }
    return worst;
  };
  const double cublas_spot = spot_err(dRef);

  const double t_cublas_fp32 = time_ms(cublas_fp32, reps);
  const double t_cublas_tc = time_ms(cublas_tc, reps);

  std::vector<Rung> rungs;
  auto add = [&](const char* name, bool tc, std::function<void()> f) {
    Rung r;
    r.name = name;
    r.tensor_core = tc;
    r.launch = std::move(f);
    rungs.push_back(std::move(r));
  };
  add("k1_naive", false, [&] {
    k1_naive<<<dim3((M + 31) / 32, (N + 31) / 32), dim3(32, 32)>>>(M, N, K, dA, dB, dC);
  });
  add("k2_coalesced", false, [&] {
    k2_coalesced<<<dim3(M / 32, N / 32), dim3(1024)>>>(M, N, K, dA, dB, dC);
  });
  add("k3_smem", false, [&] { k3_smem<32><<<dim3(N / 32, M / 32), dim3(1024)>>>(M, N, K, dA, dB, dC); });
  add("k4_regtile1d", false, [&] {
    k4_regtile1d<64, 64, 8, 8><<<dim3(N / 64, M / 64), dim3(512)>>>(M, N, K, dA, dB, dC);
  });
  add("k5_regtile2d", false, [&] {
    k5_regtile2d<128, 128, 8, 8, 8><<<dim3(N / 128, M / 128), dim3(256)>>>(M, N, K, dA, dB, dC);
  });
  add("k6_vectorized", false, [&] {
    k6_vectorized<128, 128, 8, 8, 8><<<dim3(N / 128, M / 128), dim3(256)>>>(M, N, K, dA, dB, dC);
  });
  add("k7_wmma", true, [&] {
    k7_wmma<128, 128, 32><<<dim3(N / 128, M / 128), dim3(256)>>>(M, N, K, dAh, dBh, dC);
  });

  std::printf("%-16s %10s %10s %12s %12s %12s  %s\n", "kernel", "ms", "TFLOP/s", "% of cuBLAS", "rel err vs cuBLAS",
              "err vs fp64", "check");
  std::printf("%-16s %10.3f %10.2f %12s %12s %12.2e  %s\n", "cublas_fp32", t_cublas_fp32,
              flops / (t_cublas_fp32 * 1e-3) / 1e12, "100.0", "(ref)", cublas_spot, "-");
  std::printf("%-16s %10.3f %10.2f %12s %12s %12s  %s\n", "cublas_tc_fp16", t_cublas_tc,
              flops / (t_cublas_tc * 1e-3) / 1e12, "100.0", "-", "-", "-");
  std::printf("------------------------------------------------------------------------------\n");

  FILE* csv = csv_path.empty() ? nullptr : std::fopen(csv_path.c_str(), "w");
  if (csv) {
    std::fprintf(csv, "kernel,size,ms,tflops,pct_of_cublas,baseline,rel_err,max_abs_err,err_vs_fp64,status\n");
    std::fprintf(csv, "cublas_fp32,%d,%.4f,%.3f,100.0,self,0,0,%.3e,ref\n", size, t_cublas_fp32,
                 flops / (t_cublas_fp32 * 1e-3) / 1e12, cublas_spot);
    std::fprintf(csv, "cublas_tc_fp16,%d,%.4f,%.3f,100.0,self,0,0,0,ref\n", size, t_cublas_tc,
                 flops / (t_cublas_tc * 1e-3) / 1e12);
  }

  int failures = 0;
  for (auto& r : rungs) {
    // correctness first
    CUDA_CHECK(cudaMemset(dC, 0xFF, sizeof(float) * M * N));  // NaN pattern: unwritten outputs fail
    r.launch();
    cudaError_t err = cudaGetLastError();
    if (err == cudaSuccess) err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
      std::printf("%-16s launch failed: %s\n", r.name.c_str(), cudaGetErrorString(err));
      if (csv) std::fprintf(csv, "%s,%d,0,0,0,-,0,0,0,LAUNCH_FAIL\n", r.name.c_str(), size);
      ++failures;
      cudaGetLastError();
      continue;
    }
    compare(dC, dRef, (size_t)M * N, &r.rel_err, &r.max_abs);
    r.spot = spot_err(dC);
    const double tol = r.tensor_core ? 2e-3 : 1e-4;
    r.ok = r.rel_err < tol && r.spot < tol;
    r.ms = time_ms(r.launch, reps);
    r.tflops = flops / (r.ms * 1e-3) / 1e12;
    const double base = r.tensor_core ? t_cublas_tc : t_cublas_fp32;
    const double pct = 100.0 * base / r.ms;
    std::printf("%-16s %10.3f %10.2f %11.1f%% %12.2e %12.2e  %s\n", r.name.c_str(), r.ms, r.tflops, pct,
                r.rel_err, r.spot, r.ok ? "PASS" : "FAIL");
    if (!r.ok) ++failures;
    if (csv)
      std::fprintf(csv, "%s,%d,%.4f,%.3f,%.2f,%s,%.3e,%.3e,%.3e,%s\n", r.name.c_str(), size, r.ms, r.tflops,
                   pct, r.tensor_core ? "cublas_tc_fp16" : "cublas_fp32", r.rel_err, r.max_abs, r.spot,
                   r.ok ? "PASS" : "FAIL");
  }
  if (csv) std::fclose(csv);

  std::printf("\nk1-k6 are compared with cuBLAS fp32 SGEMM; k7 (fp16 tensor cores) with cuBLAS tensor-core GEMM.\n");
  std::printf(failures ? "%d rung(s) FAILED correctness; do not trust their timings.\n" : "all rungs passed correctness.\n", failures);

  CUBLAS_CHECK(cublasDestroy(handle));
  cudaFree(dA); cudaFree(dB); cudaFree(dC); cudaFree(dRef); cudaFree(dAh); cudaFree(dBh);
  return failures ? 1 : 0;
}
