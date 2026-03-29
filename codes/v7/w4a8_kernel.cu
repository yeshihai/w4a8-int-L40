#include <ATen/cuda/CUDAContext.h>
#include <cuda.h>
#include <cuda_fp16.h>
#include <torch/extension.h>

#include <algorithm>
#include <cstdlib>
#include <cstring>

namespace {

__device__ __forceinline__ int8_t unpack_i4_low(uint8_t packed) {
  int v = packed & 0xF;
  return static_cast<int8_t>(v >= 8 ? (v - 16) : v);
}

__device__ __forceinline__ int8_t unpack_i4_high(uint8_t packed) {
  int v = packed >> 4;
  return static_cast<int8_t>(v >= 8 ? (v - 16) : v);
}

__device__ __forceinline__ int32_t pack_s8x4(int8_t x0, int8_t x1, int8_t x2,
                                             int8_t x3) {
  return (static_cast<int32_t>(static_cast<uint8_t>(x0)) << 0) |
         (static_cast<int32_t>(static_cast<uint8_t>(x1)) << 8) |
         (static_cast<int32_t>(static_cast<uint8_t>(x2)) << 16) |
         (static_cast<int32_t>(static_cast<uint8_t>(x3)) << 24);
}

// split-K with atomic accumulation (baseline fast path)
__global__ void w4a8_tiny_kernel_v73_splitk_atomic(
    const int8_t* __restrict__ a_q, const uint8_t* __restrict__ b_q_packed,
    const half* __restrict__ b_scales, float* __restrict__ out_accum, int M,
    int N, int K, int group_size, int split_k_parts) {
  extern __shared__ int8_t sh_a[];

  const int tid = threadIdx.x;
  const int m = blockIdx.y;
  const int part = blockIdx.z;

  const int N_pack = N >> 1;
  const int n_pack = blockIdx.x * blockDim.x + tid;

  if (m >= M || n_pack >= N_pack) return;

  const int a_row_base = m * K;
  for (int k = tid; k < K; k += blockDim.x) {
    sh_a[k] = a_q[a_row_base + k];
  }
  __syncthreads();

  const int n0 = n_pack * 2;
  const int n1 = n0 + 1;

  const int num_groups = K / group_size;
  const int g_begin = (part * num_groups) / split_k_parts;
  const int g_end = ((part + 1) * num_groups) / split_k_parts;

  float acc0 = 0.0f;
  float acc1 = 0.0f;

  #pragma unroll 1
  for (int g = g_begin; g < g_end; ++g) {
    int32_t dot0 = 0;
    int32_t dot1 = 0;

    const int k_begin = g * group_size;
    const int k_end = k_begin + group_size;

    int k = k_begin;
    for (; k + 3 < k_end; k += 4) {
      const int32_t a4 = pack_s8x4(sh_a[k], sh_a[k + 1], sh_a[k + 2], sh_a[k + 3]);

      const uint8_t p0 = b_q_packed[(k + 0) * N_pack + n_pack];
      const uint8_t p1 = b_q_packed[(k + 1) * N_pack + n_pack];
      const uint8_t p2 = b_q_packed[(k + 2) * N_pack + n_pack];
      const uint8_t p3 = b_q_packed[(k + 3) * N_pack + n_pack];

      const int32_t b0_4 = pack_s8x4(unpack_i4_low(p0), unpack_i4_low(p1),
                                     unpack_i4_low(p2), unpack_i4_low(p3));
      const int32_t b1_4 = pack_s8x4(unpack_i4_high(p0), unpack_i4_high(p1),
                                     unpack_i4_high(p2), unpack_i4_high(p3));

      dot0 = __dp4a(a4, b0_4, dot0);
      dot1 = __dp4a(a4, b1_4, dot1);
    }

    for (; k < k_end; ++k) {
      const int8_t a = sh_a[k];
      const uint8_t p = b_q_packed[k * N_pack + n_pack];
      dot0 += static_cast<int32_t>(a) * static_cast<int32_t>(unpack_i4_low(p));
      dot1 += static_cast<int32_t>(a) * static_cast<int32_t>(unpack_i4_high(p));
    }

    const float ws0 = __half2float(b_scales[g * N + n0]);
    const float ws1 = __half2float(b_scales[g * N + n1]);
    acc0 += static_cast<float>(dot0) * ws0;
    acc1 += static_cast<float>(dot1) * ws1;
  }

  atomicAdd(out_accum + m * N + n0, acc0);
  atomicAdd(out_accum + m * N + n1, acc1);
}

// split-K without atomic: write each part to disjoint output slice.
__global__ void w4a8_tiny_kernel_v73_splitk_store(
    const int8_t* __restrict__ a_q, const uint8_t* __restrict__ b_q_packed,
    const half* __restrict__ b_scales, float* __restrict__ out_partial, int M,
    int N, int K, int group_size, int split_k_parts) {
  extern __shared__ int8_t sh_a[];

  const int tid = threadIdx.x;
  const int m = blockIdx.y;
  const int part = blockIdx.z;

  const int N_pack = N >> 1;
  const int n_pack = blockIdx.x * blockDim.x + tid;

  if (m >= M || n_pack >= N_pack) return;

  const int a_row_base = m * K;
  for (int k = tid; k < K; k += blockDim.x) {
    sh_a[k] = a_q[a_row_base + k];
  }
  __syncthreads();

  const int n0 = n_pack * 2;
  const int n1 = n0 + 1;

  const int num_groups = K / group_size;
  const int g_begin = (part * num_groups) / split_k_parts;
  const int g_end = ((part + 1) * num_groups) / split_k_parts;

  float acc0 = 0.0f;
  float acc1 = 0.0f;

  #pragma unroll 1
  for (int g = g_begin; g < g_end; ++g) {
    int32_t dot0 = 0;
    int32_t dot1 = 0;

    const int k_begin = g * group_size;
    const int k_end = k_begin + group_size;

    int k = k_begin;
    for (; k + 3 < k_end; k += 4) {
      const int32_t a4 = pack_s8x4(sh_a[k], sh_a[k + 1], sh_a[k + 2], sh_a[k + 3]);

      const uint8_t p0 = b_q_packed[(k + 0) * N_pack + n_pack];
      const uint8_t p1 = b_q_packed[(k + 1) * N_pack + n_pack];
      const uint8_t p2 = b_q_packed[(k + 2) * N_pack + n_pack];
      const uint8_t p3 = b_q_packed[(k + 3) * N_pack + n_pack];

      const int32_t b0_4 = pack_s8x4(unpack_i4_low(p0), unpack_i4_low(p1),
                                     unpack_i4_low(p2), unpack_i4_low(p3));
      const int32_t b1_4 = pack_s8x4(unpack_i4_high(p0), unpack_i4_high(p1),
                                     unpack_i4_high(p2), unpack_i4_high(p3));

      dot0 = __dp4a(a4, b0_4, dot0);
      dot1 = __dp4a(a4, b1_4, dot1);
    }

    for (; k < k_end; ++k) {
      const int8_t a = sh_a[k];
      const uint8_t p = b_q_packed[k * N_pack + n_pack];
      dot0 += static_cast<int32_t>(a) * static_cast<int32_t>(unpack_i4_low(p));
      dot1 += static_cast<int32_t>(a) * static_cast<int32_t>(unpack_i4_high(p));
    }

    const float ws0 = __half2float(b_scales[g * N + n0]);
    const float ws1 = __half2float(b_scales[g * N + n1]);
    acc0 += static_cast<float>(dot0) * ws0;
    acc1 += static_cast<float>(dot1) * ws1;
  }

  const int base = (part * M + m) * N;
  out_partial[base + n0] = acc0;
  out_partial[base + n1] = acc1;
}

__global__ void w4a8_tiny_kernel_v73_direct(const int8_t* __restrict__ a_q,
                                            const uint8_t* __restrict__ b_q_packed,
                                            const half* __restrict__ b_scales,
                                            const float* __restrict__ a_scales,
                                            half* __restrict__ out, int M, int N,
                                            int K, int group_size) {
  extern __shared__ int8_t sh_a[];

  const int tid = threadIdx.x;
  const int m = blockIdx.y;
  const int n_pack = blockIdx.x * blockDim.x + tid;
  const int N_pack = N >> 1;

  if (m >= M) return;

  const int a_row_base = m * K;
  for (int k = tid; k < K; k += blockDim.x) {
    sh_a[k] = a_q[a_row_base + k];
  }
  __syncthreads();

  if (n_pack >= N_pack) return;

  const int n0 = n_pack * 2;
  const int n1 = n0 + 1;

  float acc0 = 0.0f;
  float acc1 = 0.0f;

  const int num_groups = K / group_size;
  #pragma unroll 1
  for (int g = 0; g < num_groups; ++g) {
    int32_t dot0 = 0;
    int32_t dot1 = 0;
    const int k_begin = g * group_size;
    const int k_end = k_begin + group_size;

    #pragma unroll 4
    for (int k = k_begin; k < k_end; ++k) {
      const int8_t a = sh_a[k];
      const uint8_t p = b_q_packed[k * N_pack + n_pack];
      dot0 += static_cast<int32_t>(a) * static_cast<int32_t>(unpack_i4_low(p));
      dot1 += static_cast<int32_t>(a) * static_cast<int32_t>(unpack_i4_high(p));
    }

    const float ws0 = __half2float(b_scales[g * N + n0]);
    const float ws1 = __half2float(b_scales[g * N + n1]);
    acc0 += static_cast<float>(dot0) * ws0;
    acc1 += static_cast<float>(dot1) * ws1;
  }

  const float a_scale = a_scales[m];
  out[m * N + n0] = __float2half_rn(acc0 * a_scale);
  out[m * N + n1] = __float2half_rn(acc1 * a_scale);
}

__global__ void w4a8_tiny_kernel_v73_epilogue_atomic(
    const float* __restrict__ out_accum, const float* __restrict__ a_scales,
    half* __restrict__ out, int M, int N) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int total = M * N;
  if (idx >= total) return;

  const int m = idx / N;
  out[idx] = __float2half_rn(out_accum[idx] * a_scales[m]);
}

__global__ void w4a8_tiny_kernel_v73_epilogue_reduce(
    const float* __restrict__ out_partial, const float* __restrict__ a_scales,
    half* __restrict__ out, int M, int N, int split_k_parts) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int total = M * N;
  if (idx >= total) return;

  float sum = 0.0f;
  for (int p = 0; p < split_k_parts; ++p) {
    sum += out_partial[p * total + idx];
  }

  const int m = idx / N;
  out[idx] = __float2half_rn(sum * a_scales[m]);
}

}  // namespace

void w4a8_tiny_cuda(torch::Tensor a_q, torch::Tensor b_q_packed,
                    torch::Tensor b_scales, torch::Tensor a_scales,
                    torch::Tensor out, int64_t group_size) {
  const int M = static_cast<int>(a_q.size(0));
  const int K = static_cast<int>(a_q.size(1));
  const int N = static_cast<int>(out.size(1));

  TORCH_CHECK((N % 2) == 0, "N must be even for packed int4 layout");

  constexpr int THREADS = 256;
  dim3 block(THREADS);
  const int grid_x = ((N / 2) + THREADS - 1) / THREADS;

  auto stream = at::cuda::getDefaultCUDAStream();

  const char* splitk_max_m_env = std::getenv("TINY_SPLITK_MAX_M");
  const char* splitk_target_cta_env = std::getenv("TINY_SPLITK_TARGET_CTAS");
  const char* splitk_reduce_mode_env = std::getenv("TINY_SPLITK_REDUCE_MODE");
  const char* splitk_dynamic_env = std::getenv("TINY_SPLITK_DYNAMIC");

  int splitk_max_m = 4;
  int splitk_target_ctas = 128;
  if (splitk_max_m_env != nullptr) {
    splitk_max_m = std::max(1, std::atoi(splitk_max_m_env));
  }
  if (splitk_target_cta_env != nullptr) {
    splitk_target_ctas = std::max(1, std::atoi(splitk_target_cta_env));
  }

  const bool use_buffer_reduce =
      (splitk_reduce_mode_env != nullptr) &&
      (std::strcmp(splitk_reduce_mode_env, "buffer") == 0);

  bool use_splitk_dynamic = true;
  if (splitk_dynamic_env != nullptr) {
    use_splitk_dynamic = (std::atoi(splitk_dynamic_env) != 0);
  }

  const int base_ctas = std::max(1, M * grid_x);
  const bool use_splitk = (M <= splitk_max_m) ||
                          (use_splitk_dynamic && (base_ctas < splitk_target_ctas));

  if (use_splitk) {
    const int num_groups = K / static_cast<int>(group_size);
    int split_k_parts = (splitk_target_ctas + base_ctas - 1) / base_ctas;
    split_k_parts = std::max(1, std::min(num_groups, split_k_parts));

    dim3 grid(grid_x, M, split_k_parts);
    const size_t shmem_bytes = static_cast<size_t>(K) * sizeof(int8_t);

    if (use_buffer_reduce) {
      auto out_partial = torch::zeros({split_k_parts, M, N},
                                      a_q.options().dtype(torch::kFloat32));
      w4a8_tiny_kernel_v73_splitk_store<<<grid, block, shmem_bytes, stream>>>(
          reinterpret_cast<int8_t*>(a_q.data_ptr<int8_t>()),
          reinterpret_cast<uint8_t*>(b_q_packed.data_ptr<uint8_t>()),
          reinterpret_cast<half*>(b_scales.data_ptr<at::Half>()),
          reinterpret_cast<float*>(out_partial.data_ptr<float>()), M, N, K,
          static_cast<int>(group_size), split_k_parts);

      const int total = M * N;
      const int epilogue_threads = 256;
      const int epilogue_blocks = (total + epilogue_threads - 1) / epilogue_threads;
      w4a8_tiny_kernel_v73_epilogue_reduce<<<epilogue_blocks, epilogue_threads, 0,
                                             stream>>>(
          reinterpret_cast<float*>(out_partial.data_ptr<float>()),
          reinterpret_cast<float*>(a_scales.data_ptr<float>()),
          reinterpret_cast<half*>(out.data_ptr<at::Half>()), M, N, split_k_parts);
      return;
    }

    auto out_accum = torch::zeros({M, N}, a_q.options().dtype(torch::kFloat32));

    w4a8_tiny_kernel_v73_splitk_atomic<<<grid, block, shmem_bytes, stream>>>(
        reinterpret_cast<int8_t*>(a_q.data_ptr<int8_t>()),
        reinterpret_cast<uint8_t*>(b_q_packed.data_ptr<uint8_t>()),
        reinterpret_cast<half*>(b_scales.data_ptr<at::Half>()),
        reinterpret_cast<float*>(out_accum.data_ptr<float>()), M, N, K,
        static_cast<int>(group_size), split_k_parts);

    const int total = M * N;
    const int epilogue_threads = 256;
    const int epilogue_blocks = (total + epilogue_threads - 1) / epilogue_threads;
    w4a8_tiny_kernel_v73_epilogue_atomic<<<epilogue_blocks, epilogue_threads, 0,
                                           stream>>>(
        reinterpret_cast<float*>(out_accum.data_ptr<float>()),
        reinterpret_cast<float*>(a_scales.data_ptr<float>()),
        reinterpret_cast<half*>(out.data_ptr<at::Half>()), M, N);
    return;
  }

  dim3 grid(grid_x, M);
  const size_t shmem_bytes = static_cast<size_t>(K) * sizeof(int8_t);
  w4a8_tiny_kernel_v73_direct<<<grid, block, shmem_bytes, stream>>>(
      reinterpret_cast<int8_t*>(a_q.data_ptr<int8_t>()),
      reinterpret_cast<uint8_t*>(b_q_packed.data_ptr<uint8_t>()),
      reinterpret_cast<half*>(b_scales.data_ptr<at::Half>()),
      reinterpret_cast<float*>(a_scales.data_ptr<float>()),
      reinterpret_cast<half*>(out.data_ptr<at::Half>()), M, N, K,
      static_cast<int>(group_size));
}



