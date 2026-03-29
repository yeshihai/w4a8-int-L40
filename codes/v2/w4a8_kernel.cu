#include <ATen/cuda/CUDAContext.h>
#include <cuda.h>
#include <cuda_fp16.h>
#include <torch/extension.h>

namespace {

__device__ __forceinline__ int8_t unpack_i4_low(uint8_t packed) {
  int v = packed & 0xF;
  return static_cast<int8_t>(v >= 8 ? (v - 16) : v);
}

__device__ __forceinline__ int8_t unpack_i4_high(uint8_t packed) {
  int v = packed >> 4;
  return static_cast<int8_t>(v >= 8 ? (v - 16) : v);
}

__global__ void w4a8_tiny_kernel(const int8_t* __restrict__ a_q,
                                 const uint8_t* __restrict__ b_q_packed,
                                 const half* __restrict__ b_scales,
                                 const float* __restrict__ a_scales,
                                 half* __restrict__ out, int M, int N, int K,
                                 int group_size) {
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

}  // namespace

void w4a8_tiny_cuda(torch::Tensor a_q, torch::Tensor b_q_packed,
                    torch::Tensor b_scales, torch::Tensor a_scales,
                    torch::Tensor out, int64_t group_size) {
  const int M = static_cast<int>(a_q.size(0));
  const int K = static_cast<int>(a_q.size(1));
  const int N = static_cast<int>(out.size(1));

  TORCH_CHECK((N % 2) == 0, "N must be even for packed int4 layout");

  constexpr int THREADS = 128;
  dim3 block(THREADS);
  dim3 grid(((N / 2) + THREADS - 1) / THREADS, M);

  const size_t shmem_bytes = static_cast<size_t>(K) * sizeof(int8_t);

  auto stream = at::cuda::getDefaultCUDAStream();
  w4a8_tiny_kernel<<<grid, block, shmem_bytes, stream>>>(
      reinterpret_cast<int8_t*>(a_q.data_ptr<int8_t>()),
      reinterpret_cast<uint8_t*>(b_q_packed.data_ptr<uint8_t>()),
      reinterpret_cast<half*>(b_scales.data_ptr<at::Half>()),
      reinterpret_cast<float*>(a_scales.data_ptr<float>()),
      reinterpret_cast<half*>(out.data_ptr<at::Half>()), M, N, K,
      static_cast<int>(group_size));
}



