#include <torch/extension.h>

void w4a8_tiny_cuda(torch::Tensor a_q, torch::Tensor b_q_packed,
                    torch::Tensor b_scales, torch::Tensor a_scales,
                    torch::Tensor out, int64_t group_size);

torch::Tensor w4a8_tiny(torch::Tensor a_q, torch::Tensor b_q_packed,
                        torch::Tensor b_scales, torch::Tensor a_scales,
                        int64_t group_size) {
  TORCH_CHECK(a_q.is_cuda(), "a_q must be CUDA tensor");
  TORCH_CHECK(b_q_packed.is_cuda(), "b_q_packed must be CUDA tensor");
  TORCH_CHECK(b_scales.is_cuda(), "b_scales must be CUDA tensor");
  TORCH_CHECK(a_scales.is_cuda(), "a_scales must be CUDA tensor");

  TORCH_CHECK(a_q.scalar_type() == at::kChar, "a_q must be int8");
  TORCH_CHECK(b_q_packed.scalar_type() == at::kByte,
              "b_q_packed must be uint8");
  TORCH_CHECK(b_scales.scalar_type() == at::kHalf,
              "b_scales must be fp16");
  TORCH_CHECK(a_scales.scalar_type() == at::kFloat,
              "a_scales must be fp32");

  TORCH_CHECK(a_q.dim() == 2, "a_q must be [M, K]");
  TORCH_CHECK(b_q_packed.dim() == 2, "b_q_packed must be [K, N/2]");
  TORCH_CHECK(b_scales.dim() == 2, "b_scales must be [K/group_size, N]");
  TORCH_CHECK(a_scales.dim() == 1 || a_scales.dim() == 2,
              "a_scales must be [M] or [M,1]");

  const auto M = a_q.size(0);
  const auto K = a_q.size(1);
  const auto N = b_q_packed.size(1) * 2;

  TORCH_CHECK(a_scales.size(0) == M, "a_scales shape mismatch");
  if (a_scales.dim() == 2) {
    TORCH_CHECK(a_scales.size(1) == 1, "a_scales second dim must be 1");
  }
  TORCH_CHECK(K % group_size == 0, "K must be divisible by group_size");
  TORCH_CHECK(b_scales.size(0) == K / group_size,
              "b_scales first dim mismatch");
  TORCH_CHECK(b_scales.size(1) == N, "b_scales second dim mismatch");

  auto out = torch::empty({M, N}, a_q.options().dtype(torch::kFloat16));
  w4a8_tiny_cuda(a_q, b_q_packed, b_scales, a_scales, out, group_size);
  return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("w4a8_tiny", &w4a8_tiny, "W4A8 tiny kernel (CUDA)");
}

