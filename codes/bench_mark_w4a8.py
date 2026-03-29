"""
Marlin/Cutlass benchmark with tiny W4A8 kernel comparison.

- W4A16 : ops.marlin_gemm (fp16 activation, int4 weight)
- W8A8  : ops.cutlass_scaled_mm (int8 activation, int8 weight)
- W4A8  : ops.marlin_gemm (int8 activation, int4 weight)
- TINY  : standalone tiny extension kernel (int8 activation, int4 weight)

Usage:
  CUDA_VISIBLE_DEVICES=3 python benchmarks/kernels/marlin_w4a8_bench_with_tiny.py     --mode all --shapes qwen --m 1 4 8 16
"""

import argparse
from pathlib import Path

import torch
import torch.utils.benchmark as benchmark
from torch.utils.cpp_extension import load

from vllm import _custom_ops as ops
from vllm.model_executor.layers.quantization.utils.marlin_utils import (
    USE_FP32_REDUCE_DEFAULT,
    marlin_act_int8_process_scales,
    marlin_make_workspace_new,
    marlin_quant_input,
    should_use_atomic_add_reduce,
)
from vllm.model_executor.layers.quantization.utils.marlin_utils_test import (
    marlin_quantize,
)
from vllm.scalar_type import scalar_types


// 根据Qwen2.5的模型配置，主要是投影，和Attention以及MLP的Shape
QWEN25_7B_SHAPES = [
    (3584, 3584),
    (3584, 512),
    (3584, 18944),
    (18944, 3584),
]

LLAMA2_7B_SHAPES = [
    (4096, 4096),
    (4096, 11008),
    (11008, 4096),
]

L40_BW_GBS = 864.0


def build_tiny_ext(tiny_dir: Path, ext_name: str, verbose: bool):
    cpp_file = tiny_dir / "w4a8_tiny.cpp"
    cu_file = tiny_dir / "w4a8_tiny_kernel.cu"
    if not cpp_file.exists() or not cu_file.exists():
        raise FileNotFoundError(
            f"tiny kernel source not found under {tiny_dir} "
            f"(expect w4a8_tiny.cpp and w4a8_tiny_kernel.cu)"
        )

    return load(
        name=ext_name,
        sources=[str(cpp_file), str(cu_file)],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        extra_cflags=["-O3"],
        verbose=verbose,
    )


def make_base_tensors(M: int, K: int, N: int):
    torch.manual_seed(42)
    w = torch.randn(K, N, dtype=torch.float16, device="cuda")
    a_fp16 = torch.randn(M, K, dtype=torch.float16, device="cuda")
    return w, a_fp16


def prepare_w8a8(w: torch.Tensor, a_fp16: torch.Tensor):
    w_scale = w.abs().max() / 127.0
    w_int8 = (w / w_scale).clamp(-128, 127).round().to(torch.int8)
    w_ref = w_int8.to(torch.float16) * w_scale

    a_scale = a_fp16.abs().max() / 127.0
    a_int8 = (a_fp16 / a_scale).clamp(-128, 127).round().to(torch.int8)

    # cutlass_scaled_mm requires b.stride(0) == 1
    n, k = w.shape[1], w.shape[0]
    w_int8_col = torch.empty(n, k, dtype=torch.int8, device="cuda").t()
    w_int8_col.copy_(w_int8)

    return {
        "a_int8": a_int8,
        "a_scale": a_scale.reshape(1).to(torch.float32),
        "w_int8": w_int8_col,
        "w_scale": w_scale.reshape(1).to(torch.float32),
        "w_ref": w_ref,
    }


def prepare_w4a16(w: torch.Tensor, a_fp16: torch.Tensor, group_size: int, m: int, n: int, k: int):
    w_ref, marlin_q_w, marlin_s, g_idx, sort_indices, _ = marlin_quantize(
        w, scalar_types.uint4b8, group_size, act_order=False, input_dtype=None
    )
    workspace = marlin_make_workspace_new(a_fp16.device)
    use_atomic_add = should_use_atomic_add_reduce(
        m=m, n=n, k=k, device=a_fp16.device, dtype=a_fp16.dtype
    )
    return {
        "a_fp16": a_fp16,
        "marlin_q_w": marlin_q_w,
        "marlin_s": marlin_s,
        "g_idx": g_idx,
        "sort_indices": sort_indices,
        "workspace": workspace,
        "use_atomic_add": use_atomic_add,
        "w_ref": w_ref,
    }


def prepare_w4a8_and_tiny(
    w: torch.Tensor,
    a_fp16: torch.Tensor,
    group_size: int,
    m: int,
    n: int,
    k: int,
):
    # Marlin W4A8 tensors
    w_ref, marlin_q_w, marlin_s_raw, g_idx, sort_indices, _ = marlin_quantize(
        w, scalar_types.uint4b8, group_size, act_order=False, input_dtype=torch.int8
    )
    marlin_s, input_global_scale = marlin_act_int8_process_scales(marlin_s_raw)
    x_int8, a_scales = marlin_quant_input(a_fp16, torch.int8)
    a_scales_final = a_scales * input_global_scale

    workspace = marlin_make_workspace_new(a_fp16.device)
    use_atomic_add = should_use_atomic_add_reduce(
        m=m, n=n, k=k, device=a_fp16.device, dtype=x_int8.dtype
    )

    # Tiny kernel tensors
    w_int4 = torch.round(w / (w.abs().amax(dim=0, keepdim=True) / 7.0 + 1e-6)).clamp(-8, 7).to(torch.int8)
    w_low = (w_int4[:, 0::2] & 0xF).to(torch.uint8)
    w_high = ((w_int4[:, 1::2] & 0xF) << 4).to(torch.uint8)
    b_q_packed = (w_low | w_high).contiguous()

    num_groups = k // group_size
    w_scales = torch.empty(num_groups, n, dtype=torch.float16, device="cuda")
    w_ref_tiny = torch.empty_like(w)
    for g in range(num_groups):
        ks = g * group_size
        ke = (g + 1) * group_size
        chunk = w[ks:ke]
        ws = chunk.abs().amax(dim=0) / 7.0
        ws_h = ws.to(torch.float16)
        w_scales[g] = ws_h
        w_ref_tiny[ks:ke] = (w_int4[ks:ke].to(torch.float32) * ws_h.to(torch.float32)).to(torch.float16)

    return {
        "x_int8": x_int8,
        "a_scales_final": a_scales_final,
        "marlin_q_w": marlin_q_w,
        "marlin_s": marlin_s,
        "g_idx": g_idx,
        "sort_indices": sort_indices,
        "workspace": workspace,
        "use_atomic_add": use_atomic_add,
        "w_ref": w_ref,
        "b_q_packed": b_q_packed,
        "b_scales": w_scales,
        "w_ref_tiny": w_ref_tiny,
    }


def run_w8a8(inp: dict):
    return ops.cutlass_scaled_mm(
        inp["a_int8"],
        inp["w_int8"],
        inp["a_scale"],
        inp["w_scale"],
        torch.float16,
    )


def run_w4a16(inp: dict, m: int, n: int, k: int):
    return ops.marlin_gemm(
        inp["a_fp16"],
        None,
        inp["marlin_q_w"],
        None,
        inp["marlin_s"],
        None,
        None,
        None,
        inp["g_idx"],
        inp["sort_indices"],
        inp["workspace"],
        scalar_types.uint4b8,
        size_m=m,
        size_n=n,
        size_k=k,
        is_k_full=True,
        use_atomic_add=inp["use_atomic_add"],
        use_fp32_reduce=USE_FP32_REDUCE_DEFAULT,
        is_zp_float=False,
    )


def run_w4a8(inp: dict, m: int, n: int, k: int):
    return ops.marlin_gemm(
        inp["x_int8"],
        None,
        inp["marlin_q_w"],
        None,
        inp["marlin_s"],
        inp["a_scales_final"],
        None,
        None,
        inp["g_idx"],
        inp["sort_indices"],
        inp["workspace"],
        scalar_types.uint4b8,
        size_m=m,
        size_n=n,
        size_k=k,
        is_k_full=True,
        use_atomic_add=inp["use_atomic_add"],
        use_fp32_reduce=USE_FP32_REDUCE_DEFAULT,
        is_zp_float=False,
    )


def run_tiny(tiny_ext, inp: dict, group_size: int):
    return tiny_ext.w4a8_tiny(
        inp["x_int8"],
        inp["b_q_packed"],
        inp["b_scales"],
        inp["a_scales_final"],
        group_size,
    )


def bench_latency_us(fn, warmup: int = 20, repeat: int = 200):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t = benchmark.Timer(
        stmt="fn(); torch.cuda.synchronize()",
        globals={"fn": fn, "torch": torch},
        num_threads=1,
    )
    return t.timeit(repeat).mean * 1e6


def compute_tflops(m: int, k: int, n: int, lat_us: float):
    return 2 * m * k * n / (lat_us * 1e-6) / 1e12


def estimate_bytes_per_gemm(m: int, k: int, n: int, mode: str, group_size: int = 128):
    a_bytes_fp16 = m * k * 2
    a_bytes_int8 = m * k * 1
    c_bytes_fp16 = m * n * 2

    if mode == "w8a8":
        w_bytes = k * n
        s_bytes = 8
        a_bytes = a_bytes_int8
    else:
        w_bytes = (k * n) // 2
        s_bytes = (k // group_size) * n * 2
        a_bytes = a_bytes_fp16 if mode == "w4a16" else a_bytes_int8

    return a_bytes + w_bytes + s_bytes + c_bytes_fp16


def compute_eff_bw_gbs(m: int, k: int, n: int, lat_us: float, mode: str, group_size: int = 128):
    total_bytes = estimate_bytes_per_gemm(m, k, n, mode, group_size)
    return total_bytes / (lat_us * 1e-6) / 1e9


def accuracy_stats(out: torch.Tensor, ref: torch.Tensor):
    out_f = out.float()
    ref_f = ref.float()
    diff = (out_f - ref_f).abs()
    max_abs = diff.max().item()
    cos = torch.nn.functional.cosine_similarity(out_f, ref_f, dim=-1).mean().item()
    return max_abs, cos


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--m", type=int, nargs="+", default=[1, 4, 8, 16, 32, 64, 128])
    parser.add_argument("--shapes", choices=["qwen", "llama", "all"], default="qwen")
    parser.add_argument("--group-size", type=int, default=128)
    parser.add_argument("--mode", choices=["w4a16", "w8a8", "w4a8", "all"], default="all")
    parser.add_argument("--no-accuracy", action="store_true")
    parser.add_argument(
        "--tiny-dir",
        type=str,
        default="/opt/workdir/vllm/standalone_w4a8_tiny_v73",
        help="directory containing w4a8_tiny.cpp and w4a8_tiny_kernel.cu",
    )
    parser.add_argument(
        "--tiny-ext-name",
        type=str,
        default="w4a8_tiny_ext_bench_v73",
        help="torch extension module name for tiny kernel",
    )
    parser.add_argument("--tiny-verbose", action="store_true")
    parser.add_argument("--disable-tiny", action="store_true")
    args = parser.parse_args()

    shapes = (
        QWEN25_7B_SHAPES
        if args.shapes == "qwen"
        else LLAMA2_7B_SHAPES
        if args.shapes == "llama"
        else QWEN25_7B_SHAPES + LLAMA2_7B_SHAPES
    )
    modes = ["w4a16", "w8a8", "w4a8"] if args.mode == "all" else [args.mode]

    tiny_ext = None
    if "w4a8" in modes and not args.disable_tiny:
        tiny_ext = build_tiny_ext(Path(args.tiny_dir), args.tiny_ext_name, args.tiny_verbose)

    print(f"\n{'='*122}")
    print(
        f"  Marlin/Cutlass/Tiny Benchmark | group_size={args.group_size} "
        f"| GPU: {torch.cuda.get_device_name(0)}"
    )
    print("  W4A16/W4A8 -> ops.marlin_gemm | W8A8 -> ops.cutlass_scaled_mm | TINY -> standalone kernel")
    print(f"{'='*122}")

    for mode in modes:
        print(f"\n-- {mode.upper()} {'-'*92}")
        if mode == "w4a8" and tiny_ext is not None:
            print(
                f"  {'Shape(K,N)':18s} {'M':>5s} {'Marlin(us)':>11s} {'Tiny(us)':>10s} {'SpdUp':>8s}"
                f" {'TFLOPS':>8s} {'EffBW':>10s} {'BW%':>8s} {'MaxErr':>10s} {'CosSim':>10s} {'TinyCos':>10s}"
            )
            print(f"  {'-'*118}")
        else:
            print(
                f"  {'Shape(K,N)':18s} {'M':>5s} {'Lat(us)':>10s} {'TFLOPS':>8s}"
                f" {'EffBW':>10s} {'BW%':>8s} {'MaxErr':>10s} {'CosSim':>10s}"
            )
            print(f"  {'-'*92}")

        for k, n in shapes:
            for m in args.m:
                w, a_fp16 = make_base_tensors(m, k, n)

                if mode == "w8a8":
                    inp = prepare_w8a8(w, a_fp16)
                    fn = lambda inp=inp: run_w8a8(inp)
                    lat = bench_latency_us(fn)
                    out = fn()
                    out_ref = torch.matmul(a_fp16.float(), inp["w_ref"].float())
                    max_err, cos = accuracy_stats(out, out_ref)
                    tflops = compute_tflops(m, k, n, lat)
                    eff_bw = compute_eff_bw_gbs(m, k, n, lat, mode, args.group_size)

                elif mode == "w4a16":
                    inp = prepare_w4a16(w, a_fp16, args.group_size, m, n, k)
                    fn = lambda inp=inp, m=m, n=n, k=k: run_w4a16(inp, m, n, k)
                    lat = bench_latency_us(fn)
                    out = fn()
                    out_ref = torch.matmul(a_fp16.float(), inp["w_ref"].float())
                    max_err, cos = accuracy_stats(out, out_ref)
                    tflops = compute_tflops(m, k, n, lat)
                    eff_bw = compute_eff_bw_gbs(m, k, n, lat, mode, args.group_size)

                else:
                    inp = prepare_w4a8_and_tiny(w, a_fp16, args.group_size, m, n, k)
                    fn_marlin = lambda inp=inp, m=m, n=n, k=k: run_w4a8(inp, m, n, k)
                    lat = bench_latency_us(fn_marlin)
                    out = fn_marlin()
                    out_ref = torch.matmul(a_fp16.float(), inp["w_ref"].float())
                    max_err, cos = accuracy_stats(out, out_ref)
                    tflops = compute_tflops(m, k, n, lat)
                    eff_bw = compute_eff_bw_gbs(m, k, n, lat, mode, args.group_size)

                    tiny_lat = None
                    tiny_speedup = None
                    tiny_cos = None
                    if tiny_ext is not None:
                        fn_tiny = lambda tiny_ext=tiny_ext, inp=inp, gs=args.group_size: run_tiny(tiny_ext, inp, gs)
                        tiny_lat = bench_latency_us(fn_tiny)
                        tiny_out = fn_tiny()
                        tiny_ref = torch.matmul(a_fp16.float(), inp["w_ref_tiny"].float())
                        _, tiny_cos = accuracy_stats(tiny_out, tiny_ref)
                        tiny_speedup = lat / tiny_lat

                bw_ratio = eff_bw / L40_BW_GBS * 100.0
                bw_flag = "*" if eff_bw > L40_BW_GBS else " "

                if args.no_accuracy:
                    max_err_str = "-"
                    cos_str = "-"
                else:
                    max_err_str = f"{max_err:.5f}"
                    cos_str = f"{cos:.6f}"

                if mode == "w4a8" and tiny_ext is not None:
                    tiny_cos_str = "-" if args.no_accuracy or tiny_cos is None else f"{tiny_cos:.6f}"
                    print(
                        f"  ({k:5d},{n:5d}) {m:6d} {lat:11.2f} {tiny_lat:10.2f} {tiny_speedup:8.3f}"
                        f" {tflops:8.3f} {eff_bw:10.1f}{bw_flag} {bw_ratio:7.1f}%"
                        f" {max_err_str:>10s} {cos_str:>10s} {tiny_cos_str:>10s}"
                    )
                else:
                    print(
                        f"  ({k:5d},{n:5d}) {m:6d} {lat:10.2f} {tflops:8.3f}"
                        f" {eff_bw:10.1f}{bw_flag} {bw_ratio:7.1f}%"
                        f" {max_err_str:>10s} {cos_str:>10s}"
                    )
            print()

    print(f"L40 theoretical peak: INT8 362 TOPS | Mem BW {L40_BW_GBS:.0f} GB/s")
    print("decode (small M): usually memory-bound, watch latency and speedup")
    print("prefill (large M): usually compute-bound, watch TFLOPS")
    print("* EffBW is model-estimated effective bandwidth, not raw DRAM counter.\n")


if __name__ == "__main__":
    main()

