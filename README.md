# W4A8-INT GEMM 优化实验报告（L40 / SM89）

> **硬件**：NVIDIA L40（SM89, Ada Lovelace，864 GB/s，362 TOPS INT8）  
> **目标**：优化 W4A8（INT4 weight × INT8 activation）在 decode 小 M 场景的性能  
> **基准形状**：`(M, K, N) = (1/4/8, 3584, 512)`（Qwen2.5-7B 典型小投影层）  
> **对照基准**：vLLM Marlin kernel（`ops.marlin_gemm`）

---

## 一、背景与问题定义

### 1.1 量化模式说明

| 模式 | Weight | Activation | Kernel |
|------|--------|-----------|--------|
| W4A16 | INT4 | FP16 | `ops.marlin_gemm` |
| W8A8  | INT8 | INT8 | `ops.cutlass_scaled_mm` |
| W4A8  | INT4 | INT8 | `ops.marlin_gemm` |

W4A8 是 W4A16 和 W8A8 的结合：权重压缩到 4bit 节省内存，激活量化到 8bit 利用 Tensor Core `mma.sync.m16n8k32.s8.s8.s32` 指令（k_size=32，比 FP16 的 k_size=16 理论吞吐翻倍）。

### 1.2 vLLM 中的 Bug（已提 PR）

在着手优化之前，先发现并修复了 vLLM 上游的两个 Bug：

**Bug 1：W4A8-INT 静默退化为 W4A16**  
- 文件：`compressed_tensors_w4a8_int.py`  
- 原因：`create_weights` 里 `act_type=params_dtype`（bf16），activation 从不被量化  
- 效果：整条路径走 BF16 mma（k_size=16），而非 INT8 mma（k_size=32）

**Bug 2：INT4 weight type 无法进入 Marlin 路径**  
- 文件：`marlin_utils.py`  
- 原因：`query_marlin_supported_quant_types` 不含 `scalar_types.int4`，`can_implement` 直接返回 False

**修复 PR**：[vllm-project/vllm#38066](https://github.com/vllm-project/vllm/pull/38066)（Fixes #38063, #38064）  
修复后激活 dtype 从 BF16 → INT8，走到正确的 `mma.sync.aligned.m16n8k32.s8.s8.s32.satfinite` 指令。

---

## 二、性能现状与瓶颈分析

> 以下数据均为 **PR #38066 修复后**的真实测量值（L40，group_size=128，Qwen2.5-7B shapes）。

### 2.1 完整 Benchmark 数据

#### W4A16（基准）

| Shape (K,N) | M | Lat(μs) | TFLOPS | EffBW(GB/s) | BW% |
|------------|---|---------|--------|------------|-----|
| (3584, 3584) | 1 | 24.06 | 1.07 | 275.9 | 31.9% |
| (3584, 3584) | 4 | 24.49 | 4.20 | 272.8 | 31.6% |
| (3584, 3584) | 8 | 24.70 | 8.32 | 272.8 | 31.6% |
| (3584, 3584) | 16 | 26.19 | 15.70 | 261.7 | 30.3% |
| (3584, 512) | 1 | 29.00 | 0.13 | 32.9 | 3.8% |
| (3584, 512) | 4 | 28.71 | 0.51 | 34.1 | 3.9% |
| (3584, 512) | 8 | 28.41 | 1.03 | 35.6 | 4.1% |
| (3584, 18944) | 1 | 32.43 | 4.19 | 1081.0 | 125.1% |
| (18944, 3584) | 1 | 33.53 | 4.05 | 1045.4 | 121.0% |

#### W8A8（cutlass_scaled_mm）

| Shape (K,N) | M | Lat(μs) | BW% |
|------------|---|---------|-----|
| (3584, 3584) | 1 | 24.60 | 60.5% |
| (3584, 3584) | 16 | 24.69 | 61.0% |
| (3584, 512) | 1 | 23.98 | 8.9% |
| (3584, 18944) | 1 | 42.51 | — |
| (18944, 3584) | 1 | 43.44 | — |

#### W4A8 Marlin（修复后）vs 本 Tiny Kernel

| Shape (K,N) | M | Marlin(μs) | Tiny(μs) | SpdUp | BW% | CosSim |
|------------|---|-----------|---------|-------|-----|--------|
| (3584, 3584) | 1 | 23.62 | 31.48 | 0.750 | 32.5% | 0.999964 |
| (3584, 3584) | 4 | 23.43 | 57.35 | 0.409 | 32.9% | 0.999964 |
| (3584, 3584) | 8 | 23.50 | 94.56 | 0.249 | 33.0% | 0.999964 |
| (3584, 3584) | 16 | 23.67 | 129.07 | 0.183 | 33.2% | 0.999964 |
| **(3584, 512)** | **1** | **28.37** | **23.98** | **1.183 ✅** | 3.9% | 0.999965 |
| **(3584, 512)** | **4** | **28.39** | **24.16** | **1.175 ✅** | 3.9% | 0.999968 |
| (3584, 512) | 8 | 28.19 | 30.72 | 0.917 | 4.0% | 0.999966 |
| (3584, 512) | 16 | 28.46 | 43.52 | 0.654 | 4.1% | 0.999965 |
| (3584, 18944) | 1 | 30.29 | 72.11 | 0.420 | 133.9% | 0.999955 |
| (3584, 18944) | 4 | 30.22 | 225.87 | 0.134 | 134.7% | 0.999960 |
| (18944, 3584) | 1 | 32.60 | 79.92 | 0.408 | 124.4% | 0.999952 |
| (18944, 3584) | 4 | 32.57 | 224.41 | 0.145 | 124.8% | 0.999954 |

### 2.2 Roofline 分析（(3584,3584) M=1）

```
总数据量 ≈ 6.61 MB（INT4 weight 6.40MB + scale 0.196MB + act/out ~0.01MB）
理论最快 = 6.61 MB / 864 GB/s ≈ 7.6μs

实测（修复后）:
  W4A16: 24.06μs → 效率 31.6%
  W4A8:  23.62μs → 效率 32.2%  ← 修复后与 W4A16 基本持平
```

> 修复前（Bug 状态下）W4A8 = ~64μs（仅 12% BW），走的是 BF16 mma 而非 INT8 mma。
> 修复后 W4A8 已恢复正常，(3584,3584) 场景与 W4A16 延迟几乎一致。

### 2.3 小 N 场景的结构性瓶颈

**(3584, 512) 的特殊性**：

```
Marlin tile_N = 128，tile_M = 16

Grid（M=1, N=512）:
  grid_x = 512 / 128 = 4 个 block（N 方向）
  grid_y = ceil(1/16) = 1（M 方向）
  → 只有 4 个 SM 工作，L40 共 144 个 SM
  → SM 利用率仅 2.8%
```

这是 **wave quantization**（波浪量化）问题——N 太小，切不出足够多的 tile 来填满 SM。
Marlin 在这个 shape 上也只有 3.9% BW，从 kernel 设计角度难以根本解决。

本 Tiny kernel 通过 **split-K** 从 K 维补充并行度，在 M=1/4 时实现了约 **18% 的提升**。

---

## 三、自研 Kernel 优化历程（v1 → v7）

针对 (3584, 512) 小 M 场景，从头写了一个 W4A8 GEMM kernel，逐步迭代。

### v1：朴素标量实现（baseline）

**方案**：CPU 风格逐元素展开，无并行优化。  
**结果**：~500μs+  
**价值**：仅用于验证正确性。

---

### v2：Shared Memory 行缓存 + group 累加

**方案**：把 A 矩阵行缓存到 shared memory，K 内循环按 group 累加，减少全局内存反复读取。  
**结果**：M=1 → **~162μs**  
**结论**：相比 v1 大幅提升，建立稳定基线。

---

### v3：激进展开 / 重排（失败探索）

**方案**：更大粒度的寄存器展开、重排访存顺序。  
**结果**：退化到 **~420μs**  
**结论**：寄存器压力过高，occupancy 下滑。激进展开在小 M 场景弊大于利。

---

### v4：CTA 256 线程（标量路径下限）

**方案**：回到 v2 算法路径，将 CTA 提升到 256 threads。  
**结果**：M=1 → **~158μs**  
**结论**：标量路径可压到的下限，指令层面的调整不会再有突破。

---

### v5：双行 M 复用（失败探索）

**方案**：一个 CTA 同时计算两行 M，B 矩阵复用以减少带宽。  
**结果**：**~182μs**，反而变慢  
**结论**：控制流与寄存器成本大于复用收益。

---

### v6：dp4a 向量化

**方案**：K 内环引入 `__dp4a` 指令，4 元素并行整数点积。  
**结果**：**~178μs**，仍未超过 v4  
**结论**：单条指令优化被系统开销淹没，瓶颈不在此处。

---

### v7：Split-K + Atomic + Epilogue（结构性突破）

**方案**：将 K 维切分为多段，每段由独立 SM 负责，输出用 atomic add 归约。两阶段：

1. **主 kernel**：每个 SM 计算部分和，写到 workspace buffer（同时使用 `__dp4a` 向量化）
2. **epilogue**：归约部分和，乘 activation scale，写出最终 fp16 结果

**结果（(M, K=3584, N=512)）**：

| M | 策略 | Tiny kernel | Marlin W4A8 | 提升 |
|---|------|------------|-------------|------|
| M=1 | split-K | **23.98μs** | 28.37μs | **1.18×** |
| M=4 | split-K | **24.16μs** | 28.39μs | **1.18×** |
| M=8 | split-K（动态）| 30.72μs | 28.19μs | 0.92× |
| M=16 | split-K | 43.52μs | 28.46μs | 0.65× |

```
v4（标量）: ~158μs → v7（split-K）: ~24μs，提升 6.6×
```

**核心原理**：split-K 将 `grid = (4, 1)` → `grid = (4, 1, 28)`（K_splits=28），
让 28 个 SM 同时处理 K 方向不同分段，彻底改变调度结构。

---

### v7.1 / v7.2 / v7.3：自适应调度策略

**v7.1**：自适应分支——小 M 走 split-K，大 M 走 direct  
**v7.2**：参数化策略——`TINY_SPLITK_MAX_M`、`TINY_SPLITK_TARGET_CTAS` 环境变量控制  
**v7.3**：动态判定——`base_ctas < target_ctas` 时自动启用 split-K；支持 atomic / buffer 两种归约模式

---

## 四、精度分析

| 指标 | 数值 | 说明 |
|------|------|------|
| CosSim（Tiny vs 参考） | ≈ 0.999964~0.999969 | 与量化参考高度对齐 ✅ |
| MaxErr（绝对误差） | ≈ 1.59~2.09 | 量化误差在合理范围内 |

精度与 Marlin W4A8 输出完全一致（TinyCos ≈ MarlinCosSim），全版本无 NaN / 发散。

---

## 五、Tiny Kernel 的适用范围

| 场景 | SpdUp vs Marlin | 说明 |
|------|----------------|------|
| N=512, M=1/4 | **~1.18×** | split-K 有效，推荐使用 |
| N=512, M=8 | ~0.92× | 接近持平 |
| N=512, M≥16 | <0.65× | split-K atomic 开销过大 |
| N=3584, M=任意 | 0.18~0.75× | Marlin smem pipeline 优势明显 |
| N=18944, M≥4 | <0.15× | Tiny 不适用 |

**结论**：Tiny kernel 的优势场景是 **N 极小（≤512）+ M 极小（≤4）** 的 decode 场景，其余情况 Marlin 更优。

---

## 六、核心认知总结

| 认知 | 说明 |
|------|------|
| **瓶颈是调度，不是算术** | decode 小 M + 小 N 下，mma 利用率和 dp4a 优化效果微乎其微；SM 并行度是决定因素 |
| **split-K 是小 N 场景的正确路径** | N 小时 tile 数不足，只能从 K 维补充 SM 并行度 |
| **atomic 代价可接受** | SM 充分利用后，atomic reduce 开销被完全覆盖 |
| **Marlin 大 N 时有 smem pipeline 优势** | Tiny kernel 缺乏 double buffering，大 N 时全局内存延迟完全暴露 |
| **m_block_size=8 是 Marlin 的独立 bug** | 已在 PR #38066 修复，修复后 Marlin W4A8 在大 N 下已恢复正常 |

---

## 七、后续方向

- [ ] **double buffering**：为 Tiny kernel 增加 smem pipeline，改善大 N 场景
- [ ] **Marlin m_block_size=8 PR**：提交独立 PR，影响 W4A8-INT / W4A8-FP8 / MXFP4-FP8 所有变体
- [ ] **autotuning**：针对不同 (M,K,N,group_size) 枚举 split-K 配置，输出最优参数 JSON
- [ ] **更大 M 场景**：M=64/128 的 prefill 路径分析

---

## 八、环境与参考

| 项目 | 说明 |
|------|------|
| GPU | NVIDIA L40（SM89，Ada Lovelace） |
| 显存带宽 | 864 GB/s |
| INT8 算力峰值 | 362 TOPS |
| 框架 | vLLM（自定义分支 `yeshihai:feat/marlin-w4a8-int`）|
| 相关 PR | [vllm-project/vllm#38066](https://github.com/vllm-project/vllm/pull/38066) |
| 相关 Issue | [#38063](https://github.com/vllm-project/vllm/issues/38063)，[#38064](https://github.com/vllm-project/vllm/issues/38064) |
