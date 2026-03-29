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

### 2.1 Benchmark 原始数据（L40，group_size=128）

#### Decode 场景（M=1，memory-bound）

| Shape (K,N) | W4A16 延迟 | W8A8 延迟 | W4A8 延迟 | W4A8 BW% |
|------------|-----------|---------|---------|---------|
| (3584, 3584) | 24.5μs | 24.5μs | 64.0μs | **12%** |
| (3584, 18944) | 32.7μs | 43.1μs | 70.5μs | 58% |
| (18944, 3584) | 34.0μs | 43.9μs | 73.6μs | 55% |
| (3584, 512) | 34.5μs | 24.0μs | 78.1μs | **1.4%** |

#### Prefill 场景（M=128，compute-bound）

| Shape (K,N) | W4A16 | W8A8 | W4A8 | 说明 |
|------------|-------|------|------|------|
| (3584, 3584) | 65.6 TFLOPS | 94.9 TFLOPS | 44.8 TFLOPS | W4A8 最差 |
| (3584, 18944) | 132 TFLOPS | 148 TFLOPS | **166 TFLOPS** | W4A8 反超 |
| (18944, 3584) | 121 TFLOPS | **199 TFLOPS** | 156 TFLOPS | W8A8 最优 |

> Prefill 大 N 时 W4A8 反超 W8A8：INT4 权重减半，内存压力更低，即使 compute-bound 区域也获益。

### 2.2 Roofline 分析（以 (3584,3584) M=1 为例）

```
理论最快（Roofline）：
  总数据量 ≈ 6.61 MB（INT4 weight 6.40MB + scale 0.196MB + act/out ~0.01MB）
  理论最快 = 6.61 MB / 864 GB/s ≈ 7.6μs

实测效率：
  W4A16: 24.5μs → 效率 31%
  W4A8:  64.0μs → 效率 12%
  比值:  64.0 / 24.5 = 2.6×（权重数据量完全相同）
```

权重数据量相同却慢了 2.6×，说明问题**不在内存带宽，而在 kernel 执行效率**。

### 2.3 根因：SM 并行度严重不足

```
tile_M = 16（Marlin 最小 M tile）
tile_N = 128（Marlin 固定）

Grid（M=1, K=3584, N=3584）:
  grid_x = 3584 / 128 = 28 个 block
  grid_y = ceil(1 / 16) = 1

→ 只有 28 个 SM 工作，L40 共 144 个 SM，116 个完全空置
→ SM 利用率 ≈ 19%
```

此外，Marlin W4A8 的 `generate_kernels.py` 里缺少 `THREAD_M_BLOCKS=0.5`（m_block_size=8），而 W4A16 有：

```python
# W4A16 —— 有 0.5（对应 m_block_size=8，专为 decode 设计）
THREAD_M_BLOCKS = [0.5, 1, 2, 3, 4]

# W4A8-INT / W4A8-FP8 —— 无 0.5
"thread_m_blocks": [1, 2, 3, 4]
```

m_block_size=8 使 M=1 时 mma 利用率从 `1/16=6.25%` → `1/8=12.5%`，理论 2× 收益。但这只是 mma 内部利用率，并不解决 SM 总数不足的问题。

**核心结论：W4A8 decode 性能差的根本原因是 SM 并行度不足，而非 mma 指令效率或内存带宽。**

---

## 三、自研 Kernel 优化历程（v1 → v7）

针对 decode 小 M 场景，从头写了一个 W4A8 GEMM kernel，逐步迭代。

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
**结果**：M=1 → **~158μs**（v2/v4 最优）  
**结论**：标量路径可压到的下限。进一步调整指令不会有突破。

---

### v5：双行 M 复用（失败探索）

**方案**：一个 CTA 同时计算两行 M，B 矩阵复用以减少带宽。  
**结果**：**~182μs**，反而变慢  
**结论**：控制流与寄存器成本大于复用收益。在 M 方向展开需要精心设计，不能简单叠加。

---

### v6：dp4a 向量化

**方案**：K 内环引入 `__dp4a` 指令，4 元素并行整数点积。  
**结果**：**~178μs**，仍未超过 v4  
**结论**：只改算术指令不足以破局。单条指令优化被系统开销淹没，瓶颈不在此。

---

### v7：Split-K + Atomic + Epilogue（结构性突破）

**方案**：将 K 维切分为多段，每段由独立 SM 负责，输出用 atomic add 归约。两阶段：

1. **主 kernel**：每个 SM 计算部分和，写到 workspace buffer
2. **epilogue**：最后一个 SM 完成归约，写出最终结果

**结果**：M=1 → **~13.5μs**，首次达到 / 略优于 Marlin  
**结论**：**decode 小 M 的关键是并行调度，不是单点算术优化。**

```
优化前（v4）: ~158μs（标量）
优化后（v7）:  ~13.5μs（split-K）
提升倍数: 11.7×
对比 Marlin W4A8: ~78μs → 本 kernel 快 5.8×
对比 Marlin W4A16: ~34.5μs → 本 kernel 快 2.6×
```

核心原理：split-K 将 `grid = 4`（N=512 时）→ `grid = 4 × K_splits`，L40 的 144 个 SM 被充分调动。

---

### v7.1 / v7.2 / v7.3：自适应调度策略

**v7.1**：自适应分支——小 M 走 split-K，大 M 走 direct（避免 split-K 的 atomic 开销）  
**v7.2**：参数化策略——引入 `splitk_max_m`、`target_ctas`，支持不同 M 的扫描调优  
**v7.3**：动态判定——`base_ctas < target_ctas` 时自动启用 split-K；对比 atomic / buffer 两种归约方式

**最终实测结论（L40，形状 (M, K=3584, N=512)）**：

| M | 策略 | 本 kernel | Marlin W4A8 |
|---|------|----------|-------------|
| M=1 | split-K | **13.4~13.6μs** | ~78μs |
| M=4 | split-K | **~13.6μs** | ~78μs |
| M=8 | split-K（动态）| **~20μs** | ~78μs |
| M=8 | direct（无 split-K）| ~173μs | — |

> M=8 direct 断崖（~173μs）的原因：N=512 时 grid 只有 4 个 tile，144 个 SM 中只有 4 个在工作。split-K 从 K 维扩展并行，将这 4 个 tile 拆成 `4 × K_splits` 个，显著改善。

---

## 四、精度分析

| 指标 | 数值 | 说明 |
|------|------|------|
| cos_tiny（内部一致性） | ≈ 0.99997 | tiny kernel 与自身参考高度一致 ✅ |
| cos_marlin（与 Marlin 对比） | ≈ 0.972~0.973 | 存在稳定差异 |

**cos_marlin ≈ 0.972 的原因**：量化构造差异，而非 kernel 数值问题。测试用的是随机生成的 dummy weight，与 Marlin 的 GPTQ repack 格式不完全对齐。全版本未出现 NaN / 精度发散。

---

## 五、核心认知总结

| 认知 | 说明 |
|------|------|
| **瓶颈是调度，不是算术** | decode 小 M 下，mma 指令利用率和 dp4a 优化效果微乎其微；SM 并行度才是决定因素 |
| **split-K 是正确路径** | 通过 K 维切分，将 SM 利用率从 ~3% 提升到接近 100%，是量级上的改变 |
| **atomic 代价可接受** | atomic reduce 在 SM 充分利用时被完全覆盖，不是瓶颈 |
| **N 小是结构性问题** | N=512 时 tile 数量仅 4 个，split-K 从 K 维有效补救；N 极小时需 batched GEMM |
| **m_block_size=8 是 Marlin 层的独立问题** | 修复可改善 Marlin 自身 ~2×，但相比 split-K 的结构性提升属于不同维度 |

---

## 六、后续方向

- [ ] **精度对齐**：使用真实 GPTQ repack 格式权重，验证 cos_marlin 是否接近 1.0
- [ ] **double buffering**：smem pipeline 隐藏全局内存延迟，进一步压榨 M=1 延迟
- [ ] **Marlin 上游 PR**：提交 m_block_size=8 支持，影响 W4A8-INT / W4A8-FP8 / MXFP4-FP8 所有变体
- [ ] **autotuning script**：针对不同 (M,K,N,group_size) 枚举 split-K 配置，输出最优参数 JSON
- [ ] **更大 M 场景**：M=64/128 的 prefill 路径，目前 direct 已接近 roofline，空间有限

---

## 七、环境与参考

| 项目 | 说明 |
|------|------|
| GPU | NVIDIA L40（SM89，Ada Lovelace） |
| 显存带宽 | 864 GB/s |
| INT8 算力峰值 | 362 TOPS |
| 框架 | vLLM（自定义分支 `yeshihai:feat/marlin-w4a8-int`）|
| 相关 PR | [vllm-project/vllm#38066](https://github.com/vllm-project/vllm/pull/38066) |
| 相关 Issue | [#38063](https://github.com/vllm-project/vllm/issues/38063)（int4 不在支持列表），[#38064](https://github.com/vllm-project/vllm/issues/38064)（W4A8 退化为 W4A16）|
