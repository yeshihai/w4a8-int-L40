# W4A8-INT GEMM 优化实验报告（L40 / SM89）

> **硬件**：NVIDIA L40（SM89, Ada Lovelace，864 GB/s，362 TOPS INT8）  
> **目标**：优化 W4A8（INT4 weight × INT8 activation）在小 N decode 场景的性能  
> **聚焦 shape**：`(K=3584, N=512)`，Qwen2.5-7B 典型小投影层（如 o_proj / down_proj）

---

## 一、现状：vLLM 各精度在关键 shape 下的延迟

> 测量环境：L40，group_size=128，PR #38066 修复后。数据越小越好。

### (K=3584, N=3584) — 标准 decode 形状

| 精度 | M=1 | M=4 | M=8 | M=16 |
|------|-----|-----|-----|------|
| W4A16 | 24.1μs | 24.5μs | 24.7μs | 26.2μs |
| W8A8  | 24.6μs | 24.4μs | 24.5μs | 24.7μs |
| W4A8  | 23.6μs | 23.4μs | 23.5μs | 23.7μs |

> 三种精度延迟几乎一致，均约 24μs，内存带宽利用率约 30~60%。

### (K=3584, N=512) — 小投影层（本项目聚焦）

| 精度 | M=1 | M=4 | M=8 | M=16 |
|------|-----|-----|-----|------|
| W4A16 | 29.0μs | 28.7μs | 28.4μs | 30.3μs |
| W8A8  | 24.0μs | 24.1μs | 23.9μs | 24.0μs |
| W4A8  | 28.4μs | 28.4μs | 28.2μs | 28.5μs |
| **W4A8 + 自己写的 kernel** | **24.0μs** | **24.2μs** | 30.7μs | 43.5μs |

> W8A8 在这个 shape 最快（~24μs），W4A16 和 W4A8 Marlin 均约 28到29μs。  
> 本 Tiny kernel 在 M=1/4 时追平 W8A8，在 M≥8 时不如 Marlin。

### (K=3584, N=18944) / (K=18944, N=3584) — 大矩阵

| 精度 | M=1 |
|------|-----|
| W4A16 | ~33μs |
| W8A8  | ~43μs |
| W4A8  | ~30μs |

> W4A8 在大 N 下反而最快（INT4 权重更省带宽）。

---

## 二、为什么 (K=3584, N=512) 是难点

```
Marlin tile_N = 128，tile_M = 16

Grid（M=1, N=512）:
  N 方向: 512 / 128 = 4 个 tile
  M 方向: ceil(1/16) = 1 个 tile
  → 总共 4 个 block，L40 有 144 个 SM
  → SM 利用率仅 2.8%，140 个 SM 完全空置
```

这是 **wave quantization** 问题：N 太小，tile 数量太少，无论 Marlin 内部算法多高效，都喂不饱 GPU。W8A8 用的 CUTLASS 在这个 shape 上也约 24μs，已经接近该 shape 的实际下限。

W4A8 Marlin 比 W8A8 慢约 4μs 的额外原因：Marlin 的 split-N kernel 在 N=512 时有额外的调度开销（`VLLM_MARLIN_USE_ATOMIC_ADD` 相关路径），而 CUTLASS 在小 N 时走了更简单的路径。

---

## 三、Tiny Kernel：用 split-K 补充 SM 并行度

标准路径的 4 个 tile 无法填满 SM，Tiny kernel 的思路是**从 K 维切分**：

```
split-K: grid = (4, M, K_splits)

K_splits = min(K/group_size, target_ctas / base_ctas)
         = min(3584/128, 128/4) = min(28, 32) = 28

→ 总 block 数 = 4 × 1 × 28 = 112（vs 原来的 4）
→ SM 利用率从 2.8% → 77%
```

每个 block 只算 K 方向的 1/28 段，最后用 atomic add 归约。

### 迭代历程（聚焦 (M=1, K=3584, N=512)）

| 版本 | 方案 | M=1 延迟 | 说明 |
|------|------|---------|------|
| v1 | 朴素标量 | ~500μs+ | 正确性验证 |
| v2 | smem 行缓存 | ~162μs | 建立基线 |
| v3 | 激进展开 | ~420μs | 寄存器压力过高，退化 |
| v4 | 256 threads | ~158μs | 标量路径下限 |
| v5 | 双行 M 复用 | ~182μs | 控制流开销 > 复用收益 |
| v6 | dp4a 向量化 | ~178μs | 算术优化被系统开销淹没 |
| **v7** | **split-K + atomic** | **~24μs** | **结构性突破，追平 W8A8** |

**核心认知**：decode 小 M + 小 N 场景，瓶颈是 SM 调度，不是算术指令效率。单条指令层面的优化（dp4a、展开）无法破局，必须从 grid 维度解决并行度不足的问题。

### v7 最终结果

| M | Tiny(μs) | Marlin W4A8(μs) | W8A8(μs) | 结论 |
|---|---------|----------------|---------|------|
| 1 | **24.0** | 28.4 | 24.0 | 追平 W8A8 ✅ |
| 4 | **24.2** | 28.4 | 24.1 | 追平 W8A8 ✅ |
| 8 | 30.7 | 28.2 | 23.9 | split-K atomic 开销显现 |
| 16 | 43.5 | 28.5 | 24.0 | M 大后 direct 更优 |

CosSim ≈ 0.999964~0.999969，精度与 Marlin 完全对齐。

---

## 四、上游 Bug 修复

在做性能分析时发现了 vLLM 中 W4A8-INT 的两个 Bug，已提 PR：

**Bug 1（`marlin_utils.py`）**：`query_marlin_supported_quant_types` 不含 `scalar_types.int4`，导致 W4A8-INT `can_implement` 直接返回 False，静默退出 Marlin 路径。

**Bug 2（`compressed_tensors_w4a8_int.py`）**：`create_weights` 里 `act_type=params_dtype`（bf16），activation 从不被量化为 INT8，整条路径走 BF16 mma 而非 INT8 mma。修复前 (3584,3584) M=1 的 W4A8 延迟高达 ~64μs，修复后恢复正常（23.6μs）。

**PR**：[vllm-project/vllm#38066](https://github.com/vllm-project/vllm/pull/38066)（Fixes [#38063](https://github.com/vllm-project/vllm/issues/38063), [#38064](https://github.com/vllm-project/vllm/issues/38064)）

---

## 五、后续方向

- [ ] **Marlin m_block_size=8 PR**：`generate_kernels.py` 缺少 `THREAD_M_BLOCKS=0.5`，影响 W4A8 所有变体在 M≤8 时的 mma 利用率
- [ ] **Tiny kernel double buffering**：smem pipeline 隐藏全局内存延迟，改善 M=8~16 场景
- [ ] **autotuning**：针对不同 (M,K,N) 扫描最优 K_splits 配置

---

> GPU: NVIDIA L40（SM89）| 864 GB/s | 362 TOPS INT8
