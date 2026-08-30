# Q2_K 目标作者树与 K1 partial 组织

日期：2026-08-27

## 结论

Q2_K 的 SG2044/VLEN128 与 K1/VLEN256 实现需要两棵作者数值树。

- SG2044 使用新增的 `group_reduced` 树：每 16 个 `q2×q8` 元素先归约，低 4-bit scale 位于该归约之外。
- K1 保留现有树：per-sub scale 先进入 q2 value，再参与 contraction。
- 选择发生在 runner/caller；同一 canonical kernel 不会被 compiler 按 target 改成另一棵树。

SG2044 的新树真实运行后，vec-dot 从 1.724 提到 3.063 GOP/s，MUL_MAT decode 从 1.732 提到 3.034 GOP/s，prefill 从 1.763 提到 4.845 GOP/s。它仍只有 source 9.462 GOP/s 的 32.4%、32.1% 和 51.2%。

K1 上把同一作者树从 16 lanes 改成 32 lanes 已经能生成和运行，但 MR2 从 1.660 降到 1.471 GOP/s，MR1 进一步降到 0.985 GOP/s。生成汇编没有 vector spill；负结果来自 32-lane mapping 没有同时形成 donor 的 scale gather 与 independent-partial combine。因此剩余缺口是 target physical program，不是再改一棵 Q2_K 作者树。

本轮按约定在这两个结论闭合后停止。runner/runtime 新增了显式的 SG 作者 entry 选择；compiler pass 与 terminal emitter 没有增加 Q2_K、格式名或 target 名特例。

## 1. 两棵作者树

### 1.1 K1 保留的树

`python/weft/std/vec_dot.py::vec_dot_q2_k_q8_k` 与
`python/weft/std/mul_mat.py::mul_mat_q2_k` 的主支路是：

```text
q2
→ widen
→ multiply per-sub scale
→ contract with q8
→ combine two 128-element halves
```

scale 位于 reduction 内。它与 `quants.c:896-930` 的 VLEN256 donor 具有相同的整数结合位置。

### 1.2 SG2044 新增的树

新增入口：

```text
vec_dot_q2_k_q8_k_group_reduced
mul_mat_q2_k_group_reduced
mul_mat_q2_k_group_reduced_decode
```

主支路是：

```text
for each 16-element sub-Level:
    partial = contract(q2[sub], q8[sub], acc=i32)
    integer += partial * low_nibble(scale[sub])
```

它显式改变了 scale 与 reduction 的结合位置，所以是另一棵作者程序。`quants.c:747-827` 的 VLEN128 donor同样先生成每个 scale group 的 partial，再用对应 scale 合并。

sub-Level 的 16 次迭代是 canonical Level multiplicity。一次持有多少个 partial、是否把多个 iteration 展开并交错，是 compiler 的寄存器与 schedule 选择，不在这棵源树里固定。

### 1.3 为什么没有在现有树上做 target rewrite

两棵树的中间 value 不同：

```text
SG: reduce(q2 * q8) * scale
K1: reduce((q2 * scale) * q8)
```

它们改变 scale 的结合位置、widening 边界和可观察的整数 overflow 行为。按照 spec 2.2，这属于作者树；target pass 不能按 VLEN 将其中一棵改写为另一棵。

runner 的具体选择是：

- `sg2044/q2_k` → `*_group_reduced` entry；
- `k1/q2_k` → 原 `production_mul_mat_q2_k` entry。

compiler 仍只看到调用方选定的一份 canonical IR。

## 2. 整数边界

Q2_K/Q8_K 的离散范围为：

```text
q2        ∈ [0, 3]
q8        ∈ [-127, 127]
scale     ∈ [0, 15]
min scale ∈ [0, 15]
```

对应上界：

```text
|q2*q8|                       <= 381
16-element group reduction    <= 6,096
one scaled group              <= 91,440
16 scaled groups              <= 1,463,040
16 bsum/min correction groups <= 487,680
combined integer magnitude    <= 1,950,720
```

因此 i16 product 与 i32 reduction/scale accumulation 均闭合。最终树没有把 operand 预先 source-widen 到 i16：那会把 SG 的 RVV lane 数从 16 降到 8。生成代码由 `contract(u2, i8, acc=i32)` 选择 `vwmulsu` 的 i16 product 和 `vwredsum` 的 i32 reduction。

## 3. SG2044 外部结果

正式运行均为 10 次、Clang 18.1.8、`-O3 -ffp-contract=fast`，数值判据为现有容差检查。

| 项目 | 修改前 GOP/s | 新树 GOP/s | 相对变化 | source GOP/s | 新树/source |
|---|---:|---:|---:|---:|---:|
| vec-dot | 1.724185 | 3.062589 | +77.6% | 9.462 | 32.4% |
| MUL_MAT decode | 1.731622 | 3.034100 | +75.2% | 9.462 | 32.1% |
| MUL_MAT prefill | 1.763080 | 4.845207 | +174.8% | 9.462 | 51.2% |

正确性：

- vec-dot：absolute/relative error 均为 0；
- decode：absolute/relative error 均为 0；
- prefill：max absolute `3.57627869e-07`，max relative `1.65838598e-05`。

生成 C 的主循环为：

```text
16-lane grouped/layered q2 load
→ shift/mask q2 plane
→ 16-lane q8 load
→ vwmulsu i16m2
→ vwredsum i32
→ scalar scale multiply
→ loop-carried i32 update
```

它已经消除了旧树中 `widen(q2)*scale` 导致的 8-lane contraction。仍然与 donor 不同的是：donor 同时保留 8 个 product/partial，再以寄存器指令合并；当前 C 每个 sub-Level 都先结束 reduction 并更新 scalar carry。

一次未归档的临时运行将 `--auto-unroll` 从 1 改为 8，prefill 为 4.837837 GOP/s，相对 4.845207 没有提升；当轮生成 C 中 sub-Level 仍是 `for (... += 16)`。临时 C 已按仓库规则删除，下面给出可复现命令。该观察说明这个 auto 参数目前只进入 operation-local schedule，没有把这个 Level 变成 8-partial register schedule。

## 4. K1 的 lane 与 partial 组织

### 4.1 同一 canonical IR 的物理实例

现有 K1 production 使用 MR2、NR1、LMUL m1 上限：

```text
scaled q / q8 partial lanes = 16
i16 partial LMUL           = m1
i32 accumulator LMUL       = m2
throughput                  = 1.659511 GOP/s
source                      = 2.410 GOP/s
ratio                       = 68.9%
```

把 `--auto-lmul-eighths` 设为 16 后，同一 canonical IR 得到 32-lane value chain：

```text
q / q8 lanes               = 32
scaled q LMUL              = m2
i32 accumulator LMUL       = m4
```

真实结果：

| 物理实例 | GOP/s | source 比例 | 相对当前 production |
|---|---:|---:|---:|
| lane16, MR2 | 1.659511 | 68.9% | 1.000× |
| lane32, MR2 | 1.471145 | 61.0% | 0.887× |
| lane32, MR1 | 0.985432 | 40.9% | 0.594× |

因此 lane32 是 compiler 已经能实例化的参数性表示，不是缺少一个 Q2_K 特例；但只扩大 lane 会退化。

### 4.2 临时汇编对照

下表来自两次 `WEFT_KEEP_ARTIFACTS=1` 的临时 K1 运行。统计完成后，本地与远端临时目录均已删除；仓库不保存 generated C/assembly，下面给出重现同一观察的命令。

| 指令/现象 | lane16 MR2 | lane32 MR2 |
|---|---:|---:|
| assembly lines | 759 | 717 |
| `vwmacc` | 16 | 12 |
| `vwmul` | 2 | 6 |
| `vredsum` | 6 | 6 |
| `vrgather` | 0 | 0 |
| `vslide` | 0 | 8 |
| `vle8` | 19 | 19 |
| `vs1r/vl1r` spill | 0 | 0 |
| `csrr vlenb` | 0 | 0 |

该次 lane32 汇编没有寄存器溢出或动态栈 spill，主要新增指令是 8 个 `vslideup`。

生成 C 对每个 128-element half 分别构造 4 个 32-lane scale value；每个 value 由两个 scalar broadcast 加一次 `vslideup` 拼成。donor `quants.c:875-909` 则一次载入 16 个 scale，再以 4 个 `vrgather` 构造 4 个 32-lane scale vector。

当前实现这个选择仍埋在 `RISCVIntrinsicC.cpp:5162-5224` 的 `regularRepeat > 1` emission 分支中：physical IR 只有 regular-repeat access facts，没有已经选好的 `load + gather` operation。因而这是 physical memory/layout 缺口，不是 leaf 拼写问题，也不是作者应再声明一个 engine 或 pack。

### 4.3 partial topology

lane32 的当前 C 对每个 half/output形成：

```text
vwmul(first plane)
→ 3 × vwmacc(remaining planes)
→ one final reduction
```

K1 donor 形成：

```text
4 independent i32m4 products
→ pairwise vector adds
→ 2 staged final reductions across the two 128-element halves
```

两者实现同一 canonical contraction，没有改变 logical value 或 Level 归属，所以 sequential accumulator chain 与 independent partial tree 的选择属于 compiler 的结构性物理选择。

当前 `rvv_widen_accumulate + rvv_finalize_widen_dot` 只承载一个 loop-carried partial；没有 typed partial set、partial-combine topology 或对应 resource contract。把 donor 结构写进 emitter 会使 emitter 重新成为结构选择者，因此本轮没有这样处理。

### 4.4 为什么本轮没有新增 Q2-only physical op

静态覆盖给出的外部事实是：

- TQ2_0 已经形成完整 storage-window/partial program；
- Q5_K 有 shaped q/qh planes，但联合 storage geometry 目前不闭合；
- IQ4_NL 已有 layered storage stream，但 codebook lookup 结果不是 partial window root；
- Q3_K/Q6_K 仍是展开的 scalar 作者树，不能作为 physical pass 的合法输入。

所以一个只让当前 Q2_K 变成 donor 指令序列的新 op，尚没有第二个现有 input 能证明它是 compiler mechanism。按本轮“Q2_K 到此停”的要求，这个缺口被记录为：

```text
typed regular-repeat load/gather
typed independent-partial set
typed partial combine/final reduction topology
```

没有为单条数字实现后端闭包。

## 5. 与 Triton/TileLang 的具体对照

本轮参考的是机制，而不是复制 GPU 物理公式。

Triton：

- `ref/triton/lib/Dialect/TritonGPU/Transforms/AccelerateMatmul.cpp:441-486`：结果 MMA encoding 是 anchor，A/B operand encoding 与 `convert_layout` 从该 anchor 构造；作者 `dot` 不携带具体 lane/fragment 公式。
- `ref/triton/lib/Conversion/TritonGPUToLLVM/ReduceOpToLLVM.cpp:237-351`：register bases、lane bases与被消去 axis 决定 thread-local/lane reduction；多个 partial按 logical slot组织后 tree-reduce。
- `ref/triton/lib/Dialect/TritonGPU/Transforms/OptimizeDotOperands.cpp:181-299`：从 sink operand layout向上回溯 view，并把 local allocation/load 提升到可共享位置。

TileLang：

- `ref/tilelang/src/transform/reducer_plan_materialize.cc:559-733`：partial storage从 update site 的 reduction axes 投影出来；多个 update site 必须得到 structurally equal plan，否则不能使用 narrow plan。
- `ref/tilelang/src/op/reduce.cc:194-230`：source/destination layout 不满足 containment 时明确报告 layout conflict，而不是在 terminal emission 中补默认行为。

这些机制共同说明：K1 的 scale repeat 与 partial combine应成为 typed physical program及其 pass 结果；它们不能由 intrinsic C emitter根据 Q2_K 的源码 closure临时重建。

## 6. Repro

```bash
examples/run/weft-quantized-vec-dot.sh sg2044 q2_k 10
examples/run/weft-mul-mat.sh sg2044 q2_k decode 10
examples/run/weft-mul-mat.sh sg2044 q2_k prefill 10

WEFT_AUTO_UNROLL=8 \
  examples/run/weft-mul-mat.sh sg2044 q2_k prefill 10

WEFT_AUTO_LMUL_EIGHTHS=16 \
  examples/run/weft-mul-mat.sh k1 q2_k prefill 10

WEFT_AUTO_LMUL_EIGHTHS=16 \
WEFT_META_BINDINGS='NC=32;MC=16;MR=1;NR=1' \
  examples/run/weft-mul-mat.sh k1 q2_k prefill 10

WEFT_KEEP_ARTIFACTS=1 \
  examples/run/weft-mul-mat.sh k1 q2_k prefill 1

WEFT_KEEP_ARTIFACTS=1 WEFT_AUTO_LMUL_EIGHTHS=16 \
  examples/run/weft-mul-mat.sh k1 q2_k prefill 1

examples/run/weft-mul-mat.sh sg2044 q4_0 prefill 1
examples/run/weft-mul-mat.sh sg2044 q4_1 prefill 1
examples/run/weft-mul-mat.sh sg2044 q5_1 prefill 1
examples/run/weft-mul-mat.sh sg2044 iq4_nl prefill 1
examples/run/weft-mul-mat.sh sg2044 q1_0 prefill 1
examples/run/weft-mul-mat.sh sg2044 q4_k_persistent prefill 1
```

SG 回归一轮运行了 Q4_0、Q4_1、Q5_1、IQ4_NL、Q1_0 与 Q4_K persistent；六条均报告 `numeric=within-tolerance`。这些是单次回归运行，不写入 10-repetition 性能 CSV。所有 `WEFT_KEEP_ARTIFACTS` 临时目录在分析后已删除。
