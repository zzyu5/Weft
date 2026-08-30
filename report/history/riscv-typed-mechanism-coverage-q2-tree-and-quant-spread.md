# Typed 机制覆盖、Q2_K 作者树与量化横向迁移报告

## 1. 本轮范围

本轮没有执行 206 条全量回归，只处理四项工作：

1. 静态审计 24 个量化格式对 layered stream、bitmask decode、storage window、partial accumulation，以及 reduction-lane/free-axis-lane anchor 的实际覆盖；
2. 按作者决定重写 Q2_K 的数值树，把 per-sub scale 放进 reduction 内部，并检查整数范围；
3. 将一部分逐元素 std 函数改成 shaped value，并将四个量化 MUL_MAT 改成 blocked prefill 与独立 GEMV decode；
4. 在 SG2044（VLEN128）和 K1/X60（VLEN256）上只重测受影响项目，并对 K1 的剩余差距尝试硬件计数器和参数隔离。

当前性能表已经用本轮 10 次重复的实测结果覆盖：
[`weft-kernel-performance.csv`](weft-kernel-performance.csv)。GGML 固定基线仍来自
[`baseline/ggml-riscv-kernel-performance.csv`](baseline/ggml-riscv-kernel-performance.csv)。

## 2. 参考实现给出的机制边界

### 2.1 Triton

Triton 的 `RemoveLayoutConversions` 不是从源代码闭包猜布局。它先找 layout anchor，再沿后代传播；冲突时插入真实 `convert_layout`，最后按 dominance order 重写程序。对应实现位于：

- `ref/triton/lib/Dialect/TritonGPU/Transforms/RemoveLayoutConversions.cpp:42-59`：anchor、传播、冲突解决、IR rewrite 四步；
- 同文件 `:218-257`：load/store、dot、gather、reshape 等 anchor 的定义；
- 同文件 `:113-179`：conversion 的 backward-slice rematerialization；
- `OptimizeDotOperands.cpp:20-81`：dot operand 的 physical layout、local allocation 与 local load 是真实 IR 重写，不是 emitter 内的字符串分支。

因此，本轮没有把 Q2_K 的 `scale * reduce(product)` 自动改成 `reduce(scale * product)`。两式改变 i16 partial、widening 位置和溢出边界；Triton 的 layout propagation 也不会执行这类数值分配律改写。

### 2.2 TileLang

TileLang 的 reducer materialization 先从 loop layout 推导 reduction thread steps，再把串行累积、线程内 partial 与 collective materialize 成显式计划。`reducer_plan_materialize.cc:627-727` 显式区分：

- reduction 变量进入线程表达式的部分形成 collective；
- 未进入线程表达式的 reduction 变量在一个线程内串行累积；
- packed partial 只有在 storage layout、thread mapping 和资源条件成立时才生成。

这与本轮结论一致：partial accumulation 必须由已经存在的逻辑 reduction axis 和 typed storage geometry 驱动，不能从若干重复 SSA operation 重新发明一条轴。

## 3. 五套 typed 机制的静态覆盖

静态检查使用统一的 SG2044 profile 和同一组 meta 生成 RISC-V IR，只统计真实出现的 physical op/leaf；没有运行性能。

| 格式 | layered stream | bitmask decode | storage window | partial accumulation | product anchor | 未触发的主要原因 |
| --- | --- | --- | --- | --- | --- | --- |
| Q1_0 | — | 是 | — | — | reduction-lane | u1 bitmask geometry 完整，但不是 layered storage |
| Q4_0 / Q4_1 | 是 | — | — | — | reduction-lane | 完整 grouped/layered Field 可直接成为 stream |
| Q5_0 / Q5_1 | 是 | 独立 high-plane merge | — | — | reduction-lane | high plane 与 low payload 已是 typed geometry，但没有多 window partial |
| Q8_0 | — | — | — | — | reduction-lane | 普通 dense signed input，不需要 packed stream |
| Q2_K | — | — | — | — | 3 个 widening product | sub-Level point 投影后的 encoded Field 不能成为 layered stream |
| Q3_K | — | — | — | — | — | 当前树没有保留可供 physicalization 使用的 shaped bit-plane 关系 |
| Q4_K | — | — | — | — | widening product | canonical tree 有 shaped reduction，但 payload 仍经 projected extracts |
| Q5_K | — | — | — | — | widening product | joined/high-plane 关系没有形成 layered window |
| Q6_K | — | — | — | — | — | 当前树仍缺 shaped high-bit/reduction 关系 |
| IQ1_S / IQ1_M | — | — | — | — | — | codebook/sign 仍由逐项数值树组织 |
| IQ2_S / IQ2_XS / IQ2_XXS | — | — | — | — | — | lookup/grid/sign 没有形成共享 typed window |
| IQ3_S / IQ3_XXS | — | — | — | — | — | grid/sign 与高位分解仍缺 shaped 关系 |
| IQ4_NL | 是 | — | — | — | reduction-lane | 16-entry lookup 后的 packed value 已落到 layered stream |
| IQ4_XS | — | — | — | — | reduction-lane | 本轮形成 shaped lookup，但没有 storage window |
| TQ1_0 | — | — | — | — | — | radix projection 仍在逐项树中 |
| TQ2_0 | — | — | 4 | 4 + finalize | reduction-lane | 唯一完整形成 typed window 与 partial program 的当前格式 |
| MXFP4 | 是 | — | — | — | reduction-lane | shaped lookup 可进入 layered operand |
| NVFP4 | — | — | — | — | reduction-lane | shaped lookup 成立，但 nested sub-Level 没有 window realization |

### 3.1 哪些条件确实过严，哪些不是

`ShareRISCVLayeredWindows` 已经不再要求 `group = 2 × layer`；当前核心条件是一个 unsigned `grouped_layered` Field、合法 group/layer 几何、唯一 lane axis，以及 `time × lane = extent`。Q4/Q5/IQ4_NL/MXFP4 的跨格式命中说明这条规则不是格式分支。

`FuseRISCVBitplanes` 对 u1、`group = layer × 8`、byte alignment 和 bit order 的要求来自 storage geometry，本轮没有放宽。放宽这些条件会改变实际位解释，不是提升覆盖。

`storage window` 和 `partial accumulation` 目前只在 TQ2_0 出现。检查后，Q2_K 并非只差“至少两个 storage window”这一条门槛：它在更早的位置就把 `Field + sub-Level point` 降成普通 extract，物理 IR 中不存在可供 partial pass 消费的 domain-projected layered stream。直接删除 partial 的窗口条件不会产生正确程序。

因此本轮只新增了可证明的 regular-index physicalization，没有把 Q2_K 塞进现有 TQ2 closure。真正缺失的是一种能同时携带 encoded Field、typed sub-Level point 和 layered geometry 的物理 window/stream；这是共享 physical IR 能力缺口，不是 Q2_K leaf 缺口。

## 4. Q2_K 作者树

### 4.1 数值结构

Q2_K 的 std 树现在把 256 元素分成两个 128-element reduction：

```text
q2 plane -> widen i16
per-sub scale -> widen i16
scaled_q = q2 * scale
q8 -> widen i16
partial = contract(q8, scaled_q, acc=i32)
two partials + min correction -> final i32
```

这项改动只发生在 `python/weft/std/vec_dot.py` 和 `python/weft/std/mul_mat.py`，因为它改变 logical intermediate values。编译器没有执行分配律重写。

### 4.2 溢出范围

- `q2 ∈ [0, 3]`，per-sub scale 的有效低 4 bit 为 `[0, 15]`；
- `q2 × scale ∈ [0, 45]`，可精确放入 i16；
- Q8_K activation 的量化整数范围乘上 45 后，单项绝对值不超过 5760；
- 256 项最坏累积与 min correction 都远小于 i32 上限。

因此 i16 scaled plane 与 i32 reduction 的类型边界是闭合的。本轮两台机器的数值结果也均通过容差；vec-dot 的最大绝对和相对误差均为 0。

### 4.3 regular-index physicalization

Q2_K 的 scale index 和 packed plane index来自 `iota/add/sub/mul/div`。memory pass 现在把这类 use-def 链推导成实体级：

```text
index_pattern = [base, stride, repeat]
```

`regular` 是 `weft_riscv.extract` 的 typed selector，verifier 检查 source/result axes、边界、正 stride/repeat 和整数溢出。VLEN128 的 `repeat=16, lanes=8` 生成同一 scalar source 的分段 broadcast；VLEN256 的 `repeat=16, lanes=32` 生成两个 broadcast 后用 `vslideup` 合成。两台 target 读取同一个 IR relation，没有 target-name 分支。

regular extract 与 lowering 后新增的 layout conversion 还需要再次 canonicalize，所以 pass pipeline 在 composite lowering 后增加了一次 layout canonicalization。结果中 `vdiv/vrem/vluxei` 消失，`vslidedown` 静态数量由 132 降到 4。

### 4.4 性能与仍缺的实体

| target / case | 本轮前 Weft | 本轮正式结果 | GGML source | source 比值 |
| --- | ---: | ---: | ---: | ---: |
| SG vec-dot | 2.767 | 1.724 | 9.308 | 18.5% |
| K1 vec-dot | 1.158 | 1.367 | 2.446 | 55.9% |
| SG MUL_MAT prefill | 3.338 | 2.136 | 9.462 | 22.6% |
| K1 MUL_MAT prefill | 1.239 | 1.823 | 2.410 | 75.6% |

SG 的 one-repetition MR/NR 隔离结果是 MR1×NR1 最快；K1 是 MR2×NR1 最快，runner 已分别绑定这两个作者 auto 实例。扩大 cohort 在当前物理程序上反而变慢，因为每个 output replica 都重复 materialize scale/payload。

当前生成程序仍有 248 个 widened MAC、24 次最终 reduction，并反复 materialize projected payload。GGML donor 先形成四个 plane/window，再对每个 window做少量 widened MAC，最后只 reduction 一次。性能差距已经不在 regular address calculation，而在 domain-projected layered window、跨 output window sharing 和 partial accumulation 没有形成。

另外，GGML 的 VLEN128 和 VLEN256 Q2_K donor 自身采用不同数值树：VLEN128 先 reduce `q2 × q8` 再乘 scale；VLEN256 把 scale 放进 reduction。本轮按用户指定采用后一棵作者树。编译器不能为了 SG 性能把它改回前一棵；这是 spec 2.2 在跨 VLEN donor 上暴露出的真实作者特化边界。

## 5. shaped row-dequant 与 vec-dot

本轮将 IQ4_XS、NVFP4 的 Python 逐元素展开改成显式 shaped value；两者都使用原有 Encoding 和数值函数，没有增加语言构造或后端格式分支。

| case | SG 旧值 | SG 新值 | K1 旧值 | K1 新值 |
| --- | ---: | ---: | ---: | ---: |
| IQ4_XS row-dequant (MElements/s) | 157.305 | 441.942 | 46.358 | 177.867 |
| NVFP4 row-dequant (MElements/s) | 318.036 | 696.188 | 111.880 | 246.296 |
| IQ4_XS vec-dot (GOP/s) | 0.655 | 0.972 | 0.216 | 0.507 |
| NVFP4 vec-dot (GOP/s) | 0.585 | 1.451 | 0.477 | 0.609 |

row-dequant 已分别达到 source 的 276%/113%（IQ4_XS，SG/K1）和 361%/244%（NVFP4）。vec-dot 的 shaped axis 已被 reduction anchor 使用，但 IQ4_XS 仍只有 source 的 36.0%/29.6%；这与静态矩阵一致：codebook/storage window 没有跨 consumer 共享。

## 6. blocked MUL_MAT 迁移

本轮完成四个原 row×column production tree 的迁移：canonical Q4_K、IQ4_XS、MXFP4、NVFP4。prefill 使用 NC/MC output tile、MR/NR accumulator、K block 和 activation reuse；decode 使用独立 GEMV 特化，避免 M=1 承担 blocked packing 成本。

| case | SG 旧值 | SG 新值 | SG/source | K1 旧值 | K1 新值 | K1/source |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| IQ4_XS prefill | 0.650 | 3.472 | 58.9% | 0.192 | 1.277 | 74.3% |
| MXFP4 prefill | 2.642 | 7.904 | 119.7% | 0.922 | 2.513 | 95.6% |
| NVFP4 prefill | 0.586 | 1.974 | 521% | 0.472 | 0.740 | 471% |
| Q4_K canonical prefill | 5.119 | 1.489 | 15.4% | 1.957 | 1.383 | baseline 为 IME，不作同引擎比值 |

MXFP4 与 NVFP4 表明相同 blocked skeleton 可以直接消费新 shaped 数值树。IQ4_XS 的提升也跨两种 VLEN 成立，但仍低于 RVV source，因为 lookup window 没有成为 output cohort 共享实体。

canonical Q4_K 是明确负结果：改成已有 staged blocked tree 后，两台机器都变慢。该树能编译且数值在容差内，但它没有命中 persistent-derived Q4_K 的 grouped window/partial program；局部 staging 反而增加了 materialization。未恢复旧 row×column 路径来掩盖这一结果。

IQ2_XXS staged 也做过迁移试验：K1 数值正确，SG2044 出现数量级错误，因此 production 入口保留原树，只新增数值正确的 GEMV decode。这个 staged 函数是仓库原有独立入口，不是 fallback；本轮没有把它冒充为跨 target 的 production 迁移。

### 6.1 尚未迁移的树

- 14 个 row×column MUL_MAT 中完成 4 个；剩余 10 个：Q3_K、Q6_K、IQ1_S/M、IQ2_S/XS/XXS、IQ3_S/XXS、TQ1_0；
- 12 个逐元素 row-dequant 中完成 IQ4_XS、NVFP4，剩余 10 个；
- 9 个逐元素 vec-dot 中完成 IQ4_XS、NVFP4，剩余 7 个。

剩余项不是同一种机械工作。Q3/Q6/TQ1 的 storage projection 尚未形成完整 shaped bit/radix axis；IQ1/IQ2/IQ3 需要把 grid/sign/codebook 组织成 typed shaped values。现有语言没有在本轮被证明“写不出来”，但直接复制 IQ4_XS 模板会丢失这些格式各自的数值结构。

## 7. K1 定位结果

K1 当前内核禁止普通用户访问 PMU，系统中也没有可用 `perf` 工具或可读 counter 接口，因此没有伪造硬件计数器归因。实际尝试的隔离变体为：

| case | default | unroll=2 | alternate MR/NR | 结论 |
| --- | ---: | ---: | ---: | --- |
| Q1_0 prefill | 3.466 | 3.469 | MR4×NR2 = 2.778 | 8% 差距不来自 unroll 或 cohort 扩大 |
| IQ4_NL prefill | 2.464 | 2.465 | — | unroll 对该差距无可测收益 |

这两项没有留下无收益配置。Q2_K 的 target-dependent MR/NR 隔离有明确结果，已进入 runner 的 meta binding；它是参数性 auto 实例，不是 pass 内 target branch。

## 8. SG 回归抽查

对上一轮已经超过 source 的六条做了 one-repetition 抽查：Q4_0 8.623、Q4_1 6.641、Q5_1 6.454、IQ4_NL 8.613、Q1_0 6.899、Q4_K persistent 10.115 GOP/s。没有观察到本轮 regular extract、scalar tuple lookup 和 late layout canonicalization 导致的回退。这些是回归抽查，不写入 10 次重复的当前 CSV。

## 9. 结论

本轮证明了三件不同的事实：

1. regular use-def index 可以作为 typed physical relation 在 VLEN128/VLEN256 上机械发射，且 shaped IQ4_XS/NVFP4 的收益跨 target 成立；
2. blocked std skeleton 对 MXFP4、NVFP4、IQ4_XS 有横向收益，但 canonical Q4_K 和 IQ2 staged 提供了反例，不能据此宣称所有格式已经机械迁移；
3. Q2_K 作者树已经按指定数值分解写出并数值闭合，但性能只达到 SG source 的 22.6%。缺口精确落在 projected encoded field 没有形成 layered window、window 没有跨 output 共享、partial program 没有形成，而不是地址 LICM、LMUL 扫描或 emitter 拼写。

没有出现“编译器必须改变这棵 Q2_K 作者树才能让当前树合法”的证据；相反，VLEN128/256 donor 的不同数值树说明需要由 std 提供作者特化。当前阻塞是同一作者树的物理表示能力不完整。
