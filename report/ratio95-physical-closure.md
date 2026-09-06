# Ratio < 0.95：物理闭合结果与待定清单

本轮快照：2026-09-07。设计依据是 [optimization-principles 第九节](../doc/compiler/optimization-principles.md#9-可复用诊断动作)，不是旧报告的瓶颈标签。

## 结果与范围

初始低于 0.95 的 **51 条全部先重跑**：SG2044 35 条、K1/X60 16 条。当前其中 **21 条达线，30 条待定**；没有把待定项算作达线，也没有证明它们已到硬件性能下界。

[当前对照表](kernel-performance-comparison.csv) 共 202 条，现有数字中 172 条 >= 0.95、30 条 < 0.95。每个成功的定向运行结果已当时更新为 `targeted-rerun`。194 条行的 measurement 虽然是这个标记，但其中有历史记录，**不表示本轮重跑了 194 条**。

本轮成功运行记录覆盖 183 个不同的表内 kernel/target/phase，包括原始 51 条、128 个回归输入及其它受影响/边界输入；各轮有重叠，共 451 条成功记录，含未保留候选。其余 19 个表内输入未运行，没有发起 202 条全量重跑。最终代码变更的受影响集合另行重跑；C 文本不变的条目沿用本轮已完成的运行，而不是把旧 full-rerun 当新测量。

一个重要负面结果：IQ2_XXS 的 SG standalone/decode 首轮重跑只有 **0.390059 / 0.396736**，不是旧表的 **0.928148 / 0.935150**。当前为 **0.409471 / 0.415074**。这是当前主链上的实际缺口，不能继续使用旧的约 93% 来归因，也不能把这次重测差额都算成本轮改动的回退。

逐条待定项见 [ratio95-pending.csv](ratio95-pending.csv)：每行都有当前数字、具体实体、工作账、责任模块、所需关系、证据位置和手动 repro。

## 保留的实现

1. **作者 ABI 的真实不别名事实。** 10 个 dequantize 入口声明 Y 与输入分离；19 个矩阵入口声明 Xq workspace/Y output 的实际分离关系。不改变量化数值程序。IQ2_XXS 的 66 B、align 2 record 用 `u16[32]` 表示原始存储，拒绝不对齐的 `u32[16]`。
2. **完整 byte accumulation 的零种子选择。** `SelectRISCVOperations::selectZeroSeedProducts` 在同坐标、完整有效域的多项链上把第一项选成 widening multiply，后续仍是原来的 MAC。不是把任何 zero-seed/tail 都强行改写。
3. **不让子域投影反向缩小完整 producer。** `PropagateRISCVLayouts` 保留父值的 lane span；`CanonicalizeRISCVLayouts` 让单用、纯 conversion 链连接的 dense unit load 匹配消费者合法 partition，保留原读取点和 effect。Q8_K 的 block sum 子域不再压缩完整输入 carrier。
4. **有界直接扩展的 indexed-entry 地址宽度。** `PlanRISCVMemory::materializeEntryByteOffsets` 仅对同坐标的直接 unsigned 8/16 -> 32 扩展，在完整 byte offset 可放入 u16 且 layout 合法时选 EEW16。不对任意已计算 u32 运算链追加 narrow。
5. **闭合 SCF carry 的 byte product + add 合为 widening MAC。** `selectLoopCarriedWidenProducts` 要求完整单轴 si16 carry 链、单用关系及对应 yield；沿用已有 typed local op。本轮 TQ1_0 SG standalone/decode 删除 15 次独立 add，没有写另一棵 radix 数值树。
6. **作者坐标入口与输出遍历。** IQ2_XXS staged body 共享，SG 绑定局部 u32 坐标入口，K1 保留原 Index 入口；runner 使用 kernel 旁的普通 entry binding，不再强制覆盖。IQ3_S/IQ3_XXS 作者层使用 16 x 16 输出 blocking，每个输出的 K 方向算术顺序不变。K1 Q6_K 的实测 unroll 从 4 改为 2。
7. **真正重放 rematerialization。** 在 `weft-opt` 注册现有收尾 passes；rematerialized clone 使旧 leaf/implementation 失效，重新选择并闭合资源。没有通过补回 final marker 或保留旧统计绕过 verifier。

参考实现的使用限于局部关系：Triton `lib/Dialect/TritonGPU/Transforms/Coalesce.cpp` 与 TileLang `src/transform/loop_vectorize.cc` 的 per-value layout / access 约束用于核对物理职责；没有引入 GPU thread 根模型或供体模板。ggml donor 的 16 x 16 output traversal 归作者层，不归 compiler 格式分支。

## 初始集合中已达线的 21 条

这里的“重跑基线”是本轮改动前的真实运行，不是旧表值；提升来自累计保留改动，不虚构每个 pass 的独立贡献。IQ2_XS SG decode 在首轮重跑时已经达线，因此不能把 21 条全称作代码优化新增收益。

| Kernel | Target | Phase | 重跑基线 ratio | 当前 ratio |
| --- | --- | --- | ---: | ---: |
| dequantize_row_iq1_m | SG2044 | tensor | 0.675653 | 0.989218 |
| dequantize_row_iq2_s | SG2044 | tensor | 0.806789 | 1.619290 |
| dequantize_row_iq2_xxs | SG2044 | tensor | 0.814311 | 1.592574 |
| dequantize_row_iq3_xxs | SG2044 | tensor | 0.703328 | 1.358189 |
| dequantize_row_q2_k | SG2044 | tensor | 0.660119 | 1.040218 |
| dequantize_row_q3_k | SG2044 | tensor | 0.931457 | 1.653471 |
| dequantize_row_q4_k | SG2044 | tensor | 0.646112 | 1.069643 |
| dequantize_row_q5_k | SG2044 | tensor | 0.931211 | 1.087556 |
| dequantize_row_tq1_0 | SG2044 | tensor | 0.599995 | 1.115979 |
| mul_mat_iq2_xs | SG2044 | decode | 1.014350 | 0.994890 |
| mul_mat_iq3_s | SG2044 | prefill | 0.914057 | 1.630038 |
| mul_mat_iq3_xxs | K1/X60 | decode | 0.934150 | 0.959472 |
| mul_mat_iq3_xxs | K1/X60 | prefill | 0.912809 | 0.952909 |
| mul_mat_iq3_xxs | SG2044 | prefill | 0.948066 | 0.955942 |
| mul_mat_q4_1 | SG2044 | decode | 0.915282 | 0.984849 |
| mul_mat_tq2_0 | SG2044 | decode | 0.921332 | 1.325378 |
| q4_1_q8_1 | SG2044 | decode | 0.873550 | 0.996793 |
| quantize_row_q8_k | K1/X60 | prefill | 0.806818 | 1.089299 |
| quantize_row_q8_k | SG2044 | prefill | 0.891705 | 1.170920 |
| tq1_0_q8_k | K1/X60 | decode | 0.941547 | 0.973616 |
| tq2_0_q8_k | SG2044 | decode | 0.894496 | 1.317986 |

原本已达线但受影响的 IQ2_XXS SG prefill 最终为 **1.013446**；没有留下新增的 < 0.95 回归行。K1 同项仍为 **0.867863**，明确列为待定。

## 待定按实体聚类

下面是 30 条的 13 类物理/合同缺口。详细行数、归属及 repro 以待定 CSV 为准。“需要”表示未实现的具体关系，不是已证明某一条指令解释全部剩余时间；尤其 Q2_K 的归约序列差异尚未通过独立 A/B 隔离时间占比。

| 类别 | 行数 | 已定位的差异 | 还需要什么 |
| --- | ---: | --- | --- |
| 01 sibling supply / partial-set | 9 | IQ1_S 的 qh raw read 已共享，但 index/delta 兄弟链的 issue 表示仍分开；IQ1_M main/correction 每块 8 loads、8 products、32 partial reductions，donor 为 2 个最终 reduction | 共同 source identity、支配点与 projection lifetime；联合 partial-set/resource 规划。物理侧未榨尽，不能默认改数值树 |
| 02 metadata read identity | 2 | IQ2_XXS sign 的 vector gather 与 scale 的 scalar read 重读同一 metadata 区域，64 B/block 对 donor 32 B | 跨 cloned field 的读点/坐标 identity 与合法 u16 raw window/deinterleave；不能换不对齐 u32 load |
| 03 staged entry memory choice | 1 | IQ2_XXS K1 prefill 的 entry ABI 已试，64-bit gather 反而更慢；Index/u32 相对坐标在双机有不同结果 | 在同一完整 entry 合同里联合比较 scalar/unit 与 vector/gather 的合法表示和资源 |
| 04 entry pairing / index lifetime | 2 | IQ3_XXS metadata 已共享；I8X4 的 16 个半 entry 对 donor 8 个整 entry，sign u32 位运算与 data/partial 同时活跃 | 成对 entry 合并、sign 广播坐标共享与联合 live-set；直接 extension 已选 EEW16 |
| 05 radix carrier | 3 | TQ1_0 当前 16 个 digit issue 对 donor 11 个，center/narrow/widen 仍分离；prefill 是不同的 replica partial 路径 | 混合 32/16 carrier、局部 decode/center 表示和资源闭合；不能宣称额外 shift 是数学下界 |
| 06 opaque table range legality | 4 | IQ4_NL/MXFP4 的 opaque i8 输入允许 -128，两项乘积之和可达 32768，不能放入 i16 | 作者常量/值域合同或合法更宽 partial；成本和格式名不能代替数值证明 |
| 07 joined metadata / bsum window | 2 | Q4_K 实际 8 scalar LBU + 2 VL8 indexed metadata reads；correction 4 strided bsum loads | joined 原始字节窗口、scalar/vector decode 共享及 bsum unit-read + register projection |
| 08 subregister reduction sequence | 2 | Q2_K load/product/scale/extract 工作相当；Weft 16 次 16-lane reduce，donor 32 次 8-lane reduce | 同一整数程序内的 subregister reduction/issue 选择及完整延迟/资源关系，不是新作者树 |
| 09 scale convergence choice | 1 | IQ2_XS 为 4 reduce 后 vector mul + 3 串行 vector MAC + 1 extract；donor 是 4 extract 后 GPR combine | `RVVPartialScale/RVVPartialToScalar` 的有限 scalar/vector 收敛选择，保持完整 product carrier |
| 10 replica reduction/extract schedule | 1 | Q4_1 K1 prefill 的 Xq 已复用；8 路 reduction/extract 交错造成 16 次 vtype 切换 | 联合 replica 的 reduction/extract 调度和 tail/resource 合同；不是缺 FMA |
| 11 scalar/address live-set | 1 | Q6_K K1 prefill unroll 4 -> 2 后 frame 784 -> 640 B，地址和标量 partial reload 仍在 | 缩短 scalar/address lifetime、联合 issue 调度，不能把它笼统归为向量 spill |
| 12 half-byte projection | 1 | IQ3_S dequant 的 entry reshape 与半字节 sign window 关系未闭合；直接 global-index 原型更慢 | 4-element entry payload 与 half-byte storage 的坐标投影复用 |
| 13 mask/replica projection | 1 | Q1_0 prefill 的 Xq 已共享，mask 仍先解码成 +/-1；现有 matcher 只支持同形状完整单 lane axis 的 scalar output | mask broadcast/free-axis/replica partial 关系，并在 -128 上给出范围证明或先 widening 后变号 |

## TQ1_0：逐段账与数值边界

此前“8 条 shift/mask 是 radix 展开固有成本”的结论不成立：同样的 256 元素，当前 SG lane16 程序是 tail16 + 5 x 3 x 16，共 **16 次 digit issue**；donor 是 5 x 32 + 5 x 16 + 16，共 **11 次**。issue 分组本身不同，不能只比较最终 reduction 次数。

当前 digit 链是 u8 wrap 乘 radix power、widen 后乘 3、右移 8、narrow 到 i8、center、乘 activation。donor 保留 16-bit digit 和 widened activation，以 8 个 MAC、3 个跨组求和收敛到 1 次最终 reduction。本轮先修闭合 carry：SG standalone/decode 汇编为 **1 vwmul + 15 vwmacc + 1 vwredsum，0 vadd**。ratio 从本轮重跑的 **0.651496 / 0.668161** 到 **0.743641 / 0.757909**，prefill 仍只有 **0.573532**。

每个 si16 lane 最多累加 16 个绝对值 <= 128 的项，上界 **2048**，原来的最终 i32 reduction 不变。本轮没有挪动 scale 或 widening 边界，也没有放宽数值容差。混合 32/16 carrier 还需要连同临时量、partial 和 lifetime 一起规划；只把 reduction 数降到 1 并不能证明收益。K1 回归的 standalone/decode/prefill 分别为 **0.973616 / 1.000671 / 1.299025**。

## 不保留的方向与反例

| 尝试 | 真实结果 | 处理 |
| --- | --- | --- |
| IQ2_XXS raw storage 改 u32 | K1 因 66 B record、align 2 的非对齐读取发生 SIGBUS，退出 135 | 撤回；保留 u16[32] |
| 过宽的 zero-seed 规则 | SG TQ1/Q1 prefill、IQ4_NL 等出现约 10-20% 回退 | 收紧为完整有效域的闭合链；负例恢复并重跑 |
| 直接把 Q8_K load 合到 m8 | SG/K1 ratio 约 0.758/0.713，更多资源代价抵消 load 数下降 | 撤回；改成源域与消费者合法 partition 联合约束 |
| 对计算后的 u32 index 追加 narrow | 29 个输入虽数值通过，但有性能回退；某些 cast 表示直接 unsupported | 撤回；只保留可重建的直接 extension，23 份不再改变的 C 恢复原绑定 |
| IQ3_S dequant 直接 global 4-index | 数值通过，SG ratio 0.352574 | 撤回；reshape/half-byte 关系缺口仍列待定 |
| IQ2_XXS staged 加 I8X8 entry ABI | SG/K1 ratio 0.808430/0.705253；确实选出 64-bit gather | 撤回，不能把“加 entry 合同”称为已解决 |
| I8X8 entry ABI + 局部 u32 coordinate | SG/K1 ratio 0.822410/0.775215 | 撤回，单纯缩小 index 并不补齐 memory/resource 选择 |
| 裸表统一改局部 u32 coordinate | SG/K1 ratio 1.010637/0.793987 | 两机不统一改；最终 SG u32、K1 原 Index 静态入口 |
| K1 IQ2_XXS 改 regular entry + 16 x 16 blocking | ratio 0.844355，低于当前 staged 0.867863 | regular 改动撤回；runner 仍按显式 entry binding |
| K1 Q6_K 缩小 MR/NR | MR1/2 x NR1/2 四个候选最高 1.962173 GOP/s，低于保留的 2.247296 | 保留 MR4/NR2、unroll 2 |
| SG Q1_0 继续加宽 LMUL | 8/16/32/64 四候选中 8 最好，约 9.428 GOP/s，其余约 7.95 | 保留 8；缺口是 mask/replica 关系 |

SG IQ3_XXS standalone 存在明显测量波动：直接 extension 新版本一次 ratio 为 0.792807，tuner 的一次结果约 0.744；同轮旧/新 artifact A/B 为 **2.201623 / 2.226201 GOP/s**，当前表记录后者，对应 **0.832684**。这些负面记录保留，不能把一次较高结果包装成稳定的独立 pass 加速比。

有限参数扫描保留完整候选日志、拒绝原因和数值结果；没有增加能覆盖数值/effect/resource 合法性的成本分数。没有静态成本估计的扫描不伪造估计值，候选域与数量在外部记录可见。

## 实际运行与机械验收

数值判据来自原有 ggml 外部 oracle 与 [实验合同](../doc/experiments/index.md)，保留原容差；没有以 parse/compile 成功代替运行。冷态各 10 次、64 MiB eviction，输入策略为 `valid-quantized-record-replicated`。SG 使用 rvv/CPU48、VLEN128、2.6 GHz；K1 使用 k1/CPU3、VLEN256、1.6 GHz。两边生成实现及 donor 均使用 `-ffp-contract=fast`，Q4_1 汇编确有 FMA。

机械重放按 [第十二节](../doc/compiler/optimization-principles.md#12-机械验收的证明范围)：清 `resources_materialized`，清三项资源统计，重放 canonicalization 的 rematerialization 分支及原有 memory/replica-load/hoist/select/finalize/snapshot/resource 后缀。resource closure 后再次 CSE 时，重新清 marker 和统计再核算，整个过程重复两轮，比较完整原始文本，**二次 diff=0**，没有忽略 SSA 名或资源属性。

| 改动边界 | 数值输入/证据范围 | 清 marker 后机械重放 |
| --- | --- | ---: |
| 内存 alias/u16 与 rematerialization 重新选 leaf | dequant、多个矩阵格式、双机；remat 改动另跑 6 个输入 | 47/47 |
| 完整 byte-chain zero seed | Q4_1、TQ2_0 及其它受影响/负例恢复输入 | 25/25 |
| domain projection / dense read | Q8_K 及多个独立矩阵输入，双机 | 53/53 |
| bounded direct index extension | IQ3_XXS standalone、decode、prefill，双机 | 6/6 |
| closed loop-carried widening MAC | TQ1_0 standalone 与矩阵 decode；prefill/另一机回归 | 2/2 |
| Q6_K selected binding | K1 prefill | 1/1 |
| staged coordinate entries | IQ2_XXS prefill，双机 | 2/2 |
| 16 x 16 output traversal | IQ3_S 与 IQ3_XXS，双机 decode/prefill | 8/8 |
| runner 最终选择原 staged entry | IQ2_XXS K1 prefill | 1/1 |

这些计数属于各次改动边界，有重复输入，不应相加声称独立覆盖量。direct-index 与 carried-MAC 的第二输入分别是同格式的其它真实 canonical 入口，**未据此宣称跨格式泛化**。另将重放闭合的 Q8_K SG IR 翻译成 C，与实际已运行 C 直接比较相同。机械稳定性不替代数值与吞吐，也不证明所有未执行 pass 都幂等。

## 手动复现与证据位置

从仓库根目录运行，使用已有 rvv/k1 SSH 和工具链配置，不设置额外 `WEFT_AUTO_*`、`WEFT_META_*` 或 `WEFT_TUNE_SELECTION`。以下入口会从 DSL emit、编译到目标机并按外部 oracle 运行；不是只生成 IR。

```bash
WEFT_KEEP_ARTIFACTS=1 examples/run/weft-kernel.sh sg2044 q8_K_quantize 10
WEFT_KEEP_ARTIFACTS=1 examples/run/weft-kernel.sh k1 q8_K_quantize 10
WEFT_KEEP_ARTIFACTS=1 examples/run/weft-quantized-vec-dot.sh sg2044 tq1_0 10
WEFT_KEEP_ARTIFACTS=1 examples/run/weft-mul-mat.sh sg2044 tq1_0 decode 10
WEFT_KEEP_ARTIFACTS=1 examples/run/weft-mul-mat.sh sg2044 iq3_s prefill 10
WEFT_KEEP_ARTIFACTS=1 examples/run/weft-mul-mat.sh k1 iq2_xxs prefill 10
```

全部 30 条待定的单条命令在 CSV 的 `repro` 列。源代码定位在 `source` 列；供体逐段对照位于 `source/c/ggml/llama.cpp/ggml/src/ggml-cpu/arch/riscv/quants.c`。

本轮仓库外证据根目录为 `/tmp/weft-ratio95-closure.BAIWpR`：`selection.json` 保存初始表与 51 条选择，`baseline-{sg2044,k1}.jsonl` 保存首次重跑，后续主题 JSONL 保存完整运行配置、数值/计时、日志及本地/远端 artifact 路径。CSV 的相对 `evidence` 路径相对此目录；绝对路径是同轮保留的 artifact，均未复制进仓库。

机械记录分别在 `matrix-halfword-resource-cse`、`zero-seed-closed-mechanical`、`domain-projection-legal-mechanical`、`entry-index-extension-mechanical`、`carried-widen-mac-mechanical`、`q6-binding-mechanical`、`staged-target-indices-mechanical`、`output-blocking-mechanical`、`iq2-prefill-selected-mechanical` 的 `results.json`，包含实际命令、退出码、final/opened/reclosed IR 及原始文本比较结果。这些是本轮临时证据，不是新的验证框架或设计规范。
