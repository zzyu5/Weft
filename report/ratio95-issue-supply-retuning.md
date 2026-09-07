# IQ2_XXS issue 调优与共享原型复核

本轮继续上一轮剩余项，按 [optimization-principles 第九节七步](../doc/compiler/optimization-principles.md#9-可复用诊断动作)核对工作账。本文是一次性结果，不替代设计规范。

## 保留结果

只保留 SG IQ2_XXS standalone/decode 的 physical unroll 绑定 `2 -> 1`，配置仍在对应 kernel 家族旁。作者程序、编译器、数值边界、K1 绑定和 runtime 容差均未改变。两个编译器共享原型实际实现并运行后因回退全部撤回，不作为完成的新能力提交。

| SG 入口 | 本轮首跑 GOP/s | 最终默认入口 GOP/s | 首跑 ratio | 最终 ratio | 吞吐变化 |
| --- | ---: | ---: | ---: | ---: | ---: |
| iq2_xxs_q8_k | 1.650001 | 1.994018 | 0.408585 | 0.493772 | +20.85% |
| mul_mat_iq2_xxs decode | 1.649730 | 1.995992 | 0.414459 | 0.501450 | +20.99% |

最终值来自普通 runner 复跑，不采用调优过程中的最高值。K1 同两条最终为 **1.098346 / 1.132237**；IQ2_XXS prefill 为 SG **1.012068**、K1 **0.867742**，dequantize 为 SG **1.631906**、K1 **2.843710**。低于线的 K1 prefill 没有因为 standalone 通过而算作解决。

当前 [对照表](kernel-performance-comparison.csv) 为 **202 行，172 行 >= 0.95，30 行待定**。本轮没有新增达线条目；待定从 29 增到 30，是相关回归发现 SG IQ1_M prefill 的旧 **0.954927** 已过时，撤回原型后的默认实测仍为 **0.868933**。详细实体、owner、所需关系及 repro 见 [本轮待定清单](ratio95-issue-supply-pending.csv)。没有把这次测量刷新称为本轮代码造成的回退。

## 修正读取工作账

上一轮把生成 C 中的重复 u16 表达式直接折成实际读取字节数，高估了共享收益。对相同 flags 的实际汇编重新核对：

- SG 每个 64-element issue 为 `group=2, entry=4, payload=8`，i8m4 product supply，两个 i16m4 partial 分别归约 32 lanes；每 256 元素共 4 个 issue、8 次 widening reduction。最终 Physical IR 的 vector resource peak=12，local storage=0；这是 IR 的资源统计，不代替最终机器指令检查。
- `word1` 的 sign 路径每块读取 32 B。scale 在 C 中像是再次拼两个 u16，但 LLVM 已缩为每 group 一个 `lbu`，每块实际另读 **8 B**，因此 sign/scale 是 **40 B，不是旧报告的 64 B**。
- grid 的 `w.q[4*group + (entry >> 1)]` 每个半字被两个 entry 使用，当前每块实际读取 64 B；donor 是直接读取 32 B grid bytes。加上 metadata，当前 q 字段读取为 104 B，donor 为 64 B，不含 d 和外部 lookup tables。
- donor `quants.c:4555-4566` 直接用两个 4-byte unit loads 供应 grid，并用两个 `memcpy` 读取 packed signs；scale 使用同一 packed value。其 lookup indices 是 EEW16，当前路径仍是 EEW32。这些是具体 physical supply/representation 差异，不能据此默认改作者数值树。
- record 仍是 66 B、align 2 的自然 u16 字段。没有恢复不安全的 u32 字段读取，也没有通过放宽 alignment 绕过合法性。

实际汇编证据：仓库外 `sg-iq2-unroll1.s:88-124`、`baseline-K1_X60-iq2_xxs_q8_k.s:120-123` 和 `:173-176`；donor 为 [quants.c](../source/c/ggml/llama.cpp/ggml/src/ggml-cpu/arch/riscv/quants.c#L4523)。

## 撤回的编译器尝试

第一版让同一个 issue 的 lhs/rhs 和 deferred scale 共用 `cloneIssueWindow` 的映射，并在 nested plan 中冻结有界共享前沿及类型/资源。第二版进一步让 conversion CSE 进入 `CanonicalizeRISCVLayouts` 的 rematerialization 固定点，避免 CSE 暴露单一消费者后不再检查其表示。

两版不改变 contraction/reduction 的数值树。实际结果如下：

| IQ2_XXS 入口 | 本轮首跑 ratio | 共用 clone map | 再加 rematerialization 固定点 | 撤回后的默认复跑 |
| --- | ---: | ---: | ---: | ---: |
| SG standalone | 0.408585 | 0.404041 | 0.407377 | 0.408959 |
| SG decode | 0.414459 | 0.397343 | 0.412235 | 0.415422 |
| K1 standalone | 1.099026 | 1.036264 | 0.933293 | 1.097879 |
| K1 decode | 1.131900 | 1.067614 | 0.965424 | 1.132025 |

共用映射的真实第二输入是 IQ1_S；IQ3_XXS 在这份 C 对照中没有变化，不能算命中正例。组合原型下 SG IQ1_S standalone/decode 为 **0.763484 / 0.769206**，撤回后恢复到 **0.805554 / 0.811497**；SG IQ2_S decode 为 **1.365554 -> 1.481759**。

K1 IQ2_XXS 的相同 flags 汇编明确显示了反作用：组合原型将 metadata 重读省掉，却新增 **8 条 vslidedown**，静态向量指令 **129 -> 141**，新增 2 处向量 spill 和 2 处对应 reload；另外有 4 处 callee-save 保存/恢复标记，不能把全部 8 个 Folded 注释都称为热循环 spill。原版与恢复版的这几项计数一致。它省的是 8 B scalar metadata 重读，而不是原来估计的 32 B。

SG IQ1_S/IQ2_S 的 Folded 标记数都未增加，不能把它们的回退也归为新增 spill。这里只有组合实现的实际 A/B 结果，没有逐一隔离 cache 与 CSE 的贡献。子代理使用简化 march 得出的额外 spill 结论未采用，以真实 runner 保留的同 flags 汇编为准。

C 影响检查范围为 **184 份已有可重生成 artifact**：第一版 9 份变化，组合版 16 份变化，均无编译失败。这不是 184 项硬件重跑，也不是 202 项完整覆盖；SG IQ1_M prefill 不在该集合，之后单独补了默认数值与机械验收。撤回差异保存于仓库外 `rejected-issue-cache-fixedpoint.diff`，没有引入 feature flag 或保留双主干。

## 有限绑定选择

使用已有 tuner，source 和其余 bindings 固定；各次搜索有独立预算，拒绝结果与实测结果同时保存，未构造覆盖合法性的成本分数。

| SG IQ2_XXS，LMUL-eighths=32 | unroll=1 | unroll=2 | unroll=4 | unroll=8 |
| --- | ---: | ---: | ---: | ---: |
| standalone GOP/s | 1.994419 | 1.649432 | 1.756176 | 1.580837 |
| decode GOP/s | 1.987533 | 1.655124 | 1.063629 | 1.061590 |

standalone 首次共 16 个 LMUL×unroll 候选：LMUL-eighths=8 的 4 个候选被未闭合的 indexed-entry part-to-lane conversion 拒绝；64 的 4 个候选被 full-product carrier 资源合同拒绝；16 的 4 个合法候选均明显更慢。没有把未测到的组合写成最优。

在选定的 32/1 上另查 4 个 pipeline/prime 候选：depth=1/2 分别为 1.993999/1.979269 GOP/s；prime=1 因不存在所需 grouped/layered storage edge 被拒绝。decode 独立枚举 4 个 unroll。共保留现有 depth=1、prime=0。

unroll 改动没有减少数学乘法或每块 reduction 数。实际 SG 内循环从 136 条指令执行 2 次，变为 61 条执行 4 次，即每块 **272 -> 244 条**；其中向量指令均为 196 条、indexed loads 均为 20 条、widening reductions 均为 8 条。差别落在地址/标量序列及其调度，不是少算了 product；两版都没有热循环 spill，18 个 Folded 注释只是 callee-save 保存/恢复。

新增边界项 IQ1_M SG prefill 另查 3 个 LMUL：8/16 分别在两路 reload 活跃区间相交处需要 **38/41** 个 vector groups，超过 32，明确拒绝；原 32 仍最快，2.288634 GOP/s，未改配置。其当前 final IR peak=31；main/correction 的 free-axis partial/reload 合同仍待补，不能靠缩小 LMUL 无条件解决。旧 C 和当前 C 还存在其它历史变化，未把旧数字下降单独归因到某一 pass。

## 仍缺的关系

本轮把 IQ2_XXS 的剩余差别收窄到了以下两个具体方向，尚未实现，不能宣称已达到硬件下界：

1. 自然 unsigned u16 的字节投影：`u32(field[word_index]) >> (8*byte_phase) & 255`。在当前输入上字节地址可化为 `8*group + entry`，但现有 `analyzeIndexedEntryRelation` 只接受直接 payload Iota，storage-window matcher 不穿过这条 widening/shift/mask 链。所需是保持读取时点、范围、alignment 和结果 layout 的完整 byte-window 关系。独立输入见 dequantize/iq2_xxs.py 与 mul_mat/iq2_xxs_staged.py。
2. word1 的 scalar/vector 消费选择：必须比较当前只读一个高字节的 scalar 路径，与共享完整向量值所增加的跨 reduction lifetime、提取和 reload。直接共用映射或无条件扩大 rematerialization 已有负例，不能继续把“读取次数更少”当作排序依据。

另一条等价的索引表达路线是 `4*group + floor(entry/2)` 的 outer-stride/inner-repeat 关系。仅给单轴 regular matcher 加 unsigned shr 不足以覆盖它，也不能把一个向量 group base 塞进要求 scalar source base 的现有 regular-repeat op。没有为这条尚未闭合的关系加一个不生效的特判。

参考职责再次对照了只读 Triton `Coalesce.cpp:82-118` 与 TileLang `loop_vectorize.cc:216-261`：从访问/consumer 和资源约束选表示，不把 kernel 名或格式名放入编译器。未触及的类别在待定 CSV 中明确标为继承上轮定位，不冒充本轮新结论。

## 验收与复现

本轮共 **83 条普通配置运行记录，覆盖 36 个不同表内输入**，全部通过现有外部数值判据，记录中的 max absolute/relative error 均为 0；这不扩展为任意输入的数值证明。另有 **27 个显式调优候选，15 个数值运行通过、12 个编译拒绝**。没有全量重跑 202 项，也没有增加单测或验证脚手架。

保留程序共做 **25 次机械重放，覆盖 21 个不同输入**：16 个恢复版、最终 IQ2_XXS 双机 8 个入口、IQ1_M SG prefill 1 个边界复跑；集合有重叠。全部 parse/verify，通过清 `resources_materialized` 与三项资源统计后的 rematerialization、memory/leaf/resource 收尾重放，二次原始文本 diff=0；重新生成 C 与对应已执行 C 直接比较一致。被撤回原型不作为最终机械验收结果。

实际重放后缀为：

```text
cse -> eliminate-dead-layouts -> canonicalize-layouts -> cse
-> eliminate-dead-layouts -> plan-memory -> materialize-replica-storage-loads
-> hoist-loop-invariants -> select-operations -> finalize-leaves
-> cse -> eliminate-dead-layouts -> materialize-read-snapshots
-> materialize-resources -> eliminate-dead-layouts -> verify-final
```

随后再清 marker/统计，执行 CSE、dead-layout、resource、final verifier；整个流程重复两次比较。上述简称对应 `weft-riscv-*` passes，完整命令保存在 results.json。

仓库根目录可直接执行：

```bash
WEFT_KEEP_ARTIFACTS=1 examples/run/weft-quantized-vec-dot.sh sg2044 iq2_xxs 10
WEFT_KEEP_ARTIFACTS=1 examples/run/weft-mul-mat.sh sg2044 iq2_xxs decode 10
WEFT_KEEP_ARTIFACTS=1 examples/run/weft-quantized-vec-dot.sh k1 iq2_xxs 10
WEFT_KEEP_ARTIFACTS=1 examples/run/weft-mul-mat.sh sg2044 iq1_m prefill 10
```

不要设置额外 `WEFT_AUTO_*`、`WEFT_META_*` 或 `WEFT_TUNE_SELECTION`。SG/K1 仍使用既有单核、10 次 cold median、64 MiB eviction、Clang flags 和 ggml oracle 合同。

证据根目录：`/tmp/weft-ratio95-issue-supply.uhgKOr`。各 `baseline/issue-cache/issue-fixedpoint/restored/selected-final/boundary-final-*.jsonl` 记录配置、数值、计时和本地/远端 artifacts；4 个 `*-tune/` 保存 search、全部 candidate 和 selected；`restored-mechanical/`、`selected-final-{sg,k1}-mechanical/`、`boundary-final-mechanical/` 保存机械验收命令。生成代码、汇编与原型差异均留在仓库外，未用 hash/checksum 校验。

保留绑定与当时已完成的表格刷新已提交为 `215a9363e`；本报告及最终边界测量另作一次性结果提交。没有把仍低于 0.95 的条目算作完成。
