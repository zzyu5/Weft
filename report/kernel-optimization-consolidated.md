# 区间优化与 95% 目标推进汇总

范围：2026-09-06 至 2026-09-07，从“重新核账，不默认把 IQ1_M 差距归为作者 variant；先看 70–90% 区间”这条要求开始，合并后续四轮结果，截止 `94bb4452b`。此前的 DSL 表面重构、读取值语义修复不计入本报告的新增成果。

本文件替代四份逐轮报告及三份待定 CSV，不简单拼接旧结论。下面的最终数字采用合并时的 [对比表](kernel-performance-comparison.csv)；机制前后对照会注明对应阶段。本文是工作结果，不是设计规范。

## 1. 最终结果与口径

- 对比表共 **202 项，172 项 ratio >= 0.95，30 项仍低于 0.95**。没有完成全部过线，也没有证明这些剩余项已经达到硬件下界。
- 区间阶段实际是 **31 项处于 70–90%**，不是旧上下文的 27 项；加边界与 SG TQ1_0 后先跑 39 项。后续扩大到全部低于 0.95 时，选中 **51 项并全部先重跑**。
- 51 项阶段结束时剩 30 项；成对读取阶段，IQ2_XS SG standalone 在改动前复跑已过线，剩 29 项；最后相关回归发现 IQ1_M SG prefill 的旧 0.954927 已过时，复跑为 **0.868933**，回到 30 项。这不是三个不同口径的“完成数”，也不能把净减少全部算成代码优化收益。
- SG IQ2_XXS standalone/decode 的旧表曾约为 0.93，当前主链首跑实际约 0.39–0.40。最后通过合法绑定调优到 **0.493772 / 0.501450**；不能用旧约 93% 的数字继续归因。
- 未修改 baseline、runtime 容差和计时边界；没有 whole-kernel 模板、编译器按格式/kernel/target 名分支、静默 fallback 或新的一层 planning IR。

只保留三个 CSV：固定 [baseline](baseline/ggml-riscv-kernel-performance.csv)、[Weft 性能](weft-kernel-performance.csv)、[对比表](kernel-performance-comparison.csv)。本次整理不改三表数值。需要明确一个尚未解决的数据同步问题：Weft 表仍是较早的完整快照，而对比表记录了后续定向重跑；归一化 kernel/target/phase 并排除小数格式差别后，**191/202 项 Weft 吞吐不一致**，baseline 对应值则 **202/202 一致**。因此现存 Weft 表不能被误称为这些定向优化之后的最新明细，也不能用它覆盖对比表的新结果。现行 [测量协议](../doc/experiments/protocol.md)仍要求该表整批替换，这项口径没有在本次文件清理中擅自改变。

## 2. 保留的实现及收益

始终按 [optimization-principles 第九节](../doc/compiler/optimization-principles.md#9-可复用诊断动作)的七步核账：物理轴与资源、动态工作、供应身份、memory edge、partial/reduction/spill、调度、donor 对照。参考只读 Triton 的 layout conversion/coalescing 与 TileLang 的 layout cost/vectorization 实现，不引入 GPU thread 根模型或供体模板。

### 2.1 索引、读取与资源关系

- **索引拼接正确性。** `rvvPartToLanePieces` 检查轴顺序，不能只看乘积和 LMUL；terminal 区分完整 carrier 容量与 active lanes。清理死 index/conversion 链后，`canReprojectIndex` 只检查最多 32 个纯节点，不复制 memory producer。修复 IQ3_XXS 的错误 sign 映射，不能将原来数值错误的快版本作为性能基准。
- **静态供应窗口共享。** `MaterializeRISCVReplicaStorageLoads` 在同 field SSA、record 坐标、常量连续范围、无干扰写入的条件下复用窗口；至多检查 32 个候选，子窗口由已有 `rvv_issue_slice` 供应，不新增宽读。IQ1_S qh load 每块由 SG 9 次、K1 5 次均降到 1 次。prefill 的动态多轴窗口不宣称命中。
- **bitmask 选择。** `FuseRISCVBitplanes` 同时接受 conversion 包裹和直接带 RVV layout 的 u1 extract，并拒绝跨 layer 的非法窗口；Q1_0 prefill 恢复可执行。这是正确性/可执行性恢复，不冒充新增加速。
- **真实不别名与存储 ABI。** 10 个 dequantize 入口声明输出与输入分离，19 个矩阵入口声明 workspace/output 的实际分离事实。IQ2_XXS 的 66 B、align 2 record 保留自然 `u16[32]`，不用未对齐的 `u32[16]`。
- **producer 与子域。** `PropagateRISCVLayouts` 保留完整 producer 的 lane span，不让 block-sum 子域反向缩小 Q8_K 输入 carrier；`CanonicalizeRISCVLayouts` 在单用纯转换链上按消费者合法 partition 重物化 dense unit load，保持原读取点。
- **受限地址宽度。** `PlanRISCVMemory::materializeEntryByteOffsets` 对同坐标直接 unsigned 8/16 -> 32 扩展，在完整 byte offset 可放入 u16 且 layout 合法时选择 EEW16；不对任意已计算的 u32 链追加 narrow。
- **完整 accumulation 与 carry。** `selectZeroSeedProducts` 只在同坐标、完整有效域中将零种子的第一项选为 widening multiply；`selectLoopCarriedWidenProducts` 只在闭合的完整单轴 si16 SCF carry/yield 链上合并 product+add，后续保持原数值边界。
- **机械重放。** `weft-opt` 注册原有收尾 passes，rematerialized clone 使旧 leaf/implementation 失效，再由原 owner 重选并重新核算资源；不能补回旧 final marker 假装通过。

上述改动及合法绑定的累计效果，使九个 SG dequantize 输入在 51 项阶段达到 0.95 以上：IQ1_M、IQ2_S、IQ2_XXS、IQ3_XXS、Q2_K、Q3_K、Q4_K、Q5_K、TQ1_0。典型的该阶段首跑 → 合并时最新 ratio 为：

| 入口 | Target | 阶段首跑 | 最新 ratio |
| --- | --- | ---: | ---: |
| IQ1_M dequantize | SG | 0.675653 | 0.989218 |
| IQ2_S dequantize | SG | 0.806789 | 1.619290 |
| IQ2_XXS dequantize | SG | 0.814311 | 1.631906 |
| IQ3_XXS dequantize | SG | 0.703328 | 1.358189 |
| Q2_K dequantize | SG | 0.660119 | 1.040218 |
| Q3_K dequantize | SG | 0.931457 | 1.653471 |
| Q4_K dequantize | SG | 0.646112 | 1.069643 |
| Q5_K dequantize | SG | 0.931211 | 1.087556 |
| TQ1_0 dequantize | SG | 0.599995 | 1.115979 |
| Q8_K quantize | SG | 0.891705 | 1.170920 |
| Q8_K quantize | K1 | 0.806818 | 1.089299 |
| TQ2_0 standalone / decode | SG | 0.894496 / 0.921332 | 1.317986 / 1.325378 |

这些是累计保留改动后的结果，不虚构各 pass 的独立贡献。完整当前数字只在对比表保存，不另建一份逐项性能表。

### 2.2 等价作者表达与绑定

IQ3_XXS 将 `entry_code = 2*entry + code` 的 [4,2] 坐标合成 [8]；原坐标由除法/取余恢复，地址、32 项归约分组、scale 和最终浮点位置不变。SG 每 64-element group 的 C 级 axis broadcast 从 4 个降为 3 个；这是等价坐标改写，不是新 numerical variant。该阶段修正索引后的正确 standalone 从 1.729791 到 2.207702 GOP/s；后续仍有测量波动，不能把这个阶段值冒充最终表值。

IQ3_S/IQ3_XXS 在作者层使用 16×16 output blocking，每个输出沿 K 的算术顺序不变。IQ3_S SG prefill 从该阶段 0.914057 到最新 **1.630038**；IQ3_XXS SG prefill、K1 decode/prefill 最新为 **0.960485 / 0.958418 / 0.951379**。IQ3_XXS standalone 双机仍未达线。

IQ2_XXS staged body 共享，SG 选择局部 u32 坐标入口，K1 保留原 Index 入口；runner 尊重 kernel 旁的 entry binding，不再覆盖它。K1 Q6_K prefill unroll 从 4 改为 2，frame 由 784 B 降为 640 B，最新 ratio **0.895079**，仍未达线。绑定保留在 `examples/kernels/*/tuning.json`，不迁入编译器，不把有限子集的 winner 称为全域最优。

### 2.3 成对自然读取

`ShareRISCVLayeredWindows.cpp::materializeSegmentPairs` 将同一 natural field 的
`field[base+2*i]` 与 `field[base+2*i+1]` 选择为一个 `rvv_segment_pair_load`，保留两个原 layout 的结果和后续 widening/add/scale/reduction。偶、奇地址不同，这不是 load CSE。

合同包括同 owner/name/typed geometry、静态 base 差 1、完整一维 part map、无干扰写、对齐、NF×LMUL<=8、exact leaf 和临时资源。emitter 只拼写已选 `vlseg2`；不处理多轴、tail、动态 base 或跨写入区间。规则在 IQ1_S、Q4_K、Q5_K 双机 standalone/decode 共 12 个输入生效；6 个 prefill 回归不作为命中正例。

SG Q4_K 的 bsum 从 **4 条 vlse16 变 2 条 vlseg2e16**；IQ1_S 双机及 Q4_K/Q5_K K1 为 2 条变 1 条，实际 tuple binding 无额外搬运。SG Q4_K standalone/decode 从该阶段 **0.786178 / 0.798548** 到 **0.882407 / 0.944265**；IQ1_S SG 从 **0.783627 / 0.790392** 到最新 **0.805554 / 0.811497**。这些都仍低于 0.95，不能写成已全面闭合。

### 2.4 TQ1_0 与 IQ2_XXS 的最终选择

TQ1_0 的“8 条 shift/mask 是固有成本”判断被推翻：当前每块是 tail16 + 5×3×16，共 **16 次 digit issue**；donor 为 5×32 + 5×16 + 16，共 **11 次**。混合 32/16 原型能减少分组，但引入 slice/widening 或 spill，standalone/decode 没有同时受益，已撤回。

保留的是通用 carry MAC：SG standalone/decode 为 **1 vwmul + 15 vwmacc + 1 vwredsum，0 vadd**。si16 lane 上界为 16×128=2048，最终 i32 reduction 不变。该阶段首跑 ratio **0.651496 / 0.668161**，最新 **0.743306 / 0.763169**；SG prefill **0.572751** 仍待定。没有为此再增加一棵 radix 作者树。

IQ2_XXS 最后一轮只保留 SG 两个 physical unroll 绑定 **2 -> 1**：standalone 从 **1.650001 到 1.994018 GOP/s**，decode 从 **1.649730 到 1.995992 GOP/s**，约 +21%，最新 ratio **0.493772 / 0.501450**。每块实际内循环指令由 272 降为 244，向量指令仍为 196、indexed loads 仍为 20、widening reductions 仍为 8；差别是标量/地址序列及调度，不是少算了 product。

standalone 扫 16 个 LMUL×unroll 候选，再检查 4 个 pipeline/prime 候选；decode 独立检查 4 个 unroll。较小/较大 LMUL 有明确的 conversion/resource 拒绝，depth=2 更慢，prime=1 缺必要 storage edge；最终仍 depth=1、prime=0。K1 绑定未改，同两条最新 **1.098346 / 1.132237**。SG/K1 prefill 为 **1.012068 / 0.867742**，不能用 standalone 的通过替代 prefill 验收。

## 3. 推翻的判断与撤回的尝试

### IQ2_XXS：必须按实际汇编数读取

旧报告按 C 中拼接 u16 的表达式，把 sign/scale 重读算成 64 B/block，**这个账不成立**。相同 flags 的实际汇编显示 sign 读 32 B，scale 被 LLVM 缩成每组一个 `lbu`、共 8 B，因此 metadata 是 **40 B**。grid 的半字重复读为 64 B，donor 直接读取 grid bytes 为 32 B；q 字段合计 **104 B 对 donor 64 B**，不含 d 与 lookup tables。

让 lhs/rhs/deferred scale 共用 `cloneIssueWindow` 映射，再让 conversion CSE 进入 rematerialization 固定点，虽然减少 C 中重读，却没有获得可保留的性能：

- 仅共用映射：IQ2_XXS K1 standalone/decode 从 1.099026/1.131900 降为 **1.036264/1.067614**。
- 两者组合：K1 进一步降为 **0.933293/0.965424**；新增 8 条 `vslidedown`、2 处向量 spill 与2处 reload，另有 callee-save 保存/恢复，不能把全部 Folded 注释当作热循环 spill。
- SG IQ1_S standalone/decode 组合原型降为 **0.763484/0.769206**，撤回后恢复 **0.805554/0.811497**；SG IQ2_S decode 为 **1.365554 -> 1.481759**。SG Folded 标记数未增加，不能套用 K1 的 spill 归因。
- 这两版都已撤回。真实第二输入是 IQ1_S；IQ3_XXS 在 C 对照中未变化，不算命中。未独立隔离 cache 与 CSE 的贡献，不报告虚构的 pass 加速比。

现在需要比较的是“省 8 B scalar 重读”与“跨 reduction 保存完整向量、提取及 reload”的成本。不是看到共享就采用，更不能因此默认去改 scale/reduction 数值树。

### 其它负结果

- **IQ2_XXS u32 storage：**66 B record 的未对齐读取使 K1 SIGBUS，退出 135；撤回，保留 u16。
- **TQ1_0 混合 32/16：**m2 standalone/decode 为 4.303212/4.249680 GOP/s；m4 为 5.579834/4.489552。分组减少但未同时胜出，不证明其它表示不能优化。
- **过宽 zero-seed：**若干 prefill/lookup 输入回退约 10–20%；收紧到完整有效域闭合链并重跑负例。
- **Q8_K 直接 m8 宽读：**SG/K1 ratio 约 0.758/0.713，资源代价抵消 load 数下降；撤回，改用 producer/consumer 合法 partition 关系。
- **任意 u32 运算链追加 narrow：**29 个数值通过输入仍出现回退，部分 cast 表示 unsupported；撤回，只保留直接 extension。
- **IQ3_S dequant global 4-index：**数值通过但 ratio 0.352574；撤回，half-byte/entry projection 仍待补。
- **IQ2_XXS staged 强加 I8X8 ABI：**SG/K1 ratio 0.808430/0.705253；再加局部 u32 坐标也只有 0.822410/0.775215。选出 64-bit gather 不等于快；最终两机保留各自实际较好的 entry。
- **放宽 Field 身份比较：**两版在 184 份可重生成 C 上均变化 0，撤回。这不是184项硬件运行，final IR 中重复 FieldOp 也不能证明更早的匹配点就是瓶颈。
- **继续盲扫参数：**Q4_K 双入口 unroll=8 仍最好；K1 Q6_K 更小 MR/NR、SG Q1_0 更宽 LMUL 均未胜出。参数无收益不是“必须新增作者树”的证明。
- **旧值与波动：**IQ3_XXS standalone 有较大波动；K1 F32 prefill 曾首测2.944075、同 C 复跑3.831085 GOP/s。不能把所有下降都归因于当轮改动，也不能选择最高单次结果冒充稳定收益。

## 4. 当前 30 项待定：按 13 类实体归并

以下条目与合并时对比表的全部 `ratio < 0.95` 项对应。standalone 指 vec-dot，decode/prefill 指 `mul_mat_*`；原始 kernel key 与数值统一以对比表为准。这里保留缺口和归属，不再创建待定 CSV。“需要”是已定位的关系，不意味着已经隔离了全部性能差距的时间占比。

1. **IQ1_S/IQ1_M 兄弟供应与 partial，9 项。** IQ1_S 双机 standalone/decode，另 SG prefill；IQ1_M 双机 standalone/decode。IQ1_S qh raw read 已唯一，bsum 成对读取已完成，但 index/delta 的 issue 表示仍分开。IQ1_M main/correction 每块仍有 8 activation loads、8 products、32 partial reductions，donor 为两个最终 reduction。需要共同坐标供应、支配点、投影 lifetime 和联合 partial-set/resource；不得直接改变 scale/widening 边界。归属：`PropagateRISCVLayouts::reductionSupplyRoles`、`ShareRISCVLayeredWindows`、`MaterializeRISCVPartialAccumulators::matchReplicaScaledDotReduction`。无条件共享已有负例。

2. **IQ2_XXS metadata/字节投影，2 项。** SG standalone/decode 为 0.493772/0.501450。自然 u16 的 `u32(field[word_index]) >> (8*byte_phase) & 255` 在本输入上可化为字节地址 `8*group+entry`；需要保持原读取点、范围、alignment 与结果 layout 的闭合 byte-window 关系。另一条路是 outer group 步距与 inner entry repeat 的二维索引，单独补 unsigned shr 不够。归属：`PlanRISCVMemory`、`RISCVPhysicalSupport::analyzeIndexedEntryRelation`、`cloneIssueWindow`。独立输入在 `dequantize/iq2_xxs.py` 与 `mul_mat/iq2_xxs_staged.py`；metadata 工作账采用40 B修正版。

3. **IQ2_XXS staged entry 选择，1 项。** K1 prefill 0.867742。entry ABI、u32 相对坐标和16×16 blocking 单独试过仍更慢；需要在同一 indexed-entry 合同内比较 scalar/unit 与 vector/gather 的合法 layout/resource，并处理重复供应。归属：`PlanRISCVMemory::selectIndexedEntryPhysicalChoice`、`mul_mat/iq2_xxs_staged.py` 与其绑定。

4. **IQ3_XXS entry 配对与 index lifetime，2 项。** 双机 standalone 为0.761414/0.942121。metadata 已共享、直接 extension 已选EEW16；每64元素仍是16个4 B entry，对 donor 8个8 B entry，sign u32 链与data/partial同时活跃。需要成对entry的源坐标关系、sign广播复用和联合live-set。归属：`PlanRISCVMemory::materializeEntryByteOffsets`、`ShareRISCVLayeredWindows`、partial materializer。

5. **TQ1_0 radix carrier，3 项。** SG standalone/decode/prefill 为0.743306/0.763169/0.572751。carry MAC已完成；16对11个digit issue、center/narrow/widen 和prefill replica carry仍需联合表示与资源合同。归属：`selectLoopCarriedWidenProducts`、`CanonicalizeRISCVLayouts`、partial materializer；额外shift不是已证明的数学下界。

6. **IQ4_NL/MXFP4 值域，4 项。** K1 两格式 standalone/decode。opaque i8 table参数允许-128，两项 `(-128)*(-128)` 之和为32768，不能无证明地合入i16；codebook load实际已经在循环外。需要可验证的作者常量/值域合同，或合法更宽partial。归属：作者table合同、`RISCVPhysicalSupport::integerRange`、`PlanRISCVPartialTopologiesPass`，不是格式名分支或新的hoist。

7. **Q4_K joined metadata，2 项。** SG standalone/decode 为0.882407/0.944265。bsum的4条strided已变2条segment，不再列为未修项；剩余为8个scalar LBU与2个VL8 indexed metadata reads，而donor先读取完整metadata再decode。需要跨joined-field原始byte-window identity及scalar/vector消费共享。归属：`PlanRISCVMemory`、`ShareRISCVLayeredWindows::MaterializeRISCVReplicaStorageLoadsPass`。

8. **Q2_K 子寄存器归约序列，2 项。** SG standalone/decode 为0.824402/0.828327。双方load/product/scale/extract工作相当；Weft为16次16-lane归约，donor为32次8-lane归约。需要同一整数程序的subregister reduction/issue选择与延迟/资源合同；未单独隔离该差别的时间占比。归属：`PlanRISCVPartialTopologiesPass`、`SelectRISCVOperations`。

9. **Q4_1 replica reduction/extract 调度，1 项。** K1 prefill 0.912010。MR4×NR2的Xq已共享，8路交错reduction/extract产生16次vtype切换；实际有FMA，不是缺FMA。需要联合调度并保留tail/resource边界。归属：partial materializer、`SelectRISCVOperations`。

10. **Q6_K scalar/address lifetime，1 项。** K1 prefill 0.895079。unroll4->2已缩小frame，但仍有地址和scalar partial reload；需要GPR压力与issue调度共同规划，不笼统归为metadata或vector spill。归属：partial materializer、physical schedule/resource。

11. **IQ3_S half-byte/entry 投影，1 项。** SG dequantize 0.732900。4-element lookup payload的reshape和half-byte sign window关系未闭合，global-index原型更慢。需要保留bit phase、8-bit storage和资源的坐标投影。归属：`RISCVPhysicalSupport`、`ShareRISCVLayeredWindows`、`CanonicalizeRISCVLayouts`。

12. **Q1_0 mask/replica 投影，1 项。** SG prefill 0.825931。Xq已跨8个输出共享，mask仍先解码成±1，再每输出两次byte MAC；现有signed-bitmask matcher只支持完整单lane axis的scalar output。需要mask broadcast/free-axis/replica partial关系，以及-128的范围证明或先widening后变号。归属：`FuseRISCVBitplanes::matchSignedBitmaskReduction`、partial materializer及signed-mask leaf合同。

13. **IQ1_M prefill replica/reload live set，1 项。** SG prefill 0.868933，是后续回归新发现的待定项。当前MR2/NR1，final vector peak=31、local storage=0；main/correction各两路输出，每组4个i16m2 reduction。LMUL-eighths=8/16在两路reload相交处需要38/41组，超过32而拒绝；原32绑定仍最快。需要free-axis sibling partial的联合供应与reload lifetime。归属：`MaterializeRISCVPartialAccumulators`、`MaterializeRISCVResources`。旧C有多处历史变化，未把旧0.954927的下降单独归因到本轮原型。

此前独立列出的 IQ2_XS SG scale-convergence 项已因后续默认复跑达到0.95以上而退出待定；不能把三份历史待定CSV的并集当作当前清单。

## 5. 验收、复现与证据

各阶段覆盖有重叠，不能相加声称唯一输入数：

- **区间阶段：**最终169个不同表内输入数值通过，其中127个原过线回归仍达到source吞吐；23个结构相关配置完成final与resource closure前layout checkpoint重放。首轮IQ3_XXS数值失败、Q1_0编译拒绝未当成功记录。
- **全体低于0.95阶段：**成功记录覆盖183个不同表内输入，共451条成功记录，包含未保留候选；没有重跑整个202项。各改动边界分别完成清marker重放，包含alias/remat 47、zero-seed 25、domain/dense 53、direct-index 6、carry-MAC 2及输出blocking 8等有重叠集合。
- **成对读取阶段：**42个不同表内输入，52条默认记录，另8个调优候选；18个cohort输入及最后4次默认复跑完成机械重放，重生成C与已执行C一致。
- **最后issue阶段：**36个不同表内输入、83条普通记录数值通过，误差记录均为0；另27个显式候选中15个运行通过、12个明确拒绝。25次机械重放覆盖21个不同输入，全部二次diff=0且重生成C与已执行C一致。

数值判据是已有ggml外部oracle；SG/core48/VLEN128与K1/core3/VLEN256，10次cold median、64 MiB eviction、相同Clang与 `-ffp-contract=fast`，未放宽容差。机械稳定不替代真机数值，不证明未执行passes幂等或全体程序等价。后续强重放对副本清 `resources_materialized` 及三项资源统计，执行rematerialization、CSE/dead-layout、memory/replica-load/hoist/select/finalize/snapshot/resource后缀；再次CSE后重新清统计并闭合资源，完整流程重复两次比较原始文本。早期checkpoint路径与后来的清marker路径不能混成一次验收。

从仓库根目录使用普通runner即可复现，每条都会完成DSL emit、编译与目标机数值/计时，而不只是生成IR：

```bash
WEFT_KEEP_ARTIFACTS=1 examples/run/weft-quantized-vec-dot.sh sg2044 iq2_xxs 10
WEFT_KEEP_ARTIFACTS=1 examples/run/weft-mul-mat.sh sg2044 iq2_xxs decode 10
WEFT_KEEP_ARTIFACTS=1 examples/run/weft-quantized-vec-dot.sh sg2044 q4_k 10
WEFT_KEEP_ARTIFACTS=1 examples/run/weft-mul-mat.sh sg2044 tq1_0 decode 10
WEFT_KEEP_ARTIFACTS=1 examples/run/weft-mul-mat.sh sg2044 iq1_m prefill 10
WEFT_KEEP_ARTIFACTS=1 examples/run/weft-row-dequantize.sh sg2044 iq3_s 10
WEFT_KEEP_ARTIFACTS=1 examples/run/weft-kernel.sh k1 q8_K_quantize 10
```

其余待定项沿用同一runner，替换target/format/phase即可；不设置额外 `WEFT_AUTO_*`、`WEFT_META_*` 或 `WEFT_TUNE_SELECTION`。供体逐段对照在 `source/c/ggml/llama.cpp/ggml/src/ggml-cpu/arch/riscv/quants.c`，不是生成代码依赖。

原始证据留在仓库外，临时目录不是永久归档：

- 区间：`/tmp/weft-band-clusters.p7HTN0/`、`/tmp/weft-band-final-runs.BxYFeq/verified-final.json`；坐标改写补跑在 `/tmp/weft-iq3-flat-runs.t06mR8/`，TQ负例在 `/tmp/weft-tq-wide-runs.o1txGr/`、`/tmp/weft-tq-m4-runs.LksbvI/`。
- 低于0.95：`/tmp/weft-ratio95-closure.BAIWpR/`，含selection、各主题JSONL及各边界的 `*-mechanical/results.json`。
- 成对读取：`/tmp/weft-ratio95-supply.ShJU9a/`，含segment-pair/segment-final运行与重放、unroll调优及两个零变化Field身份原型。
- issue：`/tmp/weft-ratio95-issue-supply.uhgKOr/`，含baseline、两版原型、restored、selected-final与boundary-final记录，四个tune目录、实际同flags汇编与 `rejected-issue-cache-fixedpoint.diff`。

本次替换的四份报告分别在 `dcb3a74e4`、`0c3b12a47`、`fa5d1582e`、`94bb4452b` 中保留原文；已删除的六份CSV亦可从清理前Git历史恢复。不新增报告副本、验证脚手架或其它CSV。
