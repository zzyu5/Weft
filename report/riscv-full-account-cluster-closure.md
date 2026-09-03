# 全量账后 failure 与性能簇收口

日期：2026-09-04

范围：本报告记录全量账之后按顺序处理的 correctness、compile-failure 与性能簇。只记录当轮
真实 IR、生成代码和双机运行结果；未重跑的历史行不在这里重新解释。

## 1. IQ3_XXS encoded-entry alignment

### 1.1 修改前工作账

- IQ3_XXS record 为 98 bytes，声明 alignment=2；metadata 是 offset=66 bytes 的 u32 field。
- 每个 indexed-entry consumer 只发生一次 metadata load；没有两个相同 logical offset 可由 CSE
  合并。因此该修复预计删除的动态 load 数为 0。
- 原 Physical IR 选择 element-width unit load，生成 `vle32`；连续 record 的奇数项只保证
  2-byte alignment。K1 standalone、MUL_MAT decode、prefill 均 SIGBUS。
- 改动不改变 logical value、Level、widening、consumer 数、reduction 或 pipeline；它只把同一
  memory edge 的合法 leaf 从 EEW32 改为 EEW8 load 后在同一 register group 内 reinterpret。
- 即使指令计数不下降也必须实现，因为原 leaf 在已证明的 alignment 上不合法；收益首先是从
  runtime fault 恢复成正确程序，而不是吞吐优化。

### 1.2 结果

从 Encoding base alignment、field offset 与 entry byte stride 推出 effective alignment，Physical
IR 显式选择 `rvv.indexed-entry-byte-load`；final verifier 拒绝夸大的 alignment 或未闭合的宽 load。
96 份 row-dequant/vec-dot Physical IR 中只有 SG/K1 的 IQ3_XXS vec-dot 选择该 leaf，其余 94 份
保持原 leaf。96/96 均可独立 parse/verify，安全运行 canonicalizer、CSE、Share 两次；同一流水
第二次运行文本 diff 为 0。

| target | entry | 修前 | 修后（10 repetitions） | source | 修后/source | numeric |
|---|---|---:|---:|---:|---:|---|
| SG2044 | standalone vec-dot | 0.283 | 2.433 GOP/s | 2.674 | 0.910× | error 0 |
| SG2044 | MUL_MAT decode | 0.283 | 2.448 GOP/s | 2.618 | 0.935× | error 0 |
| SG2044 | MUL_MAT prefill | 0.284 | 2.439 GOP/s | 2.920 | 0.836× | error 0 |
| K1 | standalone vec-dot | SIGBUS | 1.239 GOP/s | 1.448 | 0.856× | error 0 |
| K1 | MUL_MAT decode | SIGBUS | 1.246 GOP/s | 1.436 | 0.867× | error 0 |
| K1 | MUL_MAT prefill | SIGBUS | 1.245 GOP/s | 1.472 | 0.846× | error 0 |

## 2. IQ2 staged widening-dot closure：实现前预测

IQ2_S、IQ2_XS、IQ2_XXS 的 staged prefill 在 SG2044/K1 共六个入口均停在同一个 typed
widening-dot legality check。三棵 canonical tree 已显式包含 MR/NR free axes、entry/payload
reduction、K blocking、materialize 与 accumulator lifetime；当前没有作者树缺轴的证据。

七步工作账得到：

1. hot contraction 的 result 是保留 MR/NR free axes 的 scalar register tuple；两个 si8 operand
   都是 RVV carrier，target widening 与 doubled LMUL 合法。
2. 数学 MAC 数、logical entry/payload coordinate 与 source value graph不变；本修复预计不删
   canonical work。
3. supply identity不合并；storage proposal只是 producer形态，不能成为另一 operand 的
   contraction carrier。
4. 失败发生在 memory leaf选择之前；本节点不预测改变 unit/indexed/gather 数。
5. 两 operand 的 reduction lane/time decomposition不一致，导致 lane slices、stream count或
   part projection不能形成同一个 target operation。
6. pipeline尚未形成，本节点不碰 load-to-use距离。
7. Triton 先以 result MMA encoding为 parent构造两个 DotOperandEncoding并插真实
   `convert_layout`；Weft 应同样由 layout pass冻结共同 reduction carrier，Lower只验证和消费。

预测：在最后一次 use-def propagation 后重新冻结同一个 typed reduction carrier，六个入口会从
compile failure变成合法 Physical IR；Q4/Q5 等既有 widening-dot入口的最终 topology与真机数字
不变。若六个入口仍失败，说明问题不只是 carrier owner；若能编译但动态工作没有形成既有 partial
program，则该节点只修 legality，不能把性能收益归给它。第二输入不是另一个格式名，而是三种不同
IQ2 storage relation与现有 Q4/Q5 widening-dot关系共同经过同一规则。

### 2.1 实际闭合过程

预测只命中了 legality，没有命中“现有 partial program 会自然形成”这一隐含期待。实现与反例依次是：

1. layout propagation 只在两个 operand 各自带有独立 surviving free axis 的 blocked contraction 上
   冻结共同 reduction carrier。reduction axes 保持各自的 typed axis identity，最后一个 lane-bearing
   reduction axis 是 primary axis，其余 reduction axes 在同一 lane carrier 中线性化。
2. `PartialSetType.termsPerSlot` 改为完整多轴 lane carrier 的元素数；planner 冻结 primary axis、
   source set 与 combine set，materializer、verifier、emitter 共用同一 lane-count 与 slice-LMUL 合同。
3. 第一次生成的 C 数值错误且三档 LMUL 得到同一个错误结果。静态逐 lane 检查定位到 IQ2_S 的 sign
   window：逻辑布局是 `[scale_group=2, entry=2, payload=8]`，旧 memory plan 把 entry 放在 replica，
   却把每个 physical part 当作 16 个连续 bit 载入。entry 0 实际需要 bit `0..7,16..23`，`vlm`
   读成了 `0..15`。
4. bitmask window verifier 现在逐 lane 证明 logical bit offset 连续；memory planner 选择不超过 consumer
   lane width 的最宽连续 storage suffix。IQ2_S 因而形成 8-bit unit mask window，再由显式
   RVV `time_to_lane` conversion 合成消费方的 16-lane carrier。
5. 直接把 32 个 sign bit 全放进 lane 会使 live peak 达到 43/32 vector groups，因此未保留。8-bit
   window 的 peak 为 4 groups，既不伪造连续性，也不靠超预算 carrier。

这里的 typed conversion 采用完整 source/result axis→time/lane/replica 关系；terminal translation
只按已闭合映射拼接 source parts。Triton 的对应机制是 `minimalCvtLayout` 先求完整 relative layout，
`ConvertLayoutOpToLLVM.cpp:282-415` 再把 register/lane mixed mapping 分解并拼写；
`RemoveLayoutConversions.cpp:1629-1689` 则在此前独立完成传播、rematerialization 与 cleanup。Weft
不能直接采用 thread/warp basis，但“中间 conversion 允许被 pass 消除，final conversion 才必须有
terminal contract”的边界相同。

实现中主动撞出一个反例：若在通用 `ConvertLayoutOp` verifier 创建 op 时就要求 terminal
part-to-lane 拼写，IQ1_S、IQ3_XXS、TQ1_0 双机六个既有 vec-dot 会在后续 DCE 之前被拒绝。
严格合同因此只放在 final verifier；这六个入口恢复后均通过下面的机械验收，没有扩大 emitter。

### 2.2 双机结果

六个原 compile-failure 均在真实 production shape 上数值零误差并完成 10 repetitions：

| format | target | Weft | source | Weft/source | 结果 |
|---|---|---:|---:|---:|---|
| IQ2_S | SG2044 | 0.528363 GOP/s | 2.506224 | 0.211× | 正确，但慢 |
| IQ2_S | K1 | 0.771796 GOP/s | 1.469172 | 0.525× | 正确，但慢 |
| IQ2_XS | SG2044 | 0.355132 GOP/s | 4.809702 | 0.074× | 正确，但慢 |
| IQ2_XS | K1 | 0.427212 GOP/s | 1.796205 | 0.238× | 正确，但慢 |
| IQ2_XXS | SG2044 | 3.981193 GOP/s | 4.111831 | 0.968× | 正确 |
| IQ2_XXS | K1 | 1.481867 GOP/s | 1.705121 | 0.869× | 正确 |

因此本节点只把 6 个 FAIL 变成了 6 个可执行结果，不能记作六条性能闭合。final Physical IR 的
差异直接解释了分档：

| typed physical work / inner body | IQ2_S | IQ2_XS | IQ2_XXS |
|---|---:|---:|---:|
| `rvv_issue_slice` | 16 | 16 | 0 |
| `rvv_widen_accumulate` / finalize | 8 / 8 | 8 / 8 | 0 / 0 |
| partial set / combine / finalize | 0 / 0 / 0 | 0 / 0 / 0 | 2 / 4 / 2 |
| generated-C `vslideup` | 8 | 8 | 0 |
| generated-C `vslidedown` | 40 | 10 | 2 |
| generated-C widening reductions | 36 | 12 | 6 |

IQ2_XS 还在每个 inner issue 中执行 8 次 16-bit scalar scale load，随后再次载入 4 个 packed
scale byte；其 16 个 8-lane table loads 先 slide-pack，再由 indexed gather 消费。IQ2_XXS 则已把
work 物化成两个完整 partial set，combine 后才 finalize。低速来自这三棵 staged tree 进入不同
typed materialization contract 后的真实动态工作，不是 conversion legality 或一个漏掉的 CSE。

### 2.3 机械验收与回归

- 24 row-dequant + 24 vec-dot × 2 targets：96/96 生成 final RISC-V IR，96/96 独立 parse/verify，
  运行 canonicalizer、CSE、layout canonicalization、Share 两次与 final verifier；同一流水第二次
  运行 96/96 文本 diff 为零。
- 新增 IQ2 staged 三格式 × 两 target：6/6 通过同一流水，第二次运行 6/6 文本 diff 为零。
- bitmask legality 改动实际覆盖 Q1_0、Q5_0、Q5_1、IQ2_S、IQ3_S。五类 row-dequant、standalone
  vec-dot、MUL_MAT decode 在双机共 30 个真实入口全部数值零误差、10 repetitions。与 CSV 旧快照
  相比除 SG Q1_0 standalone `4.300→4.622` 外均为小幅测量波动；最大负向波动是 SG Q1_0 decode
  `4.522→4.315`（-4.6%），没有数量级退化。

当前性能边界没有被隐藏：IQ2_S/XS staged 已经有合法共同 carrier 和正确 memory edge，但仍没有
形成 IQ2_XXS 那种完整 partial set；继续优化它们属于 partial/materialization 的下一类工作，不能
归入本次 layout closure 的收益。
