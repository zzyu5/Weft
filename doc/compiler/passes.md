# RISC-V 编译 Pass

所有 pass 都改写同一份 [RISC-V IR](riscv-ir.md)。`implementation`、`schedule`、
`lane_operand` 等中间属性附着在具体 operation 上，并在其消费者 pass 中删除；它们不是
module 外的 assignment 表，也不构成新的 IR 层。

## 1. 固定顺序

```text
ConvertWeftToRISCV
→ SelectRISCVOperations
→ PropagateRISCVLayouts
→ SelectRISCVOperations
→ PlanRISCVMemory
→ CanonicalizeRISCVLayouts
→ PlanRISCVMemory
→ SelectRISCVOperations
→ LowerRISCVComposites
→ CanonicalizeRISCVLayouts
→ FuseRISCVBitplanes
→ HoistRISCVLoopInvariants
→ ScheduleRISCVLevels
→ PipelineRISCVLevels
→ SCCP
→ ShareRISCVLayeredWindows
→ PlanRISCVPartialTopologies
→ MaterializeRISCVPartialAccumulators
→ UnrollRISCVLevels
→ CanonicalizeRISCVLayouts
→ PlanRISCVMemory
→ HoistRISCVLoopInvariants
→ SelectRISCVOperations
→ FinalizeRISCVLeaves
→ MaterializeRISCVResources
→ EliminateDeadRISCVLayouts
→ VerifyFinalRISCV
```

该顺序是依赖关系，不是阶段标签。`LowerRISCVComposites` 必须先产生带 typed
operation/effect 的显式 physical loop，`schedule` 才能从真实 use-def 分配 stage/order，
`pipeline` 再只负责版本化 SSA 值并生成 prologue/steady-state/epilogue。partial planner 必须在
generic unroll 之前读取未复制的 contraction/reduction use-def，冻结 carrier、combine topology
与 issue unroll；materializer 生成 issue loop后，`UnrollRISCVLevels`才机械复制迭代。exact leaf
完成后，resource pass 才能把 leaf temporary 与 SSA live interval 一起计入峰值。

## 2. 各 pass 合同

### `ConvertWeftToRISCV`

输入是一个已经实例化的 Canonical Kernel IR candidate、一份 target profile 和一组单值
physical parameter binding。它建立 target-aware `ValueType`、`MemDescType`、Level/control、
完整 runtime `memory_view`、artifact builder 与 numerical operations，然后删除 canonical
operations。初始 layout、access 和 leaf 可以是 `unassigned`，但 source axis、Value、Level、
birth、handoff、effect、Encoding 与 ABI identity 必须一一保留。

一个 physical module 只含一组 meta/unroll/pipeline binding。候选枚举和实测发生在 driver
之外；pass pipeline 不接收列表，也不在 module 内保存备用值。

### `SelectRISCVOperations`

读取 operation semantics、typed operands、axes 和 target capability，按 target 固定优先级
写入一个 `ImplementationAttr`：scalar、RVV 或 IME 的局部结构 family。它不选择 memory
form，也不写 exact intrinsic。没有合法局部结构时当前 module 失败。

### `PropagateRISCVLayouts`

从 selected operation anchors、logical axes、producer/consumer、control carry 和 target VLEN
推导每个 shaped SSA value 的 time/lane/register-replica/local factors、SEW、LMUL、`vl`、
validity 与 register groups。普通无特殊 anchor 的 shaped value使用 target 的固定规则：
最内逻辑轴进入 lane；一元素表示保持 scalar。LMUL 是实现该 lane extent 的最小合法值，
不是 emitter 默认值。
当 encoded field 经过保持 shape/axis 的 conversion 或作为 lookup index 流向下游时，
storage layer 宽度同样沿这条 use-def 链传播；target 不能因此把超过真实
grouped/layered 宽度的逻辑 lane 伪装成一次 indexed load。

pointwise、state 与 control handoff 的要求不一致时，pass 插入真实 `convert_layout`；
`for/if/while` 的 carried/result types 同时被改写。无法保持 logical axes 或无合法 LMUL 时
失败，不通过静默 scalarization 继续。

### `PlanRISCVMemory`

读取 `MemDescType`、Encoding field mapping 和 value layout，为每条 memory edge写入
`AccessAttr` 与 transfer leaf。dense access 得到 unit/strided/indexed；encoded field 得到
natural/grouped-layered/joined 的完整 storage geometry。lookup table被保留为 descriptor edge，
不是先整表 load。contract 的 lane operand 与 lane memory form也在这里唯一确定，并在
composite lowering 中消费。

同一block内，若一个byte-aligned natural encoded scalar沿纯一元链形成一个多消费者
supply，pass插入`register_materialize(realization=physical-share)`。该op没有作者
birth；它只冻结一次已选load/decode结果，使后续不同layout consumer共享同一SSA值。
跨block或跨Level的placement仍需dominance/LCA与effect证明，不能由这条局部规则猜测。

### `CanonicalizeRISCVLayouts`

该 pass 只做已经实现的真实 rewrite：相邻 pure 逆 conversion 消除、单 use 纯 pointwise producer
的 backward rematerialization，以及同一 block 内相同 conversion 的 SSA CSE。它不声称已经
实现跨 block hoist/sink、全局 conversion algebra 或任意 producer rematerialization。

### `LowerRISCVComposites`

把 source-level physical composites 改写为终端前的真实程序结构：

- Level loop变成带 Level identity 的 `scf.for`；
- state/local materialization变成 local alloc/bind/load/store 或 register/staged SSA；
- grouped/encoded reduction变成显式 reduction loop与 typed window load/step；
- ordinary contract变成显式 reduction loop与 RVV step；
- IME contract变成 typed fragment pack/MMA/unpack；
- layout conversion获得唯一 terminal conversion leaf。

未知 conversion kind、缺失 reduction extent、缺失 access 或不闭合 fragment capability均明确
失败。该 pass 不创建 source outer traversal、Level、workspace 或 persistent Encoding。

### `ScheduleRISCVLevels` 与 `PipelineRISCVLevels`

`ScheduleRISCVLevels` 是 scheduler。它读取已 lowering 的 physical `scf.for`、SSA use-def、
loop carry 和 operation effect，把与下一迭代无关的 pure/read producer 放在 stage 0，
把依赖 carry 并生成 yield 的 consumer 放在 stage 1，并在具体 operation 上写入
stage/order。它不通过 op 数量、相邻关系或 window family 命名 cluster。嵌套 region、write/
unknown effect 目前没有 predication/ordering 合同，因此明确拒绝，不得重排。

`PipelineRISCVLevels` 是 expander。depth 1 只消费 schedule；depth 2 消费 scheduler 写入的
stage/order，为所有跨 stage SSA value 增加 loop-carried 版本，支持多个 source carries，
并生成空迭代 guard、prologue、steady-state loop 和 epilogue。当前只实现距离一迭代、
depth=2/buffer=2；嵌套 region predication、write effect、depth>2 和显式 local-storage ping-pong
尚无合法 physical program 时直接 unsupported。仅设置 `pipeline_depth=2` 而没有
dependency-derived producer/consumer cluster 仍会失败，不算 pipeline。

### `ShareRISCVLayeredWindows`

该 pass 读取 typed `grouped_layered` access geometry，并在 pure conversion、无中间
write/unknown effect 的 `local_load` 与 integer index cast 之后，仍用 field owner、logical point
与线性 byte-coordinate 关系证明两个
layer 共享一个 RVV storage window。成功时它以真实 `rvv_layered_window` op 替换两个
extract。当前仅对 full-validity layer 改写；tail cohort 没有证明安全时保持原程序。

### `PlanRISCVPartialTopologies` 与 `MaterializeRISCVPartialAccumulators`

planner 读取 contraction/reduction 的 typed axes、time/lane/replica 分解、storage window、
consumer use 与 target resource budget，唯一写入 product carrier、issue slices、partial-set、
combine/final-reduction topology、selected local instructions 和联合资源合同。顺序 fused contraction
若跨多个 issue，必须得到显式 `sequential_partial_plan`；该 plan 给出每个 issue 的 source-part
投影、唯一 widened accumulator 与最终 reduction，而不是让 composite dot 按 issue 数预留一组
临时 partial。当前该 fused program 只在 lhs/rhs 的 issue slice 与 reduction-only accumulator
具有同一完整 physical-part 映射时合法；带互异 free-axis replicas 的 outer contraction 由 planner
明确选择 `sequential_per_stream`，materializer 不得在失败后自行降级。

materializer 只核验仍存在的 SSA graph 与 plan 一致，并实例化真实
`rvv_issue_slice → rvv_widen_accumulate → rvv_finalize_widen_dot` 或对应 partial-set program。
它不能重新选择 carrier、combine tree、instruction 或 resource groups。没有闭合 plan 的已选
topology 直接失败；terminal emitter 只按 `source_parts` 投影已有 vector binding，并拼写已选
accumulate/reduction leaf。nested、layered 与 scaled/reduced-scaled plan 同样必须在 planner
中冻结 issue types、scale supply、narrow-scale type、storage-window/decode instruction、partial
reduction、finalization leaf 与同时存活的 resource groups；materializer 只能按这些字段创建
operation，不能从 shape 或当前 SSA spelling 再次推断。

### `FinalizeRISCVLeaves`

读取 `ImplementationAttr` 与已经确定的 value carrier/SEW/operand form，为 pointwise、cast、
reduce、state transfer等仍未闭合的 operation写入 exact instruction/spelling leaf，并删除
`ImplementationAttr`。结构 family只在 `SelectRISCVOperations` 产生。由
`LowerRISCVComposites`新建的window step、RVV contract step、fragment和layout-conversion op在
创建时已有自己的exact leaf；本pass只终结仍保留generic operation的那一组。两者作用于互斥的
physical op集合，不为同一个operation重复选择instruction。

### `MaterializeRISCVResources`

用 region-aware SSA live interval和 leaf temporary计算 vector/fragment peak，并把每个 leaf 的
operand/result resource groups写实。资源超限时，只对同一 block 内可合法保存的普通
`ValueType` 插入显式 local slot、`spill` 和各 use-site `reload`；fragment spill、跨 block spill
和任意 pure-producer rematerialization尚未实现时判当前 module非法。

动态 local object必须紧邻一个 typed `local_capacity_guard`。guard给出 target-bounded byte
上界，kernel resource summary按上界计算；缺少闭合上界时拒绝生成 C VLA。

### `VerifyFinalRISCV`

最终 verifier 拒绝：残留 canonical op、非 terminal RISC-V op、`unassigned` layout/access、
`implementation`、未展开 schedule、未选 leaf、丢失 Level birth/handoff、绕过 runtime
`memory_view` 的 ABI edge，以及超出 target 的 register/fragment/local-storage 使用。

operation/type verifier继续检查 logical domain、axis projection、Encoding field、conversion、
window、fragment role/packing和 exact leaf之间的局部合同；final verifier还把descriptor
storage facts与Encoding declaration、leaf widening/memory要求与target capability逐项核对。
成功只说明这份 RISC-V module满足
当前已实现的 terminal op 集合；没有实现的 spill、pipeline或extension结构必须在前序 pass
明确失败。

## 3. 可观察性

每个 pass 的输入和输出都是 program。用 MLIR pass instrumentation dump 同一 module时，应
直接看到 type/layout变化、`convert_layout` 插入或删除、memory attributes、loop/window/
fragment rewrite、pipeline expansion与spill/reload。只打印 analysis table或 assignment
dictionary不能证明程序已经物理化。

## 4. 与 Triton/TileLang 的机制关系

下列路径相对于仓库根目录位于同级reference checkout `../ref/`。Triton 的 layout encoding、`ttg.convert_layout`、Coalesce、AccelerateMatmul、
RemoveLayoutConversions 与 Pipeline同样在 TTGIR 上改写真实程序：

- `../ref/triton/include/triton/Dialect/TritonGPU/IR/TritonGPUOps.td:32`
- `../ref/triton/lib/Dialect/TritonGPU/Transforms/Coalesce.cpp:71`
- `../ref/triton/lib/Dialect/TritonGPU/Transforms/AccelerateMatmul.cpp:441`
- `../ref/triton/lib/Dialect/TritonGPU/Transforms/RemoveLayoutConversions.cpp:42`
- `../ref/triton/lib/Dialect/TritonGPU/Transforms/Pipeliner/PipelineExpander.cpp:51`

TileLang也把layout、buffer与pipeline结果落到真实block/loop，再由LowerTileOp和
InjectSoftwarePipeline重写body：
`../ref/tilelang/src/transform/layout_inference/layout_inference.cc:1196`、
`lower_tile_op.cc:1080`、`inject_pipeline.cc:3608`。

Weft复用的是 typed representation、explicit conversion 与 real rewrite；逻辑值不因此获得
thread/warp/CTA ownership。
