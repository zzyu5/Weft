# RISC-V Physical IR

## 1. Program identity

RISC-V IR 是 Weft 的第二层、也是最后一层程序 IR。它混用 `func`、`scf`、`arith` 与
`weft_riscv` dialect，直接承载同一 canonical program 在一个 RISC-V target 上的物理程序。
它不是 Kernel IR 旁边的 plan，也不另设通用 physical IR。

每个 target operation保留 `source_origin` 供诊断；真实 use-def、control、memory、Level、
conversion、schedule 与 resource entity必须存在于本 IR，emitter不能回查 source closure。

## 2. Physical value 与 layout

`!weft_riscv.value` 保留 element type、logical shape、axis identities，并把 target layout放在
type 中：

```text
logical axis
→ time factor × lane factor × register-replica factor
  × fragment factor × local-storage factor
```

layout还携带 carrier、SEW、LMUL、`vl`、register groups与validity。所有未被 numerical
operation消去的 logical axes必须保留；例如 `[M,K] -> [M]` reduction 的 M 轴不能退化为
一个 scalar 再广播。IME representation使用独立 `FragmentType`；type同时携带
lhs/rhs/accumulator role与完整`FragmentPackingAttr`，不能伪装成 generic
`ValueType(carrier=ime)`。

canonical `reshape` lower为真实的 `weft_riscv.reshape`。op保留作者给出的input-axis
order和不同的result shape/axis identities；layout pass必须为两端形成线性序号相同的carrier
partition。只有time/lane/register/fragment/local各自的总分解一致时，terminal translation才可
把它拼写成零拷贝binding；否则必须在此前插入显式physical conversion或把candidate判非法，
不能由emitter调整坐标顺序。

## 3. Memory 与 local storage

`MemDescType` 保存 pinned Encoding、shape/axes、static stride/origin facts、alignment、alias、
effect、layout identity与record geometry。kernel入口随后用 `memory_view` 的 SSA operands明确
给出每个动态 extent、stride和origin；ABI descriptor不能绕过这个 view。

canonical `subview` lower为 `weft_riscv.subview`：base descriptor、每轴static offset/extent和
缩小后的result descriptor同时存在于IR。它保留Encoding、axis、stride、origin、access与alias，
只向base address加入规则连续offset；其result只能作为`store` destination。它不是indexed
memory edge，也不允许terminal translator把它扩成scatter。

Encoding field access用 `AccessAttr` 保存 selected memory form以及 natural、grouped-layered、
joined 的完整 storage geometry。terminal lowering可以据此计算确定地址和bit extraction，
不能按量化格式名补布局。

natural field 的有效对齐由 record/base alignment、field byte offset 与动态 entry stride共同约束。
当有效对齐不足以支持 element-width unit load 时，Physical IR 必须选择显式 byte-load leaf，
由该 leaf按 byte载入并机械重解释为已经选定的RVV value；terminal emitter不能自行把`vle32`
降级成 byte load。final verifier拒绝任何超过Encoding与offset事实的alignment声明，以及任何
在不足对齐上仍选择element-width vector load的physical edge。

natural field到RVV register的完整窗口用`rvv.replica_storage_load`表示。它的
`StorageWindowPlanAttr`给出logical lane axis、unit/strided projection、extent/alignment、record
坐标与每个time/replica part对应的source window；result layout给出唯一SEW、LMUL与实际`vl`，
leaf给出exact `vle`或`vlse`。因此strided与unit不是terminal emitter根据地址表达式临时选择的
两种拼写。若后续issue materialization缩小窗口，只能投影已选plan，不能重新决定memory form。

多轴entry/payload坐标若按行主序线性化为连续source区间，使用
`rvv_unit_entry_window_load`保存scalar base、entry axes/extents、payload axis/extent与exact unit
leaf。entry axes必须是source中不存在的新逻辑轴；payload axis可以是新轴，也可以复用唯一被投影
掉的source axis，后者明确表示`entry * payload_extent + payload`的连续后缀。其它source-axis
别名、非unit系数或不闭合的layout必须保留为typed indexed edge，terminal emitter不能自行把
indexed access改写成unit load。

对一个已经闭合、只有一个raw byte window的grouped/layered load，physical parameter可以把
exact leaf选为`scalar-prime + vector-load`：先对该window中最后一个active byte执行一次有序的
scalar byte read，再执行原来已经选定的vector load。这个选择直接写在load op的`LeafAttr`中；
未选择时仍是原来的单vector-load leaf。它不读取未来iteration、不产生SSA result、不建立buffer
version，也不改变window、carrier或control，因此不是prefetch/pipeline。多window edge若没有显式
给出prime对象则非法，terminal emitter不能自行挑一个window。

当 target 的固定 record-axis 优先级选择跨 Level instance 的 lane cohort 时，
`RecordCohortType` 将一个 physical point、静态 Level partition 与 cohort width 绑定为同一坐标
实体。`rvv_record_storage_load` 明确保存 source field、cohort、storage unit、record byte stride
及该 unit 覆盖的 logical indices；sub-byte decode 是独立 pure op；`rvv_record_store` 明确保存
dense destination、logical element offset 与 record byte stride。动态 Level trip count仍由真实
主 `scf.for` 和 scalar tail 表示，不能藏在上述 leaf 或 emitter 中。该表示只在 source/destination
的一维 unit-stride encoded-record geometry、完整 active partition 与 target VLMAX 都可证明时合法；
否则保留原来的 within-record mapping。

若regular-repeat relation使同一个encoded scalar覆盖一个或多个完整lane window，
`rvv_regular_repeat_scalar_load`明确保存source axis、dynamic source base、source count、repeat与
每个physical time/replica part对应的`part_bases`。其result仍保留原logical axes以及
time/lane/replica分解，但carrier是scalar；后续pointwise边界因此只能发射已选的scalar-vector
operand form。verifier必须从layout与repeat关系重算`part_bases`，不能让terminal emitter推测
哪个scalar服务哪个lane window。该形式只适用于已证明对齐的完整repeat window；其它relation
保留为typed gather或普通memory edge。

`LocalType` 是 invocation-local、target管理的可寻址对象，明确 size、alignment、alias、
purpose、owner domain、birth与lifetime。当前真实用途包括 state/handoff object、spill slot与
pipeline window；它不能替代 caller-visible workspace。动态对象必须由紧邻的
`local_capacity_guard` 给出 target-bounded上界，否则 resource pass拒绝它。

`staged_view` 是 reload placement 的唯一 SSA birth authority；register materialization也有真实
SSA result。作者 staged birth携带 owner/birth/lifetime；编译器在同一block内冻结一次scalar
supply时使用`physical-share`，其作用域完全由SSA dominance给出，不伪造作者birth。
staged lifetime不靠无 result marker保存。

## 4. Explicit conversion

representation conflict是一个真实 `weft_riscv.convert_layout`：source/result type保存完整
layout，`ConversionAttr`保存 transfer kind/effect/temporaries，leaf保存 exact terminal form。
conversion保持 element、shape、axes和 logical Value identity，只改变 physical carrier或
coordinate decomposition。

conversion canonicalization必须重写/删除该 op。final IR可以保留合同已经闭合的 terminal
conversion；intrinsic-C translator根据 source/result layout确定地展开 slide、splat、tuple
split/merge或local transfer。这与重新选择layout不同。

RVV↔IME handoff不用 generic conversion冒充：`ime_pack`、typed fragment MMA与`ime_unpack`
显式表示 fragment packing、operation和result handoff。

## 5. Composite 与 terminal target operations

进入 terminal emission 前，canonical composite会变成真实的局部程序：

- reduction/contract 的 `scf.for` 和 loop-carried accumulator；
- grouped/encoded `WindowType`、load-window与compute-step；
- RVV contraction step和明确的 lane operand/memory form；
- 独立完整product的`rvv_partial_collect`、partial reduction与scale-combine use-def；
- IME fragment packing geometry、MMA groups/chunks与unpack；
- local alloc/bind/load/store、spill/reload；
- 对已支持window cluster展开后的pipeline prologue、steady state与epilogue。

Window与IME op可以对应一个固定 closed leaf sequence，但其 slots、terms、axis、access、
fragment packing、resources和trip count必须由 type/op attributes闭合。source Level loop、动态
outer traversal、workspace或pipeline不能隐藏在 leaf 中。

## 6. Level、ordinary control 与 schedule

canonical Level转换时保留 domain/partition/multiplicity、birth counts、carried values、handoff
和direction。Level最终成为带 `LevelAttr` 的真实 `scf.for`。ordinary `for/while/if` 仍是有序
scalar control，不因进入RISC-V IR而获得一个编译器发明的 shaped axis。

pipeline参数只有在改写成真实 guard、prologue、steady-state、epilogue与buffer-version carry后
才算实现。未消费 `ScheduleAttr` 或只有一个 depth 字段的module不能进入emission。

## 7. Final IR 不变量

`VerifyFinalRISCV` 通过时必须同时成立：

- Canonical Kernel IR operations已经全部消失；
- every shaped value/block argument有完整、target-legal layout；
- representation冲突由typed conversion或specialized fragment/local op显式承担；
- memory edge具有完整 descriptor、Encoding mapping、access form与leaf；
- source Level births/handoff、ordinary control与ABI `memory_view` identity仍可验证；
- composite、source Level op、`implementation`与未展开 schedule均已消失；
- planner使用的partial layout/combine/nested/sequential/scaled/layered plan attributes已被真实
  use-def、loop与local operations取代，不再是terminal translator输入；
- every target-local terminal op具有唯一 exact [local leaf](leaves.md)；
- register、fragment与local-storage summary不超过target profile。

携带已选 RVV stream reduction 的保留循环必须显式带 downstream-unroll-disable
合同。它的 lane/time 分解和 loop-carried accumulators 已在 Physical IR 中冻结；terminal
translator 只把该合同拼写为本地 pragma，防止 system compiler 再次展开并改变已核算的
live range。缺少该合同由 final verifier 拒绝，不能依赖 Clang/GCC 的启发式阈值。

当前没有实现的 conversion、spill、pipeline或extension形态必须在某个 physical pass中明确
unsupported，不能由terminal emitter给默认值或走另一条路径。
