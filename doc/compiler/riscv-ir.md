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

## 3. Memory 与 local storage

`MemDescType` 保存 pinned Encoding、shape/axes、static stride/origin facts、alignment、alias、
effect、layout identity与record geometry。kernel入口随后用 `memory_view` 的 SSA operands明确
给出每个动态 extent、stride和origin；ABI descriptor不能绕过这个 view。

Encoding field access用 `AccessAttr` 保存 selected memory form以及 natural、grouped-layered、
joined 的完整 storage geometry。terminal lowering可以据此计算确定地址和bit extraction，
不能按量化格式名补布局。

`LocalType` 是 invocation-local、target管理的可寻址对象，明确 size、alignment、alias、
purpose、owner domain、birth与lifetime。当前真实用途包括 state/handoff object、spill slot与
pipeline window；它不能替代 caller-visible workspace。动态对象必须由紧邻的
`local_capacity_guard` 给出 target-bounded上界，否则 resource pass拒绝它。

`staged_view` 是 reload placement 的唯一 SSA birth authority；register materialization也有真实
SSA result。staged lifetime不靠无 result marker保存。

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
- every target-local terminal op具有唯一 exact [local leaf](leaves.md)；
- register、fragment与local-storage summary不超过target profile。

当前没有实现的 conversion、spill、pipeline或extension形态必须在某个 physical pass中明确
unsupported，不能由terminal emitter给默认值或走另一条路径。
