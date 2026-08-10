# Selected Execution IR

## 20. Selected Execution IR

### 20.1 只保存物理选择

Selected IR 保存无法从 canonical IR 与固定 target facts 唯一重算的选择，例如：

```text
meta binding
VLA region → RVV/scalar realization
SEW / LMUL / repeat / unroll
memory strategy
reduction / scan strategy
contract provider
register microtile
fragment / packing / scratch
local fusion decision
math primitive strategy
multiversion specialization identity
```

### 20.2 不复制算法

Selected IR 禁止复制：

- source loop bounds；
- pointer expression；
- logical shape；
- contraction axes；
- logical predicate；
- reduce combiner；
- state identity；
- ABI；
- output order。

这些事实必须通过 canonical references 重算。

### 20.3 计划记录

概念记录：

```text
weft_execution.plan
weft_execution.meta_binding
weft_execution.axis_plan
weft_execution.memory_plan
weft_execution.reduce_plan
weft_execution.scan_plan
weft_execution.summary_plan
weft_execution.contract_plan
weft_execution.math_plan
weft_execution.primitive_plan
weft_execution.scratch_plan
weft_execution.specialization
```

`axis_plan` 可以引用 canonical VLA axis 或 logical block axis；它只保存 Scalar/RVV、SEW、
LMUL 与 unroll 等物理选择，不复制 logical extent 或 axis role。`plan` 同时绑定唯一 canonical
kernel symbol 与静态 target profile facts，使 emitter 不需要第三个事实来源。

Extension provider 可以定义 sibling selected op，但必须实现统一 selected-provider interface，并只引用 canonical anchor。

### 20.4 决策单位不是 kernel 类别

Selection 的最小输入单位是：

```text
canonical anchor
+ logical axis / relation
+ typed operands and effects
+ target facts
+ local use context
```

Compiler 禁止先回答“这个 kernel 是 GEMM、RMSNorm、q4_K 还是另一类算子”，再选择一条
互斥路径。Kernel 名、算子名、量化格式名和整体 shape pattern 都不是 selection fact。
格式名可以标识作者提供的 source variant，但不能成为 provider、plan 或 emitter route。

一个 logical axis 的语义角色由 canonical op 及其 use relation 决定，而不是由 kernel 分类器
赋予。Weft 没有 GPU program space；物理选择只描述该 axis 在当前 worker 内如何 realization：

| Canonical axis/relation | 不可改变的语义 | Selected 可以记录的选择 |
|---|---|---|
| scalar ordered axis | loop boundary、order、carry | unroll、hoist、scalar instruction schedule |
| VLA axis | logical extent、每点一次 | Scalar/RVV、SEW/LMUL、dynamic `vl` policy |
| logical free block axis | output relation | serial、register repeat、microtile、fragment free axis |
| contracted axis | paired relation、accumulator | reduction strategy、FMA/dot、fragment contracted axis |
| broadcast axis | logical replication | scalar broadcast、register reuse、materialization |
| irregular index relation | 精确 address/index relation | scalar、strided、indexed load 或 gather |
| logical predicate | active logical positions | mask、AVL absorption 或 scalar guard |

这些语义角色能从 canonical IR 重算，不应再物化成一份需要与 canonical 保持一致的长期
schedule schema。Selected 只保存“多个合法 realization 中最终选了哪个”。

### 20.5 局部候选必须可组合

Compiler 分别为每个 canonical anchor 构造 provider candidates，再联合检查 use relation、
effect、register、fragment、scratch 和 target legality。同一个 kernel 中自然允许：

- 多个 VLA region 选择不同 LMUL 或不同 Scalar/RVV realization；
- 多个 reduce、scan、summary 各自拥有独立 strategy；
- 多个 irregular index relation 同时存在；
- ordered scalar flow、irregular memory 与 staged contract 同时作用；
- scalar control、RVV memory、RVV reduction 和 extension contract 混合组成一个 entry。

新增结构应增加或扩展 primitive-local candidate family，而不是增加新的 whole-kernel 分支。
Scalar realization 是普通、显式且需要满足 legality 的 candidate，不是异常路径或静默
fallback。任何 anchor 没有合法 realization 时，selection 必须明确失败。

### 20.6 表示压缩

Weft 只有三类持久承重表示：

1. Canonical Kernel IR：唯一算法真理；
2. Selected Execution IR：只保存无法从 canonical 与固定 target facts 唯一推出的选择；
3. source、object、header、library 与 dispatcher artifacts。

Axis/use analysis、candidate set、capability query、resource equation、owner-local IR、layout
projection 和 instruction matching 都是可重算的瞬态实现。它们不得拥有独立持久 schema，
也不得通过额外 verifier 维持与 canonical/selected 的镜像一致性。

Selected record 必须引用 canonical anchor。Emitter 只能从 canonical IR、Selected IR 和静态
target facts 取得事实；不存在 kernel classifier、派生事实缓存或第三份调度结构作为额外
source of truth。

### 20.7 架构回退判据

出现以下任一实现形态即视为回退：

- `KernelKind`、operator family 或 q-format 决定整条 lowering；
- 以整体 kernel shape 匹配某个已知 emission template；
- 两个 irregular relation、两个 state flow 或 state + contract 因互斥分支不能共存；
- emitter 从 tensor shape、函数名或代码形状重新推导 source loop、state 或算法 identity；
- unsupported primitive 静默转入 catch-all emitter 或默认 fallback。

---
