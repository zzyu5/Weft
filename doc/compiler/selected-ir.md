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
weft_execution.meta_binding
weft_execution.vla_plan
weft_execution.memory_plan
weft_execution.reduce_plan
weft_execution.scan_plan
weft_execution.summary_plan
weft_execution.contract_plan
weft_execution.scratch_plan
weft_execution.specialization
```

Extension provider 可以定义 sibling selected op，但必须实现统一 selected-provider interface，并只引用 canonical anchor。

---
