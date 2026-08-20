# Canonical Kernel IR

Canonical Kernel IR 是本轮唯一持久前端表示。Python AST、inline 展开环境和 axis 分配表都是构造期对象；物理 layout、LMUL、microtile、packing candidate、fragment 和指令选择不进入该 IR。

## 类型

```text
!weft_kernel.encoding<family, kind, layout_identity>
!weft_kernel.view<encoding, shape, axis_ids>
!weft_kernel.slice<encoding, shape, axis_ids>
!weft_kernel.value<element, shape, axis_ids>
!weft_kernel.domain<axis, id, relation, extent, partition, multiplicity, tail>
!weft_kernel.point<domain>
```

标量 Value 直接使用 MLIR 的 index/integer/float 类型；正 rank Value 使用 `!weft_kernel.value`。shape 中负整数是 canonical symbolic extent identity，axis id 始终为正且在一个值内唯一。

`encoding.kind` 当前区分 `base`、`dense`、`derived_family`、`derived_instance` 与 primitive-private `ephemeral`。base、dense 与 derived instance 必须带固定 layout identity；derived family 必须不带，避免把抽象生成族伪装成已实例化 ABI。

前端 `kernel` 可以引用源码中声明的 `derived_family`，因为此时 artifact candidate 尚未选择；它不是最终函数 ABI。artifact 构造必须先执行对应 builder，把该类型实例化为带固定 identity 的 `derived_instance`，再交给 target lowering。前端不得凭空伪造 identity，target lowering 也不得直接消费抽象 family。

## 结构 operation

- `encoding_decl`：完整纯布局字段表与 padding。
- `derive` / `derive_yield`：编码族生成 region。
- `kernel` / `return`：AOT 入口及参数 View。
- `root_domain` / `domain`：Level domain 的父子关系。
- `level`：domain + carried operands/results + state/staged/body 三个 region。
- `births_yield` / `handoff`：Level 的结构终结符。
- `for` / `if` / `while` / `yield` / `condition`：普通有序控制。
- `new` / `materialize` / `admit` / `commit`：值生命周期与 View 边界。
- `slice` / `field` / `extract` / `update`：内存区域、encoded field 和局部 Value 索引。
- unary、binary、compare 与语言基本 op：完整局部数值关系。

不存在 `VLAOp`、`BlockType`、`RegionType`、`MatmulOp`、格式专用 quant op、ordered/pipeline permission 或 old command op。

## Verifier 边界

verifier 检查布局字段合法、domain/partition 结构、Level 三个 region 的签名与 terminator、普通控制 carry、Value/View domain、基本 op 的 shape/type 关系，以及 engine role 兼容。

verifier 不证明数学等价、不证明有限位宽不溢出、不替作者选择另一棵树，也不验证性能。当前 frontend-only 工具 `weft-opt` 可解析并验证 canonical IR；RISC-V target 对新 schema 的消费属于下一轮，不能通过保留旧 op 来伪装已完成。
