# Canonical Kernel IR

Canonical Kernel IR 是 DSL 与所有 target 之间唯一持久语义。它不保存候选、选中 layout、LMUL、
microtile、pipeline buffer、packing 或 fragment。

## Kernel 与 ABI

`weft_kernel.kernel` 保存 symbol、参数名、参数 kind、return type 与 source location。Pointer type
保存 element type、access、noalias/restrict、alignment、storage class 和 persistent format。

`weft_kernel.storage` 记录 `W.buffer` 的 caller-owned workspace/persistent shape，不执行分配。

## 值类型

- scalar：控制、地址与 scalar state；
- `!weft_kernel.ptr`：typed memory pointer；
- `!weft_kernel.block`：static/dynamic logical axes 上的 engine value；
- `!weft_kernel.region`：词法 VLA domain 上的 engine value；
- masked value：显式 validity；
- tuple：closed scalar state。

Block/region 的 shape、axis identity 和 element type 是逻辑事实，不是物理 layout。

## 控制

`weft_kernel.for` 保存 lower/upper/step、普通 SSA carry 与两个语义属性：

- `traversal = "ordered" | "blocks"`；
- `pipeline = false | true`。

`pipeline=true` 只表示作者允许对当前已存在循环做依赖合法的局部流水；不承诺 target 必须采用
流水，也不保存 stage/depth。`if`、`while`、`vla` 与 yield 保持普通 SSA/control contract。

## 逻辑轴

`weft_kernel.block_index` 是 source `W.axis` 的 canonical 表示，保存 extent、offset 与唯一 axis
identity。Pointwise、load/store、cast、reduce 和 structured command 必须通过这些 identity
说明对应、broadcast、free 与 reduction relation。

## Memory 与 effect

Load/store 保留 pointer footprint、predicate、fill/validity 和 alignment。地址计算仍是普通 SSA，
target 从轴对 pointer 的变化推导 unit/strided/indexed/segment relation。Canonical IR 不保存某个
target 的 memory instruction choice。

`W.transfer` 当前不增加独立 canonical op：frontend 将它展开为语义等价的一次 load 与 store，
因此所有 target 复用同一 memory facts。

## Structured command

Source command 使用稳定 canonical op：

```text
W.vdot   → weft_kernel.dot
W.gemm   → weft_kernel.matmul
W.reduce → weft_kernel.reduce
W.scan   → weft_kernel.scan
```

Op 保存 typed operands、axis relation、init 与 numerics。结果是普通 SSA value。Extension/quant
op 只保存其局部、可观察数值语义，不保存完整 microkernel。

## Verification

Verifier 检查 source semantics，而不是 target preference：

- control carry type、axis 与 dynamic extent identity；
- pointer access、storage shape、effect 与 validity；
- pointwise/broadcast domain；
- dot/matmul free/reduction axis 与 dtype；
- reduce/scan/state order；
- quant/extension operand shape、format 和 numerics；
- canonical dialect 之外的未知 op 不能混入 kernel。

Target 不支持某个合法 canonical program 时返回明确 unsupported；不能把 target 缺口伪装成
source verifier 规则。
