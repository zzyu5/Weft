# Storage Ownership 与 Lifetime

## 唯一 storage 模型

Weft 不提供源级隐式分配，也没有 `W.alloc`。作者可观察、可跨 operation 或 loop 存活的
memory object 都由调用者分配，并作为普通 kernel entry pointer 传入。DSL kernel 用 pointer
qualifier 声明 ownership，用 entry body 中唯一的 `W.storage` 声明非 external object 的逻辑
shape；Kernel IR 与生成的 C header 保存同一信息。

语言只定义以下四类 ownership：

| 类别 | 分配者与有效期 | DSL/IR 表达 | 是否进入 entry ABI |
|---|---|---|---|
| external | caller 持有；至少覆盖本次调用，也可由应用跨调用保存 | 默认 `W.external` pointer | 是 |
| persistent | caller/部署系统持有；按声明格式跨调用复用 | `W.persistent(format)` + `W.storage` | 是 |
| workspace | caller 为本 worker 的一次调用独占提供；内容可跨 loop/primitive 更新复用 | `W.workspace, W.noalias` + `W.storage` | 是 |
| primitive-private temporary | target lowering 在一个 local primitive realization 内创建和销毁 | 不进入 Kernel IR | 否 |

这四类不能互相兜底。Target 不得把缺失的 workspace 变成隐式 ABI，也不得把 external byte
pointer 自动解释成 persistent packed object；DSL kernel 也不能命名或观察 primitive-private
temporary。

## DSL 表达

External 是 pointer 的默认 storage class：

```python
input: W.ptr[W.f32, W.readonly]
output: W.ptr[W.f32, W.writeonly, W.noalias]
state: W.ptr[W.f32]
```

External 包括普通 input/output 以及由应用跨调用保存的可观察状态。对象是否在应用中长寿命，
不会自动把它变成 persistent；`persistent` 专指具有显式 target-compatible format identity 的
可复用 representation。

Persistent packed object 必须写出非空、稳定的 format identity 和完整逻辑 shape：

```python
packed_weight: W.ptr[
    W.u8,
    W.readonly,
    W.noalias,
    W.persistent("affine_i4_n16_k32_304b"),
]

W.storage(packed_weight, (column_blocks, k_blocks, 304))
```

该 identity 是 caller 与 kernel 共享的 ABI 格式名，不是 register layout、LMUL、fragment 或
target candidate。Weft kernel 不隐式构建它；若应用需要 builder，builder 必须是另一份显式
DSL kernel 或应用构建步骤，不能藏在 consumer lowering 中。

Worker-local algorithmic workspace 也必须写出完整 shape，并声明 `W.noalias`：

```python
query_scratch: W.ptr[W.f16, W.workspace, W.noalias]
accumulator_scratch: W.ptr[W.f32, W.workspace, W.noalias]

W.storage(query_scratch, (head_dimension,))
W.storage(accumulator_scratch, (head_dimension,))
```

Workspace 是算法可观察的 memory state：作者决定何时写入、跨哪些 loop/primitive 复用以及何时
内容失效。它在一次 invocation 内由当前 worker 独占；若多个 worker 共用同一地址，caller
违反声明。需要跨调用保留的普通算法状态应声明为 external，而不是 workspace。

`W.storage(pointer, shape)` 必须直接位于 kernel entry body，每个 persistent/workspace pointer
恰好一次；external pointer 禁止使用它。Shape 的每一维必须为正，并能由常量、已绑定
`constexpr` 或 runtime `index` ABI 参数的显式算术表达式求得。它描述逻辑容量，不赋予 target
改变 blocking、packing 或 lifetime 的权限。

显式消费workspace的primitive必须把自己的访问extent与该storage shape建立可证明关系。例如
`W.sort_indices(..., scratch, extent)`要求`scratch`的rank-one storage extent与`extent`为同一
SSA identity或同值常量；无法证明时在 frontend/Kernel IR 边界拒绝，而不是生成可能越界的 intrinsic C。

## Pointer facts 与 lifetime

Storage class 与以下 pointer facts 正交：

- `readonly` / `writeonly` 是 access 属性；
- `noalias` 与 `restrict` 是 alias/ownership assertion；
- `aligned(bytes)` 是调用时 minimum alignment assertion；

未写 `aligned` 时，C header 中的 alignment 值为 `0`，表示 DSL kernel 没有承诺比 element natural
alignment 更强的条件；它不是“地址无需对齐”。所有 pointer 的 bounds、实际 lifetime、格式内容
与声明一致性由 caller 保证。

Block、tuple 和 state 的 SSA lifetime 由 lexical scope 与 control-flow carry 决定；它们可以跨
scalar loop iteration 存活而不必写入 workspace。只有作者显式选择 memory staging/reuse 时才使用
workspace。Target lowering 可以把纯 SSA value 放在 register、rematerialize、reload 或局部 spill，但
这些选择不能改变 author-visible storage ownership。

## Kernel IR 与 C header

Canonical pointer type保存 element、access、alias/alignment、storage class 与
persistent format identity：

```text
!weft_kernel.ptr<
  T, access, noalias, alignment, restrict,
  external|persistent|workspace, storage-format
>
```

`weft_kernel.storage` 只绑定 kernel entry 的 persistent/workspace pointer，并保存 shape 与显式
extent operands。Primitive-private temporary 不进入 pointer type、`storage` op 或 entry ABI。
Target primitive若要求特定persistent representation，必须在physical selection前核对pointer的
format identity；caller仍负责传入内容、bounds、alignment与lifetime确实满足声明。

同一次 `weft-compile --emit=intrinsic-c` 必须同时生成 intrinsic C 和 C header。Header
保存 entry declaration；为每个 pointer给出 storage class、format、alignment、noalias/restrict metadata；
为 persistent/workspace 另外给出 rank、逐维 extent 与 element-count query。调用者据此分配和
绑定 storage。Header 是 Kernel IR 中 entry ABI 的投影，不是第二份 kernel semantics。
