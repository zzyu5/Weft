# Canonical Weft Kernel IR

## 作用

Kernel IR 是 Python DSL 与 RISC-V target lowering 之间唯一长期保存的表示。它记录作者程序的
可观察语义，不记录目标机器的一次选择。

一个 module 包含一个或多个 `weft_kernel.kernel`。每个 kernel 保存 entry name、参数类型、
返回类型和一个有序 region。未知 `weft_kernel.*` 或 `weft_ext.*` operation 会在解析时失败。

## 类型

核心类型为：

```text
scalar T
!weft_kernel.ptr<T, access, noalias, alignment,
                 restrict, external|workspace|persistent, format>
!weft_kernel.constexpr<T>
!weft_kernel.block<[D0, ...], [a0, ...], T>
!weft_kernel.region<[D0, ...], [a0, ...], T>
!weft_kernel.masked<T>
tuple<T0, ...>
```

`block` 与 `region` 同时保存 shape 和 axis identity。正 axis ID 必须对应唯一
`weft_kernel.block_index`；`0` 只表示显式 singleton dimension；VLA 维使用 `-1`。相同 extent
不等于相同 axis。

## 控制与 SSA

- `weft_kernel.for` 保存有序 scalar range 与任意 scalar/block carry；
- `weft_kernel.while` 的 condition/body region 显式传递同一组 carry；
- `weft_kernel.if` 的两个 region 产生同类型 results；
- `weft_kernel.vla` 保存逻辑 `[begin,end)` 与跨完整 VLA 域产生的 state results；
- `weft_kernel.yield`、`condition` 与普通 SSA use-def 完成所有值传递。

Block result 可以多 use、进入 pointwise、memory、state、控制 carry 或另一个合法局部运算。
IR 不编码 one-use、direct-store、相邻 consumer 或某个后端闭包。

## Storage

`weft_kernel.storage` 只声明 entry 中 workspace/persistent pointer 的 shape。Pointer type 同时保存
element type、access、alignment、alias、storage class 与 persistent format。普通 input/output
使用 external pointer，不需要 `storage`。

Primitive-private temporary 不进入 Kernel IR，也不能成为 entry 参数。

## 运算

Kernel dialect 包含：

- scalar constant/meta value、cast/bitcast、unary/binary/compare/select；
- pointer arithmetic、load/store、block index、singleton-axis view、full；
- reduce、scan、argmax、online softmax summary、sort indices；
- dot、matmul、lookup、decode、narrow；
- for、while、if、VLA 与 entry return。

`weft_ext` 只保存普通 Kernel op 无法无差别表达的局部语义：标量 little-endian f16 load、
packed quantized dot 与 RISC-V 扩展相关局部运算。Affine/symmetric i4×i8 dot 自身完整定义当前
N16×K32 packed byte relation；不另设只能在特定 consumer 中生效的 packed block view op。

## Verifier

Verifier 在最早能判断的位置拒绝：

- axis identity、shape/domain、dtype 或 validity 不一致；
- 非 scalar Python control condition；
- VLA 中非法 outer-state mutation 或 nested VLA；
- storage class、shape、workspace alias 或 persistent format 缺失；
- dot/matmul reduction axis、init/result domain 或 numerical attributes 不完整；
- extension operand 不足以独立定义其局部语义。

Verifier 不验证 LMUL、register 数、fragment 或 instruction legality；这些由一次 target lowering
根据 target profile 判断。
