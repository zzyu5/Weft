# 数值语义

数值属性定义作者允许的等价实现集合。它们不是性能 hint，也不能由 target 为了命中某条路径
静默放宽。

## `order`

- `ordered`：按语言指定的顺序执行，不允许重结合；
- `preserve`：保持 primitive 规定的分组/状态关系；
- `relaxed`：在 dtype 与 primitive 语义允许的范围内重结合局部计算。

Reduce、scan、summary 与 sequential carry 各自定义可观察顺序。Ordered state 不能因为向量化
被替换成无序 reduction。

## `math`

- `strict`：使用规范定义的精确/舍入行为；
- `native`：允许 target 的原生近似实现，但保持 command 的输入输出关系；
- `fast`：允许规范明确列出的额外有限值近似。

`native` 不等于任意 fast-math，也不允许改变 NaN、infinity、saturation 或 signedness 合同中已
明确要求的部分。

## 默认值

当前 collective command 的默认 `order="relaxed"`、`math="native"` 适合模型 kernel 的常见
性能语义。源码无需重复写默认值。需要严格复现或有序 recurrence 时，作者必须显式选择更强
合同；编译器不能根据 kernel 名猜。

## Dtype 与 accumulator

Operand dtype、output dtype 与 accumulator dtype 是语义。Widen/narrow/cast 保持逻辑元素映射，
SEW/LMUL 和寄存器 footprint 的变化由 target 推导。Saturation、rounding 和 bitcast 必须由对应
op 明确表达。

## Validity

Masked load 没有 `other` 时产生 validity-carrying value。Invalid lane 不是任意可读取数据；它在
进入不接受 masked input 的 command、store 或跨 region 前必须由显式 fill/select 闭合。Target
可使用 tail/mask 指令，但不能扩大有效域。

## Quantized numerics

Packed code、scale、minimum、zero point、codebook、correction 与 accumulation order 都是 typed
quant command 的语义。RVV/IME realization 可以改变 decode chunk、widening chain 和寄存器组织，
不能按格式名替换数学关系。
