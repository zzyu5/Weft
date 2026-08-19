# Weft 文档

`doc/` 是 Weft 当前唯一规范。它描述一门面向非 SIMT 处理器的算子 Kernel DSL，以及从
canonical Kernel IR 到本地 artifact 的唯一编译主链。`report/` 只保存一次性的推理和工作
记录，不能反过来定义语言。

Weft 的根模型是：

> 一个 worker/hart 持续执行有序控制程序，并向本 core 内可用的向量、矩阵和数据移动能力
> 发出目标无关的局部命令；作者拥有完整算子算法与数据生命周期，编译器拥有命令内部的物理
> 映射和指令实现。

Weft 只有一门 DSL、一个 canonical Kernel IR 和一条 target lowering。高层局部计算命令与
较细的轴、值和访存构造可以在同一 kernel 中混用；它们不是两门语言，也不形成两条后端。

## 语言

1. [编程模型](dsl/model.md)：执行者、抽象层次、作者与编译器的边界。
2. [Python DSL](dsl/python.md)：可写语法、控制流与 helper。
3. [逻辑轴、值与数据移动](dsl/vla-memory-and-blocks.md)：scalar、memory view、engine value、
   `W.vla`、`W.axis` 与访存关系。
4. [Storage 与生命周期](dsl/storage-and-lifetime.md)：external、workspace、persistent 和
   primitive-private temporary。
5. [局部计算命令](dsl/structured-compute.md)：`W.vdot`、`W.gemm`、reduction、state、lookup
   与 `W.quant.*`。
6. [数值语义](dsl/numerics.md)：顺序、重结合、近似数学和 validity。
7. [公开 API](dsl/api.md)：当前真实可写且可 lowering 的入口。

## 编译器

1. [架构](compiler/architecture.md)：唯一主链与模块权责。
2. [Canonical Kernel IR](compiler/kernel-ir.md)：唯一持久中间表示。
3. [RISC-V target lowering](compiler/target-lowering.md)：事实、合法性、结构选择和参数选择。
4. [Lowering 与发射](compiler/lowering.md)：selected decision 到 intrinsic C/局部 asm。
5. [构建期选择](compiler/tuning.md)：只在合法候选中编译和实测。
6. [真实 repro](compiler/verification.md)：唯一允许的执行验证方式。

## Kernel 说明

`doc/kernels/` 按算法关系说明现有 DSL kernel 如何使用同一语言，而不是为每个算子定义一条
编译路径。当前正式 target 是 RISC-V CPU，包括标准 RVV 与已接入的局部 IME realization；
其他非 SIMT target 只有在其 target facts、局部能力和真实 artifact 完整接入后才算支持。
