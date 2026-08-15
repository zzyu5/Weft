# Weft 文档索引

本目录定义当前唯一的 Weft DSL、Kernel IR 与 RISC-V lowering。`source/` 只保存 baseline；
Weft Python 程序称为 DSL kernel，`examples/` 中的入口称为 examples，编译结果直接称为
intrinsic C、C header、object 或 executable。

阅读顺序：

1. [语言定位与程序模型](dsl/model.md)
2. [Python DSL 与控制流](dsl/python.md)
3. [VLA、memory 与 block](dsl/vla-memory-and-blocks.md)
4. [Storage 与生命周期](dsl/storage-and-lifetime.md)
5. [Dot、matmul 与 state](dsl/structured-compute.md)
6. [数值语义](dsl/numerics.md)
7. [Python DSL API](dsl/api.md)
8. [编译器结构](compiler/architecture.md)
9. [Kernel IR](compiler/kernel-ir.md)
10. [RISC-V target lowering](compiler/target-lowering.md)
11. [编译与生成结果](compiler/lowering.md)
12. [构建期选择](compiler/tuning.md)
13. [真实运行复核](compiler/verification.md)

真实 DSL kernel 的组织与设计说明位于 `kernels/`：

- [Elementwise](kernels/elementwise.md)
- [Normalization](kernels/rms-norm.md)
- [Predicate、reduce 与 scan](kernels/predicate-reduction.md)
- [Online softmax](kernels/online-softmax.md)
- [Worker-local blocked GEMM](kernels/gemm.md)
- [Attention 组合](kernels/attention-composition.md)
- [Packed quantization](kernels/quantized-local.md)
- [RISC-V 局部扩展运算](kernels/extensions.md)

文档中的可变运行数字不具有规范地位。本轮问题与结果写入一次性的 `report/` Markdown；
当前 Weft 性能数字只记录在 `report/weft-kernel-performance.csv`。
