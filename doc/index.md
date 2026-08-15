# Weft 设计文档

[`WEFT_CORE_LOCAL_BLOCKED_PROGRAM_MODEL.md`](../report/WEFT_CORE_LOCAL_BLOCKED_PROGRAM_MODEL.md)
是当前唯一源语言与编程模型说明。本目录只保存该模型在Python surface、canonical IR、target
lowering与artifact上的模块投影，不得定义另一套根模型。

它定义 Weft 最终成品的目标状态；与旧设计、历史材料或实现注释冲突时，以这里为准。

Weft 是一门面向单个 RISC-V worker/hart 的 AOT kernel DSL：作者写一份持续执行的
core-local blocked program，显式拥有 ordered control、block/state 与 source-visible storage
生命周期；编译器只补全 source 不可观察的 VLA、寄存器、指令与局部扩展 realization。

## 阅读顺序

1. [Core-local blocked编程模型](../report/WEFT_CORE_LOCAL_BLOCKED_PROGRAM_MODEL.md)
2. [语言定位与程序模型投影](dsl/model.md)
3. [Python eDSL 与控制流](dsl/python.md)
4. [Storage ownership 与 lifetime](dsl/storage-and-lifetime.md)
5. [VLA、predicate、memory 与 logical block](dsl/vla-memory-and-blocks.md)
6. [Dot、Matmul 与 state algebra](dsl/structured-compute.md)
7. [数值语义](dsl/numerics.md)
8. [Python DSL API 索引](dsl/api.md)
9. [编译器总架构](compiler/architecture.md)
10. [Canonical Kernel IR](compiler/kernel-ir.md)
11. [RISC-V target lowering](compiler/target-lowering.md)
12. [Lowering、工具与 artifact](compiler/lowering-and-artifacts.md)
13. [构建期 tuning](compiler/tuning.md)
14. [Legality 与错误边界](compiler/verification.md)

典型worker-local source的canonical组合方式单独放在 `kernels/`；这些文件是语言构造示例，
不是target使用的kernel类别：

- [Elementwise](kernels/elementwise.md)
- [Normalization workers](kernels/rms-norm.md)
- [Predicate、scan 与 coordinate summary](kernels/predicate-reduction.md)
- [Online softmax](kernels/online-softmax.md)
- [Blocked GEMM](kernels/gemm.md)
- [Extension primitive](kernels/extensions.md)
- [Packed quantization 与 irregular access](kernels/quantized-local.md)
- [Causal memory、Transpose、RoPE 与 online attention composition](kernels/attention-composition.md)

## 文档分工

```text
doc/
├── dsl/       source language 的构造与语义
├── compiler/  Kernel IR、target lowering、构建配置与产物
└── kernels/   canonical DSL construct 的组合示例
```

每个概念只有一个权威落点；本轮由上面的Core-local blocked文档冻结根语义，其他文件只说明
对应模块怎样兑现它。其余工作报告仍然只是一次性快照。
