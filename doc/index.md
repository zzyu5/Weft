# Weft 设计文档

这组文档是 Weft 当前唯一的语言与编译器设计规范。它描述最终语义和模块边界，不记录
实现进度、历史版本、测试清单或迁移过程。

它定义 Weft 最终成品的目标状态；与旧设计、历史材料或实现注释冲突时，以这里为准。

Weft 是一门面向单个 RISC-V worker/hart 的 AOT kernel DSL：作者写完整的 worker-local
算法，编译器补全 source 不可观察的 VLA、寄存器、指令与扩展 realization。

## 阅读顺序

1. [语言定位与程序模型](dsl/model.md)
2. [Python eDSL 与控制流](dsl/python.md)
3. [VLA、predicate、memory 与 logical block](dsl/vla-memory-and-blocks.md)
4. [Contraction 与 state algebra](dsl/structured-compute.md)
5. [数值语义](dsl/numerics.md)
6. [Python DSL API 索引](dsl/api.md)
7. [编译器总架构](compiler/architecture.md)
8. [RISC-V target 与 primitive provider](compiler/target-and-providers.md)
9. [构建期 tuning](compiler/tuning.md)
10. [Canonical Kernel IR](compiler/kernel-ir.md)
11. [Selected Execution IR](compiler/selected-ir.md)
12. [Lowering 与 artifact](compiler/lowering-and-artifacts.md)
13. [Verifier、作者义务与禁止退化](compiler/verification.md)

典型 kernel 的 canonical 写法单独放在 `kernels/`：

- [Elementwise](kernels/elementwise.md)
- [RMSNorm](kernels/rms-norm.md)
- [Predicate reduction](kernels/predicate-reduction.md)
- [Online softmax](kernels/online-softmax.md)
- [Blocked GEMM](kernels/gemm.md)
- [Extension primitive](kernels/extensions.md)

## 文档分工

```text
doc/
├── dsl/       source language 的构造与语义
├── compiler/  编译器模块、IR、selection、lowering 与产物
└── kernels/   按 kernel 类型组织的 canonical DSL 模板
```

每个概念只有一个权威落点，其他文档只引用该定义，不复制第二套规则。实现中发现的问题、
阶段结果和阻塞写入顶层 `report/`，不得反向成为设计规范。
