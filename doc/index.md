# Weft 编程模型、DSL、物理机器与编译器规范

Weft 是一门面向非 SIMT 机器的数值 realization 语言。作者写下一个确定的、层级化的有限位宽数值程序；目标编译器在非 SIMT 物理抽象机器上为这份程序决定表示、局部指令和执行形态。本目录定义编程模型、DSL、目标物理机器和两层 MLIR 编译主干。

本文件及其链接是项目当前唯一设计规范。源码说明当前实现事实；`report/` 只保存一次性工作结果、baseline 来源和性能数据，不能反向定义语言或编译器合同。

文件分工不等于 IR 层次。Weft 的程序 IR 始终只有 Canonical Kernel IR 与 target-aware physical IR 两层。

## `model/`

- [编程模型](model/programming-model.md)：问题、执行模型、根抽象、作者与编译器的唯一职责判据，以及与 Triton、TileLang 的机制差异。
- [非目标](model/non-goals.md)：不会进入 Weft 语言的替代根模型与隐藏权限。

## `dsl/`

- [Encoding、View 与 Artifact](dsl/encoding-and-artifacts.md)：packed storage、内存对象、派生布局和调用边界。
- [Value、Level 与有序控制](dsl/values-and-levels.md)：逻辑值、生命周期、domain、cohort、handoff 和普通控制。
- [函数、Operation 与编译配置](dsl/functions-and-operations.md)：标准库、`auto`、基本 operation、显式逻辑轴，以及 core DSL 与 target requirement 的边界。
- [DSL 示例](dsl/examples.md)：Q4_K vec-dot、blocked MUL_MAT、GEMV、online attention 和 Top-K。
- [Canonical Kernel IR](dsl/canonical-ir.md)：语言事实必须保存什么，以及不得包含什么物理信息。

## `machine/`

- [非 SIMT 物理抽象机器](machine/physical-machine.md)：逻辑值怎样沿时间、lane、寄存器副本、extension fragment 与局部存储形成物理表示，以及 target profile 必须提供什么。

## `compiler/`

- [编译器总览](compiler/index.md)：两层 MLIR、candidate authority 与完整文档入口。
- [RISC-V Physical IR](compiler/riscv-ir.md)：physical value、memory、conversion、Level/control 与 final invariants。
- [编译 Pass](compiler/passes.md)：layout、memory、schedule、resource 与 target-op rewrite。
- [RISC-V Local Leaf](compiler/leaves.md)：RVV intrinsic、IME/opaque asm 的合同与选择。
- [Intrinsic C / ASM Emission](compiler/emission.md)：terminal translator 与 system compiler 边界。

## `experiments/`

- [实验与 Baseline](experiments/index.md)：baseline怎样决定作者侧DSL实现、case identity与性能归因。
- [测量协议](experiments/protocol.md)：correctness、Clang、目标机、timing与CSV合同。

这些文件只描述设计，不记录实现进度、性能结果或迁移过程。
