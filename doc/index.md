# Weft 编程模型与 DSL 规范

Weft 是一门面向非 SIMT 机器的数值 realization 语言。作者写下一个确定的、层级化的有限位宽数值程序；目标编译器为这份程序决定物理表示、局部指令和执行形态。本目录是编程模型与 DSL 的规范性定义；目标 physical IR 与 pass 结构不在本部分定义。

本规范描述编程模型和 DSL：

1. [编程模型](programming-model.md)：问题、执行模型、根抽象、唯一职责判据，以及与 Triton、TileLang 的机制差异。
2. [Encoding、View 与 artifact](language/encoding-and-artifacts.md)：packed storage 的逻辑布局、内存对象和跨调用派生表示。
3. [Value、Level 与控制流](language/values-levels-and-control.md)：逻辑值、生命周期、domain、cohort、handoff 和普通有序控制。
4. [函数、基本 op 与 engine role](language/functions-ops-and-engines.md)：同语言标准库、显式逻辑轴、闭合 primitive、重载和目标抽象角色。
5. [完整示例](examples.md)：Q4_K vec-dot、blocked MUL_MAT、GEMV、online attention 和 Top-K。
6. [Canonical Kernel IR 边界](canonical-kernel-ir.md)：源程序必须保存什么，以及不得包含什么物理信息。
7. [不采用的模型](non-goals.md)：会破坏语言边界的替代设计及原因。

本文档不规定某个目标后端怎样组织 physical IR，也不记录实现进度、性能结果或迁移过程。
