# Weft

Weft 是一门面向 RISC-V worker/hart 的高性能 AOT kernel DSL 与编译器。它以
worker-local 标量控制、VLA iteration region、logical block、state algebra 和可组合的
primitive realization provider 为核心，生成 Scalar、RVV、IME 及未来扩展的 RISC-V
source/object/static-library/header artifact。

## 从哪里开始

1. [`WEFT_FINAL_SPEC.md`](WEFT_FINAL_SPEC.md)：语言和编译器的唯一规范性设计。
2. [`doc/README.md`](doc/README.md)：现行工程文档索引与权威关系。
3. [`doc/ARCHITECTURE.md`](doc/ARCHITECTURE.md)：从规范到组件边界的整体架构。
4. [`doc/REPOSITORY_RECONSTRUCTION.md`](doc/REPOSITORY_RECONSTRUCTION.md)：为什么选择
   在本仓库净室重建，以及旧代码如何成为只读材料。
5. [`doc/CURRENT_STATE.md`](doc/CURRENT_STATE.md)：当前源码与最终规范的真实差距。

## 当前状态警告

当前 `include/`、`lib/`、`python/` 和 `tools/` 仍主要实现被最终规范否决的
`grid/task + whole-kernel selector` 原型。它们是待抽取的 donor，不是最终架构，也不应
作为新实现的兼容入口。本文档重构没有移动或重写这些源码；下一次实现切换应按照仓库
重建决策一次性建立新的依赖边界。
