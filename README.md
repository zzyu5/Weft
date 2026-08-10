# Weft

Weft 是一门面向 RISC-V worker/hart 的高性能 AOT kernel DSL 与编译器。它以
worker-local 标量控制、VLA iteration region、logical block、state algebra 和可组合的
primitive realization provider 为核心，生成 Scalar、RVV、IME 及未来扩展的 RISC-V
source/object/static-library/header artifact。

## 当前仓库状态

仓库已经完成重建边界切换：旧编译器、旧 examples、历史实验和旧 artifacts 全部隔离在
[`materials/`](materials/README.md)。根目录没有活动的旧源码，也没有兼容入口；新的
`CMakeLists.txt` 可以独立配置，但在 canonical IR 合同落地前故意不定义编译 target。

这意味着当前仓库是“准备开始新实现”的干净骨架，不是一个已经可用的 compiler。

## 阅读顺序

1. [`doc/WEFT_FINAL_SPEC.md`](doc/WEFT_FINAL_SPEC.md)：语言和编译器的唯一规范性设计。
2. [`doc/README.md`](doc/README.md)：现行工程文档索引与权威关系。
3. [`doc/ARCHITECTURE.md`](doc/ARCHITECTURE.md)：worker-local、VLA、block、state 与三层表示。
4. [`doc/REPOSITORY_RECONSTRUCTION.md`](doc/REPOSITORY_RECONSTRUCTION.md)：已经完成的仓库
   隔离、当前根结构与下一步实现顺序。
5. [`materials/README.md`](materials/README.md)：旧代码里哪些内容可抽取、哪些语义禁止复用。

## 第一里程碑

第一步直接完成完整 Python reference frontend，并同步完成其 canonical Kernel IR/types/
verifier。详细边界见 [`doc/PYTHON_DSL.md`](doc/PYTHON_DSL.md)。

```text
complete Python DSL source
  -> canonical Kernel MLIR
  -> independent parse / print / verify
```

这一里程碑不做 provider/backend。语言闭环后，第二里程碑才建立 Scalar + RVV 的最小
可执行纵向链。新代码只能在根目录重新建立，不能调用 `materials/legacy-source/`。
