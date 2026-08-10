# Weft

Weft 是一门面向 RISC-V worker/hart 的 AOT kernel DSL 与编译器。它以 worker-local 标量
控制、一维 VLA iteration region、logical block、state algebra 和可组合的 primitive
realization provider 为核心；它不是计算图编译器，也不把 GPU grid/SIMT 模型搬到 RISC-V。

## 当前状态

仓库已经完成净室重建的第一里程碑：

```text
Python @weft.kernel source
  -> canonical weft_kernel / weft_ext MLIR
  -> registered MLIR parse + print + verify
```

活动根包含新的 Python reference frontend、canonical Kernel/Extension dialect、独立
`weft-opt`，以及按 language/upstream/operator/variant 组织的
[`source/`](source/README.md) 语料库。旧编译器、历史实验与旧 artifacts 仍被硬隔离在
[`materials/`](materials/README.md)，不进入 CMake、Python import、链接或运行时。

这一里程碑只完成语言与 canonical IR 闭环。仓库目前没有 Selected Execution IR、target
profile、Scalar/RVV/IME provider、RISC-V source/object/header artifact 或 IntentDSL bridge；
因此不能把当前结果描述成已经可执行的 RISC-V kernel compiler。

## 构建与最小使用

当前开发环境使用 LLVM/MLIR 20：

```bash
cmake -S . -B build \
  -DMLIR_DIR=/usr/lib/llvm-20/lib/cmake/mlir \
  -DLLVM_DIR=/usr/lib/llvm-20/lib/cmake/llvm
cmake --build build --target weft-opt
```

将一个 Python kernel 降到 canonical MLIR，并由独立方言解析与验证：

```bash
PYTHONPATH=python python3 -m weft \
  source/weft/weft/elementwise/add_bias/add_bias.py \
  | build/tools/weft-opt/weft-opt
```

`python -m weft SOURCE --kernel NAME` 可在一个 source 文件含多个 kernel 时选择入口。
Kernel definition 是 AOT source object，不能作为普通 Python 函数 eager 调用。

十项固定 baseline、相邻 C runtime 与两条手工 repro 入口见
[`source/README.md`](source/README.md)。这些 reference source 不进入默认 CMake，也不表示
Selected IR 或 RISC-V artifact 已经实现。

## 阅读顺序

1. [`doc/WEFT_FINAL_SPEC.md`](doc/WEFT_FINAL_SPEC.md)：语言与完整编译器的唯一规范性设计。
2. [`doc/CURRENT_STATE.md`](doc/CURRENT_STATE.md)：已实现、已验证和尚未实现的精确边界。
3. [`doc/PYTHON_DSL.md`](doc/PYTHON_DSL.md)：第一里程碑 source surface、canonical schema
   与验收记录。
4. [`doc/ARCHITECTURE.md`](doc/ARCHITECTURE.md)：worker-local、VLA、block、state 与三层表示。
5. [`materials/README.md`](materials/README.md)：donor 中可抽取与禁止复用的内容。
