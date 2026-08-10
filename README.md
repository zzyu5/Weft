# Weft

Weft 是面向 RISC-V worker/hart 的 AOT kernel DSL 与编译器。它以 worker-local 标量控制、
一维 VLA iteration region、logical block、state algebra 和可组合的 primitive realization
provider 为核心；它不是计算图编译器，也不把 GPU grid/SIMT 模型搬到 RISC-V。

## 构建

```bash
cmake -S . -B build -DCMAKE_PREFIX_PATH=/path/to/llvm
cmake --build build --target weft-opt
```

将一个 Python kernel 降到 canonical MLIR：

```bash
PYTHONPATH=python python3 -m weft \
  source/weft/weft/elementwise/add_bias/add_bias.py \
  | build/tools/weft-opt/weft-opt
```

`python -m weft SOURCE --kernel NAME` 用于显式选择 source 中的 kernel。Kernel definition
是 AOT source object，不能作为普通 Python 函数 eager 调用。

## 入口

- [`doc/WEFT_FINAL_SPEC.md`](doc/WEFT_FINAL_SPEC.md)：唯一语言与编译器规范。
- [`source/README.md`](source/README.md)：固定 operator corpus 与 source 布局。
- [`examples/README.md`](examples/README.md)：手工 repro 入口。
- [`materials/README.md`](materials/README.md)：隔离的历史代码与硬件知识供体。
