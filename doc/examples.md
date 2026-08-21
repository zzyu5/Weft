# 前端 examples

`examples/kernels/` 只保留手工复现入口；算法源码位于 `python/weft/std/`，避免保存第二份实现。

| 入口 | 同语言源码 | 主要约束 |
|---|---|---|
| `quantization/q4_k_gemv.py` | `std/encodings.py`、`std/quant.py` | grouped/layered q layout、joined K-scale layout、derive、三层位宽、min 支路、`ds` 的 block 层位置、嵌套 handoff |
| `dense/gemm.py` | `std/dense.py::gemm` | NC/KC/MC/MR/NR/KB 层、staged panel、KC-scope accumulator |
| `dense/gemv.py` | `std/dense.py::gemv` | GEMM 的一维退化，使用普通 `contract` 而非新原语 |
| `attention/flash_attention.py` | `std/attention.py` | staged query、三个 state、非归约 handoff、顺序性来自 use-def |
| `selection/top_k.py` | `std/selection.py` | 无 Level 的普通有序 `for/while/if` 与 Value 更新 |

每个入口可用下面同一条前端复现方式生成并验证 canonical Kernel IR：

```bash
PYTHONPATH=python python -m weft examples/kernels/dense/gemm.py > /tmp/gemm.mlir
build/tools/weft-opt/weft-opt /tmp/gemm.mlir -o /dev/null
```

这条命令验证到 canonical IR 边界，不声称已经经过本轮明确禁止修改的 RISC-V backend。
