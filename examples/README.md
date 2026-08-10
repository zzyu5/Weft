# Manual repro entrypoints

算法 source 位于 [`../source/`](../source/README.md)。`examples/` 不再保存第二份 kernel
source，只放能够手工重放当前边界的入口：

- `repro/canonical/run.sh`：Weft Python source → canonical MLIR → `weft-opt`；
- `repro/source/run_c_baselines.sh`：编译并运行十份相邻的 C reference/runtime。

这些入口是开发期的真实 repro，不是测试套件，也不进入默认 CMake target。
