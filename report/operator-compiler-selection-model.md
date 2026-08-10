# 算子编译器 Selection 模型报告

## 结论

Weft 的 selection 单位已经在规范中明确为 canonical anchor + logical axis/relation + typed
operands + target facts，而不是 kernel 类别。RISC-V worker-local 模型没有 GPU program
space；逐轴选择决定 scalar、VLA/RVV、register repeat、microtile、fragment、memory 与
predicate realization。

## 关键边界

- 一次输入是一份完整 worker-local kernel，不是计算图或等待识别的算子 graph；
- kernel 名、算子名、量化格式名和整体 shape 不进入 selection；
- 多个 state、irregular relation 和 contract 可以在同一个 kernel 中自然组合；
- Scalar 是显式 provider candidate，不是 fallback；
- 持久表示只有 canonical、selected 和 artifacts；
- emitter 只投影 canonical + selected + target facts，不重新推断算法或物理计划。

## 实现状态

本轮只固定设计规范，没有新增 selection/provider/emitter 实现，也没有产生新的 RISC-V
artifact。后续代码若引入 whole-kernel classifier、format route 或 catch-all emitter，应按
规范直接视为架构回退。
