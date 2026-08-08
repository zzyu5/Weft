# 模块 D — Physical Plan 解析推导（核心研究贡献）

## 职责

对选中 owner 内的一个结构化原语实例，推导出具体的 Physical Plan：LMUL/
subtiling 选择、storage（哪部分数据驻留寄存器/scratch）、`vsetvl` 放置、
tail/边界处理等——对应 Intent 语言规格里故意留白、交给 Realizer 决定的
那一整块内容（`RegionBinding`/`Subtiling`/`WorkerMapping`/`Storage`/
`WorkingLayout`/`Pipeline`/`Boundary`/`Launch`）。

这是全新工作，不是旧架构任何现有代码的直接推广。

## 目标形态

不是"人工列出几个候选、公式在候选表里打勾"（数据重新排版，不算真正的
公式），而是尽量接近"给定硬件能力事实，真正解析/推导出合法且高质量的
tile/资源分配"，参考方向包括（均需独立验证，非既定结论）：

- 类 QIGen 的约束求解（寄存器/cache 容量作不等式约束，解出分块参数）；
- 类 Hidet task-mapping 的组合代数（`spatial`/`repeat` 等组合子 + 结合律
  描述任务分配层级）——但需要补齐 Hidet 完全没建模的 RVV 专属维度：
  运行时可变的 worker 数量（VLEN-agnostic）、worker 内部资源折叠（LMUL）；
- 类 CUTLASS/CuTe 的 layout algebra（`Layout=(Shape,Stride)` 及其代数
  运算）——但只覆盖仿射映射，K-quant 一类的 codebook/grid 非仿射查表需要
  独立机制。

## 明确的方法论边界

- 输出可以是"合法/更优的区间或方向"（偏序关系、约束边界），不必是"唯一
  确定的数值"——在离散硬件粒度（LMUL 只能是 mf2/m1/m2/m4/m8 等）上，公式
  收窄候选区间，最终对齐硬件粒度做选择，这不退化为"人工候选表"，因为
  区间本身是解析推导出来的，不是预先枚举的。
- 绝不是开放式在线搜索/cost-model autotuner；不能引入 Ansor/AutoTVM 式的
  "生成大参数网格 + 演化搜索"。

## 待裁问题

- 这套推导方法论对不同结构化原语种类（`contract` vs `reduce` vs
  `state_stream`）是否共享同一套数学框架，还是各自独立；
- 如何验证"推导出来的 Plan 确实是解析构造的，不是暗中退化为人工修正表"
  ——需要独立于设计意图的验证手段（例如：新增一个此前未覆盖的硬件能力
  组合时，若推导结果错误，必须能定位到"公式未建模某个约束"而不是
  "缺少一条候选分支"）。
