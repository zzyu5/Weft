# 模块 A — Front-end 契约（消费 Intent Kernel IR）

## 职责

定义 Weft 从 Intent kernel source 实际消费什么，作为新的 canonical problem
表示，取代旧架构里 9 个手写 TableGen problem op（`Int8MACProblemOp` 等，
已归档）的封闭问题目录模式。

Intent kernel 的内容分两类，Weft 必须区分对待：

- **纯值表达式**（算术、位运算、`I.exp` 等）——不是结构锚点，编译器可以
  自由做 CSE、融合、rematerialize、向量化；不需要为每种表达式模式注册
  专门的问题类型。
- **结构化原语**（`reduce`/`scan`/`contract`/`state_stream`/`group_by`/
  `gather`/`scatter`/`sort`/`topk`/`histogram`）——算法锚点，Realizer 不能
  把它换成另一种算法，但必须为它给出真正的物理实现策略。这是模块 C/D
  真正要处理的对象。

## 输入契约（来自 Intent 侧，非 Weft 自定义）

- `I.domain`/`I.partition`（`extent=`/`count=`，可观察性由是否进入
  host-visible shape 决定，不由参数名决定）；
- `I.parallel`/`I.ordered` 声明的顺序语义；
- 结构化原语列表（见上）及其各自固定的逻辑关系（如 `reduce` 的
  axis/identity/combiner，`contract` 的 operand/indexing/reduce 轴）；
- `I.Constexpr`（用户 specialization，可控制算法分支）与 `I.auto(...)`
  （compiler-owned realization hole，不可被读取/分支/逃逸为可观察值）
  的严格区分——这条区分必须端到端保持，见模块 E。

## 与旧架构的关系

旧的 9 个手写 problem op 是"预先登记的封闭问题目录"，本质上是 K-quant 等
具体格式在 IR 层的固化。新契约要处理的是开放组合空间：任意合法的结构化
原语嵌套 + 控制流 + effects，都应该可以被消化，不要求预先为每种组合注册
新 op。

## 待裁问题

- Weft 内部是否需要一个独立于 Intent 语言本身的 MLIR dialect 层来承载这份
  IR，还是直接消费 Intent 编译器产出的某种既有中间表示？（取决于 intentdsl
  项目自身的实现进度和交接形态，目前未知。）
- 纯值表达式部分的"更好向量化决策"（例如识别位打包/仿射折算这类惯用
  模式）算不算 Weft 的研究范围，还是完全交给通用向量化 pass？
