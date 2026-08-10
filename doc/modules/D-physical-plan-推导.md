# 模块 D — Physical Plan 解析推导（核心研究贡献）

## 职责

对 canonical Weft blocked program 推导 selected RISC-V execution/layout：
block tensor 到 scalar/RVV/IME execution group 的映射、VLEN-agnostic
strip-mining、SEW/LMUL、memory mode、reduction/dot strategy、owner handoff、
canonical mask realization 和必要 scratch。

这是全新工作，不是旧架构任何现有代码的直接推广。

### Value-local layout 与多组 contract

`#weft_layout.blocked` 只保存一个 canonical block value 的局部轴遍历顺序和
SIMD 轴，不保存 shape、extent 或 scope-global axis id。selector 先根据点算、
broadcast、pointer 与 memory use 建立 value-layout 等价分量；`contract` 的
lhs/rhs/result 不被粗暴合并，它们的轴对应只从 canonical
`lhs_axes/rhs_axes` 重算。

彼此独立的等价分量也不再被 whole-scope topology gate 强迫由一个 contract 协调。
`examples/repro_mixed_axis_sums.py` 在同一 flat kernel 中放置两个不共享 block SSA 的
rank-2 reduction：第一个 use 要求 axis 1 进入 VLA，第二个 use 要求 axis 0 进入 VLA。
selector 分别生成 `order=[1,0], vector_axes=[1]` 与
`order=[0,1], vector_axes=[0]` 两个 group，emitter 在同一函数内顺序发射两个 scope。
这证明“kernel 属于 rowwise 还是 columnwise”已不是入口分类；带 rank-2 contract 的
scope 目前仍只支持其 site-local 三组，跨 component contract composition 尚未完成。

### Rank-N 逐轴角色与 owner-local 数值策略

普通 pointwise/memory scope 不再以 rank-1/rank-2 分类入口。selector 对每个 value-local
component 选择一个 vector axis，把它放到 traversal 最快端；`order` 中剩余轴都是有序
serial roles。RVV emitter 反向遍历这些 serial axes 生成从慢到快的嵌套循环，再对
vector axis 做 VLA strip-mining。selector 逐轴检查 canonical store unit-stride、load
affine legality 和 reduction role，不再默认最后一轴。`examples/repro_rank3_affine.py` 的
axis 1 是唯一 unit-stride store axis，`order=[1,2,0], vector_axes=[1]` 实际发射为
`axis0 → axis2 → axis1-VLA`；shape、stride
和每个轴的动态上界仍从 canonical SSA 重算。当前 rank>2 只开放 pointwise/memory；
reduction、contract、多 vector axis 和 register-tile factorization 仍 fail closed。

canonical reduction 的 `kind/init/axis/ordered` 只在 `weft_kernel.reduce`；owner record
只保存最终 strategy。RVV 已实现 f32 `sum/max/min`，其中 max/min 直接使用 ISA 的
`vfredmax/vfredmin`，不复制 kind 到 selected record。canonical vector `exp` 则有真实的
数值实现选择，因此新增：

```text
weft_rvv_execution.unary_config {
  source_node,
  group,
  strategy = "exp_poly_v1"
}
```

`exp_poly_v1` 是 owner-local 选择；多项式系数、exceptional-range merge 与 intrinsic
拼写属于 emitter 静态表，不进入 plan。寄存器压力模型把这段内部 working set 作为
transient 计入 lane-ratio/LMUL 选择。`examples/repro_softmax.py` 的三个 canonical
阶段因此分别得到 max reduction、exp+sum 和 exp+store 的独立 group/config，而不是
匹配一个 softmax kernel 模板。

当前 rank-2 f32 contract 的 selected 形状是：

```text
lhs group    #weft_layout.blocked<rank=2, order=[1,0], vector_axes=[]>
rhs group    #weft_layout.blocked<rank=2, order=[1,0], vector_axes=[1]>
result group #weft_layout.blocked<rank=2, order=[1,0], vector_axes=[1]>

weft_rvv_execution.contract_config {
  source_node,
  groups = [lhs_group, rhs_group, result_group],
  strategy = "sequential"
}
```

`groups` 只是对已选 layout group 的角色引用，不复制 M/N/K、shape 或
contract axes。rhs/result 的 lane ratio 在合并寄存器压力下联合选择；
serial lhs 不伪造 `lane_ratio`。这一严格切片已由 source emitter 实现为
`M serial → N VLA strip → K serial`。

第一条 IME owner 使用同一个 shared 层，但不把 IME fragment 假装成 RVV vector
axis：

```text
lhs group    #weft_layout.blocked<rank=2, order=[1,0], vector_axes=[]>
rhs_t group  #weft_layout.blocked<rank=2, order=[1,0], vector_axes=[]>
result group #weft_layout.blocked<rank=2, order=[1,0], vector_axes=[]>

weft_ime_execution.contract_config @ime_c0 {
  source_node,
  groups = [lhs_group, rhs_t_group, result_group],
  mac_m = 4, mac_n = 4, mac_k = 8,
  vlen_bits = 256,
  strategy = "fragment_tiled"
}
```

同一个 symbol 共同拥有三个 group，因为 `vmadot` fragment 是不可拆开的物理操作；
shared `weft_execution.group.owner` 因而允许 owner record 以非空 `groups` 回指多个
group。所有 owner-local selected record 必须实现 `SelectedOwnerOpInterface`；shared
verifier 不再因某个任意 operation 碰巧带有 `group/groups` 属性就接纳它。
`si8/si32`、signedness、contract axes、logical M/N/K、pointer/mask 与 instruction
spelling 均从 canonical IR 加静态 ISA 表重算，不在 config 中复制。当前 selector 只在
target 携带精确 `xsmtvdotii` extension token、显式 VLEN=256，且 canonical M/N/K
extent 可解析为正、角色一致的整数时选择；不根据 constexpr 参数名称解释 M/N/K，
这些角色仍由参数在 `arange`/contract SSA 中的使用决定。逻辑 extent 可以大于物理
fragment 或不能整除它；`4×4×8` 只描述已选 hardware leaf。

这个切片的 tail 不是 IME mask：owner 按 canonical load mask 与逻辑块边界把每个
K fragment 的有效元素 pack 到零初始化 scratch；物理 helper 只从 config 读取的
`mac_m/mac_n/mac_k` 推导循环与 storage，并在 v2/v3 中跨 `ceil(K/8)` fragment 累加，
最后按 canonical store mask scatter。每个 canonical contract 都独立形成一个 site；
多个 site 使用各自的三组 layout group 和 owner symbol，selector 以所有 site 的 producer
closure 并集证明没有忽略算法工作。当前仍只支持这个 signed widening leaf，
IME→RVV/Scalar cleanup 与更多 fragment/layout family 尚未实现。

block-carried GEMM 在此基础上增加一个外层 result group 和循环体的三个 contract
role group。canonical For 的四个状态端点不是四份物理值；selected plan 使用两条
按-use handoff 连接它们：

```text
outer init/result group
  -- For init operand --> body result-role group
  -- Yield operand    --> outer loop-result group

body contract groups = [serial lhs, N-vector rhs, N-vector carried result]
```

handoff 两侧必须使用相同 blocked layout 与 lane ratio。selector 分别在外层 loop
point 和内层 contract point 计算活跃寄存器，再选择一个共同 ratio；emitter 将其实现
为同一 RVV accumulator 的作用域穿越，不复制或重排数据。K0 的 lower/upper/step 与
每次 BK contraction extent 都继续从 canonical For/Contract 重算。

当前 shared selected IR 已有最小的按-use转换记录：

```text
weft_execution.layout_conversion {
  consumer_node,
  consumer_operand,
  group = target_group
}
```

source value 由 canonical consumer operand 唯一得到，source group 由
`value_layout` 反查，source/target layout 由两个 group 得到；因此 record 不重复保存
producer node/result、layout 或 conversion kind。verifier 已检查 canonical use、native
source group、target rank/scope、重复 use 与空转换。当前有三种真实 realization：

1. For/Yield 的 layout-preserving carried handoff，两侧共享 layout/lane ratio，作为
   同一 accumulator 的 alias；
2. 普通 reduction operand 的 splat rematerialization。同一个 canonical splat 若被
   两个不同逻辑轴归约，native use 保持原 group，另一个 use 进入 incoming-only
   target group。`getVectorFastestBlockedLayout` 保证被选 vector axis 位于 traversal
   最快端；RVV owner 从原 scalar 重新 splat 到 target lane domain；
3. 普通 reduction operand 的 masked-load rematerialization。source 必须是 direct
   canonical load；selector 在 target axis 上重新验证整个 pointer/mask producer closure
   的 affine legality，并要求 load 与该 consumer 之间没有 store。emitter 从 canonical
   pointer/mask 重新发出目标 layout 的 `vle`/`vlse`，而不是搬运 native 寄存器值。

后两种都不是寄存器 transpose。load 路径确实重新读取 canonical memory，因此一旦中间
存在 store（包括中间 operation 的嵌套 region 内 store）且没有 alias proof就
fail closed；rematerialized producer closure 的 pointer/index/mask/load 临时值也进入
target group 的瞬态寄存器压力，但不生成第二份 `value_layout`。任意非-load value 的 vector-axis
permutation、寄存器数据重排和跨 Scalar/RVV/IME owner handoff 仍明确拒绝。
`isIndexCoordinateProvenance`、logical extent
和 projection-to-store 结构闭包共同约束 target group 的坐标/尾部，不能只凭两个
动态 extent 的数值相等把无关 block 当成同一 projection。

### Affine lane stride

`#weft_layout.blocked.order` 的最快轴不再被写死为 rank-2 axis 1。selector 根据 reduction
axis 选择单个 vector axis，再由 `getVectorFastestBlockedLayout` 构造 traversal order。
pointer legality 沿 canonical arange/expand/index binary 图重算 lane stride：lane-invariant
项、unit arange、add，以及“一个 lane-affine 因子乘一个 lane-invariant scalar”保持
affine；两个 lane-varying 因子相乘、lane-dependent subtraction、负 constant multiplier
均 fail closed。

RVV emitter 保存 pointer 当前 chunk 的 base expression 与 element stride。stride 为 1
发射 masked `vle`；其他已证明非负 affine stride 转成
`ptrdiff_t(element_stride) * sizeof(float)` 后发射 masked `vlse`。当前 f32-only 路径已在
`examples/repro_columnwise_sum.py` 上以 runtime `stride=77` 实机验证。vector store 仍只
接受 unit stride，尚未把旧代码中的 `vsse`/indexed/segment 机制接入新 owner contract。

### Meta 参数选择

canonical Kernel IR 只保留 `!weft_kernel.constexpr<index>` 参数及其 `meta_value` SSA
用途，不在算法层写死块尺寸。物理选择用 canonical `arg_names` 寻址输入的
`NAME=VALUE`，随后立即转成 selected plan 内按 entry argument index 记录的
`weft_execution.meta_binding {argument, value}`。参数名称不是角色分类：selector 不会
根据 `BM`、`BN` 或 `BK` 的拼写选择 GEMM 分支，extent/loop-step 的意义仍由它们在
canonical SSA 中连接到哪个 `arange`/`range` operand 决定。

`weft-compile --meta BM=8 --meta BN=32 --meta BK=16` 表示独立选择；
`--block-elements=32` 仅表示用户显式要求所有 constexpr 统一为 32。两种模式互斥，
不存在默认块尺寸；缺项、重复项、未知名称和非正值都会在 plan 物化前失败。因此
canonical 算法事实与 selected 物理决策仍各有唯一归属。

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
- 可以对 Weft source 声明的 meta-parameters 和合法 execution layouts 做
  bounded/offline tuning；搜索不能创造算法结构、跳过 legality 或改变 kernel
  的 pointer/index/mask/control 语义。

## 待裁问题

- block layout、RVV register-group 约束和 IME tile 约束共享到什么边界；
- 如何验证"推导出来的 Plan 确实是解析构造的，不是暗中退化为人工修正表"
  ——需要独立于设计意图的验证手段（例如：新增一个此前未覆盖的硬件能力
  组合时，若推导结果错误，必须能定位到"公式未建模某个约束"而不是
  "缺少一条候选分支"）。
