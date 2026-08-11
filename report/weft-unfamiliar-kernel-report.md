# 十个陌生 Kernel 与 Weft 外推能力

这份报告记录第三组十个 kernel 的当前实现、架构变化与真机读数。十项 workload 在检查
当前 lowering 能力之前先确定，目的不是扩大已有 family 的 case 数量，而是观察一份自然
worker-local 算法能否依靠已有语义 anchor 组合出 RISC-V realization。

固定的 `report/baseline/` 没有修改。Weft 数字继续集中记录在
[`weft-kernel-performance.csv`](weft-kernel-performance.csv)；本文中的 GGML 数字来自本轮
新增的 public-op 单 kernel repro，只作为同机对照，不反向成为 compiler 输入。

## 结论

十项都已通过唯一 production 主链在 SG2044 上生成并执行：

```text
natural Weft Python DSL
→ canonical worker-local Kernel IR
→ RISC-V primitive-local physical decisions
→ intrinsic C
→ GCC 15 system compiler
→ single-hart SG2044 executable
```

本轮没有新增 kernel name、operator name、source op-count 或 whole-region shape 分支，也没有
引入 normalization pass。新增能力落在 VLA memory/predicate、ordered scalar state、scalar
coordinate math 与 local contract projection 上；作者写下的 traversal、state、grouping、
window 与 layout保持在 DSL/Kernel IR 中。

从结构上看，Weft 已经表现出初步外推能力：十个事后选择的算法都没有被改写成后端模板，
两个等价 source 也获得相同类别的 physical realization。从物理选择和性能看，结论仍是
部分成立而不是“Triton级通用性已经证明”：很多新路径只有一个固定合法配置，top-k仍比
GGML慢3.12倍，state tiling、window reuse、tap unroll和更完整的LMUL/microtile候选空间尚未
成为真实选择维度。

## 先确定的十个 workload

| Kernel | 真实结构压力 | 自然 DSL / Kernel IR 所有的算法结构 | 使用的局部 lowering anchor |
|---|---|---|---|
| MoE router Top-K | selection、重复summary、coordinate tie-break | row/rank traversal、此前已选集合、`summary_fold(value,index)` | VLA predicate、tuple summary、max/equal/first-lane realization |
| Mamba short convolution | VLA与短有序tap loop组合 | sequence/channel traversal、token VLA、tap loop和strides | nested scalar loop中的VLA access decision、unit/strided RVV memory |
| Mamba2 SSM scan | ordered recurrence、grouped parameters、special math | token/head/dimension/state traversal、softplus、decay与state update | scalar `log/exp`、ordered memory state、dimension VLA |
| RWKV-WKV6 | matrix state recurrence | token/head/row/column顺序、state equation和独立final state | scalar carried traversal与column VLA load/store |
| ADD_ID | indexed expert row与VLA pointwise组合 | token/slot traversal、显式expert index、hidden VLA | scalar indexed base、unit-stride RVV add/store |
| GET_ROWS_BACK | duplicate-aware scatter-add | output zero-fill、source顺序、duplicate index累加 | scalar indexed base、VLA load/add/store；不猜冲突处理 |
| MUL_MAT_ID | source-owned grouping加indexed contraction | expert count/offset/cursor/items、expert traversal与local contract | grouping仍是source control；contract只选择row microtile realization |
| Depthwise Conv2D | padding predicate、window、channel VLA | NHWC/C-fast layout、spatial/window loops、stride/pad/dilation | scalar coordinate predicate、channel RVV memory/FMA |
| MaxPool2D | window max与width VLA | NCHW/WHCN layout、channel/row/window traversal | width VLA load/max/store |
| Bilinear upscale | floor/clamp、四点indexed load、channel VLA | half-pixel坐标、四邻域、C-fast input/output layout | 显式scalar `floor`/min/max与channel RVV arithmetic |

这些 source 不是十个新 operator op。Top-K 沿用 `summary_fold`，两个 recurrence 使用普通有序
control与memory state，vision kernels使用普通coordinate/predicate/VLA，`MUL_MAT_ID` 的
expert grouping也由作者显式写出。DSL只补了算法确实需要的scalar `floor` 语义，没有增加
`top_k_kernel`、`ssm_kernel`、`vision_kernel`或`mul_mat_id_kernel`一类入口分类。

## Lowering 重构

### VLA 决策穿过作者的 scalar control

VLA region 的 physical scan 现在递归进入其中的普通 scalar `for`，分别为每个compare、
load、store和narrow产生decision。这样短卷积的tap loop、Top-K的previous-rank loop以及
vision window中的局部访问可以保留原始控制结构，而不需要把region改写成标准形状。

禁止的仍是第二个active VLA axis。普通有序scalar loop可以嵌在VLA中；其effect与顺序保持
source-visible。Emitter只消费每个access已经选定的lane relation、unit/strided mode、
predicate realization和LMUL，不再根据周围op数量重建一次memory策略。

### Ordered state 没有被伪装成 graph pattern

SSM scan 与RWKV recurrence都以作者写下的token/state顺序执行。Target没有从multiply/add
use graph发现“Mamba”或“RWKV”，也没有自动创建scan primitive。当前 realization把适合的
dimension/column域映射到RVV，把真正有顺序依赖的state loop保留为scalar control；这是一条
合法组合路径，但还没有state register tiling、persistent placement或近似math policy的候选
选择。

### Local F32 contract 不再要求 enclosing-loop closure

`ContractDecision` 现在保存row axis、reduction axis/extent、typed producer/consumer、row
microtile与LMUL。Emitter从这些已选事实把block pointer与predicate投影到intrinsic C。
`MUL_MAT_ID` 因而可以把同一个local contract放在expert→row→item的新上下文中；target不再
要求固定的row-loop/column-loop parent关系，也不负责发明expert grouping。

这仍是局部 `[BM,K] × [K]` F32 family，而不是任意rank contract已经完成。当前row tile上限
6、LMUL4和accumulator组织只有一个实现点，尚未形成有意义的候选空间。

### System compiler 边界保持不变

Weft只决定dynamic `vl` placement、LMUL、memory form、local microtile和intrinsic family，
输出普通 intrinsic C。最终register allocation、machine scheduling、peephole、constant fold
与instruction encoding继续由GCC完成；本轮没有增加LLVM-backend式机器级IR或调度层。

## 等价 source 敏感性

只对两个适合的算法写了自然等价表达；没有增加程序规范化pass。

| Kernel | 等价变化 | Primary | Equivalent | Physical realization |
|---|---|---:|---:|---|
| Top-K | pointer base hoist与predicate operand书写次序 | 3.184054 ms | 3.131253 ms | intrinsic集合与计数一致 |
| SSM convolution | `(sequence,channel)` 两层坐标与线性position解码互换 | 2.747092 ms | 2.768892 ms | intrinsic集合与计数一致 |

两个版本都选择相同的VLA memory、LMUL和summary类别，没有因为source closure不同而切换fast
path。这里不要求二进制或时间完全相同：scalar coordinate表达仍由system compiler处理，
而中位数也保留普通运行波动。判据只是target decision来自axis/use-def/memory/state事实，而
不是完整source文本形状。

## SG2044 真机性能

Weft 与 GGML 都固定单线程/单 hart、RVV VLEN128和同一逻辑shape。每个样本前执行64 MiB
cache eviction并取中位数。Weft计时generated kernel entry；GGML计时单op graph的
`graph_compute + synchronize`。因此数字反映两条实际可调用路径，不把微小graph dispatch
成本伪装成intrinsic本体差异。

| Kernel | Shape | Weft ms | GGML public op ms | Weft / GGML | 可比边界 |
|---|---|---:|---:|---:|---|
| Top-K | `tokens=512,experts=256,k=8` | 3.184054 | 1.021364 | 3.117 | 同一top-k集合；GGML不保证rank顺序 |
| SSM convolution | `T=128,C=2048,taps=4` | 2.747092 | 5.491962 | 0.500 | math、shape、layout相同 |
| SSM scan | `T=128,H=32,D=64,state=16` | 4.845841 | 9.114867 | 0.532 | math/layout相同；GGML拼接output/state ABI |
| RWKV-WKV6 | `T=128,H=32,S=64` | 16.281330 | 16.707418 | 0.974 | math/layout相同；GGML拼接output/state ABI |
| ADD_ID | `tokens=128,slots=8,hidden=4096` | 6.572648 | 6.746318 | 0.974 | math、shape、layout相同 |
| GET_ROWS_BACK | `rows=32000,items=128,hidden=2048` | 86.614495 | 89.849307 | 0.964 | duplicate scatter与zero-fill相同 |
| MUL_MAT_ID F32 | `tokens=64,slots=2,N=4096,K=4096` | 939.236930 | 1134.325136 | 0.828 | grouping与full contraction均在计时内 |
| MaxPool2D | `N=1,C=64,H=W=112,K=S=2` | 1.795128 | 9.226990 | 0.195 | math、shape、WHCN layout相同 |
| Depthwise Conv2D | `N=1,H=W=112,C=32,K=3` | 8.797379 | 113.517962 | 0.077 | 只作语义参照：Weft C-fast，GGML WHCN |
| Bilinear upscale | `128²×64 → 256²×64` | 6.557349 | 84.307306 | 0.078 | 只作语义参照：Weft C-fast，GGML WHCN output |

严格或近严格可比的前八项中，Weft有七项更快，Top-K慢211.7%。SSM convolution与scan分别
快50.0%和46.8%；RWKV、ADD_ID与GET_ROWS_BACK只领先2.6%、2.6%和3.6%，属于同档；F32
`MUL_MAT_ID` 快17.2%；MaxPool快80.5%。Depthwise与bilinear的约13倍时间差混有native
layout和被向量化轴的差异，不能作为严格speedup。

### GGML 对照实际走了什么

| GGML op | SG2044 CPU path |
|---|---|
| Top-K | C++ `partial_sort`，无手写RVV |
| SSM convolution | scalar accumulation |
| SSM scan | RVV分支仍为TODO，实际走scalar state loop |
| RWKV-WKV6 | RVV构建走scalar path；手写SIMD只覆盖x86/Arm |
| ADD_ID / GET_ROWS_BACK | `ggml_vec_add_f32`，RVV构建下该helper仍是scalar loop |
| MUL_MAT_ID F32 | `ggml_vec_dot_f32` 的真实RVV intrinsic路径 |
| Depthwise direct WHCN | scalar direct convolution；另一条C-fast path才有SIMD macro |
| MaxPool2D / bilinear | scalar window/interpolation loops |

所以这组数字证明的是：Weft可以从陌生source组合出有效RVV realization，并在唯一一项真实
RVV对照 `MUL_MAT_ID` 上取得17.2%的本次读数优势。它不能证明已经全面超过成熟手写RVV
库，因为另外九个GGML public op并不是手写RVV基线。

## 当前仍存在的结构与性能缺口

- Top-K每个rank重新遍历完整expert域，没有candidate reuse、rank并行或selection-specific
  register organization；这是唯一严格对照中明显落后的项。
- SSM convolution没有tap unroll/load reuse decision；SSM scan与RWKV没有state tiling、
  persistent register placement或math approximation policy。
- ADD_ID与GET_ROWS_BACK只有基础indexed-base加unit-stride VLA路径，没有alias-aware scheduling、
  prefetch或duplicate contention physical choice。
- `MUL_MAT_ID` 的row tile、LMUL和register reuse仍是固定local family；source grouping正确但
  target还不能在多个microtile/packing方案间选择。
- Vision kernels没有spatial/channel tile选择、sliding-window reuse或FMA contraction；scalar
  edge branch和coordinate hoist也没有候选decision。
- VLA内nested scalar loop的memory/predicate可以被physical scan看到，但reduce/scan/summary
  decision尚未对所有nested上下文递归；第二个active VLA axis仍明确unsupported。
- 较早的F16 conversion/fill/dot/update/normalize、online softmax、F16 GEMM和affine Q4_K IME
  仍依赖较精确的exact closure。本轮只移除了local F32 contract的enclosing-loop closure，
  没有宣称整个`RISCVLowering`已经closure-free。

最终判断是：**selection bias 被实质削弱，Weft 已能让十个事后选择的自然worker-local程序
通过逐实体语义事实进入同一条RISC-V主链；但当前物理候选空间仍窄，且成熟RVV公共对照只
覆盖一项。因此它已经开始具备算子编译器的外推性质，尚未完成“Triton for RISC-V”的性能
自由度与广泛证据。**
