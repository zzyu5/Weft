# 语言定位与程序模型

## 最终定义

Weft 是一门面向**单个 RISC-V worker / hart** 的高性能 kernel DSL 与 AOT 编译器。它的根
程序是一份在一次 invocation 中持续执行的 core-local blocked program：一个 worker 顺序处理
一个或多个 block，并让 SSA accumulator、state、显式 staging 与 workspace 跨 loop 存活、更新
和复用。

作者使用普通控制流、一等 VLA iteration region、显式指针与逻辑 predicate、局部 logical block value、结构化 state algebra，以及 dot / matmul / scan / lookup / decode 等 primitive 编写完整的 worker-local kernel。编译器把这些结构映射到动态 `vl`、LMUL、寄存器组织、memory instruction、register microtile、矩阵 fragment 和 RISC-V 扩展指令。

多核线程创建、工作切分、线程池、OpenMP、affinity 与 NUMA 均由外部 runtime 负责，不属于 Weft core。

Weft 的扩展原则是：

> **已有语义扩展局部 target realization；新增可观察语义增加一个局部 primitive；永不为某个完整算子、模型格式或 kernel 名增加整段后端模板。**

Weft 不是 Tensor graph compiler，也不是 Triton CPU backend 的重写。它与 Triton 的关键区别
不是有没有 tile，而是 block 的 owner、control contract 与 lifetime：

- Weft 没有 program grid、`program_id` 或隐式 launch identity；
- Weft 的入口是一个由当前 worker 持续执行的普通 callable kernel，而不是 grid 中的一次性
  program instance；
- VLA region 的 strip 边界与 `vl` 在 source 中不可观察；
- 跨 strip 的 reduce、scan、summary state 是一等语义；
- logical block 是普通 SSA value，由 worker control program 持有，可以跨 loop carry、多 use、
  pointwise、state、memory 与 structured primitive 自然组合；
- caller-provided external/persistent/workspace 与 compiler-private primitive temporary 具有互不
  混淆的 ownership/lifetime；
- RISC-V 的 V、矩阵、量化、重排及 vendor extension 在一次 primitive-local target lowering 中组合接入。


## 规范用词

本文件使用以下约束词：

- **必须 / MUST**：符合 Weft 的实现不可违反；
- **禁止 / MUST NOT**：符合 Weft 的实现不可提供该行为；
- **应该 / SHOULD**：除非有明确且可说明的原因，否则应遵守；
- **可以 / MAY**：可选能力，不影响核心语义。


## 设计目标与非目标

### 设计目标

Weft 必须同时满足：

1. **worker-local kernel 可写性**：作者能直接表达循环、blocking、staging、pointer/index、mask、state 与 structured compute；
2. **VLA 原生性**：source 不观察固定 VLEN、exact `vl` 或 hardware lane count；
3. **高性能物理自由度**：target lowering 能决定 LMUL、register microtile、unroll、packing、fragment 与 instruction family；
4. **RISC-V 扩展局部接入**：新增扩展不要求新增完整 GEMM、Softmax、RMSNorm、量化算子模板；
5. **AOT 可嵌入性**：产物是普通 object / static library / header，运行期不依赖 Python、LLVM 或 JIT；
6. **外部 runtime 兼容性**：llama.cpp、ggml、框架线程池或应用自己的调度器可直接调用生成的 worker-local entry；
7. **算法与 realization 分离**：source 与 Kernel IR 固定 worker-local 算法，物理选择只存在于一次 target lowering 调用中。

### 非目标

Weft 不负责：

- framework graph 导入；
- 多算子 fusion、partition 或 end-to-end model compilation；
- 自动从任意 SSA 图发现 GEMM、attention、量化格式或完整算子；
- 自动发明作者未写出的 cache loop、staging skeleton 或算法 variant；
- 创建或管理 CPU 线程；
- OpenMP、pthread pool、work stealing、NUMA placement；
- 多个 hart 协作同一个 logical tile 的同步编程模型；
- 运行期 JIT；
- 通用正确性证明、实现审批或 certification framework；
- 用 kernel 名、格式名、route string 驱动 codegen。


## 根程序模型

### 一个 kernel、一种组合规则

Weft kernel不是scalar、VLA、block、state与extension几种kernel的并集。它只有一个根模型：
作者写下一个由普通region、typed SSA value、显式storage/memory effect和局部semantic primitive
组成的worker-local持续有序程序。Scalar control与VLA决定logical execution domain；scalar、
block与region只是同一value system中的shape kind；load/store、state algebra、dot/matmul和
extension primitive都消费这些value，并服从同一extent、validity、use-def与effect规则。

这些构造彼此正交：一个ordered loop可以包含VLA，一个VLA可以产生block-bearing region value，
一个dot可以位于其中，dot两侧也可以来自indexed memory；sequential state update可以在相邻处
使用reduce。出现组合不产生新的kernel类别，也不授权target接管外围结构。

### Worker-local kernel

一个 Weft kernel 是一个普通可调用函数。它接收显式 pointer / scalar / descriptor 参数，并由
当前 worker 从入口到返回持续执行。一次调用可以顺序遍历多个 cache/output block；block、
accumulator、state 与 source-visible workspace 的 lifetime 不受某个 VLA strip 或 local primitive
边界限制，而由作者写下的 lexical/control/storage relation 决定。

```text
external runtime
    → 选择 worker-local work slice
    → 调用 Weft kernel(args..., slice descriptor...)
    → Weft kernel 在当前 hart 上执行 scalar + VLA + extension code
```

Weft 不规定 work slice 的统一形状。应用可以传入：

- `row_begin / row_end`；
- 一个 tile index；
- expert range；
- quant block range；
- ragged descriptor；
- 任意普通 scalar / pointer 描述符。

因此，Weft ABI **不得**强制所有 kernel 使用 `work_begin/work_end`，但允许库提供该常见约定的 helper。

### 无 program grid

Canonical Weft language 中不存在：

- `program_id`；
- grid rank；
- launch grid；
- hart ID；
- worker ID；
- 隐式 task identity。

需要工作坐标时，调用者必须将其作为普通参数传入。

### 普通控制流

作者显式拥有：

- scalar `for` / `while` / `if`；
- cache / algorithmic blocking 的位置；
- staging 与重算骨架；
- outer K loop；
- block 之间的顺序与状态；
- memory effect 及其顺序。

普通 scalar `for` / `while` 是作者写下的有序 traversal。编译器只能做不改变 source-visible
iteration、carry 与 effect order 的普通实现优化，不能重新分类、交换或替换作者的 traversal，
也不得从普通 scalar multiply/add 猜出dot/matmul。
只有 source 显式写出的 `W.vla` / `W.dot` / `W.matmul` 才分别授权 SIMD logical axis 与局部
乘加域。VLA 内部可以做 strip-mining；dot/matmul 内部可以重组 reduction。两者都
不得改变 source-visible iteration/effect 语义、显式算法边界，或创建 source 中不存在的
algorithmic loop、state 与 staging 骨架。


## 两类一等数据域

Weft 不把所有计算统一成一个 Triton 风格 tile。语言有两类互补的一等数据域。

### VLA iteration region

VLA region 表示一个运行时长度的一维逻辑迭代域：

```python
with W.vla(begin, end) as i:
    ...
```

语义是：

```text
i ∈ [begin, end)
```

该逻辑域由编译器和目标实现分解为任意数量的连续动态 `vl` strip。source 不得观察：

- strip 数量；
- 当前 `vl`；
- strip ordinal；
- physical lane ID；
- fixed VLEN；
- LMUL。

VLA region 是普通 pointwise、memory、reduction、scan 与跨 strip summary 的主要执行域。

### Logical block value

Logical block 是具有显式 shape 的局部 SSA region value，例如：

```text
block<BM × BK, f16>
block<BK × BN, f16>
block<BM × BN, f32>
```

Shape不是block的完整identity。每次`W.block(extent)`建立一个唯一source axis，并产生该轴的
logical index block；block/region type同时保存shape与axis identity。两个extent相等但来自不同
`W.block`调用的axis不能互换。`W.full/W.zeros`直接消费这些axis value，而不是裸整数shape。

Logical block：

- 是局部数据域；
- 可以由`W.block`、singleton-axis view、full/zeros、load、pointwise与structured result产生；
- 是普通 SSA value，可以有多个 consumer，可以进入 pointwise、state、memory、control carry
  或另一个 structured primitive；
- 可以作为 `W.dot`、`W.matmul`、block reduction、decode 等 primitive 的 operand，
  也可以是它们的 result；
- 可以作为 `for` / `while` 的 accumulator 或 state 跨 logical iteration 存活；
- 不等于 cache block；
- 不等于 register microtile；
- 不等于 IME fragment；
- 不对应独立 worker、program instance 或 launch task；
- 可以是编译器中的 lazy region value，不要求先物化为实际数组或寄存器集合。

Structured primitive result 不能成为 fast-path terminal。`dot → add → store`、一个 result 的
多个 consumer、`matmul → pointwise → store` 和 block/state loop carry 都服从普通 SSA
composition。Target 若不能为合法 composition 建立 physical handoff，必须明确 unsupported；
不得靠要求精确 producer/use closure 来改变语言语义。

### 组合形态

一个 region value 可以具有：

- 零个或一个 VLA axis；
- 零个或多个 logical block axes；extent可以是static/meta，也可以由显式runtime value给出。

Canonical 类型可概念性表示为：

```text
region<[* , D0, D1, ...], [vla, a0, a1, ...], T>
```

其中 `*` 表示当前VLA axis；`Dk` 是static dimension，或以 `-1` 配合显式extent operand
表示的dynamic/meta dimension；`ak`是source-owned block axis identity。显式singleton broadcast
使用axis identity `0`，不能伪装成另一条真实axis。

当前 canonical language 在同一 lexical scope 中只允许一个活跃 VLA axis。嵌套第二个
VLA region 必须被拒绝；这是当前语言能力边界，用来保持 region identity 与 state 语义
唯一，并不是把 RVV lane count 暴露给 source。未来若引入多维 VLA，必须定义新的语言
语义，不能由 target 自动猜测。

该限制不禁止VLA body中的普通scalar `for` / `while` / `if`。短window、此前已选元素检查、
coordinate decode等有序scalar control可以嵌在VLA内；它们不会产生第二个lane domain，且
其source-visible顺序、state与effect必须保持不变。

`W.dot`与`W.matmul`不得缩并VLA axis；跨VLA axis的聚合必须使用reduce、scan或显式typed summary primitive。
VLA axis只能作为dot的free/batch axis；当前matmul只接受local block operands。


## Tile 与分块层次

Weft 必须区分以下四层，不得混用同一个 `tile` 概念：

### Algorithmic / cache block

由作者决定是否存在、位于哪个循环层、如何影响 memory reuse。典型参数为 `BM/BN/BK`。

这些参数可以是 build-time meta-parameter，但其**存在和使用位置**属于 source algorithm。

### Logical operand block

由 source 构造并由 structured primitive 消费的 shaped semantic value。

它只描述局部坐标域与数据关系，不声明寄存器或 ISA fragment。

### Register microtile

例如 `mr × nr` accumulator、register repeat、LMUL 组合及 K-unroll。

它属于 target lowering 的物理配置空间，由 lowering 检查 legality，构建期 tuning 循环选择。

### ISA fragment

例如某个矩阵扩展规定的 `4×4×8`、accumulator register class 或 encoded operand tile。

它是具体扩展的硬件叶子，只存在于 target lowering 的瞬态状态与生成代码中，不进入通用 source block 类型。


## Storage 与 lifetime

作者可观察的 memory object 都由 caller 分配并作为 entry pointer 传入。语言固定三类
source-visible storage：普通 external buffer/state、带显式 format identity 且跨调用复用的
persistent object，以及当前 worker 在一次调用内独占并可跨 loop/primitive 复用的 workspace。
Target 只可在 local primitive 内创建 source 不可观察的 primitive-private temporary。

Weft 没有源级隐式 allocation；storage class、shape、alignment、alias 与 lifetime 的完整合同见
[Storage ownership 与 lifetime](storage-and-lifetime.md)。这条边界保证 target 不会为了命中某个
实现偷造 workspace ABI、persistent repack 或另一份算法 state。


## GEMM 判据：不是 Triton-CPU 的另一层语法

一个 blocked GEMM worker 的 source 必须显式拥有 M/N/K traversal、BM/BN/BK 的使用位置、
operand block、accumulator、mask、staging、persistent packing 和 store。Accumulator 是普通
block SSA value，由同一个 worker 持有并跨作者写下的 K loop 更新；这个 worker还可以在外围
control 中继续处理下一个 M/N block。

`W.matmul(lhs, rhs, init=acc)` 只授权当前 `[BM,BK] × [BK,BN] + [BM,BN]` local block
product 的物理化。Realizer可以在该边界内选择 LMUL、register microtile、multiple
accumulators、K-unroll、短生命周期 packing、pipeline 与 RVV/IME fragment；它不得创建 outer
K loop、persistent repack、另一种 traversal 或新的 GEMM algorithm。

Triton 的典型 program/CTA 围绕一个逻辑拥有的 result tile 组织 collective execution；Weft
围绕一个 CPU worker 的持续 ordered control program 组织多个 block 及其 lifetime。两者都可
使用 tile 和 autotuned block size，但 owner 与授权边界不同，这才是 Weft 独立的程序模型。
