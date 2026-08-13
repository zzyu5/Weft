# 语言定位与程序模型

## 最终定义

Weft 是一门面向**单个 RISC-V worker / hart** 的高性能 kernel DSL 与 AOT 编译器。

作者使用普通控制流、一等 VLA iteration region、显式指针与逻辑 predicate、局部 logical block value、结构化 state algebra，以及 dot / matmul / scan / lookup / decode 等 primitive 编写完整的 worker-local kernel。编译器把这些结构映射到动态 `vl`、LMUL、寄存器组织、memory instruction、register microtile、矩阵 fragment 和 RISC-V 扩展指令。

多核线程创建、工作切分、线程池、OpenMP、affinity 与 NUMA 均由外部 runtime 负责，不属于 Weft core。

Weft 的扩展原则是：

> **已有语义扩展局部 target realization；新增可观察语义增加一个局部 primitive；永不为某个完整算子、模型格式或 kernel 名增加整段后端模板。**

Weft 不是 Tensor graph compiler，也不是 Triton CPU backend 的重写。它与 Triton 的关键区别是：

- Weft 没有 program grid、`program_id` 或隐式 launch identity；
- Weft 的入口是一个普通 worker-local callable kernel；
- VLA region 的 strip 边界与 `vl` 在 source 中不可观察；
- 跨 strip 的 reduce、scan、summary state 是一等语义；
- logical block 只作为局部数据域和 structured primitive 的 operand，不是整个程序的根执行单元；
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

### Worker-local kernel

一个 Weft kernel 是一个普通可调用函数。它接收显式 pointer / scalar / descriptor 参数，并在当前 worker 上完成一段工作。

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
- memory effect 与 atomic/fence。

普通 scalar `for` / `while` 是作者写下的有序 traversal。编译器可以对它做保持语义的
unroll、interchange、hoist、rematerialization、software pipelining 和局部 scheduling，
但不得把它重新分类为 VLA logical axis，也不得从普通 scalar multiply/add 猜出dot/matmul。
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

Logical block：

- 是局部数据域；
- 可以由 block axis、broadcast、load、reshape、transpose 和 pointwise 产生；
- 可以作为 `W.dot`、`W.matmul`、block reduction、permute、decode 等 primitive 的 operand；
- 不等于 cache block；
- 不等于 register microtile；
- 不等于 IME fragment；
- 不对应独立 worker、program instance 或 launch task；
- 可以是编译器中的 lazy region value，不要求先物化为实际数组或寄存器集合。

### 组合形态

一个 region value 可以具有：

- 零个或一个 VLA axis；
- 零个或多个 logical block axes；extent可以是static/meta，也可以由显式runtime value给出。

Canonical 类型可概念性表示为：

```text
region<[* , D0, D1, ...], T>
```

其中 `*` 表示当前VLA axis；`Dk` 是static dimension，或以 `-1` 配合显式extent operand
表示的dynamic/meta dimension。

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
