# Weft 编程模型与 DSL 文档重构

## 1. 本轮结果

本轮只重写并冻结 Weft 的编程模型与 DSL 规范，没有修改 Python 前端、Canonical Kernel IR 实现、RISC-V lowering、emitter 或性能数据。

最终文档不再描述设计演化、实现进度或历史轮次，而是直接回答：

- Weft 作者在写什么程序；
- 哪些结构必须由作者显式决定；
- 哪些变化只属于目标编译器的物理表示；
- Encoding、Value、Level、普通控制、engine role 与标准库分别具有什么语义；
- Canonical Kernel IR 必须保存什么，以及不得保存什么；
- Weft 与 Triton、TileLang 的机制差异是什么。

## 2. 最终目录

文档只保留两个目录，由一个顶层索引统一导航：

```text
doc/
├── index.md
├── model/
│   ├── programming-model.md
│   └── non-goals.md
└── dsl/
    ├── encoding-and-artifacts.md
    ├── values-and-levels.md
    ├── operations-and-engines.md
    ├── examples.md
    └── canonical-ir.md
```

没有为每个章节建立子目录或重复 README。五个语言示例保留在同一个文件中，避免目录继续膨胀。

## 3. 冻结的编程模型

### 3.1 Weft 所处的位置

Weft 承载的是数学算子定义与手写 intrinsic 之间的数值 realization：作者明确写出逻辑值、有限位宽、代数支路、频率层次、供应与复用关系；编译器负责将这些结构物理化为 lane、register tuple、fragment、memory operation、intrinsic 或局部 asm。

Q4_K 用来说明这一边界：

- min correction 与 q-product 是两条作者写出的数值支路；
- element、sub-block、block 三种计算频率由作者树表达；
- i16 partial、i32 accumulator 与 f32 block result 的位置属于数值语义；
- 编译器不能从普通 dequant 表达式自动发现这些结构。

### 3.2 执行模型

一份 Weft kernel 是一次普通函数调用中的完整有序程序。语言没有隐式 grid、program id、worker/hart identity、thread/warp/CTA ownership，也不假定每个调用实例天然拥有一个 tile。

普通 `for/while/if` 是有序标量控制，不会被自动恢复成 shaped axis 或 VLA domain。需要 shaped logical axis、cohort 或层级 lifetime 时，作者显式使用 Value 与 Level。

### 3.3 唯一职责判据

最终判据是：

> 对开放的 canonical SSA，改变逻辑值集合或改变逻辑值的 Level 归属，由作者写；不改变这两者的机器实现，由编译器决定。

closed primitive 内部不可观察的 physical partial、归约树、spill slot 和 fragment temporary 不属于新的 source logical value。

这个判据区分了两类容易混淆的变化：

- accumulator 跨 KC 还是跨整个 K、persistent interleave 是否存在、一个 logical cohort 是 16 还是 4，都是作者程序；
- LMUL、lane/register/fragment mapping、memory form、局部 pipeline、spill/reload，则属于编译器表示。

## 4. 冻结的 DSL 构造

### 4.1 Encoding、View 与 Artifact

Encoding 只定义 logical field coordinate 到 storage unit/bit range 的布局关系，不携带 dequant 公式、scale/min 语义、字段不变量或 ISA 选择。

规范明确了：

- `elements` 必须显式声明，不能由最大字段长度猜测；
- bit order、byte order、record alignment、field span 与 padding；
- `natural/grouped/layered/bit_planes/joined` 的组合语义；
- record-local field axis 与 Level point 的唯一投影规则；
- base encoding、derived family 与 pinned derived instance 的区别；
- artifact phase 与 invocation phase 的边界；
- ordinary View、caller-visible workspace、persistent packed object、invocation-local staged pack 与 primitive-private temporary 的不同生命周期。

离散 storage bytes 与 packed field 可以要求逐 byte/field 一致；浮点 kernel 使用 operation 语义与数值容差，不要求 bit-exact。

### 4.2 Value、Level 与控制

Value 是带 element type、logical shape 和 axis identity 的普通 SSA value。structured operation 的结果仍可拥有多个 consumer、进入 pointwise/state/memory/另一个 primitive，不能退化成 direct-store closure。

Level 正式保存：

- domain 与 parent relation；
- partition、multiplicity 与 tail；
- `births.state`；
- `births.staged`；
- body；
- typed handoff。

表面 handoff 使用普通 SSA 赋值；Canonical Kernel IR 必须保留层归属。跨 engine 的 `stage_handoff` 与 Level terminator handoff 是两种不同结构。

### 4.3 函数、Operation 与 Engine Role

std 与应用 kernel 使用同一门语言。普通 helper 在形成 canonical program 前 inline；`@weft.derive` builder 则在 artifact phase 单独求值。

语言核心没有内置 whole-kernel GEMM。GEMM/GEMV 由 Level、`pack/materialize/admit/commit`、state 与局部 `contract/outer_contract` 组成。

`auto` 只实例化作者声明的 source 参数。每个具体 binding 形成一棵固定 candidate；target lowering 不能回头改变 Level topology。

`scalar/wide/matrix/transfer` 是硬 engine role。未标注表示没有额外 source 约束；一旦写出 `@wide` 或 `@matrix`，目标不能静默换成另一 role。

## 5. 示例承担的判据

单一示例文档保留五个互补程序：

- Q4_K × Q8_K multi-output vec-dot：Encoding、派生 artifact、代数支路、频率层次与有限位宽；
- blocked MUL_MAT：NC/KC/MC、panel materialization、MR/NR cohort 与 accumulator 的 KC scope；
- GEMV：GEMM 去掉 column Level 后的自然退化，不引入新 primitive；
- online attention：多个 state、非归约 handoff 与顺序性来自 use-def；
- Top-K：明确只使用普通有序标量控制，不使用 Level。

示例中同时修正了 Q4_K record-local scale 投影、min 支路 accumulator 类型、attention 的 f32 accumulation/f16 narrow，以及 Top-K 的静态域和 tie/NaN policy。

## 6. Canonical Kernel IR 边界

Canonical Kernel IR 是作者数值树的持久 authority，必须保存：

- Encoding/View/Value 的逻辑类型与 axis；
- ordinary control 与 SSA carry；
- Level domain、births、body 与 handoff；
- operation 的数值、精度、overflow、effect 与 engine role；
- derived layout identity 与调用边界。

它不保存 SEW、LMUL、physical lane、register tuple、fragment、selected instruction、memory form、layout conversion、pipeline、spill 或 intrinsic spelling。

本轮没有规定 physical IR 应采用什么 dialect、pass 或改写机制；这部分没有被混入 DSL spec。

## 7. Triton 与 TileLang 对照

对照结论按参考仓库中的真实机制改写：

- Triton 以 launch grid/program instance 为公开执行分解，TritonGPU layout encoding 描述 register/lane/warp/block/CTA ownership；Weft 不把这种 physical ownership 作为 source 根模型。
- TileLang 源程序显式承载 launch/thread binding、buffer scope、copy intent、tile operation 与 pipeline annotation；Weft 保存的是 encoding-aware 数值树、logical lifetime 与 Level handoff。
- Triton 与 TileLang 都可以手写量化 decode、attention 或状态程序；差异不是“能不能写”，而是哪些结构是一等 source contract，并能在后续物理化中自然保存。

## 8. 清理结果

- 删除了旧的 language/model/compiler planning 文档，未保留双份规范。
- 文档目录收缩为 `model/` 与 `dsl/` 两类。
- 删除仓库中 69 个空目录，包括旧文档目录、历史 examples 占位目录和空的 include/lib/build 子目录；没有删除任何非空目录或文件。
- 用户此前清理的 `report/history/` 历史快照一并从仓库提交，不再保留空 history 目录。

本报告记录的是这一轮文档重构结果，不作为后续实现状态或性能数字的维护入口。
