# Weft 当前状态与重构检查点

## 一句话结论

Weft 现在已经不再沿旧的 graph/problem/whole-kernel variant 骨架修补，而是在同一仓库中
长出了一条独立的新主干：

```text
Weft Python kernel / 其他前端 bridge
  → canonical Weft Kernel MLIR
  → selected RISC-V execution/layout MLIR
  → RVV 或 IME owner 发射
  → RISC-V source/object
```

这条主干已经能忠实编译多种由 pointwise、memory、reduction、contraction 和 carried
loop 组成的算法结构，并真实生成 RVV object；但它仍是一个严格、可执行的 compiler
slice，还不是“RISC-V 版 Triton 已完成”。当前最关键的缺口不是再加一个算子，而是把
layout/owner composition 从“单 owner 的若干可运行切片”推进成统一的 Scalar/RVV/IME
region compiler。

## 一、我们现在到底在做什么

### 1. 项目身份

Weft 是一门 standalone RISC-V blocked/VLA kernel DSL 与编译器，目标地位类似 Triton
之于 GPU，但执行模型不是 GPU SIMT 的翻版。

Weft source 描述一个 kernel 内的：

- task grid 与 task identity；
- logical block tensor、逐轴 shape 和 broadcast；
- pointer/index arithmetic、mask、load/store；
- pointwise SSA、reduction、contraction；
- structured loop 与 carried state；
- 可由编译器选择实例值的 meta-parameter。

Weft source 不描述：

- framework graph 或多-kernel graph 优化；
- 固定 VLEN、hardware lane 数、`vl`、LMUL 或 vector register number；
- RVV/IME 指令拼写；
- register allocation、spill 和最终 instruction schedule；
- 由 softmax、GEMM、量化格式或 kernel 名触发的整段模板。

所以当前工作不是“做一个算子库”，而是在建立一门能够承载算法骨架、再把它映射到
RISC-V execution resources 的 kernel language/compiler。

### 2. 与 IntentDSL 的关系

依赖方向已经确定为：

```text
Intent DSL
  → Intent side target plugin / 独立 bridge
  → canonical Weft Kernel MLIR
  → Weft compiler
```

Weft core 不解析 `intent.*`，也不链接 Intent verifier。Intent bridge 要完成一次真正的
target-oriented algorithm lowering：把 logical domain、view、validity、reduce、contract、
state 和 effects 变成 task、blocked loop、pointer/index、mask、load/store、reduce、contract
和 carried SSA。之后 Weft program 必须能脱离 Intent 独立保存、验证和编译。

当前这条 bridge 尚未实现；`intentdsl` 仍是只读上游。现阶段先用手写 Weft kernel
证明语言和后半条 compiler path 独立成立，是有意的实施顺序，不是绕开集成问题。

### 3. 这次属于重写还是增量重构

判决是：**新主干重写，低层能力增量抽取**。

- 不修补旧的 canonical problem → whole-kernel variant → format body → route 主干；
- 新增自己的 Python eDSL、Kernel dialect、Layout/Execution dialect、selector、owner config、
  source emitter 和 compiler CLI；
- 旧代码只作为硬件知识供体，逐件抽取 RVV intrinsic、IME leaf、ABI 和 object packaging；
- 新路径不读取旧 problem、variant、brick、route 或 kernel family identity；
- 旧主干尚未删除，但已不在 `weft-compile` 的正式数据流中。待新主干覆盖关键结构后，
  应按功能替代关系删除，而不是建立 compatibility layer。

这不是继续抢救旧骨架，也不是把整个仓库一次性清空重写；它是在现有仓库中建立一条
语义与物理边界都独立的新 compiler trunk。

## 二、当前承重架构

### Layer 1：canonical Weft Kernel MLIR

这是算法唯一真理。当前已建模：

- typed scalar/pointer/constexpr 与 kernel ABI；
- `task_id`；
- rank-N logical block、dynamic extent provenance、`arange`、`expand_dims`、singleton
  broadcast 和 `splat`；
- pointer arithmetic、masked load/store；
- unary/binary/compare/cast；
- `sum/max/min` reduction，显式 `init/axis/kind/ordered`；
- 显式成对轴的 contraction，`W.dot` 只是 source convenience；
- `for/yield` 与显式 carried SSA。

Python frontend 解析 AST 后直接生成注册的 `weft_kernel.*` MLIR。Python 侧只有通用的
operation/value/region assembly builder，没有第二套 typed Weft IR 或第二套 verifier。

尚未完整建模的 source language 面包括 `if/while`、atomic/fence、scan、一般
reshape/transpose、helper/call、正式 launch-grid extent contract 和更丰富 dtype 语义。

### Layer 2：selected RISC-V execution/layout MLIR

这一层只保存“多个合法物理实现中选了哪个”，不复制算法。

共享层当前包含：

- plan 的 source kernel 与 target identity；
- task/meta binding；
- value-local layout group；
- canonical value result → native group；
- canonical consumer operand → target group 的 `layout_conversion` use edge；
- group → owner symbol。

owner-local sibling dialect 当前包含：

- RVV `group_config`：lane ratio 与 register budget；
- RVV unary/reduction/contract strategy；
- IME contract fragment、VLEN 与 realization strategy。

已经删除未被真实发射消费的 group-global `tail_policy/mask_policy` 字段。masked load 的
inactive-lane 语义由 canonical `mask + other` 唯一决定，emitter 机械选择 `_tumu`；这不是
一个需要持久化的全 group 搜索决策。

### Layer 3：artifact

当前 `weft-compile` 支持：

- `selected-mlir`：canonical Kernel IR 加一份 selected execution/layout；
- `source`：按 selected owner 生成 source；
- `object`：RVV source 继续交给 target compiler 生成 RISC-V relocatable object。

IME 当前只到 selected MLIR/source；没有 IME object toolchain、public header、正式 runtime
driver 或 launch packaging。旧 EmitC/typed body 可以作为代码供体，但不是新主干的第四层
authority。

## 三、当前的 RISC-V 执行抽象

### 1. 不把整体叫作 SIMT

当前更准确的模型是：

```text
Kernel invocation
  → logical task grid
    → blocked tensor program
      → selected execution groups
         scalar | RVV VLA | IME fragment
```

- task 是 runtime 可调度的逻辑 work instance，不是 hart ID；
- block tensor 是 source-visible 的逻辑值，不是 vector register；
- RVV lane domain 是运行时 `vl` 决定的一维 VLA SIMD domain；
- IME 是独立 matrix/fragment execution，不是更大的 LMUL；
- Scalar 应成为一等 owner，而不是无 body 的 fallback。

因此 Weft 的核心抽象不是 GPU-style SIMT，而是 **task-grid + blocked program +
owner-local execution group**。

### 2. 当前 layout 表达了什么

当前 `#weft_layout.blocked` 是最小的 owner-independent value layout：

```text
rank          logical rank
order         value-local traversal order，最快轴在前
vector_axes   当前 RVV slice 中至多一个 VLA vector axis
```

它刻意不保存 shape、extent、stride、broadcast、reduction axis 或 contraction axis；这些都从
canonical SSA 重算。selector 先按 value/use 关系建立 layout component，再逐轴检查：

- 哪个轴是 reduction role；
- 哪个轴上的 store 可以证明 unit stride；
- load 是 unit-stride 还是 affine strided；
- 哪个轴实际变化而非 singleton；
- 剩余轴以什么顺序成为 serial loops。

例如 rank-3 affine 的 unit-stride store 位于 logical axis 1，所以选择
`order=[1,2,0], vector_axes=[1]`，并发射 `axis0 → axis2 → axis1-VLA`，而不是默认最后一轴。

### 3. “成熟 layout 体系”下一步应是什么

下一步不应简单地把 `vector_axes` 从一个放宽成多个。RVV 的硬件 lane domain 本质上是一维；
真正缺少的是把 logical axes 可组合地映射到不同物理角色：

```text
logical axes
  → task/serial loop axes
  → one RVV VLA lane axis
  → register-tile/repeat factors
  → reduction/contract roles
  → memory affine/indexed relation
  → IME fragment axes
```

因此成熟体系应增加的是：

1. 可组合的逐轴 role/mapping，而不是 kernel topology 分类；
2. logical coordinate → serial/VLA/register-tile coordinate 的映射代数；
3. value-native layout 与 consumer-use layout 的显式关系；
4. register tile/repeat/factorization，而不是伪造第二个 RVV lane axis；
5. owner-local IME fragment relation，不把 IME tile 塞进 RVV layout；
6. layout conversion 的真实 realization：rematerialize、register shuffle、memory
   transpose、owner handoff，各自有明确 legality 和 cost。

当前 `rank/order/one-vector-axis + use-edge conversion` 是这个体系的可执行地基，但还不是
终态。

## 四、已经真实成立的 compiler slice

### 1. RVV selection 与发射

当前 selector/emitter 已实现：

- rank-N pointwise/memory：一个 VLA vector axis + 任意数量有序 serial axes；
- f32 unit-stride masked load/store；
- f32 affine strided masked load；
- scalar、lane-varying和 vector-invariant mask 到 RVV predicate 的正确物化；
- f32 pointwise arithmetic、compare、negation；
- f32 `sum/max/min` reduction；
- rank-1 full f32 dot；
- 单 scope rank-2 f32 contraction；
- one-carried-state 的 tiled GEMM K loop；
- owner-local f32 `exp_poly_v1`，包括有限输入近似、`+Inf → +Inf`、`-Inf → 0` 和 NaN
  lane 恢复；
- VLEN-agnostic strip-mining、lane ratio/LMUL legality 与 register-budget selection；
- selected MLIR verifier/round-trip 和 RISC-V object packaging。

当前真实 use-edge layout conversion 只有：

1. For/Yield carried state 的 layout-preserving alias/handoff；
2. splat 为不同 consumer layout 重新物化；
3. direct masked load 在无 intervening store、目标 pointer affine 可证明时重新 load。

它们都不是通用 register transpose。

### 2. 算法结构与证据边界

| 算法结构 | 新主链到达位置 | 证据边界 |
|---|---|---|
| RMSNorm | canonical → selected → RVV object → 真机数值 | 已有标量参考对照 |
| rank-2 affine | 同上 | 已有真机数值对照 |
| rowwise/columnwise reduction | 同上 | 已有真机数值对照；columnwise 使用 runtime `vlse` stride |
| rank-1 dot | 同上 | 已有真机数值对照 |
| rank-2 direct contract | 同上 | 已有真机数值对照 |
| carried-loop GEMM | 同上 | 已有不同 BM/BN/BK 绑定下的真机数值对照 |
| splat dual-axis conversion | 同上 | 已有真机数值对照 |
| masked-load dual-axis conversion | canonical → selected → RVV object | 尚无本轮真机数值 |
| mixed-axis independent reductions | canonical → selected → RVV object | 尚无本轮真机数值 |
| rowwise max | canonical → selected → RVV object | 已发出 `vfredmax`，尚无本轮真机数值 |
| three-stage softmax | canonical → selected → RVV object | 已发出 max、exp、sum、normalize；尚无新主干真机数值 |
| rank-3 affine | canonical → selected → RVV object | 已按 layout order 发射三层结构；尚无本轮真机数值 |
| signed widening IME contract | canonical → selected → IME source | syntax-only；无新主干 IME object/真机数值 |

这里的 softmax 不是一个 fused `softmax` op，也没有名称分支。source 明确写出 max pass、
exp+sum pass 和 exp+normalize/store pass；selector 分别为每个 stage 选择 group 和 owner-local
strategy，emitter遵守该 algorithm skeleton。

### 3. 本检查点的最新收紧

提交前又完成了以下结构修复：

- scalar i1 和 vector-invariant blocked i1 memory mask 显式广播为 RVV predicate；
- mask splat 可真实发射 `vmset/vmclr`；
- arange/expand/binary 的逐轴 provenance DFS 使用 branch-local visited set，避免共享 DAG
  的一条路径抹掉另一条 logical axis；
- contract verifier 的 lhs/rhs 使用 consumer effective-use group，允许未来的 operand
  `layout_conversion`，result 仍使用 native value group；
- group-global tail/mask policy 从 selected schema 删除；
- vector exp 先把非有限 lane 隔离出有限多项式，再显式恢复 Inf/NaN value semantics；
- rematerialized load、masked load/store 与 exp owner strategy 的临时寄存器压力进入 lane-ratio
  选择。

当前构建成功；现有算法 source 均重新走过 canonical、selected verifier/round-trip、source
与其当前支持的 object/syntax 边界。另一个一次性 scalar-mask kernel 已实际生成包含
`vmset/vmclr + vle_tumu + vse_m` 的 RVV object；没有向仓库加入 test/fixture。

## 五、目前还不能声称什么

Weft 当前不能声称：

- 已经具有 Triton 同等成熟度；
- 一个 kernel 内已经能统一混合 Scalar/RVV/IME owner；
- IME object、IME runtime correctness 或 IME performance 已完成；
- 一般 register layout conversion、transpose 或跨 owner handoff 已完成；
- 多维 register tiling、RVV register-blocked microkernel 已完成；
- rank>2 reduction/contract、沿 physical serial axis 或多轴 reduction 已完成；
- strided store、indexed gather/scatter、segment memory 已接入新 owner contract；
- 多个 carried state、多阶段 GEMM、一般 control/state/ragged/effects 已完成；
- 自动 BM/BN/BK tuning、cost model、profile-guided selection 已完成；
- public header、runtime launch 和稳定外部 compiler library API 已完成；
- Intent→Weft bridge 已完成。

尤其要区分：

- selected MLIR 通过只证明 plan 自洽；
- source/object 生成只证明 artifact 闭合与 target compiler 接受；
- 只有真实目标运行和数值对照才是 runtime correctness；
- 当前 exp 特殊值与新的 scalar mask 已到 object，但尚无本轮真机数值；
- exp 的 `8 scalable + 2 mask` 是 owner-local 的保守峰值摘要，目前仍是静态契约，尚未
  与 emitter 内部临时值 liveness 机械共享或通过反汇编/实机压力验证。

## 六、当前真正的承重问题

### 1. 统一 mixed-owner selection/emission

当前 CLI 仍按 target extension 在 whole-kernel RVV selector 与 IME selector 之间二选一。
shared `group/owner/value_layout/layout_conversion` schema 已经能表达一部分组合，但缺少：

- 每个 canonical operation/region 的唯一 owner binding；
- 不产生 block result 的 Scalar operation owner record；
- RVV、IME、Scalar analyzer 向同一个 orchestrator 返回候选/选择的接口；
- 一个只生成一次函数 ABI/控制结构，再让各 owner 发射 fragment 的 composite emitter；
- 跨 owner blocked value handoff 和 cleanup。

这是下一阶段最重要的架构工作。若继续在 RVV/IME 两个 whole-kernel emitter 上叠算子，项目
会重新形成互斥模板树。

### 2. layout algebra 与真实 conversion

当前 value-local layout 已经消除了 whole-kernel shape 分类，但仍只有 traversal order 和一个
VLA axis。下一步需要 register tile/repeat、operand-role relation、memory mapping 与
use-edge conversion cost/realization。否则 contraction 性能只能停在 sequential
outer-product，而不能长成专家级 RVV microkernel。

### 3. Scalar 是一等 owner

没有 Scalar owner，就无法自然完成：

- control-heavy region；
- small problem；
- RVV/IME tail cleanup；
- unsupported vector instruction 的合法实现；
- mixed-owner region 的 glue。

Scalar 不能是 silent fallback；它也必须有 selected ownership、真实 body 和 fail-closed
emission。

### 4. resource/numerical contract

register budget 目前已影响 LMUL/lane ratio，但复杂 owner strategy 的内部峰值仍由静态摘要
给出。需要建立 selector 与 emitter 共享的 owner-local resource contract，且不把内部算法
临时 SSA 复制进 canonical/selected schema。

同样，近似数学函数需要明确 value semantics、误差边界、exceptional values 与 target
evidence。`exp_poly_v1` 现在有明确特殊值恢复，但仍需真实 RVV 数值和寄存器行为证据。

### 5. artifact/runtime 与 IME 工具链

RVV object 已成立，但完整系统还缺 header、launch/runtime driver 和稳定 callable packaging。
IME 则缺 object assembler/toolchain、目标机和新主链数值闭环。没有这些条件时只能诚实停在
source。

## 七、建议的下一阶段顺序

1. 建立 unified selection orchestrator，把 owner 选择从 whole-kernel 二选一改为
   operation/region ownership；同时定义最小 Scalar owner。
2. 把现有 RVV/IME selector 拆成 owner-local legality/plan contributor，不再各自 materialize
   完整 plan。
3. 建立 composite emitter：共享 ABI、control skeleton 和 canonical traversal，各 owner 只
   发射被分配的 fragment。
4. 在这个统一骨架上推进第一条 IME contract + RVV/Scalar cleanup 的 mixed-owner kernel。
5. 再扩展 layout algebra 到 register tile/repeat 与真实 operand conversion，并用它产生一条
   RVV register-blocked contraction，而不是增加 GEMM 名称模板。
6. 最后接 Intent→Weft bridge；bridge 只需生成已经独立成立的 canonical Weft program，
   不参与 RISC-V owner 或 artifact 选择。

这个顺序的理由是：当前最危险的退化方向不是“少一个算子”，而是每加一个算法结构就在
whole-kernel selector/emitter 中再长一个互斥分支。先完成统一 ownership/composition，后续
reduction、scan、ragged、IME 和新 extension 才能以局部 owner 增量接入。

## 八、检查点判决

当前 Weft 已经证明以下命题：

> 一门独立的 blocked kernel DSL 可以保留算法 skeleton，并从 canonical SSA 逐轴选择
> RISC-V VLA layout，生成真实 RVV object；RVV reduction、contraction、carried loop、
> use-edge rematerialization和 owner-local numerical strategy 不需要依赖 kernel 名模板。

但还没有证明更强的命题：

> 同一个一般 Weft kernel 可以由统一 compiler 把不同 region/value/use 分配给
> Scalar、RVV、IME，并通过成熟 layout algebra 和真实 handoff 生成高性能 artifact。

因此当前项目状态应描述为：

**独立 DSL 与 RVV 主链已经成立；value-local VLA layout 的可执行地基已经成立；IME 有一条
严格 source slice；统一 mixed-owner 和成熟 register-layout system 仍是下一阶段核心。**
