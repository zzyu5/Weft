# Weft：独立 RISC-V Blocked/VLA Kernel DSL 与 Intent 集成总设计

## 最终结论

Weft 应当成为一门独立、可手写、可由其他编译器生成的 RISC-V kernel DSL，
而不是直接消费 Intent MLIR 的一组后端 passes。

正确依赖方向是：

~~~text
Intent DSL
  → Intent 的 Weft target plugin / 独立 bridge
  → canonical Weft Kernel IR
  → Weft compiler
  → selected Scalar/RVV/IME execution
  → RISC-V source/object/header
~~~

同时，用户可以完全绕过 Intent：

~~~text
hand-written Weft Python kernel
  → canonical Weft Kernel IR
  → 同一个 Weft compiler
  → RISC-V artifact
~~~

所以这里选择的不是“纯外部进程”或“纯嵌入式”中的一个，而是：

> 语义上独立，仓库和编译器边界独立；技术上保持 MLIR-native，可以同进程
> dialect-to-dialect lowering，也可以通过 textual MLIR 调用外部 weft-compile。

Weft core 永远不解析 intent.*，也不依赖 intentdsl。只有 bridge 同时认识两种
dialect，依赖方向只能是 Intent/bridge → Weft。

---

## 一、对上一版设计的纠正

上一版把 Weft 定义成：

~~~text
canonical Intent Kernel MLIR
  → Weft physical decisions
  → RISC-V
~~~

这个方案短期容易接通，因为可以直接读取 Intent 的 domain、reduce、contract、
state_stream 和 effects；但长期会产生三个问题：

1. Weft 只在有 Intent 输入时才完整，无法成为独立 kernel language；
2. Weft core 会持续跟随 intentdsl 的 IR、verifier 和 target refactor；
3. 每增加一种 Weft 能力，都可能同时改 Intent adapter、Weft plan 和 emitter，
   最终变成 intentdsl 内部一个越来越大的 RISC-V 子系统。

这正是用户担心的“越做越乱”：项目看似分仓，语义上却仍然嵌在 Intent 内部。

正确做法不是让 Weft 永远引用 Intent node/value ID，而是进行一次正常的、
单向的 compiler lowering：

~~~text
Intent algorithm
  → 一个完整、可独立解释和编译的 Weft blocked program
~~~

生成后的 Weft program 是更低层算法表示。它确实重新表达了循环、load/store、
reduce/dot 和 mask，但这是正常 lowering 结果，不是与 Intent 并行维护的第二份
真理：

- 不做 Weft→Intent round-trip；
- 不要求两个 IR 长期双向同步；
- bridge 完成后，Weft compiler 只认 Weft IR；
- source location/provenance 可以保留，但不成为语义依赖；
- Intent 改动只需要调整 bridge，不要求 Weft core 理解 Intent。

PyTorch/Inductor 生成 Triton program、Triton 再编译 GPU kernel，本质上也是这种
边界。

---

## 二、三种方案的裁决

### 方案 A：Weft 直接嵌入 intentdsl

~~~text
intent.* → intent_plan → RISC-V passes/emitter
~~~

优点：

- 同一个 MLIRContext；
- 可以直接访问 Intent node、region、effects；
- source diagnostic 容易保持；
- 早期少一个显式 IR 边界。

缺点：

- Weft 不再是独立语言；
- RISC-V capability、RVV/IME、toolchain 和 artifact 进入 intentdsl；
- intentdsl 当前 GPU target/plan 正在重构，Weft 会跟着一起变化；
- 其他前端无法独立使用 Weft；
- 手写 Weft kernel 会退化成旁路；
- 项目很容易再次围绕“某个上游算子怎么接”生长。

判决：不选。

### 方案 B：Weft core 直接消费 Intent MLIR

~~~text
intent.* → weft-compile → RISC-V
~~~

优点：

- 仓库仍然分开；
- 可以复用 canonical Intent IR；
- Weft 可以掌握全部 RISC-V realization。

缺点：

- Weft 必须链接 Intent dialect/verifier；
- Weft 输入仍不是自己的语言；
- standalone 手写能力需要另造入口；
- Intent IR 变化会直接改变 Weft core；
- Weft 的抽象容易只是一份 Intent physical plan，而不是 Triton-like kernel DSL。

判决：不作为核心架构。可以有独立实验 adapter，但不能成为 weft-compile 的正式
输入。

### 方案 C：独立 Weft DSL + MLIR-native bridge

~~~text
Intent → ConvertIntentToWeft → weft.* → weft-compile → RISC-V
                           ↑
                  hand-written Weft
~~~

优点：

- Weft 是真正独立的语言和编译器；
- hand-written 与 generated kernel 是同一入口；
- Weft 可以被 Intent、其他编译器和人工直接使用；
- Intent 只拥有 bridge，不拥有 RISC-V lowering；
- Weft 只需要稳定自己的 Kernel IR；
- 仍可在同一 MLIR 进程中直接 conversion，不牺牲 MLIR 优势；
- 也可使用 textual MLIR/外部 CLI 获得进程隔离。

代价：

- 必须认真设计 Weft Kernel IR；
- Intent→Weft bridge 是真实的 target lowering，不是几行 adapter；
- 需要明确哪些 blocked decisions 在 bridge 中产生，哪些由 Weft compiler选择；
- 多一个可读 IR 边界和编译器 package。

判决：选择。

---

## 三、Weft 到底是什么

Weft 最短定义：

> 一门面向 RISC-V Scalar、RVV、IME 与未来扩展的 blocked/VLA kernel DSL，
> 以及把该 DSL 编译成高质量 RISC-V kernel artifact 的编译器。

它与 Triton 相似：

- 用户或上游生成器写一个 blocked kernel program；
- source 显式 task/grid、tile/block、地址、mask、load/store、reduce/dot；
- compiler 负责把 block tensor 映射到真实执行资源；
- 生成可读 source 和可调用 artifact；
- meta-parameters 可以由 tuner 选择。

它与 Triton 不同：

- GPU 的 CTA/warp/lane 不是共同执行模型；
- RVV 的 vector width 在编译时可以未知；
- 一个 logical block 可能需要多次 AVL→vl strip-mining；
- Scalar、RVV、IME 可以在一个 kernel 内分工；
- IME 不是“更宽的 RVV”，有独立 tile/accumulator contract；
- task 可以由 runtime 调度到 hart，不等于 hardware hart ID。

Triton 官方把其模型描述为 blocked program，而不是 blocked threads：
[Triton introduction](https://triton-lang.org/main/programming-guide/chapter-1/introduction.html)。
GPU layout 再把 block elements 分布到 warp/lane/register：
[Triton/Gluon layouts](https://triton-lang.org/main/getting-started/tutorials/gluon/layouts.html)。

RVV 的关键事实来自：

- [RISC-V V 1.0](https://docs.riscv.org/reference/isa/unpriv/v-st-ext)
- [LLVM RISC-V Vector Extension](https://llvm.org/docs/RISCV/RISCVVectorExtension.html)

包括 VLEN-agnostic scalable vector、AVL/vl、SEW/LMUL、tail/mask policy、
unit/strided/indexed/segment memory 和 ordered/unordered reduction。

---

## 四、Weft DSL 的层级位置

### Intent DSL

描述较高层、portable 的完整 kernel algorithm：

- logical domain/region；
- structured tensor-flow；
- state_stream；
- ragged relation；
- source-visible ABI/effects；
- compiler-owned I.auto。

用户不写 pointer arithmetic、program/task ID 或目标 blocked layout。

### Weft DSL

描述 RISC-V-oriented blocked kernel program：

- 一个 task instance 做什么；
- task grid 如何覆盖工作；
- block tensor 的 logical extent；
- pointer/index arithmetic；
- logical mask 和 fill；
- explicit load/store；
- block-level reduce/scan/dot；
- explicit structured loops 和 loop-carried state；
- compile-time meta-parameters 与合法 tuning space。

它不写 fixed VLEN、LMUL、vector register 或具体 instruction。

### RISC-V Execution IR

Weft compiler 选择：

- block tensor 的 physical layout；
- Scalar/RVV/IME owner；
- RVV strip-mining、SEW、LMUL，以及不能从 canonical memory op 唯一决定的
  mask/tail realization；
- memory instruction class；
- reduction tree 与 cross-strip accumulator；
- IME tile/accumulator；
- owner handoff 和 cleanup；
- 已选 meta-parameter。

### target lowering

生成：

- Scalar C/C++；
- RVV intrinsics 或必要的 lower-level IR；
- IME intrinsic/inline-assembly leaf；
- object/header/callable entry。

---

## 五、Weft source 应显式什么

### 1. Kernel entry 与 ABI

Weft source 是单-kernel language。一个 entry 至少包含：

- typed pointer/scalar parameters；
- address space 或 memory class constraint；
- alignment、alias/noalias；
- runtime shape/stride；
- compile-time meta-parameters；
- output/effect contract。

Weft 不决定 framework output allocation、多-kernel graph 或 autograd。

### 2. Task grid

类似 Triton program_id，但不叫 hart_id：

~~~python
row = W.task_id(0)
~~~

task_id 是 launch grid 中的逻辑 task identity。Runtime 可以：

- 单 hart 顺序执行多个 tasks；
- thread pool 把 tasks 分配到多个 harts；
- persistent worker 重复领取 tasks。

Weft source 不能假设 task_id 等于物理 hart。

### 3. Block tensor

~~~python
cols = W.arange(0, BLOCK_N)
~~~

W.arange 产生 logical block elements。BLOCK_N：

- 可以大于 VLMAX；
- 不等于 hardware lane count；
- 可以是 compile-time meta-parameter；
- 可以在 selected execution 中被分成多个 vl chunks；
- 可以映射到 Scalar、RVV 或 IME-compatible layout。

Block tensor 支持：

- rank/shape/dtype；
- broadcast、reshape、transpose；
- pointwise SSA；
- explicit cast；
- logical predicate。

### 4. Pointer/index 与 memory

Weft source 显式：

~~~python
offsets = row * stride_row + cols * stride_col
values = W.load(ptr + offsets, mask=valid, other=0.0)
W.store(out + offsets, result, mask=valid)
~~~

这与 Intent 不同：Weft 已经进入 target-oriented blocked program，地址和 blocked
index 是算法 lowering 的一部分。

Compiler 再选择：

- unit-stride；
- strided；
- indexed gather/scatter；
- segment；
- scalar cleanup；
- load/store grouping。

### 5. Mask 与 validity

Source mask 是 block-program 语义：

~~~python
valid = cols < n
values = W.load(ptr + cols, mask=valid, other=0.0)
~~~

Compiler 选择：

- AVL/vl 直接收紧；
- RVV mask；
- mask + tail policy；
- scalar branch；
- IME full tile + RVV/Scalar cleanup。

source mask 不等于 RVV v0，也不等于 tail policy。

### 6. Reduction

~~~python
partial = W.sum(values * values, axis=0)
~~~

Source 固定：

- block values；
- axis；
- identity；
- combine；
- accumulator dtype；
- ordered/unordered requirement。

Compiler 决定：

- scalar loop；
- RVV horizontal reduction；
- vector accumulator + trailing reduction；
- cross-strip scalar/vector accumulation；
- IME partial 与 RVV/Scalar fold。

### 7. Dot / contraction

~~~python
acc = W.contract(
    a_block,
    b_block,
    acc,
    lhs_axes=(-1,),
    rhs_axes=(0,),
    ordered=False,
)
# rank-1/rank-2 convenience：acc = W.dot(a_block, b_block, acc)
~~~

Source 固定 blocked operands、成对 contraction axes、accumulator value/type、输出轴
顺序和 ordered requirement。canonical op 不保存 target layout 或 instruction。

Compiler 可以选择：

- scalar FMA；
- RVV outer-product、dot 或 register-blocked microkernel；
- IME matrix tile；
- widening/composite instruction sequence。

Source 不出现 vmadot、vfmacc 或具体 fragment register。

### 8. Control 与 state

Weft 必须支持：

- if；
- for/range；
- while；
- loop-carried SSA；
- early/guarded region；
- helper function；
- explicit atomic/fence。

Intent state_stream 会在 bridge 中 lowering 成明确的 blocked loop 与 carried state。
Weft core 不需要认识 state_stream 这个上游概念，但必须能表达其 lowering 结果。

### 9. Meta-parameters

Weft source 可以显式：

~~~python
BLOCK_M: W.constexpr
BLOCK_N: W.constexpr
BLOCK_K: W.constexpr
~~~

允许 tuner 在合法候选中选择：

- block extent；
- task grouping；
- unroll/prefetch 参数；
- algorithm不变前提下的 blocked layout。

不允许 meta-parameter：

- 改变 source-visible ABI；
- 偷换算法；
- 让非法 owner 合法；
- 直接等同于 LMUL/register number；
- 进入 runtime data-dependent branch。

---

## 六、Weft source 应隐藏什么

- fixed VLEN；
- vlenb；
- hardware lane count；
- exact vl；
- LMUL 与 vector register number；
- vsetvl placement；
- RVV instruction spelling；
- RVV v0/mask register allocation；
- tail/mask agnostic/undisturbed implementation；
- IME source/accumulator physical registers；
- register allocation、spill；
- final instruction scheduling；
- compiler/sysroot invocation details；
- artifact route string。

这些属于 selected RISC-V execution 或更低层。

---

## 七、为什么这不是第二个 Intent

Intent 与 Weft 的抽象差必须足够大。

| 问题 | Intent | Weft |
|---|---|---|
| 输入身份 | portable logical kernel | RISC-V-oriented blocked kernel |
| work identity | domain/region | task grid + block elements |
| memory | logical view/index relation | typed pointer + explicit offsets |
| tile | I.auto，source 不观察 | block/meta-parameter 显式 |
| mask | logical predicate/fill | blocked load/store mask |
| reduction | structured algorithm node | block reduction |
| contraction | logical tensor contraction | blocked dot/microkernel anchor |
| state | state_stream/carry semantics | explicit loops + carried SSA |
| ragged | relation/descriptor | offsets/indices/pointer loops/masks |
| owner | 不存在 | compiler选择 Scalar/RVV/IME |
| VLEN/LMUL | 不存在 | target profile 给出 VLEN 约束，execution 选择 LMUL/strip-mining；运行时 `vl` 保持动态 |

如果 Weft source 仍然只写 domain、parallel、state_stream、ragged relation，
它就是第二个 Intent，边界失败。

如果 Weft source 已经写 vsetvl、LMUL、vmadot、register number，它又太低，
无法继承 compiler 和未来硬件的改进。

正确中间点是：

> 显式 blocked program，隐藏具体 RISC-V execution resources。

---

## 八、RISC-V 执行模型

### 1. 不称整体为 SIMT

Weft 的外层 task grid 具有 SPMD-like 形式，但 RVV 内层是 VLA SIMD，IME 是
matrix/tile execution，Scalar 没有 lane。

建议称：

> blocked/VLA RISC-V kernel model

或：

> task-grid + execution-group model

### 2. 三层物理层级

~~~text
Kernel Invocation
  → Task Grid
    → Block Tensor Program
      → Execution Groups
         scalar | rvv | ime
~~~

Task 是逻辑 work instance，不是 hart。
Block Tensor 是 source-visible blocked values，不是 vector register。
Execution Group 才是 selected target mapping。

### 3. RVV mapping

一个 block extent B 的 RVV lowering：

~~~text
remaining = B
base = 0
while remaining > 0:
    vl = choose_vl(remaining, SEW, LMUL)
    execute block[base : base + vl]
    base += vl
    remaining -= vl
~~~

Selected execution 需要建模：

- scalable vector shape；
- SEW/LMUL；
- AVL来源；
- canonical mask 到 AVL/predicate 的 realization；inactive-lane policy 若已由
  canonical `mask + other` 唯一确定，则不重复持久化；
- live vector groups/register budget；
- memory addressing mode；
- reduction state across strips。

### 4. IME mapping

IME owner 单独提供：

- capability identity；
- supported tile shapes；
- input/accumulator dtype；
- signedness/widening；
- source/accumulator register class；
- load/store contract；
- edge/tail ability；
- ABI/intrinsic/assembly；
- RVV/Scalar handoff。

IME 不继承 RVV 的 vl、LMUL 或 mask 语义。

### 5. Scalar mapping

Scalar 不是空 fallback envelope。它必须可以真实实现：

- whole kernel；
- small problem；
- control-heavy region；
- RVV/IME cleanup；
- unsupported vector/matrix operation。

无真实 body 就编译失败。

---

## 九、Weft 内部表示

### Layer 1：canonical Weft Kernel IR

由 Python eDSL 或其他 frontend 直接生成，保存完整 blocked algorithm：

- entry ABI；
- task/grid；
- block tensor SSA；
- pointer/index；
- mask/load/store；
- pointwise；
- reduce/scan/dot；
- structured control/state；
- meta-parameters；
- source locations。

Python frontend 不维护平行 typed IR。

### Layer 2：selected RISC-V Execution IR

保存不可从 Kernel IR 唯一推导的选择：

- target profile；
- task scheduling/layout；
- block tensor physical layout；
- Scalar/RVV/IME owner；
- RVV SEW/LMUL、AVL 构造、strip-mining 与非派生的 mask/tail 选择；
- IME tile/accumulator；
- memory lowering；
- reduction/dot strategy；
- owner handoff；
- selected meta-parameters；
- private scratch。

共享 execution ops 与 owner-local extension ops 可以处于同一层：

~~~text
weft execution core
  + rvv execution extensions
  + ime execution extensions
  + scalar execution extensions
~~~

不是 giant optional struct。

### Layer 3：artifact

- readable generated source；
- object；
- header；
- callable entry；
- 可选 lower-level IR。

### 瞬态 lowering

typed RVV/IME ops、EmitC、LLVM IR 可以存在，但：

- 单一 producer；
- 不新增 selection；
- 不作为独立 front door；
- 不反向驱动 Kernel/Execution IR；
- 不形成第四份长期 authority。

---

## 十、概念性 Weft Python 语法

### 1. RMSNorm

~~~python
import weft
import weft.language as W

@weft.kernel
def rms_norm(
    x: W.ptr[W.f32],
    weight: W.ptr[W.f32],
    y: W.ptr[W.f32],
    rows: W.i64,
    cols: W.i64,
    stride: W.i64,
    BLOCK: W.constexpr,
):
    row = W.task_id(0)

    sum_sq = W.scalar(0.0, dtype=W.f32)
    for base in W.range(0, cols, BLOCK):
        lane = W.arange(0, BLOCK)
        index = base + lane
        valid = index < cols
        values = W.load(
            x + row * stride + index,
            mask=valid,
            other=0.0,
        )
        sum_sq += W.sum(values * values, axis=0)

    scale = W.rsqrt(sum_sq / W.cast(cols, W.f32))

    for base in W.range(0, cols, BLOCK):
        lane = W.arange(0, BLOCK)
        index = base + lane
        valid = index < cols
        values = W.load(x + row * stride + index, mask=valid, other=0.0)
        weights = W.load(weight + index, mask=valid, other=0.0)
        W.store(y + row * stride + index, values * scale * weights, mask=valid)
~~~

launch grid 是 rows。BLOCK 是 logical block extent，不是 VLEN 或 lane count。

Weft compiler 可以：

- 用 Scalar 实现；
- 每个 BLOCK 内做多次 RVV vl strip；
- 选择 LMUL；
- 使用 deferred-wide vector accumulator；
- 对最后一个 block 选择 AVL/mask/tail；
- 重新物化第一遍 values，而不是强制保存。

### 2. GEMM

~~~python
@weft.kernel
def gemm(
    a: W.ptr[W.f16],
    b: W.ptr[W.f16],
    c: W.ptr[W.f16],
    m: W.i64,
    n: W.i64,
    k: W.i64,
    BM: W.constexpr,
    BN: W.constexpr,
    BK: W.constexpr,
):
    pid_m = W.task_id(0)
    pid_n = W.task_id(1)

    offs_m = pid_m * BM + W.arange(0, BM)
    offs_n = pid_n * BN + W.arange(0, BN)
    acc = W.zeros((BM, BN), dtype=W.f32)

    for k0 in W.range(0, k, BK):
        offs_k = k0 + W.arange(0, BK)
        a_block = W.load(
            a + offs_m[:, None] * k + offs_k[None, :],
            mask=(offs_m[:, None] < m) & (offs_k[None, :] < k),
            other=0.0,
        )
        b_block = W.load(
            b + offs_k[:, None] * n + offs_n[None, :],
            mask=(offs_k[:, None] < k) & (offs_n[None, :] < n),
            other=0.0,
        )
        acc = W.dot(a_block, b_block, acc)

    W.store(
        c + offs_m[:, None] * n + offs_n[None, :],
        W.cast(acc, W.f16),
        mask=(offs_m[:, None] < m) & (offs_n[None, :] < n),
    )
~~~

同一个 W.dot 可以被选择为：

- Scalar FMA nest；
- RVV register-blocked microkernel；
- IME matrix tile；
- RVV/IME mixed path。

bridge 可以从 Intent GEMM 生成这段 Weft program；Weft compiler 不需要知道它来自
Intent contract node。

---

## 十一、Intent→Weft bridge 的职责

bridge 是真正的 target lowering，需要：

- 把 Intent entry ABI 转成 Weft pointer/scalar ABI；
- 把 logical domains 变成 task grid 和 block loops；
- 把 views/index relations 变成 pointer/index arithmetic；
- 把 logical validity 变成 Weft masks；
- 把 reduce/contract 保留成 Weft block reduce/dot anchor；
- 把 state_stream 变成 explicit loop + carried SSA；
- 把 ragged relation 变成 offsets/indices loads、loops 和 masks；
- 把 effects 变成 explicit store/atomic/fence；
- 把 I.auto 变成 Weft meta-parameter 或合法 search space；
- 保留 source location/provenance。

bridge 不负责：

- RVV LMUL、AVL/`vl` 构造与策略；
- IME tile/register；
- Scalar/RVV/IME owner selection；
- target instruction；
- register allocation；
- artifact emission。

如果 bridge 需要按 kernel 名称选择模板，说明 Intent backend 仍然不通用。
如果 Weft core 需要读取 Intent op 才能完成 lowering，说明独立边界失败。

### 1. 三方各自决定什么

这里最容易重新混乱的是“调度到底归谁”。边界必须按语义层级划分，而不是按
仓库或 pass 名划分：

| 层 | 必须决定 | 不得决定 |
|---|---|---|
| Intent | logical domain、数据依赖、reduce/contract/state/ragged/effect 等算法语义 | block size、RVV LMUL、IME tile、具体指令 |
| Intent→Weft bridge | 如何把逻辑轴表达成 task grid、blocked loops、pointer/index、mask 和 block reduce/dot；可以留下有约束的 meta-parameter | RVV `vl`、LMUL、寄存器和指令；不得按 kernel 名套模板 |
| Weft compiler | meta-parameter 的合法实例、physical block layout、VLA strip-mining、Scalar/RVV/IME owner、handoff 和指令实现 | 改写 source-visible 算法、重新猜 Intent 结构或选择另一个数学算法 |

bridge 做的是一次真正的 target schedule lowering：它选择**逻辑 blocked program
的形状**，但可以把 `BLOCK_M/BLOCK_N/BLOCK_K` 等值保留为受约束的编译期参数。
Weft 再根据 target profile 实例化这些参数，并决定 block elements 如何落到动态
`vl`、RVV register groups 或 IME tile。两者都涉及“调度”，但前者是算法到
blocked program，后者是 blocked program 到物理执行，不能混成同一份 plan。

如果某种算法需要完全不同的 traversal、reduction order 或 staging skeleton，
bridge 应生成不同的 Weft program；Weft 不应根据 kernel 名在后端偷偷替换整段
算法。反过来，LMUL、尾部处理或 IME/RVV 混合只是同一 Weft program 的不同合法
实现，应由 Weft 选择。

### 2. bridge 的语义完备判据

转换完成后必须满足：

- 保存出的 `weft.*` module 在不加载 Intent dialect、verifier 或源码的情况下，
  可以独立 parse、verify、optimize 和 compile；
- 若两个 Intent program 生成完全相同的 canonical Weft IR，Weft 从此应无法也
  无需区分它们；
- 任何会影响数值、内存效果、顺序或 ABI 的信息都必须已经进入 Weft IR；
- provenance/source location 只用于诊断，不参与 legality 或 codegen；
- bridge 输出 canonical Weft MLIR，不输出 Python DSL 文本，也不调用另一条
  handwritten-only compiler path。

这几条比“是否同进程”更能判断 Weft 是否真正独立。

---

## 十二、MLIR-native 集成方式

### 1. Weft 导出的组件

Weft repo 应导出：

- Weft Kernel dialect/type/op library；
- Weft verifier；
- selected RISC-V Execution dialect；
- compiler/pass library；
- Python eDSL；
- weft-opt；
- weft-compile；
- artifact/runtime adapter。

weft-compile 的正式输入只有 canonical Weft MLIR。

### 2. Intent side

Intent 侧增加：

- ConvertIntentToWeft pass；
- Weft target option；
- artifact materializer；
- 可选 source diagnostic mapping。

其中 ConvertIntentToWeft bridge library 只链接 Weft Kernel dialect，不直接依赖
RVV/IME lowering；同进程 integration driver 可以再组合独立的 WeftCompiler
library。

### 3. 同进程模式

~~~text
intent.* module
  → ConvertIntentToWeft
  → weft.* module
  → WeftCompiler library
  → artifact
~~~

优点：

- 无文本往返；
- source location 和 MLIR diagnostic 保持最好；
- 直接 dialect conversion；
- 共享 MLIRContext。

要求：

- LLVM/MLIR toolchain 完全一致；
- 依赖方向单向；
- Weft compiler library 有清楚 API；
- Intent target code不进入 Weft core。

### 4. 外部进程模式

~~~text
Intent compiler
  → textual Weft MLIR
  → weft-compile
  → artifact
~~~

优点：

- 两边构建/进程隔离；
- 早期实现最清楚；
- Weft 可以独立调试；
- textual Weft IR 成为可读交接物。

缺点：

- parse/print 开销；
- LLVM/MLIR dialect兼容需要显式管理；
- diagnostic 需要携带 source location；
- artifact contract 需要明确。

建议：

- 第一条 integration repro 使用外部 weft-compile；
- Weft DSL/IR 稳定后再提供同进程 library；
- 两种模式必须产生同一个 canonical Weft IR，不得有两套 compiler path。

---

## 十三、旧代码怎么处理

### 直接退出新主干

- bounded Exec canonical problems；
- old source front doors；
- whole-kernel VariantMaterialization/Selection/Dispatch；
- format/kernel-name formula catalog；
- Typed*PreRealizedBody 作为 canonical source；
- GgmlRepack 作为语言 op；
- empty scalar fallback envelope；
- route string/metadata 决定 compute；
- 旧 test/coverage/certification/campaign authority。

### 可以抽取

- RVV scalable vector、mask、index types；
- setvl/with_vl 的语义和 lowering；
- unit/strided/indexed/segment memory；
- widening/narrowing/reduction primitives；
- IME base MMA/intrinsic/assembly；
- capability provides/implies/conflicts；
- object/header/toolchain packaging；
- fully-legalized/fail-closed emission gate；
- ABI type/ownership validation。

### 只作硬件知识参考

- deferred-wide reduction；
- LMUL/register budget；
- packed decode/codebook/repack；
- q4/q8 scale/min fold；
- IME quant bricks；
- compiler asymmetry和真实硬件负结果。

这些知识以后挂到 Weft block ops 与 selected execution，不迁旧 family identity。

“99%发射都不对”不应理解成 99% 行数要删除，而应理解成：

> 旧 source→problem→variant→body→route 的组织方式整体不能作为新骨架；
> 末端指令、intrinsic、ABI 和 artifact 代码需要逐件重新接线。

---

## 十四、新目录

~~~text
include/Weft/
├── Dialect/
│   ├── Kernel/                 canonical Weft blocked Kernel IR
│   ├── Execution/              selected RISC-V execution core
│   ├── RVV/                    RVV execution/lowering extensions
│   ├── IME/                    IME execution/lowering extensions
│   └── Scalar/                 Scalar execution/lowering extensions
├── Frontend/
│   └── Python/                 Python eDSL → Kernel MLIR
├── Analysis/                   derived/recomputable facts
├── Transforms/                 Kernel → selected Execution
├── Conversion/
│   ├── Scalar/
│   ├── RVV/
│   └── IME/
├── Artifact/
└── Runtime/

lib/Weft/                       与 include 镜像

python/weft/
├── language/
├── frontend/
└── runtime/

tools/
├── weft-opt/
└── weft-compile/
~~~

Weft core 中不出现 Intent、IntentAdapter 或 ConvertIntentToWeft 的实现；这里的
`Conversion/` 只保存 Weft→Scalar/RVV/IME 的目标 lowering。

Intent bridge 位于：

- intentdsl 的 target plugin；或
- 一个独立、同时依赖 IntentIR 与 WeftKernel 的 integration package。

不能位于 Weft core。

---

## 十五、实施顺序

### 阶段 1：独立 Weft Kernel IR

定义最小但真实的：

- kernel/function/ABI；
- task_id/grid；
- block tensor type；
- arange/broadcast/reshape/cast；
- pointer/index；
- mask/load/store；
- pointwise；
- reduction；
- contraction；
- for + carried SSA；
- constexpr/meta-parameter。

Python frontend 直接生成 canonical Weft MLIR，不维护 Python typed IR。

当前已经落地这一阶段的可运行切片：独立 `weft_kernel` dialect、typed pointer、
rank-N dynamic logical block、constexpr、task_id、arange、`expand_dims`、逐轴
singleton broadcast、`splat/full_like/zeros_like`、pointer arithmetic、pointwise、
masked load/store、逐轴 reduce shape projection、显式成对轴的
`weft_kernel.contract`、`W.contract`/`W.dot`、for/yield carried SSA，以及
Python AST→generic MLIR assembly。dynamic extent identity 由 canonical SSA 图派生，
用于验证 broadcast、memory footprint、contract 配对轴和 loop-carried block；裸
`-1 == -1` 不是相等证明。
Python 侧只有与 MLIR operation/value/region 一一对应的瞬态 assembly builder，
不另立 Weft op schema 或 verifier。

尚未落地 reshape/transpose、launch-grid extent contract 和完整控制流。除已经落地的
rank-1 full f32 dot、单 scope rank-2 f32 contract 与 signed widening IME contract 的
物理 `4×4×8` fragment tiling 外，一般/rank-N contract 的 selected operand layout、
更多 IME fragment/microkernel family 与 cleanup handoff尚未实现。
任意 rank-N 已有 pointwise/memory 物理切片：恰好一个逻辑轴进入 RVV VLA 并被放到
traversal 最快端，其余轴按 `order` 形成有序嵌套 serial loops；canonical
`arange/expand_dims/broadcast/pointer` graph 证明 unit 或 runtime affine load stride。
`examples/repro_rank3_affine.py` 的 canonical address graph 让 axis 1 成为唯一 unit-stride
store axis，因而以 `order=[1,2,0], vector_axes=[1]` 走通
selected/source/object。沿 selected vector axis 的 f32 sum/max/min 已实现 layout
projection：selected plan 保留 rank-2 输入 group 和
reduction strategy，rank-1 结果作为每个 serial row 的派生标量发射，不建立零 vector
axis 的伪 RVV group。selected Execution 已加入最小 `layout_conversion` record，只保存
consumer node/operand 与 target group；source value/group/layout 均从 canonical SSA 和
`value_layout` 推导。除 canonical For/Yield 的 layout-preserving carried-state
handoff 外，`examples/repro_dual_axis_sum.py` 已形成第一条普通逐-use conversion：同一
canonical splat 的 axis-1/axis-0 reduction 分属两个 vector-fastest group，第二个 use
通过 incoming-only group 从原 scalar rematerialize。真实 RVV 上以
`tasks=2, rows=19, cols=70, value=1.25` 得到 `max_abs=0`。这不等于任意数据重排；
`examples/repro_dual_axis_load_sum.py` 已进一步允许 direct masked load 在无 intervening
store、target affine pointer 可证明时为第二个 reduction use 重新 load，并分别生成
axis-1 `vle32` 与 axis-0 `vlse32`。selected round-trip/object 已闭合但尚无新真机数值；
register/owner conversion、多 vector axis、strided store/indexed gather、rank>2
reduction/contract 与沿物理 serial axis 归约仍未落地。canonical vector exp 已通过
owner-local `unary_config(strategy="exp_poly_v1")` 发射；三阶段
`examples/repro_softmax.py` 已完成 selected round-trip/source/object，但尚无新主干真机
数值。单组 columnwise sum 已选择
`order=[0,1], vector_axes=[0]` 并发射 masked `vlse32`；真实 RVV 上以
`tasks=2, rows=19, cols=70, stride=77` 得到 `max_abs=0`。因此阶段 1 仍不能视为完整结束。

### 阶段 2：第一条 standalone repro

手写 Weft RMSNorm：

~~~text
Weft Python
  → canonical Weft Kernel MLIR
  → owner-local selected execution
  → RVV source/object 或 IME source
  → 真实 RISC-V 数值对照
~~~

先证明语言和编译器独立成立，不接 Intent。

当前 `weft-compile` 可以把 RMSNorm source 输出的 canonical MLIR 在同一进程内
推进到 selected layout/execution、RVV C++ source 或 RISC-V relocatable object。
本轮再用临时 target harness 在真实 RVV 机器上以 `rows=3, cols=513` 对照标量参考，
得到 `max_abs=0, max_rel=0`。新 compiler 与两条 selected-execution route 不经过旧
Exec/Variant/route；旧 route 仍在仓库并行注册。这个结果证明阶段 2 的首个独立切片
成立，但仓库 example 本身仍只打印 canonical MLIR，也不代表 rank-N、一般 IME、
header/runtime launch 或完整调优系统已经完成。

第二个结构化 kernel `examples/repro_rank2_affine.py` 已通过同一 compiler path：

~~~text
rank-2 expand_dims + singleton broadcast + row-major pointer graph
  → #weft_layout.blocked<rank=2, order=[1,0], vector_axes=[1]>
  → serial row loop + VLA RVV column strip
  → RISC-V object
  → 真实 RVV 运行（rows=5, cols=513, stride=520, max_abs=0）
~~~

这条路径仍按 canonical SSA 结构选择和发射，没有 `rank2_affine` 名称分支。

第三个结构化 kernel `examples/repro_rowwise_sum.py` 在相同 rank-2 layout 上加入
`reduce(axis=1)` 与 rank-1 serial store。真实 RVV 运行覆盖两个 task、每个
`rows=5, cols=513`，得到 `max_abs=0`；selector/emitter 没有新增 rowwise-sum 名称
分支。

第四个算法 source `examples/repro_contract.py` 的单 scope rank-2 f32 contraction 已走通
canonical→selected→RVV C++/object。`W.contract(lhs, rhs, 0.0,
lhs_axes=(-1,), rhs_axes=(0,))` 保留为单个 `weft_kernel.contract`；输出 shape 由未收缩
lhs axes 接未收缩 rhs axes 推导。selector 建立 distinct lhs/rhs/result value-local
groups，emitter 产生 `M serial → N VLA strip → K sequential` outer-product；真实 RVV
上以 `tasks=2, M=5, N=37, K=29` 得到 `max_abs=0, max_rel=0`。这个结果只覆盖单
scope 严格切片，不代表通用 contraction 已完成。

第五个算法 source `examples/repro_vector_dot.py` 使用相同 canonical ContractOp 的
full rank-1 f32 子集，已由结构驱动 selector 生成 `contract_config`，再 lower 为 VLA
`vfmul + vfredusum` 与 scalar store。真实 RVV 运行覆盖三个 task、每个
`count=1027`，得到 `max_abs=0`。这只证明 rank-1 full contraction，不外推为 GEMM。

### 阶段 3：从 direct contraction 到 tiled GEMM 与 IME

当前已进一步落地：

- direct rank-2 RVV contraction 的 block-carried accumulator handoff 与 K0/BK
  algorithmic loop realization；
- For init/body argument/yield/result 之间按 canonical use-edge 记录的物理 handoff；
- `BM=BN=BK=32, tasks=2×3, M=37, N=70, K=45` 的真实 RVV 数值对照，
  `max_abs=0, max_rel=0`；
- 按 canonical constexpr 参数独立选择并持久化的 meta binding；使用
  `BM=8, BN=32, BK=16, tasks=3×3, M=19, N=70, K=45` 再次真实运行，覆盖
  M/N/K 尾部并得到 `max_abs=0, max_rel=0`；
- canonical signedness 与 `si8×si8→si32` widening contract；
- 独立 `weft_ime_execution` owner：每个 canonical contract site 的一个 config 共同拥有
  lhs/rhs/result 三个 shared group，只持久化物理 `4×4×8` fragment、VLEN=256 与
  `fragment_tiled` strategy；
- 结构驱动 IME selector/source emitter：任意正的 selected logical M/N/K 被切成
  `ceil(M/4)×ceil(N/4)×ceil(K/8)`，按 canonical SSA 标量化 pointer/mask，完成
  tail-safe scratch pack、register-resident K-loop、真实 `vmadot` leaf 与 masked scatter；
  同一 flat kernel 的多个不重叠 contract site 独立成立，不再要求整个 kernel 匹配
  “两个 load/一个 contract/一个 store”。selected MLIR 已 round-trip，source 已通过
  RISC-V clang syntax check；没有新主干 IME object/真机结论。

仍需加入：

- 更一般的 operand layout relation 与按 canonical use-edge 的 layout conversion；
- 更多 IME instruction/fragment family 与显式 fragment lane layout relation；
- block transpose/broadcast；
- RVV microkernel；
- IME object/toolchain、真实目标运行与跨 owner cleanup；
- Scalar/RVV cleanup；
- BM/BN/BK tuning。

如果实现依赖 gemm kernel 名，架构失败。

当前 `examples/repro_gemm.py` 已经形成并发射
`splat → for(init/body-arg/yield/result carried block) → contract → store` 的 canonical
MLIR。RVV source 保持这条 SSA 状态与 source 中的 `range(0, K, BK)`；没有调用一个
自造完整 M/N/K 循环的 whole-kernel emitter 来替换 canonical algorithm skeleton。
命令行可用重复的 `--meta NAME=VALUE` 按 canonical `arg_names` 独立选择 BM/BN/BK；
selected plan 仍只保存参数索引和值，发射器不解释参数名称。`--block-elements=N`
只作为显式统一绑定简写，两种输入互斥且没有隐藏默认值。当前尚未实现的是从候选
空间自动搜索这些值，而不是表达或发射独立值的能力。

### 阶段 4：Intent bridge

让 Intent 的 RMSNorm/GEMM backend 生成与手写入口相同的 canonical Weft IR。

验收判据：

- Weft core 零 intent.* 依赖；
- generated Weft IR 可以单独保存、修改、编译；
- 手写与 Intent-generated 走同一 compiler path；
- bridge 只改 Intent side/integration package；
- 新增 bridge 不修改 RVV/IME emitter。

### 阶段 5：control/state/ragged/effects

按 Weft 自身通用能力推进：

- explicit carried loop；
- gather/scatter；
- atomic/fence；
- dynamic/ragged pointer loops；
- multi-stage blocked program。

Intent state_stream/ragged 只在 bridge 中 lowering，不进入 Weft core 词汇。

### 阶段 6：旧主干删除

新 Weft DSL 可以独立产生真实 artifact 后，删除旧：

- problem front doors；
- variants/dispatch；
- format bodies；
- route graph；
- 只为旧 authority 服务的 metadata。

不保留 compatibility production path。

---

## 十六、唯一验证方式

不迁旧 test，不建立新 test 目录，不累计 coverage、边界或 certification。

只保留少量能代表算法结构的手工 repro，不为边界枚举累计测试 corpus。每轮相关
repro 走下列真实链路，尚无 owner 的 canonical feature 则诚实停在 verified MLIR：

~~~text
Weft source
  → canonical Weft MLIR
  → selected execution
  → RISC-V source/object
  → 真实目标运行
  → 数值对照
~~~

Intent 集成阶段再把入口替换为：

~~~text
Intent source
  → generated canonical Weft MLIR
  → 同一后半条 repro
~~~

没有真实 RISC-V 机器或相应扩展工具链时，只能诚实停在 selected MLIR、source 或
实际能够生成的 object；这些停点不得写成 runtime correctness。

---

## 十七、拒绝标准

出现以下任一情况，说明架构正在走回旧路：

1. weft-compile 的正式输入包含 intent.*；
2. Weft core 链接 Intent dialect/verifier；
3. Weft op 按 softmax/gemm/q4_K 等 kernel/format 命名；
4. Intent bridge 按 kernel 名套整段 Weft 模板；
5. Weft source 不可脱离 Intent 单独编译；
6. 手写 Weft 与 Intent-generated Weft 走两套 lowering；
7. W.arange/BLOCK 被当成 fixed RVV lane/VLEN；
8. IME 被压成 RVV LMUL 的一种；
9. emitter 从 kernel 名、route string 或 metadata 猜 compute；
10. selected execution 与 Kernel IR 同时可修改算法；
11. 为兼容旧 artifact 保留 problem/variant 旁路；
12. 在第一条真实 repro 前先造庞大 plugin、测试或治理基建。

---

## 最终回答

### Weft 应该单独做成 DSL 吗？

应该。

只有成为独立 blocked kernel DSL，Weft 才能真正像 Triton：

- 可手写；
- 可由上游生成；
- 可独立优化和调试；
- 不依赖某一个上游 IR；
- 对 RISC-V hardware model 拥有完整控制；
- 可以被 Intent 纳入，而不是寄生在 Intent 内部。

### 直接嵌入有没有价值？

有技术价值，但不应成为语义架构。

同进程 MLIR conversion、共享 MLIRContext、直接 pass pipeline 都可以保留；它们只是
部署方式。只要 canonical Weft Kernel IR 边界存在、依赖方向是 Intent→Weft，
同进程并不等于嵌入式混乱。

### 最合理的形态是什么？

> standalone Weft language/compiler + MLIR-native Intent bridge。

Intent 负责编译出 Weft DSL/IR；Weft 负责把这个 blocked/VLA program lowering 到
Scalar、RVV、IME。两边各自完整，边界清楚，又不放弃 MLIR 的直接 lowering。
