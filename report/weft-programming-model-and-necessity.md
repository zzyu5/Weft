# Weft 最终编程模型：必要性、用户接口与编译边界

## 报告结论

Weft 不应被定义为“RISC-V 版 Triton”，也不应被定义为“隐藏了 RVV intrinsic 的 C 语法”。本报告冻结的语义模型是：

> **Weft 是一门面向单个 CPU worker/hart 的有序 kernel 语言。作者通过显式 control、addressable storage/view、storage lifetime、data movement、state 与目标无关的局部计算操作组织完整的 core-local algorithm；需要新局部数据流时可以下降到 runtime VLA、logical axis 与 block SSA；编译器只在这些显式语义授权的边界内生成 SIMD、register microkernel、memory schedule、local pipeline 与 RISC-V extension realization。**

该定义的核心不是“工作如何分配”，而是 layout 发生之前，源程序已经正式说明了什么：

```text
ordered worker control
+ runtime scalable VLA domain
+ axis-identified logical block values
+ caller-visible storage ownership/lifetime
+ explicit state algebra/effect order
+ target-independent local semantic operations
```

Weft 的独立性不是技术上的绝对必然。Triton-CPU、TileLang 的 CPU target 都可以继续扩展来承载类似程序。当前选择独立 Weft 的工程判断是：若要把上述内容变成可由 compiler 消费的正式语义，而不是 raw-pointer convention，需要协调改变 frontend、type/value model、layout、storage、ABI、runtime 与 verifier；这不是只增加一个 CPU/RVV codegen backend。这个判断不是“不可能复用”的理论下界，而是对需要改变哪些层的明确核算。复用它们的编程思想与编译技术是正确的，直接沿用其现有 canonical contract 则不能自动得到 Weft 所需的语义。

这项选择有一个明确的可证伪条件：

> **如果 runtime VLA、axis identity、ordinary block/state lifetime、workspace/persistent contract 与 local extension semantics 只存在于文档或 verifier，而没有真实影响 physical mapping、handoff、resource、schedule 和 artifact，那么 Weft 没有独立建语言的必要，应退回 Triton-CPU/TileLang-CPU 路线。**

因此，本报告不是靠重新命名保护现有设计，而是同时定义成立条件、失败条件与当前缺口。

### 本文的状态标签

为避免把设计目标冒充成当前实现，全文使用以下四种状态：

| 标签 | 含义 |
|---|---|
| **当前事实** | 已由现有 Python DSL、canonical Kernel IR、RISC-V lowering 或生成接口承载 |
| **冻结语义** | 本报告确定且后续实现不得随意改变的编程模型与权责边界 |
| **拟议表面** | 为降低书写负担而建议的 Python spelling/source helper；尚不是当前能力，且不得增加第二份 canonical authority |
| **未决项** | 当前证据不足，不纳入冻结模型；真实 kernel 证明需要后再决定 |

其中 `Buffer/View/read/write/copy` 是下文采用的**概念名称和拟议表面 spelling**。冻结的是“有地址 storage → logical view → 无地址 SSA value”的语义分层，不要求 Kernel IR 新增同名对象：当前 typed pointer、`W.storage`、axis、pointer/index、validity 与 `W.load/store` 可以承载其 canonical 含义。受限 lexical scratch 则仍是未决项，不属于本次冻结结果。

---

## 一、首先排除三条错误论据

### 1. Triton 不是只能一 program 处理一个 tile

Triton 的典型教程使用 launch grid，让每个 program instance 处理一个输出 tile；但官方 persistent-kernel 教程明确让一个 program 在外层循环中处理多个 tile，并复用 MMA state 与 buffer。因此：

```text
Triton program 通常处理一个 tile
```

是常见调度，不是语言不可突破的边界。[Triton Persistent Kernels](https://triton-lang.org/main/getting-started/tutorials/gluon/persistence.html)

所以 Weft 不能仅以“一个 worker 处理多个 block”证明独立性。

更强的反例是：若Weft最终只有persistent loop、fixed block values和dot/matmul，那么它只是Triton persistent program换了CPU ABI。跨block accumulator/state本身也不构成新语言；只有当runtime VLA、axis identity、ordered effect以及storage lifetime共同成为canonical contract并被physical compiler消费时，差异才成立。

### 2. CPU/RVV codegen 本身不要求新语言

官方实验性 Triton-CPU 仓库及其示例已经足以证明：为 Triton 增加 CPU backend 是现实路线，而不是概念上的不可能。[Triton-CPU](https://github.com/triton-lang/triton-cpu) 这项证据只证明路线存在；它不证明当前仓库已经完整覆盖 Weft 所需的 reduction、quant、ordered state、RVV 或 IME 语义。

因此：

```text
我要生成 RVV
→ 所以必须重新发明 DSL
```

这个推理不成立。

### 3. “能写出来”不等于“是语言语义”

Triton 已经有 typed pointer、block tensor 和 tensor descriptor；C 与 Triton 也都可以用 pointer、循环和 helper 编码 persistent weight、workspace、state carry 与 packed quant。问题不在“有没有比 raw pointer 更高的类型”，而在这些现有对象是否共同、正式地承载以下完整合同，使 verifier、layout 与 target lowering都能消费：

- pointer 是 external、persistent 还是 worker workspace；
- packed bytes 具有什么 format identity；
- 两个相同 extent 的维度是不是同一个 logical axis；
- 某循环是 ordered recurrence 还是可重结合 reduction；
- 一个 runtime domain 是否允许 arbitrary VLA strip-mining；
- state/value 跨哪些 loop 与 control boundary 存活；
- 某个 i4×i8 relation 是普通 bitwise graph 还是显式 local semantic primitive。

如果这些只靠名字、注释或调用者约定，verifier、physical planner 与 extension selector都不能可靠消费。Weft 的必要性必须建立在“正式语义合同”，而不是“语法上可以写”。

---

## 二、需要解决的真正问题

当前 Weft 已经拥有 scalar control、`W.vla`、`W.block`、memory、state、matmul 与 quant primitive，但它们在 public surface 中几乎平级出现：

```text
W.range
W.vla
W.block
W.load/store
W.storage
W.matmul
W.reduce/scan
typed quant/extension operations
```

这使语言看起来像“C control + vector region + tile intrinsic + storage annotation + extension builtins”的拼接，而不是一个自然的 CPU kernel model。

问题不是这些语义本身都错，而是缺少连接它们的用户对象与接口层级。尤其缺少：

```text
Buffer：什么是有地址的存储对象
View：当前操作的是 Buffer 上哪块逻辑区域
Value：从 View 读取或由计算产生的无地址 SSA block
Operation：对 Value 做何种局部语义计算
```

若没有这条链，所谓“普通用户只调用 matmul”没有意义：matmul 的 operand 是整张矩阵、raw pointer、固定 tile、staged buffer，还是 packed view，并不清楚。

---

## 三、最终根执行模型

### 3.1 一个 kernel 是什么

一份 Weft kernel 是一个普通 AOT callable，由当前 worker/hart 从入口持续执行到返回：

```text
external runtime
→ 选择 worker-local work slice
→ 调用 generated C/object entry
→ 当前 hart 执行 ordered control + local parallel domains + local operations
```

Canonical language 中没有隐式：

- grid；
- `program_id`；
- hart ID；
- task identity；
- thread block；
- warp；
- 多 worker barrier。

应用若需要 row range、tile coordinate、expert range 或 ragged descriptor，必须作为普通 entry 参数传入。线程池、OpenMP、NUMA、affinity 与多核工作分配属于外部 runtime。

### 3.2 Author-owned ordered control

作者拥有：

- scalar `for / while / if`；
- loop order；
- cache/algorithmic blocking；
- staging 与 recomputation；
- outer K/key/token traversal；
- state transition；
- author-visible effect order；
- persistent packing 与算法 variant。

普通 scalar loop 默认是有序程序。Compiler 不得因为一段 loop 看起来“可并行”，就自动将其改为 VLA、reduce、scan、matmul 或另一个算法。

### 3.3 Explicit local reorganization authority

只有显式构造授权 target 重组：

| 显式构造 | 授权范围 |
|---|---|
| VLA/map logical domain | 当前 logical elements 可被动态 strip-mine/SIMD 化 |
| reduce | 按声明的 algebra/order 聚合当前 axis |
| scan | 按声明顺序产生 prefix |
| typed summary | 重组该 summary algebra，不拥有外围 traversal |
| dot/matmul | 重组当前 local reduction/product domain |
| lookup/decode/quant | 实现当前完整局部 byte/numerical relation |
| extension semantic op | 实现新增的局部可观察语义 |

Target 不从普通 multiply/add 猜 matmul，不从 max/exp/sum 猜 online softmax，也不从 bitwise graph猜 packed format。

### 3.4 Runtime VLA的精确定义

`W.vla(begin, end)`不是“普通loop交给auto-vectorizer”，也不是一个固定长度tile。它表示作者显式授权：`[begin,end)`中的logical elements可被target按任意合法strip宽度实现，logical结果不得依赖一次取了多少lane。当前RISC-V target profile仍在compile time明确VLEN（例如128或256 bit）；runtime变化的是logical extent与tail处的`vl`，而不是在同一object中动态发现未知硬件VLEN。

因此：

- source不能读取`vl`并据此改变算法结果；
- 同一source可针对不同target profile重新编译成不同lane/LMUL mapping；
- state若跨strip存活，必须由显式state/carry语义定义；
- scalar `W.range`没有这项授权，compiler不能自动升级；
- 当前实现禁止nested VLA，这是明确能力边界，不应被“scalable”一词掩盖。

---

## 四、四种核心语义对象

### 4.1 Buffer：有地址的 storage object（冻结语义；拟议表面）

`Buffer` 是本文对真实可寻址 storage 合同的概念名称。当前实现仍以 typed pointer、entry 参数与 `W.storage` 表达；建议的 `W.Buffer[...]` 只是其 source wrapper，不是另一种持久 IR。该合同至少具有：

```text
element type
rank/shape
strides or explicit address relation
alignment
alias/access qualifier
ownership/lifetime
optional persistent format identity
```

拟议 Python 表面可以使用 descriptor：

```python
A: W.Buffer[W.f16, shape=(M, K), readonly=True, noalias=True]
C: W.Buffer[W.f32, shape=(M, N), writeonly=True, noalias=True]
```

Frontend 必须把它机械展开为现有 canonical pointer、shape、stride、storage ownership 与 alias facts；生成的 AOT ABI 再展平为 C 参数，不要求运行期存在 Python/Weft object。

Raw pointer 与显式 pointer arithmetic 继续作为 expert surface，但不应是所有普通用户唯一的 memory interface。

### 4.2 View：Buffer 上的逻辑区域（冻结语义；拟议表面）

`View` 是非 owning logical region 的概念名称：

```text
base Buffer
+ logical axes
+ offset
+ extent
+ stride/address map
+ validity/bounds
```

拟议 Python spelling：

```python
m = W.tile(m0, BM, bound=M)
k = W.tile(k0, BK, bound=K)
a_view = A.view(m, k)
```

每次 tile/view construction 必须建立真实 logical axis identity。即使两个 axis extent 都是 32，也不能因为 shape 相等就被 compiler 合并。

本项目应作出单一选择：`View` 是 source sugar，frontend 必须把它完整展开为 canonical axis identity、pointer/index、extent/stride 与 validity。Kernel IR 不再并列保存一份独立 view authority；backend 也不得从 shape 或 source helper 名反推这些事实。

### 4.3 Block Value：无地址的局部 SSA value（当前事实；冻结语义）

从 View 读取后产生 Block Value：

```python
a = W.read(a_view, fill=0)
```

Block Value：

- 具有 dtype、shape、axis identity 与 validity；
- 没有 source-level physical address；
- 可以由 read、full/zeros、pointwise、reduce、matmul、decode 等产生；
- 可以多 consumer；
- 可以跨 `for/while/if` carry；
- 可以进入 state、memory 或另一个 local op；
- 不等于 cache block、register microtile、RVV vector 或 IME fragment。

Compiler 决定它是保持 register、reload、rematerialize、shuffle、local pack 还是必要 spill。

### 4.4 State：由 control 与 algebra 定义的长期 value（当前事实；冻结语义）

State 不是独立硬件 storage 类别。它可以是 scalar、Block Value 或 addressable Buffer 内容；其语义来自：

- 创建位置；
- loop/control carry；
- ordinary use-def；
- reduce/scan/summary algebra；
- effect order；
- external/workspace/persistent ownership。

普通 sequential carry、reduce、scan 与 typed summary 必须继续是不同语义，不能因为底层实现共享而合并。

### 4.5 数据与计算主链

拟议表面必须展开到当前 canonical 语义，因此最终用户模型与 compiler 输入之间只有这一条链：

```text
Buffer
  ↓ view/slice
View
  ↓ read
Block SSA Value
  ↓ pointwise / reduce / matmul / quant local op
Block SSA Value
  ↓ write
View
  ↓
Buffer
```

Algorithmic materialization则是：

```text
source View
  ↓ explicit copy/stage
destination Buffer/View
```

---

## 五、CPU storage 与 allocation 模型

TileLang 的 `alloc_shared` 和 `alloc_fragment` 暴露 GPU thread-block/shared/register-fragment hierarchy。这对 GPU 合理，但 CPU 不应照搬成 `alloc_l1`、`alloc_vector_register` 或 `alloc_ime_fragment`。[TileLang Language Basics](https://www.tilelang.com/programming_guides/language_basics.html)

CPU 作者应该表达 addressability、ownership 与 lifetime，而不是物理 cache/register placement。

### 5.1 External Buffer

由 caller 持有的输入、输出或普通跨调用 state：

```text
lifetime：至少覆盖当前调用
ABI：是
format：普通 element/stride contract
```

### 5.2 Persistent Buffer

由 caller/部署系统跨调用保存的 representation：

```text
lifetime：跨调用
ABI：是
必须有稳定 format identity
```

例如：

```python
packed_weight: W.PersistentBuffer[
    W.u8,
    shape=(column_blocks, k_blocks, 288),
    format="q4_0_n16_k32_288b",
]
```

Persistent format是 caller 与 kernel 的 ABI，不是 LMUL、register layout 或 target candidate。当前实现已把`storage_format`与shape/extent metadata写入pointer/header，并由caller按生成query分配；尚不存在跨kernel的builder-consumer registry。冻结的ABI规则是：format identity最终必须足以固定byte/bit layout、endianness、element/group shape、scale/zero/minimum/correction语义、alignment以及必要padding，不能只依赖模糊格式名。拟议builder与consumer应声明同一个identity和shape，能在frontend/build边界判断的冲突应直接拒绝；runtime内容正确性仍由caller负责。这里不引入版本号、checksum或同步机制。Builder应是另一份显式algorithm recipe/kernel，而不是consumer lowering偷做repack。

### 5.3 Caller Workspace

由 caller 为当前 worker/invocation 提供的大型或 runtime-sized scratch：

```text
lifetime：一次 invocation，可跨 loop/primitive
ABI：是
ownership：当前 worker 独占
```

适用于：

- activation quantization buffer；
- 大型 attention accumulator；
- runtime-size staging；
- 需要 caller 控制内存预算的 scratch。

### 5.4 Lexical Scratch（未决项，不冻结）

当前模型不增加 `W.scratch`。作者可寻址、跨 operation 或跨 loop 存活的临时存储继续由 caller workspace 明确提供；无地址 block SSA 和 primitive-private temporary 不进入 ABI。

受限 lexical scratch 的确可能减少小型静态 staging 对 entry ABI 的污染，但现有真实 kernel 尚未证明“caller workspace + SSA/private temporary”无法自然表达必要算法。现在加入它会同时引入 stack budget、noescape、动态大小、递归/调用、alignment 与资源失败合同；这不是一个可以凭便利性冻结的构造。

只有同时出现以下证据，才重新讨论该构造：

- 真实 kernel 需要一个作者可寻址、跨多个 local operation 存活的静态小对象；
- 将它做成 caller workspace 会实质破坏合理 ABI，而不是仅多一个参数；
- block SSA 或 primitive-private temporary 无法表达其可观察 address/alias关系；
- target 能提供明确的静态上界、noescape和失败语义。

若未来满足这些条件，允许的设计也只能是 static/meta bounded、noescape、resource-checked，并且不承诺 L1、register 或某种硬件 storage。现在不把它写入 public surface、Kernel IR 或冻结原则。

### 5.5 Block SSA 与 primitive-private temporary

两者都不是 addressable Buffer：

| 对象 | 作者可见 | 可寻址 | lifetime owner |
|---|---:|---:|---|
| Block SSA value | 是 | 否 | lexical/use-def + compiler placement |
| primitive-private temporary | 否 | 否 | selected local realization |

Register accumulator 应是 Block SSA，不应通过 `alloc_fragment` 暴露。Decode temporary、short pack 与 IME fragment 应是 primitive-private physical state。

---

## 六、CPU data movement 应如何表达

### 6.1 Read/Write（拟议 source spelling）

`read`/`write` 是现有 `W.load/store` 的拟议 source spelling，不增加另一种 canonical memory op：

```python
value = W.read(view, fill=0)
W.write(view, value, where=predicate)
```

它们只表达 logical memory relation，不承诺：

- vector instruction；
- cache level；
- register placement；
- segment/gather form；
- async behavior。

Compiler 根据canonical axes/strides/index/predicate选择unit、strided、indexed或segment realization。只有作者本来写的是scalar value/control时才生成scalar memory；显式VLA或block operation没有合法vector realization时必须unsupported，不能把scalar当作静默fallback。

### 6.2 Copy（拟议 source helper）

可以提供窄而明确的 source helper：

```python
W.copy(dst=stage.view(...), src=A.view(...), where=valid)
```

语义是同步、保持element value/representation的bulk transfer。Source与destination必须具有相同dtype/representation、显式extent/predicate，并满足不重叠或明确的`noalias`前提；`copy`不是`memmove`，重叠时直接拒绝。首个可实现表面只覆盖当前canonical语义能无歧义表达的单VLA或有界静态block copy，并在frontend展开为既有read/write；runtime-sized rank-2、嵌套VLA或需要reshape/transpose的copy在相应canonical能力完成前明确unsupported。不能为了让helper“看起来完整”引入opaque copy IR或第二套lowering。

Compiler 可以 vectorize、group、hoist address、选择 memory instruction；它不能：

- 隐式分配 destination；
- 隐式改变 dtype；
- 隐式 pack；
- 隐式 async；
- 隐式创建 barrier或double buffer；
- 宣称数据进入某级 cache。

### 6.3 Stage

`stage`不是新的神秘 storage op，而是作者明确选择：

```text
把 source value materialize 到指定 caller workspace
并在后续 loop/primitive 中复用
```

它可以由 `read/cast/write` 或 `copy` 组成。Destination、shape、lifetime和representation必须显式。

### 6.4 Pack

Persistent或workspace packing改变可观察 representation，必须是显式 algorithm operation/builder recipe：

```text
source layout
→ scale/zero-point/codebook/bit packing
→ destination format
```

Primitive 内部的短生命周期 register pack仍属于 target realization，不进入 DSL。

### 6.5 Prefetch

Prefetch没有 correctness effect，应留在 physical planner/backend config。当前没有多个真实 realization时，不应提前增加 public API。

### 6.6 Segment/Interleaved

`segment load`通常是机器 realization，不应平铺成默认 local semantic op。作者通过 View/stride/field relation表达 AoS/interleaved data；compiler决定使用 segment instruction。

只有当数据关系具有独立可观察语义时，才增加 typed operation。

---

## 七、Pipeline 的最终边界

TileLang 的 `T.Pipelined` 同时服务 GPU async copy、shared buffers、barrier 与 tensor-core pipeline；专家还可指定 stage/order。[TileLang Software Pipeline](https://www.tilelang.com/programming_guides/software_pipeline.html)

Weft 的普通作者只写显式 loop：

```python
for k0 in W.range(0, K, BK):
    a = W.read(A.view(...))
    b = W.read(B.view(...))
    acc = W.matmul(a, b, init=acc)
```

Compiler 根据：

```text
use-def
loop invariance
alias/effect
state order
lifetime
resource
```

最终目标是在作者显式loop内生成合法的load/compute overlap、unroll、single/double buffering与prologue/steady/epilogue。当前compiler已有dot/matmul等primitive-local的unroll、pipeline-buffer候选和资源过滤；跨`read → decode/cast → compute → write`的通用loop-local scheduler仍未闭合，不能把局部candidate称为完整自动pipeline。

“合法”必须有具体含义。Compiler只可提前或重排纯计算，以及在alias/effect facts证明安全时的readonly load、address calculation、cast、decode与primitive-private pack；移动范围只能位于作者已经写出的同一loop和control-dominance边界内。以下内容不得跨越作者可观察顺序：

- write、atomic、fence或其他external effect；
- 与write存在未知alias的read；
- ordered state update、sequential carry或不可重结合summary；
- control-dependent effect；
- 会改变异常、bounds或predicate可观察行为的操作。

因此，未来共享scheduler中的“自动pipeline”不是把ordered loop改写成另一种算法，而是在已证明无观察差异的局部依赖图上安排compiler-private temporary。证明不了就不生成该candidate；当前尚无通用scheduler的operation则明确保留为能力缺口，不能用silent fallback或文档声明掩盖。

必须区分：

```text
compiler-local register pipeline
→ physical decision，用户不见

algorithm-visible workspace/ring buffering
→ 作者显式写 Buffer/View/copy/control

pipeline depth/prefetch distance
→ backend config/build-time tuning
```

当前不应把 `num_stages`、stage number、register group 或 prefetch distance加入 canonical DSL。只有真实 workload证明 compiler无法安全确定某种非语义授权时，才讨论最小 hint。

---

## 八、三种库必须严格分开

“高性能模板库”不是一种东西。Weft 需要三种完全不同的复用层。

### 8.1 Algorithm/Recipe Source Library

这是作者显式选择的 DSL implementation recipe，例如：

```python
W.algorithms.blocked_matmul(...)
W.algorithms.online_attention(...)
W.algorithms.q4_projection(...)
```

它可以拥有：

- entry 参数与 ABI；
- outer traversal；
- BM/BN/BK；
- staging/workspace；
- persistent layout；
- state algorithm；
- epilogue；
- 完整 algorithm variant。

它必须是可展开的 source helper。展开必须保留并显式产生全部axis identity、storage ownership/lifetime、pointer/index、validity、effect、state和local-op facts；不能只展开成一个opaque call或遗失ABI关系。展开后target lowering只看canonical Kernel IR，不读取recipe名，不按helper名选择实现。

这不违反“作者拥有算法”：作者通过调用某个 recipe，显式选择了这份 algorithm。

### 8.2 Canonical Local Semantic Operation Library

例如：

```text
pointwise/map
reduce/scan/typed summary
dot/matmul
lookup/decode
typed quantized compute
```

它拥有：

- typed operands/results；
- logical axes；
- local numerical/byte relation；
- validity；
- init/state algebra；
- rounding/saturation/reassociation等精确数值合同。

它不能拥有：

- outer loop；
- cache blocking；
- workspace/persistent ABI；
- 完整 state machine；
- target/ISA选择。

这层形成 canonical Kernel IR op。

### 8.3 Target Realization/Microkernel Schema Library

例如：

```text
RVV outer-product matmul
RVV reduction-lane matmul
RVV widening FMA
RVV packed decode+dot
IME i4×i8 fragment
segment/gather memory
```

它拥有：

- physical layout；
- LMUL/SEW；
- register microtile；
- multiple accumulators；
- operand reuse window；
- K-unroll；
- local pack；
- compiler-private pipeline temporary与schedule；
- fragment/resource；
- intrinsic/asm spelling。

它不能拥有：

- entry ABI；
- outer traversal；
- author workspace；
- persistent format builder；
- complete GEMM/attention/quant kernel；
- kernel-name/format-name route。

### 8.4 三者的信息流

```text
作者选择 algorithm recipe
        ↓ source expansion
worker control + Buffer/View/storage + local semantic ops
        ↓ canonical Kernel IR
target realization schemas
        ↓ physical candidate/selection
intrinsic C / local asm
```

只有 canonical Kernel IR 是持久 compiler authority；source helper 与 target candidate都不能形成第二份长期 IR。

### 8.5 “Weft为什么更粗，是否应继续做细”的最终答案

Weft在**local semantic op**层有意比裸FMA/bit operation粗，在**physical compiler**层必须比当前实现更细。这不是折中措辞，而是两种粒度服务不同责任：

| 例子 | 应放在哪一层 | 原因 |
|---|---|---|
| blocked GEMM、online attention、q4 projection完整算法 | algorithm recipe/作者程序 | 拥有outer traversal、staging、packing或state variant，不能成为backend op |
| `[M,K]×[K,N]+init` | canonical `matmul` | 局部、完整、可观察的数值关系；授权reduction/microkernel/IME重组 |
| packed i4×i8 affine dot | typed local quant op | byte/scale/correction关系无法由普通SSA无歧义恢复，但不拥有外围loop |
| add、cast、predicate、read/store | ordinary canonical ops | 普通value/data effect，沿同一axis/layout传播 |
| LMUL、MR×NR、accumulator tuple、decode chunk | transient physical mapping | 是机器结构选择，不是作者算法语义 |
| `vle32_m4`、IME MMA opcode | target leaf/spelling | 是具体ISA，不进入portable DSL |

把`matmul`继续拆成普通multiply/add/reduce，会丢失作者对局部product的明确授权，compiler只能重新做pattern discovery；把整个GEMM或q4 projection塞进一个op，又会夺走作者的outer control和storage ownership。正确粒度就是：

> **canonical op覆盖一个局部且完整的可观察关系；op结果必须是ordinary SSA；外围control/storage不属于op；机器layout比op更细并全部留在transient lowering。**

新primitive只有同时满足以下条件才可进入canonical IR：

1. 现有ordinary ops无法无歧义表达其可观察数值、byte、order或effect语义；
2. 语义局限于当前operands/results，不拥有entry、outer traversal或persistent allocation；
3. 结果可以普通multi-use、carry、store或进入另一个合法op；
4. target可以仅替换其local realization，而不替换作者算法；
5. 名称描述语义关系，不描述kernel、格式route、VLEN或硬件指令。

性能不足时先按这一边界归因：缺blocking/staging/persistent layout是DSL/recipe问题；已有local relation的lane/register/fragment、reuse、pipeline与resource不够细，是compiler内部问题；只有可观察语义确实缺失时才新增DSL primitive。不能因为当前layout compiler较粗，就把LMUL/fragment泄漏给用户。

---

## 九、三种用户层级，而不是三种语言

这是拟议source-interface层级：当前仓库主要实现的是kernel/expert层的低层spelling，尚未提供完整recipe与Buffer/View表面。冻结的是所有层级必须展开到同一canonical semantics，而不是下面的函数名已经可用。

### 9.1 Application/Recipe User

可以选择现成 algorithm recipe：

```python
W.algorithms.blocked_matmul(
    A, B, C,
    work=rows,
    blocks=(BM, BN, BK),
    epilogue=activation,
)
```

不需要直接书写 VLA、axis或pointer arithmetic。

### 9.2 Kernel/Algorithm Author

主要使用：

```text
worker control
Buffer/View/tile
read/write/copy
workspace/persistent
state
matmul/reduce/scan/quant operations
```

### 9.3 Expert Local-Program Author

需要新局部结构时下降使用：

```text
explicit runtime VLA
explicit axis/block constructors
raw pointer/index arithmetic
masked values
typed summary/state composition
custom local operation composition
```

### 9.4 Backend Author

只接触：

```text
physical layout
reuse/resource/schedule
RVV/IME capability
intrinsic/asm
```

这些是接口层级，不是四条lowering route。实现这些拟议表面时，前三层必须在frontend后汇入同一canonical Kernel IR；当前target不得提前依赖尚不存在的recipe/helper名字。

---

## 十、用 GEMM 走完整模型

### 10.1 Algorithm recipe 层

普通使用者可以调用可展开 recipe：

```python
W.algorithms.blocked_matmul(A, B, C, work=rows, blocks=(BM, BN, BK))
```

作者通过选择 recipe 接受其 M/N/K traversal、blocking与staging策略。

### 10.2 Worker DSL 层

下面是**拟议表面**的概念写法；它必须展开到当前唯一canonical IR，不代表这些Python API已经实现：

```python
@W.kernel
def gemm_worker(A, B, C, work, M, N, K, BM, BN, BK):
    for m0 in W.range(work.begin, work.end, BM):
        for n0 in W.range(0, N, BN):
            m = W.tile(m0, BM, bound=M)
            n = W.tile(n0, BN, bound=N)
            acc = W.zeros((m, n), dtype=W.f32)

            for k0 in W.range(0, K, BK):
                k = W.tile(k0, BK, bound=K)
                a = W.read(A.view(m, k), fill=0)
                b = W.read(B.view(k, n), fill=0)
                acc = W.matmul(a, b, init=acc, acc_dtype=W.f32)

            W.write(C.view(m, n), acc)
```

作者明确写出：

- worker slice；
- M/N/K traversal；
- BM/BN/BK；
- logical views与memory effects；
- accumulator lifetime；
- local matmul位置。

作者没有写：

- raw pointer expression；
- `vl`/LMUL；
- RVV type；
- register accumulator tuple；
- K-unroll/load pipeline；
- RVV或IME；
- intrinsic/asm。

### 10.3 显式 staging

若算法要求 f32→f16 staging，当前冻结模型由caller提供带shape、dtype、alignment和lifetime的worker workspace：

```python
stage = stage_workspace.view(m_local, k_local)
value = W.read(A.view(m, k), fill=0)
W.write(stage, W.cast(value, W.f16))
a = W.read(stage)
```

是否以及何时建立这个staging是作者算法决定；compiler不能把一次cast偷偷升级成跨loop materialization。若以后真实需求证明受限lexical scratch不可替代，再按5.4节重新审议，而不是先假定它存在。

### 10.4 Local semantic operation

`W.matmul`只声明：

```text
[M,K] × [K,N] + init[M,N] → result[M,N]
```

它不拥有 outer K loop、staging、persistent packing或store。

### 10.5 Target realization

Compiler 根据 canonical axis、pointer/index/stride、value lifetime、reuse与target facts形成：

```text
N-lane/M-register RVV outer product
M-lane/N-register realization
F16 widening FMA
IME fragment（若语义与target支持）
```

并选择 LMUL、microtile、accumulator数、unroll、compiler-private pipeline buffer与handoff。

### 10.6 IME

用户不写 `matmul_ime`。若现有 `W.matmul` 或typed quant op的语义与某IME fragment兼容，target profile增加一个合法realization。当前IME1 candidate明确绑定`SpacemitIME1I4I8MMA`与symmetric/affine i4×i8 local primitive（`lib/Target/RISCVPhysicalPlanning.cpp:1099-1234`），因此不能推出普通f16/f32 matmul或其他quant关系已有IME candidate。缺失时应报告realization unsupported，而不是给DSL增加`ime`关键字或静默转向另一条kernel路径。

### 10.7 相比直接写C，究竟省略了什么

比较对象必须是“可竞争的RVV/IME intrinsic C”，而不是没有优化的三重标量循环：

| 内容 | 直接写高性能C/asm的作者 | Weft kernel作者 | Weft/系统编译器 |
|---|---|---|---|
| M/N/K traversal、BM/BN/BK、cache blocking | 手写 | 仍然手写或显式选择recipe | 不发明 |
| staging、workspace、persistent packed B | 手写并管理ABI | 仍然显式写 | 只实现已声明关系 |
| accumulator跨K-loop生命周期 | 用C变量/数组手写 | 用block SSA显式表达 | 决定physical placement |
| tail、mask、pointer/stride关系 | 每个target路径手写 | 写logical view/validity | 选择unit/strided/indexed与tail realization |
| `vl`、SEW、LMUL、RVV type | 手写并维护VLEN差异 | 不写 | Weft target决定 |
| MR/NR、vector accumulator tuple、K-unroll | 手写微内核 | 不写 | Weft target构造/选择；当前空间仍需加强 |
| load reuse、local pack、pipeline buffer | 手写并人工核算live registers | 只写算法可见staging | Weft target负责compiler-private部分；跨op scheduler尚是缺口 |
| RVV intrinsic与IME asm | 手写不同实现 | 只写`matmul`或typed quant relation | Weft发射局部realization |
| 最终寄存器分配、machine scheduling | 交给C compiler | 交给C compiler | system compiler负责 |

因此Weft不是省掉GEMM算法，而是省掉**当前local block内部的target-specific physical program**。如果用户仍需写LMUL、fragment、寄存器组与RVV/IME分支，抽象失败；如果compiler替用户创建外层K-loop、persistent pack或另一种GEMM算法，权责同样失败。

---

## 十一、为什么仅用 GEMM 不能证明 Weft

Fixed tile GEMM、pointwise和普通 reduction都能自然落入 Triton-CPU。Weft 的独立模型必须由更难的组合证明。

### 11.1 Quantized projection

自然算法可能是：

```text
ordered row/block traversal
→ VLA max reduction
→ scale + explicit narrow/round/saturate
→ write activation workspace
→ traverse output blocks
→ consume persistent packed weight
→ typed i4×i8 local dot
→ RVV or IME realization
```

这里同时需要：runtime VLA、workspace lifetime、persistent format、explicit quantization algorithm、typed local op 和 extension selection。

### 11.2 Online attention/state kernel

自然算法可能是：

```text
stage query
for key in ordered traversal:
    dot
    update maximum/scaled sum
    update value accumulator state
normalize
store
```

Compiler不得从普通 graph改成另一种 softmax/attention variant；但应能在显式 key loop内联合安排load、dot、summary和state representation。

### 11.3 SSM/RWKV/Gated state

Token loop具有可观察顺序；state同时进入next-state与output。普通loop carry不能自动变成reduce/scan。该语义在GPU SPMD tile模型中可以编码，但不是其默认canonical owner/effect contract。

这些程序共同刻画了Weft必须验证的目标workload：mixed ordered control、scalable local domain、typed storage/state与local extension需要同时成立。它们本身不证明Triton或TileLang不能承载，也不证明Weft已经成功；证明只能来自这些facts在真实DSL→IR→physical decision→artifact中产生可追踪因果作用。

---

## 十二、为什么不是直接复用 Triton-CPU 编程模型

### 12.1 可以直接复用的部分

Weft应积极借鉴或复用：

- explicit local tensor/block operations；
- op-specific lowering；
- layout propagation思想；
- target capability/legality；
- build-time autotuning；
- MLIR pass与dialect基础设施；
- intrinsic/backend代码生成方法。

这些不是Weft需要重新发明的研究问题。

### 12.2 Triton已有而Weft也需要的能力

- loops与SSA；
- masked load/store；
- block pointwise；
- dot/reduction/scan；
- persistent kernel；
- multiple hardware backends；
- autotune。

所以不能把这些写成Weft独有创新。

### 12.3 若直接扩展 Triton，实际会触及哪些层

下面不是“必须改十项才可能”的理论证明，而是把Weft冻结语义做成Triton正式contract时的工程影响面。部分内容可以先用wrapper、metadata或backend convention试验；只有当compiler需要可靠推理时，才必须进入core IR/type/effect/ABI。

| 所需合同 | Triton可复用起点 | 若要成为正式Weft语义，至少影响 | 仅靠约定是否足够 |
|---|---|---|---|
| worker-call与work slice | persistent program、host launcher | launch/AOT ABI/runtime | 原型可；稳定产品不够 |
| runtime scalable VLA，`vl`不可观察 | dynamic shape、mask、loop | type/region/layout/lowering | 不够，若要跨op传播与验证 |
| logical axis identity与validity | block tensor、descriptor、mask | frontend/type/value facts/verifier | 不够，若layout与handoff依赖它 |
| ordered carry、reduce、scan、summary分离 | SSA loop、reduce/scan | op/effect/algebra contract | helper可表达，compiler推理不足 |
| external/persistent/workspace format | pointer/descriptor/ABI metadata | storage type、ABI/header、alias/lifetime | 可做metadata；必须有唯一consumer |
| typed quant/extension relation | custom op/inline asm | dialect/op legality/target realization | inline asm只解决拼写，不解决语义 |
| RVV lane/register/IME fragment mapping | layout与backend机制 | CPU/RVV target layout/resource | 属于backend核心工作 |
| 跨ordered control的lifetime/resource | SSA/use-def | analysis/selection | 后端局部约定无法完整覆盖 |

这说明差异跨越多个协调层，但不说明Triton从理论上不能承载。真正的选择是：维护一个以这些CPU语义为唯一根模型的小系统，还是长期维护一组Triton core extensions。

### 12.4 用同一个persistent quant projection对照authority

两种模型都能“写出”该算法，差别是哪些事实是默认canonical、哪些需要项目约定或extension：

| 关系 | Triton当前可用表达 | Weft冻结合同 | 结论 |
|---|---|---|---|
| 一个execution instance处理多个output blocks | persistent program内显式loop | worker内显式ordered loop | 不是Weft差异 |
| runtime row/token长度 | runtime dimension + mask/descriptor + compile-time block shape | explicit runtime VLA domain，target选择strip | 都可表达；canonical VLA权责不同 |
| activation workspace | pointer/descriptor与host convention | typed workspace storage + extent/header metadata | Weft要求lifetime/ownership被ABI消费 |
| packed weight | raw/typed pointer、descriptor或custom metadata | persistent storage + format identity | Weft要求format成为producer/consumer合同 |
| i4×i8 local relation | ordinaryops、custom op或inline asm | typed local semantic op | 两者均可扩展；Weft禁止从bitwise graph猜出 |
| RVV/IME选择 | CPU backend/custom op lowering | 同一local op的target realization | 区别在是否共享同一typed candidate space |
| outer blocking与packing算法 | source program显式写 | source program显式写 | 都不能归给backend猜测 |

因此，不存在“这个kernel只有Weft能写”的论证。Weft要证明的是：把右列storage/VLA/local-op facts设为默认contract后，能以更少的source convention、更直接的legality与更共享的RVV/IME编译获得实际收益。

### 12.5 理论上仍可做成 Triton fork

完全可以fork Triton并实现上述合同。那可能复用更多基础设施，但需要长期维护与上游语义差异；最终得到的是在Triton基础设施中承载的Weft式CPU contract，而不是“只加一个RVV codegen即可”的当前模型。

因此独立Weft不是“否则做不到”，而是：

> 当前项目选择让core-local CPU contract成为唯一根模型，并只复用可共享的编译技术；如果未来实证表明上述合同可由很薄的Triton extension稳定承载，那么迁移到Triton基础设施反而是更合理的选择。

### 12.6 什么时候应放弃Weft、选择Triton-CPU

如果目标最终收缩为：

```text
fixed compile-time tiles
raw pointer ABI
pointwise/reduce/dot/matmul
每个program或persistent program处理blocks
不需要typed workspace/persistent format
不需要runtime VLA/state semantics
```

则应直接采用Triton-CPU/RVV backend。继续维护独立DSL没有合理收益。

---

## 十三、为什么不是直接复用 TileLang-CPU 编程模型

先校正事实：2026-08-19核验的TileLang 0.1.13文档明确把自己描述为GPU/CPU kernel DSL，target文档列出LLVM CPU execution，官方API暴露`tilelang.cpu` backend namespace及codegen、pipeline、scalar GEMM等子模块。因而不能把TileLang说成“只有GPU、以后才可能有CPU”，也不能反过来把已有CPU backend误写成已经成熟的RVV/Weft式编程合同。[TileLang](https://www.tilelang.com/index.html) [TileLang Targets](https://www.tilelang.com/get_started/targets.html) [TileLang CPU API](https://tilelang.com/autoapi/tilelang/cpu/index.html)

### 13.1 TileLang真正值得复用的部分

TileLang将不同使用层次分开：hardware-unaware beginner、使用Tile Library的developer、thread primitives expert；同一kernel可混用。其核心思想是dataflow与thread binding、layout、tensorization、pipeline分离。需要同时注意：官方文档明确写着hardware-unaware beginner interface尚未完整实现，因此不能把规划中的最高层当成当前成熟能力。[TileLang Overview](https://www.tilelang.com/get_started/overview.html)

Weft应复用：

- 多层用户接口；
- algorithm/tile source library；
- target-independent local semantic ops；
- effectful producer/consumer pipeline分析；
- layout inference与target realization分离；
- user可以逐层下降而不切换整个语言。

### 13.2 TileLang当前公开主路径与Weft冻结合同的差别

TileLang已经支持CPU target；但当前公开language-basics与GEMM主路径仍突出以下GPU hardware-aware构造：

```text
T.Kernel(grid, threads)
alloc_shared
alloc_fragment
copy global↔shared↔fragment
Pipelined stage/order
gemm/tensor-core op
```

这些直接反映GPU的软件管理层级。这一观察限定于当前公开主路径，不声称TileLang IR或CPU模块永远不能表达其他模型。

Weft不应复用为canonical语义的部分：

- grid/block/thread identity；
- shared/fragment hardware storage classes；
- barrier与协作copy；
- physical thread binding；
- source-visible fragment/register layout；
- GPU async-copy pipeline contract。

### 13.3 CPU等价物不是简单改名

右列不是TileLang-CPU已有API的逐项翻译，而是Weft冻结/拟议的CPU semantic replacement：

| TileLang公开GPU-facing construct | Weft选择的CPU semantic replacement |
|---|---|
| grid/thread block | worker callable + explicit work slice |
| shared memory | 无直接硬件scope等价；caller workspace只表达lifetime，不承诺cache level |
| fragment allocation | non-addressable Block SSA accumulator，由compiler选择register/fragment |
| parallel copy | logical read/write；拟议`copy`只是同步source helper |
| thread binding | compiler-private VLA/lane/register mapping，不进入source |
| `T.gemm` | target-independent local `matmul`；官方CPU API公开`gemm_scalar`子模块，但这不等于已经具有Weft的RVV/IME contract |
| `T.Pipelined` | 无直接source等价；目标是dependency/resource-driven local scheduler，当前仍有跨op缺口 |
| hardware layout | transient target physical arrangement |

### 13.4 为什么已有CPU target仍不等于Weft合同

LLVM CPU target回答“生成到哪里”，并不自动回答source是否以worker-call、runtime VLA、axis-identified block、ordered state和lifetime-oriented storage为canonical contract。TileLang完全可以在现有frontend/library/pass基础设施上补齐这些语义；如果它们能以很薄、稳定、可被layout与resource compiler消费的方式加入，那么直接复用TileLang会比独立Weft更经济。

当前独立Weft的选择，是让这组CPU语义从一开始成为唯一根模型，而不是在hardware-aware GPU主路径旁再加一套约定。这个选择的合理性必须由后端真实消费这些语义来维持，不能由“TileLang最初面向GPU”维持。

---

## 十四、为什么这不会重新变成Graph Compiler

### 14.1 Buffer/View不等于Tensor Graph

Buffer/View存在于一份显式ordered worker program内。Compiler不负责：

- 全模型graph import；
- operator fusion/partition；
- 自动发现attention/GEMM；
- 创建outer loop；
- 替换algorithm variant；
- 调度多个kernel。

View只让pointer/axis/bounds成为正式事实，不把kernel变成graph。

### 14.2 Algorithm recipe不是backend route

Algorithm recipe由作者在source中显式调用并展开。Target lowering看不到recipe名字，只看到canonical operations。因此它是代码复用，不是whole-kernel matcher。

### 14.3 Local operation不是完整operator

`matmul`只拥有local block product；typed quant op只拥有local byte/numerical relation。它们没有entry ABI、outer control或persistent layout。

### 14.4 Compiler不从普通SSA发现新语义

所有reorganization authority来自显式VLA、state algebra或local semantic op。普通SSA只提供use/lifetime/handoff，不赋予算法替换权。

---

## 十五、数值语义也必须分层

Local op必须显式保存真正影响observable result与legal realization的数值合同，例如：

```text
input dtype
accumulator dtype
reassociation policy
FMA contraction policy
rounding
saturation
approximation policy（仅适用的op）
```

当前public contract接受`order ∈ {ordered, preserve, relaxed}`与`math ∈ {strict, native, fast}`（`python/weft/frontend/compiler.py:1894-1906`）。冻结语义为三值`order`赋予明确区别——逐元素顺序、保持逻辑分区顺序、允许target相关tree/lane/strip重组——因此它应继续保留；frontend当前只校验字符串，完整语义仍需各target兑现。真正有问题的是把同一个`math`三值空间平铺到matmul：它同时混入FMA contraction、目标原生product precision和近似算法，而当前RISC-V matmul仅接受`order="relaxed"`且`math="native"`这一配对（`lib/Target/RISCVKernelCompiler.cpp:9419-9426`），其他组合没有realization。

`acc_dtype`是必要且正确的语义，因为它决定局部product的accumulator类型。最终matmul canonical op应保留`order`，并把multiply-add contraction拆成独立事实；只有确有替代product precision的硬件时，才为matmul增加一个精确的product-precision属性。Transcendental的`strict/native/fast`继续由exp/tanh等各自op解释。例如拟议替代是：

```python
W.matmul(
    a,
    b,
    init=acc,
    acc_dtype=W.f32,
    order="relaxed",
    contraction="allow",
)
```

若作者要求不同顺序或禁止FMA，应使用精确值：

```python
order="ordered" | "preserve" | "relaxed"
contraction="forbid" | "allow"
```

当前target没有对应realization时必须明确unsupported，不能把其他`order/math`组合静默当成`relaxed/native`，未来也不能把`contraction="forbid"`静默当成`allow`。上述`contraction` spelling是拟议替代方案，不是当前已经完成的API；因此当前`order`语义保留，matmul的`math`属性需要收缩，而不是把整个数值合同推倒重来。

---

## 十六、编译器获得什么，作者保留什么

### 作者拥有

- worker slice与entry ABI；
- ordered control；
- algorithm/cache blocking；
- staging、caller workspace与persistent layout；
- state algorithm/lifetime；
- Buffer/View与data movement；
- local semantic operations；
- algorithm variant与observable numerical policy。

### Weft compiler拥有

- VLA strip与dynamic `vl`；
- logical axis到sequential/lane/register/unroll/fragment的mapping；
- value physical layout与handoff；
- memory instruction/form；
- register microtile与multiple accumulators；
- operand reuse、reload、rematerialize与local pack；
- loop-local pipeline与resource legality；
- RVV/IME realization；
- intrinsic C与typed asm。

### System C compiler拥有

- 最终physical register allocation；
- 最终machine scheduling；
- peephole；
- constant folding；
- machine code encoding。

### Weft内部必须严格区分三类信息

这是compiler model的冻结部分，不能再混成一个family selector：

| 类别 | 内容 | 是否可搜索 |
|---|---|---:|
| 输入事实 | DSL已经写下的axis identity/control/value/use/memory/effect/lifetime/local op，加target的ISA/ABI/VLEN/register/extension facts | 否 |
| 唯一语义推导 | 由显式axis identity与op/use relation得到free/reduction/broadcast role，以及pointer relation、validity/tail、live range、dtype conversion | 否；同一输入只能有一个答案 |
| 结构性选择 | sequential/lane/register/fragment映射、state placement、RVV或IME local realization、share/reload/rematerialize/local pack、pipeline topology | 是；但只能在显式授权内部 |
| 参数性选择 | LMUL、MR/NR、accumulator数、K-unroll、pipeline buffer count、prefetch distance | 是；只实例化已选结构 |

BM/BN/BK等algorithm/cache blocking默认属于作者。只有作者在recipe/meta config中显式给出候选集合时，build-time tuner才能在不同DSL实例之间测量选择；target不能把它们伪装成local physical parameter并暗中改变outer algorithm。

Candidate生成后，先由ISA/shape约束与完整live-resource accounting逐项删除非法项，再由target cost或构建期实测在合法结构/参数中选择。实测不创造新结构，也不能让非法candidate变合法。每项最终decision只有一个producer；emitter只能读取selected mapping、schedule、resource与leaf，不得再次按shape/dtype/format猜测。这是冻结的selection流程；当前实现仍使用固定lexicographic penalty，通用构建期实测选择尚未完成。

### Physical layout具体是什么

Weft的layout不是source-visible storage layout，也不是一个`RVVStrip/RegisterMicrokernel/IMEFragment`类别名。它是每个logical value的轴分解与机器表示，例如：

```text
logical axis
→ sequential factor
→ scalable lane factor
→ register repetition / accumulator factor
→ reduction unroll factor
→ optional extension fragment coordinate
```

它必须沿整个value-use chain传播：pointwise保持同轴映射，broadcast axis只能singleton扩展，cast保持logical element correspondence但重算SEW/LMUL footprint，load/store用pointer relation约束lane方向，reduce/scan消去或携带指定axis，matmul连接operand free/reduction axis与result axis，loop carry保持跨iteration representation一致。多个consumer要求冲突时，compiler明确选择共享、reload、rematerialize、shuffle或primitive-private local pack；不能因one-use/direct-store closure不匹配而丢失整个op能力。

当前代码已经有transient mapping、resource与candidate结构，但operand-aware arrangement、跨op传播、完整live resource和通用scheduler仍是20.5节的实现缺口。冻结的是上述信息流和唯一authority，不是假装当前算法已经达到Triton级成熟度。

---

## 十七、该模型带来的收益：当前可确认与实现后预期

这里不把设计目标写成已经证明的效果。前四项主要是冻结语义带来的可推理性；后几项只有在编译器真正消费这些事实并产生高质量代码后才成立。

### 17.1 当前可确认：比无额外合同的raw C结构化保存更多canonical事实

C当然可以用pointer、loop、struct和注释编码同一算法；差别不是C“不能表达”，而是当前Kernel IR把K axis、VLA domain、state algebra、packed relation与workspace storage class/metadata保存为可验证、可被lowering直接消费的结构化事实，不要求后端从任意C习惯用法恢复。当前storage class只是lifetime/ownership的有限代理，还不是完整effect interval或跨kernel runtime verifier。

### 17.2 设计收益：source不绑定RVV物理参数

同一source不写fixed VLEN、LMUL、vector type与register microtile，因此允许在SG2044/K1形成不同实现。是否已经对每项kernel选到高质量的不同实现，属于compiler与性能事实，不能由语言设计直接推出。

### 17.3 冻结语义收益：CPU算法结构有唯一owner

Cache blocking、ordered token/key loop、caller workspace、persistent packed weights与跨block state自然属于同一worker程序。

### 17.4 拟议表面收益：高层可用、低层可下降

普通用户选择algorithm recipe；kernel作者用Buffer/View与local ops；新算法作者下降到VLA/axis/block；backend作者处理ISA。无需所有人都面对最底层构造。

### 17.5 架构收益：新扩展不应增加完整operator backend

IME或未来RISC-V extension只为现有local semantic op增加realization。相同matmul/quant relation位于GEMM、MoE、Conv或Attention时应共享能力；是否已经做到，必须从production lowering而不是接口图判断。

### 17.6 当前产品方向：AOT嵌入CPU生态

Generated object/header可以进入llama.cpp、GGML、线程池或应用runtime；Weft不接管多线程和model graph。

### 17.7 拟议复用收益：Algorithm library与compiler能力分离

可以拥有高质量blocked GEMM/attention/quant recipes，同时保持target lowering不按recipe/kernel名路由。

### 17.8 创新边界：不能把已有概念重新命名成贡献

以下不是Weft独有创新：persistent worker、block/tile SSA、dot/matmul/reduce/scan op、masked memory、multi-level library、layout propagation、autotuning、RVV codegen或IME asm。Triton、TileLang、TVM/MLIR及手写kernel体系已经分别证明了这些思想。

Weft可以主张的研究假设只有它们在CPU contract中的**组合方式与compiler消费方式**：

1. **Core-local mixed-control canonical program**：同一份IR把author-owned ordered control、runtime VLA、axis block SSA、typed state与caller-visible storage放在同一个worker lifetime中，而不是把CPU仅视为另一个fixed-tile target。
2. **Lifetime/value-chain-driven physical mapping**：layout不是逐op/family模板选择，而是沿ordinary multi-use、control carry、memory relation和state lifetime联合生成sequential/lane/register/fragment实现。
3. **Extension as local realization**：RVV、IME及未来RISC-V扩展只扩张已有local semantic op的合法physical space，不增加whole-kernel route；同一语义在dense、quant、attention、MoE等外围程序中复用。
4. **Algorithm transparency without hardware-scope exposure**：作者保留blocking、staging、packing与state algorithm，compiler接管LMUL、register microtile、fragment与compiler-private pipeline；CPU用户表达lifetime/reuse原因，而不是`shared/fragment/register`硬件scope。

前两项是核心compiler研究，后两项是架构和扩展性贡献候选。它们目前仍是需要production lowering与真实性能共同支持的主张：若最终还是family selector、one-use closure或kernel-name leaf，论文和项目都不能把上述内容称为已实现创新。

---

## 十八、必须承担的代价

### 18.1 比增加Triton backend成本更高

需要维护独立frontend、Kernel IR、verifier、AOT ABI和library surface。

### 18.2 生态更小

不能自动继承Triton/TileLang所有教程、工具、backend与调优生态。

### 18.3 Compiler责任很重

既然不把LMUL、fragment、pipeline暴露给用户，Weft必须真正拥有强layout、reuse、resource、schedule与cost model。隐藏机器细节但生成很慢的代码，不构成成功的抽象。

### 18.4 容易primitive膨胀

需要严格 admission rule：新op必须有独立、完整、局部、可观察语义；不能每个kernel/format一个op。

### 18.5 容易产生两套authority

Algorithm helper、semantic op与realization schema必须分别停在source、canonical IR与transient lowering，不能都保存一份program truth。

### 18.6 易用表面不能偷偷增加新authority

`Buffer/View/copy/recipe`若实现，必须完整展开到现有canonical facts。若它们在Kernel IR中又保存一份可与pointer/axis/storage不一致的truth，或者backend开始按helper/recipe名字分流，那么易用层反而破坏了模型。

---

## 十九、对该模型的主要反驳及回答

### 反驳1：Triton persistent kernel也能做这一切

它能编码许多相同算法，这是事实。Triton支持runtime problem dimensions、mask、descriptor和persistent loop；但其program/grid与block tensor contract并不会自动等价于“arbitrary runtime VLA + axis identity + typed storage lifetime + ordered state”。若这些事实必须跨op参与layout、resource与ABI推理，就需要core extension，而不只是多写一层persistent loop。

若愿意长期维护这些Triton core extensions，使用Triton fork是可行工程选择；这不否定Weft模型，只改变实现载体。

### 反驳2：Buffer/View只是重新发明tensor DSL

Buffer/View仅描述一份worker程序内的address relation与storage lifetime，不引入graph scheduling、fusion、partition或operator discovery。它更接近typed pointer region，而不是framework tensor graph。

### 反驳3：高层algorithm library又变成whole-kernel模板

Source recipe可以拥有完整algorithm，因为作者显式选择并展开它。禁止的是target根据kernel/name/shape自动选opaque whole-kernel implementation。两者authority位置不同。

### 反驳4：隐藏`W.block/W.vla`后backend还是要猜axis

高层surface只隐藏spelling，不隐藏semantics。`tile/view/map/reduce`必须在frontend expansion时创建真实axis/VLA op，canonical IR完整保存。Backend不得从shape猜。

### 反驳5：所有作者可见scratch都用caller workspace太笨重

它确实可能多出参数，但当前好处是ownership、size、alignment与memory budget全部可见，且已有模型能够承载。仅凭“ABI不够漂亮”不足以引入新的stack/private allocation语义。5.4节列出的真实需求出现前，冻结模型继续只用caller workspace；未来若加入lexical scratch，也必须是受限、noescape、resource-checked的新决定，而不是默认答案。

### 反驳6：让compiler自动pipeline会改变作者顺序

Compiler只能移动被use-def、alias、effect与state语义证明不可观察的operation，并保持author-visible iteration/effect result。Write/atomic/unknown-alias read/ordered carry/control-dependent effect不得越界；algorithm-visible buffering必须由作者显式写。无法证明时不生成该pipeline candidate。

### 反驳7：新extension直接用inline asm就够了

Inline asm可以拼指令，但不能独立提供semantic legality、operand layout、resource、handoff与跨target alternative。已有语义应先进入同一local-op candidate space，asm只是最终leaf。

### 反驳8：C compiler已经会自动向量化

C compiler适合最终register allocation与machine scheduling，但不会可靠恢复axis identity、packed quant relation、author storage contract或IME fragment语义。Weft提供比普通C更高的局部semantic information，同时不承担LLVM backend工作。

### 反驳9：这套模型还是太低级

Application user可以调用algorithm recipe；只有kernel/algorithm author需要写worker loops、View和storage。Weft是kernel DSL，不承诺从纯数学operator自动发明CPU algorithm，但不要求每个使用者重复手写所有loops。

### 反驳10：这套模型还是太高级，不能控制性能

作者仍控制blocking、loop order、staging、workspace、packing、state algorithm与recipe。隐藏的是LMUL、register、fragment和ISA，不是CPU algorithm。Expert surface还允许直接组合VLA/axis/block values。

---

## 二十、当前实现、冻结语义与拟议表面的边界

### 20.1 当前已经具备

- worker-local ordered control；
- raw pointer/index/predicate；
- runtime VLA；
- axis-identified block SSA；
- frontend/Kernel IR层的ordinary multi-use与control carry；
- external/persistent/workspace caller contract；
- dot/matmul/reduce/scan/summary；
- typed quant/extension operations；
- one RISC-V lowering与AOT C/header；
- transient physical decisions。

当前源码证据如下；这是报告快照的事实边界，而不是由设计文档反推：

| 事实 | 当前源码证据 | 精确边界 |
|---|---|---|
| Python DSL→canonical Kernel IR | `python/weft/frontend/compiler.py:301-408` | frontend直接构造`weft_kernel.kernel`；`weft-compile`只消费canonical MLIR |
| ordered `for/while/if`与carry | `python/weft/frontend/compiler.py:2775-2964`；`include/Weft/Dialect/Kernel/IR/KernelOps.td:142-164`；`lib/Target/RISCVKernelCompiler.cpp:3890-4041` | target只闭合scalar、部分VLA value及有界rank≤2 f32 block carry，其他明确unsupported |
| runtime VLA与axis block | `python/weft/frontend/compiler.py:1254-1289,2818-2876`；`include/Weft/Dialect/Kernel/IR/KernelOps.td:53-73,166-171` | 当前禁止nested VLA；target仍要求明确fixed VLEN profile |
| external/persistent/workspace ABI | `python/weft/language/annotations.py:18-93`；`lib/Target/RISCVHeader.cpp:251-327` | `W.storage`不分配内存；entry仍是flattened pointer/scalar C ABI |
| 唯一RISC-V主链与瞬态decision | `lib/Target/RISCVLowering.cpp:10-34`；`lib/Target/RISCVKernelCompiler.cpp:478-516,958-1068` | physical plan是一次compile内存对象；输出只持久化C/header/object |
| 结构化op与局部hardware realization | `include/Weft/Dialect/Kernel/IR/KernelOps.td:289-336`；`lib/Target/RISCVKernelCompiler.cpp:3395-3544` | op dispatch真实存在，但只对有合法typed candidate的subset发射 |

这些条目不等于所有组合都已有高性能target artifact，也不等于compiler/runtime会验证caller实际遵守跨调用ownership。尤其runtime-shape二维block materialization/reduction等范围仍有明确unsupported；“IR能表达”“ABI带metadata”和“当前RISC-V target对所有合法consumer都能发射”必须分开叙述。

### 20.2 已冻结、但当前主要由低层spelling承载的语义

- addressable storage、logical region、block SSA三者分离；
- view必须展开为axis/pointer/index/validity的单一truth；
- recipe必须source-expand并在target前消失；
- persistent format是caller-visible ABI；
- pipeline只移动effect/alias证明安全的compiler-private工作。

这些边界已经冻结，但当前用户往往需要通过raw typed pointer、`W.storage`、显式axis、`W.load/store`写出。

### 20.3 拟议、尚未实现的易用表面

- ergonomic Buffer descriptor；
- logical View/tile API；
- read/write over View；
- narrow copy helper；
- 清楚分层的algorithm recipe library；
- 对普通用户隐藏axis plumbing但保留canonical identity的frontend expansion。

这些是source ergonomics，不得被报告成当前public能力，也不得新增第二条lowering。

### 20.4 明确未冻结

- lexical scratch/private addressable allocation；
- 用户可见pipeline stage、prefetch distance或hardware storage scope；
- 任何将LMUL、RVV type、IME fragment暴露给普通DSL作者的接口。

### 20.5 当前compiler仍需完善

- operand/result-aware physical arrangement；
- accumulator correspondence；
- packed/decode/fragment layout；
- RVV↔IME handoff；
- cross-op loop-local scheduling；
- complete live-resource model；
- target cost与compile-and-measure选择。

当前dot/matmul已有LMUL、unroll、pipeline buffer与resource candidate，并用固定penalty排序（`lib/Target/RISCVPhysicalPlanning.cpp:2815-3192`）；尚未形成运行实测驱动的通用cost model。因此“有candidate字段”不能被报告成完整compile-and-measure系统。这些compiler缺口影响性能，但不改变本报告定义的source programming model。

---

## 二十一、最终冻结原则

1. 一个kernel是一份单worker/hart持续执行的ordered CPU program；
2. 多核work allocation属于external runtime；
3. addressable storage表达ownership、shape、stride、alignment、alias与lifetime；`Buffer`只是建议的source wrapper；
4. logical view表达storage上的axes、offset、extent、stride与validity，并必须在frontend完整展开为canonical facts；
5. Block Value是non-addressable、axis-identified ordinary SSA；
6. scalar control默认ordered，不能自动升级为parallel/reduction；
7. VLA/map/reduce/scan/matmul/typed op显式授予局部重组权；
8. 作者拥有outer traversal、blocking、staging、packing与state algorithm；
9. algorithm recipe可以拥有完整algorithm，但必须source-expand；
10. canonical local op只拥有局部semantic relation；
11. target realization只拥有physical layout/microkernel/pipeline/ISA；
12. source helper、canonical IR与physical plan不能形成重复authority；
13. CPU storage暴露lifetime/addressability，不暴露cache/register/fragment层级；
14. 当前不加入lexical scratch；真实需求成立后也必须static bounded/noescape/resource-checked并重新冻结；
15. compiler-private pipeline只能由compiler在显式loop与effect边界内生成；author-visible staging/ring buffer仍由作者拥有；
16. RVV/IME是local realization，不是DSL route；
17. emitter机械消费selected decision；
18. system compiler负责最终register allocation与machine scheduling；
19. 无合法realization时明确unsupported，不允许fallback；
20. 所有高层表面最终汇入唯一canonical Kernel IR与RISC-V lowering；
21. `Buffer/View/read/write/copy/algorithm recipe`的具体Python spelling属于拟议表面，不把未实现API冒充成冻结后的当前事实。

---

## 二十二、独立Weft是否成立的最终判据

先给出反事实选择表。它防止用“我们已经写了很多代码”倒推独立语言必然正确：

| 实际需求 | 更合适的起点 | 原因 |
|---|---|---|
| fixed block pointwise/reduce/matmul，program或persistent program处理work | Triton/Triton-CPU | 已有block-op、layout、autotune和backend生态，Weft没有独立收益 |
| 作者愿意显式管理hardware scope、copy、pipeline和thread binding | TileLang | 当前developer/expert模型直接服务这种控制 |
| 完全手控RVV/IME指令、ABI稳定且kernel很少 | intrinsic C/local asm | DSL与compiler成本可能高于收益 |
| ordered worker control、runtime VLA、axis SSA、storage lifetime、typed state/quant需共同驱动物理编译 | Weft模型 | 这是本报告定义的专门contract；仍需实现证据证明独立载体值得 |

因此，Weft不是前三类需求的“更高级统一答案”。它只对最后一类成立。

### 成立

若实际实现证明：

- 同一runtime VLA程序在不同VLEN形成不同合法strip/layout；
- axis identity而非shape/name决定matmul/reduction与handoff；
- ordinary consumer/control carry真实影响lifetime/resource但不破坏primitive能力；
- workspace/persistent合同约束ABI与合法realization；
- 同一local semantic op在RVV/IME间获得不同实现；
- algorithm recipe展开后target不依赖recipe/kernel名；
- stateful、quantized、irregular与dense kernel共享同一physical compiler；
- 性能来自共享layout/reuse/schedule能力。

则这些结果共同支持Weft具有独立编程模型与编译研究价值。这里没有用单一kernel、单一性能数字或文档完整度作结论。

这些不是要求新增test matrix或验证脚手架；证据来自项目既有真实DSL→Kernel IR→intrinsic C/asm→目标机repro与性能记录。关键是每个结论都能指向生成代码中的因果作用：去掉某项semantic fact后，对应legality/layout/ABI/schedule确实无法保持，而不是仅在IR打印结果中少一个字段。

### 不成立

若最终系统只是：

```text
Python语法
→ fixed block ops
→ 按kernel/format选择RVV/IME helper
```

或者Buffer/View/VLA/storage只用于verifier，physical compiler仍按完整source closure与leaf模板工作，那么Weft只是低成熟度Triton/TileLang重写，应停止独立路线。

---

## 最终定义

Weft 不是因为“CPU无法运行Triton”而存在，也不是因为“RISC-V需要一门新语法”而存在。

它成立的必要条件与核心研究假设是：

> **面向目标工作负载的高性能RISC-V kernel，受益于一份以单worker ordered control为根、同时原生表达runtime VLA、logical block SSA、state lifetime、caller-visible storage和typed local compute的canonical CPU程序；compiler再从这些事实生成RVV/register/IME的物理程序。当前Triton和TileLang合同不会因为增加CPU target就自动提供这组语义，但它们完全可能通过core extension承载；Weft选择独立实现，必须以更直接的语义消费、更小的CPU专用contract和真实高性能结果证明这项工程取舍。**

最终产品结构应是：

```text
Algorithm recipes（拟议source layer）
       ↓ source expansion
Worker control + addressable storage/view + lifetime
       ↓ read/write（copy仅为拟议source helper）
Block SSA + local semantic operations
       ↓ canonical Kernel IR
Shared physical layout/reuse/resource/pipeline compiler
       ↓
RVV / IME local realizations
       ↓
intrinsic C / asm / object / header
```

冻结语义允许algorithm作者直接控制CPU blocking、staging与state，同时隐藏机器寄存器与扩展指令；拟议recipe/Buffer/View表面用于减少样板，但还不是当前实现事实。只有当唯一compiler真实消费这些语义、跨target形成共享高性能realization，并且不退化为kernel/format route时，Weft才不是Triton-CPU或TileLang-CPU的重复建设。
