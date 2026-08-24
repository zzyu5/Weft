# Weft 规范

## 摘要

Weft 是一门层级化、encoding-aware、有限位宽的**数值 realization 语言**。

作者写下一份具体的数值实现：哪些中间值存在、每个值在哪一层诞生、以什么类型的关系交回外层、哪些代数支路被拆出去、同时保留多少个输出、哪些重排发生在调用之外。

编译器不发明代数结构。它在非 SIMT 物理抽象机器上落实这份实现，并严格区分三类工作：唯一合法推导、由 target 固定规则完成的结构性选择、由构建期实测完成的参数性选择。`std` 特化和 source `auto` 仍由作者提供；target 不生成新的程序树。

一句话：

> **Weft = C 的数值语义 + 表示自由。**
>
> 你写的是一个确定的数值程序，一句都不含糊；但你不写它长什么样、放哪儿、用哪条指令。

---

# 第一部分 · 定位

## 1.1 缺失的那一层

一个算子的数学定义，和它的高性能实现之间，隔着一段东西。

以 q4_K × q8_K 的 vec_dot 为例。数学定义：

```
y[m] = Σ_k dequant(W[m,k]) · x[k]
dequant(q_i) = d·sc_b·q_i − dmin·m_b        (b = i/32)
```

真实的高性能 kernel 算的是它的改写形态：

```
y = ds · [ d · Σ_b ( sc_b · Σ_{i∈b} q_i·x_i )  −  dmin · Σ_b ( m_b · Σ_{i∈b} x_i ) ]
             └────────── scale 支路 ─────────┘      └───────── min 支路 ────────┘
```

两式数学相等，机器上是两个完全不同的程序。发生了三件事：

**(1) 一条支路被代数地拆出去，并落到预存字段上。**
min 支路里 `q` 消失了，剩下的是对 activation 的求和；而 Q8_K 的 block 预存了 `bsum`。256 次乘法塌缩成十几次。

**(2) 乘法按频率被重新分层。**
`q_i·x_i` 在元素层 256 次，`·sc_b` 在 sub 层 8 次，`·d` 与 `·ds` 在 block 层 1 次。改写后内层循环是纯整数乘加。

**(3) 累加类型跟着层拆开。**
元素层 i8×i8→i16（覆盖 2 项），sub 层 widen 到 i32 再乘 scale，block 层进 f32。

这三件事：不是算法（数学结果一样）、不是调度（改变了存在哪些中间值和运算总次数）、也不是 ISA（不提任何寄存器）。

> **几十种 vec_dot 之所以是几十种，不是因为几十种数学，是因为几十种分解结构。**

**这段东西目前没有语言承载。Weft 承载它。**

## 1.2 为什么已有语言承载不了

| 语言 | 根构造 | 承载不了的原因 |
|---|---|---|
| Intent / 计算图 | 逻辑域 + 原子 contract | contract 内部不可见，"i16 partial"、"min 支路走 bsum" 无处安放 |
| Halide / TVM / Exo | 算法 + schedule，**前提是 schedule 不改数值** | 上面的改写改变了数值结构，在 Halide 里那是另一个 algorithm，而 Halide 对新 algorithm 一无所知 |
| Triton | program grid + block tensor | distributed layout 围绕 thread/warp/CTA ownership；packed bit mapping 与层级有限位宽 realization 不是其根模型的一等结构 |
| TileLang | launch/thread binding + storage scope + tile op | shared/local/fragment、copy、pipeline 与 tensorization 是 source contract；Encoding 与 Level lifetime 不是其根模型的一等结构 |
| intrinsic | 全部手写 | 能写，但分解与表示写死在同一份代码里 |

**注意措辞：** 不是"别人写不出来"。Triton/TileLang 里当然可以手写 decode、乘法、归约和 correction 支路。真正的区别是 source contract 保存什么：Triton 保存 program/block 与 SIMT distributed layout，TileLang 保存 thread/storage/copy/tile structure，Weft 保存 Encoding、有限位宽数值树与 Level lifetime。具体机制见 7.2 与第八部分。

## 1.3 与 Intent 的边界

```
Intent      定义一个语义等价类，contract 保持原子
Weft        选定这个等价类里的一个具体的、有限位宽的、层级化的数值实现
intrinsic   把这个实现绑定到一台具体机器
```

同一个 `I.contract`，在 Weft 里有无穷多种合法写法，数学等价、性能差两个数量级。

**这条边界是设计出来的，不是自然法则。** 如果 Intent 也允许打开 contract，两者会重新重叠。所以 Intent 的 collective 必须保持原子 —— 这是两个项目并存的前提。

---

# 第二部分 · 根抽象

## 2.1 为什么根不是一个"对象"

Triton 一类 SIMT 语言把 logical coordinate 到 thread/warp/CTA owner 的分布作为物理 layout 的核心关系。单控制器 vector/matrix 目标不需要先建立这类虚拟 owner 才能调度整个逻辑值。

这不表示单控制器目标没有强物理对象；它只说明 Weft 不把某一种 owner、tile、VLA region、space-time block 或 resident-state pattern 选作所有程序的源语言根类型。

**准确的表述（不上升为硬件定理）：**

> 单控制器机器上确实存在很强的物理对象（RVV register group、IME fragment、AMX/SME tile state）。
> **但 Weft 的源语言不选择其中任何一种作为统一根类型。** 这是设计选择。

于是表示成为自由变量，只在两端被夹住：

```
一端：内存里的比特编码        Q4_K 的 nibble 排布、scale/min 位置
另一端：指令的操作数格式      RVV 的 SEW×LMUL、IME fragment 形状
中间整段：自由，且互相约束
```

程序结构那一侧剩下的，是**层级化的数值合成结构**。其他硬件事实——向量宽度、寄存器组、矩阵引擎、存储层级——全部落在表示一侧。

## 2.2 唯一判据

整份规范只有一条分界线：

> **改变 canonical operation/value/effect graph，或改变 Value 的 Level / artifact 归属 → 属于源程序（树），作者写。**
> **只改变同一程序的物理表示 → 属于编译器。**

canonical logical Value 由 producing operation 的 typed operands、result type、logical axes、数值语义与 effects 定义。目标 engine、invocation-local pack、memory form、lane/register/fragment layout 和 pipeline 都不进入 Value identity；它们是在同一 canonical graph 上形成的物理程序。

它的用法：

| 决策 | 逻辑值集合变了吗 | 归属 |
|---|---|---|
| 累加器拆成 4 路 partial | 变了（1 个值 → 4 个值 + 合并） | 作者 |
| `reduce(v)` 内部用树归约 | 没变（源程序只有一个结果值） | 编译器 |
| nibble 用 and/shift 还是 gather | 没变 | 编译器 |
| min 支路存不存在 | 变了 | 作者 |
| `group=16` 拆成 4 次 `group=4` | 变了（4 套 admit + 4 套状态） | **禁止** |
| 把循环不变的地址计算提出去 | 没变（本来就不随内层变） | 编译器 |
| 把 `w.d * i32_acc` 下沉到 sub 层 | 变了（中间类型、转换与舍入位置全变） | **禁止** |
| 为一个 staged Value 选择 local pack 及连续方向 | 没变 | 编译器 |
| `contract` 用 RVV 还是 IME | 没变 | 编译器；硬目标要求写在 build config |
| LMUL / vl / 寄存器分组 | 没变 | 编译器 |
| strip-mining、spill/reload、rematerialize | 没变 | 编译器 |

**这条判据同时定义了作者写什么和编译器做什么，它比任何一个"根抽象的名字"更根本。**

## 2.3 语义、配置与 hint 必须分层

成熟 kernel DSL 并不排斥 hint 或编译 meta。Triton 的 `tl.assume`、`tl.multiple_of`、`tl.max_contiguous`、`tl.range(num_stages=...)` 与 load/store cache modifier，TileLang 的 `T.Parallel`、`T.Pipelined` 和 `T.copy` annotation，都把事实承诺、schedule 参数或 target preference 附着在明确的 IR 实体上；它们不被伪装成数值 Value 的 identity。

Weft 采用以下分层：

```text
core DSL
    numerical operations / logical axes / effects / Level / pinned Encoding

build config
    source auto bindings / target profile / hard target requirements

target physical machine
    engine / layout / local pack / memory form / register-fragment mapping /
    pipeline / spill / reload / rematerialize / intrinsic-asm
```

core DSL 不提供 soft-hint 语法。target preference 或“必须使用某扩展，否则失败”属于 build config，不进入 canonical Kernel IR，也不改变 Value identity。

## 2.4 三条被否决的诱惑

下面三类构造不属于 Weft 的语言或编译器合同：

**(1) 用权限位代替语义或配置分层** —— `allow_reassociation` / `allow_repack` / `allow_low_precision`。

若一个开关改变数值、effect 或 artifact 语义，它必须进入相应 operation、type 或 Encoding；若只限制目标实现，它属于 build config；若两者都不改变，它没有独立语义。`-ffast-math` 看似权限，实则改变浮点语言；Triton 也把精度写进 `tl.dot` 的参数，而不是塞进一个通用 engine permission。

**(2) 数学等价证明** —— 编译器不需要证明 min 支路等价于原式，因为**编译器根本不知道原式存在**。Weft 程序里没有那个数学定义。作者写 `dot(w.m, mins)` 就是要算这个式子，和 C 里写 `a*b+c` 一样。分配律是作者做的，做错了就是算错。

**(3) 溢出证明** —— `into=i16` 是作者的选择，不是断言。溢出是作者的责任，和 C 里写 `int16_t sum` 一样。最多在可选调试模式插检查。

**通用规则：**

> **每当想加一个标志位，先问：谁读它？如果读它的那个优化被"不改树"禁止了，这个标志就不存在。**

`ordered` 就是这样被删掉的：它本来要区分"这个累加能不能拆 partial"，但拆 partial 本身已被判据禁止（那是改树），所以没有任何消费者读它。**依赖关系写在程序里，编译器读 use-def 就够了。**

---

# 第三部分 · 语言规范

## 3.1 Encoding —— 逻辑字段到 storage 的纯布局映射

```python
@weft.encoding
class Q4_K:                          # 144 B / 256 elems
    layout = bitorder.lsb_first, byteorder.little
    elements = 256
    d:     f16
    dmin:  f16
    sc:     u6[8] @ joined(4, 2, 4, lo_first)
    m:      u6[8] @ joined(4, 2, 4, lo_first)
    q:     u4[256] @ grouped(64) @ layered(32, lo_first)

@weft.encoding
class Q8_K:                          # 292 B / 256 elems
    layout = bitorder.lsb_first, byteorder.little
    elements = 256
    ds:    f32
    q:     i8[256]
    bsum:  i16[16]                   # per 16 elems
```

Encoding 不是“字段起始 offset + 元素宽度”的 C struct 字段表。它定义：

> **一个逻辑字段坐标怎样映射到 storage unit 与其中的 bit range。**

`elements` 是每条 storage record 对应的逻辑元素数。它是必填布局事实，不能从
字段 shape 猜测：TQ1 的 48 个 radix storage bytes、IQ 的 grid/index 字段都只覆盖
record 的一种存储分解，而一条 record 仍对应 256 个逻辑元素。

字段仍按源码顺序占据互不重叠的 storage span；字段内部的逻辑元素不要求按 `offset + i × width` 连续排列。作者只组合少量布局关系，不写逐元素 bit-slice 表，也不写地址公式。

从 GGML 的 Q4/Q5/Q2_K..Q6_K/IQ 系列归纳出的关系只有五类：

```text
natural
    logical element 按顺序占据自然 storage unit；Q8_0/Q8_1 属于此类。

grouped(N)
    每 N 个逻辑元素形成一个独立重复的 storage group。

layered(P, lo_first | hi_first)
    group 内按 P 个元素一层；同一层内位置相同的元素共享一个
    storage unit 的不同 bit range。Q4_0 是 grouped(32)+layered(16)，
    Q4_K.q 是 grouped(64)+layered(32)。

bit_planes(w0, w1, ...)
    一个逻辑值的不同 bit width 位于独立重复平面；Q5/Q6/Q3 的
    qh/hmask 属于此类。

join
    把多个 plane 中的片段重组成一个逻辑字段值；K-scale 与 IQ
    metadata 使用它。join 只重组 bits，不执行 scale、zero-point、
    codebook 或 dequant 数学。
```

`natural/grouped/layered/joined` 构成本轮公开且可 lowering 的基础布局表面。`bit_planes` 是全格式调查得到、但尚无本轮 consumer 的下一类纯布局关系；在它闭合前，相关 metadata 必须作为真实 storage 字段出现，不能伪装成连续逻辑字段，也不能让 emitter 按格式名恢复。

`joined(group=4, fields=2, low_bits=4, order=lo_first)` 表示一组连续逻辑字段共享规则化 storage：每个字段的前 `group` 项以完整宽度占据独立 byte；后 `group` 项的低 `low_bits` 在尾部 bytes 按 field 顺序分层，剩余高 bits 填入前半 bytes 的空余高位。Q4_K 的 `sc/m` 因而共享 12 bytes，作者不写任何 bit slice。

以 Q4_K.q 为例，`grouped(64) @ layered(32, lo_first)` 唯一表示：

```text
g = i // 64
r = i % 64
storage byte = 32*g + (r % 32)
storage bits = low nibble  if r < 32
               high nibble otherwise
```

这是布局关系的解释，不是作者写入源程序的索引公式。

Encoding 必须同时显式规定：

```
bit order        位在字节内的方向
byte order       字节序
elements         一条 storage record 对应的逻辑元素数
grouping         logical array 到重复 storage group 的划分
layering         group 内逻辑层与 storage bit range 的对应
bit planes/join  一个逻辑值跨多个 storage plane 时的纯 bit 重组
alignment        对齐
padding          填充位置与内容
```

**Encoding 不携带：**
- 不变量。编译器不知道 `bsum[j] == Σ q[16j:16j+16]`，也不需要知道。
- 解码语义作为推理基础。`dequant()` 如果存在，它是**可以被调用的函数**，不是给编译器推理用的关系。

**唯一作用：** 让作者按逻辑字段访问 packed storage，而不是自己计算 byte/nibble 地址。布局 pass 可以把组合关系展开为 canonical index mapping；那是编译产物，不是源程序表面。

**推论：** 编译器不能自行从 `q` 重建或替换 `bsum`。若某个派生编码需要生成字段，那是它的 builder 的责任。D 组可以选择 and/shift、gather 或专用 unpack 来实现同一 mapping，但不能改变 mapping。

> **注意 Q4_K 的 sc/m 粒度是 32、Q8_K 的 bsum 粒度是 16。这个错配是真实的，程序必须显式处理（见 4.1 的 `fold2`）。**

## 3.2 View

```python
View[Encoding, shape]
```

一个带编码类型的内存对象。`View[f32, (M,N)]` 是稠密特例。

## 3.3 Value

局部值有逻辑 shape 和数值类型，**没有物理表示**。

不是 tile：不要求静态 shape、不要求 2 的幂、没有 owner、不归属任何存储层级。

四个构造：

```python
new(dtype, shape, init=...)      # 在本层诞生一个状态值，随迭代更新
materialize(expr)                # 在本层物化一次，被后代只读复用
admit(view[region])              # 把一个区域从内存带进本层
commit(value, view[region])      # 写回
```

**`new` 与 `materialize` 必须分开：**

| | 语义 | 例 |
|---|---|---|
| `new` | 在本层诞生、被内层更新、有跨迭代状态 | accumulator |
| `materialize` | 在本层形成一个 staged Value、被后代只读复用 | staged panel |

`admit` 的层归属就是它的全部信息量：**"这个数据在哪一层被供应一次" = "它被复用几次"。** 普通 load 表达不了这件事。

## 3.4 Level —— 语言的核心

Level 与普通 `for` 的区别只有一条：

> **`for` 只说"重复 n 次"；level 说"哪些逻辑值在我这一层诞生、以什么类型的关系交回外层"。**

一个 level 在 IR 中携带：

```
domain          覆盖的逻辑域 + 与父层的 partition 关系（含 tail）
multiplicity    每个父实例中执行多少次
births.state    在此诞生的状态值（new）
births.staged   在此物化一次、被后代只读复用的值（materialize）
handoff         内层结果如何合成回外层状态
```

**语法上，handoff 就是普通 SSA 赋值。** 作者不需要学一套 handoff 代数：

```python
i32_acc += reduce(widen(p16, i32)) * w.sc[s]       # 归约式 handoff
o = o * alpha + contract(p, v, over="tk")          # 重标定式 handoff
```

**但 IR 必须保留层归属。** 这是 level 不退化成 `for` 的唯一保证。删掉它，"值在哪一层诞生、以什么关系交回外层"就无处记录。

**一层承载一段程序**，可包含多棵子树、多个 contract、pointwise 与多个状态更新，不是"一层一个归约"。

**层是可选结构，不是强制的根。** 有稳定层级数值合成的算子用层；数据依赖控制流的算子（sort、Top-K、CSR gather）用普通控制流（见 3.9）。硬要所有东西套一个形状，正是 TileLang 遇到 Ascend 时的病。

## 3.5 逻辑 cohort

```python
with L.rows(M, group=16) as mb:
```

**语义：** 这一层的**一个逻辑实例**共同产生 16 个输出；它们共享同一份该层的输入供应和逻辑状态作用域。

编译器**可以**：分多个物理 issue、用多个 fragment 承载、spill/reload、以不同 LMUL 组合实现。

编译器**不可以**：把它改写成四个源程序级的独立 `group=4` —— 那会重新定义四次 `admit` 和四套逻辑状态，是另一个程序。（与 Triton 不会偷偷把 `BLOCK_M=128` 改成 32 同理。）

```
group=16                       source-visible 的逻辑 realization cohort
register / fragment / issue    它的物理分解
```

**不要说"16 个必须同时物理 live"。**

## 3.6 Core DSL、build config 与目标物理机器

`scalar`、`wide`、`matrix` 与 `transfer` 是目标物理机器中的 engine 类别，不是 core DSL annotation。源程序写：

```python
acc += contract(a, b, over="k", acc=i32)
```

该 operation 的 typed operands、logical axes、Encoding、use-def 和数值语义进入 canonical IR。target profile 再判断有哪些 scalar/RVV/IME realization 合法，并按结构规则选择一种。选择哪类 engine 不改变 canonical result。

若作者要求产物必须使用 IME，否则失败，要求写入 build config：

```text
target = K1
require = uses_extension(IME)
```

requirement 是对 target physical program 的硬谓词。它在结构性选择前过滤不能满足要求的 physical structures，并在 selected program 上再次验证；不满足就拒绝当前 build，不静默退回 RVV。它不进入 canonical Value identity，也不允许编译器修改作者树。

如果 IME 高性能实现需要不同的数据供应树，例如额外的 `materialize`、不同 Level 或 persistent Encoding，作者选择另一份 std 函数。那两份程序的区别来自真实的 logical birth、lifetime 或 artifact，而不是 `@matrix` 权限位。

跨 physical engine 的 register/fragment/local-storage 交接是 target physical conversion。core DSL 不提供 `stage_handoff`；普通 SSA use-def 与 `materialize` 已经给出值边界和生命周期。若某种多引擎执行需要 source-visible 的同步或 effect，必须定义具有该可观察语义的 operation，不能用 engine annotation 暗示。

## 3.7 派生编码与阶段

```python
@weft.derive
def Q4K_I16(W: View[Q4_K, (M, K)]) -> View[Q4K_I16, (M, K)]:
    return interleave(W, rows=16)
```

**这不是"更外面的一层循环"，是类型生成。**

两个阶段必须分开：

```
artifact phase    build / load          产出派生编码类型
execution level   一次 invocation 内     频率层
```

**抽象族与已实例化物理编码要分开：**

```
Q4K_I16<rows=16>              抽象派生编码族
Q4K_I16_R16_LayoutA           已实例化物理编码（含具体字节交错顺序）
```

作者可以用不同 derived builder 或显式 build config 选择 LayoutA / LayoutB，**但一旦 artifact 建好，kernel entry 接收的就是固定的 layout identity，lowering 不能再选另一种交错顺序。** builder 必须产出：

```
builder / size / alignment / endianness / layout identity / target compatibility
```

**为什么必须写在源程序里：** 持久重排改变 artifact、ABI、内存占用、初始化成本、跨调用生命周期。这是跨越调用边界的事实。

**与"权限"的区别：** 写 `repack=allowed` 只有 1 bit（我同意你去找）；写 `interleave(W, rows=16)` 是整个结构（重排是什么、发生几次、谁看得见，全在源码里）。

## 3.8 auto 参数

```python
group=auto(1, 4, 8, 16)
extent=auto("KB")
```

`auto` 是作者声明的有限 source 参数域，不是许可位。它说"这个数由构建过程绑定"，而"这个宽度存在"是作者写的。与 Triton 的 `BLOCK_M` 同类；它可能实例化不同 logical cohort 或 Level extent，因此仍属于作者程序空间。

target 还可以为同一 source tree 声明 LMUL、已固定 schema 内的 physical microtile extent、unroll、pipeline depth 与 buffer count 等有限物理参数。这些参数不进入 DSL 或 canonical IR，由构建期实测选择。source `auto` 与 target physical parameter 可以由同一工具测量，但前者实例化作者程序，后者只实例化机器表示。

没有明确语义、只说“编译器随便找”的 permission bit 不属于语言。

## 3.9 普通控制流

```python
for i in range(N):
    if cond: ...
```

**默认语义：有序标量。编译器不把循环迭代自动变成并行或宽执行。**

逻辑轴必须由作者显式表达。把 `for i in range(8)` 展开成八份独立 SSA，
与写下一个 shape 为 `[8]` 的值，是两份不同的程序：前者只有八次有序标量执行，
后者才存在一个可被表示 pass 映射的 8 元素逻辑轴。编译器不得从同构 SSA、
固定 op 数量或 source closure 中把前者重新识别成后者，也不得自动向量化前者。

宽执行只来自三处：
1. 调用了库中已写好的 realization（`matmul` / `reduce` / `sort` ...）
2. 打开的层级数值 realization（level + cohort）
3. 明确 shaped 值上的局部 operation

编译器仍对普通控制流中的地址计算、常量、局部标量表达式做常规优化。

**理由：** 全篇的原则是"编译器不发明结构"。如果在 irregular kernel 上又说"编译器自己想办法向量化"，边界就崩了。

## 3.10 函数与标准库 —— 只有一个层级

**语言里没有内置的 `matmul`。**

语言只有：level、value、encoding、以及 3.11 的基本 op。

`gemm`、`gemv`、`q4k_gemv`、`sort` 全部是 `weft/std/` 下**用同一门语言写的普通函数**，与你自己写的 kernel 在语言地位上没有任何区别。你可以打开源码读它、复制一份改一行、或者不用它。

**四条规则：**

**(1) 没有黑盒。** 标准库是同语言可读的源码，不是 C++ selector、不是模板特化、不是 emitter 分支。

**(2) 调用是 inline 的。** 不是运行时 call。调用点展开成被调用者的那棵树，与外围代码一起进入同一串 physicalization pass；值的表示沿普通 use-def 继续传播，冲突处显式插入局部 convert，而不是让库和调用者各自隐藏选择。（与 C 的 `static inline` 同类，但更强：C 只 inline 代码，我们 inline 时物理表示尚未决定。）

**(3) 一个名字下可以有多个实现，靠签名选。**

```python
gemv(f32, f32)                    → 一棵树
gemv(Q4K_I16, Q8_K)               → 另一棵树
gemv_panelized(Q4K_I16, Q8_K)      → 带不同 materialize / Level 的另一棵
```

这是普通重载/特化，不是编译器搜索结构。前端根据调用的函数、参数类型与 derived Encoding 唯一解析到一棵树；匹配歧义是前端错误。build config 可以约束 target capability 或最终 physical program，但不能替作者在多棵 std tree 之间选算法结构。构建过程只枚举该树声明的有限 source `auto` 绑定，target compiler 不在多棵 std tree 之间竞赛。

**(4) 参数化的是数值决策，不是机器参数。**

```python
def gemv_quant(W, X, Y, *, cohort=auto(4,8,16), mac_width=2): ...
```

`cohort`、`mac_width` 是数值决策的口子。LMUL、vl、寄存器**永远不出现在签名里**。

**推论：** llama.cpp 那 48 条 quantized vec_dot 的对应物是 **48 个函数** —— 不是 48 个后端路径、不是 48 个模板特化，就是 48 段彼此独立、互相可抄的库代码。新增一个格式 = 一个 encoding 声明 + 一个函数。

> **没有两个层级，没有"普通用户 vs 库作者"，没有特权原语。所有人都在写同一种程序。**

## 3.11 基本 op

每个 op 的定义必须写清 result shape、覆盖范围与数值行为。示例：

```
mac_pairs(a, b, into=i16)
    每 2 个相邻元素的乘积累加为一个 i16
    result shape: [n/2]
    溢出：wrap（作者负责范围）

mac_groups(a, b, n=4, into=i16)
    每 n 个相邻元素的乘积累加为一个 i16

widen(v, dtype)         位宽提升，逻辑值数量不变
iota(n, dtype=u32)      显式建立 shape 为 [n] 的逻辑索引值
reduce(v)               整个逻辑向量归约为一个值
                        （内部可用树归约/多物理 partial——不改变逻辑值集合）
fold2(v)                相邻两项相加，[n] → [n/2]
dot(a, b)               逐元素乘后归约
contract(a, b, over=)   指定轴上的缩并
outer_contract(a,b,over=) 外积式缩并，产出二维结果
lookup(table, idx)      查表
reshape/transpose/index 显式改变 logical axes 或坐标关系
interleave(view, rows=) 跨行交错（派生编码专用）
```

`reshape/transpose/index` 必须完整规定输入输出 logical axes 和坐标映射。它们是 canonical operation，不是 local layout hint。invocation-local pack 不进入 core DSL；target compiler 可以在不改变 logical axes、Level 或 effects 的前提下，为任意合法 Value/use edge 选择 local pack。跨调用的重排仍必须使用 derived Encoding 固定 bytes 与 ABI。

**注意：** `reduce` 内部可以产生多个物理 partial 与树归约 —— 因为源程序里它只有一个逻辑结果值，逻辑值集合没变。而 `i32_acc += ...` 拆成 4 路是从 1 个逻辑值变成 4 个，属于改树，作者写。这不是给 `reduce` 的豁免权，是判据的直接应用。

---

# 第四部分 · 示例

## 4.1 q4_K × q8_K vec_dot

### 派生编码（build phase）

```python
@weft.derive
def Q4K_I16(W: View[Q4_K, (M, K)]) -> View[Q4K_I16, (M, K)]:
    return interleave(W, rows=16)
```

### kernel

```python
@weft.kernel
def q4k_gemv(W: View[Q4K_I16, (M,K)], X: View[Q8_K, (K,)], Y: View[f32, (M,)]):

  with L.rows(M, group=16) as mb:                        # cohort = 16
    f32_acc = new(f32, [16], init=0)                     # births.state

    with L.blocks(K, extent=256) as kb:
      w = admit(W[mb, kb])
      x = admit(X[kb])

      i32_acc = new(i32, [16], init=0)                   # births.state

      with L.subs(extent=32) as s:                       # 8 次 / block
          # 元素层：成对乘积落 i16。2 × 15 × 127 = 3810，安全。
          # 整 32 项压 i16 会溢出（60960），故不这么写。
          p16 = mac_pairs(w.q[s], x.q[s], into=i16)
          # handoff: element → sub，关系是 widen-reduce 后乘 scale
          i32_acc += reduce(widen(p16, i32)) * w.sc[s]

      # min 支路：完全不碰 w.q
      # bsum 粒度 16、sc/m 粒度 32，故 2:1 折叠
      mins = fold2(x.bsum)                               # i16[16] → i32[8]
      min_term = dot(w.m, mins)

      # handoff: block → row。ds 属于当前 Q8_K block，必须在本层乘。
      f32_acc += x.ds * (w.d * i32_acc - w.dmin * min_term)

    commit(f32_acc, Y[mb])
```

### 这段程序写下了什么

| 知识 | 在哪里 |
|---|---|
| 分配律拆出 min 支路 | 两条并列表达式，`w.q` 不出现在 min 支路 |
| min 支路落到预存 bsum | `fold2(x.bsum)` + `dot(w.m, mins)` |
| bsum/sc 粒度错配的处理 | `fold2` |
| `sc` 在 sub 层 | 出现在 `L.subs` 内 |
| `d`、`dmin`、`ds` 在 block 层 | 出现在 `L.blocks` 内 |
| 三层位宽 | `into=i16` / `widen(_, i32)` / `f32_acc` |
| i16 只覆盖 2 项 | `mac_pairs`（作者算过范围） |
| 持久重排 | `Q4K_I16` 派生编码 |
| 同时保留 16 个输出 | `group=16` |

### 格式动物园

| 格式 | 相对上面的差别 |
|---|---|
| Q4_0 | 对称，无 min 支路 → 少一条支路 |
| Q4_1 / Q5_1 | 有 min 无 per-sub scale → min 项在 block 层直接算 |
| Q6_K | 6 bit 跨两个字段 → 编码声明变，分解结构不变 |
| IQ2_XXS | codebook → `L.subs` 里多一个 `lookup(cb, idx)` |

新增一个格式 = **一个 encoding 声明 + 一个函数**。若引入新的 bit extraction 形态或新的 lookup，则加**一个共享的局部 op + 它的 lowering**。禁止的是每格式一条 whole-kernel backend、按格式名分支 —— 不是禁止一切后端扩展。

## 4.2 dense GEMM（BLIS 结构）

```python
@weft.kernel
def gemm(A: View[f32,(M,K)], B: View[f32,(K,N)], C: View[f32,(M,N)]):

  with L.tiles(N, extent=auto("NC")) as nc:
    with L.tiles(K, extent=auto("KC")) as kc:                    # ← KC 层

      Bp = materialize(admit(B[kc, nc]))                          # staged，被所有 MC 复用

      with L.tiles(M, extent=auto("MC")) as mc:
        Ap = materialize(admit(A[mc, kc]))                        # staged，每 MC 一次

        with L.rows(mc, group=auto("MR")) as mb:
          with L.cols(nc, group=auto("NR")) as nb:

            # accumulator 只跨当前 KC：从 C 读入、累加、写回。
            # 这是一个数值决策，不是实现细节 —— 见下。
            acc = new(f32, [MR, NR], init=admit(C[mb, nb]))

            with L.blocks(kc, extent=auto("KB")) as kb:
              a = Ap[mb, kb]
              b = Bp[kb, nb]
              acc += outer_contract(a, b, over="k")

            commit(acc, C[mb, nb])
```

### accumulator 的作用域是一个数值决策

两种写法都合法，它们是**两棵不同的树**：

| 写法 | C 的读写频率 | 累加顺序 |
|---|---|---|
| accumulator 只跨 KC（上面这份） | 每个 KC 读一次写一次 | 分段累加，段间在 f32 上合并 |
| accumulator 跨整个 K（KC 层退化） | 只写一次 | 一路累到底 |

**这正是这门语言要表达的东西。** 前者是经典 BLIS：把 `Bp` 的 staged logical domain 限制在 KC×NC，使该工作集有机会装入 cache，代价是 C 的 read-modify-write。后者省了 C 的往返，代价是 staged working set 可能过大。作者选 logical domain 与 lifetime，target 选择具体 local representation。

### 层级位置就是复用倍数

| 分派 | 含义 |
|---|---|
| `materialize(B)` 在 KC 层 | 每个 (NC, KC) region 形成一个 staged Value，被所有 MC 复用 |
| `materialize(A)` 在 MC 层 | 每个 (MC, KC) region 形成一个 staged Value，被所有 NR 复用 |
| `acc` 在 MR/NR 层 | 跨当前 KC 驻留 |
| `admit` 在 KB 层 | 每 K 块供应一次 |

把 `materialize(B)` 挪进 MC 层，程序仍然正确，但那是另一个程序 —— B 的 staged birth 与供应次数被复制 MC 次。staged Value 最终是原样引用、register window、local-storage panel、RVV tuple 还是 IME operand，由 target physical machine 决定。

## 4.3 GEMV

**GEMV 不是另一个原语，是同一棵树的退化：**

```python
@weft.kernel
def gemv(W: View[f32,(M,K)], X: View[f32,(K,)], Y: View[f32,(M,)]):
  with L.rows(M, group=auto("MR")) as mb:
    acc = new(f32, [MR], init=0)
    with L.blocks(K, extent=auto("KB")) as kb:
      x = admit(X[kb])                       # 被 MR 行共用
      acc += contract(admit(W[mb, kb]), x, over="k")
    commit(acc, Y[mb])
```

相对 GEMM：去掉 `L.cols` 层、accumulator 降为一维，并且 source tree 没有额外的 staged panel birth。target 仍可在同一 Value lifetime 内选择 register window 或 local pack。

**"小 N 走 row-dot、大 N 走 panel GEMM"这个 regime 判断不进语言** —— 作者写的就是他要的那棵树。

对照：把 `admit(X)` 写进 `L.rows` 内部、`group=1`，就是朴素行 dot，X 被读 M 遍。**唯一的区别是 `admit` 在哪一层 + cohort 宽度。**

## 4.4 attention（online softmax）

```python
@weft.kernel
def flash_attn(Q: View[f16,(Tq,D)], K: View[f16,(Tk,D)], V: View[f16,(Tk,D)],
               O: View[f16,(Tq,D)]):

  with L.rows(Tq, group=auto("BQ")) as qb:
    q = materialize(admit(Q[qb, :]))                 # staged
    m = new(f32, [BQ],   init=-inf)                  # state
    l = new(f32, [BQ],   init=0)                     # state
    o = new(f32, [BQ,D], init=0)                     # state

    with L.blocks(Tk, extent=auto("BK")) as kb:
      k = admit(K[kb, :])
      v = admit(V[kb, :])

      s     = contract(q, k, over="d", acc=f32)
      m_new = max(m, rowmax(s))
      p     = exp(s - m_new)
      alpha = exp(m - m_new)

      l = l * alpha + rowsum(p)
      o = o * alpha + contract(p, v, over="tk")
      m = m_new

    commit(o / l, O[qb, :])
```

**这个例子确立三条语义：**

1. **handoff 不限于归约。** `o = o*alpha + ...` 是"先重标定外层状态、再并入内层结果"，比 `+=` 严格更强。
2. **顺序性来自 use-def，不来自标志位。** `alpha` 依赖上一次的 `m`，这条依赖边在程序里显式存在，编译器读 use-def 即可。**没有 `ordered`。**
3. **一层承载一段程序。** 这一层有两个 contract（第二个消费第一个的产物）、多个 pointwise、三个状态更新。

## 4.5 Top-K —— 明确退化

```python
@weft.kernel
def topk(X: View[f32,(N,)], out: View[i32,(K_,)]):
    heap = new(f32, [K_])
    idx  = new(i32, [K_])
    for i in range(N):                    # 普通 for，有序标量
        if X[i] > heap[K_-1]:
            insert(heap, idx, X[i], i)
    commit(idx, out)
```

**这里没有 level，也不该有。** 没有稳定的层级数值合成，没有跨层 typed handoff，只有数据依赖的控制流。

按 3.9，编译器**不会**把这个循环自动向量化。要宽执行的 Top-K，就调用库里用 level 写好的那个版本，或者自己写一个。

**层套不上就不套 —— 这是设计的一部分，不是缺陷。**

---

# 第五部分 · 编译器

## 5.1 作者程序与编译器表示

手写一份 q4_K intrinsic 时，人同时承担两组不同性质的工作。

### 作者写入 canonical tree

1. 代数支路：是否拆 min 支路、使用 bsum 还是重新求和；
2. 逻辑粒度：bsum/sc 怎样折叠，mac 一次覆盖几个元素；
3. 有限位宽：i16、i32、f32 的转换与舍入位置；
4. 频率层次：`d/sc/ds` 分别在哪个 Level 参与；
5. logical values/state/partial 的数量；
6. cohort、Level extent、accumulator lifetime 与 handoff；
7. `admit` 与 `materialize` 位于哪层；
8. persistent derived Encoding 是否存在、bytes 与 ABI 是什么；
9. ordinary scalar control 与显式 shaped axis 的选择。

### 编译器形成 physical program

- logical axes 到 time、lane、register replica、fragment 与 local storage 的表示；
- value 的 SEW、LMUL、`vl`、tail、register group 与 physical partial；
- memory edge 的 load form、unpack、broadcast、conversion 与 reuse；
- local operation 使用 scalar、RVV、IME 或 transfer engine 的 realization；
- invocation-local pack 的存在、axis orientation、physical schema 与 parameterized extents；
- local cluster、pipeline、unroll、prefetch、buffer、spill/reload/rematerialize；
- resource legality、intrinsic 与 local asm。

第二组不是一个统一的“前向推导”，而是三种性质不同的决定。

## 5.2 非 SIMT 物理抽象机器

目标 lowering 把每个 source Value 的 logical coordinates 关联到下列物理分量：

```text
logical coordinates
    → issue/time
    × SIMD lane
    × register replica
    × extension fragment
    × local storage
```

这是一条可组合、可部分定义、允许 broadcast/replication 的表示关系，不是所有 Value 都必须拥有的五维笛卡尔积。scalar、RVV、IME、staged local object 可以只使用其中一部分。

每个物理实体必须保留 canonical Value、logical axes、Level instance、ordinary control、producer/consumer、effect 与 validity 的 identity。物理化可以分片、复制、暂存或重算同一 Value，不能改变 logical value 集合、Level 归属、handoff、pinned Encoding 或 operation 数值语义。

完整机器合同见 `doc/machine/physical-machine.md`。

## 5.3 唯一合法推导

这类事实由语言与 target legality 唯一决定，不能使用启发式：

```text
op/value          free、reduction、broadcast axes；result 保留/消去哪些 axes
encoded use       logical coordinate → storage bits
typed chain       cast/widen/narrow 的 element-width 与数值关系
memory/effect     alias、order、validity、mask 与可移动性
target op         dtype、shape、mask/tail、engine/fragment/resource 是否合法
```

例如 `[M,K]` 沿 K reduce 的结果必须保留 M axis；把它变成 scalar 再广播不是较差实现，而是错误表示。一个 codebook axis 只有在 source shaped Value 中存在时才能映射到 lane，编译器不能从八个独立 scalar lookup 重新发明轴。

同一事实只有一个 producer。上下游 representation 不一致时插入保持 logical identity 的 physical conversion；后续 pass 可以消除或重物化 conversion，但不能用默认字段静默替代。

## 5.4 结构性选择

这类决定不改变 canonical tree，但存在多个合法 physical realization：

```text
axes               time / lane / register replica / fragment 的分配
engine             scalar / wide / matrix / transfer realization
memory             unit / strided / indexed / segment
pack               是否建立 invocation-local pack，以及 axis orientation、carrier与handoff关系
materialization    share / reload / rematerialize / spill / local pack
quant              decode materialize / decode-compute fusion
schedule           sequential / local software pipeline
extension          fragment family 与 handoff structure
```

build config 可以对最终 physical program施加硬 target requirement，例如必须使用 IME；它只过滤结构选择结果，不进入 canonical tree。persistent derived Encoding 的 byte layout 也不是结构选择；它已经由作者和 ABI 固定。

结构性选择使用 target 提供的确定规则和固定优先级。规则可以读取 typed use-def、producer/consumers、Encoding mapping、target legality、live values 与资源上限，并选取第一个合法结构。

Weft 不建立静态 cost model，不生成多个合法结构后用估计或真机运行比较性能。较低优先级结构只在较高优先级结构不合法时使用，不能因为“可能更快”成为隐式备用路径。结构一旦产生，后续 pass 与 emitter 只消费它。

## 5.5 参数性选择

结构固定后，下面参数可以拥有多个有限合法绑定：

```text
value/op            LMUL、physical lane factor、schema内physical microtile extent
Level/cluster       unroll、pipeline depth、buffer count
memory edge         prefetch distance
```

target profile 声明有限参数域；构建期 tuner 为当前 target/shape 编译并实测这些绑定。每个绑定完整经过 legality、resource check 与 emission；非法绑定在运行前拒绝。tuner 不生成新结构，不改变结构优先级，也不让一个非法实现变合法。

source `auto` 与此不同。它由作者/std 声明，可能实例化不同 cohort、Level extent 或其它 source tree；target 不发明其值域。构建工具可以同时测 source `auto` 与 physical parameter，但必须保存二者的不同 authority。

## 5.6 编译流程

```text
按函数调用、类型与静态 source 参数选定 std overload
        ↓
绑定一组作者声明的 source auto 参数
        ↓
Canonical Verify
        ↓
ConvertWeftToRISCV
        ↓
target-aware RISC-V Physical IR
        ↓
layout / memory / operation / pipeline / resource passes
        ↓
VerifyFinalRISCV
        ↓
terminal translation：intrinsic C / local asm
```

Weft 只有两层 MLIR 程序表示：

```text
Canonical Kernel IR             类似 TTIR
RISC-V Physical IR              类似 TTGIR
```

intrinsic C 是第二层的 terminal translation，不是第三层 MLIR。非SIMT物理抽象机器是 RISC-V IR 的语义，不单独形成一层通用 Physical IR。`func/scf/arith` 与 `weft_riscv` 可以同时出现在同一 RISC-V module 中；dialect namespace、target profile、build config、tuner 和 pass 数量都不增加 IR 层次。

构建 driver 在进入 lowering 前绑定 source candidate。结构固定后，有限 physical parameter 的每组绑定各自从 canonical module 建立一份独立 RISC-V module并完整编译；tuner比较的是这些module产生的可执行artifact，不是一张全局assignment表。

没有全局带回边求解器，也没有结构 tournament：唯一事实在所属实体上产生一次；结构 pass 按固定规则选择；layout conflict形成显式conversion；资源不足则插入结构合同允许的spill或判当前module非法；target lowering不修改作者tree，也不转去另一个std overload。

## 5.7 RISC-V Physical IR

RISC-V IR 是一份瞬态、target-aware、typed SSA physical program，不是 canonical program 旁边的 decision record。它至少必须直接表达：

```text
physical value type     logical axes → time/lane/register-replica/fragment mapping
memory descriptor       Encoding、storage mapping、extent/stride/origin/alignment
convert_layout          保持logical identity的真实SSA conversion
memory operations       load/store/encoded access与selected memory form
local objects           local pack、pipeline buffer、spill slot及其lifetime
target operations       selected scalar/RVV/IME/transfer operation
schedule structure      physical loop、cluster、buffer version、prologue/steady/epilogue
resource operations     spill/reload或pure-producer rematerialization
```

layout属于physical value type；representation conflict在具体use edge形成typed `weft_riscv.convert_layout`。conversion elimination必须重写或删除真实operation。local pack、pipeline和spill必须形成真实local object、region与SSA use-def。不能用`realization="..."`、`pipeline_depth=2`或缺字段默认值代替尚未生成的程序结构。

每个physical entity携带source origin，但真实use-def、control、memory和schedule必须存在于RISC-V IR本身；emitter不得借origin回查canonical closure再补程序。

设计不采用第三层`weft_phys` dialect，不采用`weft_riscv.problem` / `weft_riscv.assignment`字典，也不保留兼容路径。未来target实现同一非SIMT机器合同时，替换第二层target-aware physical IR，不插入新的source或physical层。

## 5.8 RISC-V Pass

所有pass改写同一份RISC-V IR：

```text
ConvertWeftToRISCV
    建立target-aware types、ABI descriptor、source origin与一一对应的Level/control

SelectRISCVOperations
    按typed facts、target profile、requirements与固定优先级选择scalar/RVV/IME anchors

PropagateRISCVLayouts
    沿use-def传播完整layout，在不兼容use edge插入typed conversion

PlanRISCVMemory
    物化unit/strided/indexed/segment、encoded access与invocation-local pack

CanonicalizeRISCVLayouts
    conversion propagation/elimination、rematerialization、CSE/DCE与reuse

PipelineRISCVLevels
    在已有Level/loop内生成cluster、physical loop、buffer version与真实pipeline结构

MaterializeRISCVResources
    计算live resources，插入合法spill/reload/rematerialized producer或拒绝module

LowerRISCVComposites
    把composite conversion/memory/contract/schedule降为scf与primitive RVV/IME ops

VerifyFinalRISCV
    确认layout、conversion、memory、schedule、resource与requirements全部闭合
```

这些是pass，不是IR层。pass-local analysis map可以存在，但跨pass结果必须写回physical type、operation、region或附着于真实实体的typed attribute。每个pass后dump同一份RISC-V module，必须能直接看见type变化、conversion插入/删除、memory/target op替换、pipeline展开与spill/reload。

## 5.9 瞬态实体归属

一次 target lowering 内，物理事实归属于具体实体：

```text
value              representation / axes mapping / SEW / LMUL / vl / spill
value-use edge     conversion / broadcast / handoff / rematerialization
op                 selected local realization / physical partial / fragment
memory edge        Encoding mapping / load form / stride / alignment / unpack
Level/cluster      time mapping / tail / unroll / pipeline / buffers / prefetch
target profile     ISA / ABI / VLEN / engines / resources / priorities / parameter domains
build config       source-auto bindings / target requirements
```

链的一端是 pinned memory Encoding，另一端是 selected target operation operand/result contract。内部 conversion、local pack、spill 和 pipeline temporary 不成为 canonical values 或跨调用 artifact。

## 5.10 可改与不可改

```
可以改：
    地址计算的位置
    纯常量计算的位置
    循环不变值的提取（其逻辑诞生层本来就不随内层变）
    一次逻辑 load 落成多少条物理 load
    strip-mining、尾循环
    spill / reload / rematerialize
    闭合 op 内部的物理 partial 与树归约

不可以改：
    把一棵树变成另一棵树
    逻辑值的诞生层
    handoff 的关系与位置
    cohort 的逻辑宽度
    代数支路的存在与否
    中间数值类型
```

**判据永远是 2.2 那一条：移动之后，逻辑值集合和它们的层归属变没变。**

例：把与 sub index 无关的地址加法提出去 —— 可以（数值树没变）。把 `w.d * i32_acc` 下沉到 sub 层 —— 不可以（转换、舍入位置、中间类型全变）。

## 5.11 Verifier 与 emitter

每个 RISC-V pass 必须声明：

```text
允许出现哪些op/type
读哪些program facts
产生、替换或消除哪些program entities
pass后必须满足哪些结构不变量
```

同一决定只有一个 producer。后面的pass只消费，不重新推导或重新选择。pass顺序由MLIR legality、type verifier、operation interface与pass failure约束，不由`stage="representations"`字符串约束。

**需要**（结构与类型）：

```
层的父子域是否合法、partition 是否覆盖（含 tail）
admit / new / materialize / commit 的域是否对应
operand 类型与 shape 是否匹配
encoding 字段宽度、偏移、bit/byte order 是否合法
scope 是否越界
派生 encoding 的 builder 与消费端 layout identity 是否一致
```

**不需要**：

```
证明这个分解等价于某个 I.contract
证明作者为什么选 i16
证明不溢出
证明 bsum 由调用方正确生成
```

> **检查它是不是一份合法的 Weft 程序，不证明它是不是作者想写的那个数学公式。**

数值 repro 的正确性判据与 encoding repro 分开：encoding 布局验证比较同一段
storage bytes 或从中解出的离散字段，允许逐字节一致；浮点 kernel 输出使用有限性
检查与明确的绝对/相对误差容差，不要求 bit-exact。不同合法的乘加结合与 contraction
可以产生末位差异，不能为了复刻 reference 的舍入位置而改作者数值树或阻断合法指令融合。

Terminal translator只接收通过final verifier的RISC-V module。它把`func/scf/cf/arith`写成普通C，把已选RVV/IME operations写成确定intrinsic或typed local asm，并打印已经物化的ABI和pointer arithmetic。它不能生成Level/pack/pipeline/microkernel结构，不能推导layout、memory form、engine、fragment或spill，也不能同时读取canonical module与side record合成physical program。缺失信息必须回报final-RISC-V verifier错误。

## 5.12 编译器绝不做的四件事

```
发现 q4_K 的分配律
发现 min 支路应该走 bsum
决定应该出现 i16 partial
决定应该 persistent interleave 16 行
```

这些由作者或库写下。

> **Weft 不替你想出算法改写；它让你把想出来的那个写下来一次，并把"换一个写法"的代价从两天降到一行。**

## 5.13 输出

RISC-V IR terminal translation生成 intrinsic C（`__riscv_v*` / IME intrinsic），不增加LLVM dialect层，也不把未决定的向量形态交给LLVM自动向量化。系统C compiler继续负责最终寄存器分配、机器调度、peephole和机器码生成。

---

# 第六部分 · RVV / IME 的收益（单目标视角）

**这一节不依赖多后端。** 一个只写 RVV、一辈子不碰别的机器的人，为什么用 Weft。

## 6.1 他现在跟什么搏斗

**(1) LMUL 传染。** 每次 widen，LMUL 翻倍，类型名全变（`u8m1` → `i16m2` → `i32m4`）。元素层选了 `m1`，后面每一层被钉死。想改成 `m2`？整个函数所有类型名、vl 计算、中间变量全部重写 —— 不是一处改动，是几十处。

**(2) 寄存器预算靠脑算。** 16 个输出 × `i32m4` = 64 个寄存器，RVV 只有 32 个。**算错了不报错，是悄悄 spill，性能掉一半。**

**(3) vl 与尾循环。** 每层循环都要算，每个 kernel 都要写，写错就越界。

**(4) VLEN 分裂。** VLEN 128 和 256 上同一份代码性能差很多。llama.cpp 的 RVV 文件里 Q2_K 按 VLEN128/VLEN256 分了两个实现 —— 这是被迫的。

**(5) kernel-local layout ↔ 指令选择绑死。** 决定用 `vwmaccsu`，nibble 的 local unpack 顺序、register window 与 consumer shape 都随之钉死。想试 `vluxei` gather，往往要重写从 load 到 accumulator 的整段 intrinsic。跨调用的 `Q4K_I16` bytes 仍是作者固定的 derived Encoding，不在这项自由里。

## 6.2 Weft 里他不写的

上面五类 target-internal 细节都不写；跨调用可见的 derived Encoding 仍由作者写。他写的是：

```python
p16 = mac_pairs(w.q[s], x.q[s], into=i16)
i32_acc += reduce(widen(p16, i32)) * w.sc[s]
```

没有 `m1`/`m2`/`m4`，没有 `vl`，没有 `__riscv_` 前缀，没有尾循环，没有寄存器预算。

## 6.3 最值钱的一条：他可以**试**了

现在他有一个想不清楚的问题：

> q4_K 的 mac，用 `vwmaccsu` 成对做、还是展开成 i8 用 `vqmaccsu` 四路、还是查表展开？

要知道答案，**要写三遍整个 kernel**，每遍重新想交错顺序、重算 LMUL、重写尾循环。一遍两天。所以现实中他根本不试，选一个凭经验最像的，写完就不动。

Weft 里：

- **展开式拆 nibble 用 and/shift 还是 gather** —— 逻辑值集合没变，是 target 的结构性选择；编译器按固定 legality 规则与优先级决定，作者不写，也不把两种结构交给 tuner 竞赛。
- **成对 mac 换成四路 mac** —— 逻辑值集合变了（i16 partial 的数量和覆盖范围变了），**是另一棵树，但作者只改一行**：`mac_pairs(...)` → `mac_groups(..., n=4)`，该候选重新走完整 physicalization pass 序列。

> **收益不是编译器替你找到最好的结构，而是让"换一个结构"的代价从两天降到一行。**

同时，作者不再手写同一棵树内部的 layout、pack shape、register/fragment mapping、memory form 和 pipeline。这个主张要求编译器拥有真实的物理抽象机器、typed physical program 与编译算法，不能只把固定 emitter 分支包装成 pass。

## 6.4 VLEN 无关与多引擎

- **VLEN** 是 target fact。同一份分解在 VLEN 128 与 256 上经 target rules 与 physical parameter tuning 产生不同的 LMUL、cohort 物理分组与 unroll。**不写两份。**
- **IME** 不是"另一个后端"，是同一台物理机器上的另一个引擎。若同一 source tree 的 axes、materialization 与 operands 同时允许 RVV 和 IME，target 按结构规则选择；若 IME 需要不同的 logical materialization 或 Level，作者写另一份 std tree。`require=uses_extension(IME)` 可以要求 build 结果必须实际使用 IME，否则失败；它不进入 DSL。

## 6.5 抽象的外部检验

不能靠职责表或 pass 名称自证。要用结构不同的 source、consumer 与 target 实际检验：

> **改作者树的任意一项，target 物理表示是否仍能在不手写 LMUL、pack shape、layout、memory form 或 pipeline 的前提下重新形成？**

| 改动 | target physical program | 人要管吗 |
|---|---|---|
| cohort 16 → 8 | LMUL、寄存器分组、交错全变 | 不用 |
| mac 2 元素 → 4 元素 | unpack、instruction、partial、SEW/LMUL 与 memory mapping 全变 | 不用 |
| 加一条新支路 | 多一组值要排 | 不用 |

这不是纸面自证。必须用结构不同的 source、consumer 与 target 实际编译：若作者仍要回去手调 LMUL、物理 pack 或 fragment，说明 source tree 与 physical machine 没有真正分开。

---

# 第七部分 · 多目标

多目标是**特性**，不是附录。但当前聚焦 RVV / IME。

## 7.1 什么跨机器不变

```
根抽象（层级数值 realization）                   不变
语言词汇（level / new / materialize / admit /
          commit / encoding / operation）        不变
数值分解的主体                                    基本不变
数据供应结构                                      仅在作者选择另一 source tree 时变
target profile、物理表示、指令、资源、流水          全变
```

同一 source tree 在不同 target 上可以选择不同 engine、layout、local pack、conversion 和 pipeline。只有当高性能实现确实需要改变 logical materialization、Level、effect 或 persistent artifact 时，作者才写同一门语言中的另一棵程序。把 decode 与 contract 分配到不同物理 engine 本身不要求改树；它们之间若需要 source-visible staged birth 或同步语义，才需要新的 source operation/Level 结构。

## 7.2 与 TileLang 的机制对照

TileLang source 明确暴露 launch/thread binding、shared/local/fragment allocation、copy instruction preference、pipeline stage/order 和 layout。具体位置包括：

- `ref/tilelang/tilelang/language/kernel.py:149`：block/thread indices 与 launch frame；
- `ref/tilelang/tilelang/language/allocate.py:1`：shared/local/fragment/global scope；
- `ref/tilelang/tilelang/language/copy_op.py:54`：copy/TMA/cp.async preference；
- `ref/tilelang/tilelang/language/loop.py:13`：parallel layout 与 software-pipeline metadata。

TileLang compiler 仍通过 `ref/tilelang/src/transform/layout_inference/layout_inference.cc:92`、`ref/tilelang/src/transform/pipeline_planning.cc:36` 与 `ref/tilelang/src/transform/lower_tile_op.cc:43` 推导 layout、分析 pipeline 并 lower tile operation。在所参考版本中，CPU pipeline 将 thread binding 串行化，CPU GEMM lowering 选择 `cpu.scalar`；见 `ref/tilelang/tilelang/cpu/pipeline.py:15` 与 `ref/tilelang/src/cpu/op/gemm.cc:21`。

Weft 不复用这些 source-level storage/thread constructs，而用同一门语言写目标相关的程序树：

```python
# 一份直接消费 decode value 的 source tree
with L.subs(...) as s:
    p16 = mac_pairs(w.q[s], x.q[s], into=i16)
    i32_acc += reduce(widen(p16,i32)) * w.sc[s]

# 若另一实现确实需要不同 logical materialization，作者写另一份 std tree
with L.blocks(...) as kb:
    panel = materialize(decode_all(w))
    acc  += contract(panel, x)
```

同一门语言、同一套 Level 规则与 Encoding。若上面两棵树的 logical births 不同，它们就是两个作者程序；若 logical tree 相同，RVV/IME engine、register↔fragment conversion、local pack 与 physical pipeline 都由 target profile 和物理抽象机器决定。跨调用 persistent artifact 仍由作者固定。

## 7.3 目标贡献什么

```text
ISA / ABI / VLEN
physical engine classes、可实现的 canonical ops 与 operand/result contracts
time / lane / register-replica / fragment / local-storage representations
合法 conversions、memory forms、mask/tail 与 alignment
register、fragment、temporary、buffer 与 local-storage resources
transfer、wait、barrier、issue 与 ordering constraints
固定 structural rules 与 priorities
有限 physical parameter domains
intrinsic / local asm availability
```

target profile 不只是“加几个存储类型”，也不能按 kernel 或格式名提供 whole-kernel route。它改变物理表示和规则，不改变 canonical tree 或 pinned bytes。build requirement 只过滤不满足目标约束的 physical program，不能反向改树。

## 7.4 关于 TPU 一类的诚实评估

TPU TensorCore 也是单控制器，前提成立，层级分派有意义。但 MXU 是 128×128 固定形状，对 contract 的约束**极强**，强到接近 SIMT 的 ownership —— 一旦走 MXU，accumulator 形状、operand 形状、K 分块粒度基本被钉死，编译器自由度小得多。而 `N=1` 的 GEMV 在 TPU 上真实的解是改变对外可见的东西（攒 batch、融合 projection、padding），那是算子边界变了。

> **根抽象对 TPU 成立，但价值小于 RVV/IME。甜区是"表示自由度大 + 单控制器"的机器：RVV、IME、DSP、各类 VLIW NPU。TPU 在边缘。**

不为 TPU 调整根抽象。

## 7.5 语言贡献与编译器贡献分开主张

同一棵树也可以被编到 GPU（行 group → warp，sub/block → 循环，mac → MMA）。所以不能只靠层级结构推出"原生非 SIMT"。

```
语言贡献：   层级化、encoding-aware 的有限位宽数值实现
             —— 对 GPU 也成立，不是非 SIMT 专属

编译器贡献： 面向单控制器 vector/matrix 机器的物理抽象机器、
             typed physical program 与实体级编译算法
             —— 这才是非 SIMT 的部分
```

**语言不必"原生非 SIMT"，它只需要不携带 SIMT ownership。** 非 SIMT 的正面定义是：logical Value 不属于虚拟 thread；编译器把完整值沿 issue/time、lane、register replica、extension fragment 与 local storage 作物理分解。只说“没有线程”不够，完整合同见 `doc/machine/physical-machine.md`。

---

# 第八部分 · 与其他语言的关系

| | 根构造 | 数值分解可否成为一等结构 | 换机器时改什么 |
|---|---|---|---|
| Intent | 逻辑域 + 原子 contract | 否 | — |
| Halide / TVM / Exo | 算法 + 保数值的 schedule | 否（改数值即换 algorithm） | schedule |
| Triton | program grid + block tensor；TTGIR distributed encoding | 可以手写，但 Encoding/Level 数值树不是根模型的一等结构 | source/meta 改 program/block；compiler 改 layout/MMA/pipeline |
| TileLang | thread binding + storage scope + copy/tile op | 可以手写，但 Encoding/Level 数值树不是根模型的一等结构 | source 改 storage/thread/copy structure；compiler 做 layout/tile lowering |
| intrinsic | 全部手写 | — | **全部重写** |
| **Weft** | **层级数值 realization + 非 SIMT 物理机器** | **是** | **作者改数值/Level/ABI；target 改表示、结构规则与参数** |

**论证方式：** 不依赖"别人写不出来"，而依赖 —— **别人能写，但无法在其根模型中自然保存、组合并重新物理化这些信息。**

---

# 第九部分 · 冻结的命题

**1.** Weft 的源语言不选择任何一种目标物理对象作为统一根类型。这是设计选择，不是硬件定理。

**2.** 唯一判据：改变 canonical operation/value/effect graph，或改变 Value 的 Level / artifact 归属 → 树，作者写；只改变同一程序的物理表示 → 编译器。

**3.** Core DSL 不用权限位混淆语义、build config 与物理实现。目标要求属于 config，engine/local pack 属于物理机器。

**4.** Encoding 是逻辑字段到 storage unit/bit range 的纯布局映射，由 grouped/layered/bit-plane/join 等关系组合；不携带不变量或解码数学。

**5.** Level 与 `for` 的区别是"值在哪一层诞生、以什么关系交回外层"。语法上是普通 SSA，IR 必须保留层归属。

**6.** 层是可选结构。普通 `for/if/while` 默认有序标量，编译器不自动向量化。

**7.** `scalar/wide/matrix/transfer` 是物理 engine 类别，不进入 canonical operation 或 Value identity；硬 target requirement 写在 build config。

**8.** 跨调用可见的表示必须写在源程序里；派生编码是类型生成，字节布局在 build 时固定。

**9.** **只有一个层级、一门语言。** 标准库是用同一门语言写的普通函数，inline 展开，无黑盒，无特权原语，无第二类用户。

**10.** 编译器决定分三类：唯一合法推导、固定规则与优先级完成的结构性选择、构建期实测完成的参数性选择。Weft 不使用静态 cost model，也不生成多个物理结构后竞赛。

**11.** 非 SIMT 的正面机器模型是 logical Value 到 issue/time、lane、register replica、extension fragment 与 local storage 的可组合表示关系；不是“没有线程”这一句否定定义。

**12.** 语言贡献与编译器贡献分开主张。

---

# 附录 A · 未闭合问题

1. **跨 engine 异步流水。** register/fragment/local-storage conversion 与 wait/barrier 怎样保持 source use-def/effect；若同步本身对源程序可观察，需要定义真实 effect operation，不能恢复通用 `stage_handoff`。
2. **fragment 的部分 spill 与 handoff。** opaque fragment 能否局部拆分、怎样保持 source Value identity、哪些 reduction 必须在 fragment 内闭合。
3. **local storage 的跨 engine ordering。** 不同 engine 共享 local object 时的 alias、coherence、wait/barrier合同。
4. **动态 loop 与 tail 的 physical time identity。** pipeline version、dynamic trip count、tail mask 与 source Level instance 的完整对应规则。
5. **bit_planes 的公开表面。** 全格式调查已证明它是重复出现的纯布局关系；在没有真实 lowering 消费者前不暴露半成品接口。
6. **Intent → Weft 的生成路径。** 当前设计不依赖它；Weft 独立成立。

# 附录 B · 已排除的候选（防重走）

```
worker 与 program tile 的区别
VLA / ordered control / state / workspace 的能力清单
作者显式写 lane / register / fragment
reuse scope、working set、materialization intent
space-time block
reduction stream + resident state
Kernel Program + Local Realization Program 两份用户程序
"CPU = vdot，GPU = gemm"
权限位（allow_reassociation / allow_repack / ...）
invocation-local `pack(along=...)` 作为 core DSL operation
source-level `@wide/@matrix/@transfer` 与 `stage_handoff`
数学等价证明器与溢出证明器
Level 上的 ordered 属性
闭合原语作为特权层（"普通用户 vs 库作者"两个层级）
全局 C/D dictionary、arc propagation 与 DFS 回溯求解器
静态 cost model 作为结构选择权威
生成多个合法物理结构后静态排序或真机竞赛
```
