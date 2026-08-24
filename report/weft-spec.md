# Weft 规范

**版本 0.2 · 面向 RVV / IME**

---

## 摘要

Weft 是一门层级化、encoding-aware、有限位宽的**数值 realization 语言**。

作者写下一份具体的数值实现：哪些中间值存在、每个值在哪一层诞生、以什么类型的关系交回外层、哪些代数支路被拆出去、同时保留多少个输出、哪些重排发生在调用之外。

编译器不发明代数结构。它用一串前向 physicalization pass 落实这份实现：SEW/LMUL 沿值链传播、vl 与尾循环、寄存器分组、packed value 的读取形态、指令选择、引擎绑定、unroll 与流水。`std` 特化和 `auto` 参数在最外层形成候选；每个候选独立走完整 pass 序列，资源不合法就被删除。

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
| Triton | 带 ownership 的静态 tile | layout 系统围绕 SIMT ownership 建模；layout 维度里没有"编码"，表达不了 nibble→i8→i16 pair 这条链 |
| TileLang | 存储层级 + copy/gemm | shared/local/fragment 是源语言词汇，换机器要改语言 |
| intrinsic | 全部手写 | 能写，但分解与表示写死在同一份代码里 |

**注意措辞：** 不是"别人写不出来"。Triton/TileLang 里当然可以手写 decode、乘法、归约和 correction 支路。真正的区别是：**这些结构不是它们根模型的一等对象，无法在其中自然保存、组合并重新物理化。**

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

SIMT 硬件**强制** ownership：一个值必须被一组线程共同拥有，layout 就是"逻辑坐标 → owner"这个函数，`tile` 是它的名字。tile 一写下来，编译空间就被结构性划定 —— 这不是审美，是硬件逼出的特权对象。

单控制器机器没有这个强制。控制器看得见整个值，没有谁拥有谁。

这解释了为什么一系列候选全部失败：worker、VLA region、space-time block、reduction stream + resident state —— 它们都在找特权对象。

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

> **改变了逻辑值集合，或改变了逻辑值的层归属 → 属于源程序（树），作者写。**
> **不改变这两者 → 属于编译器（表示）。**

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
| LMUL / vl / 寄存器分组 | 没变 | 编译器 |
| strip-mining、spill/reload、rematerialize | 没变 | 编译器 |

**这条判据同时定义了作者写什么和编译器做什么，它比任何一个"根抽象的名字"更根本。**

## 2.3 三条被否决的诱惑

设计过程中反复出现、必须拒绝的三类构造：

**(1) 权限位** —— `allow_reassociation` / `allow_repack` / `allow_low_precision`。

编译器里没有"权限"。真实编译器只有语义和类型。`-ffast-math` 看似权限，实则换了一门语言（float 加法定义为可结合的语言）。权限的正确归宿是变成 op 的定义或类型的一部分 —— Triton 把结合性写进 `tl.sum` 的定义、把精度写成 `tl.dot` 的参数，而不是给全局开关。

**(2) 数学等价证明** —— 编译器不需要证明 min 支路等价于原式，因为**编译器根本不知道原式存在**。Weft 程序里没有那个数学定义。作者写 `dot(w.m, mins)` 就是要算这个式子，和 C 里写 `a*b+c` 一样。分配律是作者做的，做错了就是算错。

**(3) 溢出证明** —— `into=i16` 是作者的选择，不是断言。溢出是作者的责任，和 C 里写 `int16_t sum` 一样。最多在可选调试模式插检查。

**通用规则（三次教训的共同形式）：**

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
| `materialize` | 在本层物化一次、被后代只读复用 | packed panel |

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

## 3.6 Engine role

四个抽象角色，**不是 ISA 名字**：

```
scalar     标量控制与地址
wide       宽向量计算
matrix     矩阵引擎
transfer   数据搬运能力
```

**硬绑定：** `op @ wide` 表示这个 realization 明确要求宽向量引擎，编译器**不会**把它落到矩阵引擎。要用 IME，就写另一棵标注 `@matrix` 的树（可以是同一个函数名的另一个特化）。

含糊的第三种状态（"写了 wide 但编译器可能用 matrix"）不存在 —— 那等于把 engine role 变成 hint，即变相的权限位。

**`transfer` 是能力不是硬件。** RVV 的普通 load/store 可以实现 transfer role；有独立搬运单元的目标映射到它自己的机制。不要求目标存在独立 DMA 单元。

两个动词：

```python
materialize(expr)     # 在这里形成一次真实的跨阶段值
handoff(value)        # 一个阶段的值交给另一个阶段
```

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

build-time 可以在 LayoutA / LayoutB 之间选择，**但一旦 artifact 建好，kernel entry 接收的就是固定的 layout identity，lowering 不能再选另一种交错顺序。** builder 必须产出：

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

`auto` 是**搜索空间的声明**，不是许可位。它说"这个数由候选选择定"，而"这个宽度存在"是作者写的。与 Triton 的 `BLOCK_M`、TileLang 的 `num_stages` 同类。

要删除的只是没有明确含义、只说"编译器随便找"的 permission bit —— 前者把语义责任推给编译器，后者只是留一个数不填。

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
gemv(Q4K_I16, Q8_K, engine=matrix) → 又一棵
```

这是普通重载/特化，不是编译器搜索结构。**编译器仍不发明树，它只在几棵人写好的树里挑。**

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
pack(view, along=)      重排为便于连续访问的形态
interleave(view, rows=) 跨行交错（派生编码专用）
```

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
      w = admit(W[mb, kb]) @ transfer
      x = admit(X[kb])     @ transfer

      i32_acc = new(i32, [16], init=0)                   # births.state

      with L.subs(extent=32) as s:                       # 8 次 / block
          # 元素层：成对乘积落 i16。2 × 15 × 127 = 3810，安全。
          # 整 32 项压 i16 会溢出（60960），故不这么写。
          p16 = mac_pairs(w.q[s], x.q[s], into=i16)             @ wide
          # handoff: element → sub，关系是 widen-reduce 后乘 scale
          i32_acc += reduce(widen(p16, i32)) * w.sc[s]          @ wide

      # min 支路：完全不碰 w.q
      # bsum 粒度 16、sc/m 粒度 32，故 2:1 折叠
      mins = fold2(x.bsum)                               # i16[16] → i32[8]
      min_term = dot(w.m, mins)                          @ wide

      # handoff: block → row。ds 属于当前 Q8_K block，必须在本层乘。
      f32_acc += x.ds * (w.d * i32_acc - w.dmin * min_term)     @ wide

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

      Bp = materialize(pack(B[kc, nc], along="k")) @ transfer    # staged，被所有 MC 复用

      with L.tiles(M, extent=auto("MC")) as mc:
        Ap = materialize(pack(A[mc, kc], along="k")) @ transfer  # staged，每 MC 一次

        with L.rows(mc, group=auto("MR")) as mb:
          with L.cols(nc, group=auto("NR")) as nb:

            # accumulator 只跨当前 KC：从 C 读入、累加、写回。
            # 这是一个数值决策，不是实现细节 —— 见下。
            acc = new(f32, [MR, NR], init=admit(C[mb, nb]))

            with L.blocks(kc, extent=auto("KB")) as kb:
              a = admit(Ap[mb, kb]) @ transfer
              b = admit(Bp[kb, nb]) @ transfer
              acc += outer_contract(a, b, over="k") @ wide

            commit(acc, C[mb, nb])
```

### accumulator 的作用域是一个数值决策

两种写法都合法，它们是**两棵不同的树**：

| 写法 | C 的读写频率 | 累加顺序 |
|---|---|---|
| accumulator 只跨 KC（上面这份） | 每个 KC 读一次写一次 | 分段累加，段间在 f32 上合并 |
| accumulator 跨整个 K（KC 层退化） | 只写一次 | 一路累到底 |

**这正是这门语言要表达的东西。** 前者是经典 BLIS：把 `Bp` 限制在 KC×NC 使 panel 装得进 cache，代价是 C 的 read-modify-write。后者省了 C 的往返，代价是 panel 可能过大。作者选，不是编译器选。

### 层级位置就是复用倍数

| 分派 | 含义 |
|---|---|
| `pack(B)` 在 KC 层 | 每个 (NC, KC) panel 打包一次，被所有 MC 复用 |
| `pack(A)` 在 MC 层 | 每个 (MC, KC) panel 打包一次，被所有 NR 复用 |
| `acc` 在 MR/NR 层 | 跨当前 KC 驻留 |
| `admit` 在 KB 层 | 每 K 块供应一次 |

把 `pack(B)` 挪进 MC 层，程序仍然正确，但那是另一个程序 —— B 被重复打包 MC 次。

## 4.3 GEMV

**GEMV 不是另一个原语，是同一棵树的退化：**

```python
@weft.kernel
def gemv(W: View[f32,(M,K)], X: View[f32,(K,)], Y: View[f32,(M,)]):
  with L.rows(M, group=auto("MR")) as mb:
    acc = new(f32, [MR], init=0)
    with L.blocks(K, extent=auto("KB")) as kb:
      x = admit(X[kb]) @ transfer            # 被 MR 行共用
      acc += contract(admit(W[mb, kb]), x, over="k") @ wide
    commit(acc, Y[mb])
```

相对 GEMM：去掉 `L.cols` 层、accumulator 降为一维、X 太小不 pack。

**"小 N 走 row-dot、大 N 走 panel GEMM"这个 regime 判断不进语言** —— 作者写的就是他要的那棵树。

对照：把 `admit(X)` 写进 `L.rows` 内部、`group=1`，就是朴素行 dot，X 被读 M 遍。**唯一的区别是 `admit` 在哪一层 + cohort 宽度。**

## 4.4 attention（online softmax）

```python
@weft.kernel
def flash_attn(Q: View[f16,(Tq,D)], K: View[f16,(Tk,D)], V: View[f16,(Tk,D)],
               O: View[f16,(Tq,D)]):

  with L.rows(Tq, group=auto("BQ")) as qb:
    q = materialize(admit(Q[qb, :])) @ transfer      # staged
    m = new(f32, [BQ],   init=-inf)                  # state
    l = new(f32, [BQ],   init=0)                     # state
    o = new(f32, [BQ,D], init=0)                     # state

    with L.blocks(Tk, extent=auto("BK")) as kb:
      k = admit(K[kb, :]) @ transfer
      v = admit(V[kb, :]) @ transfer

      s     = contract(q, k, over="d", acc=f32)  @ wide
      m_new = max(m, rowmax(s))                  @ wide
      p     = exp(s - m_new)                     @ wide
      alpha = exp(m - m_new)                     @ wide

      l = l * alpha + rowsum(p)                  @ wide
      o = o * alpha + contract(p, v, over="tk")  @ wide
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

## 5.1 决策清单与实体归属

手写一份 q4_K RVV intrinsic，人要定的全部决策：

### A 组 · 数值分解（6 项，**作者**）

1. 分配律怎么拆 —— 拆不拆出 min 支路
2. min 支路落到哪个字段 —— 用 bsum 还是重新求和
3. bsum 粒度 16 vs sc 粒度 32 怎么折叠
4. 中间位宽的层次 —— i16 → i32 → f32
5. `d` / `sc` / `ds` 各自乘在哪一层
6. mac 一次吃几个元素（2 / 4 / 8）—— 决定 i16 能装多少

### B 组 · 数据供应结构（4 项，**作者**）

7. 同时保留几个输出（cohort 宽度）
8. 几路 partial 累加器
9. activation 在哪一层被读一次
10. weight 要不要跨调用重排、重排成什么形状

### C 组 · 向量形态（9 项，**编译器**）

11. 某个 **value** 的 SEW
12. 某个 **value** 的 LMUL（widen 会沿 use-def 传播）
13. 某个 **Level/value** 的 vl
14. 某个 **Level** 的尾循环
15. 某组 **state/result values** 怎样分组进寄存器
16. 整个候选的寄存器预算是否合法；spill 归属于具体 **value**
17. 某个分组乘加 **result value** 的 partial layout
18. 某个 **reduce op** 的横向归约位置
19. 当前 **target/entry** 的 VLEN specialization

### D 组 · 指令与内存形态（7 项，**编译器**）

20. 某个 packed **value-use edge** 的拆法：and+shift / 专用 unpack / gather
21. 某个 **mac/contract op** 的指令：`vwmaccsu` / `vqmaccsu` / IME
22. 某个 scale **value-use edge** 怎样广播到 consumer mapping
23. 某个 **memory object / derived encoding instance** 的字节级顺序
24. 某个 **memory edge** 的 load form、步长与对齐
25. 某个 **Level + memory edge** 的 prefetch 距离
26. 某个 **Level / local op cluster** 的软流水深度与 unroll

```
写 intrinsic：26 项
写 Weft：    10 项（A + B）
编译器 passes 决定：16 类实体级属性（C + D）
```

这些不是 16 个 kernel-global 开关。一个 kernel 可以同时有多个 packed value、多个 reduce、多个 memory edge 和多个 Level；每个实体都可以得到不同决定。只有 target facts 与总寄存器预算是全局输入/检查。

## 5.2 前向 pass 序列，不是全局求解器

编译器采用和常规 MLIR compiler 一致的形态：每个 pass 读取 canonical IR 与前序 pass 已写下的瞬态 attribute，决定一类实体事实，再交给后续 pass。

```text
std specialization × auto bindings
        ↓ 最外层枚举完整 source candidate
InferEncodingMappings
        为每个 encoded field/value-use 展开 grouped/layered mapping
PropagateValueRepresentations
        为每个 value 决定 SEW、LMUL、vl 与必要 convert
SelectLocalOperations
        为每个 op / memory edge 选择 local RVV/IME operation
ScheduleLevels
        为每个 Level 决定 unroll、local pipeline 与 prefetch
CheckResources
        汇总 live values；合法则保留，不够则使用已明确决定的 spill，
        否则把当前完整 candidate 标为非法
EmitIntrinsicC
        只读所有决定，机械拼写 intrinsic C / local asm
        ↓
rank 或真机实测所有合法 candidate，选择 winner
```

**pass 内前向，pass 之间靠 entity-local attribute 传递。没有回边。**

- layout 冲突不回头重选：在冲突的 value-use edge 插入明确 convert，后续 pass 可以消除冗余 convert；
- 资源不足不回头修改 LMUL、instruction 或作者树：当前 candidate spill 或非法；
- 要探索另一 LMUL、mac realization、unroll 或 std 数值特化，就形成另一个外层 candidate，重新走完整流水；
- 一个 pass 不读取 emitter 状态，也不修改更早 pass 已冻结的实体决定。

这不意味着各项彼此独立。instruction 会读取 operand representation，schedule 会读取 instruction latency，resource check 会汇总所有 live values；依赖通过前向数据流体现，而不是通过全局 dictionary 上的 arc propagation 与 DFS 回溯体现。

## 5.3 瞬态决定的归属

一次 target lowering 内，决定直接挂在它描述的实体上：

```
value attr         physical kind / SEW / LMUL / vl / register bundle / spill
op attr            selected local realization / operand-result handoff
memory-edge attr   encoding mapping / load form / stride / alignment / unpack
Level attr         tail / unroll / pipeline / prefetch schedule
target facts       ISA / ABI / VLEN / extensions / global resource budget
```

链的两端仍被钉死：一端是内存 encoding mapping，一端是 selected instruction operand form。中间的 value-use 冲突由显式 convert 表示。

**pin 只出现在跨调用边界**：kernel 参数、持久 buffer、调用方可见 workspace。**pin 处不许插 coercion**；内部的 coercion 是编译器自己的成本。

## 5.4 候选在 pass 流水之外枚举

```
程序 schema
  × 库函数的多个特化（不同的树）
  × auto 参数实例
        ↓
  对每个完整候选，运行一遍完整前向 pass 序列
        ↓
  删除不可行候选（寄存器不足、无合法表示链、引擎不匹配）
        ↓
  静态代价排序 或 真机实测
        ↓
  选出完整 winner
```

候选之间可以具有不同 std 特化、`auto` 参数和 target-local physical config；候选内部不回退重选。因而“树不可改”的准确含义仍是：

> **对每一个已实例化的候选而言，树不可改。**
> 资源检查只能接受或拒绝这个候选，不能把它改成另一棵树。

**编译器仍然不发明树** —— source candidate 来自库中人写好的特化和 `auto` 的取值范围，不来自结构搜索。

## 5.5 可改与不可改

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

## 5.6 Pass 契约、Verifier 与 emitter

每个 physicalization pass 必须声明：

```text
读哪些 canonical facts / 前序 attributes
写到哪类具体实体
缺失或冲突时插入什么 convert，或返回什么 unsupported
```

同一决定只有一个 producer。后面的 pass 只消费，不重新推导。

**需要**（结构与类型）：

```
层的父子域是否合法、partition 是否覆盖（含 tail）
admit / new / materialize / commit 的域是否对应
operand 类型与 shape 是否匹配
encoding 字段宽度、偏移、bit/byte order 是否合法
scope 是否越界
engine role 是否与 op 兼容
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

Emitter 是最后一个 pass。它只读 selected value/op/memory-edge/Level attributes，负责普通 C、RVV intrinsic、ABI 和 typed local asm 拼写。Emitter 缺信息必须回报前序 pass 契约缺口；不得扫描 source closure、根据 dtype/shape/VLEN/格式名补选结构，也不得写回任何 physical attribute。

## 5.7 编译器绝不做的四件事

```
发现 q4_K 的分配律
发现 min 支路应该走 bsum
决定应该出现 i16 partial
决定应该 persistent interleave 16 行
```

这些由作者或库写下。

> **Weft 不替你想出算法改写；它让你把想出来的那个写下来一次，并把"换一个写法"的代价从两天降到一行。**

## 5.8 输出

生成 intrinsic C（`__riscv_v*` / IME intrinsic），不生成 IR 交给 LLVM 做向量化。理由：向量形态已经由 Weft passes 明确决定，不能再交给系统编译器重新猜。

---

# 第六部分 · RVV / IME 的收益（单目标视角）

**这一节不依赖多后端。** 一个只写 RVV、一辈子不碰别的机器的人，为什么用 Weft。

## 6.1 他现在跟什么搏斗

**(1) LMUL 传染。** 每次 widen，LMUL 翻倍，类型名全变（`u8m1` → `i16m2` → `i32m4`）。元素层选了 `m1`，后面每一层被钉死。想改成 `m2`？整个函数所有类型名、vl 计算、中间变量全部重写 —— 不是一处改动，是几十处。

**(2) 寄存器预算靠脑算。** 16 个输出 × `i32m4` = 64 个寄存器，RVV 只有 32 个。**算错了不报错，是悄悄 spill，性能掉一半。**

**(3) vl 与尾循环。** 每层循环都要算，每个 kernel 都要写，写错就越界。

**(4) VLEN 分裂。** VLEN 128 和 256 上同一份代码性能差很多。llama.cpp 的 RVV 文件里 Q2_K 按 VLEN128/VLEN256 分了两个实现 —— 这是被迫的。

**(5) 交错顺序 ↔ 指令选择绑死。** 决定用 `vwmaccsu`，nibble 拆出来的顺序就被钉死，`Q4K_I16` 的字节排法就被钉死。想试 `vluxei` gather？整个 kernel 从 load 到 store 全部重写。

## 6.2 Weft 里他不写的

上面五条，一条都不写。他写的是：

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

- **展开式拆 nibble 用 and/shift 还是 gather** —— 逻辑值集合没变，**同一份源程序的两个表示解**，编译器解。
- **成对 mac 换成四路 mac** —— 逻辑值集合变了（i16 partial 的数量和覆盖范围变了），**是另一棵树，但作者只改一行**：`mac_pairs(...)` → `mac_groups(..., n=4)`，该候选重新走完整 physicalization pass 序列。

> **收益不是编译器替你找到最好的结构，而是让"换一个结构"的代价从两天降到一行。**

这个主张不需要编译器有任何超人能力，只需要实体级 physicalization pass 与候选枚举做对。

## 6.4 VLEN 无关与双引擎

- **VLEN** 是 target fact。同一份分解在 VLEN 128 与 256 的候选流水中产生不同的 LMUL、cohort 物理分组与 unroll。**不写两份。**
- **IME** 不是"另一个后端"，是同一台机器上的另一个引擎。同一份数值分解，在 `@wide` 树和 `@matrix` 树之间选 —— 两棵树共享 A 组的全部 6 项，只在 B 组和 engine 标注上不同。

## 6.5 抽象是否正确的纸面检验

不需要机器就能判：

> **改 A/B 组的任意一项，C/D 组是否需要人重新介入？**

| 改动 | C/D 组 | 人要管吗 |
|---|---|---|
| cohort 16 → 8 | LMUL、寄存器分组、交错全变 | 不用 |
| mac 2 元素 → 4 元素 | 20/21/23/17/11/12 全变 | 不用 |
| 加一条新支路 | 多一组值要排 | 不用 |

**三条都成立 → 抽象对。任何一条里人还得回去调 LMUL → A/B 与 C/D 没真分开 → 抽象假。**

---

# 第七部分 · 多目标

多目标是**特性**，不是附录。但当前聚焦 RVV / IME。

## 7.1 什么跨机器不变

```
根抽象（层级数值 realization）                   不变
语言词汇（level / new / materialize / admit /
          commit / encoding / engine role）      不变
数值分解的主体（A 组 6 项）                       基本不变
数据供应结构（B 组）                              大部分不变
引擎边界与物化位置                                变
物理表示、指令、资源、流水                         全变（编译器解）
```

**"接新硬件源程序一个字不改"是错的。** 正确的说法：

> **同一门语言的两个不同程序，而不是两门语言。**

一台把 decode 和 matmul 放在两个物理单元上的机器，本来就该有不同的树。要求树跨机器不变，等于要求性能跨机器不变。

## 7.2 与 TileLang-Ascend 的对照

TileLang 为支持 Ascend，往**源语言**里加了：Cube/Vector 执行域、UB/L1/L0 存储关键字、跨引擎 handoff、隐藏 workspace、同步原语、重定义 `T.Parallel`。**GPU 的 TileLang 程序和 Ascend 的 TileLang 程序不是同一门语言的两个程序。**

Weft 的做法：

```python
# RVV：decode 与 contract 交错
with L.subs(...) as s:
    p16 = mac_pairs(w.q[s], x.q[s], into=i16)      @ wide
    i32_acc += reduce(widen(p16,i32)) * w.sc[s]    @ wide

# 分离引擎的机器：整块 decode → 物化 → 交给 matrix engine
with L.blocks(...) as kb:
    panel = materialize(decode_all(w) @ wide)
    acc  += contract(handoff(panel), x) @ matrix
```

同一门语言、同一套 level 规则、四个抽象角色 + 两个动词。存储层级不在源语言里 —— 它是目标贡献的表示格点。

> **它们改语言，我们改程序。**

## 7.3 目标贡献什么

```
representation      可用的表示格点与合法转换
engine binding      哪些 op 能落到哪个角色
handoff / transfer  跨引擎交接的代价与同步要求
schedule 约束       流水深度、issue 限制、资源方程
```

不只是"加几个存储类型"。

## 7.4 关于 TPU 一类的诚实评估

TPU TensorCore 也是单控制器，前提成立，层级分派有意义。但 MXU 是 128×128 固定形状，对 contract 的约束**极强**，强到接近 SIMT 的 ownership —— 一旦走 MXU，accumulator 形状、operand 形状、K 分块粒度基本被钉死，编译器自由度小得多。而 `N=1` 的 GEMV 在 TPU 上真实的解是改变对外可见的东西（攒 batch、融合 projection、padding），那是算子边界变了。

> **根抽象对 TPU 成立，但价值小于 RVV/IME。甜区是"表示自由度大 + 单控制器"的机器：RVV、IME、DSP、各类 VLIW NPU。TPU 在边缘。**

不为 TPU 调整根抽象。

## 7.5 语言贡献与编译器贡献分开主张

同一棵树也可以被编到 GPU（行 group → warp，sub/block → 循环，mac → MMA）。所以不能只靠层级结构推出"原生非 SIMT"。

```
语言贡献：   层级化、encoding-aware 的有限位宽数值实现
             —— 对 GPU 也成立，不是非 SIMT 专属

编译器贡献： 面向单控制器 vector/matrix 机器的实体级物理化 pass
             —— 这才是非 SIMT 的部分
```

**语言不必"原生非 SIMT"，它只需要不携带 SIMT 包袱。** Triton 的问题不是它能表达 tile，是 layout 系统围绕 ownership 建模，搬到单控制器机器上必须先拆掉再重新恢复。Weft 的层不携带 ownership，所以不需要被拆 —— 这就够了。

---

# 第八部分 · 与其他语言的关系

| | 根构造 | 数值分解可否成为一等结构 | 换机器时改什么 |
|---|---|---|---|
| Intent | 逻辑域 + 原子 contract | 否 | — |
| Halide / TVM / Exo | 算法 + 保数值的 schedule | 否（改数值即换 algorithm） | schedule |
| Triton | 带 ownership 的静态 tile | 否（layout 无编码维度） | 后端重新恢复被抹掉的结构 |
| TileLang | 存储层级 + copy/gemm | 否（一等结构是 buffer/copy） | **改抽象机器** |
| intrinsic | 全部手写 | — | **全部重写** |
| **Weft** | **层级数值 realization** | **是** | **程序的引擎边界；语言不变** |

**论证方式：** 不依赖"别人写不出来"，而依赖 —— **别人能写，但无法在其根模型中自然保存、组合并重新物理化这些信息。**

---

# 第九部分 · 冻结的命题

**1.** Weft 的源语言不选择任何一种目标物理对象作为统一根类型。这是设计选择，不是硬件定理。

**2.** 唯一判据：改变了逻辑值集合或它们的层归属 → 树，作者写；没变 → 表示，编译器解。

**3.** 源程序里没有"许可"，编译器里没有"证明"。想加标志位时，先问谁读它。

**4.** Encoding 是逻辑字段到 storage unit/bit range 的纯布局映射，由 grouped/layered/bit-plane/join 等关系组合；不携带不变量或解码数学。

**5.** Level 与 `for` 的区别是"值在哪一层诞生、以什么关系交回外层"。语法上是普通 SSA，IR 必须保留层归属。

**6.** 层是可选结构。普通 `for/if/while` 默认有序标量，编译器不自动向量化。

**7.** Engine role 硬绑定。`transfer` 是能力不是硬件。

**8.** 跨调用可见的表示必须写在源程序里；派生编码是类型生成，字节布局在 build 时固定。

**9.** **只有一个层级、一门语言。** 标准库是用同一门语言写的普通函数，inline 展开，无黑盒，无特权原语，无第二类用户。

**10.** "树不可改"是对每个已实例化的候选而言；候选在最外层枚举，每个候选独立走完整前向 pass 序列，资源检查只接受或拒绝。

**11.** 语言贡献与编译器贡献分开主张。

---

# 附录 A · 未闭合问题

1. **多引擎流水的表达。** `materialize` + `handoff` 够不够表达双缓冲式的跨引擎 overlap，还是需要第三个动词。
2. **tail 语义。** level 的 partition 在非整除域上如何定义，尤其是 cohort 宽度除不尽时。
3. **bit_planes 的公开表面。** 全格式调查已证明它是重复出现的纯布局关系；在没有真实 lowering 消费者前不暴露半成品接口。
4. **代价模型。** 静态排序候选需要什么粒度的机器模型；什么时候必须落到真机实测。
5. **Intent → Weft 的生成路径。** 当前设计不依赖它；Weft 独立成立。

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
数学等价证明器与溢出证明器
Level 上的 ordered 属性
闭合原语作为特权层（"普通用户 vs 库作者"两个层级）
全局 C/D dictionary、arc propagation 与 DFS 回溯求解器
```
