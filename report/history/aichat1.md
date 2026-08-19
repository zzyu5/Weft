我先把结论说死：

> **当前 Weft 不是单纯的“高性能模板选择器”，但也还不是 Triton 那种成熟的布局编译器。它现在是一个由显式 op 约束的、面向单核局部程序的物理程序生成器。**

它真正生成的已经包括轴映射、寄存器重复、局部循环、load/compute 顺序、部分流水、值之间的交接和具体 intrinsic；不是只选择一个完整手写 kernel。
但很多慢项的根因是：**编译空间里根本没有包含 baseline 使用的高质量微内核结构**。这时候调参和 cost model 都救不了。

这份报告记录的主线确实已经从“识别人工高性能结构，再填 LMUL”改成了 op 语义、轴、值链、访存和生命周期共同产生映射、局部调度、资源过滤与硬件操作。 

---

# 一、Weft 的编译空间现在到底是什么

可以把当前实现压成一个明确的集合：

[
\mathcal C(P,T)
===============

{(M,S,H,p)\mid
\text{语义合法、访存合法、依赖合法、资源合法}}
]

其中：

* (P)：作者写下的 DSL kernel；
* (T)：目标机器事实；
* (M)：逻辑轴到机器维度的映射；
* (S)：局部执行顺序与流水；
* (H)：最终使用的 RVV、IME 等局部硬件操作；
* (p)：LMUL、微块、unroll、buffer 数量等数值参数。

## 1. 程序首先提供事实，而不是候选模板

编译器从 Kernel IR 读取：

```text
逻辑轴是谁
哪些轴是 free / reduction / broadcast / packed / group
一个值由谁产生、被谁使用、活到哪里
pointer 沿每条轴怎样变化
哪些 operation 有副作用
state 是否有顺序要求
作者显式写了 dot、matmul、decode 还是 lookup
```

目标提供：

```text
VLEN
合法 SEW / LMUL
向量寄存器数量
segment / indexed / widening 能力
IME fragment
```

这些都不是性能模板名称。报告中已经删除了以 `VLEN128/256`、固定 32/64 lanes 为实现身份的路径。

## 2. 轴映射构造机器形态

当前共同坐标是：

```text
sequential
lane
register
unroll
fragment
```

例如一个 `W.matmul` 的 M、N、K 轴，可能被映射成：

```text
N 的一部分 → RVV lanes
M 的一部分 → 多个 vector accumulator
K 的一部分 → unroll
剩余因子   → primitive 内顺序循环
```

或者：

```text
M/N/K 的局部因子 → IME fragment
剩余因子          → 顺序循环
```

这一步不是从一个 `gemm_k1()` 和 `gemm_sg()` 中二选一，而是在同一个轴空间里形成不同实例。

## 3. 映射继续生成局部程序

报告里新增的这些结构很关键：

```text
LocalOperandWindow
LocalPipelineAction
LocalPipelineSchedule
LocalDecodeSchedule
LocalMicrokernelSchedule
```

`buildLocalMicrokernelSchedule()` 会从已经选出的轴映射计算：

* primitive 内顺序迭代多少次；
* SIMD lane factor；
* 有多少个寄存器 accumulator；
* K-unroll；
* 一个 operand load 服务多少 accumulator；
* load window；
* buffer 数量；
* 明确的 Load / Compute 动作序列；
* quant decode 的 chunk、group、reduction segment。

也就是说，它确实在**生成微内核循环和执行顺序**，而不是只给预写 C 模板填一个 LMUL。

## 4. 资源过滤决定哪些生成程序能存在

候选中的这些东西会共同消耗寄存器：

```text
输入 operand
多个 accumulator
index / mask
state
decode temporary
cast / widen temporary
pipeline buffer
IME fragment
value handoff
```

资源按 operation 所在位置和生命周期计算，资源超限的候选在生成 intrinsic C 前删除。Quant、dense、state 和 extension fragment 已经开始使用同一套资源计算。 

## 5. 最后才落到硬件 leaf

最终 leaf 仍然必须人工实现，因为编译器必须知道：

```text
RVV FMA 怎么拼
segment load 怎么拼
IME asm 怎么拼
某种 packed bitfield 的局部指令序列怎么拼
```

但这些 leaf 不应该再拥有：

```text
完整 K 循环
完整 output traversal
整个量化 projection
完整 workspace 和 ABI
```

报告声称 emitter 只消费已选的映射、shape、projection 和 leaf，不再自己决定 LMUL、decode chunk 或 fragment。

---

# 二、这是不是 Triton 类型的编译

## 是同一类问题，但不是同一成熟度

Triton 的高性能也不是来自“选择一个 GEMM 模板”。

它的 NVIDIA 后端显式执行：

```text
coalesce
remove layout conversions
optimize thread locality
accelerate matmul
optimize dot operands
assign latencies
schedule loops
software pipeline
prefetch
allocate shared / tensor memory
lower MMA
```

也就是说，Triton 用统一的 tile/layout 表示承接数据分布，再由大量 op-specific 和 target-specific pass 生成线程分工、访存、矩阵指令和流水。([GitHub][1])

Triton 的 Linear Layouts 更进一步：它不是继续枚举更多布局，而是用统一数学表示描述逻辑 tensor 元素到硬件资源的映射及 layout conversion。([arXiv][2])

Weft 当前的对应关系是：

```text
Triton
tile element
→ CTA / warp / lane / register / shared-memory layout

Weft
block axis + lifetime
→ sequential / lane / register repetition / fragment
```

但两者有一个实质差距：

### Triton 的 layout 描述更细

它能描述：

> tile 中的某个逻辑元素具体落到哪个 warp、lane、register。

Weft 当前的 mapping 更像**轴因子分解**：

```text
这一轴有多少进入 lane
多少成为 register repetition
多少顺序执行
```

它还不是一个完整的 per-element layout algebra，也没有 GPU 那样成熟的 layout conversion、shared-memory 分配和多级执行层次。

### Triton 的 pipeline 是成熟 pass 链的一部分

Weft 当前虽然能生成 Load/Compute action，但主要还是 primitive-local 的寄存器流水。它尚未拥有 Triton 那种成熟的 latency assignment、loop scheduling、异步 copy 和多级 storage pipeline。

因此最准确的评价是：

> **Weft 现在已经采用 Triton 类型的编译方法：从高层程序值和轴关系生成目标内布局与调度。它不是 Triton 类型的模板库。**
>
> **但 Weft 当前的物理表示与调度器比 Triton 粗很多。**

---

# 三、什么证据说明它真的在编译，而不只是选模板

## 1. F32 K1 prefill

旧 F32 程序使用逐列 rank-one dot，后来改成了显式二维 `W.matmul`。作者因此明确暴露 M/N/K 的局部 block product；同一个 F16/F32 matmul planner 再根据 dtype 选择 native FMA 或 widening FMA，共用 mapping、microtile、unroll、pipeline 和 emitter skeleton。

结果：

```text
K1 F32 prefill
5205.112 ms
→ 1563.137 ms
baseline 1606.776 ms
```

这里不是调用了 K1 F32 GEMM 模板。真正发生的是：

```text
DSL 暴露正确 structured op
→ 共同轴映射生成二维微内核
→ load/reuse/schedule改变
```



这也是一个重要事实：

> 性能不仅由编译器决定。DSL 如果只暴露 rank-one dot，编译器就没有二维输出复用的授权和信息。

## 2. Q3 vector partial

原实现每个 decode segment 后立刻 scalar reduction。新实现变成：

```text
decode / widen product
→ i32 vector partial accumulation
→ primitive末尾一次 reduction
```

K1 从 66.766 ms 降到 44.194 ms，最终略快于 baseline。

这是 state placement 和 reduction schedule 的变化，不是更换整个 Q3 kernel。

## 3. 42 个失败被一次 value-chain 修复关闭

第一次全量运行有 42 个失败，涉及 Q2、Q4_K、NVFP4 和多种 dequant。它们不是 42 个独立 kernel bug，而是 decode、store、reduce 和 indexed value 各自拥有第二份 physical shape 决定。

修复后它们共享同一个 block value-chain 和同一份 operation decision，42 个失败同时消失。

这也是编译器行为，不是给 42 个 kernel 加 42 条路径。

---

# 四、为什么编译出来仍然慢

最重要的结论是：

> **多数严重慢项不是“选错了一个好候选”，而是当前空间里根本没有 baseline 使用的那种高质量候选。**

---

## 1. F16：缺的不是 LMUL，而是微内核结构

最终结果：

```text
SG F16 decode    14.036 vs 7.387 ms
SG F16 prefill  702.321 vs 299.138 ms

K1 F16 decode    39.518 vs 9.785 ms
K1 F16 prefill 3755.370 vs 782.657 ms
```



当前已经有：

```text
row/column mapping
LMUL
K-unroll
single/double buffer
```

但 K1 prefill 只从 4267 ms 改善到 3755 ms。这个幅度说明：

> 排序不是主要问题，现有候选本身就不够好。

缺少的很可能是：

* 足够大的 cross-row / cross-output register tile；
* 更多独立 accumulator；
* A/B panel 的跨输出复用；
* 更好的 widening FMA 组织；
* F32→F16 staging 与 matmul 之间的数据流融合或摊销；
* 更深、真正能隐藏 load 延迟的流水。

其中还有一部分明确属于 DSL：真实 workload 是 F32 activation，作者需要显式 staging 到 F16 workspace。以前直接给 F16 input 的数字不真实；现在 staging 成本进入了计时。

所以 F16 慢由两层组成：

```text
DSL 算法确实多了一次 staging
+
编译器没有生成足够强的 F16 widening microkernel
```

---

## 2. F32：说明轴映射本身可以有效，但目标选择不稳定

K1 F32 prefill已经接近甚至略快于 baseline，但：

```text
SG F32 prefill 仍慢 34%
K1 F32 decode 仍慢 21%
```

同一套 mapping 在不同 phase/target 表现不同，说明：

* 合法性模型基本可用；
* reuse、microtile 和 cost model不够准确；
* decode 与 prefill需要不同的计算/访存平衡。

这不是再加一个 `K1F32PrefillLeaf` 能解决的。需要让候选空间明确包含不同 register tile、load window 和 pipeline，并由真实测量选点。

---

## 3. Quant：合法 mapping 不等于好的 decode 组织

Q4_0、Q5_0 的改进来自优先选择更少 register chunks、减少重复 decode 的 mapping：

```text
Q4_0 SG 81.730 → 44.659 ms
Q4_0 K1 119.486 → 59.084 ms
Q5_0 K1 120.114 → 59.947 ms
```

但 Q2 变成：

```text
SG 73.851 ms vs baseline 12.617
K1 164.273 ms vs baseline 48.013
```



这说明当前排序只理解了一部分代价：

```text
register chunks多少
decode重复多少
```

却没有完整理解：

* bitplane 解码指令数；
* scale 的复用；
* gather/slide 的真实代价；
* decoded values 的同时存活；
* activation 在多个输出间的复用；
* reduction 在 vector partial 与 scalar partial 之间的代价；
* 不同机器上 widening、gather、shift 的吞吐差异。

因此 Q2 不是“编译器没识别 Q2”，而是：

> **编译空间没有表达出 baseline 的 decode/reuse 结构，或者 cost 排序无法识别它。**

同一 Q5_0 实现在 SG 慢约 2.38 倍、K1却快约 7%，更直接说明当前 target profile主要描述**合法性**，没有描述足够的微架构成本。

---

## 4. IME：指令 leaf 基本可用，fragment 编排不行

IME decode：

```text
只慢 8%—17%
```

IME prefill：

```text
慢 40%—50%
```



这组数字很干净地定位了问题：

* 单次 IME fragment 指令与 asm 拼写并不差；
* 连续调用很多 fragment 时，输入 packing、fragment 复用和 pipeline 没有摊开。

缺的是：

```text
一个 packed input服务多少fragment
activation fragment能复用多久
多个fragment是否同时在途
load/pack与mma如何交错
结果fragment如何批量写回
```

所以 IME prefill 的问题不在 leaf，而在**fragment 级局部循环生成**。

---

## 5. FlashAttention：当前 scheduler 的作用域太小

最终：

```text
SG 32.267 vs 21.667 ms
K1 61.429 vs 44.422 ms
```



FlashAttention 包含：

```text
load K/V
dot
max/sum state更新
exp
accumulator rescale
weighted accumulation
normalize
```

当前 microkernel scheduler主要围绕一个 local primitive 和它的 operand window生成 Load/Compute action。

但 FlashAttention 的性能依赖的是：

> 同一个作者已写出的循环中，dot、summary state、exp、V load 和 weighted accumulator 之间的联合寄存器复用与流水。

这不是让编译器发现 FlashAttention 算法；算法已经在 DSL 中。
需要的是让 loop-local scheduler覆盖多个明确 op 组成的局部依赖链。

---

## 6. Transpose：报告还无法确定是 DSL 问题还是编译器问题

K1 contiguous transpose：

```text
20.522 ms vs baseline 4.251 ms
```

报告把它指向 coordinate reuse、load ordering 和 memory schedule。

但仅凭报告还不能确定：

* baseline是否用了 blocked transpose；
* Weft DSL是否只是直接 strided load/store；
* 两边是否具有相同 cache blocking。

如果 baseline 有显式二维 blocking，而 DSL 没有，那么这是 DSL kernel 写得不对，编译器不该自动发明。

如果算法完全相同，才是编译器 memory schedule 和寄存器组织的问题。

这一项必须直接并排看两份代码，报告本身没有给出足够证据。

---

# 五、Weft 达到高性能的真实条件

从已经成功和失败的项目看，高性能不是一句“轴映射”就能产生，必须同时满足五件事：

1. **DSL 暴露正确的局部结构**
   F32 rank-one dot 改成二维 `W.matmul` 后，编译器才获得二维复用信息。

2. **映射产生足够大的寄存器复用**
   不是只决定哪一轴进 lane，而要决定多少输出同时驻留、一个 load 服务多少 accumulator。

3. **局部 scheduler 保持中间结果不落地**
   Q3 的 vector partial 就是典型例子。

4. **资源模型能预测真实峰值与 spill 风险**
   当前还只是 operation-level snapshot，并非完整全程序 live interval；某些候选可能在 C 编译器处产生额外 spill 或 `vsetvl` 开销。报告明确说整体 live interval尚未完全合成。

5. **目标相关 cost 或实测选择**
   VLEN、寄存器数只能判合法，不能告诉编译器 gather、widen、shift、load、FMA在 SG 和 K1 上谁更贵。

---

# 六、后续性能工作应该按这个顺序

## 第一优先：补“空间里不存在的结构”

先处理：

```text
F16 register tile
IME fragment reuse
quant cross-output decode reuse
FlashAttention局部多op流水
```

这些不是调参能解决的。

## 第二优先：把 pipeline 扩到作者写下的局部循环依赖链

不是通用图编译，也不是只识别 GEMM。

输入是：

```text
一个显式 loop
+
其中明确的 load/decode/dot/state op
+
effect、alias和loop-carried dependence
```

输出是：

```text
prologue
steady state
epilogue
buffer rotation
```

## 第三优先：完整 target cost / compile-and-measure

静态模型负责：

* 合法性；
* 大致资源；
* 删除明显坏点。

真实目标机测量负责：

* LMUL；
* microtile；
* unroll；
* buffer count；
* fragment choice。

## 第四优先：对数量级慢项检查 DSL 与 baseline 是否同算法

重点检查：

```text
F16 staging
transpose blocking
FlashAttention workspace/reuse
quant persistent layout
```

如果算法结构不同，先改 DSL kernel；不能让编译器偷偷恢复 baseline 算法。

---

## 最终判断

当前 Weft 的编译空间已经可以清楚写成：

```text
op语义
→ 轴映射
→ 复用与局部schedule
→ 资源合法性
→ 局部硬件operation
→ intrinsic C
```

这确实属于 Triton 类型的编译，而不是整 kernel 模板选择。

它现在慢的核心原因也很明确：

> **统一坐标和生成流程已经出现，但很多 workload 需要的高复用微内核、跨 op 流水和目标成本还没有进入候选空间。**

所以后续不是继续“统一更多名字”，也不是先做一个更大的 autotuner，而是：

> **把 baseline 已经证明有效的复用、fragment 编排、decode 组织和流水，变成由轴、生命周期和显式 op 生成的机器结构。**

[1]: https://github.com/triton-lang/triton/blob/main/third_party/nvidia/backend/compiler.py "triton/third_party/nvidia/backend/compiler.py at main · triton-lang/triton · GitHub"
[2]: https://arxiv.org/abs/2505.23819?utm_source=chatgpt.com "Linear Layouts: Robust Code Generation of Efficient Tensor Computation Using $\mathbb{F}_2$"
