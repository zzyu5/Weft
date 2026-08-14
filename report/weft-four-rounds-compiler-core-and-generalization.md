# Weft 四轮编译器核心重构与泛化报告

## 结论

这四轮没有把 Weft 扩成图编译器，也没有继续积累整 kernel emitter。实际完成的是一条更接近
CPU kernel compiler 的唯一主链：

```text
Weft Python DSL
→ canonical worker-local Kernel IR
→ transient RISC-V physical entity plans
→ intrinsic C / primitive-local inline asm
→ system C compiler
→ object / executable
```

四轮分别解决了四个不同层次的问题：

1. 把原先分散在 VLA、block、dot、quant、IME 与 emitter 中的决定接成统一物理规划核心；
2. 收缩 DSL 与 Kernel IR，使 scalar control、VLA、logical block、state、dot/matmul 和 extension
   primitive 服从同一套授权规则；
3. 把统一抽象扩成受 value/use、memory、state 与寄存器资源共同约束的候选空间；
4. 用事后选择、接入前冻结的自然程序和 RVV Segment2 能力检验外推，没有增加 kernel route、精确整段模板或
   备用后端。

当前累计状态是 64 个 distinct kernel、134 条真机性能记录：SG2044/RVV128 66 条，
K1/RVV256 66 条，K1/IME1 2 条。这个数量是四轮结束后的累计结果，不是第一轮开始时的语料规模。

结构上，Weft 已经有了真实 kernel compiler 的骨架；性能上，它还不是 Triton 成熟度的系统。
局部 RVV/IME realization 已经能从共享事实中获得明显收益，但 matmul、VLEN256 quant、state、
vision、pipeline 与 build-time selection 仍有明确缺口。

## 范围与证据

本报告以四轮开始时已有的60个kernel为固定基础，不把更早从十/二十/三十/四十逐步扩到
六十个kernel的历史过程重新计算成本四轮工作：

| 轮次 | 主要提交 | 本轮问题 |
| --- | --- | --- |
| 第一轮 | `2a6b7c4ba`、`1d401bfbe`、`c3210b815`；性能闭合到 `ba86bb8ec` | 建立统一物理编译核心 |
| 第二轮 | `e2398e8ee` | 收敛 DSL、Kernel IR 与 primitive owner |
| 第三轮 | `30a8829c5` | 扩大共享、受资源约束的物理候选空间 |
| 第四轮 | `38b7c83c0`、`861177997` | 用接入前冻结的事后程序和 Segment2 检验泛化与扩展性 |

事实来源分为三类：

- `doc/`：当前语言与编译器边界；
- 当前源码与上述提交：真实实现；
- `examples/run/weft.sh` 的目标机执行与 `report/weft-kernel-performance.csv`：正确性与性能结果。

SG2044 与 K1 的 core、compiler 和 memory system 都不同，因此跨设备数字只用于观察
target-specific realization，不能当作 VLEN128/VLEN256 的单因素因果实验。

前三轮用于前后比较的126条同key记录固定为62条SG RVV、62条K1 RVV和2条K1 IME；文中
“K1 64项”即62条标准RVV加2条IME。第四轮新增4个kernel的双硬件记录后，当前总表才增长为
134条。每轮几何平均都只比较该轮前后均存在的同kernel、phase、hardware、target、shape和scope。

## 四轮共同冻结的编译器边界

### 作者、Realizer 与系统编译器的所有权

| 所有者 | 拥有的内容 |
| --- | --- |
| DSL 作者 / 上游 frontend | worker-local ABI、scalar/ordered traversal、VLA logical axis、blocking、staging、persistent layout、pointer/index/predicate/effect、state algebra、算法 variant、显式 local primitive |
| Weft RISC-V Realizer | dynamic `vl`、LMUL、value/register shape、memory form、handoff、state placement、register microtile、K-unroll、primitive-local packing、RVV/IME realization、局部 pipeline/prefetch 决定 |
| system C compiler | 最终寄存器分配、机器调度、peephole、常量折叠与机器码生成 |

普通 scalar loop 不能被 target 自动改成 VLA；普通 multiply/add 不能被猜成 dot/matmul；
blocking、staging 与 persistent packing 不能被 target 暗中创造。只有显式 `W.vla`、`W.dot`、
`W.matmul`、state 或 extension primitive 授权对应局部物理重组。

### 唯一持久 authority

长期存在的编译器表示只有 canonical Kernel IR。它保存 worker-local 算法、typed value、axis、
validity、pointer/effect、state 与显式 primitive，不保存 exact `vl`、LMUL、microtile、fragment、
instruction spelling 或 build measurement。

一次 target lowering 内可以建立 capability、legality、candidate、resource 和 schedule 信息，但
它们只能是短生命周期的 physical entity plan，不能形成第二份 IR、provider registry 或独立
pipeline front door。Emitter 只读取已经产生的决定；缺失决定时明确失败，不能重选，也不能
fallback 到 GGML、materials 或旧 emitter。

## 第一轮：建立统一物理编译核心

### 第一轮之前的问题

第一轮不增加kernel，以既有60个kernel、每个标准RVV目标62条phase作为固定压力语料。
VLA、block、dot/matmul、quant 和 IME 已经各自拥有一些高性能 lowering，但 value shape、LMUL、
memory form、state、block strip 与 fragment 仍由不同局部结构保存。部分 emitter 还会根据周围
use 或 block closure 再推一次 vector shape。结果是同一个 value 在 load、cast、state、dot 与
store 之间缺少统一物理身份，资源预算也无法闭合。

### 实际重构

第一轮建立了统一的 `PhysicalEntityPlan`。每个显式 VLA region 或 local primitive 都以相同结构
保存：

```text
value physical shapes
memory/value handoffs
temporary shapes
primitive realization
loop-local schedule
resource budget
```

Handoff 被明确为 `share`、`convert`、`rematerialize`、`reload`、`shuffle` 或
`primitive-local pack`。资源预算统一计算 live value、memory/index、predicate、state、primitive、
pipeline 与 peak register groups；没有显式 pipeline 或 prefetch 授权时，计划明确记录
single-stage 与 zero-prefetch，而不是由 emitter 猜测。

`preparePhysicalDecisions()` 在输出 C 之前统一产生 VLA、dot/matmul、block store/reduce、decode、
quant 与 extension decision。`PlannedPhysicalDecision<T>` 将 primitive-specific realization 与同一
entity plan 绑定；同一 canonical owner 不允许产生两份决定。

随后又删除了 primitive payload 中重复保存的 block/decode/store/F16 shape 字段。Emitter 改为
只查询 entity-owned `PhysicalValueDecision`、handoff 与 temporary。Dot 的 owner 固定为
`weft_kernel.dot`，下游 store 只表示 output handoff，不再拥有第二份 dot decision。

### materials 的知识所有权

第一轮同时明确了历史材料的利用边界：

| materials 中的知识 | 归属 |
| --- | --- |
| outer loop、blocking、staging、persistent organization、跨阶段 reuse | DSL 算法知识，必须由作者写入 kernel |
| SEW/LMUL、register footprint、decode packing、widen/reduce 组织、local scheduling | Realizer 的 physical candidate/legality/resource 知识 |
| intrinsic/asm spelling、operand constraint、clobber | primitive-local leaf |

`materials/` 没有进入 CMake、include、link、runtime 或 production fallback。

### 真机闭合与性能事实

第一轮结束时，固定语料在唯一主链上完成：

```text
SG2044 / RVV VLEN128 : 62 / 62 phase
K1/X60 / RVV VLEN256: 62 / 62 phase
K1/X60 / IME1       : 2 / 2 local primitive
```

这证明了统一 plan 能生成可执行代码，但没有自动带来整体性能提升。以第一轮开始前与
`ba86bb8ec` 的同 key 记录比较，126 项 median ratio 的几何平均为 `1.035×`；其中 SG 的
62 项为 `1.074×`，K1 的 64 项为 `0.999×`。也就是说，本轮整体回退 `3.54%`，主要来自 SG
回退 `7.42%`，K1 基本不变。代表性 SG 回退包括：

- F32 GEMM decode：`14.816357 → 35.077880 ms`；
- F32 GEMM prefill：`756.964108 → 1558.790406 ms`；
- Weighted EmbeddingBag：`64.886151 → 139.693869 ms`；
- Out Product backward：`127.211509 → 263.495145 ms`。

同时，IQ1_M×Q8_K 从 `77.098845` 降到 `39.266377 ms`。这些相反方向的变化与物理实现选择
发生改变一致，也说明默认 candidate ordering 仍不成熟。

因此第一轮完成的是 authority 与资源骨架，而不是“统一以后自然更快”。

## 第二轮：收敛核心语言与 IR 抽象

### 冻结已经收缩的 structured compute 边界

进入第二轮时，一般化`W.contract(lhs_axes, rhs_axes, output_order)`与generic summary fold已经在
前序重构中删除。本轮没有恢复或重新实现它们，而是清理编译器中残留的`contract`命名与owner
含义，并把最终核心边界固定为真实kernel所需的两个structured product授权点：

```text
W.dot    : 固定收缩双方最后一个 logical block axis
W.matmul : [M,K] × [K,N] → [M,N]
```

VLA axis 只能作为 free/batch axis，不能被 dot/matmul 隐式缩并。作者仍拥有 operand block、
outer traversal、blocking、staging、pointer/index、accumulator 与 store。

Backend config与CLI中的`contractLMUL/contractKUnroll`相应收敛为`dotLMUL/dotKUnroll`；F16
候选明确归属matmul。前序删除generic summary fold后保留的显式`argmax`与
`online_softmax_summary`也在这一轮被纳入同一canonical value/effect模型，target不再从任意
helper SSA closure猜summary algebra。

### 一个 kernel model，而不是多套执行路径

第二轮统一了 scalar、block、region 与 masked validity 的 canonical value system。VLA、state、
dot/matmul、quant 和 IME 不再各讲一套 kernel 模型：

```text
ordered control regions
+ zero or one active VLA axis per lexical scope
+ scalar / block / region SSA values
+ explicit extent, validity, pointer relation and effects
+ explicit local semantic primitives
```

Sibling extension dialect 必须复用核心 logical value、extent、validity 与 use-def/effect 查询；它只
定义新的局部数值语义，不能重新定义 execution model。量化 extension 名称中的 `contract` 仍是
固定 typed block 的局部数值 primitive，不是已经删除的一般化 contraction。

### primitive-local owner

F16 matmul 不再以外围 N loop 为 owner；affine IME 不再扫描完整 N/K loop body寻找唯一 store；
online-softmax summary 也不再接管 normalize consumer。每个 canonical primitive 根据自身 typed
operands、axis、memory、state 与 target facts 产生 decision，外围 traversal 原样保留。

这一轮得到的是一个清楚的定义：作者写的是完整 worker-local kernel；`W.vla`、`W.dot`、
`W.matmul`、state 与 local extension primitive 分别授予局部重组权；Realizer 输入四类事实——
domain、value、effect、primitive——输出同一种 transient entity plan。

相同 126 条记录与第一轮结束快照相比，第二轮 median ratio 几何平均为 `0.963×`：SG 的
62 项为 `0.927×`，K1 的 64 项为 `0.999×`。这对应整体改善 `3.71%`，其中 SG 改善
`7.30%`。同步观察到F32 GEMM、EmbeddingBag、Out Product与Causal Mask恢复，而Dense Conv
同时出现回退，因此仍不能只以平均数判断成熟度，也不能只由时间变化反推单一根因。

## 第三轮：把统一抽象扩成受资源约束的候选空间

### 从固定值变成受资源约束的 candidate

第三轮没有增加 kernel，重点是让已经统一的实体拥有真实候选空间。

Local F32 dot 的 candidate facts 包含 model、row tile、reduction extent、unit/strided/indexed
operands、predicate/state/handoff groups、materialized init 与 reduction predicate。Selector 枚举：

```text
LMUL ∈ {1, 2, 4, 8}
K-unroll ∈ {1, 2, 4}
```

候选按 accumulator、streamed operand、predicate、state、handoff 与 32 个 architectural vector
registers 的 headroom 过滤，再考虑 tail、memory、LMUL、unroll 与 peak-resource penalty。

VLA candidate 同样联合计算 element shape、indexed address vector、predicate mask、state placement、
narrow intermediate、local dot hard constraint 与 register budget。Load、
cast、state、dot、store 不再分别挑 LMUL；冲突时该候选整体非法。

### Quant 与 IME 仍然是局部 realization

Quant lowering 由 typed local block、storage type、extent、VLEN 与 resource facts产生 semantic lanes、
byte shape、decode/reduction segment 与 leaf family。IME 只实现明确的 local N16×K32 fragment；
activation quantization、outer N/K traversal、persistent packed layout、accumulator carry 和 ABI 仍属于
source。

当前性能表中，同一 K1 target 上的两项局部 RVV/IME realization 为：

| Local primitive | K1 RVV256 | K1 IME1 | IME / RVV throughput |
| --- | ---: | ---: | ---: |
| affine q4_K×q8 | 42.912404 ms / 0.781929 GOP/s | 3.958822 ms / 8.475863 GOP/s | 10.84× |
| symmetric q4_0×q8 | 44.643618 ms / 0.751606 GOP/s | 3.176632 ms / 10.562896 GOP/s | 14.05× |

两项 activation code mismatch 都为 0。这证明上述两个明确的q4/q8局部primitive可以由target
facts选择RVV或IME realization；它不证明其他量化关系或任意matmul已经能自动映射到IME。

### 第三轮没有完成的部分

当前合法 K-unroll winner 仍为 1；multi-axis microtile、multiple accumulators、pointer/load schedule、
prefetch、local reuse 与 software pipeline 的空间仍窄。CLI 已能显式约束 LMUL、dot K-unroll、
F16 input LMUL 与 row microtile，但仓库尚没有覆盖完整候选空间的成熟 compile-and-measure tuner。

所以第三轮完成的是“可以表达并过滤多个合法物理实现”，不是“已经总能选择最快实现”。

与第二轮结束快照相比，第三轮相同 126 条记录的 median ratio 几何平均为 `0.951×`，整体改善
`4.86%`。K1 的 64 项改善 `11.09%`，SG 的 62 项反而回退 `2.03%`。共享候选让 SG
Q4_K×Q8_K 从 `12.194864` 降到 `3.702316 ms`，但 IQ1_M×Q8_K 从 `37.850765` 回退到
`248.047282 ms`。这正是“候选空间已经存在、跨 target winner 选择仍未成熟”的直接证据。

## 第四轮：接入前冻结的事后语料与 RISC-V 扩展性

### 接入前固定的后验语料

第四轮在接入compiler之前固定了四个此前未围绕当前lowering设计的自然程序；名单固定后才开始
补语言事实与target realization：

| Kernel | 自然结构 | 用来压力什么 |
| --- | --- | --- |
| `interleaved_complex_mul_f32` | 两路交错复数输入与交错输出 | 同一局部 interleaved memory relation 被多次使用；load/store 顺序等价写法 |
| `interleaved_rope_f32` | scalar token/head traversal 内的 VLA pair rotation | Segment2 出现在陌生 outer context 与多个输入中 |
| `dilated_causal_conv1d_f32` | VLA token + scalar tap + causal predicate + strided memory | predicate、state-like carry 与 masked strided access 组合 |
| `codebook_lookup_affine_f32` | explicit local table lookup + affine compute | lookup、narrow/widen、gather 与普通 pointwise/memory 组合 |

Complex 的等价 source 改变了常数乘法、load/store 顺序和临时 SSA；lookup 的等价 source 改变了
地址临时量、显式 `where=True` 与算术组织。Complex 两种写法都由同一 Segment2 memory decision
family处理；lookup两种写法都由显式`LookupOp`的同一table-gather realization处理。这里没有
normalization pass，也没有按symbol识别。

### 新增的共享 lowering 能力

#### RVV Segment2 memory realization

Segment2 不是新的算法 primitive。它是显式 VLA、typed f32 pointer 与局部 interleaved address
relation授权下的一种 memory realization。当前 legality 要求：

- 两个 access 位于同一 block；
- 同一 root/base，field 为 0/1；
- 同为 load 或同为 store；
- f32、all-active、原始关系为 stride-2；
- LMUL 不超过 4。

Load 在首字段发出一次 `vlseg2e32` 并产生两个 value；store 在末字段发出一次 `vsseg2e32`，让
普通 SSA producer 按原 source 顺序生成。实现没有 deferred producer replay，也没有把两个完整
kernel 交给专用 emitter。

#### Masked strided memory

VLA masked load现在消费已经选定的 predicate mask、strided/indexed memory mode与scalar `other`，
发出 RVV `_tumu` intrinsic。Dilated causal conv 的左 padding由 DSL source显式写出；target没有
暗中创造 padding、改变 traversal 或替换 causal algorithm。这样 mask-off lane 即使不访存，也
不会先在 C pointer arithmetic中构造下溢地址。

#### Explicit lookup realization

Lookup 只有在作者显式写出 `W.lookup` 后才会进入该路径。当前 RVV realization接受 all-active
`f32 table<16>` 与 VLA `u8` indices，在 VLEN至少128时形成 table reload、u8→u16→u32 widening
与 `vrgather`。Lookup result extent由indices传播；value shape、handoff与temporary仍由统一entity
plan拥有。普通SSA图不会被猜成lookup。

### 双目标真实结果

四个 kernel均在SG2044/VLEN128与K1/VLEN256上完成真实数值执行。当前性能CSV记录：

| Kernel | SG2044/VLEN128 | K1/X60/VLEN256 | Correctness |
| --- | ---: | ---: | --- |
| Interleaved complex | 3.158514 ms / 7.967615 GB/s | 9.152613 ms / 2.749578 GB/s | primary/equivalent max abs = 0 |
| Interleaved RoPE | 0.709423 ms / 8.868413 GB/s | 1.116770 ms / 5.633618 GB/s | max abs = 5.96046448e-08 |
| Dilated causal conv | 32.822124 ms / 0.223631 GFLOP/s | 86.627341 ms / 0.084731 GFLOP/s | max abs = 0 |
| Codebook lookup affine | 1.799868 ms / 582.584945 MElements/s | 4.611405 ms / 227.387532 MElements/s | primary/equivalent max abs = 0 |

Complex与RoPE的单-kernel runtime还包含同ABI、同shape、同数据布局、相同64 MiB eviction与7次
median协议的普通RVV strided intrinsic baseline。它只存在于repro runtime，不进入compiler、
`source/`或generated artifact：

| Kernel | Target | Segment2 Weft | Strided RVV baseline | Segment2 speedup |
| --- | --- | ---: | ---: | ---: |
| Interleaved complex | SG2044 | 3.158514 ms | 14.645564 ms | 4.637× |
| Interleaved complex | K1/X60 | 9.152613 ms | 13.844980 ms | 1.513× |
| Interleaved RoPE | SG2044 | 0.709423 ms | 2.524751 ms | 3.559× |
| Interleaved RoPE | K1/X60 | 1.116770 ms | 2.612199 ms | 2.339× |

四个strided baseline median来自本轮目标机runtime标准输出；仓库没有另存raw stdout或日志，
因此它们由本报告做一次性快照，而不属于`weft-kernel-performance.csv`。CSV只保存表中四项
Weft生成实现的median与throughput。

GGML没有这两个交错布局的同算法实现，因此没有伪造GGML分母。这里证明的是：一个局部RVV
memory capability在Complex与RoPE这两个程序中复用了同一decision family，并且本次在两台目标
机器上都优于同目标普通strided realization；这两例不等于对任意交错程序的普遍性能结论。

## 四轮结束后的累计状态

### 当前性能记录构成

| 项目 | 数量 |
| --- | ---: |
| 性能记录 | 134 |
| Distinct kernel | 64 |
| SG2044 / RVV VLEN128 | 66 |
| K1/X60 / RVV VLEN256 | 66 |
| K1/X60 / IME1 VLEN256 | 2 |
| Forward phase | 96 |
| Decode phase | 24 |
| Prefill phase | 4 |
| Backward phase | 6 |
| Forward-backward phase | 2 |
| Update phase | 2 |

不同单位、shape、scope、preprocessing归属与correctness scope不能横向混算。当前表中既有full也有
sampled correctness，也有source-owned preprocessing计入计时的条目；134条记录不能概括成一套
完全相同的测量范围。

### 已经成立

- Weft 是单 worker/hart 的 kernel DSL与AOT compiler，不是graph compiler；
- 唯一主链只消费canonical Kernel IR，不解析或链接IntentDSL；
- scalar、VLA、block、state、dot/matmul、quant与extension服从同一value/effect/primitive模型；
- general contract与generic summary fold已删除，没有双语义或compatibility path；
- 主要physical decision在emission之前由canonical owner唯一产生；
- 相同DSL可以在RVV128、RVV256与IME profile上生成并真实执行；
- 接入前冻结的事后kernel没有导致kernel-name、q-format、whole-region或legacy route重新出现；
- `source/c/ggml/`只作GGML baseline，`materials/`只作知识供体，二者均不进入production fallback。

### 仍不能声称

- 任意dot/matmul source envelope都能获得高性能lowering；
- candidate ordering已经等同成熟build-time autotuning；
- multiple accumulators、load scheduling、prefetch、reuse和pipeline空间已经完整；
- VLEN256 quant、prefill GEMM、state与vision contraction已经具有竞争力；
- IME支持已经超出两个明确的N16×K32 local primitive；
- lookup已经支持任意dtype、任意table extent或masked lookup；
- Segment2已经支持masked、非f32、多字段或跨block组合；
- 全部134条记录都采用full correctness或统一不含preprocessing的计时范围。

## 最终判断

四轮之后，Weft 不再只是“若干 kernel 能走通”的 emitter 集合。当前实现已经形成三个可由
源码与真机结果直接定位的编译器性质：

1. 一个清楚且唯一的 worker-local语言/IR模型；
2. 一个由逐实体typed facts联合产生、受资源约束且由emitter机械消费的physical planning core；
3. 一个能让同一局部语义跨kernel、跨书写形式和跨target复用realization的扩展机制。

第四轮尤其排除了最危险的成功方式：新增程序没有促使Weft增加整kernel分类；新增RVV能力只扩张
memory physical space；自然等价写法没有依赖normalization；GGML/materials没有进入runtime路径。

但性能事实同样说明，Weft当前应被称为“已经形成真实编译器核心、仍在扩充高性能搜索空间”，
而不是成熟的RISC-V Triton。当前主要缺口是更宽且可实测选择的共享physical candidates，
而不是更多kernel recognizer。
