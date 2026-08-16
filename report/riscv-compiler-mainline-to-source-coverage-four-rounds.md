# Weft RISC-V 编译主干到 source baseline 全覆盖：四轮推进报告

## 结论

这四轮不是此前报告中已经出现过的其他“四轮”。本文严格按下面四个 prompt 对应的连续
实现节点划分：

| 轮次 | prompt | 实现边界 | 实际作用 |
|---|---|---|---|
| 第一轮 | 重构 Weft 的编译主干 | `f45a9f694` | 收敛同一物理事实的唯一 owner，清除 lowering 内部的重复决定 |
| 第二轮 | 完成高性能编译核心与 intrinsic C 发射解耦 | `5cc80ee8f` | 将 facts、physical planning、kernel compiler 与 RVV/IME/quant 拼写拆成明确模块 |
| 第三轮 | 继续扩充高性能能力并收敛编译器方法 | `7687bf0b7` | 增加 VLEN256 grouped quant leaf、显式 semantic-lane leaf，并继续把 target 判断移出 local leaf emitter |
| 第四轮 | 覆盖 `source/` baseline 并完成编译器收敛 | `aebcf4392`—`fcb535dff` | 用约 11 小时为 123 个 GGML 逻辑 case 形成 250 条 Weft target realization/执行记录 |

第四轮从 2026-08-16 05:40:09 到 16:32:41，提交时间跨度为 **10 小时 52 分 32 秒**，
共 39 个提交。前三轮建立和整理编译器骨架，第四轮才把这个骨架压到完整的 GGML RISC-V
baseline 上。

四轮后的真实结论是：

> **Weft 已经形成唯一的 Kernel IR → typed facts → physical selection → intrinsic C / local asm
> 主链，并能覆盖当前固定的 GGML RISC-V baseline；但是 performance space 仍不成熟，尤其是
> K1 dense/quantized matrix multiplication、F16/F32 prefill 与部分 packed decode。**

因此，“source 全覆盖”证明的是编译路径、语义覆盖与 target realization 的广度，不等于
250 个 kernel，也不等于所有实现已经追平 GGML。

## 报告边界与数据来源

本文是一次性工作快照，不是规范。当前规范仍由 [`doc/index.md`](../doc/index.md) 及其链接的
模块定义。性能数据来自：

- 当前 Weft 结果：[`weft-kernel-performance.csv`](weft-kernel-performance.csv)；
- 固定 GGML 结果：[`baseline/ggml-riscv-kernel-performance.csv`](baseline/ggml-riscv-kernel-performance.csv)；
- GGML 测量说明：[`baseline/ggml-riscv-kernel-baseline.md`](baseline/ggml-riscv-kernel-baseline.md)。

本文所说“比 GGML 快/慢”的比率统一为：

```text
GGML median_ms / Weft median_ms
```

大于 1 表示 Weft 更快，小于 1 表示 Weft 更慢。Weft source coverage 与 GGML 使用相同硬件、
算法和 shape，但 warmup、repetition、correctness scope 与部分 preprocess 计时范围并未完全统一，
所以这些比率用于判断方向和数量级，不作为接近 1.0 时的最终胜负结论。

## 四轮之前的问题

进入本轮前，Weft 已经有 Core-local blocked DSL、canonical Kernel IR，以及 axis、use、memory、
lifetime 和 target resource facts。真正的问题不是“没有 Plan”，而是这些信息进入物理 lowering
之后仍存在以下裂缝：

1. operation decision、entity plan 和 emitter 各自保存一部分 LMUL、shape、schedule 或 resource；
2. 同一个 dot/matmul 的 physical config 会被选择、复制、汇总，然后在 emission 附近再次解释；
3. lookup、decode、block store/reduce 等路径缓存自己的 vector shape，和 value/handoff authority 重复；
4. `RISCVLowering.cpp` 同时承担 facts、选择、资源、Kernel IR traversal、C 控制流和 intrinsic/asm 拼写；
5. quant、IME 和 RVV leaf 虽然已经是局部 primitive realization，但 target/VLEN 判断仍有一部分留在
   compiler/emitter 交界处；
6. 完整 `source/` baseline 尚未全部有对应的自然 DSL kernel、runtime 与真实 target 记录。

四轮推进的顺序因此是：先关闭重复 authority，再拆模块，再扩共享 realization，最后用完整
baseline 检验这条主链是否真的能承受不同格式、不同 target 和真实模型 shape。

---

## 第一轮：重构 Weft 的编译主干

### 范围

提交：`f45a9f694`，相对上一节点共修改 2 个文件，`452 insertions / 493 deletions`。代码变化集中
在当时仍然单体的 `lib/Target/RISCVLowering.cpp`；另一次性刷新了性能 CSV。

这一轮没有新增 DSL、Kernel IR op 或 kernel。它处理的是 lowering 内部“同一事实由谁决定”的问题。

### 1. Dot/matmul 的 physical config 回到 operation decision

旧结构中，VLA dot 同时存在：

```text
VLADotDecision
VLADotCandidate::physical
VLA region dotPhysical map
entity-level LoopLocalScheduleDecision
```

`kUnroll`、LMUL、pipeline 等信息会在这些结构间复制。第一轮删除 `VLADotCandidate` 和 region
的第二份 `dotPhysical` authority，改为：

```text
VLADotDecision::physical
```

作为该 operation 唯一的 selected config。F16 matmul 采用相同原则：selector 返回
`SelectedF16MatmulPhysical { config, resources }`，最终 physical config 只写入
`F16GemmNTileDecision::physical`。

Emitter 不再从 entity schedule 重新取得 `kUnroll` 或 microtile，而是读取 operation 已经选好的
decision。

### 2. Resource 与 realization 一起产生

新增或收敛为以下 selected result：

```text
SelectedBlockStorePhysical
SelectedBlockReducePhysical
SelectedF16MatmulPhysical
```

旧 block store/reduce selector 只返回 realization，prepare 阶段再计算一次 resource 和 schedule；
新结构让 legality、realization 与 resource budget 在同一个选择点一起产生。实体层只接收最终
resource 占用，不再保存另一份 schedule。

### 3. Shape authority 统一为 typed physical value/handoff/temporary

VLA candidate 从裸的：

```text
dataSEW / dataLMUL / indexSEW / indexLMUL
```

改为 typed `RVVVectorShape dataShape/indexShape`，cast、predicate、state、lookup 等相关 shape
统一通过同-lane shape 关系构造。

`VLALookupDecision` 和若干 block decode decision 删除自己缓存的 code/index/table/result shape；
生成阶段改为读取 entity 已经拥有的：

```text
PhysicalValueDecision
PhysicalHandoffDecision
PhysicalTemporaryDecision
```

这使一个 value 从 load、cast、lookup、dot 到 store 只拥有一份机器形态，而不是每个 primitive
分别记一份“我认为它是什么向量”。

### 4. Memory fact 保存语义关系，不提前物化 emitter 地址

`LocalBlockMemoryFact` 从预先保存一个 `base` 地址，改成保存：

```text
semantic value
pointer
storage axis
pointer/axis relation
```

实际 block base 由 `projectLocalBlockMemoryBase` 按当前 storage axis 投影。这样 address expression
仍来自 Kernel IR pointer relation，而不是分析阶段制造一个 emitter 私有地址。

segment2 memory 的判断也从“两个字段必须具有完全相同的 pointer SSA value”放宽为比较去掉
coordinate/field 后的 invariant address。等价 base 写法不再因为 SSA 拼写不同失去 segment
realization。

### 5. 这一轮真正清掉的重复 authority

- `VLADotCandidate` 与 region `dotPhysical` 的第二份 dot config；
- entity-level `LoopLocalScheduleDecision` 对 unroll/microtile/pipeline 的重复保存；
- lookup/decode decision 内重复的 vector shape 集合；
- VLA candidate 中裸 SEW/LMUL 的手工推导；
- `LocalBlockMemoryFact.base`、`activationBlockBase` 等预物化地址；
- segment2 对 pointer SSA identity 的过窄要求。

### 6. 本轮没有完成什么

这一轮仍把绝大多数逻辑放在一个 `RISCVLowering.cpp` 中。它收敛了 authority，却尚未把事实分析、
机器选择和 intrinsic 拼写拆成文件级职责。

提交修改了性能 CSV，但没有增加 repro 入口或保留独立运行日志。因此可以确认结果表发生了刷新，
不能仅凭该提交声称全仓 runtime 在此节点完成了一次可追溯重跑。第一轮的主要产物是编译主干的
所有权修正，不是性能闭合。

---

## 第二轮：完成高性能编译核心与 intrinsic C 发射解耦

### 范围

提交：`5cc80ee8f`，20 个文件，`13,951 insertions / 13,216 deletions`。

原 `RISCVLowering.cpp` 在父提交中为 13,221 行，本提交后缩到 28 行；新建的
`RISCVKernelCompiler.cpp` 为 11,070 行。也就是说，这一轮既有真实职责拆分，也有大量内容迁移；
不能把全部行数都解释成新增能力。

### 1. 新的模块边界

| 模块 | 本轮后的职责 |
|---|---|
| `RISCVLowering.cpp` | target gate 与顶层 facade；调用唯一 compiler，再生成所需 prelude |
| `RISCVKernelFacts.{h,cpp}` | logical axis、lane relation、value use、memory access 与 Kernel physical facts |
| `RISCVPhysicalPlanning.{h,cpp}` | RVV shape/LMUL、resource budget 与 typed candidate selector |
| `RISCVKernelCompiler.{h,cpp}` | Kernel IR traversal、decision materialization、C control/address emission 协调 |
| `RISCVIntrinsicC.{h,cpp}` | exact `IntrinsicCLeaf`、leaf 名称与 selected leaf 集合 |
| `RISCVIntrinsicCPrelude.cpp` | 公共 C/RVV helper、数学函数与只按 selected leaves 拼接的 prelude |
| `RISCVRVVIntrinsicC.cpp` | RVV local fragment 拼写 |
| `RISCVIMEIntrinsicC.cpp` | IME1 typed local inline asm |
| `RISCVQuant*IntrinsicC.cpp` | IQ1/IQ2/IQ3/Q6 等 quant local leaf 拼写 |

顶层主链变成：

```text
canonical Kernel IR
→ analyzeKernelPhysicalFacts
→ typed candidate selectors / resource legality
→ per-operation selected decision + exact IntrinsicCLeaf
→ KernelCompiler 生成普通 C 控制与调用点
→ local leaf modules 对已选 leaf 机械拼写 RVV intrinsic / IME asm
```

### 2. 不只是移动文件的部分

`RISCVKernelFacts` 提供 typed `LogicalAxisFact`、`ValueUseFact`、`MemoryAccessFact` 和
`KernelPhysicalFacts`；`RISCVPhysicalPlanning` 提供 typed selector，例如：

```text
selectI4I8FragmentRealization
selectReductionStatePlacement
selectQuantI8DotPhysical
selectE2M1E8M0I8Physical
selectF32DotPhysicalConfig
selectF16MatmulPhysicalConfig
```

以 symmetric i4×i8 为例，compiler 的 decide 阶段先选择 realization 与 exact leaf；emit 阶段只
读取 plan、handoff 和 `intrinsicCLeafName`，不再按 format 名或 target 字符串决定调用哪个 helper。

被拆出的 leaf 均只实现局部数值 primitive，包括：

- RVV symmetric/affine i4×i8 N16K32；
- IME1 symmetric/affine i4×i8 N16K32；
- grouped affine i4×i8；
- E2M1/E8M0×i8；
- IQ2_S、IQ3_S、IQ1_M、Q6_K fixed/scalable realization；
- 公共 RVV math/prelude helper。

它们不拥有 entry ABI、outer traversal、activation quantization、workspace、persistent layout
或完整 matrix multiplication。

### 3. 性能记录说明了什么

该提交刷新了 11 行性能数据。代表性变化包括：

| target / case | 前值 | 后值 | 判断 |
|---|---:|---:|---|
| SG dense prefill | 6.140 | 5.573 GOP/s | 回退 |
| SG IQ1_M local dot | 0.477 | 3.010 GOP/s | 明显改善 |
| K1 dense prefill | 1.160 | 1.815 GOP/s | 改善 |
| K1 MXFP4 local dot | 0.649 | 2.147 GOP/s | 明显改善 |
| K1 IQ3_S local dot | 1.746 | 0.122 GOP/s | 严重回退 |
| K1 IME local fragment | 8.412 | 8.453 GOP/s | 基本持平 |

这些数字说明文件拆分没有自动带来统一性能收益，也暴露了 K1 IQ3 这种 target realization
仍不成熟的路径。与第一轮相同，本提交未保留独立命令日志，表内小幅变化不能全部归因到模块
拆分本身。

### 4. 本轮没有完成什么

“planning 与 emission 解耦”在文件职责上已经成立，但还不是绝对纯化：

- `RISCVKernelCompiler.cpp` 仍约 12.5K 行，许多 `decide*` 与 `emit*` 仍在同一文件；
- 独立 `RISCVPhysicalPlanning` 只覆盖共享 shape/resource/leaf selector，VLA、block 与部分 handoff
  planning 仍由 KernelCompiler 协调；
- selected leaf set 仍是 compiler 与 prelude/emitter 之间的共享接口；
- 部分 header 依赖仍然较紧。

因此第二轮完成的是“从一个大 lowering 文件拆出真实职责和 typed API”，不是宣称 emitter 已经
完全没有复杂控制逻辑。

---

## 第三轮：继续扩充高性能能力并收敛编译器方法

### 范围

提交：`7687bf0b7`，19 个文件，`385 insertions / 142 deletions`。本轮没有修改 examples 或
`materials/`，只修改 compiler、intrinsic leaf、文档和性能 CSV。它检验的是：同一 DSL 与
Kernel IR 是否能仅通过扩展 shared physical realization，在不同 target 上获得新实现。

### 1. Grouped affine i4×i8 获得 VLEN256 realization

`selectGroupedAffineI4I8Physical` 不再只在 emitter 附近用 `vlenBits > 128` 选 helper，而是形成
三种 explicit realization：

```text
VLEN128 → RVVVLEN128GroupedDot
VLEN256 → RVVVLEN256GroupedDot
其他合法 VLEN > 128 → RVVScalableLocalBlockDot
```

每个 candidate 同时产生 packed、scale、activation、activation-sum 的 vector shape 和 resource
budget，并检查 peak register groups。

新增的 VLEN256 leaf 用 32-lane e8m1 完成 nibble decode、signed widening multiply、i32 reduction，
并只处理当前八组 scale/minimum metadata。它仍然是 `grouped_affine_i4_i8_dot` 的局部实现，
不拥有 Q4_K row traversal 或 projection。

### 2. Fixed quant leaf 从隐式 VLEN 分支变成显式 semantic-lane 决定

IQ2_S 和 Q6_K 增加：

```text
FixedLanes32
FixedLanes64
```

compiler 根据 selected decision 的 `semanticLanes` 选择 exact leaf；不存在对应 lane realization 时
直接 unsupported。local leaf emitter 不再接收 `vlenBits` 再决定 32-lane 还是 64-lane helper；
block/use relation 的协调仍在 KernelCompiler 中完成。

同样地，`emitIntrinsicCPrelude` 与 quant leaf emission 删除了无意义的 VLEN 参数传递。target
事实只在 planning/selection 阶段影响 realization；这里的 emission 特指 local leaf 的
intrinsic/asm 拼写，不表示 KernelCompiler 已经没有控制与 handoff 协调。

### 3. Dense/state resource accounting 继续收紧

- F32 carried state 与 transient state 分开计数；多个 vector carry 按实际同时存活累加，而不是
  只取最大值；
- F16 pipeline temporary 变为 selector-local resource，不再作为持久 Plan 字段跨阶段保存；
- F16 matmul 继续枚举 row microtile、input LMUL、K-unroll 与一/二阶段 schedule，并在 selector
  内完成 peak resource 检查；
- 删除不再消费的 `BlockStoreGroupDecision::interstitial` 与重复重新发射路径。

### 4. 性能变化与证据边界

SG2044 Q4_K×Q8_K decode 从 `9.590` 到 `9.860 GOP/s`，约提升 2.8%。K1 对应记录从
`0.911` 到 `2.239 GOP/s`，但前后 scope 文本同时发生变化，不能把 2.46× 全部当成同协议
leaf speedup。

本提交没有新建 runtime，也没有留下新 leaf 的独立执行日志；它证明 shared physical ability
进入了 compiler，随后第四轮的完整 source 运行才对这些 target 路径进行了更广覆盖。

### 5. 本轮没有完成什么

- grouped affine 仍要求明确的 local dtype、shape、little-endian 与 VLEN/resource legality；
- IQ3 等部分 fixed/scalable realization 仍只有很窄的 candidate space；
- dense 默认 candidate ordering 尚未因这一轮变成熟；
- 没有从 `materials/` 直接调用或复制实现，但本提交本身也没有留下逐条 donor-to-rule 记录；
- emitter 文件虽只拼写 exact leaf，KernelCompiler 内部仍保留局部 block/use relation 的协调代码。

---

## 第四轮：覆盖 `source/` baseline 并完成编译器收敛

### 范围与规模

边界：`aebcf4392`—`fcb535dff`，39 个提交，90 个文件，累计
`15,469 insertions / 1,298 deletions`。

文件分布反映了本轮不是简单填 CSV：

| 区域 | 主要作用 |
|---|---|
| `examples/kernels/` | 自然表达 forward、quantize/dequantize、vec-dot 与 production mul_mat 算法 |
| `examples/repro/weft/` | 为对应 source case 提供真实 shape、数据、正确性与计时 runtime |
| Kernel/Extension dialect + Python frontend | 补齐真正独立的局部量化数值 primitive |
| `lib/Target/` | typed decision、resource legality、RVV/IME leaf 与 intrinsic C 拼写 |
| `report/weft-kernel-performance.csv` | 记录 SG2044、K1 RVV 与 K1 IME 的最终运行数字 |

本轮的核心不是把 GGML 函数接进 Weft，而是把 GGML 已经证明重要的算法组织和局部硬件知识
按所有权重新表达：

```text
outer loop / blocking / workspace / packed layout → DSL kernel
local numerical relation                         → Kernel/Extension IR primitive
LMUL / vector shape / gather / widening / leaf   → physical planning
RVV intrinsic / IME asm spelling                 → local emitter leaf
```

`source/` 与 `materials/` 均未进入 Weft compiler 或生成代码的 production include/link/call/fallback。
部分 repro runtime 会显式 include 或 link GGML reference 来做数值、性能对照；reference 不参与
Weft lowering，也不是生成 kernel 的备用实现。

### 1. 05:40—06:17：先覆盖 forward、activation quantize 与 block dequant

`aebcf4392` 将 GGML forward primitives 接入已有 VLA、reduce、state、memory lowering。这里没有
创建“forward op”或按名字分派；add/sub/mul/div、activation、normalization、layout、gather、RoPE
和 attention 仍由现有 DSL 构造组合。

随后：

- `6208faad8`：用显式 reduce、scale、RNE+saturating narrow 和 store 表达 Q8_0/Q8_1/Q8_K
  activation quantization；
- `ba5732847`：建立共享 block dequant DSL kernel 与 runtime。

Q8 quantization 的 block stride、scale 计算、zero maximum 行为和 packed output layout由 DSL 拥有；
target 只实现 VLA/reduce/narrow/store 的机器形态。

### 2. 06:36—08:14：把 24 种 row dequant 作为普通 block/value 组合接入

这一阶段连续覆盖：

- packed block 在 ordered control 和 nested reduction 中的 carry；
- Q4/Q5/Q8 grouped affine 与 nonlinear decode；
- Q2_K/Q3_K/Q4_K/Q5_K/Q6_K bitplane 和 metadata；
- TQ1/TQ2 ternary；
- MXFP4/NVFP4；
- IQ2 table、IQ3 grid、IQ1 grid/delta、IQ4 nibble codebook。

重要的架构取舍是：**没有创建一个按格式分派的 `DequantizeOp`。** Row dequant 的 outer block
traversal、field offset、scale/zero-point、table pointer 和 output relation 均由 generic
load/decode/select/cast/pointwise/store 组合显式表达。只有无法从普通 SSA 图安全猜出的局部数值
关系，才成为 typed extension primitive。

这一步检验了 block value 能跨 ordered loop、nested reduction 和普通 consumer 继续使用，而不依赖
direct-store 或某一条完整 producer closure。

### 3. 08:25—09:03：完成 26 种 production `MUL_MAT` DSL 映射

从 Q4_0 开始，依次覆盖 legacy Q4/Q5/Q8、Q1、K-quant、ternary、IQ 与 FP4 projection。每个
production kernel 仍显式拥有：

```text
M/N/K traversal
activation quantization或已有activation layout
workspace与scale/code storage
packed weight block relation
row/block accumulator
最终store
```

target 只能为当前 local dot/matmul 选择 RVV 或 IME realization，不能调用 GGML，也不能把完整
projection 包成一个 leaf。

截至 `e8caa6969`，26 个 weight formats 的 decode/prefill production projection 都有自然 DSL
表达与对应 runtime。

### 4. 09:19—11:20：补齐 packed dot 与 IME local fragment

这一段加入或完善：

- packed i5×i8 typed RVV leaf，服务 Q5_0/Q5_1；
- Q4_1 affine IME fragment；
- 四行 i4 fragment 在 RVV 与 IME 中的局部 realization；
- IQ3/IQ1 在 VLEN256 上的 vector realization；
- 24 个标准 quantized vec-dot baseline runtime 与性能记录。

IME leaf 仍只实现当前 N16×K32 或 M4×N16×K32 block product。outer M/N/K、workspace、persistent
format 和 ABI 均来自相同 DSL kernel。

### 5. 13:21：source lowering 的第一次集中收敛

`6f71b0e04` 是本轮最大的单次 compiler 收敛节点：30 个文件，`1,605 insertions / 371 deletions`。
它把此前分批接入的 source mul_mat、vec-dot、quantize、dequantize 和 forward 路径统一到 typed
quant leaves 与 per-operation physical plan，并新增独立 ternary intrinsic C 模块。

这一节点同时做了两类清理：

1. 不合法 resource candidate 在生成代码前直接失败，例如 block reduction 没有合法资源时返回
   明确 unsupported；
2. 不再保留“某种 quant helper 不行就换另一条旧实现”的备用路径。

它不是把所有量化格式合成一个黑盒 op，而是让同一 vector shape、widening、gather、reduction
和 resource machinery 被多个局部 primitive 共享。

### 6. 13:47—14:50：用 typed codebook primitives 关闭剩余 vec-dot 缺口

这一阶段连续加入五组真正独立的局部数值关系：

| typed primitive | 服务的真实关系 | compiler 共享的机器能力 |
|---|---|---|
| `signed_codebook_i8_dot` | IQ2_XXS / IQ3_XXS signed table lookup + dot | indexed gather、widen、reduction |
| `packed_u9_u7_codebook_i8_dot` | IQ2_XS 的 u9/u7 packed index 与双 nibble scale | packed index、codebook gather |
| `packed_u11_grid_delta_i8_dot` | IQ1_S 的 grid index、delta 与 activation-sum correction | grid gather、correction handoff |
| `nibble_codebook_i8_dot` | IQ4_NL / IQ4_XS 16-entry nibble table dot | nibble decode、gather、widen |
| `packed_i3_grouped_i8_dot` | Q3_K three-bit grouped-scale dot | packed i3、group scale、reduction |

每个 primitive 同时具备 Python builtin、正式 Extension IR op/verifier、candidate facts、physical
selection 与独立 intrinsic C leaf。它们只拥有表中所列的局部数值语义；format stride、outer block
数量、scale 层次、row traversal 和 output ABI 仍属于 DSL kernel。

这五个节点没有形成 `if q_format == ...` 的 whole-kernel route。相反，多个格式会落到同一个
`CodebookGatherI8` physical family，再根据 typed operands、semantic lanes 和 target facts选择
具体 leaf。

### 7. 15:13：aligned F16 load 由 pointer facts 决定

`e17374786` 把 `load_f16_le` 的 aligned halfword realization 接到 pointer alignment 和 even-offset
关系。只有证明地址满足 alignment 时才选择 half load，否则使用合法的 byte spelling。

这不是按 Q5、F16 GEMM 或某个 kernel 名选择快路径；同一 pointer fact 可以被所有需要 little-endian
F16 load 的局部计算复用。

### 8. 两次完整 source 执行与最终 F16 candidate

第一次全量执行结束于 `a34cffe33`。随后没有因“已经 250 行 PASS”停止，而是继续处理 F16
dense 差距。

`616dc380d` 给 F16 matmul 增加真实 `columnMicrotile` candidate。它与 row microtile、input LMUL、
K-unroll 和一/二阶段 pipeline 一起进入 resource legality；emitter 按 selected column group复用
LHS，而不是从完整 GEMM 名称进入另一条实现。

当轮未归档到 CSV 的临时候选测量为：

- SG2044：column=1 约 `17.33 ms`，column=2 约 `15.57/15.33 ms`，因此 VLEN128 默认选择 2，
  该局部对照约改善 11%；
- K1：column=2 约 `42.08 ms`，column=1 约 `39.32 ms`，因此 VLEN256 保留 column=1。

这些 A/B 数字用于当轮选择，当前仓库只保留最终 winner 的全量数字，没有把临时候选另建一张
长期表。随后在最终实现上第二次完整执行，`fcb535dff` 记录最终 250 行。约 11 小时最后形成的
闭环是“第一次全量运行暴露问题 → F16 candidate 实测 → 第二次全量运行确认最终结果”，而不只是
补齐 CSV。F16 candidate 确实影响生成代码，但最终结果仍明显落后 GGML，不能称为 dense
performance 已闭合。

### 9. 被实测否决并清理的 Q5 尝试

本轮还依据 GGML donor 尝试过把 Q5 reduction zero seed 的 VL 从 1 改为 `vl32`，意图减少一次
VL 重设。该修改在 SG2044/K1 上没有形成可重复收益，SG q5_1 还出现退化/噪声，因此在提交前
完整撤回：

- 没有保留 config 字段；
- 没有保留备用 leaf；
- 没有把失败结果写入当前性能 CSV；
- 没有因一次失败删除整个 Q5 physical family。

这符合本轮方法：失败 candidate 可以删除，不能留下死分支，也不能把一次机器波动固化成永久规则。

## 第四轮最终覆盖到底是多少

固定 GGML baseline 是 **123 个逻辑 case、244 条硬件实测行**：

| 类别 | GGML 逻辑 case | GGML/Weft 标准硬件行 |
|---|---:|---:|
| Production `MUL_MAT`：26 formats × decode/prefill | 52 | 104 |
| Quantized vec-dot helper | 24 | 48 |
| Activation quantize | 3 | 6 |
| Row dequantize | 24 | 48 |
| Forward primitive | 20 | 38 |
| **标准合计** | **123** | **244** |

Weft 额外记录 K1 上 Q4_0/Q4_1/Q4_K 的 decode/prefill IME realization，共 6 行：

```text
244 条标准 target 行 + 6 条 K1 IME alternative = 250 条 Weft source realization
```

按 hardware 分：

| target | 标准行 | 额外 IME | Weft 最终行 |
|---|---:|---:|---:|
| SG2044 / RVV VLEN128 | 121 | 0 | 121 |
| K1/X60 / RVV VLEN256 | 123 | 0 | 123 |
| K1/X60 / IME1 VLEN256 | 0 | 6 | 6 |
| **合计** | **244** | **6** | **250** |

250 不是整个 CSV 的行数。当前 CSV 还保存此前 149 条 compiler-pressure 性能记录，所以数据行
总数为 399。

## 真实执行与正确性范围

最终 source coverage 的执行结果是：

```text
SG2044: 121 / 121 compiled, correctness-checked, executed
K1 RVV + IME: 129 / 129 compiled, correctness-checked, executed
总计: 250 / 250
```

所有行都走：

```text
DSL kernel
→ canonical Kernel IR
→ 当前唯一 RISC-V compiler
→ intrinsic C / local asm
→ system C compiler
→ SG2044 或 K1 真机
```

correctness scope 分布为：

| scope | 行数 | 含义 |
|---|---:|---|
| `full` | 90 | 完整结果对照 |
| `full_rows` | 48 | 完整 row-dot 输出对照 |
| `sampled` | 112 | 模型级大输出上的抽样数值对照 |

244 条标准行使用 `warmup=0`；6 条 IME 行使用 `warmup=3`。148 行不把 preprocess 计入 timed
region，102 行计入。所有行都有 `median_ms`，但这组协议差异也是本文只做方向性 GGML 比较的
原因。

全量 orchestration 只在 `/tmp` 中临时存在，执行完成后删除，没有把一次性的 case matrix 或验证
框架带进仓库。第一轮全跑期间修正了两个临时映射问题：Q4_K×Q8_K 使用 FFN-up decode
repetition；CSV 小写 IQ 名称映射到正式 DSL entry。这些是 orchestration 名称映射，不是 compiler
fallback。

Git 中保留的是两次 250 行整体刷新与最终结果，不保留另一套长期运行日志 authority。

## 第四轮性能分析

### 1. 按类别的整体方向

下表将 244 条标准 Weft 行和固定 GGML 行按 case、phase、hardware、shape 名义对齐。协议仍有
前述差异；几何平均与中位比率大于 1 表示 Weft 更快。

| 类别 / target | 行数 | 几何平均相对速度 | 中位相对速度 | 快于或等于 GGML |
|---|---:|---:|---:|---:|
| `mul_mat` / SG2044 | 52 | 0.851× | 0.836× | 14/52 |
| `mul_mat` / K1 | 52 | 0.619× | 0.708× | 15/52 |
| vec-dot / SG2044 | 24 | 0.941× | 0.857× | 10/24 |
| vec-dot / K1 | 24 | 0.824× | 0.849× | 6/24 |
| activation quantize / SG2044 | 3 | 0.834× | 0.865× | 0/3 |
| activation quantize / K1 | 3 | 0.884× | 0.926× | 0/3 |
| row dequantize / SG2044 | 24 | 1.273× | 1.004× | 12/24 |
| row dequantize / K1 | 24 | 1.359× | 1.661× | 15/24 |
| forward / SG2044 | 18 | 1.230× | 1.007× | 9/18 |
| forward / K1 | 20 | 1.172× | 1.122× | 13/20 |

整体图景很明确：forward 和许多 row dequant 已经有竞争力；production matrix multiplication 是
当前最大性能缺口，K1 比 SG2044 更明显。

### 2. 代表性 matrix multiplication

| case | SG2044 Weft / GGML | K1 Weft / GGML | 判断 |
|---|---:|---:|---|
| F32 decode | 2.304 / 1.628 GOP/s = 1.42× | 1.689 / 2.007 = 0.84× | SG 已领先，K1 仍慢 |
| F16 decode | 2.070 / 4.542 = 0.46× | 0.878 / 3.429 = 0.26× | 两 target 均明显落后 |
| F16 prefill | 5.833 / 14.358 = 0.41× | 1.816 / 5.488 = 0.33× | column candidate 尚未闭合 dense 差距 |
| Q4_K decode | 9.822 / 8.923 = 1.10× | 2.246 / 9.766 = 0.23× | K1 标准对照为 GGML IME |
| Q4_K prefill | 9.006 / 9.672 = 0.93× | 2.247 / 24.577 = 0.09× | K1 RVV 与 GGML IME 不可解释为 RVV 对 RVV |
| Q5_K decode | 2.385 / 1.305 = 1.83× | 0.699 / 1.231 = 0.57× | SG 有优势，K1 仍落后 |

K1 的 Q4_0/Q4_1/Q4_K 还应看同一 Weft DSL 的 IME realization：

| K1 IME case | Weft | GGML IME | 相对速度 |
|---|---:|---:|---:|
| Q4_0 decode | 10.660 GOP/s | 11.409 | 0.93× |
| Q4_0 prefill | 19.306 | 28.431 | 0.68× |
| Q4_1 decode | 8.400 | 9.535 | 0.88× |
| Q4_1 prefill | 17.469 | 24.581 | 0.71× |
| Q4_K decode | 8.388 | 9.766 | 0.86× |
| Q4_K prefill | 17.452 | 24.577 | 0.71× |

IME decode 已进入接近 baseline 的区间；prefill 仍缺更好的 fragment/packing/pipeline 与 local reuse。

### 3. Forward、quantize 与 dequantize

- SG/K1 RMSNorm 分别约为 GGML 的 2.60×/1.60×；
- SG/K1 flash attention 仅约 0.62×，summary/state 与数据复用仍不足；
- K1 contiguous transpose 约 0.21×，说明识别 strided/indexed memory 不等于已经拥有成熟 schedule；
- 三个 Q8 activation quantizer 在两台机器都略慢于 GGML，整体约 0.83—0.88×；
- row dequant 的几何平均已领先，但格式差异很大：K1 Q8_0 约 3.96×，而 K1 IQ2_S 约 0.46×；
- SG IQ4_NL dequant 约 4.55×，而 SG IQ1_M 约 0.47×。

这些差异说明“共享 VLA/decode lowering 已覆盖”与“所有 packed organization 都已成熟”是两件事。

## 四轮后真正形成的编译器结构

当前 production 路径可以概括为：

```text
Core-local blocked DSL kernel
        ↓
canonical Kernel IR
        ↓
axis / use / memory / lifetime / primitive facts
        ↓
resource-legal physical selection
  - value and handoff shape
  - LMUL / memory form / state placement
  - microtile / unroll / local pipeline
  - quant gather / widening / reduction
  - RVV / IME exact leaf
        ↓
KernelCompiler emits ordinary C control and addresses
        ↓
RVV intrinsic C / primitive-local IME asm spelling
        ↓
system compiler
```

生产路径审计未发现：

- 按 kernel 名选择实现；
- 按 q-format 字符串接管 whole kernel；
- exact op-count 或 direct-store fast path；
- 生成的 Weft kernel 或 compiler 对 GGML/materials 的 include、link 或 runtime call；repro runner
  中显式链接的 GGML reference 只用于同输入对照；
- legacy/scalar 静默 fallback；
- emitter 根据 VLEN、format 或完整 source closure 再选一次 exact leaf。

`RISCVLowering.cpp` 对没有实现的 target 明确报告 no fallback；CLI 的 `--kernel` 只选择多-entry
文件中的入口，不参与 lowering 语义。

## 没有被四轮结果掩盖的问题

### 1. KernelCompiler 仍然偏大

文件级职责已经拆开，但 `RISCVKernelCompiler.cpp` 仍同时协调多类 operation 的 prepare/decide/emit。
其中仍有 `collectBlockClosure`、block store/reduce 的局部 use-def closure 收集。它不是按完整 kernel
分类的 whole-region matcher，但也说明 value-chain planning 尚未完全收敛为更小的共享机制。

因此不能声称“所有 closure 代码已经删除”。准确说法是：高性能入口不再依赖 kernel 名、精确
op 数、one-use、direct-store 或完整外围 loop closure；局部 block producer/use relation 仍用于
合法 fusion、rematerialize、store/reduce projection。

### 2. Candidate space 宽度不均衡

F32 dot、F16 matmul、VLA LMUL 和 reduction state 已有多个真实 candidate；很多 quant/codebook
leaf 仍主要是在 VLEN128/VLEN256/fixed/scalable 之间选择，microtile、cross-output reuse、gather
schedule 和 pipeline 空间较窄。

IME 也已经是同一 primitive 的 local realization，但 fragment family、local packing 和 resource
model 仍比较固定，尚未形成成熟的多候选 compile-and-tune space。

### 3. Dense target selection 还不成熟

F16 column microtile 证明 target 可以选择不同合法实现，但 F16/F32 prefill、尤其 K1 结果仍落后。
问题已经不再是“没有 matmul primitive”，而是 row/column microtile、LMUL、K-unroll、load reuse、
pipeline 与真实 target cost 的联合选择质量不足。

### 4. Generic composition 与 typed primitive 的边界仍需持续保持

quantize/dequantize 当前主要由普通 block/VLA/load/decode/arithmetic/store 组合表达，这符合作者
显式拥有 packed layout 和算法组织的边界。Vec-dot 中那些普通 SSA 无法无歧义表达的 packed
数值关系则成为 typed local primitives。

不能为了让 backend 更容易而把每个格式变成黑盒 op；也不能反过来要求 compiler 从任意 bitwise
SSA 图猜出 IQ/Q/K-format。第四轮的实现选择了这两者之间的局部语义边界，但未来增加格式时仍需
继续遵守。

### 5. 性能记录不是统一 benchmark framework

250 条记录覆盖真实硬件与模型级 shape，但 warmup、repetition、preprocess 和 correctness scope
仍按 case 变化。它足以暴露 2×—10× 的性能问题，不足以对 0.95× 和 1.05× 做最终判断。

这张 CSV 只是一份数字记录，不承载阈值、同步、检查或验收逻辑。

## 总判断

这四轮的真正推进可以压缩成四句话：

1. 第一轮让 physical shape、resource 与 schedule 各自只有一个决定 owner；
2. 第二轮把 facts、physical selection 和 intrinsic/asm spelling 从单一 lowering 文件中拆出；
3. 第三轮证明新增 VLEN256/quant realization 可以扩张同一 local primitive，而不增加 kernel route；
4. 第四轮用 123 个真实 GGML 逻辑 case、244 条标准 target 行和 6 条额外 IME realization，证明
   这条唯一主链具有完整 source 覆盖能力。

但最终状态仍应诚实表述为：

> **Weft 已从“有一批能跑的高性能 case”推进到“拥有可解释、可扩展、能覆盖真实 baseline 的
> RISC-V kernel compiler”；它尚未推进到“各主要 primitive 都有成熟候选空间并普遍达到 GGML
> 性能”的阶段。**
