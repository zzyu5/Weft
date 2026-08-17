# Weft RISC-V 编译决策模型到 baseline 最终收敛：四轮推进报告

## 结论

本文记录的是上一份 source 全覆盖工作之后的四轮推进，严格对应以下四个 prompt：

| 轮次 | 本轮问题 | 提交边界 | 提交数 |
|---|---|---|---:|
| 第一轮 | 重构 Weft 的编译决策模型 | `68b2c2a2b` | 1 |
| 第二轮 | 完成高性能 RISC-V 编译器 | `d16cd26bc`—`5f5787e6a` | 10 |
| 第三轮 | 完成全部现有 Kernel 的统一高性能编译 | `7fadd81cb`—`e98a02fb3` | 7 |
| 第四轮 | 覆盖有 baseline 的项目并完成最终收敛 | `9416278bb`—`0853aa234` | 5 |

这四轮没有重新设计 Core-local blocked DSL，也没有再增加一批 kernel。它们解决的是此前
source 全覆盖后仍存在的编译器内部问题：目标事实、结构实现、参数和 exact intrinsic leaf 仍然
有历史 family 身份；memory、state、dense、quant 和 emission 各自保留部分决定；一些性能能力
在重构后没有被真实全量运行及时发现已经退化。

最终状态可以概括为：

> **Weft 现在以 axis、value/use、memory、lifetime、primitive 和 target facts 为输入，先形成唯一的
> 结构实现与参数实例，再选择 exact RVV/IME local leaf，最后生成 intrinsic C 或局部 asm。最终
> 123 个 baseline 逻辑 case 对应的 244 条标准 target 记录和 6 条额外 K1 IME realization 已在最终
> 代码上完成第二次全量真机运行，250/250 均完成编译、数值检查和执行。**

这一结论不等于主要性能问题都已解决。Forward 与大量 row dequant 已有竞争力；SG2044 Q4_K
production 已接近 GGML；F16/F32 prefill、K1 dense、部分 quant/codebook 和 flash attention 仍有
明确差距。最终 CSV 保留这些差距，没有通过旧实现或特殊路径掩盖。

## 报告边界

本文是一次性工作记录，不定义语言或编译器规范。当前规范仍由
[`doc/index.md`](../doc/index.md) 及其链接模块定义。本文使用的当前数据为：

- Weft 当前性能数字：[`weft-kernel-performance.csv`](weft-kernel-performance.csv)；
- 固定 GGML 数字：[`baseline/ggml-riscv-kernel-performance.csv`](baseline/ggml-riscv-kernel-performance.csv)；
- baseline 硬件与计时说明：[`baseline/ggml-riscv-kernel-baseline.md`](baseline/ggml-riscv-kernel-baseline.md)。

性能比率统一写成：

```text
baseline median_ms / Weft median_ms
```

大于 1 表示 Weft 更快，小于 1 表示 Weft 更慢。只比较相同硬件、算法、shape、dtype/量化格式和
phase；K1 的 GGML Q4_0/Q4_1/Q4_K production baseline 是 IME，因此不能拿它和 Weft K1 RVV
结果形成 RVV 对 RVV结论。

当前 Weft 与固定 baseline 的 warmup、repetition 和 correctness scope 仍不是逐项完全相同。
64 MiB eviction、kernel invocation 范围和 median 记录一致，但接近 1.0 的比率只表示方向，不是
最终胜负判定。数量级差距则足以暴露实现空间问题。

## 四轮之前的真实状态

进入第一轮之前，以下基础已经存在：

- 唯一的 Core-local blocked Python DSL 与 canonical Kernel IR；
- `RISCVKernelFacts`、`RISCVPhysicalPlanning`、`RISCVKernelCompiler` 和局部 intrinsic/asm 模块；
- 123 个 baseline 逻辑 case 的 DSL/runtime 映射与 250 条 Weft target 记录；
- RVV VLEN128、RVV VLEN256 和局部 IME realization；
- value shape、handoff、resource 与 selected leaf 的初步物化。

但当时的编译器仍保留了以下历史结构：

1. `RVVVLEN128*`、`RVVVLEN256*`、`FixedLanes32/64` 等名字把 target 参数误写成 realization 身份；
2. 一些 family selector 同时决定 primitive、结构、参数和 leaf，难以解释“为什么合法”；
3. memory emission 仍从地址表达重新恢复部分 memory 选择；
4. state、dense pipeline、quant decode 和 fragment 的资源与实现决定尚未各自闭合；
5. exact leaf、C ABI 和 RVV intrinsic type spelling 仍与 KernelCompiler 交织；
6. 所有 examples 虽已有 DSL 表达，但没有在新决策模型完成后重新做完整 baseline 全跑。

因此这四轮的顺序不是再增加功能列表，而是：

```text
先重建决定模型
→ 再关闭 memory/state/dense/quant/leaf 的唯一 authority
→ 再用全部现有 kernel 消除对偶然源码形状的依赖
→ 最后只对真实 baseline 项目做两次全量真机运行并修复回退
```

---

## 第一轮：重构 Weft 的编译决策模型

### 范围

提交 `68b2c2a2b`，修改 22 个文件，净变化为 `3,065 insertions / 2,652 deletions`。主要文件为：

- [`include/Weft/Target/RISCVLowering.h`](../include/Weft/Target/RISCVLowering.h)；
- [`lib/Target/RISCVPhysicalPlanning.h`](../lib/Target/RISCVPhysicalPlanning.h)；
- [`lib/Target/RISCVPhysicalPlanning.cpp`](../lib/Target/RISCVPhysicalPlanning.cpp)；
- [`lib/Target/RISCVKernelCompiler.cpp`](../lib/Target/RISCVKernelCompiler.cpp)；
- [`lib/Target/RISCVIntrinsicC.h`](../lib/Target/RISCVIntrinsicC.h) 及 RVV/IME/quant leaf 文件。

本轮没有修改 examples、DSL 或 Kernel IR。它重构的是 target lowering 对机器决定的表达方式。

### 1. 明确区分输入事实、结构实现和参数

公开 backend config 被收敛为两组不同性质的选择：

```text
RISCVStructuralConfig
  reductionStatePlacement
  i4I8FragmentImplementation

RISCVCandidateParameters
  vlaLMUL
  dotLMUL / dotKUnroll
  f16InputLMUL
  f16RowMicrotile / f16ColumnMicrotile
  f16KUnroll / f16LoadBufferCount
  narrowLMUL
  sortRadixBits
```

Target profile 提供 VLEN、SEW/LMUL legality、寄存器数量、indexed/segment memory、Zvfh 与 IME
等硬件事实。它们都只是候选合法性和资源计算的输入，不能再充当 kernel route 或 realization 名称。

结构实现与参数也不再混为一体。局部实现统一表达成：

```text
LocalImplementation
  primitive   : 它实现哪个显式局部语义
  structure   : RVV register microkernel / RVV strip loop / IME fragment
  parameters  : semantic lanes、microtile、vector shape、entry width
  leaf        : 已经选定的 exact intrinsic/asm spelling
```

这使“使用 RVV register microkernel”与“使用 16 或 32 semantic lanes”成为两个不同问题。

### 2. 删除由 VLEN 或 lane 数命名的 realization 身份

本轮删除了多组历史枚举和 leaf 身份，包括：

```text
RVVVLEN128LocalBlockDot / RVVVLEN256LocalBlockDot
RVVVLEN128GatherDot / RVVVLEN256GatherDot
RVVVLEN128TableDot / RVVVLEN256TableDot
RVVVLEN128GroupedDot / RVVVLEN256GroupedDot
IQ2/IQ1/Q6 FixedLanes32 / FixedLanes64 leaf
```

替代方式不是把条件藏到另一个 emitter，而是让 selector 根据 primitive typed facts 和 target facts
先构造 `LocalImplementationStructure + LocalImplementationParameters`，再在 planning 中映射到 exact
leaf。VLEN128/VLEN256 只通过 lane capacity、legal vector shape 和 resource budget影响结果。

### 3. 把所有现有 local family 迁入同一决定模型

同一次改动迁移了：

- F32 math；
- symmetric/affine/grouped affine i4×i8；
- E2M1/E8M0、packed i4/i5/i3、ternary；
- signed/packed/nibble codebook；
- IQ1/IQ2/IQ3/Q6；
- RVV register、RVV strip 与 SpacemiT IME fragment。

因此本轮不是为一个量化格式做试点。旧 enum、旧 leaf、旧 prelude 参数和旧 quant helper 接口在
同一提交中删除，没有保留新旧两套 production authority。

### 4. 这一轮建立的核心判据

从此一个局部实现必须能回答：

1. 它实现哪个显式 DSL/IR primitive；
2. 它采用哪种机器结构；
3. 参数由什么 typed facts 和 target resource 约束得到；
4. exact leaf 是什么；
5. 它需要和产生什么 physical shape；
6. emitter 为什么只需要拼写这一 leaf。

VLEN、target 名或量化格式不再能单独回答这些问题。

### 5. 第一轮没有完成什么

第一轮完成的是决定类型和 family 迁移，不是全部 physical authority 的闭合：

- memory emission 仍需退出地址重新推导；
- state placement、dense pipeline、quant resources 仍需进一步形成唯一 selected record；
- exact leaf、C ABI 与 RVV API spelling 仍需从 KernelCompiler 分离；
- 没有在这一提交上做完整 baseline 重跑。

---

## 第二轮：完成高性能 RISC-V 编译器

### 范围

提交 `d16cd26bc`—`5f5787e6a`，共 10 个提交；相对第一轮修改 17 个文件，净变化为
`1,898 insertions / 1,342 deletions`。

本轮没有增加新的长期 IR 或 registry。每个提交都关闭一项已存在能力的 owner，形成以下信息流：

```text
Kernel IR facts
→ structural realization
→ resource-legal parameter instance
→ selected value/memory/state/primitive decisions
→ exact local leaf
→ intrinsic C / local asm spelling
```

### 1. Memory emission 改为消费 address decisions

`d16cd26bc` 扩充 `RISCVKernelFacts` 中的 address projection，使 semantic pointer、root pointer、axis
relation、coordinate 与 invariant base 在 facts/decision 阶段闭合。生成 load/store 时只消费已经
选定的地址关系，不再从当前 C expression 或 producer closure 猜 unit/strided/indexed 形式。

这个变化直接服务等价 pointer-base、不同 SSA 临时值和 structured value 多 consumer：源码拼写改变
不应导致 memory realization 消失。

### 2. Dense pipeline 成为真实 selected decision

`bce739ccc` 将 dense 的寄存器组织与 load pipeline 写成正式结构：

```text
DenseVectorOrganization
DenseLoadSchedule
F32DotParameters
F16MatmulParameters
DenseMicrokernelResourceFacts
```

当前真实候选包括：

- F32 的 LMUL 与 K-unroll；
- F16 的 row/column microtile、input LMUL 与 K-unroll；
- streamed load 与 register double buffering；
- accumulator、lhs/rhs load window、predicate、state 和 handoff 的联合寄存器预算。

资源不足的组合在生成 C 之前被删除。Emitter 只按 selected `DenseLoadSchedule` 生成 streamed 或
double-buffered code，不再自行判断是否“像 GEMM”。

### 3. State 使用一份完整物理决定

`36c4fafd6` 把 state 的以下内容合成 `SelectedVLAStatePhysical`：

```text
carry representation
strip update
finalize
whole-VLA lifetime
```

Reduce、scan、segmented scan 和 widening reduction 仍保留各自语义。Selector 可以决定 scalar 或
vector carry，但不能把 ordered state 改成 relaxed reduction。资源规划与 emission 读取同一个
selected state，不再分别维护 state schedule 和 state resource schema。

### 4. Exact intrinsic leaf 在 emission 前选定

`1b78c3f44` 将 exact leaf 的选择移入 planning。`collectSelectedLocalImplementations()` 在生成 kernel
body 前收集实际使用的 leaf；prelude 和各 quant/RVV/IME 模块只输出这些 leaf 所需 helper。

如果 selected decision 缺少 leaf、operand shape 或 handoff，lowering 明确失败。Emitter 不会改用
另一个 helper、旧 implementation 或 scalar 版本。

### 5. C ABI 与 RVV intrinsic spelling 独立

`e7ecd6691` 新建 `RISCVCABI.{h,cpp}`，统一 kernel body 与 C header 的 scalar/pointer/type spelling。
`13c1b569e` 新建 `RISCVRVVSpelling.{h,cpp}`，统一 vector type、LMUL suffix 与 intrinsic type suffix。

这两个模块没有 compiler authority：

- C ABI 模块不决定 storage、shape 或 lifetime；
- RVV spelling 模块不选择 LMUL、microtile 或 leaf；
- RVV intrinsic API 拼法变化不需要回到 axis/resource selector 修改逻辑。

### 6. Quant decode、widening 与 resource 形成共享结构

`a4476ab7e` 与 `b96986a7a` 删除 family 内散落的临时向量计数，统一通过
`QuantDecodeResourceFacts` 表达：

```text
loaded values
index values
temporary values
predicate groups
```

`calculateQuantDecodeResources` 将它们换算为同一时间存活的 register groups，再与 target vector
register budget比较。Codebook gather、nibble decode、ternary、packed i3/i4/i5、IQ 和 grouped
affine 因此共享同一资源公式，而不是每个格式各写一份。

### 7. RVV 与 IME 是同一 primitive 的结构候选

`d05464311` 把 symmetric/affine i4×i8 的 RVV register microkernel 和 IME fragment放入同一个
`selectI4I8FragmentPhysical`。Target profile 决定某个 fragment 是否合法，backend config 可以在
合法结构间选择；outer M/N/K traversal、activation quantization、workspace 与 output ABI 都不属于
fragment。

`5f5787e6a` 最后把 dense 的 load buffering 名称收敛为实际代码含义，删除模糊的 pipeline stage
说法：当前只有真实生成 streamed 与 register-double-buffered code 的维度被保留。

### 8. 第二轮的模块职责

本轮后当前核心模块的职责基本形成：

| 模块 | 当前职责 |
|---|---|
| `RISCVTargetProfile` | ISA、ABI、VLEN、SEW/LMUL、memory 与 extension legality |
| `RISCVKernelFacts` | axis、ordered parent、value use/lifetime、memory relation |
| `RISCVPhysicalPlanning` | 结构候选、参数实例、resource filtering、exact local leaf |
| `RISCVKernelCompiler` | Kernel IR traversal、相连 value 的 physical plan、C 控制流协调 |
| `RISCVCABI` / `RISCVHeader` | 已知 ABI 与 metadata 的 C 拼写 |
| `RISCVRVVSpelling` | 已知 vector shape 的 RVV type/intrinsic suffix |
| `RISCVIntrinsicC*` | 已选 RVV/quant helper 与 IME local asm 的机械输出 |
| `RISCVLowering` | fixed-RVV target gate 与唯一 compiler facade |

这不是把一个大 if-else 换成空接口。Dense、state、quant resource 和 RVV/IME fragment均已有
两个以上真实实现或明确结构差异；没有真实第二个实现的字段被删除或精确改名。

---

## 第三轮：完成全部现有 Kernel 的统一高性能编译

### 范围

提交 `7fadd81cb`—`e98a02fb3`，共 7 个提交；修改 12 个文件，净变化为
`394 insertions / 261 deletions`。其中前 6 个提交继续修改 compiler，最后一个提交只把已经形成的
方法写回现行文档。

本轮仍没有修改 examples。原因不是 examples 没有经过新 compiler，而是 Core-local blocked DSL
已经在更早的迁移中成为全仓唯一语言；本轮缺口属于 target realization，不属于作者算法结构。

### 1. Dense structure 由轴事实推导

`7fadd81cb` 将 dense 的结构入口收敛到 `analyzeStructuredProductFacts`。它从 lhs/rhs/result 的 axis
identity 推导：

```text
lhs free axes
rhs free axes
reduction axes
result axes
```

再结合哪一条 free axis 位于显式 VLA 中，选择以下结构类别：

```text
lhs VLA free axis → RVV VLA vector dot
rhs VLA free axis → RVV VLA microtile
无 VLA free axis → local-row register microkernel
两侧同时是 VLA free axis → 当前 target 明确 unsupported
```

外围 loop 数量、kernel 名、direct store、固定 producer 数量都不是入口条件。结构确定后才枚举
LMUL、K-unroll、microtile 和 load schedule。

### 2. Structured value handoff 被显式物化

`4d089b45c` 为每个 `(consumer, value)` 记录唯一 handoff：

```text
Share
Convert
Rematerialize
Reload
LocalPack
```

每条记录同时带 source shape 与 result shape。一个 dot/matmul 结果经过 pointwise、出现第二个
consumer、跨控制流保存或延后 store 时，emitter不再从当前位置猜它应继续作为哪种向量。

同一个 value 若收到冲突的 physical shape，或同一 consumer/value 的 handoff kind/shape不一致，
physical plan 在 emission 前失败，而不是选择一个方便生成的版本。

### 3. Segment memory 的最终 shape 成为 decision

`5486a27ef` 让 segment access 的以下信息一次决定并一直携带到 emission：

```text
load/store kind
field count
element SEW
coordinate scale
selected RVV shape
```

Emitter 不再根据 pointer pattern 重新算 field/SEW/LMUL。普通 unit、strided、indexed 与 segment
memory 因此使用同一套 pointer/axis relation facts，但拥有不同合法 physical realization。

### 4. State、quant 与 fragment resource 只保留被消费的结果

`36ae921d8` 使 state 和 quant plan 直接携带 selected resources；`86b59ab09` 删除没有任何下游消费
者的 quant decision 字段；`f02c1876a` 则从 i4/i8 operand/accumulator shapes 推导 fragment resource，
不再复制一份固定寄存器常数。

这一组改动的重要结果不是字段变少，而是每个物理事实只有一个 producer：

- shape 由 selected value/primitive record 持有；
- handoff 由 consumer/value record 持有；
- resource 与 selected realization 一起产生；
- exact leaf 在 body emission 前确定；
- emitter 只能核对并消费这些结果。

### 5. 现有 examples 为什么大多没有修改

所有正式入口都通过 [`examples/run/weft.sh`](../examples/run/weft.sh) 执行：

```text
python3 -m weft <DSL kernel>
→ canonical Kernel IR
→ weft-compile --emit=intrinsic-c
→ target C compiler
→ target runtime
```

因此一个 example 的 Python 文件三四天没有变化，只表示作者写下的 traversal、blocking、state、
workspace 和 primitive 语义没有变化；每次运行仍会重新进入当前 `weft-compile` 和当前 target
lowering。State、vision、memory、quant、codebook 和 dense examples 都由统一 runner 映射到各自
DSL source，不存在冻结的旧 generated C。

这一轮没有为了证明 compiler 改写 DSL，也没有增加 normalization pass。只有当作者算法本身与
baseline 不一致时，才应修改 example；这一情况最终在第四轮的 F16 production 中真实出现。

### 6. 方法写回现行文档

`e98a02fb3` 更新编译器与 kernel 文档，记录代码中已经成立的方法：

```text
typed program/target facts
→ 唯一合法性与轴关系推导
→ structural implementation candidates
→ parameter instances 与 resource filtering
→ selected physical decisions
→ intrinsic C / typed local asm
```

文档明确：长期表示只有 canonical Kernel IR；physical plan、candidate、LMUL、fragment 和 leaf
都是一次 lowering 内的瞬态值，不成为第二份可输入 IR。

### 7. 第三轮没有声称什么

第三轮证明所有当前 primitive 已迁移到同一种编译方法，但没有做全量性能运行，也没有把
“代码路径统一”当成“性能没有回退”。第四轮专门用固定 baseline 语料检查这一点。

---

## 第四轮：覆盖有 baseline 的项目并完成最终收敛

### 范围

提交 `9416278bb`—`0853aa234`，共 5 个提交；修改 10 个文件，净变化为
`605 insertions / 282 deletions`。

本轮没有运行无 baseline 的 examples。执行范围冻结为 source 中有真实 baseline 且已有对应 DSL
kernel 的 123 个逻辑 case。先做一次完整运行暴露问题，随后只重跑受影响项目；共享修复收敛后，
清空临时结果并在最终代码上做第二次完整运行。

### 1. 最终执行范围

| 类别 | 逻辑 case | 标准 target 行 | 额外 IME 行 |
|---|---:|---:|---:|
| Production `MUL_MAT`：26 formats × decode/prefill | 52 | 104 | 6 |
| Quantized vec-dot | 24 | 48 | 0 |
| Activation quantize | 3 | 6 | 0 |
| Row dequantize | 24 | 48 | 0 |
| Forward primitive | 20 | 38 | 0 |
| **合计** | **123** | **244** | **6** |

按 target 分：

| target | 行数 |
|---|---:|
| SG2044 / RVV VLEN128 | 121 |
| K1/X60 / RVV VLEN256 | 123 |
| K1/X60 / IME1 VLEN256 | 6 |
| **合计** | **250** |

250 是 source coverage 记录，不是 kernel 数量，也不是整个 CSV 的行数。当前 CSV 另有此前保留的
149 条 compiler-pressure 数字，因此总数据行数为 399。

### 2. 第一遍全量运行暴露的四个问题

#### 2.1 K1 fractional-LMUL cast 无法拼写

K1 flash attention 首先在 intrinsic C 边界失败：

```text
VLA cast shape has no intrinsic-C spelling
```

`9416278bb` 没有把 LMUL 改回整数或增加 K1 特例，而是让 `emitCast` 使用统一的
`rvvVectorType` 与 `rvvIntrinsicTypeSuffix` 消费 selected source/result shape。F16/F32 的 fractional
LMUL 因此和其他 RVV value 使用同一 spelling 机制。SG/K1 flash attention 随后均能编译执行。

#### 2.2 F16 production 的算法输入和 preprocessing 归属不真实

旧 `blocked_gemm` example 直接接收 F16 activation，和真实 F16 production baseline 的 F32
activation 输入不一致。这个差异属于作者程序，不应由 target 猜测。

`c31f3c05c` 因此只修改这一份 DSL kernel：

```text
F32 activation input
→ 作者显式 W.vla cast
→ W.workspace 中的 F16 staging
→ 原有 W.block + W.matmul
```

Workspace 通过 `W.storage` 声明 shape/lifetime；runtime 同步传入 F32 input 与 F16 workspace。
当前性能 CSV 将这一 scope 记录为
`activation-f16-staging-plus-direct-generated-matrix-multiplication`，并把 preprocessing 计入 timed
region。性能数字因而更慢，但 workload 与所有权边界变得真实。

这是四轮中唯一修改的 example。其余 examples 无需变化，因为它们的作者算法结构已经正确，
问题属于 target realization。

#### 2.3 Q4_K grouped affine 被通用 strip 实现拖慢

第一遍运行发现 Q4_K production 明显回退：SG prefill 约 3.16 s，K1 prefill 约 2.86 s。检查生成
代码后确认，`GroupedAffineI4I8` 在决定模型重构时只剩通用逐 strip realization，丢失了同一局部
primitive 已有的 register microkernel 结构。

`386595bba` 恢复的是共享结构，不是 Q4_K kernel route：

- selector 从 e8m1 的实际 lane capacity得到 16 或 32 semantic lanes；
- 选择 `RVVRegisterMicrokernel` 或合法的 `RVVStripLoop`；
- register implementation 同时携带 packed/scale/activation/activation-sum shapes 与资源；
- SG 使用对应的 VLEN128 register organization，K1 使用 32-lane RVV intrinsic organization；
- production mul_mat 与独立 vec-dot 共用 `GroupedAffineI4I8` local primitive。

最终 SG Q4_K prefill 恢复为 463.928 ms，K1 RVV 为 1,912.203 ms。SG 已回到 baseline 附近；K1
RVV 仍慢，但没有用 K1 IME baseline 冒充 RVV 对照。

#### 2.4 重构后的统一默认值忽略了 target resource 差异

第一遍运行还暴露了 reduction、VLA cast 和 F32 dense 的统一默认选择在两台机器上含义不同。
`f254cf2f1` 将选择改回 target-fact-driven：

- relaxed single reduction 在 VLEN128 可保留 vector carry，VLEN256 根据资源与代价保留 scalar；
- 带 local primitive 的 F32↔F16 cast 在 SG 选择 LMUL2、K1 选择 LMUL4；
- cast-only staging 仍可选择更宽 VLA；
- F32 local-row dense 的 desired lanes 从 e32m1 在当前 target 的实际 lane capacity推导；
- K-unroll 同时读取 reduction extent、handoff、predicate、indexed memory 与 row/resource 关系。

这里使用的是 VLEN、lane capacity 和寄存器预算等 target facts，不是 `if target == K1` 或
`if kernel == ...`。

### 3. 最终第二遍全量运行

所有修复完成后，旧临时结果被清空，在最终提交上重新完成 250 条运行：

```text
source_forward          38 / 38
source_mul_mat         104 / 104
source_vec_dot          48 / 48
source_quantization      6 / 6
source_dequantization   48 / 48
source_mul_mat_ime       6 / 6
--------------------------------
total                  250 / 250
```

每一行都完成：

```text
DSL kernel
→ canonical Kernel IR
→ 当前唯一 RISC-V lowering
→ intrinsic C / local asm
→ system C compiler
→ SG2044 或 K1 真机
→ runtime 自身定义的 correctness check 与计时
```

当前 250 行的执行元数据为：

| 项目 | 分布 |
|---|---|
| repetitions | 132 行为 3 次；118 行为 10 次 |
| warmup | 244 条标准行是 0；6 条 IME 行是 3 |
| correctness | 90 full；48 full_rows；106 sampled；6 activation_full_output_sampled |
| preprocess timed | 144 no；106 yes |
| eviction | 全部 64 MiB |

仓库没有保留新的 case matrix、test framework 或第二份运行 authority。正式单项入口仍是
[`examples/run/weft.sh`](../examples/run/weft.sh)；全量 orchestration 是本轮临时组合，完成后没有
进入仓库。最终 250 行由 `0853aa234` 写回当前性能 CSV。

### 4. 当前类别级性能

下表仅对齐相同 hardware、kernel、phase 和 shape。K1 `mul_mat RVV` 排除了 baseline 实际使用
IME 的 Q4_0/Q4_1/Q4_K 六行；这六行单独作为 `K1 IME` 比较。

| 类别 / target | 可比行 | 几何平均相对速度 | 中位相对速度 | Weft 不慢于 baseline |
|---|---:|---:|---:|---:|
| Production mul_mat / SG RVV | 52 | 0.851× | 0.832× | 15/52 |
| Production mul_mat / K1 RVV | 46 | 0.757× | 0.774× | 15/46 |
| Production mul_mat / K1 IME | 6 | 0.790× | 0.782× | 0/6 |
| Quantized vec-dot / SG RVV | 24 | 0.944× | 0.858× | 10/24 |
| Quantized vec-dot / K1 RVV | 24 | 0.824× | 0.849× | 6/24 |
| Activation quantize / SG RVV | 3 | 0.823× | 0.866× | 0/3 |
| Activation quantize / K1 RVV | 3 | 0.937× | 1.128× | 2/3 |
| Row dequantize / SG RVV | 24 | 1.283× | 1.041× | 13/24 |
| Row dequantize / K1 RVV | 24 | 1.323× | 1.608× | 15/24 |
| Forward / SG RVV | 18 | 1.221× | 1.010× | 9/18 |
| Forward / K1 RVV | 20 | 1.138× | 1.085× | 14/20 |

整体方向没有被全量 PASS 掩盖：row dequant 与 forward 的共享 VLA/memory/state lowering总体已有
竞争力；production mul_mat、尤其 K1 dense/quant，仍是最主要的性能缺口。

### 5. Dense 与 Q4_K 的代表结果

| case | SG2044 | K1/X60 | 判断 |
|---|---|---|---|
| F32 decode | 14.847 vs 20.612 ms，1.388× | 19.999 vs 16.715 ms，0.836× | SG 领先，K1 落后 |
| F32 prefill | 847.535 vs 574.193 ms，0.678× | 5205.112 vs 1606.776 ms，0.309× | prefill candidate/复用仍不足 |
| F16 decode | 13.123 vs 7.387 ms，0.563× | 40.005 vs 9.785 ms，0.245× | 显式 staging 后仍有大差距 |
| F16 prefill | 743.040 vs 299.138 ms，0.403× | 4267.736 vs 782.657 ms，0.183× | 当前最大 dense 差距之一 |
| Q4_K decode | SG RVV 3.396 vs 3.760 ms，1.107× | K1 IME 4.031 vs 3.436 ms，0.853× | SG 已有竞争力；IME decode 接近 baseline |
| Q4_K prefill | SG RVV 463.928 vs 444.056 ms，0.957× | K1 IME 245.440 vs 174.758 ms，0.712× | SG 接近；IME prefill 仍缺 reuse/pipeline |

K1 RVV Q4_K 的 14.911 ms decode 和 1,912.203 ms prefill 仍保留在 CSV，但 K1 baseline 对应项是
IME1 asm，因此不计算 RVV 对 RVV 比率。它们的意义是观察同一 DSL primitive 的另一种合法 target
realization，而不是形成错误的速度结论。

### 6. Forward 与 state/memory 的代表结果

| case | SG2044 相对速度 | K1 相对速度 | 判断 |
|---|---:|---:|---|
| norm | 3.551× | 1.385× | reduction/state placement有效 |
| rms_norm | 2.632× | 1.536× | 两台机器均领先 |
| sum_rows | 6.852× | 1.068× | SG vector carry收益明显，K1接近 baseline |
| softmax | 0.918× | 1.018× | 接近区间，不作小差距胜负判断 |
| flash_attn | 0.609× | 0.584× | summary state、局部复用与 memory schedule仍不足 |

`cont`、indexed/gather 与部分 codebook 也继续暴露“能够选择合法 memory form”不等于拥有成熟的
coordinate hoist、window reuse 和 prefetch schedule。第四轮没有为这些单项加入 special route。

---

## 四轮后真正保留的编译主干

当前 production 信息流为：

```text
Core-local blocked DSL kernel
        ↓
canonical Kernel IR
        ↓
RISCVKernelFacts
  axis identity / ordered parent
  value consumers / definition / last use / control carry
  pointer root / index / predicate / lane relation / effect
        ↓
唯一语义与合法性推导
  free / reduction / broadcast axes
  memory form / validity / type conversion
        ↓
RISCVPhysicalPlanning
  structural candidates
  parameter instances
  target/resource filtering
  exact local leaf
        ↓
RISCVKernelCompiler transient physical plan
  value shape
  consumer/value handoff
  memory / state / dense / quant / fragment decisions
        ↓
ordinary C control and address emission
        ↓
RVV intrinsic spelling / quant helper / local IME asm
        ↓
system C compiler
```

`KernelCompiler::preparePhysicalDecisions()` 在生成函数体之前逐 operation 建立 decision，并拒绝同一
primitive 注册第二份决定；`collectSelectedLocalImplementations()` 随后收集 exact leaf。生成阶段
按 operation 类型读取 physical plan，缺 decision、shape、handoff 或 leaf 就报错，不重新选择。

## DSL 与 examples 实际改变了什么

这四轮没有改变 Core-local blocked 编程模型，也没有新建第二套 DSL：

- scalar `for/while/if` 仍是作者的 ordered traversal；
- `W.vla` 仍是显式 SIMD logical axis；
- `W.dot/W.matmul` 仍只授权局部 block product；
- workspace、staging、persistent layout 和 accumulator lifetime仍由作者拥有；
- LMUL、vector type、fragment、register group 和 exact leaf仍不进入 DSL。

唯一 example 变化是 F16 `blocked_gemm` 的 F32 activation → explicit F16 workspace staging。它证明
性能归因遵守了所有权边界：算法输入和跨 primitive workspace缺失时修改 DSL kernel；机器 shape、
register microkernel 和 target choice不足时修改 compiler。

其他 examples 文件没有变化但仍正确运行，是因为 runner 每次从 Python DSL 重新生成 Kernel IR，
再调用当前唯一 compiler。仓库不保存可绕过当前 lowering 的 generated kernel。

## 旧 authority 与 fallback 审计

最终 production compiler 未发现以下路径：

- kernel-name 或完整算子 route；
- q-format 字符串接管 whole-kernel lowering；
- VLEN128/VLEN256 作为 realization 身份；
- one-use、direct-store、exact-op-count 或完整外围 loop closure 作为高性能入口；
- emitter 根据 target、format 或 source closure 再选择一次 LMUL、microtile、fragment 或 leaf；
- legacy emitter、silent scalar fallback、GGML/materials runtime fallback。

Target profile 中仍会检查 VLEN、SEW/LMUL、寄存器数量和 IME1 对 VLEN256 的真实硬件约束。这些是
legality，不是目标名字分支。`selectLocalImplementationLeaf` 仍按 local primitive、structure 与 typed
parameters 映射 exact helper；它不拥有 outer traversal 或 kernel ABI。

`examples/run/weft.sh` 按命令行 kernel 名选择 DSL source、runtime 和 meta values，这是手工 repro
harness dispatch，不是 compiler implementation selector。部分量化 runtime 会链接 GGML helper 计算
expected value；实际 workload 始终调用 Weft generated entry，GGML reference不参与生成代码，也不
作为运行 fallback。

`source/` 不进入 Weft compiler 或 generated kernel 的 production path；部分 repro runtime会显式
引入其中的 GGML header/helper 计算 correctness reference。`materials/` 不进入 compiler 或 runtime。

## 尚未解决的问题

### 1. Dense candidate 与 cost 仍不成熟

F32/F16 已有 LMUL、microtile、K-unroll、row/column organization 和单/双 load buffer，但当前排序
仍主要是 resource/legal heuristic。K1 prefill 的 3—5× 时间差说明 cross-row/output reuse、真正的
microkernel blocking、load/compute overlap 与 target cost选择尚未成熟。

F16 显式 staging 关闭了算法输入不一致，却也使 staging cost真实进入数字。它不能再靠假定 F16
activation 规避，但 compiler 仍需要更好的 cast/store/matmul dataflow复用。

### 2. Quant candidate 宽度不均衡

Q4_K grouped affine register microkernel证明共享 local structure可以同时改善 production 与 vec-dot。
但 IQ1/IQ2/IQ3、MXFP4/NVFP4 等不同关系仍有明显分化；许多 primitive只有 register与strip、
32/64 semantic lanes或 RVV/IME 之间的有限选择，cross-output reuse、gather organization与pipeline
空间仍窄。

### 3. Flash attention 与复杂 memory/state 仍缺局部复用

Flash attention 在 SG/K1 都约为 baseline 的 0.6×。当前 compiler 能正确处理 VLA、predicate、
summary state与mixed dtype，但尚未形成足够好的 window reuse、state placement和load schedule。

### 4. KernelCompiler 仍然很大

文件职责已经分开，但 `RISCVKernelCompiler.cpp` 仍超过 11K 行，同时协调多类 operation 的 decision
materialization、C control flow与地址生成。当前关键边界是“所有机器选择发生在 emission 前”，而
不是声称每个 decide/emit 函数已经拆到最小文件。

### 5. 性能 CSV 是记录，不是统一测试框架

250 条结果来自真实模型级 shape与真机运行，但 repetition、warmup、correctness scope 和 preprocess
按 runtime实际语义变化。CSV 不附带 threshold、同步或校验逻辑；它用于保存数字和暴露数量级
问题，不构成另一套 compiler authority。

## 总判断

四轮真正完成的是：

1. 第一轮把 target facts、结构实现、参数和 exact leaf分成可解释的不同决定，并删除以 VLEN/lane
   命名的历史 realization；
2. 第二轮让 memory、dense pipeline、state、quant resource、RVV/IME fragment和 leaf selection各自
   只有一个 owner，并把 C ABI/RVV spelling从编译推理中分离；
3. 第三轮让 dense structure、value handoff、segment memory与资源真正由 axis/use/memory facts驱动，
   并证明 examples不需要为了后端闭包而改写；
4. 第四轮用最终 250 条 baseline realization 做两遍完整真机运行，修复 fractional LMUL、F16 workload
   所有权、Q4_K register microkernel和 target-aware selection，最终 250/250 通过。

因此当前最准确的结论是：

> **Weft 已经拥有一条形式上可解释、实现上唯一、能覆盖固定 GGML baseline 的 RISC-V kernel
> compiler 主干。它不再依赖 target/VLEN/family 身份或旧 emitter维持能力；但其高性能候选空间
> 仍不均衡，dense prefill、复杂 quant和attention仍需由共享物理实现继续缩小差距。**
