# Weft Core-local blocked 编程模型四轮推进报告

## 报告范围

这份报告记录从“建立完整的 Core-local blocked 编程模型”开始的连续四轮推进。它是一份一次性历史快照，不是第二份设计规范；当前语言与编译器规范仍以 [`doc/index.md`](../doc/index.md) 及其链接文档为准。

四轮对应的提交边界是：

| 轮次 | 提交 | 本轮实际主题 |
| --- | --- | --- |
| 第一轮 | `9b4383d86` | 建立 Core-local storage、lifetime 与 artifact 合同 |
| 第二轮 | `0fb7308f4` | 收紧 structured value、canonical verifier 与 RISC-V target contract |
| 第三轮 | `c231f95cd` | 将既有 kernel catalog 迁移到 canonical artifact ABI |
| 第四轮 | `444d6d84e` | 用七个事后选择的陌生 kernel 扩展共享 lowering |

这里把证据分成三种，不混在一起：

1. [`doc/`](../doc/index.md) 描述四轮后希望长期成立的正式合同；
2. 上述提交和当前源码说明实际进入了 DSL、Kernel IR、Realizer、emitter 与 artifact 的内容；
3. [`weft-kernel-performance.csv`](weft-kernel-performance.csv) 只记录已经进行过的真机正确性和性能执行数字。

提交标题中的 `build`、`close`、`migrate` 和 `generalize` 不是成熟度证明。尤其“structured value contract 已关闭”不等于任意 structured result 都已经可以像 Triton block 一样自由组合；本报告会把已经实现和仍然欠缺的部分分别列出。

## 结论先行

这四轮完成的核心变化不是多支持了七个 case，而是把 Weft 的根模型从“一个 worker-local callable，加上一批彼此不完全统一的 VLA、block、quant 和 IME fast path”推进为：

```text
caller/runtime 把一段工作和明确 storage 交给一个 worker/hart
        ↓
worker 顺序执行一份持续存在的 Core-local blocked program
        ↓
普通 SSA value、block、state、workspace 可按作者写下的控制流存活和复用
        ↓
显式 VLA / dot / matmul / state / extension primitive 授权局部物理化
        ↓
一次 RISC-V lowering 产生 target-specific transient decisions
        ↓
intrinsic C / primitive-local asm + generated C header
        ↓
system compiler 生成 object/library，caller 按同一 ABI 调用
```

四轮后已经明确并实际落地的内容包括：

- kernel 的执行者是一个 worker/hart，不是 GPU grid 中的一次性 program instance；
- outer traversal、blocking、staging、persistent packing、workspace 生命周期和算法 variant 由作者拥有；
- `W.vla`、`W.dot/W.matmul`、state 和 extension op 只是局部授权点，不能接管完整 kernel；
- external、worker-local workspace、persistent packed storage 与 primitive-private temporary 不再被当成同一种 raw scratch；
- public DSL、canonical op/verifier、RISC-V capability 与 generated artifact 开始按同一能力边界收敛；
- RISC-V target facts 不再主要依赖散落的字符串判断，而有了 typed target profile；
- 七个新程序通过扩展 unary、index、predicate、indexed memory、workspace 和 local quant/IME facts 接入，没有新增 kernel-name route 或 whole-kernel emitter。

但四轮没有证明 Weft 已达到 Triton 级成熟度。当前最重要的未闭合项仍是：

- `dot/matmul result → 任意 pointwise / 多 consumer / state / memory` 的一般 SSA composition 仍未完整证明；
- 若干高性能 family 仍依赖比理想模型更窄的 producer/use/region envelope；
- persistent format 已能进入 ABI，但 canonical builder、兼容性与跨 realization 的通用 packed-object 产品合同仍不完整；
- physical candidate space 仍偏窄，尚没有成熟的 compile-and-measure 选择系统；
- 第四轮部分复杂 kernel 只做 sampled correctness，性能数字也没有形成与 GGML 同算法、同 shape、同 preprocessing ownership 的闭合 baseline 对比。

因此，四轮后最准确的定位是：

> Weft 已经把 Core-local blocked program 的关键语言、storage、target 与 artifact 边界做成了真实主链，并开始用陌生程序检验逐实体 lowering；它仍是正在形成完整 composition 与性能搜索空间的 RISC-V kernel compiler，而不是已经完成的“RISC-V Triton”。

## 四轮开始前的问题

四轮开始前的审计见 [`weft-programming-model-and-compiler-architecture-audit.md`](weft-programming-model-and-compiler-architecture-audit.md)。当时 Weft 已有真实的：

```text
Python DSL
→ canonical Kernel IR
→ RISC-V intrinsic C / local asm
→ system compiler
→ SG2044 / K1 执行
```

也已经明确不是图编译器：source 是完整 callable，没有 grid、program ID、隐式 hart ID、graph scheduler 或 GGML/materials fallback。

问题在于同一仓库中仍混合着三种没有完全接上的作者模型：

```text
C-like worker function
+ lexical VLA program
+ backend-shaped local block / quant / IME program
```

最直接的裂缝有六类。

### 1. Worker-local 只有“谁执行”，没有完整 storage/lifetime 合同

Attention、Conv、MoE 与 IME source 已经通过 pointer 参数使用 scratch 或 packed object，但 pointer 本身无法说明：

- 它是 public input/output、invocation-local workspace，还是跨调用 persistent object；
- shape、alias、alignment 与 lifetime；
- caller 应分配多少空间；
- generated artifact 如何向调用者公开这些要求；
- 哪些 temporary 只是 primitive 内部的 target 实现细节。

### 2. Block/structured primitive 还不是稳定的普通 SSA value

审计中的反例是：已有 `dot → store` 能 lowering，但只在 dot result 后加入一个合法逐元素 add，canonical IR 仍可生成，RISC-V lowering 却失败。说明 backend 仍把 structured result 当作某条 fast-path closure 的 terminal，而不是有明确 physical handoff 的普通 producer。

### 3. Public surface 大于正式 target 能力

`widen`、`atomic_add`、`fence`、`valid/fill`、`permute`、多种 block transform、lookup 和 matmul 的表面语义与当前 RISC-V artifact 并不一致。明确 unsupported 比 silent fallback 正确，但不能把“frontend 能写、target 明确失败”继续称作 public support。

### 4. Canonical verifier 没有完整守住语义边界

Storage、dynamic extent identity、validity、masked value、lookup/decode、structured init 和 sort workspace 等合同存在缺口。有些错误会晚到 resource selection 才间接失败，导致 canonical IR 还不是完全可信的唯一算法 authority。

### 5. Physical planning 与 emitter 的权责仍然交叉

已有 `PhysicalEntityPlan`、VLA decision、dot candidate 和资源约束，但若干 family 仍依赖 one-use、direct store、固定 producer 链、邻接关系或外围 loop closure。部分 shape、layout、decode grouping 与 fragment 选择仍贴在 emission 附近。

### 6. Artifact 仍过度依赖 benchmark harness

`weft-compile` 可以输出 Kernel IR 或 intrinsic C，但普通 caller 没有同源 generated header 来得知准确 C ABI、workspace 和 persistent storage metadata。很多 runtime 保存了手写 prototype，runner 的 catalog 事实上承担了产品接口的一部分。

四轮就是围绕这些裂缝推进，而不是重新发明 GPU SIMT、恢复一般 contraction，或者把更多整算子塞进 backend。

## 第一轮：建立完整的 Core-local blocked 编程模型

### 本轮要回答的问题

第一轮首先冻结一件事：Weft 用户写的不是“一个隐式映射到 CPU lane 的 tile”，而是一份由一个 worker/hart 持续执行的有序程序。

作者可以在普通 control flow 中：

- 顺序处理多个 cache/algorithm blocks；
- 让 accumulator、state、staging 和 workspace 跨 loop iteration 存活；
- 显式决定 blocking、packing、reuse 和 storage lifetime；
- 把 local structured primitive 的结果继续作为普通 SSA value 使用。

Target 只能在显式授权的局部 domain 内决定 VLA strip、LMUL、memory form、microtile、短生命周期 packing、RVV/IME realization 与 primitive-local pipeline。

### 1. 冻结了唯一根模型

[`doc/dsl/model.md`](../doc/dsl/model.md) 将 kernel 定义为一个普通 C ABI 可调用、由单个 worker/hart 执行的 persistent ordered program：

```text
entry pointer/scalar ABI
+ ordered scalar control
+ zero or one lexical VLA axis
+ logical block/state values
+ explicit storage lifetime
+ explicit local structured primitives
```

这里没有引入 grid、task identity、implicit hart identity，也没有把每个 block 变成一次独立 launch。一个 worker 可以在外层循环中连续拥有多个 block，并让 state 和 workspace 跨循环复用。

这也明确了与 Triton 的差异：Triton 通常围绕 grid 中 program/CTA 逻辑拥有的结果 tile 组织执行；Weft 围绕 CPU worker 的持续控制程序以及它持有的 block、state 和 storage lifetime 组织执行。两者都可以使用 block/tile，但 block 的执行合同和生命周期不同。

### 2. 将 storage ownership 做成 DSL 与 Kernel IR 合同

第一轮新增并冻结了三种 caller-visible storage 标注：

| Storage | Source 表达 | 所有者与生命周期 |
| --- | --- | --- |
| External/public | 默认 pointer | caller 持有的普通输入输出 buffer |
| Worker-local algorithmic workspace | `W.workspace, W.noalias` | caller 为本次 worker invocation 提供；作者可跨 loop/primitive 使用 |
| Persistent packed storage | `W.persistent(format)` | caller/model loader 提供，跨 kernel 调用复用，format identity 稳定 |

Entry body 中的 `W.storage(pointer, shape)` 为 workspace/persistent pointer 闭合逻辑 shape。Canonical pointer type 同时携带 storage class 和 persistent format identity。

Primitive-private temporary 被明确排除在 public DSL、Kernel IR ABI 与 artifact metadata 之外；它只存在于一次 target lowering 的 transient decision 中。

这不是让 compiler 自动发明 staging 或 repack。相反，它把 ownership 明确分开：

```text
algorithmic workspace / persistent layout → 作者和 caller 的合同
primitive-local register pack / temp      → target 的实现细节
```

### 3. 给 storage 增加最早的 canonical 限制

第一轮已经建立基本限制：

- `W.storage` 只能直接出现在 entry body；
- workspace 和 persistent pointer 必须分别有且只有一条 storage declaration；
- external pointer 不能伪装成显式 storage object；
- workspace 必须是 `noalias`；
- persistent storage 必须有非空 format identity。

更完整的 shape、extent 与 entry ABI verifier 在第二轮继续补齐。

### 4. 收回没有真实 target artifact 的 public 构造

为了让“公开能力 = canonical schema = 正式 target artifact”成为可执行原则，第一轮没有给不成熟能力增加 fallback，而是从 public surface 收回：

- `prefetch`；
- `atomic_add`；
- `fence`；
- `valid`；
- `fill`；
- `permute`；
- `widen`。

Helper effect 也收缩到实际闭合的 read/write。未实现能力不再以 public builtin 的形式假装存在。

### 5. 生成正式 C artifact header

第一轮新增 `RISCVArtifacts.cpp` 的 header emitter，并给 `weft-compile` 增加 `--header`：

```text
canonical Kernel IR
→ intrinsic C
+ generated C header
```

Header 不只是手写函数声明的替代物，还公开：

- entry C ABI；
- pointer storage kind；
- persistent format identity；
- workspace/persistent rank 与 extents；
- caller 分配所需的 element-count 查询。

`--emit=intrinsic-c` 要求 C 与 header 使用不同输出路径，避免 artifact 相互覆盖。Runner 随后开始让 runtime include generated header，而不是为每个 kernel 维护另一份 prototype authority。

### 6. 迁移了已有显式 workspace/persistent source

本轮机械更新了 attention、dense/transpose convolution、selection、MoE 和两个 IME projection 等 source，使原本只靠 raw pointer 私约定的 workspace/persistent operand进入正式 annotation 与 `W.storage` 合同。

这些修改没有改变它们的 traversal、blocking、state、staging 或 persistent layout；只是把作者已经拥有的信息变成 canonical source/ABI 事实。

### 第一轮实际关闭和没有关闭的内容

第一轮真正关闭的是：

- worker/hart-local persistent ordered program 成为唯一根模型；
- caller-visible storage category、shape declaration 与 artifact metadata 有了正式表达；
- 未闭合 public surface 被主动收缩；
- generated C/header 成为同一编译动作的 artifact 边界。

第一轮没有关闭：

- structured result 的一般多 consumer/pointwise/state composition；
- persistent object 的通用 builder 与 target compatibility；
- 所有 validity/extent/lookup/decode verifier；
- family-specific exact closure；
- 成熟的 physical candidate/tuning space。

该提交没有改写性能 CSV，也没有保存一组新的目标机 runtime 日志，因此不能把第一轮描述成“新模型已对全部 kernel 完成真机重测”。它首先完成的是语言、IR 与 artifact 合同。

## 第二轮：让编译器兑现已冻结的模型

### 本轮要回答的问题

第二轮不再保留“surface 先存在、以后 target 再补”的双层事实。目标是让公开 DSL、canonical verifier、target profile 和真实 emission 对同一能力给出一致答案，并让错误尽量在最早的语义边界失败。

### 1. 补全 kernel 与 storage verifier

`KernelOp::verify` 开始统一检查：

- entry 参数 metadata、kind、name 与 return contract；
- persistent/workspace 参数是否各自恰有一条 storage declaration；
- external pointer 是否错误地进入 `W.storage`。

`StorageOp::verify` 进一步检查：

- storage 位于 entry block；
- storage class 不是 external；
- shape rank 与 extents 合法；
- static shape/extent 一致。

因此，workspace/persistent 不再只是 Python annotation，而成为 canonical IR 可以独立拒绝错误输入的合同。

### 2. 闭合 extent、validity 与 masked memory 的关键语义

本轮把动态 extent identity 继续带入 `for`、`while` 与 logical value；`full` 明确 shape；`invalid` 被限制在允许的 sentinel 位置。

Memory 方面：

- load 必须产生与 pointer/predicate 一致的 validity；
- store 不能直接接受尚未消解的 masked value；
- dynamic extent 的比较不再只看一个模糊 `-1` shape；
- canonical error 更早落在 value/memory boundary，而不是 resource selector。

### 3. 将排序 workspace 从 backend 私有假设变成作者合同

`sort_indices` 改为显式消费 caller-provided、`noalias` 的 `u32` workspace。Canonical verifier 要求该 storage 是 rank-1，并且 extent 与 sort domain 一致；对应 runtime 负责分配和传入。

这体现了 Core-local blocked 模型的边界：排序所需、算法可观察且跨 primitive 使用的 workspace 属于 source/artifact；leaf 内部临时寄存器才属于 target。

### 4. 继续收缩没有统一语义的 block transform

第二轮删除了 public/canonical：

- `broadcast_to`；
- `reshape`；
- `transpose`。

保留的 `expand_dims` 只允许 singleton-axis 语义。这里不是追求“原语越少越好”，而是避免把只能被特定 closure 吸收、没有独立一般 artifact 的操作继续宣传成普通 block transformation。

### 5. 把 structured compute 的真实 envelope 写进 canonical contract

本轮没有恢复 arbitrary-axis contraction，而是进一步收紧当前真实实现：

- `dot`：F32 operands 与 F32 accumulator；
- `matmul`：F16 operands 与 F32 accumulator；
- init 必须显式且不能携带未定义 mask；
- lookup/decode 的 table、code、dtype 与 block extent 收紧到 target 真正拥有的 local semantic envelope。

这一步的意义不是“形态越固定越高性能”，而是停止让 frontend 承诺 target 还没有定义的任意形态。

### 6. 建立 typed RISC-V target profile

原先 target legality 中散落着 march 字符串、VLEN 与 extension 判断。第二轮把它们收敛进 `RISCVTargetProfile`，正式承载：

- RV32/RV64 与 ABI/XLEN；
- full V 与 Zve 类能力关系；
- F16/F32 element capability；
- fixed VLEN；
- memory/widening facts；
- vector register/resource facts；
- matrix extension identity 与 legality。

IME1 明确要求 RV64 与 VLEN256，而不是由 emitter 临时看到一个字符串就选 asm。CLI 解析 target facts 后将 typed profile 传给同一 lowering。

### 7. 把 decision preparation 前移到 emission 之前

第二轮继续强化：

```text
canonical primitive + typed operands + axes + use relation + target facts
        ↓
prepare physical decisions
        ↓
intrinsic/asm spelling
```

未知 op 直接报 unsupported，不进入 fallback。Scan、argmax 和 online summary 等现有能力仍通过它们明确的 enclosing VLA owner 发射；region dot、F16 GEMM 等形态仍有窄 envelope，不能据此声称一般 composition 已经完成。

### 第二轮的真实证据边界

该提交修改了四条 SG2044 性能记录，但没有保存完整的新一轮远程 stdout/raw samples，也没有增加新 kernel 或新硬件覆盖。因此它证明的是 canonical/target contract 进入了既有真实路径，不是一次系统性能结论。

第二轮之后仍存在的关键问题包括：

- local/region dot 仍没有一般普通 consumer lowering；
- nested VLA 仍明确不支持；
- 某些 state/summary 只在 enclosing VLA 的现有实现域内闭合；
- typed target profile 已建立，但 candidate family 与 emitter authority 仍需继续扩展。

## 第三轮：迁移并验证既有 kernel catalog

### 本轮要回答的问题

前两轮改变了 source annotation、generated header 与 canonical ABI。第三轮不增加算法，而是检查现有 catalog 是否仍偷偷依赖手写 prototype、旧 quant ABI 或 source spelling 私约定。

### 1. 删除 quant runtime 的第二份函数声明 authority

旧 quant repro 的公共 header 保存了一组手写 typed declarations。第三轮删除这些 declarations，让 Q4_0、Q4_1、Q4_K、Q5_0、Q5_1、Q8_0 等 runtime 直接消费 generated header 中的 canonical byte-pointer ABI。

这样 caller ABI 的唯一来源变成：

```text
Kernel IR entry signature + storage metadata
→ generated header
→ runtime compile
```

而不是 Kernel IR 一份、手写 quant header 再维护一份。

### 2. 统一 persistent format identity

两个 IME runtime 改用 canonical persistent format 名称：

- `q4_0_n16_k32_288b`；
- `q4_k_n16_k32_304b`。

这些名字描述 caller/model loader 提供的 packed representation；IME leaf 仍只拥有局部 fragment realization，不能创建 persistent object 或接管 outer traversal。

### 3. 让 runner 对 C、header、caller 和 archive 使用同一 ABI

Runner 为每个 source 生成：

```text
kernel.c
kernel.h
```

随后 kernel 与 caller 均以同一 generated header 编译。等价写法/多实现 repro 会分别生成两套 C/header/object，再打入 archive，而不是靠共享手写 declaration 假定二者 ABI 相同。

Quant C caller 补上 `_POSIX_C_SOURCE=200809L`，使其时间与分配接口在目标编译器下有明确声明。这是 artifact 调用修复，不是新的 compiler stage。

### 4. 修复 benign identity cast 对 physical ownership 的破坏

第三轮在 Realizer 中加入 `RVVIdentity` cast/share handoff，并让 dot/matmul/F16 GEMM 的 producer tracing 可以穿过不改变数值和 shape 的 identity cast。

它解决的是：同一个 typed value 因 ABI/source 迁移插入一个无害 cast 后，不应丢失既有 physical owner。该修改仍然是一个局部 handoff 改进，不等于 arbitrary `dot → add → store` 已经闭合。

### 5. catalog 与性能记录的实际变化

第三轮前后性能 CSV 都是：

| 项目 | 数量 |
| --- | ---: |
| 性能记录 | 134 |
| Distinct kernel | 64 |
| SG2044 / RVV VLEN128 | 66 |
| K1/X60 / RVV VLEN256 | 66 |
| K1/X60 / IME1 | 2 |

本轮重写/重测了已有数字表，但没有增加 134 个 kernel，也没有增加新算法；134 是 hardware/target/phase 记录数。仓库没有保存这次远程执行的逐次 raw stdout，因此报告只能把 CSV 视为当时的数值快照，不能反推每个 sample 的完整运行日志。

既有 runner 中的 segmented scan、Top-K、SSM conv、ROI Align、AdamW、weighted embedding bag、interleaved complex 和 lookup 等 equivalent spelling 路径继续通过 canonical artifact ABI。它们提供部分“无关 source spelling 不应改变能力类别”的证据，但仍不能代替 structured result 多 consumer、跨 region state 或 packed storage ownership 的一般证明。

### 第三轮的真实结论

第三轮关闭的是“已有 catalog 仍由手写 caller ABI 驱动”的问题，并证明新 storage/header 合同可以承载原有 64 个 distinct kernel 的记录。

它没有：

- 新增算法能力；
- 建立通用 persistent builder；
- 证明任意 block/structured result composition；
- 扩大 IME 到任意 matmul；
- 建立成熟 compile-and-measure tuner。

## 第四轮：用陌生 kernel 反证过拟合

### 选题方式

第四轮先确定七个真实、自然的程序，再接 compiler；没有根据当前 matcher 能做什么反向换题。

| Kernel | 自然算法结构 | 主要压力点 |
| --- | --- | --- |
| `rms_norm_mul_f32` | RMS reduction 后乘独立 per-element multiplier | reduction result、普通 pointwise、多输入 VLA 与等价书写 |
| `timestep_embedding_f32` | log/exp frequency、sin/cos 双输出与 odd tail | scalar/VLA 数值转换、transcendental leaf、尾部 predicate |
| `qwen3vl_mrope_f32` | 四路 position plane、angle cache 与 head-local rotation | workspace、跨 loop staging、VLA trig 与多源索引 |
| `col2im_1d_f32` | output-owned VLA + scalar tap loop + predicate + indexed load | control+VLA、native index arithmetic、masked irregular memory |
| `conv3d_f32` | source-owned im2col/weight staging + local dot | unfamiliar outer traversal、workspace lifetime、contract leaf复用 |
| `flash_attn_ext_f32_f16` | F32/F16 typed attention、GQA、mask、softcap、sink 与 online state | state、mixed dtype、workspace、nontrivial producer-consumer context |
| `q4_k_mul_mat_id` | MoE grouping、activation quantization、persistent Q4_K、RVV/IME fragment | quant+indexed selection+workspace+persistent storage+extension realization |

这些 source 位于 `examples/kernels/`，各自有独立真实规模 runtime，算法没有为 backend 改写成统一模板。RMS Norm Mul 和 Col2Im 另有自然等价书写，用来检验 decision 是否依赖无关临时 SSA 或 index 表达。

### 接入过程暴露的实际缺口

七个程序第一次进入主链时暴露的不是七个 kernel recognizer 缺失，而是四类共享事实缺口：

1. `tanh` 没有完整 frontend/canonical/target realization；
2. VLA logical index 不能自然转换到 F32 数值计算；
3. typed index arithmetic、predicate、select、nested pointer decomposition 与 indexed load 不能在同一 value/use 模型中组合；
4. 某些生成的 `if` 有未被消费的结果槽，emitter 仍假定每个结构结果都必须物化。

这些是当轮交互诊断中实际遇到并修复的问题，但失败的 raw stdout 没有作为仓库 artifact 保存。因此它们在这里用于解释实现动机，不作为可重放日志引用。

### 1. 完成 `tanh` 和 VLA unary realization

第四轮补齐 `tanh` 的 Python builtin、canonical op/verifier 和 RISC-V lowering。VLA unary decision family 现在覆盖：

- `exp`；
- `tanh`；
- `sin`；
- `cos`。

`sin/cos` 当前通过一个明确的 primitive-local RVV↔scalar libm temporary realization完成；它属于显式 unary primitive 的 local leaf，不是遇到不支持就静默把完整 kernel 换成 scalar backend。该实现仍然很窄，也没有资格被描述成成熟 RVV transcendental library。

### 2. 打通 index value 到普通数值和 predicate 的组合

本轮新增/修复：

- logical index → F32 conversion；
- index-vector arithmetic；
- typed compare 与 select；
- nested pointer decomposition；
- 32-bit/64-bit indexed load；
- 对应 value shape、mask/index relation、temporary 与 register resource 计算。

因此，Col2Im、MRoPE 和 timestep embedding 不需要把自然 index 关系提前改写成 emitter 喜欢的固定 op chain。

### 3. 修正共享 resource 与 structured emission 问题

LMUL 计算改为按 data LMUL、XLEN 与 data SEW 的关系形成 index/address shape，而不是沿用对 VLEN128 偶然成立的固定比例。

`if` emitter 只物化真正有 consumer 的 result slot，避免 emitter 根据结构结果个数重新推断 value ownership。

Q4 activation block 的 group/block division 也在共享 quant lowering 中修正；修复位置是 local quant primitive 的 typed relation，不是 `q4_k_mul_mat_id` symbol 分支。

### 4. 保持 local primitive 与算法结构的所有权边界

七个 kernel 中：

- MRoPE 的 angle cache 是 source-owned workspace；
- Conv3D 的 patch/weight packing 是 source-owned staging；
- Flash Attention 的 query/accumulator scratch 与 online state 是 source-owned；
- Q4_K MoE grouping、activation quantization、outer expert traversal 和 persistent 304-byte layout 是 source/caller-owned；
- RVV/IME 只实现当前 local N16×K32 Q4_K fragment。

Backend 没有把这些结构从普通 SSA 猜出来，也没有用 local leaf 接管完整 ABI、outer loop 或 persistent layout。

### 5. 没有重新长出整 kernel route

`RISCVLowering.cpp` 中没有增加这七个 kernel symbol 的分支；生成实现也没有调用 GGML 或 `materials/` runtime。Q4 source 会复用 DSL quant helper，但 target 仍只看 canonical local primitive 与 typed operands。

新增能力分别落在：

```text
frontend/canonical local semantic op
typed value/index/predicate facts
physical handoff/resource decision
intrinsic C / primitive-local leaf spelling
```

而不是：

```text
if kernel == conv3d / flash / q4_k_mul_mat_id
    emit whole implementation
```

### 6. 七个 kernel 的真机结果

第四轮新增 15 条记录：七个 kernel 均在 SG2044/RVV VLEN128 与 K1/RVV VLEN256 执行，其中 Q4_K MoE 另在 K1/IME1 执行。

| Kernel | SG2044 / RVV128 | K1/X60 / RVV256 | K1/X60 / IME1 | Correctness scope |
| --- | ---: | ---: | ---: | --- |
| RMS Norm Mul | 1.029204 ms / 509.411157 MElements/s | 1.522383 ms / 344.386399 MElements/s | — | full；等价写法同类 realization |
| Timestep Embedding | 4.690820 ms / 34.955082 MElements/s | 7.638025 ms / 21.467330 MElements/s | — | full；最大绝对误差 `2.23815441e-05` |
| Qwen3-VL MRoPE | 2.322691 ms / 1.805795 GB/s | 3.548630 ms / 1.181950 GB/s | — | full |
| Col2Im 1D | 21.625475 ms / 24.232161 MElements/s | 98.418075 ms / 5.324550 MElements/s | — | full；最大绝对误差 0；等价写法同类 realization |
| Conv3D | 135.651734 ms / 6.678644 GOP/s | 550.133043 ms / 1.646819 GOP/s | — | sampled；最大绝对误差 0 |
| Flash Attention Ext | 397.949080 ms / 2.698189 GOP/s | 707.640981 ms / 1.517354 GOP/s | — | sampled；误差为浮点舍入量级 |
| Q4_K Mul Mat ID | 2654.656934 ms / 1.617899 GOP/s | 5256.822103 ms / 0.817027 GOP/s | 518.722195 ms / 8.279899 GOP/s | activation full + output sampled |

Q4_K Mul Mat ID 的 activation code mismatch 为 0，activation scale 最大绝对误差为 0；output 最大绝对误差为 `1.90734863e-06`，RVV/IME 最大相对误差分别为 `9.53674316e-07` 和 `1.56462193e-06`。

在同一 K1、同一 kernel/shape/scope 下，IME throughput 相对 RVV 为约 `10.13×`。这证明该明确 Q4_K local fragment 的 IME realization有效；它不是 GGML baseline speedup，也不能外推为任意 quant matmul 的 IME 支持。

### 第四轮后的性能表构成

第四轮前后数字表变化为：

| 项目 | 第四轮前 | 第四轮后 |
| --- | ---: | ---: |
| 性能记录 | 134 | 149 |
| Distinct kernel | 64 | 71 |
| SG2044 / RVV VLEN128 | 66 | 73 |
| K1/X60 / RVV VLEN256 | 66 | 73 |
| K1/X60 / IME1 | 2 | 3 |

这些是 kernel/phase/hardware/target 记录，不是统一实验总体。不同记录的 correctness scope、repetition、shape、throughput unit 和 preprocessing ownership并不完全相同，不能把 149 行横向混成一个整体性能结论。

## 四轮合起来改变了什么

### DSL 层

四轮后，作者写的是一份 Core-local blocked program，而不是一组待识别的 operator graph：

- entry 是 caller 可调用的 worker-local ABI；
- scalar control 是有序 traversal，不会被 target 偷偷改成 VLA；
- `W.vla` 显式授权一个 lexical SIMD logical axis；
- block/state 是同一 value system 的 shaped SSA value；
- `W.dot/W.matmul` 只授权当前 local product；
- workspace/persistent storage 由作者标注并声明 shape；
- outer blocking、staging、packing、state algorithm 和 algorithm variant 始终属于 source。

同时，public surface 通过删除不真实能力和收紧 structured op envelope，开始与当前唯一 target artifact 保持一致。

### Canonical Kernel IR 层

Canonical IR 增加或强化了：

- pointer storage class 与 persistent format；
- `W.storage` 对应的 registered canonical op；
- kernel parameter/storage verifier；
- extent identity、validity 与 masked memory contract；
- sort workspace contract；
- unary/index/predicate/indexed-memory typed relation；
- structured compute 的当前真实 dtype/shape envelope。

长期 authority 仍只有这份 Kernel IR。LMUL、microtile、fragment、candidate、resource budget 与 intrinsic spelling没有被物化成第二份持久 IR。

### RISC-V Realizer 层

四轮主要把 target 输入从“一个大致相符的完整 closure”继续推向：

```text
primitive identity
+ typed operands/results
+ logical axes and extents
+ validity/predicate
+ ordinary use relation
+ memory/storage relation
+ state facts
+ typed RISC-V target facts
```

然后在 emission 前产生 value shape、LMUL、memory form、handoff、temporary、local primitive realization 与资源决定。

实际新增的共享能力包括 storage-aware artifact、typed target profile、identity/share handoff、VLA unary、index→numeric conversion、native index-vector/predicate/select、nested pointer decomposition、indexed load 和 local Q4_K RVV/IME realization修复。

### Emitter 与 artifact 层

Emitter 的正式职责进一步收敛到：

- RVV intrinsic C 拼写；
- typed primitive-local asm；
- C ABI；
- generated header metadata。

Runner 不再用手写 prototype 维持第二份 ABI，kernel C、header、caller object 和 archive均来自同一次 canonical compile contract。

这仍不是一个完整 package/install 产品：system compiler/object/archive 驱动仍主要由当前 runner 完成，外部 frontend 的稳定 construction API 也仍以 canonical textual IR 为主。

## 从开始审计到四轮结束的问题关闭表

| 开始时的问题 | 四轮后的状态 | 准确判断 |
| --- | --- | --- |
| Worker-local 根模型含糊 | 已冻结为 persistent ordered Core-local blocked program | 已关闭设计歧义，并进入 source/IR/artifact |
| Workspace 只是 raw pointer 私约定 | workspace/persistent annotations + `W.storage` + generated metadata | 主要合同已关闭；alignment/builder/更丰富 descriptor 仍有限 |
| Public DSL 大于 target | 删除一批假能力，收紧 transform/dot/matmul/lookup/decode envelope | 大幅收敛；以后每个新增 op 仍需三层同时闭合 |
| Canonical verifier 太弱 | kernel/storage/extent/validity/sort/structured envelope 得到加强 | 关键问题已修；不能外推为所有组合均完整 |
| Target profile 是字符串启发式 | typed ISA/ABI/XLEN/VLEN/element/resource/IME facts | 主体关闭 |
| Caller ABI 有手写第二份 authority | generated C header 被 catalog runtime统一消费 | 当前 catalog 已迁移 |
| Benign cast 破坏 fast path | identity/share handoff 与 producer tracing | 该类无关 spelling 已修 |
| `dot → 普通 consumer` 失败 | 只修复部分 handoff；未展示一般 dot/matmul composition 的完整真机证据 | 仍未关闭 |
| Exact source closure | 七个陌生 kernel 通过共享 unary/index/memory/storage facts接入 | 明显减弱，但若干旧 family 仍有窄 envelope |
| Persistent packed ABI 私有 | format identity、shape 和 header metadata已进入合同 | builder、target compatibility 与通用跨 realization合同仍未关闭 |
| Candidate space 固定 | typed target/resource facts更完整 | 仍没有成熟 microtile/pipeline/prefetch/tuning space |
| 只能靠既有 catalog 证明能力 | 七个接入前固定的新 kernel，含两组自然等价表达 | 得到更强外推证据，但规模仍不足以证明普遍泛化 |

## 这四轮没有做什么

为了避免把实现范围写大，下面这些不属于四轮成果：

- 没有引入 GPU grid、SIMT task model 或 graph-level operator scheduler；
- 没有恢复一般化 `contract(lhs_axes, rhs_axes, output_order)`；
- 没有增加 normalization pass 去重写作者 blocking、staging 或 state；
- 没有把 ordinary SSA multiply/add 猜成 dot、softmax、quant 或 IME；
- 没有调用 GGML 或 `materials/` 旧函数作为 production fallback；
- 没有为七个新 kernel 增加 symbol/name route 或 whole-kernel microkernel；
- 没有证明所有 structured values 已能任意组合；
- 没有证明 149 条记录具有统一 full correctness、统一 warmup 或统一 preprocessing scope；
- 没有完成与 `source/` GGML baseline 的全量公平性能闭合；
- 没有完成 Triton 式成熟 autotuning 或通用 IME matmul backend。

## 最终状态

四轮真正建立的是一条更可信的编程模型闭环：

```text
作者显式写一份 worker/hart-local persistent control program
        ↓
block、state、workspace 与 persistent storage 拥有明确语义和生命周期
        ↓
局部 primitive 只授权自己的机器重组域
        ↓
canonical Kernel IR 成为唯一长期算法 authority
        ↓
typed target facts 和普通 use/storage relation 产生 transient physical decisions
        ↓
emitter 机械生成 intrinsic C / local asm 与同源 header
        ↓
caller 通过正式 ABI 在 SG2044/K1 上执行
```

第一轮让这个模型有了 storage 和 artifact 的骨架；第二轮让 canonical 与 target 边界更诚实；第三轮让已有 catalog 真正消费新 ABI；第四轮则第一次在这套模型冻结后，用七个陌生程序逼出共享 unary、index、predicate、indexed memory、workspace 和 quant/IME lowering。

它已经比四轮前更接近真正的 RISC-V kernel compiler，原因不是“71 个 kernel 能跑”，而是越来越多能力可以指出：由哪个显式 DSL 构造授权、由哪些 typed facts决定、由哪个 physical decision 唯一拥有、最后由哪个 local leaf 拼写。

它距离 Triton 级别仍最缺两件承重能力：structured value 的普遍 composition，以及受资源约束且能通过构建期实测选择的宽 physical candidate space。在这两项真正闭合前，Weft 应被描述为已经建立 Core-local blocked 主模型并具备初步外推能力的编译器，而不是完成态系统。
