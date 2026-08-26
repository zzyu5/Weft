# Weft 从 Core DSL 重构到双目标全量回归的完整自查

日期：2026-08-26  
审计代码状态：`5068a1dcd Complete dual-target RISC-V regression`  
审计起点：用户提出“现在开始真正的 DSL 阶段的重构……完整修改 examples 中的 DSL”之后的全部推进  
审计对象：当前源码、MLIR dialect、pass 主链、生成与运行脚本、现行设计文档、当前性能 CSV、对应 Git 提交和当轮真实运行记录

## 1. 这份报告回答什么

本报告不把“代码已提交”“构建通过”“有一个 pass”“能打印 intrinsic C”当作编译器成熟的证据。它回答五个可从仓库外部检查的问题：

1. 从上述 DSL 重构请求开始，后续每一轮实际改了什么、运行到了哪里、没有证明什么；
2. Weft 现在是否已经形成一套明确的编程模型，以及 DSL 是否真正按这套模型工作；
3. Weft 是否已经拥有真正的 Canonical Kernel IR 和 target-aware Physical IR，还是仍然在程序旁边维护 decision 表；
4. physical pass 是否真的改写程序、覆盖哪些结构、在哪些输入上仍退化为窄 matcher 或固定投影；
5. 为什么 206 条 case 全部数值正确之后，仍有 128 条低于 baseline 的 50%。

证据采用以下优先级：

```text
当前源码和当前 IR 定义
> 当前可手动执行的 target repro 与当前 CSV
> 对应提交的 diff
> 当轮一次性报告和对话结论
```

历史报告只用于恢复当轮证据边界，不能覆盖当前源码事实。性能数字只使用当前的
[`weft-kernel-performance.csv`](weft-kernel-performance.csv) 和固定的
[`baseline/ggml-riscv-kernel-performance.csv`](baseline/ggml-riscv-kernel-performance.csv)。

## 2. 总结论

### 2.1 直接结论

| 问题 | 结论 | 外部证据 |
|---|---|---|
| 是否已有明确编程模型 | **有** | 现行模型把作者程序定义为 encoding-aware、层级化、有限位宽的数值 realization，并用“改变 canonical value/effect graph 或 Level/artifact 归属”划分作者与编译器职责 |
| 是否已有非 SIMT 物理抽象机器 | **有正式定义，也有一部分可执行实现** | logical coordinates 被分解到 time、SIMD lane、register replica、fragment、local storage；这些分量已进入 RISC-V types/ops，但异步流水、fragment spill、通用 local-storage ordering 等仍未闭合 |
| Canonical Kernel IR 是否是真 IR | **是** | `weft_kernel` 是注册的 MLIR dialect，使用 typed SSA、regions、block arguments、results 和 verifier；不是 Python 字典或 emitter 旁表 |
| RISC-V Physical IR 是否是真 IR | **是** | layout 在 physical value type 中，conversion/spill/reload/window/IME 是真实 SSA op；pass 输入输出都是 program；旧 `problem/assignment` 主干已删除 |
| 是否已经达到 Triton 级 IR 机制 | **核心机制已经同类，成熟度没有达到** | typed layout、显式 conversion、program-to-program pass 已成立；但 Weft 的 conversion algebra、pipeline、resource allocation、target specialization 和 op coverage 远窄于 TTGIR |
| emitter 是否只做机械拼写 | **没有完全做到** | 它主要消费 typed leaf，但仍重建 encoding/address geometry、解析普通字符串属性和 `auto:`，并在若干复合 leaf 内组织 load/cache/投影逻辑 |
| 是否已经是高性能通用算子编译器 | **尚不能这样定性** | 206/206 数值正确，但中位性能比为 0.264，128/206 低于 baseline 的 50%；多数 K/IQ/TQ tree 没有 baseline 的 blocked/reuse 结构，F16 即使有 blocked tree 仍严重落后 |

### 2.2 当前最准确的项目定位

Weft 已经不再是“几段 emitter 外面套一个 DSL”，因为：

- 作者程序进入真实 Canonical Kernel IR；
- target lowering 产生真实 typed RISC-V IR；
- layout conversion、spill/reload、window、fragment 和 pipeline rewrite 已经是 IR 实体；
- 206 条双目标 case 确实走统一主链并在真机得到正确结果。

但它也还不是成熟的高性能算子编译器，因为：

- 大量 std 实现仍是逐 row、逐 column 调用 vec-dot，canonical tree 本身没有输出 blocking 和跨输出复用；
- pass 对 conversion、reuse、pipeline、resource 和 target-specific structure 的推导范围很窄；
- K1 production MUL_MAT 没有接入已经能独立运行的 IME leaf；
- emitter 与最终 leaf 的边界仍有泄漏；
- 当前性能只有 50/206 超过 baseline，不能用“至少十条超过 baseline”的门槛替代总体判断。

因此当前状态应定性为：

> **Weft 已建立真实的两层 MLIR 编译器骨架和一套明确的非 SIMT 数值编程模型；它已证明较宽的正确性覆盖，但高性能 physicalization 仍只在部分作者树、部分 representation 和少量 leaf 上成熟。**

## 3. 从指定起点开始的逐轮推进

这里按实际用户轮次恢复，而不是按提交数量重新包装。指定起点之后共有六个阶段：三轮结构重构、一轮自查 prompt、十项 baseline 闭合、一次双目标全量回归。

### 3.1 第一轮：examples 与 std 切换到新的 Core DSL

对应提交：`85eb68714 refactor: switch examples to core DSL`  
时间：2026-08-25 01:25:52 +0800  
规模：31 个文件，463 行新增，1055 行删除

#### 实际改动

- 全部当时 examples/std 去掉 source-level `@wide/@matrix/@transfer`；
- 去掉 invocation-local `pack(along=...)`；
- 去掉 source `stage_handoff`，保留 Level handoff 和普通 SSA；
- Q4_K 派生编码改为参数化 `Q4K_I[rows]`，persistent ABI 使用 `Q4K_I[16]`；
- dense、attention、quant、row-dequant、vec-dot、MUL_MAT 的 std 函数改成当前 Level/Value/Encoding 语言；
- 同时删除一批仍围绕旧 source permission/pack 语义的 target 代码。

这轮不是简单替换名字。作者可观察的跨调用重排被放进 derived Encoding；invocation-local physical pack 不再由 DSL 指定；scalar/RVV/IME engine 也不再成为 source operation 身份。

#### 当轮外部结果

- SG2044 `q8_0_quantize` 完成 `DSL → IR → intrinsic C → 真机`，离散量化结果 bit-exact，记录为 388.30 MElements/s；
- 同一份 target-neutral Q4_K DSL 能分别生成 SG2044 RVV C 与 K1 IME C/asm；
- Dense/F16 blocked tree 能生成 canonical facts，但发射在 reduction Level carry 的 layout conversion 处失败。

#### 这轮没有证明什么

- Q4_K 的 K1 路径只到生成代码，没有 K1 真机性能；
- Dense/F16 的失败说明新 DSL 已把问题送到 physical representation 边界，但不说明后端已经能兑现模型；
- 没有全量 canonical 验证，也没有正式性能 CSV。

#### 自查判断

这轮完成了 source surface 的方向切换，但当时 Canonical IR 和 RISC-V lowering 仍带有旧结构。它证明“新语言可以写代表性程序”，没有证明“全仓只有一套完整语言和编译路径”。

### 3.2 第二轮：重建第一层 Canonical Kernel IR

对应提交：`8b305a0ea refactor: rebuild canonical kernel ir`  
时间：2026-08-25 03:16:02 +0800  
规模：20 个文件，1696 行新增，575 行删除

#### 实际改动

- 第一层统一为 `weft_kernel + arith + scf`；
- Encoding、View、Value、Slice、Domain、Point 变成正式 MLIR types；
- domain 的 extent、partition、multiplicity 和 parent/axis identity 进入 SSA/type/attribute 合同；
- Level 拥有 carried values/results、`births.state`、`births.staged`、body 和 handoff regions；
- 删除旧自定义 `for/if/while/yield/condition` Kernel op，让普通控制使用标准 `scf`；
- 删除旧 `mac_pairs` IR op、axis/engine/pack 路径；
- 将 `cast`、`widen`、`narrow`、lookup、reduce、dot/contract 等边界重新交给 canonical verifier。

当前实现的核心定义位于：

- [`include/Weft/Dialect/Kernel/IR/KernelOps.td`](../include/Weft/Dialect/Kernel/IR/KernelOps.td)；
- [`lib/Dialect/Kernel/IR/KernelDialect.cpp`](../lib/Dialect/Kernel/IR/KernelDialect.cpp)；
- [`python/weft/frontend/compiler.py`](../python/weft/frontend/compiler.py)。

#### 当轮外部结果

- 87 个现有 DSL kernel 全部完成 `DSL → Canonical IR → weft-opt`：87/87；
- SG2044 Q8_0 quantize 保持 bit-exact；
- SG2044 Q4_K GEMV 数值误差为 0，三次中位数 8.309 GOP/s。

#### 这轮没有证明什么

- 87/87 是 canonical lowering 与 verifier 覆盖，不是 87 个 target runtime；
- 8.309 GOP/s 是重构 repro，不是同协议全量性能结果，因此当轮没有更新性能 CSV；
- 当轮仍明确承认第二层是 `weft_riscv.problem/assignment` side record，physical IR 尚未成立。

#### 自查判断

这一轮之后，第一层已经是真正的 MLIR program IR。它不是“AST 加一组检查字段”：Level 是 region op，carried values 和 handoff 是 SSA，encoding/value/domain 由 types 与 verifier 约束。但 Python frontend 仍先构造自己的短生命周期 Python SSA，再渲染 textual MLIR，由 C++ parser 建立正式 MLIR。这与 Triton frontend 直接调用 MLIR builder 不同，不过 parser 之后的 Canonical IR 仍是真 IR。

### 3.3 第三轮：重建第二层 typed RISC-V Physical IR

对应提交：`ef85cf0e6 refactor: build typed riscv physical ir`  
时间：2026-08-25 17:55:01 +0800  
规模：47 个文件，13673 行新增，9335 行删除  
当轮报告：[`riscv-physical-ir-second-level-refactor.md`](riscv-physical-ir-second-level-refactor.md)

#### 实际改动

- 删除 `RISCVPlanningDialect`、`ConstructProblems`、`Constrain*`、`Solve`、旧 layout conversion resolution 和旧 schedule side-record 主干；
- 建立 `weft_riscv` dialect；
- layout 进入 `ValueType`/`FragmentType`；
- memory descriptor、local storage、window 和 fragment 成为正式 types；
- `convert_layout`、spill、reload、register materialization、IME pack/MMA/unpack 成为真实 SSA operations；
- pass 之间通过同一份 RISC-V program 传递结果；
- final verifier 拒绝残留 canonical op、未选 layout/access/leaf、未展开 schedule 和资源超限；
- intrinsic-C/asm 只接收已通过 final verification 的 physical module。

当前第二层的核心定义位于：

- [`include/Weft/Dialect/RISCV/IR/RISCVOps.td`](../include/Weft/Dialect/RISCV/IR/RISCVOps.td)；
- [`lib/Dialect/RISCV/IR/RISCVDialect.cpp`](../lib/Dialect/RISCV/IR/RISCVDialect.cpp)；
- [`lib/Target/RISCVCompiler.cpp`](../lib/Target/RISCVCompiler.cpp)。

#### 当轮外部结果

- 编译器完整构建；
- K1/X60 signed-i8 IME local contract 真机运行，`M=4,N=4,K=8`，数值 exact，三次中位数 0.542 µs；
- 可手动执行命令是 `./examples/run/weft-kernel.sh k1 ime_i8_contract 3`。

#### 当轮明确未支持

- dynamic encoded-dot；
- 通用 depth>2/multi-buffer/multi-carry cluster pipeline；
- fragment spill 与 cross-block spill；
- i4 IME 和其它 fragment family；
- production IME MUL_MAT。

#### 自查判断

这一轮解决了“有没有真正 physical IR”的结构问题，但只解决了必要载体。typed dialect 本身不会自动带来 layout propagation、coalescing、conversion elimination、reuse、pipeline 和 resource allocation 的横向能力。一个局部 IME contract 成功，只证明这一条 physical op 链闭合，不能证明其它 operation family 都已迁移成熟。

### 3.4 第四轮：完整自查 prompt，而不是代码完成轮

这一轮用户要求“给一个完整的自查 prompt”。交付是下一轮实施指令，没有代码提交。

当轮已经明确写出：

- Physical IR 的结构成立；
- old side-record 主干已删除；
- K1/IME 有一条真机链；
- ordinary RVV op、layout conversion、跨控制流 value、pipeline、spill、memory form 和双目标差异大多只经过构建或静态审计；
- 需要用反例检查 emitter 是否仍补决定、pass 是否只对单一路径有效。

因此不能把这一轮计成“问题已修复”。它的实际产物是把第二层从“结构完成”重新降格为“横向完整性待验证”。

### 3.5 第五轮：十个 SG2044 baseline kernel/phase 闭合

对应提交：`90b15cfa9 refactor: close riscv physical lowering for baseline kernels`  
时间：2026-08-26 06:04:23 +0800  
规模：40 个文件，3963 行新增，615 行删除

#### 实际改动

- 增加 `ShareRISCVLayeredWindows` 与 `UnrollRISCVLevels`；
- 扩充 composite lowering、layout propagation、typed verifier 和 intrinsic-C 生成；
- 增加 quantized vec-dot、row dequantize、MUL_MAT 的正式运行入口；
- 将 layered field window、register gather/lookup、encoded contract 等物理形态接进统一 RISC-V IR；
- 修复受影响的 DSL/std，使它们能到达现有 physical operations。

#### 当轮外部结果

- SG2044 上十个 kernel/phase 与同目标 baseline 真实比较；
- 十项全部超过 baseline，最低领先 1.84%；
- 浮点使用预先规定的 tolerance，离散量化存储使用 bit-exact；
- Weft 和 baseline 使用 Clang 18.1.8、`-O3`、`-ffp-contract=fast`、相同 ABI 与计时范围；
- 当轮只声明 SG2044，不包含 K1。

#### 自查判断

这轮证明十条受针对性推进的路径能够在新两层 IR 上达到手写 baseline。它没有证明横向泛化：只挑十条达标会放大成功路径，无法显示其它量化格式、F16 或 K1 的总体分布。

### 3.6 第六轮：206 条双目标全量回归

对应提交：`5068a1dcd Complete dual-target RISC-V regression`  
时间：2026-08-26 13:27:16 +0800  
规模：31 个文件，1710 行新增，420 行删除

#### 实际改动

- 增加 register gather/extract/lookup 与 bitplane merge 的 typed verifier；
- 增加 `FuseRISCVBitplanes` 和 `HoistRISCVLoopInvariants`；
- 修复 register lookup、materialization、layout conversion、encoded access 与 leaf 合同；
- 扩充 vec-dot、row-dequantize、MUL_MAT、quantize runner，统一 SG2044/K1 远端交叉编译和运行；
- 当前性能 CSV 覆盖写成 206 行真实 target 结果。

#### 当轮外部结果

- 206/206 数值正确：SG2044 103 条，K1/X60 103 条；
- 工具链统一为 Clang 18.1.8；
- 正式 flags 为 `-O3 -ffp-contract=fast -mabi=lp64d`，`-march` 随 target；
- 一次 warmup，每次计时前遍历 64 MiB eviction buffer，十次取中位数；
- 50/206 超过 baseline，涉及 26 个不同的 family/kernel；
- K1 IME local contract 仍可真机 exact，十次中位数 0.375 µs；
- baseline 另有 38 行 activation/norm/memory/attention，没有对应 Weft runner，未进入这 206 条。

#### 自查判断

“全量”在这里指当前 production manifest 的四个 family、103 个逻辑 case × 两台机器，不是仓库所有 example，也不是 baseline CSV 的所有 244 行。该轮完成了正确性全覆盖和既定“至少十条超过 baseline”的门槛；它没有完成“所有性能要求”，因为 128/206 低于 baseline 的一半。

## 4. 当前编程模型是否真正成立

### 4.1 模型定义已经明确

现行设计权威是 [`doc/index.md`](../doc/index.md) 及其链接，不是历史 report。核心定义见
[`programming-model.md`](../doc/model/programming-model.md)：

```text
数学/算子语义
    ↓ 作者选择有限位宽、带 Level/lifetime 的具体 realization
Canonical Kernel program
    ↓ target compiler 物理化
time / lane / register replica / fragment / local storage / leaf
```

作者决定：

- logical values、axes 和 numerical branches；
- partial 的有限位宽与合并位置；
- Level、birth、handoff、ordinary control；
- staging/materialization 的逻辑生命周期；
- persistent Encoding、artifact 和 ABI；
- source `auto` 的有限程序候选。

编译器决定：

- lane/register/fragment/time/local-storage representation；
- scalar/RVV/IME realization；
- local pack、memory form、conversion、spill/reload/rematerialize；
- local pipeline 和 exact intrinsic/asm leaf；
- 结构固定后的 LMUL、microtile、unroll、pipeline depth 等物理参数。

唯一职责判据是：

> 改变 canonical operation/value/effect graph，或者改变 Value 的 Level/artifact 归属，属于作者；只改变同一程序的物理表示，属于编译器。

这个判据不是“授权位”。DSL 没有 `@wide/@matrix` 或 `pack(along)`；显式 operation 和 typed value 本身就是程序语义。target requirement 属于 build config，不进入 canonical value identity。

### 4.2 为什么它不是 Triton-CPU

Triton TTIR 的源模型有 grid、program id 和 block tensor；TTGIR layout 进一步给逻辑元素增加 thread/warp/CTA ownership。Weft 没有虚拟线程或 program instance ownership。一个完整 logical Value 由 target compiler 直接分解到 time、lane、register replica、fragment 和 local storage。

两者都让作者写算法 realization，而不是让编译器发现算法。Triton FlashAttention 也由作者写 online-softmax recurrence、causal phase 和 QK/PV 次序。差异在源程序的一等结构：

- Triton 保存 program-grid/block-tensor 结构；
- Weft 保存 packed Encoding、有限位宽数值树、Level birth/lifetime/handoff；
- Triton 的 tensor layout 围绕 SIMT ownership；
- Weft 的 physical layout 不预先归属于虚拟 thread。

因此 Weft 与 Triton 的不同不是“不 target-aware”。Weft 同样 target-aware，只是 target awareness 从第二层 physical IR 开始，源语言不以某种目标物理对象作为根类型。

### 4.3 当前 DSL 是否真的使用这套模型

外部可见的正面证据：

- current examples/std 已删除 source engine role、local pack permission 和旧 stage handoff；
- dense GEMM 用 NC/KC/MC/MR/NR/KB Level、accumulator birth 与 `outer_contract`；
- attention 用 m/l/o state 和非归约 handoff 表达 online recurrence；
- Top-K 使用普通有序 `for/while/if`，没有为向量化强套 Level；
- Q4_K 同时存在 persistent derived Encoding 与 canonical-input staged path，两者的 ABI/lifetime 差异由作者程序表达；
- 87 个 DSL kernel 都能生成并验证 canonical IR。

仍然存在的 implementation restriction：

1. Python frontend 的 Level birth 分类只识别 Level body 顶层直接赋值给 `new`/`materialize` 的形式；嵌套控制中的 birth 不会自动变成 birth region。这是当前 source lowering 的语法限制，不是抽象机器能力。
2. kernel argument alias 目前把所有 View 放进同一保守 alias set。它安全，但失去 no-alias 事实，直接限制 hoist、reuse 和 schedule。
3. 字符串形式的 source `auto("NAME")` 不在 canonical source 中携带非空 choices；每次 Convert 依赖显式 meta binding。设计中的“作者声明有限源候选域”尚未由这一表面形式自包含实现。
4. 部分量化 Encoding 只声明 raw storage arrays，具体 bit-plane/join 的一部分仍在 std tree 里通过 index/shift 表达。这里必须逐格式区分“真实 numerical decode”和“本该唯一属于 Encoding 的 storage mapping”；当前代码还不足以证明 24 个格式全部遵守同一 mapping 边界。
5. 大量 row-dequant/vec-dot std 使用 Python 普通循环逐元素展开。按语言语义这是合法有序程序，但它不会产生 shaped logical axis，编译器也不应把同构标量迭代重新猜成 axis。

### 4.4 编程模型的结论

编程模型不是当前性能差的临时解释，它已经能一致说明 dense、quant、attention、state 和 scalar control。当前没有证据证明职责判据本身失效：多数严重慢项的 tree 尚未表达 baseline 的 blocking/reuse；F16 则说明同一正确 tree 的 physical compiler 仍不成熟。两者都不要求 target 暗中改 canonical value/Level。

但全量结果也没有“验证 spec 2.2 已经正确完备”。只有当 tree、Encoding、std 特化和 physical passes 都到位后，仍必须改变 canonical values 或 Level 才能达到目标，才构成反例。当前尚未达到这个前提。

## 5. Canonical Kernel IR：是否像 TTIR 一样是真 IR

### 5.1 它是真实 MLIR program IR

`weft_kernel` 在 [`KernelOps.td`](../include/Weft/Dialect/Kernel/IR/KernelOps.td) 中定义并由
[`KernelDialect.cpp`](../lib/Dialect/Kernel/IR/KernelDialect.cpp) 注册。当前具有：

- `EncodingType`、`ViewType`、`ValueType`、`SliceType`、`DomainType`、`PointType`；
- typed SSA operands/results；
- `kernel`、`derive`、Level 的 regions；
- Level carried block arguments、birth regions、body 与 handoff terminator；
- `scf.for/while/if` 的普通有序控制；
- encoding storage span、domain multiplicity、admit/commit、field、reduce、dot/contract、lookup 等 C++ verifier。

Python frontend 使用自己的短生命周期 dataclass graph，再输出 textual MLIR；`weft-compile` 用 MLIR parser 建 module 并调用 `mlir::verify`。因此：

- Python 内部对象不是持久 IR；
- parser 之后的 Canonical Kernel IR 是 MLIR；
- 不能因为 frontend 用文本构造，就把后续 dialect/SSA/region 叫作假 IR。

### 5.2 与 Triton TTIR 的机制差异

| 机制 | Triton TTIR | Weft Canonical Kernel IR | 后果 |
|---|---|---|---|
| frontend 构造 | Python/C++ binding 直接调用 MLIR builder | Python 临时 SSA 后渲染 textual MLIR，再 parse | Weft 多一道文本边界，frontend 类型错误更依赖 C++ parser/verifier 才闭合 |
| 值类型 | builtin ranked tensor + pointer/descriptor types | 自定义 View/Value/Slice/Domain/Point | Weft 可直接保存 Encoding、Level/domain identity；生态兼容与通用 MLIR interface 较弱 |
| op 约束 | TableGen traits/interfaces/type inference 较丰富 | 多个 operand 是 `AnyType`，语义主要由 C++ verifier 检查 | Weft 仍是 typed IR，但重写 pass 更难依赖通用 interfaces |
| op kind | 大量专用 op/enum/interface | 一部分 kind/predicate/bounds 是 `StrAttr` | 某些错误只能由手写 verifier 或 pass 分支发现，结构性重构也更依赖字符串 |
| declaration 解析 | symbol/interface 体系较成熟 | Encoding/derive 仍通过 module walk 按 family/layout/parameters 查找 | 合法但不够规范化，跨模块和 symbol tooling 能力更弱 |

### 5.3 第一层当前成熟度

Canonical IR 已满足“作者程序唯一 authority”的必要条件，并且 87/87 canonical lowering 是横向语料证据。它仍不是成熟到可以无损承接任意 frontend 的稳定 dialect：alias、source-auto domain、nested birth surface 和若干 encoding mapping 尚未闭合；大量 string/array metadata 也使 pass 复用弱于 TTIR。

结论：

> **Canonical Kernel IR 是真正的 MLIR IR，不是旁表；它已经承载当前 DSL 的主要语义，但 dialect interface、alias、source-auto 与全格式 Encoding 闭合程度仍低于成熟 TTIR。**

## 6. 非 SIMT 物理抽象机器是否存在

### 6.1 设计上的机器已经明确

[`physical-machine.md`](../doc/machine/physical-machine.md) 定义：

```text
logical coordinates
    → issue/time coordinates
    × SIMD lane coordinates
    × register-replica coordinates
    × extension-fragment coordinates
    × local-storage coordinates
```

它不是一个五元组统一根类型，而是 logical coordinate 与物理 carrier 的部分、可组合关系。它明确规定：

- scalar 可以只有 time/scalar carrier；
- RVV value 使用 lane 和 register replicas；
- IME value 使用 typed fragment，并通过显式 handoff 与 RVV/local storage 相接；
- staged/spilled value 可驻留 local storage；
- reduce 只能消去 reduction axes，必须保留所有 free axes；
- ordinary scalar loop 不会因同构而获得 shaped axis；
- multiple consumers 通过显式 conversion/reload/rematerialize/spill 解决表示冲突；
- pipeline 必须形成真实 prologue/steady/epilogue 和 buffer versions。

这套定义正面回答了过去 Q4_0 `[M,K] → [M]` 丢失 M 轴、IQ lookup 把 codebook/output 轴映射反的问题。它不是“VLEN+LMUL 清单”，而是一套 logical identity preservation 和 physical decomposition 规则。

### 6.2 在 RISC-V IR 中已经物化的部分

当前 `LayoutAttr` 已携带 carrier、axes、time/lane/replica/fragment/local factors、SEW、LMUL、VL、register groups 和 validity；`ValueType`、`FragmentType`、`MemDescType`、`LocalType`、`WindowType` 是真实 types。

当前真实 physical ops 包括：

- `convert_layout`；
- load/store/field/extract/lookup；
- materialize/staged view/local alloc/bind/load/store；
- spill/reload/register materialize；
- RVV grouped-MAC/encoded-dot window load/step；
- RVV contract step；
- IME pack/MMA/unpack；
- pipeline 展开后的 `scf` control。

这说明抽象机器已经不只存在于文档。

### 6.3 尚未形成完整机器实现的部分

- 只实现 depth=1 和一种 depth=2、two-window、single-carry pipeline；
- 没有 async transfer/wait/barrier；
- 没有通用 register contract multi-stage pipeline；
- fragment spill、cross-block spill 未实现；
- local-storage slot 没有 lifetime coloring，每个 static object 保守独占；
- 跨 engine local-storage ordering、部分 fragment handoff、dynamic tail/time identity 仍是未冻结合同；
- target profile 只覆盖完整 V、显式 VLEN、部分 memory capability 和一个 IME fragment family。

结论：

> **抽象机器已经存在，并已进入 types/ops；但当前 RISC-V backend 只执行了它的一个窄子集。不能把“机器定义完整”与“所有机器机制都已实现”混为一谈。**

## 7. RISC-V Physical IR：是否像 TTGIR 一样是真 IR

### 7.1 旧 side-record 主干已经被真实替换

当前主链没有 `weft_riscv.problem`、`assignment` 或 module 外 solver dictionary。实际顺序位于
[`RISCVCompiler.cpp`](../lib/Target/RISCVCompiler.cpp)：

```text
ConvertWeftToRISCV
→ SelectRISCVOperations
→ PropagateRISCVLayouts
→ SelectRISCVOperations
→ PlanRISCVMemory
→ CanonicalizeRISCVLayouts
→ FuseRISCVBitplanes
→ LowerRISCVComposites
→ HoistRISCVLoopInvariants
→ PipelineRISCVLevels
→ UnrollRISCVLevels
→ ShareRISCVLayeredWindows
→ FinalizeRISCVLeaves
→ MaterializeRISCVResources
→ VerifyFinalRISCV
```

layout 冲突会插入真实 `convert_layout`; resource pass 会插入真实 local slot、spill、reload；composite lowering 会产生真实 loops/windows/fragments；pipeline 会重写成真实 control。每个 pass 后 dump IR 可以看到 program 改变。

### 7.2 与 TTGIR 的共同机制

- layout 进入 value/tensor type，而不是 emitter 邻接表；
- conversion 是 SSA op，可被插入、CSE、rematerialize 或删除；
- memory descriptor 与 local storage 是 typed entities；
- matrix fragment 是 target IR type/op；
- pass 以 program→program 方式运行；
- final translator 从 typed physical program lower 到更低层。

这些是 Triton TTGIR 的关键机制，Weft 已经采用。它们与 SIMT ownership 无关，因此可以复用到非 SIMT 机器。

### 7.3 与 TTGIR 的成熟度差异

1. Triton 的 `RemoveLayoutConversions` 有更丰富的 backward rematerialization 和成本规则；Weft 只消除相邻逆转换、单 use 纯 pointwise rematerialization、同 block conversion CSE。
2. Triton pipeline expander 可按任意 `maxStage` clone stage、predicate 和 prologue/steady/epilogue；Weft 只接受 depth=2、buffer=2、single carry、body 恰好为 window load+step。
3. Weft `FuseRISCVBitplanes` 只识别一个严格局部 expression closure；`ShareRISCVLayeredWindows` 只合并相邻两层。它们格式无关，但 source-spelling 敏感。
4. 一些关键决定仍以普通 string attrs 传递，如 `lane_operand`、`lane_memory_form`、loop direction、system unroll、canonical provenance；它们在 IR 内，但不是强 typed entity。
5. `SelectRISCVOperations` 对 IME fragment 按 target capability 顺序取第一个合法项；当前不是广泛的 target structural rule 系统。
6. `LowerRISCVComposites` 仍按 implementation family、operation/access 字符串进入大段不同 rewrite；这比 whole-kernel route 好，但还没有收敛成通用 axis/layout operation algebra。

结论：

> **RISC-V Physical IR 在机制意义上是真正的 TTGIR 类 physical IR；当前差距是 pass 算法和 target operation coverage，而不是缺少 IR 这一层。**

## 8. Pass 逐项自查

现行文档 [`passes.md`](../doc/compiler/passes.md) 只列十项主 pass，当前源码实际运行十五次，其中 `Select` 运行两次，并增加 Fuse、Hoist、Unroll 和 Share。这个文档漂移说明“设计描述”和“实际编译序列”尚未同步；以下以源码为准。

| 实际 pass | 真实写入/改写 | 已有外部证据 | 当前窄点 |
|---|---|---|---|
| ConvertWeftToRISCV | Canonical op/type → target-aware values、memdesc、loop/control、artifact pack | 87 个 canonical kernels 可到第一层；206 个 case 后续可执行 | 仍复制 `source_origin/canonical_op`；source auto 需外部单值绑定 |
| SelectRISCVOperations #1 | 给 op 写一个 `ImplementationAttr` | scalar/RVV/IME local contract 可闭合 | fragment first-match；跳过 memory/convert；结构规则覆盖窄 |
| PropagateRISCVLayouts | 沿 use-def/control 推导 layout，改 value/block types，插 `convert_layout` | Q4/Q8、lookup、contract、双 target 都实际使用 | 默认最内轴+最小 LMUL；跨 block/复杂 multi-consumer algebra 很弱 |
| SelectRISCVOperations #2 | 在 layout facts 已知后补结构选择 | 解决部分依赖 layout 的 local op | 文档遗漏；仍不是一般 target strategy system |
| PlanRISCVMemory | 写 `AccessAttr`、memory leaf、lane operand/form | unit/strided/indexed、encoded mapping、register lookup 有 runtime consumer | lane operand/form 仍是字符串；mapping 规则少；K1 target facts常只改变 lane 数 |
| CanonicalizeRISCVLayouts | 删除逆 conversion、单-use rematerialize、同 block CSE | typed conversion 路径真实运行 | 无全局 conversion algebra、跨 block hoist/sink、multi-use cost/remat |
| FuseRISCVBitplanes | 严格局部表达式树 → `RVVBitplaneMergeOp` | 全量 IQ/K case 可经过 verifier/runtime | exact local closure；换 SSA spelling 可能不命中 |
| LowerRISCVComposites | 产生 reduction loops、windows、RVV steps、IME fragments、local ops | 206 case 的主要结构由此产生 | 体量大、family/operation/access 分支多；仍有结构性 matcher |
| HoistRISCVLoopInvariants | 移动可证明 pure 的 arith/RISCV value ops | 全量生成路径使用 | 只覆盖白名单 pure ops，受保守 alias set 限制 |
| PipelineRISCVLevels | window cluster → prologue/steady/epilogue | Q4_K grouped window 具备实际消费者 | 仅 depth2/buffer2/single carry/恰好两 op body |
| UnrollRISCVLevels | 消费 unroll factor，调用 MLIR loop unroll | 当前 case physical config 可记录 unroll | 多数 Level 仍为 1；不是构建期全面实测 winner |
| ShareRISCVLayeredWindows | 相邻两层 extract → 双结果 shared window | layered quant formats 有消费者 | 只认 adjacent pair 和 layer×2 |
| FinalizeRISCVLeaves | Implementation → exact LeafAttr，删除 implementation | final verifier 要求每个 terminal op 有 leaf | 普通 op 能闭合；复合 op 的 exact leaf 常在 LowerComposites 已先写入 |
| MaterializeRISCVResources | region-aware liveness、resource peak、spill/reload/local slot | 206 modules 均通过资源验证 | 无 local-slot coloring；fragment/cross-block spill 不支持 |
| VerifyFinalRISCV | 拒绝 canonical leftovers、unassigned facts、pending schedule/leaf、资源越界 | 206 case 都通过并生成可执行 C | 只能证明当前 terminal op 集合法，不证明优化质量 |

### 8.1 pass 不是假的，但很多 pass 仍只覆盖一个窄投影

“假 pass”与“窄 pass”需要区分：

- Propagate、Canonicalize、LowerComposites、Pipeline、Resource 确实改写 IR，不是给 emitter 填一张表；
- 但 Pipeline 只有一种 depth-2 schema，Share 只有两层，Fuse 只有一个 closure，fragment 只有一个已验证 family；
- 因此换输入后没有匹配结构时，pipeline/reuse 能力会消失。这正是“有 pass 不等于泛化”的外部证据。

## 9. Leaf 与 emitter 自查

### 9.1 已经成立的边界

`LeafAttr` 固定 engine、family、instruction、spelling、mask/tail、resource groups 和 local bytes；Value/Fragment/Access/Conversion/Window attributes 固定 shape、layout、memory geometry 和 handoff。多数普通 op 的 intrinsic family由 `FinalizeRISCVLeaves` 选择，emitter 读取 leaf instruction 拼写 C。

当前 active target code 中没有按 `q4/q5/q6/iq/mxfp/nvfp` 格式名注册 whole-kernel route，也没有调用 GGML/materials fallback。K1 IME leaf 只实现一个局部 signed-i8 `M4×K8 × K8×N4 → M4×N4` fragment product，不拥有完整 GEMM traversal 或 ABI。

### 9.2 emitter 仍然过重的事实

[`RISCVIntrinsicC.cpp`](../lib/Target/RISCVIntrinsicC.cpp) 约 8300 行。它不只是 intrinsic 名字适配：

- 入口再次运行 `validatePhysicalProgram`，与 final verifier 职责重复；
- 读取普通 attrs `weft.riscv.level/direction/system_unroll` 决定 C loop 方向和 pragma；
- 建立 `EncodingInfo/MemoryInfo/FieldInfo/Binding`，重新解释 Encoding declaration 和 storage geometry；
- `singleStorageFragment` 按 mapping 计算 byte/shift；
- `resolve("auto:...")` 和 `a|b` 仍能在 emitter 内解析绑定或取首项；
- encoded contract step 内组织 lane load cache、byte/shift/mask、widen、register projection 和 MAC；
- stream contract 内组织 repeated/lane operand 的 load reuse；
- IME leaf 内打印固定 pack loops 和 asm sequence。

这里有两类行为：

1. **合理 terminal lowering**：根据已选 AccessAttr/Encoding mapping 打印 pointer arithmetic，根据 typed layout 展开 static register parts，根据 closed Window/Fragment op 打印有限序列。这与 TTGIR→LLVM pattern 同类，并不等于重新选择算法。
2. **边界泄漏**：`auto:`/`a|b` 解析、关键普通字符串 attrs、重复 physical validation，以及某些复合 op 仍要求 emitter 自己重建足够多的 memory/schedule geometry。前序 IR 没有把这些 facts 全部变成更闭合的 typed op 时，emitter 事实上仍是局部第二解释器。

因此不能写“emitter 已完全机械”。更准确的结论是：

> **emitter 已不再是 whole-kernel selector，但它仍是一个过大的 terminal lowering，部分物理结构只在其 C++ 控制逻辑中完全展开。**

## 10. 206 条全量结果

### 10.1 覆盖与协议

当前 CSV 包含：

| family | 双目标总行数 |
|---|---:|
| quantized vec-dot | 48 |
| MUL_MAT | 104 |
| row dequantization | 48 |
| activation quantization | 6 |
| 合计 | 206 |

每台机器各 103 条。浮点正确性使用预先规定容差；packed bytes、field 和 exact integer storage 使用 bit-exact。数值失败不记录性能。正式计时协议见
[`doc/experiments/protocol.md`](../doc/experiments/protocol.md)。

baseline 还有 38 条 activation/norm/memory/attention 结果没有对应 Weft runner，故当前报告不能称为“覆盖 source 中全部 baseline”。

### 10.2 总体性能分布

| Weft / baseline | 条数 | 比例 |
|---|---:|---:|
| ≥ 100% | 50 | 24.3% |
| 90%–100% | 5 | 2.4% |
| 70%–90% | 15 | 7.3% |
| 50%–70% | 8 | 3.9% |
| < 50% | 128 | 62.1% |

补充分布：

- 63/206 低于 10%；
- 101/206 低于 25%；
- 总体中位 ratio 为 0.264；
- SG2044 中位 ratio 为 0.347，60/103 低于 50%；
- K1 中位 ratio 为 0.228，68/103 低于 50%。

按 family：

| family | <50% / 总数 |
|---|---:|
| MUL_MAT | 72 / 104 |
| quantized vec-dot | 31 / 48 |
| row dequantization | 25 / 48 |
| activation quantization | 0 / 6 |

### 10.3 代表性结果

| case | SG2044 ratio（decode/prefill） | K1 ratio（decode/prefill） | 说明 |
|---|---:|---:|---|
| F32 MUL_MAT | 1.48 / 1.05 | 0.97 / 0.95 | blocked dense tree 和普通 RVV physicalization 已接近或超过 baseline |
| F16 MUL_MAT | 0.089 / 0.224 | 0.093 / 0.045 | tree 已 blocked，但 conversion/load/microkernel/leaf 仍严重不足 |
| Q4_0 MUL_MAT | — | 0.196 / 0.043 | K1 baseline 使用 production repack + IME；Weft production runner 仍是 RVV |
| Q4_K MUL_MAT | 约 0.52 / 0.53 | 0.200 / 0.080 | SG 普通路径仍低；K1 未接 production IME |
| IQ2_XS | 0.037 / 0.031 | 0.035 / 0.034 | codebook/bit extraction/reuse tree 与 pass 都不足 |
| IQ1_M | 0.029 / 0.029 | 0.049 / 0.049 | prefill 几乎等于重复 decode，没有跨 row/output reuse |

## 11. 为什么性能不行

性能差不是一个原因，也不能全部归给 DSL 或全部归给 backend。当前证据支持下面的分层因果链。

### 11.1 第一主因：多数 production std tree 没写出 baseline 的 blocked/reuse realization

[`python/weft/std/mul_mat.py`](../python/weft/std/mul_mat.py) 中，多数量化格式执行：

```text
activation quantize
→ for row
→ for column
→ 调用一次 quantized vec-dot
```

这里说的是上述逐 row、逐 column 的 fallback tree；它没有：

- N/output cohort blocking；
- 多输出 accumulator；
- activation 在多个输出间的共享；
- persistent/repacked weight panel；
- KC/MC/NC 层的跨输出 materialization lifetime；
- production IME fragment 所需的数据供应树。

因此，对采用这类 fallback 的多数量化 K/IQ/TQ 路径，M=128 prefill 主要是在重复 M=1 decode 的逐输出工作；其 prefill/decode 吞吐几乎相等，是直接外部现象。Q4_0、Q4_K persistent/staged 和 F16 已有不同程度的 blocked tree，不属于这句概括；它们的低性能还必须继续归因到 physical pass、leaf 与 target coverage。

按职责判据，compiler 不能把普通 row/column scalar loops 变成新的 shaped axis、cohort、state 或 Level；那会改变 canonical value 集合和 materialization 次数。F32 tree 明确写了 blocking、multi-accumulator 和 outer contract，所以能达到 SG 87%–148%、K1 95%–97%。同一个 backend 下的巨大差异首先证明作者 tree 对性能是决定性的。

这不是说“baseline 决定 backend route”。baseline 决定我们要比较的算法 realization；若 baseline 使用 blocking、persistent repack 或 staged activation，DSL/std 应尽量表达同一算法与 artifact scope，否则比值测到的就是两份不同 realization 的端到端差距。

### 11.2 第二主因：physical representation 与 reuse pass 的横向能力太窄

即使作者 tree 提供 shaped values，当前 pass 仍有下列限制：

- ordinary shaped value 默认把最内轴映射到 lane，并选满足 extent 的最小 LMUL；缺少更丰富的多 consumer / widening / register-pressure propagation；
- conversion canonicalization 只覆盖局部逆转换、单-use pointwise rematerialization 和同 block CSE；
- bitplane fusion 和 layered-window sharing 对 source closure/adjacency 敏感；
- memory plan 对 encoded mapping 有少量固定 form，没有成熟的 coalescing/reordering/reuse system；
- resource pass 没有 local-slot coloring，也不支持 fragment/cross-block spill；
- 多数 target physical config 仍是 unroll=1、pipeline=1。

所以“有 typed IR”没有自动把 IQ lookup window 共享到多个输出，也没有自动产生跨 row/register cohort 的高质量 microkernel。

### 11.3 第三主因：软件流水几乎只在一个窄 cluster 上成立

当前 depth-2 pipeline 要求 loop body 恰好是一个 window load 和一个 window step，且只有一个 accumulator carry。它适用于被专门改写成 grouped/encoded window 的路径，不适用于普通 register contract、复杂 lookup、多个 state、多个 load/decode chain。

这解释了为什么曾经在 Q4_K grouped path 上有效的跨 group 流水，没有自动推广到 IQ、TQ、F16 或普通 dense：这些 tree 没有被 LowerComposites 投影成同一种 window cluster，而 Pipeline pass 本身也不会构造一般 cluster。

### 11.4 第四主因：K1 target capability 没有进入 production operator

K1 的 local signed-i8 IME contract 已经真机运行，但 production runner 显式传入 `--matrix-extension=none`，没有启用 IME；当前 q4 production tree 也没有形成该 leaf 所需的 fragment input/packing。baseline 的 K1 Q4_0/Q4_K 使用 production repack + IME1，Weft 则走普通 RVV。

因此 K1 Q4 的 4%–20% ratio 不表示“VLEN256 RVV 编译错了”这么简单；它比较的是 baseline 的 IME/repack artifact 与 Weft 的 RVV canonical path。差距同时包含作者 tree、artifact scope、physical fragment coverage 和 runner configuration。

### 11.5 第五主因：F16 暴露了纯编译器/leaf 缺口

F16 std 已有 staging/convert、NC/KC/MC/MR/NR/KB Level 和 `outer_contract`，不是简单 row×column vec-dot，但只有 baseline 的 4.5%–22.4%。这说明不能把总体性能差全部归给作者 tree。

目前只能确定差距位于以下组合：

- f16→f32 widening 与 value handoff；
- operand load reuse 和 memory schedule；
- register microtile/multiple accumulator representation；
- contract leaf 与 system compiler 最终代码形态。

当前 runner 退出时删除 generated C，仓库没有保留 206 条对应汇编或硬件计数器，因此不能从 CSV 进一步量化这四项各占多少。把 F16 简单归成“转换太慢”或“GCC/Clang 问题”都没有足够证据。

### 11.6 第六主因：参数 tuner 尚未成为 production 选择系统

当前 compiler API 每次接收一组单值 binding；runner 只对少数格式硬编码 meta，其余使用默认值。没有证据表明 206 条都对 LMUL、microtile、unroll 和 pipeline 参数做了 compile-and-measure 并使用 target winner。

因此部分差距属于参数性工程缺口。但这不能解释所有问题：当 tree 没有 output cohort，或者 pipeline pass 没有合法 cluster 时，扫参数不会产生不存在的结构。

### 11.7 emitter 可能影响代码质量，但现有数据无法单独归因

emitter 内部有大段 address/decode/load organization，system Clang 还负责最终寄存器分配和机器调度。两者可能造成额外差距。但当前实验只保存最终时间，不保存每条 generated C/assembly/perf counters；不能把 128 条慢项统一归为 emitter spelling。

### 11.8 实验口径不是主要解释

当前 Weft/runtime/baseline 都使用同一 Clang 和 contraction policy，并使用相同目标、shape、warmup、cache eviction 和十次中位数。数值错误不会记性能。因此总体低 ratio 不是 GCC/Clang 不对称或 bit-exact 尺子造成的。

需要保留的口径差异是算法/artifact 本身：baseline 可能含 production repack/IME，Weft 当前 tree 可能没有。这不是测量作弊，而是当前实现覆盖不足；报告必须把它归到作者 tree/physical capability，不能拿这个比值证明某个局部 pass 单独慢了多少。

## 12. 当前证据真正证明和没有证明的内容

### 12.1 已经证明

1. 一套无 source engine role/local pack permission 的 Core DSL 可以表达当前 87 个 canonical kernels。
2. Canonical Kernel IR 是 typed MLIR program，Level/domain/Encoding/value/effect 不只存在于 Python 对象。
3. RISC-V Physical IR 是 typed MLIR program，旧 problem/assignment mainline 已被删除。
4. layout conversion、spill/reload、window、IME fragment、pipeline expansion 至少各有真实 IR 实体，而不是 emitter side record。
5. 当前 production manifest 的 206 条双目标 case 都能走统一主链、生成 C、交叉编译、真机运行并通过正确性。
6. 部分作者 tree 能达到或超过手写 baseline：50/206 超过 baseline，F32 dense 在两台机器接近或超过 baseline。
7. 一个局部 IME fragment path 能在 K1 真机正确执行。

### 12.2 没有证明

1. 没有证明 87 个 DSL kernels 都有 target runtime；87/87 是 canonical coverage。
2. 没有覆盖 baseline 的全部 244 行；38 行没有 Weft runner。
3. 没有证明 K1 production MUL_MAT 使用 IME；当前只证明 local contract。
4. 没有证明同一 physical pass 能对自然等价 SSA、不同 consumer、不同 Level context 均产生同类 realization；Fuse/Share/Pipeline 仍有明显 closure 限制。
5. 没有证明 physical target rules 能充分利用 VLEN128/256 的结构差异；大量 K1 case 只改变 lane count，memory/schedule/engine 未显著变化。
6. 没有证明 tuner 在 production 中为每个 target/shape 选了真实 winner。
7. 没有证明 emitter 已退化为纯 spelling adapter。
8. 没有证明 spec 2.2 已被全面验证，也没有发现足以证伪它的 case。
9. 最重要的是：206/206 correctness 没有证明高性能泛化；当前中位 ratio 0.264 反而证明这部分尚未完成。

## 13. 最终成熟度判断

| 层次 | 当前判断 | 理由 |
|---|---|---|
| 编程模型 | **已形成，可继续作为设计 authority** | 根抽象、职责判据、Triton/TileLang 差异、正确性边界已经一致；当前慢项可在该边界内归因 |
| DSL/std | **表面已切换，std realization 不均衡** | dense/Q4_K 有层级 blocked tree；多数 K/IQ/TQ 仍是 row×column vec-dot；部分 encoding mapping 仍未完全闭合 |
| Canonical Kernel IR | **真实、可验证、中等成熟** | typed SSA/regions/verifier 已成立；alias、source auto、nested births、string-heavy contracts 仍弱 |
| 非 SIMT 抽象机器 | **定义完整度高，执行子集有限** | time/lane/replica/fragment/local storage 已定义并部分进入 types/ops；异步、fragment spill、通用 pipeline 未实现 |
| RISC-V Physical IR | **真实、不是 side record** | typed layout/conversion/memory/local/fragment/window，真实 program rewrite；但仍含关键字符串 attrs 与窄 composite ops |
| Pass 系统 | **结构真实，横向能力不足** | 15-stage program pipeline 已运行；conversion/reuse/pipeline/resource/target rules 覆盖窄，文档还落后于源码 |
| Local leaf | **有清楚合同，覆盖窄** | RVV 普通 leaf 和一个 signed-i8 IME fragment 闭合；没有 production q4 IME 与多 fragment family |
| Emitter | **不再是 whole-kernel selector，但仍过重** | 消费 typed facts，同时重建 encoding/memory/复合 leaf 局部结构并保留少量决策泄漏 |
| Correctness coverage | **当前 production manifest 宽** | 206/206 双目标正确；不是全部 baseline/全部 examples |
| Performance maturity | **明显不足** | 128/206 <50%，中位 0.264；F16、K/IQ/TQ 和 K1 IME integration 是实质缺口 |

## 14. 最后结论

从指定起点开始的重构不是“什么都没推进”。它完成了三件过去长期没有真正完成的基础工作：

1. source DSL 去掉了物理权限位和 local-pack hint，回到 encoding-aware、Level/lifetime-aware 的数值程序；
2. Canonical Kernel IR 与 RISC-V Physical IR 都成为真正的 MLIR SSA program，旧全局 planning/assignment 主干被删除；
3. 当前量化/dense production manifest 的 206 条 case 已经通过同一主链在 SG2044/K1 真机正确执行。

但如果问题是“现在是不是已经有 Triton 级的高性能编译器”，答案仍然是否定的。失败不在“完全没有编程模型”或“IR 还是假的”；失败在模型落实之后的两侧都不均衡：

- 作者侧只有少数 std tree 写出了与 baseline 相当的 blocked、staged、persistent/reuse realization；
- 编译器侧只有少数 physical patterns 拥有成熟的 layout、reuse、pipeline 和 leaf；
- K1 的 target extension 只在局部 leaf 上闭合，没有进入 production operator；
- terminal emitter 仍承担过多复合 physical lowering；
- 构建期参数选择没有覆盖全量 production cases。

所以当前最诚实的判断是：

> **Weft 已经拥有一套可辨认、可执行、两层 IR 真实成立的算子编译器架构；它不是旧高性能路径的纯包装。但它还没有证明可以仅靠编写一棵新的自然 std tree，就在不同格式、不同外围程序和不同 RISC-V target 上自动得到接近手写的 realization。当前全量性能分布表明，这个核心编译能力仍未成熟。**
