# 基于逻辑轴分解的 Core-local 物理编译器：本轮推进报告

## 报告范围

本文只记录上一条“建立基于逻辑轴分解的 Core-local 物理编译器”prompt 所推动的那轮工作，
即约十小时连续推进形成的提交区间：

```text
1134eaaf7 generate local implementations from axis mappings
    ...
cea3a353d derive local primitive layouts before emission
```

该区间共 61 个提交，修改 27 个文件，代码变化为 `+7771/-4928`。核心变化集中在
`lib/Target/`：本轮没有修改 DSL kernel、examples、source baseline 或 materials，也没有创造
第二条 lowering 路径。

本文中的性能数字只来自这十小时推进期间已经完成的真实运行。本文整理时没有重新执行
目标机命令；之后误启动的运行不属于本轮，也没有进入本文或性能 CSV。

## 本轮原本要替换什么

本轮开始前，编译器已经拥有 typed facts、target profile 和若干 physical decision，但实际主体
仍接近：

```text
局部轴事实
→ 判断属于 VLA dot、local-row、dense、某种 quant 等人工结构
→ 枚举该结构自己的 LMUL、microtile、unroll
→ 进入对应 leaf 或 emitter
```

这仍然是“识别一种已有高性能形态，再填参数”，而不是从作者写下的 Core-local blocked
program 生成 core 内部执行布局。本轮的目标是将主体替换为：

```text
Kernel IR 中的轴、值、memory、lifetime 和 primitive 事实
→ operation 与 target 共同给出 mapping constraints
→ time / lane / register / unroll / fragment 分解
→ value-use 链上的 handoff、resource 与 pipeline
→ typed local leaf / projection
→ intrinsic C 或局部 asm
```

作者拥有的 outer traversal、blocking、staging、workspace、persistent layout 和算法 variant
没有被 backend 改写。

## 一、建立了统一的逻辑轴分解核心

本轮新增并贯穿主干的核心是 `RISCVAxisMapping.h/.cpp`。

`CoreMappingProblem` 只描述待物理化的事实和合法空间：

- logical axis 及其 `Free / Reduction / Broadcast / Packed / Group / State` 角色；
- extent、ordered 约束、能否或必须进入 lane；
- register factor、unroll、pipeline buffer 和 fragment 的候选；
- lane element width、RVV instruction family 和 target fragment capability。

`enumerateCorePhysicalMappings` 将它们统一投影为：

```text
PhysicalAxisDecomposition
  sequentialFactor
  laneFactor
  registerFactor
  unrollFactor
  fragmentFactor

CorePhysicalMapping
  instruction
  laneAxis / laneShape
  axes[]
  pipeline.bufferCount
  fixedFragmentGroups
```

因此 VLEN、SEW、LMUL 和 fragment 不再是高层实现身份。它们只是 target facts 和候选约束的
输入；最终 sequential factor 由逻辑 extent 除以 lane、register、unroll、fragment 已实现因子
统一得到。

对应的第一组提交是：

```text
1134eaaf7  generate local implementations from axis mappings
cb9e6821d  generate dense kernels from axis decompositions
cc6571204  map VLA values memory and state through axis plans
9a652e47c  generate quantized mappings from logical axes
```

这四个提交完成了从“各 family 自己描述一种结构”到“各实体向同一个轴分解器贡献约束”的
主干切换。

## 二、物理映射开始覆盖连接的值链，而不是孤立 operation

本轮随后把 mapping 从 primitive 自身扩展到了普通 value-use、storage 和 handoff：

- block pointwise/value chain 从已选 axis mapping 获得物理 shape；
- structured result 的 store 不再要求 direct-store 或相邻 consumer；
- matmul accumulator、输入和结果 storage handoff 在 emission 前形成；
- VLA dot 的 operand handoff、cast、math、load 和结果映射沿普通 use-def 传播；
- block store、block reduce、ordered loop 与 carried state 保留同一物理表示；
- 无法获得正式 mapping 的 block value 会明确失败，不再进入隐藏 scalar 或旧 emitter。

代表提交包括：

```text
490321d30  generate block value chains from axis mappings
1585703a7  map structured result stores onto core axes
6803af923  materialize matmul value and storage handoffs
125c6046c  materialize VLA dot operand handoffs
be18bb6ed  propagate VLA axis mappings through value chains
575915a8c  preserve author ordered loops in emission
b38de7ebe  reject unmapped block value operations
```

这里形成的边界是：普通 consumer 可以改变 handoff 和资源占用，但不能决定某个 structured
primitive “还有没有 lowering”。

## 三、资源合法性由最终映射统一计算

本轮建立了统一的 `PhysicalLiveRange` 与 `PhysicalResourceRequirements`。资源类别包括：

```text
Value
Memory
Index
Predicate
State
Temporary
Fragment
Handoff
```

`calculatePhysicalResources` 根据已选 RVV shape、数量、固定 fragment groups 和 target 的 vector
register 数计算峰值占用，超出预算的 mapping 在代码生成前被判为非法。量化路径又通过
`AxisMappedLiveValue` 将 register factor、lane capacity、logical chunk 等 multiplicity 投影为同一
套 live ranges。

因此 accumulator 数量、operand window、mask/index、state、decode temporary、pipeline buffer
和 extension fragment 至少开始在同一资源预算内相互竞争，而不是每个 family 各自假设还有
一组可用寄存器。

关键提交包括：

```text
a9ffbb739  combine VLA resources from typed live ranges
882996f95  derive quant resources from axis mapped live values
24cb8ee69  select i4 i8 resources from fragment mappings
691a84369  select state and fragment mappings from target facts
```

## 四、dense dot/matmul 共用 structured-product 轴投影

本轮删除了 VLA dot、local F32 dot 和 F16 matmul 分别手造 M/N/K mapping 的主体。
`StructuredProductFacts` 保存 operand/result 的真实轴关系，`projectStructuredProductAxes` 再把：

- result/free axes；
- reduction axes；
- operand 上的 broadcast 或缺失轴；
- 当前显式 VLA lane axis；
- target 支持的 lane/fragment 能力；

投影为一个 `CoreMappingProblem`。

由此得到的 RVV strip、register microtile、multiple accumulators 或 IME fragment 是轴分解结果，
而不是进入 lowering 之前先选中的 dense template。F32 dot 与 F16 matmul 的生成也改为读取
selected lane/register/unroll/pipeline decomposition。

相关提交：

```text
c34914f0f  derive VLA dot realization from operand axes
a48f5d928  derive local dense mappings from shared axis facts
fbebdfa5f  generate local dot pipelines from axis mappings
b0f549a43  generate dense programs from selected lane axes
97dea2b69  separate dense axis facts from realizations
642856e6b  project structured products through one axis mapping
```

## 五、VLA、memory 与 state 进入相同的轴和生命周期体系

VLA 不再只提供一个 `vl`。本轮将同一 region 中的 value shape、memory access、mask、index、
state placement 和 local primitive resource 合并到 `VLAEntityCandidateFacts`：

- unit/strided/indexed/segment memory 由 typed address facts 与 lane mapping共同决定；
- cast/widen/narrow 保持逻辑元素映射，并传播 SEW/LMUL 变化；
- reduce、scan、ArgMax、online summary 的 carry placement 根据 state 语义、control carry、
  cross-region lifetime 和 target resources 选择 scalar/vector realization；
- ordered state 保留顺序语义，不会因为存在 RVV reduction 而被自动改成可重结合算法；
- segment memory 与 state 使用同一 mapping/resource authority。

代表提交：

```text
e56f1a5e0  select VLA state placement with axis resources
251fa4adf  materialize VLA state vector shapes
eb32440ac  derive segment memory from typed address facts
6aa8c19af  derive state placement from value axis facts
5ba79d268  propagate VLA mappings by value axes
bb5af3c18  derive state placement from typed axis facts
```

## 六、quant、decode、codebook 与 IME 被迁到轴映射主干

本轮不是建立一个“quant mapping”总开关，而是把 packed/decode primitive 的实际 operand extent、
reduction axis、group/packed relation、widening chain 和 codebook gather 逐项接入公共 mapping 和
resource 计算。

已迁移的局部结构包括：

- packed I4/I5、grouped I3；
- base-3 与 packed-I2 ternary；
- signed codebook、nibble codebook、packed U9/U7 与 U11 grid/delta；
- IQ1/IQ2/IQ3、Q6_K；
- E2M1/MXFP4；
- grouped affine I4×I8；
- RVV register realization 与 IME fragment realization。

IME 不再独立拥有完整 quant/GEMM route。它由 target fragment capability 向同一个 mapping problem
增加 fragment constraints，最终 leaf 只实现当前局部 primitive。

相关提交从：

```text
7c61faad3  generate packed quant dots from axis mappings
78c63dbed  generate nibble codebook from value mappings
95ecc424e  generate packed ternary dots from reduction mappings
2fd8fb48a  generate base3 ternary dots from typed axis facts
f47cdb3dd  generate signed codebook dots from value mappings
1c7b0cc22  derive IME fragments from target capabilities
```

一直覆盖到 IQ1/IQ2/IQ3、Q6、grouped-I3 和最终 operand-driven quant mapping。

## 七、physical decision 与 intrinsic C 的权责进一步闭合

本轮后半段持续删除 emitter 中的第二次选择：

- `LocalLeafKind` 是 selected local primitive realization 的唯一身份；
- grouped affine L16/L32 在 planning 阶段成为不同 typed leaf；
- packed dot 的 lane-slide、lane-create、register-chunks 成为显式 `LocalLeafProjection`；
- nibble codebook combined/split 与 sign-source direct/extend 在 emission 前决定；
- block store 的 multi-strip、register repetition、sequential strip 在 planning 阶段物化；
- emitter 只根据 selected shape 拼 RVV intrinsic 类型/后缀，根据 selected leaf/projection 机械生成
  intrinsic C 或局部 asm。

收尾提交是：

```text
549d14791  materialize typed local leaf decisions
1167781e8  make typed leaves the sole local mapping authority
080600137  select grouped quant leaf width before emission
d4f2dd493  materialize packed dot projection decisions
cea3a353d  derive local primitive layouts before emission
```

这一步没有消灭底层 leaf：RVV intrinsic 和 IME asm 仍然必须知道具体 ISA spelling。删除的是 leaf
或 emitter 根据 VLEN、shape、format 或周围 closure 再决定“使用哪种机器结构”的权力。

## 八、本轮已有真实运行结果

### 数据如何写回 CSV

十小时推进期间留下了 31 个与现有 CSV 协议严格一致的真实测量结果。它们被写回
`weft-kernel-performance.csv` 的 43 行；行数多于测量数，是因为同一真实 runner 同时出现在早期
压力语料和后来的 `source/` coverage 分类中。

写回时保留了原有 shape、repetitions、warmup、64 MiB eviction、correctness scope 和 preprocess
scope，只替换实际重测得到的误差、`median_ms` 与 throughput。没有同口径结果的行保持原值。

下表中的“相对旧记录”按 `旧 median / 新 median` 计算，大于 1 表示本轮结果更快。这只是同一
Weft runner 在两次真实记录之间的变化，不等于对 GGML baseline 的速度比。

### Dense 与 structured product

| Kernel | Target | 旧记录 ms | 本轮 ms | 本轮吞吐 | 相对旧记录 |
|---|---|---:|---:|---:|---:|
| F16 blocked GEMM decode | SG2044 | 16.656145 | 12.058155 | 2.782717 GOP/s | 1.381× |
| F16 blocked GEMM decode | K1 RVV | 40.378719 | 39.967162 | 0.839550 GOP/s | 1.010× |
| F32 blocked GEMM decode | SG2044 | 15.755328 | 14.902427 | 2.251609 GOP/s | 1.057× |
| F32 blocked GEMM decode | K1 RVV | 20.003544 | 22.107040 | 1.517817 GOP/s | 0.905× |
| Dense Conv2D | SG2044 | 783.915562 | 2425.425363 | 1.992161 GOP/s | 0.323× |
| Dense Conv2D | K1 RVV | 3795.740917 | 3625.416686 | 1.332768 GOP/s | 1.047× |
| Out Product | SG2044 | 185.177917 | 198.458862 | 3.719650 GOP/s | 0.933× |
| Out Product | K1 RVV | 1084.935589 | 1300.862175 | 0.567468 GOP/s | 0.834× |
| Lower-triangular solve | SG2044 | 6.373089 | 5.852147 | 5.733696 GOP/s | 1.089× |
| Lower-triangular solve | K1 RVV | 14.804404 | 21.599128 | 1.553509 GOP/s | 0.685× |

这一组结果说明 dense 结构已经进入共同轴映射，但 candidate quality 并未因此自动成熟。SG F16
decode 改善，K1 F32 和 ordered dot 退化；最严重的是 SG Dense Conv，说明同一 matmul relation
已经共享编译机制，不等于陌生 outer context 下已经拥有正确的 microtile、load reuse 和 pipeline。

### Quant、decode 与 codebook

| Kernel | Target | 旧记录 ms | 本轮 ms | 本轮吞吐 | 相对旧记录 |
|---|---|---:|---:|---:|---:|
| IQ2_S × Q8_K | SG2044 | 66.217287 | 43.131915 | 2.722822 GOP/s | 1.535× |
| IQ2_S × Q8_K | K1 RVV | 76.754420 | 76.286148 | 1.539474 GOP/s | 1.006× |
| IQ3_S × Q8_K | K1 RVV | 965.733121 | 72.219338 | 1.626164 GOP/s | 13.372× |
| IQ1_M × Q8_K | SG2044 | 39.012231 | 51.185675 | 2.294402 GOP/s | 0.762× |
| Q6_K × Q8_K | SG2044 | 19.068306 | 39.071997 | 3.005746 GOP/s | 0.488× |
| Q6_K × Q8_K | K1 RVV | 52.532771 | 120.061536 | 0.978169 GOP/s | 0.438× |
| Q4_0 × Q8_0 | SG2044 | 44.692426 | 81.730441 | 1.436925 GOP/s | 0.547× |
| Q4_0 × Q8_0 | K1 RVV | 58.997957 | 119.486150 | 0.982880 GOP/s | 0.494× |
| Q5_0 × Q8_0 | SG2044 | 44.046073 | 43.718828 | 2.686269 GOP/s | 1.007× |
| Q5_0 × Q8_0 | K1 RVV | 59.819003 | 120.114153 | 0.977741 GOP/s | 0.498× |
| IQ2_XXS × Q8_K | SG2044 | 48.989514 | 48.610799 | 2.415935 GOP/s | 1.008× |
| IQ2_XXS × Q8_K | K1 RVV | 102.603209 | 105.859795 | 1.109397 GOP/s | 0.969× |
| IQ4_NL × Q8_0 | SG2044 | 20.391029 | 20.369092 | 5.765623 GOP/s | 1.001× |
| IQ4_NL × Q8_0 | K1 RVV | 58.202514 | 58.322318 | 2.013646 GOP/s | 0.998× |
| TQ1_0 × Q8_K | SG2044 | 26.886398 | 26.839782 | 4.375613 GOP/s | 1.002× |
| TQ1_0 × Q8_K | K1 RVV | 34.704337 | 34.779839 | 3.376684 GOP/s | 0.998× |
| MXFP4 × Q8_0 | SG2044 | 19.835046 | 53.550382 | 2.193084 GOP/s | 0.370× |
| MXFP4 × Q8_0 | K1 RVV | 54.709999 | 64.391743 | 1.823844 GOP/s | 0.850× |
| Codebook lookup affine | K1 RVV | 5.177485 | 4.640791 | 225.947689 MElements/s | 1.116× |
| Dequantize Q4_0 | SG2044 | 5.690335 | 5.746055 | 729.944980 MElements/s | 0.990× |
| Dequantize Q4_0 | K1 RVV | 15.362145 | 13.085457 | 320.531717 MElements/s | 1.174× |

量化结果呈现的是结构重构的真实代价，而不是单向提升：IQ3_S 的 K1 病态路径被消除，IQ2_S
和 K1 Q4 dequant 有明显改善；但 Q6、Q4_0、K1 Q5_0 和 MXFP4 明显退化。这些退化说明新
mapping 已成为 authority，但 decode-compute fusion、register organization、chunk projection 与
leaf cost/selection 尚未恢复旧高性能实现中的关键组织。

### 正确性

上述写回 CSV 的运行都完成了原 runtime 的真实数值对照：

- dense、Out Product、Dense Conv、Q4/Q5/Q6、IQ2_S、IQ3_S、IQ1_M、IQ4_NL、TQ1、MXFP4、
  codebook lookup 和 Q4 dequant 的记录为 exact 或原 runtime 允许的既有数值范围；
- lower-triangular solve 的最大绝对误差为 `1.1920929e-07`，最大相对误差为
  `9.50217072e-05`；
- IQ2_XXS 的最大绝对误差为 `0.00016784668`，最大相对误差为 `1.52587891e-05`。

### 本轮有运行、但没有写回 CSV 的数字

开发过程中还有两组真实 spot run，但它们与 CSV 既有 protocol 不一致，因此只在这里说明，
不伪装成正式 CSV 更新：

- Q4_K × Q8_K、FFN-up decode、单次 repetition：SG2044 Weft `11.661373 ms`、GGML
  `12.345236 ms`；K1 Weft `52.833057 ms`、GGML `45.629553 ms`。它说明同一 grouped quant
  mapping 在 SG 有竞争力，在 K1 仍落后，但不能替换 CSV 中 repetition 30/10 的行。
- Q4_0 projection 的 SG RVV、K1 RVV、K1 IME quick run 分别得到 `6.487249 ms`、
  `21.093232 ms`、`7.251294 ms`；该组使用了与 CSV 不同的 repetition，而且 K1 IME 受到同机
  并发运行干扰，所以没有写回性能表。

## 九、这十小时真正逼出的编译能力

本轮不是增加了一组 case，而是形成了以下共享能力：

1. **一个物理轴分解器。** sequential、lane、register、unroll 和 fragment 由同一 mapping
   problem 生成，不再分别属于 dense、VLA、quant 或 IME 的高层模板。
2. **连接的 value-use physical handoff。** structured result、普通 consumer、storage、loop carry
   和 state 开始共享同一物理 value authority。
3. **统一资源预算。** operand、accumulator、memory/index/mask、state、decode temporary、pipeline
   buffer 与 fragment 进入同一 register-group 预算。
4. **target capability 驱动的 realization。** VLEN、合法 LMUL、vector register 数和 IME fragment
   约束参与候选 legality，不再作为实现路径名称。
5. **typed leaf 与 projection。** emitter 不再凭 lanes、shape equality 或周围 closure 重新选择
   grouped width、packed projection、nibble/sign layout 与 block-store realization。
6. **真实的局部 pipeline 形态。** dense dot/matmul 已能根据已选 mapping 生成 single-buffer 与
   register double-buffer 代码，而不是只有一个无效的 pipeline 字段。

## 十、仍未完成的部分

这轮完成了主干替换，但还没有完全达到 prompt 中“从约束系统生成所有局部机器程序”的最终
状态。

### 1. Pipeline 仍主要服务 dense reduction

当前 dependence facts 和 banked emission 已经是真实代码，但主要围绕 F32 dot/F16 matmul 的两条
operand load stream、一个 accumulator 和一个 reduction axis。它还不是任意 local value DAG 的
通用 load/compute pipeline 生成器。

### 2. Candidate construction 仍存在 family-specific 层

公共 `CoreMappingProblem` 和资源计算已经建立，但 dense、state、block memory 与不同 quant
primitive 仍各自组装部分 candidate facts、偏好和 live ranges。它们不再按 kernel 名或 VLEN
route 选择 whole-kernel 实现，但也还没有完全收敛为“每个 operation 只贡献约束”的单一求解。

### 3. 量化 leaf 仍较多，且性能空间没有闭合

`LocalPrimitiveKind/LocalLeafKind` 仍保留 IQ1/IQ2/IQ3/Q6、nibble、signed codebook 等局部数值
primitive 的 typed leaf。这些 leaf 没有外围 traversal 或 ABI，因此不是 whole-kernel route；
但 Q6、Q4_0 和 MXFP4 的退化证明，其 mapping、decode topology、chunk reuse 和局部流水还不足以
从同一数值 primitive 自动恢复高性能实现。

### 4. Dense mapping 已共享，dense 性能尚未共享

GEMM、Conv、Out Product 和 ordered dot 已经走共同 structured-product/axis 机制，但 SG Dense
Conv 的 3.09× 退化以及 K1 Out Product/solve 的退化表明，陌生 outer context 下的 microtile、
operand reuse、load ordering 和 pipeline candidate 仍过窄。

### 5. 协调代码仍然过大

`RISCVKernelCompiler.cpp` 仍承担大量 facts 提取、candidate 构造、selected program projection 与
普通 C 控制流生成。信息 authority 已比之前清楚，但模块边界尚未完全体现新的 mapping 主干。

### 6. 本轮没有完成全量 baseline 复跑

本轮使用代表性的 dense、quant、state、memory 与 IME 程序持续检查主干，并保存了上述已有
性能结果；没有重跑全部 399 条性能记录。因此不能从本报告推出“全仓所有现有记录已在
`cea3a353d` 上重新确认”。CSV 只更新了本轮实际获得且协议一致的数字。

## 最终判断

这十小时最重要的结果不是新增了多少 leaf，而是 Weft 第一次拥有了真实的 core-local physical
mapping 主干：

```text
logical axes / values / memory / lifetime
→ mapping constraints
→ sequential / lane / register / unroll / fragment decomposition
→ live resources and handoff
→ typed local leaf / projection
→ intrinsic C / local asm
```

因此，原先“先判断属于哪一种高性能结构”的主体已经被实质替换；dense、VLA、state、memory、
quant 与 IME 都开始共享轴分解、资源和 selected-decision authority。

但本轮不能被描述为性能闭合。已有数字同时证明两件事：统一 mapping 能消除某些病态路径并
让 target facts 真正影响实现；它也暴露出当前 candidate generation、decode organization、
microtile、reuse 和 pipeline 仍然不足。当前 Weft 已经从高性能模式库向物理编译器跨过了核心
结构门槛，但还没有达到 Triton 式“统一布局生成同时稳定地产生高质量实现”的成熟度。
