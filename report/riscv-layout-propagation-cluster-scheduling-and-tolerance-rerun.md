# RISC-V 表示传播、局部流水与容差口径重测

## 报告范围

本轮同时处理两件事：

1. 撤销把浮点 bit-exact 当作 kernel 正确性标准的错误口径，统一前端、runtime、runner 与性能记录；
2. 针对 IQ2_XXS 暴露的跨消费者重复提取、layout 不匹配和无跨迭代流水，建立 use-edge conversion、encoded storage mapping 传播和 loop-local cluster schedule，并在 SG2044/VLEN128 与 K1/VLEN256 上运行受影响的不同结构。

这不是一次全量 206 项重测。当前性能 CSV 只保存本轮在同一编译口径下真实重测的 20 条 Weft 结果；固定 GGML baseline 仍单独保存在 `report/baseline/ggml-riscv-kernel-performance.csv`，没有与当前 Weft 数字混写。

## 结论先行

本轮得到的是一组真实但尚未完整的横向能力，而不是性能闭合：

- Q4_K 派生布局路径在两台机器都没有退化：SG2044 decode/prefill 分别达到同轮 GGML 的 96.94%/103.93%，K1 分别达到 127.44%/164.28%。此前约 9.4 GOP/s 的 SG2044 结果仍然有效；末位浮点差异不再被误判为失败。
- typed conversion 已经成为具体 value-use edge 的瞬态决定，lane-to-register 提取具有同一 use cluster 内的缓存；但是当前显式 conversion 只覆盖 pointwise、cast、widen/narrow 和 lookup 的 mapped operand，Level carry、handoff、structured operation 等仍使用 emitter 的通用坐标投影。因此不能声称 Triton 式 conversion insertion/elimination 已完整实现。
- encoded storage identity 可以沿 slice、admit、stage handoff 和 local materialize/use-def 传播，dense pack 与 encoded pack 共用同一 pass；但这还没有让 canonical K/IQ 路径自然达到高性能。
- generic loop-local scheduler 确实生成了 prologue/steady-state/epilogue，而不只是保存 `pipeline_depth=2`。它让 standalone IQ2 vec-dot 在 SG/K1 分别提升 56%/44%，却仍只有手写的约 5%；Q4_0 上 depth 2 反而变慢，实测 winner 都是 depth 1。
- IQ2 blocked tree 的主要瓶颈不能由本轮 pass 合法消除：entry/codebook partial 在作者树中跨 group Level 存活。把 reduction 移进 group 会改变逻辑值集合和 Level 归属，属于作者树变更，正好触发 spec 2.2 的边界。本轮没有替作者改树。
- 42 份把逻辑轴写成标量展开的 std 树按要求未改。pass 没有从同构 SSA 重新发明轴；这些程序继续表现为标量或窄映射是当前语言语义的直接结果。

## 一、正确性口径修正

### 1.1 Encoding 与浮点 kernel 分开

bit-exact 只继续用于离散 storage bytes、packed field 和 encoding layout 的验证。浮点 kernel 现在采用统一判据：

```text
所有输出有限
并且 abs(actual - expected) <= 1e-4 + 2e-3 * abs(expected)
```

runtime 同时打印最大绝对误差、最大相对误差和：

```text
numeric=within-tolerance
```

这项改动覆盖 GEMM、GEMV、MUL_MAT、Q4_K GEMV、row dequantize 和 vec-dot runtime。activation workspace 等离散 encoding storage 仍逐字节比较；没有把 encoding 错误藏进浮点容差。

当前 Q4_K 派生路径的最大误差如下：

| target / phase | max abs | max rel | 判定 |
|---|---:|---:|---|
| SG decode | 0.03125 | 9.01e-8 | within tolerance |
| SG prefill | 0.125 | 1.10e-3 | within tolerance |
| K1 decode | 0.03125 | 9.01e-8 | within tolerance |
| K1 prefill | 0.125 | 1.10e-3 | within tolerance |

这些差异来自合法 contraction/乘加结合，不再为复刻 reference 的舍入位置交换 scale 顺序或阻断 contraction。本轮源码中没有保留 flat-32 iota、强制 scalar-f32 materialization 或 scale reassociation 的实验路径。

### 1.2 `iota` 前端与 canonical IR 对齐

`W.iota(extent, start=0, dtype=u32)` 现在是正式 canonical op，并显式产生一个带新逻辑轴的 shaped value。前端和 verifier 都只接受 unsigned integer dtype，不再出现前端接受 `index`、canonical verifier 再拒绝的错位。

spec 同时明确：

```text
一个 shape=[8] 的 iota value
!=
Python/DSL 中八次有序标量展开
```

编译器只映射作者显式写出的逻辑轴，不从 exact-op-count 或 source closure 重新识别轴。

## 二、参考 Triton 与 TileLang 后采用的边界

本轮参考了 `ref/triton` 的 layout conversion/rematerialization 组织和 `ref/tilelang` 的 pipeline 依赖与展开组织，采用了两条具体约束：

1. layout 冲突属于 value-use edge，不属于 kernel 或格式；source/target mapping、SEW、LMUL、part projection 和 lane extraction 必须在 emission 前成为 typed transient fact；
2. schedule 分成两部分：scheduler 根据 loop-local use-def/effect 形成 producer、consumer、frontier 和 buffer 计划，emitter只把选定计划展开为 prologue、steady-state 和 epilogue。

Weft 没有复制 Triton 的持久 GPU layout IR，也没有复制 TileLang 的 GPU pipeline stage surface。新增信息仍只存在于一次 RISC-V lowering 内的 `riscv.problem` 瞬态 assignment。

## 三、当前 RISC-V pass 主干

本轮后的主干是：

```text
ConstructRISCVProblems
→ AssignRISCVRepresentations
→ ResolveRISCVLayoutConversions
→ PropagateRISCVStorageMappings
→ SelectRISCVLocalOperations
→ ScheduleRISCVLevels
→ CheckRISCVResources
→ SelectRISCVWinner
→ intrinsic C / local asm
```

各 pass 的实际读写如下：

| pass | 读取 | 写入/决定 | 当前边界 |
|---|---|---|---|
| Construct problems | canonical op、Level、meta/auto、target profile | value/op id、use-def、control path、候选参数实例 | 不改变作者树 |
| Assign representations | axis、free/reduction 关系、producer/users、handoff、VLEN/SEW | value 的 lane/register/stream、SEW、LMUL、live range | 仍有未显式轴的树只能得到标量形态 |
| Resolve layout conversions | source/consumer value mapping | 每个 mapped operand 的 typed `use_conversion` | 当前只覆盖 pointwise/cast/widen/narrow/lookup |
| Propagate storage mappings | encoding/derive、pack/materialize、handoff/use-def | base family、interleave rows、layout identity、pack axis | 支持 dense 与 encoded local/derived pack |
| Select local operations | value mapping、memory relation、primitive、target facts | RVV memory/lookup/MAC 等局部 realization | 不按格式名选择 whole kernel |
| Schedule Levels | loop control path、producer/users、effect、frontier | local cluster、producer/consumer order、pipeline structure | 只支持 depth 1/2；不是所有 loop 都能形成 cluster |
| Check resources | liveness、mapping、primitive temp、pipeline frontier | peak、reload/rematerialize 或 candidate invalid | 纯 computed aggregate 可按 register part 重算；任意 computed spill 未实现 |
| Select winner | legal parameter instances 与真机结果入口 | 当前 invocation 的 measured winner | tuner 不形成持久 compiler stage |
| emitter | 上述 assignment | C control、intrinsic、局部 asm | 仍有部分 control/handoff 使用通用 projection，conversion 消除未完全前移 |

这张表也说明了没有完成的部分：当前不是“所有 layout conflict 都插 typed conversion、全局 canonicalize 后 emitter 纯读 conversion op”的完整状态。

## 四、三项横向能力的真实结果

### 4.1 Typed value-use conversion

`ResolveRISCVLayoutConversions` 为具体 operand edge 写入：

```text
source / target value
source / target lane axis
source / target SEW 与 LMUL
source part
target part
lane offset
relation = identity | scalar-broadcast | project | lane-to-register
```

emitter 的 pointwise、cast、widen/narrow 和 lookup 读取这些字段。lane-to-register 的 scalar extraction 按 source value/part/lane 缓存，同一局部使用簇不再为每个 consumer 重复发 `vslidedown + vmv.x.s`。

IQ2 blocked assignment 中观察到：

- 183 条 value-use conversion edge；
- 7 条 lane-to-register conversion；
- 14 个不同 conversion materialization 名；
- 生成 C 中 30 次 slide/extract，而不是每个后续 consumer 无条件重建。

这证明 conversion 已经受真实输入影响；但它还没有覆盖所有 use edge，也没有独立的 conversion canonicalization/fixpoint pass。当前结果应称为 typed use-edge conversion 的第一段，而不是完整 Triton 等价实现。

### 4.2 Packed/index/codebook storage mapping

`PropagateRISCVStorageMappings` 从 encoding declaration 和 derive 读取：

```text
base encoding family
record storage bits
logical elements
interleave rows
作者显式 pack axis
```

并沿 slice、admit、stage handoff 与 materialize 后的普通 use-def 传播同一 storage identity。局部 encoded pack 的 rows 来自 downstream physical width，而不是格式名；dense pack 没有被错误要求必须带 encoding declaration。

IQ2 blocked 当前 memory 结果包括：

- `scalar-indexed` encoded access：2 个；
- `unit-stride-interleaved` encoded access：16 个；
- `unit-stride-lookup-window`：4 个。

这比“所有 canonical encoding 一律 scalar-indexed”更宽，但没有消除全部 indexed path。尤其 standalone IQ2 std 树没有 output/codebook shaped axis，storage pass 无法从八份标量操作中合法形成共享 vector window。

### 4.3 Loop-local cluster 与真实跨迭代流水

`ScheduleRISCVLevels` 从普通 scalar `for` 内的 use-def 向后追踪 pure producer，要求：

- producer 中存在 load/extract/lookup/admit；
- consumer 使用 producer frontier；
- commit/materialize/pack 等 effect 不被跨越；
- frontier 与 loop-carried state 的 lifetime 可知。

depth 2 现在会让 emitter 生成：

```text
prologue: load/decode current frontier
steady:  load/decode next frontier → compute current → rotate
epilogue: compute final frontier
```

生成 C 中可以直接看到 `pipeline_current_*`、`pipeline_next_*` 和独立 steady loop。frontier 当前 bank 已属于普通 value liveness，resource pass 额外计入 next bank；因此不是把同一对 buffer 重复计成四份。

参数 A/B 结果：

| kernel / target | depth 1 GOP/s | depth 2 GOP/s | 结果 |
|---|---:|---:|---|
| Q4_0 vec-dot / SG | 0.754196 | 0.541361 | depth 2 慢 28.2% |
| Q4_0 vec-dot / K1 | 0.289974 | 0.241144 | depth 2 慢 16.8% |
| IQ2_XXS vec-dot / SG | 0.131265 | 0.204752 | depth 2 快 56.0% |
| IQ2_XXS vec-dot / K1 | 0.061229 | 0.088407 | depth 2 快 44.4% |

这些结果证明 pipeline 参数影响真实生成结构，也证明“存在 depth 2”不等于它普遍更快。tuner 现在忽略数值失败 candidate，接受 `within-tolerance`，并用真实 metric 排序后重新运行 winner；Q4_0 选择 depth 1，IQ2 选择 depth 2。

## 五、资源与 spec 2.2 边界

### 5.1 已增加的资源处理

资源 pass 现在可以把以下纯 computed aggregate 改为 per-register-part rematerialization：

```text
iota / extract / lookup
unary / binary
cast / widen / narrow
```

可重读 admitted value 仍可变成 reload-per-use。pipeline frontier 不参与这两种降压，因为它必须跨迭代保存。

### 5.2 IQ2 blocked 的具体阻塞

IQ2 blocked 的作者树在每个 group 产生 `entry=4 × codebook=8` partial，并在八个 group 之后才 reduce。depth 1 在 SG 当前可落到 29/32 vector groups；depth 2 需要再保留一套 21-group frontier，候选需要 42/32，因而被正确判非法。K1 同类候选需要 41/32。

把 entry/codebook reduction 移进 group 可以显著缩短 live interval，但这会改变：

```text
logical partial value 的数量
reduction 所属 Level
跨 group carry 的类型与生命周期
```

因此它不是 layout conversion、spill 或 scheduler 可以自行完成的表示变化，而是 spec 2.2 所定义的作者树变化。本轮在这里停止，没有让 backend 暗中改树。

任意 non-pure computed value 的 stack spill 仍未实现；若未来出现必须保留语义但寄存器不足的 candidate，目前会明确 invalid，不会回退到隐藏 scalar path。

## 六、双机编译与测量口径

两台机器的 Weft kernel、runtime wrapper 和 GGML wrapper 都使用 Clang 18、`-O3`、`-ffp-contract=fast`、`lp64d`。GGML library 的 CMake cache 也确认：

- SG2044：`/opt/tcrv-toolchains/llvm-18.1.8/bin/clang{,++}`；
- K1：`/usr/bin/clang{,++}-18`。

SG 的 Clang 必须配合：

```text
--gcc-toolchain=/opt/tcrv-toolchains/gcc-15.2.0
-B/opt/tcrv-toolchains/binutils-2.46.1/bin
-fno-integrated-as
```

其中 `-B` 是必要条件；仅指定 GCC toolchain 不会让 Clang 使用支持当前 RVV ISA 的 GNU assembler。

计时都是同 shape 的单 kernel cold protocol、`repetitions=3`。本轮 live GGML 数字是用上述同轮 Clang 口径读取的比较快照；固定全量 baseline CSV 没有被部分覆盖成混合口径。

## 七、本轮双机结果

### 7.1 MUL_MAT 与 dense

| kernel / phase | target | Weft GOP/s | 同轮 GGML GOP/s | ratio |
|---|---|---:|---:|---:|
| Q4_K I16 / decode | SG | 9.102575 | 9.390173 | 96.94% |
| Q4_K I16 / prefill | SG | 10.003027 | 9.624393 | 103.93% |
| Q4_K I16 / decode | K1 | 3.127684 | 2.454186 | 127.44% |
| Q4_K I16 / prefill | K1 | 4.129769 | 2.513934 | 164.28% |
| Q4_0 local pack / decode | SG | 1.427457 | 7.423087 | 19.23% |
| Q4_0 local pack / decode | K1 | 0.442279 | 2.200878 | 20.10% |
| Q2_K canonical / decode | SG | 0.880328 | 9.203091 | 9.57% |
| Q2_K canonical / prefill | SG | 0.879756 | 9.643044 | 9.12% |
| Q2_K canonical / decode | K1 | 0.282354 | 2.449528 | 11.53% |
| Q2_K canonical / prefill | K1 | 0.282541 | 2.451342 | 11.53% |
| IQ2 local pack / decode | SG | 0.677998 | 3.997222 | 16.96% |
| IQ2 local pack / prefill | SG | 0.715108 | 4.110204 | 17.40% |
| IQ2 local pack / decode | K1 | 0.424408 | 1.665004 | 25.49% |
| IQ2 local pack / prefill | K1 | 0.453822 | 1.707583 | 26.58% |
| F32 / prefill | SG | 5.354495 | 7.318487 | 73.16% |
| F32 / prefill | K1 | 1.874143 | 2.635056 | 71.12% |

Q4_K 的 tuning winner 随目标变化：SG prefill 为 unroll 4/depth 1，K1 为 unroll 4/depth 2。K1 并非只把 lane 数从 128 改成 256后复用固定 meta。

### 7.2 Vec-dot

| kernel | target | Weft GOP/s | 同轮 GGML GOP/s | ratio |
|---|---|---:|---:|---:|
| Q4_0 × Q8_0 | SG | 0.756869 | 7.533400 | 10.05% |
| Q4_0 × Q8_0 | K1 | 0.289974 | 2.225180 | 13.03% |
| IQ2_XXS × Q8_K | SG | 0.204752 | 4.059860 | 5.04% |
| IQ2_XXS × Q8_K | K1 | 0.088407 | 1.683167 | 5.25% |

generic pipeline 对 IQ2 的相对收益真实存在，但不能掩盖绝对形态仍然错误：standalone IQ2 std 树没有表达 codebook lane 与 output register cohort，pass 不能从标量展开恢复它们。

### 7.3 同口径局部 A/B

IQ2 blocked 在同一 SG Clang 18 环境中，加入当前 use-edge conversion、storage propagation、rematerialization 与 scheduling 后：

```text
0.508873 → 0.715108 GOP/s，+40.5%
```

这说明本轮改变不是单纯目录拆分；但 0.715108 仍只有同轮 GGML 的 17.40%，所以它也没有证明跨输出共享已完整解决。

## 八、当前能力边界

### 已由不同输入证明的部分

- 同一 conversion pass 在 identity、project、scalar broadcast 和 lane-to-register edge 上产生不同结果；
- 同一 scheduler 在 Q4_0 上选择 sequential，在 IQ2 standalone 上选择并实际生成 double-buffer；
- 同一 storage pass 同时处理 dense pack、canonical encoded local pack 和 derived interleave；
- 同一 Q4_K 作者树在 VLEN128/VLEN256 上得到不同 measured winner，并保持数值有效和高性能。

### 尚未成立的部分

- conversion 尚未覆盖所有 use edge，也没有独立 conversion elimination/canonicalization；
- codebook/index window 尚未普遍成为跨 output consumer 的共享 physical value；
- canonical Q2/Q4/IQ 的 memory、lookup、register cohort 和 cluster 仍然窄，双机多数只有手写的 5%–27%；
- non-pure computed spill 尚未实现；
- 42 份缺显式逻辑轴的作者树未迁移，后端按语义不能替它们恢复轴；
- K1 上 VLEN 变化会改变 lane count、LMUL 与合法资源，但 memory relation 本身相同的输入不会仅因 VLEN 不同而凭空换成另一种 memory form；本轮没有制造这种无事实依据的 target 分支。

因此本轮的诚实结论是：表示冲突、encoded storage 和局部流水已经从 emitter 私有判断前移成可观察的实体级 pass 结果，并在不同输入上产生了不同代码与性能；但它们只关闭了横向缺口的一部分。当前仍不能由 canonical Q/IQ tree 普遍生成接近手写 intrinsic 的跨输出共享微内核。

## 九、可手动复现的入口

```bash
# Q4_K 回归与 measured winner
WEFT_TUNE_RUNNER=mul-mat \
WEFT_TUNE_UNROLLS=1,2,4 WEFT_TUNE_PIPELINE_DEPTHS=1,2 \
examples/run/weft-kernel-tune.sh sg2044 q4_k_i16:prefill 3

# generic pipeline A/B
WEFT_TUNE_RUNNER=vec-dot \
WEFT_TUNE_UNROLLS=1 WEFT_TUNE_PIPELINE_DEPTHS=1,2 \
examples/run/weft-kernel-tune.sh sg2044 q4_0 3

WEFT_TUNE_RUNNER=vec-dot \
WEFT_TUNE_UNROLLS=1 WEFT_TUNE_PIPELINE_DEPTHS=1,2 \
examples/run/weft-kernel-tune.sh sg2044 iq2_xxs 3

# blocked MUL_MAT 双机
examples/run/weft-mul-mat.sh sg2044 iq2_xxs_local prefill 3
examples/run/weft-mul-mat.sh k1 iq2_xxs_local prefill 3

# dense 双机
examples/run/weft-kernel.sh sg2044 gemm_f32 3
examples/run/weft-kernel.sh k1 gemm_f32 3
```
