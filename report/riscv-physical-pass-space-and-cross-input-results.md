# RISC-V physical pass 空间与跨输入结果

## 结论

这一轮没有用“pass 已拆开”证明编译器已经泛化，而是把结构不同的现有程序送进同一主干。
结果分成三类。

第一类真正改善了：

- F32 blocked GEMM 从错误的 M-lane 改为 N-lane，dense load/store 由完整轴坐标生成；
  SG2044 的 MR4×NR16 为 6.665204 GOP/s，是同目标 source 的 94.60%；K1 的
  MR4×NR32 为 2.479803 GOP/s，是 source 的 94.10%。
- IQ2_XXS local-pack 没有改作者 tree；SG2044 decode 从 0.553423 提到
  0.779411 GOP/s，prefill 从 1.309321 提到 2.001647 GOP/s。
- Q4K_I16 仍走原 grouped/layered 路径并保持 bit-exact：decode 9.389858、
  prefill 10.961397 GOP/s。也就是说，新表示推导没有破坏已经成熟的 packed 路径。

第二类直接否定了当前 winner：F32 的 MR/NR 形成 8 个 concrete candidate，SG2044
资源检查允许其中 7 个，但静态 `min(register peak)` 选择 MR2×NR4；真机更快的是
MR4×NR16。K1 的真机更快点又变成 MR4×NR32。当前确实有多个候选，但编译器内部的
winner 不是 compile-and-measure。

第三类没有走通，因而不能声称“所有 operation family 已横向拉齐”：

- `online_flash_attention` 在 dense `materialize(admit(...))` 处被 operation pass 判非法；
- `top_k_f32` 的有序标量程序能形成完整 assignment，但 emitter 尚不能发射
  sequential aggregate `new/update/while`；
- 较早的 `q4_k_gemv` 手工 repro 能生成和运行，却出现
  `expected=12384.4346, actual=12384.4219`，所以没有把它算作数值通过。

因此，本轮证明的是 representation/memory/resource 主干对 dense、grouped packed、
codebook 三种输入有真实横向进展；它没有证明整个编译器空间完成。layout conflict、
measured winner、attention staging 和普通标量 aggregate 仍是代码中的实际缺口。

## 对照参考实现后采用的组织

### Triton：传播、冲突和 conversion 是三件不同的事

[`RemoveLayoutConversions.cpp`](/home/kingdom/phdworks/ref/triton/lib/Dialect/TritonGPU/Transforms/RemoveLayoutConversions.cpp:42)
明确给出四段流程：初始化 anchor、向 users 传播、解决多个 encoding 的冲突并插
`ConvertLayoutOp`、按 dominance 重写。它不是“给每个 value 取第一种合法 layout”。

同一文件的 `LayoutPropagation::initAnchorLayout` 在
[`218`](/home/kingdom/phdworks/ref/triton/lib/Dialect/TritonGPU/Transforms/RemoveLayoutConversions.cpp:218)
为参数、昂贵 load/store、dot、atomic、有效 gather/reshape 建 anchor；
`propagateToUsers` 在
[`283`](/home/kingdom/phdworks/ref/triton/lib/Dialect/TritonGPU/Transforms/RemoveLayoutConversions.cpp:283)
处理 loop carry、yield、elementwise、reduce 和 shape op；冲突处理在
[`375`](/home/kingdom/phdworks/ref/triton/lib/Dialect/TritonGPU/Transforms/RemoveLayoutConversions.cpp:375)
优先 load/store 的 blocked layout 或 MMA layout，否则用每线程元素数作压力近似。
`getValueAs` 在
[`491`](/home/kingdom/phdworks/ref/triton/lib/Dialect/TritonGPU/Transforms/RemoveLayoutConversions.cpp:491)
才真正插 conversion。

这给 Weft 的直接约束是：memory 可以是 mapping anchor；沿 use-def 传播可以在
representation pass 中做；但 consumer 冲突不能靠“整个连通分量共用一个 lane axis”冒充
conversion。当前前两件已开始存在，第三件仍缺。

Triton 的 allocation 也不是 emitter 猜预算：
[`Allocation.cpp:197`](/home/kingdom/phdworks/ref/triton/lib/Analysis/Allocation.cpp:197)
先收集 typed size，再做 liveness，最后分配 offset；alias 在
[`303`](/home/kingdom/phdworks/ref/triton/lib/Analysis/Allocation.cpp:303) 合并到同一 allocation。
这对应本轮把 initializer/new 和 deferred outer/add 的资源所有权显式化。

### TileLang：layout 是映射，pipeline 读取真实依赖

TileLang 的 `Layout` 直接保存 logical index 到 forward index 的映射，见
[`layout.py:12`](/home/kingdom/phdworks/ref/tilelang/tilelang/layout/layout.py:12)；
`repeat/expand/inverse/reshape` 都是显式 layout 变换，不是格式名分支。

layout inference 对同 storage alias 传播 layout；shape/element width 改变时做 reshape，
冲突则 merge 或明确失败，见
[`layout_inference.cc:162`](/home/kingdom/phdworks/ref/tilelang/src/transform/layout_inference/layout_inference.cc:162)
和 [`251`](/home/kingdom/phdworks/ref/tilelang/src/transform/layout_inference/layout_inference.cc:251)。

pipeline planning 保存每个 stage 的 read/write region、scalar def/use、原顺序、stage、
条件执行和最后 consumer，见
[`pipeline_planning.cc:406`](/home/kingdom/phdworks/ref/tilelang/src/transform/pipeline_planning.cc:406)。
所以 depth=2 只有在 dependency plan 真正生成 prologue/steady/epilogue 时才是第二种实现；
一个数字字段本身不算 pipeline。

TileLang autotuner 只枚举调用者提供的 config，见
[`tuner.py:994`](/home/kingdom/phdworks/ref/tilelang/tilelang/autotuner/tuner.py:994)；
它不会从 IR 猜作者 tree。这与 Weft 的 spec 2.2 边界一致。

## 实际修改

### 1. representation 读取完整 use-def 和 memory affinity

`AssignRISCVRepresentations` 先收集全部 producer/user/handoff，再建立本次 lowering 内的
use-def 邻接关系。wide op 的 free-axis 平局现在按以下事实决定：

```text
显式 reduce/contract 轴关系
→ 与该 op 的 value chain 相连的 memory layout
→ dense 最末轴或 packed.along.<axis> 的连续性
→ 所在 Level 的 cohort
```

F32 GEMM 的 `[M,N]` result 在 MR=NR=4 时，原实现因遍历顺序选择 M-lane；本轮选择
N-lane。这个判断随后传播到 admitted values、accumulator 和 commit，而不是由每个 emitter
分支各选一次。

reduce、dot/contract/outer-contract 的被消去轴成为 operation-local `eliminated_axis`。
free axes 以有序 `register_axes/register_extents` 保存，`register_parts` 是各 extent 的乘积。
这使“消去 K、保留 M/N”不再依赖 result rank 的偶然顺序。

### 2. memory form 由 lane 与声明布局共同确定

`SelectRISCVLocalOperations` 对每条 transfer edge 产生：

- dense 最末轴或 `packed.along.<axis>` 与 lane 一致：`unit-stride`；
- dense 的其他轴：`runtime-strided`；
- encoded record：`record-address`；
- 标量：`scalar-address`。

dense outer-contract 也在 operation pass 中得到 `lane_operand`、`lane_memory_form` 和一个
lane operand vector 的 temporary budget。emitter 不再查看 C stride 后临时决定 vle/vlse。

### 3. emitter 按轴坐标投影，不再按扁平 part 下标碰运气

本轮审查发现，先前虽然 assignment 已有 `register_axes`，但下列路径仍在做
`zip(parts)` 或 `part / streams`：Level/for/if carry、cast、widen、reduce、lookup、dense
load/store 和 outer-contract。这会在轴顺序改变但 part 数恰好相同时静默交换数据。

现在统一使用两种投影：

```text
projectPart(source, result, result_part)
projectRegisterPart(source, result, result_register_part)
```

dense load/store 从所有 register axis 的坐标形成地址；outer-contract 对 lane operand 的
相同坐标窗口只 load 一次，再供多个 accumulator 使用。若 source/result 的 lane mapping
无法由坐标投影，emitter 明确报“需要 physical layout conversion”，不再原序复制。

encoded outer MAC/Fold 当前只真正实现一个 accumulator axis；遇到多 register axis 会明确
拒绝，而不是把扁平 part 误当该轴坐标。

### 4. accumulator 和 primitive temporary 进入资源事实

单 consumer 且 mapping 相同的：

```text
outer_contract → add(acc, product)
```

被选择为 deferred product + accumulator update；product 不单独物化。一次使用且 mapping
相同的 `admit → new` 形成 tied initialization。resource pass 将 tied values 放入同一
resource class，但仍按该 class 的最大实际 groups 计数。

dense outer FMA 的 lane operand vector作为 operation temporary 单独计数。SG2044 的
MR4×NR32 因 32 个 accumulator groups、lane temporary 和保留组继续非法；没有在 emitter
里偷偷缩小 microtile。

## 每个 pass 当前实际在做什么

这里的“结果数”区分 outer candidate enumeration 与 candidate-local 确定推导。不同输入
经过规则得到不同结果，不等于同一个 candidate 内有多个可选实现。

| pass | 读取 | 写入 | 当前真实结果空间 |
|---|---|---|---|
| ConstructRISCVProblems | canonical tree、`auto` ranges、target、Level | concrete Problem | `Π(auto choices) × Π(Level unroll choices × pipeline choices)`；F32 MR={2,4}、NR={4,8,16,32} 得到 8 个 candidate |
| AssignRISCVRepresentations | axes、Level path、use-def/handoff、memory layout、SEW、VLEN | lane、register tuple、SEW/LMUL/vl、materialization | 每 candidate 1 个确定结果；跨输入可产生不同 lane/register/stream 结果；LMUL仍是第一个能容纳当前 lane extent 的合法值 |
| SelectRISCVLocalOperations | typed op、mapping、encoding、use count、target flags | realization、memory edge、tied/deferred cluster | 每 op 每 candidate 1 个结果；本轮实际出现 unit/runtime-strided、record/scalar address、tied/untied、deferred/materialized；matrix fragment仍取第一个合法 fragment |
| ScheduleRISCVLevels | Level、candidate schedule、local temporary facts | level mapping、unroll、pipeline structure | depth 1=`sequential-stream`，depth 2=`cross-iteration-double-buffer`；只有这两种；不是所有 Level 都有第二种 |
| CheckRISCVResources | live interval、resource class、temporary、fragment、target budget | legal、reload admitted value、invalid | 确定地产生 3 类结果；只对可重读 admit 做 reload，computed value 超预算仍 invalid |
| SelectRISCVWinner | 所有 legal Problem 的静态 cost | 1 个 assignment | 比较项数=legal candidate 数；当前只按 register peak，真机已反证其排序 |
| intrinsic-C emitter | frozen assignment | C/局部 asm | 0 个选择；无法投影的 layout 明确失败 |

仍然属于假选择或单实现的部分：

1. LMUL 没有 candidate alternative；只是按 target legal list 取第一个足够值。
2. 大多数 op 的 instruction domain 只有一个 realization；IME 只有 M1×N16×K32 fragment。
3. layout conflict 尚无 transient conversion insertion/elimination。
4. winner 不是实测。
5. pipeline 只有 depth 1/2；attention 所需的 dense staging 甚至尚未进入该 scheduler。
6. resource spill 只支持重读 admitted value，computed aggregate 没有 spill。

## 外部运行结果

所有列为本轮真实手工 repro：DSL → Kernel IR → 当前六个 RISC-V pass → intrinsic C →
目标 C 编译器 → 真机。Weft 数值均为 bit-exact 才记性能。source 数字沿用 CSV 中同硬件、同 shape
的既有 baseline；source 使用其原 warmup 口径，本轮新增 Weft concrete/local 行为 cold、
`repetitions=3,warmup=0`，所以 ratio 用于同目标数量级比较，不伪装成同一次配对重测。

### SG2044 / VLEN128

| kernel / path | 修改前 Weft | 本轮最终 Weft | source | 最终/source | 结果 |
|---|---:|---:|---:|---:|---|
| IQ2_XXS local decode | 0.553423 | 0.779411 | 4.067610 | 19.16% | +40.8%，bit-exact |
| IQ2_XXS local prefill | 1.309321 | 2.001647 | 4.239853 | 47.21% | +52.9%，bit-exact |
| Q4K_I16 persistent decode | 9.388334 | 9.389858 | 9.555431 | 98.27% | 基本不变，bit-exact |
| Q4K_I16 persistent prefill | 11.108464 | 10.961397 | 9.720730 | 112.76% | -1.3%，bit-exact |
| F32 prefill MR4×NR16 | 6.881550（本轮较早 emitter） | 6.665204 | 7.045805 | 94.60% | 完整轴地址与 temporary 计数后的最终结果 |

F32 的结构候选在本轮较早阶段还实测过：MR4×NR4=5.162587、MR4×NR8=6.151110、
MR4×NR16=6.881550、MR2×NR32=5.456639；MR4×NR32 因资源超预算没有性能数字。
这些点用于反证静态 winner，没有作为完整 candidate matrix 写进 CSV。

### K1 / VLEN256

| kernel / path | 修改前 Weft | 本轮最终 Weft | source | 最终/source | 结果 |
|---|---:|---:|---:|---:|---|
| IQ2_XXS local decode | 0.509911 | 0.510604 | 1.664643 | 30.67% | 基本不变，bit-exact |
| IQ2_XXS local prefill | 1.178748 | 1.182625 | 1.705884 | 69.33% | 基本不变，bit-exact |
| F32 prefill MR4×NR32 | 2.344175（本轮较早 emitter） | 2.479803 | 2.635297 | 94.10% | +5.8%，bit-exact |

SG 与 K1 的实测优选 NR 不同，证明 target facts 会改变有效 concrete binding；也证明
“最小寄存器峰值”不是足够的 winner。

## 负向输入

负向结果是本轮判断 pass 是否为“假泛化”的主要证据。

| 输入 | 停止位置 | 事实 |
|---|---|---|
| `online_flash_attention`, BQ=4/BK=32 | SelectLocalOperations | dense `materialize(admit(...))` 只接受显式 pack request；attention 的作者 staging 尚无 realization |
| `top_k_f32` | intrinsic-C emitter | assignment 已 complete；`new(f32,[K_])` 被标成 sequential aggregate，但 emitter 只实现 scalar/vector state，且 `update/while` 也无完整 aggregate emission |
| `q4_k_gemv` / `q4_k_gemv_groups4` | 真机数值 | 都得到同一 row-0 mismatch；生产 Q4K_I16 路径仍 bit-exact，因此不能把旧 GEMV repro 当作通过证据 |

这些失败没有通过 fallback 或格式分支掩盖，也没有被写成性能数字。

## CSV 处理

`report/weft-kernel-performance.csv`：

- 保留原 canonical F32 SG/K1 行，不再用 concrete MR/NR 覆盖历史 canonical 记录；
- 追加独立 scope 的 F32 MR4×NR16（SG）和 MR4×NR32（K1）行；
- 更新本轮最终 Q4K_I16 与 IQ2_XXS local 双机数字；
- concrete/local Weft 行明确写 `repetitions=3,warmup=0`；source 列是既有 baseline，
  没有声称共同 warmup 重测。

没有增加 CSV 校验、阈值或同步逻辑。

## spec 2.2

本轮没有发现“必须让编译器改变作者逻辑值集合或 Level 归属才能获得已记录提升”的证据：

- F32 的 NC/KC/MC、MR/NR、packing 和 accumulator scope 没变；只改变轴到 lane/register 的
  表示、memory form 和 handoff。
- Q4_K grouped/layered tree 没变。
- IQ2_XXS local tree 没变；它仍受作者写下的 8 次有序 scalar codebook loop 限制，
  compiler 没有把它偷换成 VLA reduction。

因此 spec 2.2 未被本轮证伪。attention 和 Top-K 的失败是缺 operation realization/emission，
不是要求 compiler 改 tree；旧 q4 GEMV 的数值差异尚未完成归因，不能拿来判断 2.2。

## 卡住的地方

两个缺口已经被外部反例钉住且位于编译空间核心：

1. 不兼容 consumer layout 的 conversion insertion/elimination 不存在。当前 use-def 图传播只在
   可共享一个 mapping 时有效，尚未达到 Triton 的 anchor/propagate/conflict/convert 结构。
2. measured winner 未接入。静态 resource cost 在 SG 与 K1 上都选错 concrete dense candidate。

另外，attention staging、Top-K sequential aggregate、旧 q4 GEMV 数值差异是独立的 operation/
emission 完整性缺口。它们说明当前主干不能被描述成“全部 kernel 形态完整”，但没有要求修改
作者 tree，也没有迫使后端增加 whole-kernel route。
