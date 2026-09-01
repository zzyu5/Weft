# VLEN-independent 作者树与 vec-dot/decode 物理化报告

日期：2026-08-31

## 1. 结论

本轮把两类此前被 target/VLEN 掩盖的问题拆开处理：

1. **作者程序不再由 VLEN 选择。** IQ2_XS 删除 SG2044 的 scalar Python 展开入口，standalone
   vec-dot 与 MUL_MAT decode 现在共同调用一棵带 `scale_group × entry × payload` 逻辑轴的
   canonical tree。Q2_K 的两棵不同数值树完成双 target 交叉运行后，`group_reduced` tree 在
   SG2044 和 K1 都胜出；另一棵 tree、runtime define 与 target entry 分支已删除。
2. **VLEN 差异进入 physical program。** 同一 IQ2_XS tree 在 SG2044/VLEN128 上形成
   `4 × 2 × 8` issue window，在 K1/VLEN256 上形成 `16 × 2 × 8` window；source-level logical
   axes、reduction 和 ABI 不变。
3. **vec-dot 与 MUL_MAT decode 的公共底座得到真实验证。** IQ2_XS、Q1_0、Q6_K 的
   standalone 与 production decode 在同一 target 上同步变化，核心 physical-op 计数一致。
4. **IQ2_XS 仍未超过 source。** 最终正式结果为 SG2044 source 的 `72.6%/72.1%`，K1
   source 的 `78.1%/80.1%`（standalone/decode）。剩余差距已定位到 widened product 的载体
   和 final reduction topology，不是 GEMV wrapper，也不是 target 选择了另一棵作者 tree。

本轮没有跑 206 条全量，也没有修改 IQ1/IQ3/TQ1 prefill tree。

## 2. 第九节七步诊断记录

### 2.1 现象与基线

IQ2_XS 的两份旧 source program 数学和 ABI 相同，但 SG2044 使用 Python 展开的 scalar
entry tree，K1 使用显式 entry/payload axes。旧正式值是：

| 路径 | SG2044 | K1/X60 |
|---|---:|---:|
| standalone vec-dot | 0.692936 | 0.300673 |
| MUL_MAT decode | 0.736263 | 0.599084 |

这不是算法 variant：table bytes、weight/activation encoding、C ABI、blocking、staging 和
persistent artifact 全部相同。差异只在 canonical IR 是否保存逻辑轴。因此 scalar tree 是
信息缺失的写法，不是应由调用方长期保留的 target specialization。

Q2_K 的两棵 tree 则确实不同：一棵先形成 `q2 × q8` group partial 再乘 scale；另一棵把
scale 放进 reduction。它们改变中间 value、widening 和整数结合位置，不能由 physical pass
互换。本轮的交叉运行结论是 `group_reduced` tree 在两个 target 都胜出，所以调用方无需保留
两棵程序。正式保存的唯一 tree 结果见第 6 节；落败候选的临时运行数字没有写入 CSV，不能
把它们伪装成可长期引用的正式数据。

### 2.2 决策 owner

本轮采用的边界是：

- logical axis 是否存在、选择哪个 std function、reduction 结构和 Value/Level 集合归作者；
- LMUL、VL、lane/time 分解、memory instruction、partial topology、pipeline 和 RVV/IME
  realization 归 target lowering；
- MR/NR/unroll 等同一 tree 的参数性实例可以由实测 tuner 选择。

因此 IQ2_XS 只有一棵 shaped tree；SG 与 K1 的差异必须出现在 RISC-V IR。Q2_K 先比较两棵
作者程序，但最终选择也不能写成 `VLEN == 128` 的 source 条件：结果既然是一棵在两机都赢，
另一棵就直接删除。

这一边界已写入 `doc/compiler/optimization-principles.md`。

### 2.3 修改前的静态工作量

旧 SG scalar tree 把 8 个 group、每组 4 个 entry、每个 entry 的 8 个 payload 写成 Python
循环展开。canonical IR 中不存在统一的 entry/payload shaped value，因而也不存在可供 pass
消费的 typed unit-entry window、indexed-entry load 或 issue-time topology。编译器不能从
展开后的 SSA closure 猜回轴；普通 Python/scalar control 不被自动向量化是正确行为。

旧 typed entry path 还暴露两个 physical contract 缺口：

- `RVVUnitEntryWindowLoadOp` 只接受单个 full-lane window，不能表示同一 logical axis 的
  time-striped issue；
- regular-repeat scale access 把 storage source axis 与 logical reduction axis 混为一个轴，
  动态 issue base 只能由 emitter 临时重建。

### 2.4 修改后的 typed physical 计数

同一 standalone IQ2_XS tree 在最终 RISC-V IR 中的计数如下：

| target | unit-entry window | indexed-entry load | regular-repeat gather | widen-dot | partial topology | reduce |
|---|---:|---:|---:|---:|---:|---:|
| SG2044/VLEN128 | 6 | 6 | 3 | 3 | 3 | 3 |
| K1/VLEN256 | 2 | 2 | 1 | 1 | 1 | 1 |

SG2044 的每个 typed window 是 `scale_group4 × entry2 × payload8`，外层 issue-time 覆盖完整
16 个 scale groups；K1 的 window 是 `scale_group16 × entry2 × payload8`，layout 内部以
time factor 处理 VLEN256 下的载体。standalone 与 MUL_MAT decode 的 contraction 部分计数完全
相同；MUL_MAT 只多出 final stream/reduce-store op。

最终 intrinsic C 中，table index 已经是 `u16`：SG 有 6 个、K1 有 4 个
`__riscv_vluxei16` 静态实例，`vluxei32` 为 0。scale source 和 scale-group result 分别由
`source_axis`、`reduction_axis` 描述；动态 `source_base` 是 typed scalar operand，不由 emitter
猜 issue offset。

### 2.5 资源与动态工作

改变是有可计数收益的，不是只增加 op 名称：

- SG standalone/decode 分别从 `0.693/0.736` 提升到 `2.530/2.517` GOP/s；
- K1 standalone/decode 分别从 `0.301/0.599` 提升到 `1.393/1.409` GOP/s；
- scalar compatibility tree、target entry switch 和 flat byte-table ABI 已删除；
- Q6_K 暴露的非法 indexed-entry 结果没有用 fallback 吞掉：planner 现在只在 result carrier
  确实是 RVV value 时建立该 op；
- Q1_0 暴露的 direct loaded Record owner 由 bitmask op/verifier 正式接受，不再误判为非法
  Field origin。

不过当前 IQ2_XS 仍不是 donor 的 product organization。逐段对照生成 C 与
`source/c/ggml/llama.cpp/ggml/src/ggml-cpu/arch/riscv/quants.c:4270-4437` 后，差异已经可以计数：

| target/每个 donor issue | GGML donor | 当前 Weft |
|---|---|---|
| VLEN128，64 elements | 1 个 `i16m8` widened product，随后 4 个 16-lane reduction | 4 个 `i16m2` widened product，4 个 reduction，另有 3 个 `vslide1up` 组装结果 |
| VLEN256，128 elements | 1 个 `i16m8` widened product，随后 8 个 16-lane reduction | 16 个 `i16m2` widened product，16 个 reduction；生成 C 另有 30 个 `vslide1up`、16 个 i8 `vslidedown` 和 2 个 u16 `vslidedown` |

也就是说，entry/load/index 已经被物理化，但 product 在进入 reduction 前被过早切成 16-lane
slices。剩余缺口属于 P1（完整 product 骑在哪个 carrier）和 P4（何时切片/收敛），并受联合
resource contract 约束。它不是再改 LMUL 就能解决的问题，也不是 MUL_MAT 外层遍历造成的。

### 2.6 双入口、双 target 验证

正式 10-repetition 结果：

| format | target | standalone | source | ratio | MUL decode | source | ratio |
|---|---|---:|---:|---:|---:|---:|---:|
| IQ2_XS | SG2044 | 2.530066 | 3.482858 | 72.64% | 2.516819 | 3.489556 | 72.12% |
| IQ2_XS | K1/X60 | 1.392501 | 1.783955 | 78.06% | 1.408712 | 1.759671 | 80.06% |
| Q1_0 | SG2044 | 4.402068 | 2.264629 | 194.38% | 4.695073 | 2.333519 | 201.20% |
| Q1_0 | K1/X60 | 2.797600 | 3.772674 | 74.15% | 2.808747 | 3.696397 | 75.99% |
| Q6_K | SG2044 | 5.977505 | 4.895281 | 122.11% | 5.510133 | 4.851051 | 113.59% |
| Q6_K | K1/X60 | 2.983024 | 2.384436 | 125.10% | 2.974804 | 2.332811 | 127.52% |

IQ2_XS 的 SG standalone 第一次 10-repetition 运行曾得到 1.808 GOP/s；紧接着同一代码、同一
命令复跑为 2.530，且与 decode 2.517 对齐。1.808 被判为目标机瞬时干扰，没有写入 CSV。

### 2.7 donor 与参考编译器对照

对 GGML donor 的横向归纳没有得到“每个格式一条特殊路径”，而是下列共同关系：

- Q4_0：packed byte unit load，low/high nibble 形成 shaped lanes，widened product 后才 reduce；
- Q5_K：base plane 与 high plane 共同形成一个 joined payload；
- IQ2_XS：u16 entry code 产生 8-byte grid/sign indexed load，完整 product 保留后按 16-element
  scale group reduction；
- TQ2_0：radix/plane 先形成 independent partial，再执行 final topology；
- Q8_0：普通 contiguous signed-byte supply，不需要 packed decode。

这五类分别压住 P3 storage geometry、P2 producer reuse、P1 carrier placement 和 P4 delayed
convergence；它们不能由格式名分支代替。

Triton 的 `OptimizeDotOperands.cpp:181-220` 会沿 dot sink 已确定的 memdesc/layout 向 producer
回传，并把 local allocation 与 view chain 改写成真实 IR；Weft 不能直接复用 GPU memdesc，
但同样需要 producer、window、load 与 consumer 在 physical IR 中成为可验证的关系。
Triton `PipelineExpander.h:79-102` 明确把 schedule 决策与 prologue/steady/epilogue 的机械展开
分开。TileLang `pipeline_planning.cc:271-309` 先收集 buffer access region，
`inject_pipeline.cc:1195-1229` 再计算 buffer versions 并展开 pipeline。这些参照说明：issue-time
window 和 lifetime 必须先是 physical program；不能等 emitter 看源码 closure 时临时组织。

## 3. 全部 standalone/decode 配对审计

当前 CSV 中 24 个格式 × 2 targets 共 48 对 standalone vec-dot/MUL_MAT decode：

- 45/48 对的吞吐差异在 ±10% 内；
- IQ2_XS、Q1_0、Q6_K 本轮重新测量的 6 对全部同步；
- 三个历史 outlier 是 MXFP4 双 target（MUL decode 比 standalone 高约 143%/154%）和 K1
  TQ2_0（约 +10.7%）。源码上 MXFP4 decode 调用的仍是同一个 `vec_dot_mxfp4_q8_0`
  helper，因此这两行首先说明 CSV 来自不同编译器快照或运行状态，不足以证明 GEMV wrapper
  有独立优化。本轮没有为消灭统计 outlier 重跑未受影响格式。

所以可以对本轮三个重新测量格式作出硬结论：GEMV outer traversal、store 和 ABI 不是主要
瓶颈；不能把尚未同状态复测的 48 对全部宣称为现时硬结论。

## 4. 实际代码改动

### 4.1 作者 tree 与入口

- IQ2_XS standalone 与 MUL decode 共享 `_iq2_xs_entry_reduce`；
- table View 使用 `I8X8[entry, payload]`，不再把相同 8-byte record展平为 byte table；
- code/index/payload 使用 u16，activation address 在需要时显式 widen 到 u32；
- 删除 SG scalar compatibility path；两台 target 使用同一个 canonical entry；
- 删除 Q2_K 的 alternate tree、entry、runtime define 和 VLEN source branch。

### 4.2 Physical IR 与 pass

- `RVVUnitEntryWindowLoadOp` 支持单一 logical axis 的 time-striped physical window；
- `RVVRegularRepeatGatherOp` 分离 storage `source_axis` 与 logical `reduction_axis`，并把动态
  `source_base` 变成 operand；
- memory planner 从 typed `FieldFacts` 读取 encoding geometry，而不是依赖 memory pass 前仍为
  `opaque` 的 access attribute；
- issue-window materialization 能克隆 iota、unit/indexed entry load、regular-repeat、broadcast、
  arithmetic/cast/lookup 等 producer chain；
- identity `convert_layout` 在 memory materialization 后显式删除；
- indexed-entry planner 要求 RVV result carrier，避免 Q6_K 构造 verifier 不合法的 op；
- bitmask decode verifier 与 fusion pass共同接受 direct loaded Record owner，修复 Q1_0。

### 4.3 Emitter

Emitter 只消费 `partOffset` 和 typed `sourceBase`。没有增加 format、kernel 或 VLEN 分支，也没有
在 terminal emission 中选择 issue topology、entry width 或 scale access form。

## 5. 负结果与被否定的假设

1. **只看 pre-memory access attribute 无效。** 该阶段 IQ2 field access 仍是 `opaque`，所以
   typed indexed-entry rule不会触发。最终改为读取 encoding-derived `FieldFacts`。
2. **只放宽 result guard 不够。** 它修复 Q6_K 的合法域，却不会自动给 IQ2 产生 issue-time
   window；P1 与 P3 必须同时闭合。
3. **初次直接把现有 u32 payload cast 成 u16 失败。** target selection没有合法 terminal
   cast leaf。最终 tree 从一开始把 payload logical axis定义为 u16，只在 activation address
   处 widen；没有增加 emitter cast fallback。
4. **standalone/decode wrapper 不是 IQ2 的主要差距。** 两份最终 IR 的 contraction op 计数相同，
   10-repetition 吞吐也同步。
5. **原预测不完整。** 预测表把 IQ2 主要归为 P2/P4。实际第一主因是 P1 issue-time carrier 与
   P3 typed entry/indexed load，P2 regular-repeat scale supply随后生效；P4 仍是当前未闭合的
   widened-product/final-reduction问题。这一项计为预测错误，不改解释迁就结果。
6. **Q2_K 的 target 作者特化假设被否定。** 两棵数值 tree 都能在两机运行，交叉结果选择同一
   tree；源码级 VLEN branch 被删除。

## 6. 当前正式 Q2_K 状态

`482c6a589` 后只有 `group_reduced` canonical tree。正式 10-repetition CSV：

| target | standalone vec-dot | MUL decode | MUL prefill |
|---|---:|---:|---:|
| SG2044 | 6.548500 | 6.611409 | 9.908940 |
| K1/X60 | 1.535602 | 1.531145 | 3.055156 |

这说明不同 VLEN 可以生成不同 lane/time、regular-repeat 和 partial topology，但不能成为选择
不同 std function 的理由。

## 7. 没有掩盖的边界

1. **IQ2_XS 尚未过线。** 剩余具体结构是 product fragmentation：donor 先形成一个宽 i16
   product，再切片 reduction；当前 Weft 对每个 16-lane slice 分别乘、reduce、再用 slide
   组装。这个决定需要 joint input/product/partial/output resource contract，不能由 emitter补。
2. **K1 Q1_0 尚未过线。** bitmask owner 修复使 standalone/decode同步到 source 的约
   `74%/76%`，但本轮没有继续改 leaf 拼写或 schedule。
3. **partial materializer 的 owner 仍过宽。** 当前 `MaterializeRISCVPartialAccumulators` 在
   `nested_scaled_stream` 路径中同时实例化 issue window、lane slices、partial topology 和
   widen-dot leaf。它没有把决定下沉到 emitter，但 topology selection 与 mechanical
   materialization 还没有完全分成两个 pass；不能据此宣称 decision owner 已彻底收口。
4. **48 对 CSV 有 3 个历史快照 outlier。** 本轮没有用未重测条目支撑现时性能结论。

## 8. 可复现命令

正式 paired repro：

```bash
./examples/run/weft-quantized-vec-dot.sh sg2044 iq2_xs 10
./examples/run/weft-mul-mat.sh sg2044 iq2_xs decode 10
./examples/run/weft-quantized-vec-dot.sh k1 iq2_xs 10
./examples/run/weft-mul-mat.sh k1 iq2_xs decode 10

./examples/run/weft-quantized-vec-dot.sh sg2044 q1_0 10
./examples/run/weft-mul-mat.sh sg2044 q1_0 decode 10
./examples/run/weft-quantized-vec-dot.sh k1 q1_0 10
./examples/run/weft-mul-mat.sh k1 q1_0 decode 10

./examples/run/weft-quantized-vec-dot.sh sg2044 q6_k 10
./examples/run/weft-mul-mat.sh sg2044 q6_k decode 10
./examples/run/weft-quantized-vec-dot.sh k1 q6_k 10
./examples/run/weft-mul-mat.sh k1 q6_k decode 10
```

当前 C++ 构建：

```bash
ninja -C build
```

构建完成；上述 12 条正式 repro 数值均为 `within-tolerance`。
