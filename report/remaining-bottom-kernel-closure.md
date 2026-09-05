# IQ1 reduction topology 与底部入口收口

日期：2026-09-05  
起点：`99d46d001`

本文只记录本轮的预测、实现、实测、负结果和确定边界。设计权威仍是 `doc/`。

## 1. 第九节工作账与实现前预测

### IQ1_M

IQ1_M 的 multi-axis topology 已在此前提交中闭合。当前 Physical IR 已经显式形成：

```text
partial_set
  -> partial_repack
  -> partial_reduce(entry, payload)
  -> partial_scale_combine(scale_part)
  -> partial_combine(group)
  -> partial_finalize
```

因此本轮没有再放宽 matcher。剩余 donor 差异是把 scale 融进 i32 lane accumulator、最后再
reduce；当前作者程序明确写的是先完成 `contract(entry,payload)`，再按 `scale_part` 和 `group`
reduce。把前者改成后者会改变 scale/reduction 结合位置和 widening 归属，属于作者数值程序，
不能由 Physical pass 偷改。

在不改数值图的范围内，静态扫描只找到一项：main/correction 两个 sibling issue loop 会分别
读取同一 Q8 activation window。SG 每 record 的动态 load 可从 4 次降到 2 次，K1 可从 8 次
降到 4 次；但 96 份入口里没有第二个同构 input。本轮没有为单个格式冻结 loop-fusion 合同。

### IQ1_S prefill

作者树已有 `entry=4`、`payload=8`，但八个 group 被写成串行 `L.subs` carry。实现前预测是：

- 显式写出 `group=8`，保持 `contract(entry,payload)` 后乘 scale、再 reduce group；
- pointwise broadcast 即使改变轴顺序，只要保留每条轴的 identity 和 extent，layout propagation
  仍应看到后续 group reduction；
- planner 必须冻结 lhs/rhs/scale 的 issue 物化次序和 scale 所处资源阶段；materializer 只消费；
- 数学 product 数、widening、Level、ABI 不变；可见收益来自较宽 product carrier 和较短 live
  overlap，而不是减少数学 MAC。

整数范围也在修改前核过：每组 32 个 `i8*i8` product 的绝对上界小于 516,128，乘最大 scale
15 后累加八组仍小于 62,000,000，落在 i32 内。把已有 group 从 Python/Level 串行写法改成
shaped reduction 不改变溢出结果。

参考实现提供的是职责边界，不是可以直接照搬的 RVV 规则：

- Triton 在 `AccelerateMatmul.cpp:464-484` 把结果 MMA encoding 与 DotOperandEncoding 写进真实
  IR；`DotOpToLLVM/FMA.cpp:18-44` 的 terminal lowering 只消费已经给定的 register values。
- TileLang 在 `inject_pipeline.cc:3095-3175` 用显式 stage/order 检查 producer-consumer
  dependency，再机械展开 schedule。
- Weft 没有 warp/MMA owner，必须从 logical axes、同一具体 Field 的 use-def、selected memory
  form、output replica 数和 RVV register budget 得到自己的 issue order 与 stage。

若 shaped group 没有减少 product 切片，或 stage/order 没有降低资源峰值，这些改动就不成立。

## 2. 实现

### 作者程序

`examples/kernels/mul_mat/iq1_s.py` 现在显式写出 `group=8`：main 与 correction 都先
`contract(entry,payload)`，再 `reduce(group)`。输入输出、table bytes、ABI 和 scale/reduction
结合位置未变。

### layout propagation

`PropagateRISCVLayouts` 不再要求 pointwise 前后 shape/axis 顺序逐项相同；它按 axis identity
检查原有轴及 extent 是否完整保留。indexed lookup 的展开判定也从“类型只有一条轴”改成
“恰有一条非 singleton 物理轴”，singleton output axes 不再挡住事实传播。

`RVVWidenDotOp` verifier 同时要求 nested plan 与 partial topology 的 output-axis 顺序一致，
并要求二者的 axis 集合等于结果剩余 free axes。它允许等价的轴排列，但不允许 plan 与
topology 各自采用不同排列。

### typed issue schedule

`NestedPartialPlanAttr` 新增两项会影响 use-def 和资源合法性的决定：

- `issueMaterializationOrder`：`0/1/2` 分别表示 lhs/rhs/scale，verifier 要求恰为一个排列；
- `scaleSupplyStage`：`before-product` 或 `after-partial-reduce`；后者只允许单 output replica。

planner 只在 scale 与 indexed operand 可追溯到同一个具体 `FieldOp` 时调整顺序；单 output
把 scale 放在 partial reduction 后，多 output 把共享 Field 的 operand 与 scale 连续物化。
materializer 按 plan 生成 op，不重新扫描选择。

同一个 scale producer 还被 correction 在 issue loop 后使用时，materializer 把纯、同 block、
无 region、所有外部用户均位于 loop 后的 backward slice 放到第一个 post-loop user 前。
memory read 不会被移动。这个 placement 的作用是缩短 live interval，不改变动态算术或访存数。

## 3. 隔离实验

这组实验把组合收益拆开了：

| 形态 | SG2044 | K1/X60 | 结论 |
|---|---|---|---|
| 只有 shaped group + axis propagation | 33/32，非法 | 33/32，非法 | topology 已形成，但资源阶段未闭合 |
| 再加 scale stage，不做 post-loop placement | 33/32，非法 | 33/32，非法 | stage attr 本身不会自动缩短已有 SSA lifetime |
| stage + post-loop pure-slice placement，默认 lhs/rhs/scale order | 3.501 GOP/s | 33/32，非法 | SG 的单-output overlap 闭合；K1 多-output 仍差 1 group |
| 再消费 selected issue order | 3.505 GOP/s | 2.837 GOP/s | SG peak=31，K1 peak=29；双机合法 |

“计数不变为什么值得实现”的答案因此是机械可见的：这些 schedule 决定把 register peak 从
33 降到 31/29，使已经选定的完整-product topology 从非法变为合法；不是用吞吐猜测顺序。

还有一个主动撞出的反例：曾按“出现 indexed supply”做过宽排序，SG IQ2_XS 从约 4.84 降到
2.86 GOP/s。规则收窄为“scale 与 operand 共享同一个具体 Field”后，用同一 canonical input、
同一 target flags 比较 current 与 clean `99d46d001`，IQ2_XS 和 IQ2_XXS 的 final RISC-V IR
均为零行 diff。过宽规则没有保留。

## 4. 真机结果

所有表中数值都是当前工作树、Clang 18、合法量化 record、10 repetitions；CSV 对应行已标为
`targeted-rerun`。

| entry | target | 本轮前 Weft | 本轮后 Weft | source | 当前比值 |
|---|---|---:|---:|---:|---:|
| IQ1_S MUL_MAT prefill | SG2044 | 4.451287 | 3.505333 | 4.854846 | 72.20% |
| IQ1_S MUL_MAT prefill | K1/X60 | 1.735747 | 2.836568 | 2.740584 | 103.50% |
| IQ1_S standalone vec-dot | SG2044 | 3.419425 | 3.520539 | 4.861243 | 72.42% |
| IQ1_S MUL_MAT decode | SG2044 | 3.419057 | 3.548349 | 4.839919 | 73.31% |
| IQ1_S standalone vec-dot | K1/X60 | 1.999198 | 2.025068 | 2.711676 | 74.68% |
| IQ1_S MUL_MAT decode | K1/X60 | 2.019245 | 2.043251 | 2.663380 | 76.72% |

K1 prefill 过线；SG shaped prefill 相比旧串行树反而从 91.69% 降到 72.20%，不能写成双机
闭合。standalone/decode 在两台机器同步，仍说明它们的剩余差距在共享 contraction 内部，
不是 GEMV wrapper。

当前 SG prefill 每 256-record/output 仍有可数的 supply 差异：qh/metadata 为 9 次 load/gather
而 donor 为 1 次，q window 为 4 次而 donor 为 1 次，scale 有 8 次 slide 加 8 次 scalar
extract，bsum 是 2 次 indexed gather 而 donor 是一次连续 load。明确可删除的上限约 28 条
supply/movement 指令；product/reduction 数不应再减少。已有隔离负结果包括整条 scale
标量化、强行扩大 qh carrier、pipeline=2 和更宽 unroll，均未产生可保留收益。

## 5. 结构验收与回归

本轮按新 attr schema 重新生成 24 个 vec-dot + 24 个 row-dequant × 2 targets，共 96 份
Physical IR，不复用旧 final artifact：

- 96/96 可独立 parse/verify；
- 96/96 可安全运行 canonicalizer、CSE、layout canonicalization、Share 两次和 final verifier；
- 整条流水再次运行后，96/96 第二轮文本 diff 为零。

双机单次回归中，IQ2_XXS、Q3_K、Q6_K、TQ2_0 均数值正确且保持原档。IQ2_XS staged
prefill 的 full-product carrier legality failure 在 clean `99d46d001` 上同命令复现，属于既有
未闭合入口，不是本轮回归。

当前性能对照表共 202 行：133 行达到 source，29 行在 90%-100%，34 行在 70%-90%，
5 行在 50%-70%，1 行低于 50%。

## 6. 当前确定边界

- IQ1_M multi-axis partial topology 已完成；剩余 donor 形态要求改变 scale/reduction 结合和
  widening 归属，需要作者数值程序决定。本轮没有让 compiler 越界。
- IQ1_S K1 prefill 已过线；SG prefill 与双机 standalone/decode 仍是 qh/q/scale/bsum 的联合
  supply 与资源问题。当前 typed issue schedule 只有 IQ1_S prefill 一个正性能输入，不能宣称
  已跨格式泛化。
- TQ1_0 SG standalone/decode 当前约 59%。donor 的一个 i16 carrier + 一次最终 reduction 会把
  当前三个 canonical contract/reduce 改成另一棵 reduction/widening 程序；不能由 Physical pass
  暗改。
- IQ2_XS staged prefill 双机仍有既有 full-product carrier legality failure。

