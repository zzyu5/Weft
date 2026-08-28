# RISC-V partial topology 统一、IR 语义与双机性能报告

日期：2026-08-29

范围：本报告只记录本轮对 partial topology、相关物理 IR/pass 合同、通用 canonicalizer/CSE 可用性，以及 IQ4_XS、Q3_K 和受影响回归入口的结果。没有铺新格式，也没有把历史全量数字解释为当前结果。

所有正式性能均使用 production shape、Clang 18、`-O3`、相同浮点 contraction 配置和 10 repetitions；source 数字来自 `report/baseline/ggml-riscv-kernel-performance.csv`。数值正确性采用项目当前的容差判据。

## 一、外部结论

| 目标入口 | SG2044 Weft / source | SG 比值 | K1 Weft / source | K1 比值 | 结论 |
|---|---:|---:|---:|---:|---|
| IQ4_XS MUL_MAT prefill | 7.666652 / 5.894772 | 130.1% | 2.325529 / 1.718922 | 135.3% | 双机过线 |
| Q3_K MUL_MAT prefill | 3.599146 / 7.218739 | 49.9% | 1.181831 / 2.449247 | 48.3% | 双机正确，但未过线 |

IQ4_XS 从本轮前的 3.471709 / 1.277462 GOP/s 提升到 7.666652 / 2.325529 GOP/s。Q3_K 从 0.886730 / 0.431666 提升到 3.599146 / 1.181831 GOP/s，但仍只有 source 的约一半。因此本轮证明了一项物理组织能跨 VLEN 生效，也同时留下了一个没有被 topology 统一自动解决的反例。

## 二、动手前采用的参考机制

本轮没有把 Triton/TileLang 的实现直接移植到 CPU/RVV，而是先确定它们如何把“一个 reduction 的物理组织”变成单一 IR 合同。

### 2.1 Triton

- `tt.reduce` 是一个 variadic、带 combine region 的值语义 operation，而不是多个 emitter closure；定义见 `/home/kingdom/phdworks/ref/triton/include/triton/Dialect/Triton/IR/TritonOps.td:761-788`。
- `ReduceOpToLLVM` 依据 register bases、lane bases 和消去轴组织 thread-local、lane-local 与 tree reduction；对应实现见 `/home/kingdom/phdworks/ref/triton/lib/Conversion/TritonGPUToLLVM/ReduceOpToLLVM.cpp:228-351`。
- dot operand 的 local allocation/load 是真实 IR 重写；对应 `/home/kingdom/phdworks/ref/triton/lib/Dialect/TritonGPU/Transforms/OptimizeDotOperands.cpp:308-331`。

直接照用 Triton 的具体 warp/thread reduction 不成立，因为 RVV 没有虚拟线程归属、warp shuffle 或 shared-memory reduction。但它给出的结构约束可以直接采用：一个 reduction operation 只能有一个已选 physical topology；后续 lowering 消费这一个合同，不能由不同 materializer/emitter 再竞争 ownership。

### 2.2 TileLang

TileLang reducer materialization 以 epoch 为单位构造一份 reducer plan，并要求多个 update site 得到 structurally equal 的 plan；对应 `/home/kingdom/phdworks/ref/tilelang/src/transform/reducer_plan_materialize.cc:559-735,784-920,1202-1338` 与 `/home/kingdom/phdworks/ref/tilelang/src/op/reducer.h:171-219`。

Weft 不能直接采用 TileLang 的 thread binding 和 shared-memory plan，但采用了同一条 owner 约束：partial slots、剩余 axes、combine topology 和资源合同必须先组成一份 plan，再物化所有 update；不能让 fused stream 和 independent partial 各自从局部形状重新判断。

## 三、partial topology 现在怎样成为一个决定

### 3.1 唯一 typed 合同

`RVVWidenDotOp` 现在携带一个 `#weft_riscv.partial_topology`。该属性记录：

- topology kind；
- 唯一 layered root（若存在）；
- partial axes 与 output axes；
- source slots、partial slots、output replicas；
- lane split；
- combine arity 与 slot order；
- 预计寄存器组。

旧的 `stream_reduction` 实现字段已从当前源码删除。fused layered stream、independent set、scaled/level-scaled partial 和顺序 topology 不再各自拥有一个外围布尔门控；它们是同一属性的互斥取值。

### 3.2 planner 与 materializer 的职责

编译流水中现在连续执行：

```text
PlanRISCVPartialTopologies
→ MaterializeRISCVPartialAccumulators
```

planner 只从 typed operands、轴、use-def、layered root、slot 几何和资源事实写入 topology；materializer 只读取该 topology 并生成 `rvv_partial_set/repack/reduce/scale_combine/combine/finalize` 或 layered load/decode/widen-accumulate。最终 verifier 不允许已要求物化的 topology 继续残留为普通 widening dot。

这次同时抽出了两项共享推导，避免 planner 和 materializer各算一遍条件：

- `deriveScaledSourceLaneGeometry`：唯一计算 scaled source 的 lane/slot 几何；
- `canProjectWindowSlice`：在选择 layered topology 前，对完整 use-def slice 做可投影性预检。

后者直接阻止了 K1 Q2_K 曾经出现的非法 `[32] × [128] -> [32]` widening multiply。资源不足或 projection 不闭合时，该 topology 在生成 leaf 前判非法，不由 emitter 缩窄或改路。

### 3.3 verifier 补上的关系

`PartialTopologyAttr` 现在验证 axes 唯一、partial/output axes 不重叠、slot order 是完整 permutation、combine arity 可整除、lane split 与 source/partial slot 数闭合，并且只有 layered topology 可携带 root operand。

`RVVPartialSetOp` 的 verifier 与 emitter 也统一了 subregister slice 的合同：LMUL 小于 8、从更大 source 中取 exact slice 时，以 m1 carrier 加 lane offset 表示，不再由 verifier 和 emitter 使用两套规则。

## 四、静态计数后才保留的优化

IQ4_XS 的 activation/window 共享在修改前按 `(record replica, logical offset)` 计数；旧 key 忽略 replica，既不能证明跨 output 共享安全，也会把不同物理 record 混在一起。修改后：

- window identity 显式包含 `recordReplica` 与 `windowOffset`；
- verifier 禁止一个 source window 跨 register replicas；
- emitter 只根据已选 window 重建 record pointer。

这项修改把多个 output consumer 的相同 activation record load 合成一个 typed window，同时没有合并不同 logical offset。IQ4_XS 双机 10-repetition 的 2.21× / 1.82× 提升是其外部结果。

Q3_K 的 natural affine activation access 也从 indexed gather/index-vector construction 变成 unit load；最终生成 IR 中不再为这条自然连续轴构造 gather index。它减少了动态工作，但不足以解决 Q3 的全部差距。

## 五、等价作者写法主动撞坑

### 5.1 预定的强检验没有通过

Q4_K 的 `mac_pairs` 与 `mac_groups(n=4)` 是最直接的等价作者写法：外围 Level、Encoding、scale/min epilogue 相同，只改变 partial grouping。两者当前都在进入 partial topology 之前失败：

```text
reduction layout conflict requires an unsupported non-scalar lane remap
```

因此本轮不能声称“高性能 partial topology 对等价作者树稳定”。失败发生在 reduction-layout propagation，而不是 topology 门控再次选错；这是一个更早的 physical layout 缺口。

### 5.2 能成功的弱检验

对 Q2_K group-reduced vec-dot，把 `partial * scale` 改成交换律等价的 `scale * partial`。两份程序均成功生成：

```text
2 × partial_topology = sequential_per_stream
0 × rvv_partial_*
```

这证明 sequential topology 的识别不依赖乘法 operand 的直接 SSA 顺序。但它没有覆盖 independent/layered topology，不能替代上一节失败的强检验。

## 六、通用 canonicalizer/CSE 审计

### 6.1 当前 physical IR 仍不能安全运行通用 CSE

九类 `RVVPartial*` operation 目前声明为 `Pure`。`PartialSetType` 与 topology 属性描述 slots、axes、lane offsets 和资源，却没有独立 partial cohort/lifetime identity。于是两个 opcode、SSA operands、attributes、result types 相同但必须独立存活的 partial，在标准 CSE 看来是同一个值。

这不是理论风险：历史 TQ2_0 负实验中，通用 CSE 将 SG2044 prefill 从约 20.47 降到约 10.13 GOP/s。当前自定义 `CanonicalizeRISCVLayouts` 因而只处理 conversion inverse/chain、受限的 single-use rematerialization 和同 block conversion CSE，没有把 partial program交给通用 CSE。

### 6.2 trait 不能为单个 pass 临时修改

本轮曾把 `PhysicalPoint`/`RootPoint` 标为 `Pure`，目的是让 Level scheduler 把它们视为只读。最终 DCE 随即删除这些显式 Level points，final verifier 报：

```text
final physical Level has no explicit point
```

该尝试已撤销。当前做法是保持它们非 Pure，并让 `ScheduleRISCVLevels` 在自己的 effect 判定中显式把 point 当作只读。这一事实说明现有 op effect/trait 合同还没有成熟到可直接运行通用 canonicalizer。

### 6.3 当前可观测性也不完整

尝试对最终 RISC-V IR 执行：

```bash
build/tools/weft-opt/weft-opt final.riscv.mlir --pass-pipeline="builtin.module(cse)"
```

`weft-opt` 在首个 `weft_riscv.encoding_decl` 即因 dialect 未注册而失败。因此目前甚至没有一条独立工具命令可以对 physical IR 重放通用 MLIR pass。旧 canonicalizer 产生 emitter 合同外 `arith.select` 的历史问题也已不在当前源码中，无法在本轮重现到具体 fold。

结论不是“禁用通用 pass 即可”。当前事实是 partial lifetime/cohort 的值语义尚未由 type/op/region 合同表达，通用 CSE 不安全；physical dialect 又未注册进 `weft-opt`，通用 pass 的外部复现入口也缺失。

## 七、性能与回归

### 7.1 正式结果

| entry | SG2044 GOP/s | SG source | ratio | K1 GOP/s | K1 source | ratio |
|---|---:|---:|---:|---:|---:|---:|
| IQ4_XS prefill | 7.666652 | 5.894772 | 130.1% | 2.325529 | 1.718922 | 135.3% |
| Q3_K prefill | 3.599146 | 7.218739 | 49.9% | 1.181831 | 2.449247 | 48.3% |
| Q2_K prefill | 10.164706 | 9.462195 | 107.4% | 2.740874 | 2.410350 | 113.7% |
| TQ2_0 prefill | 20.559046 | 6.982594 | 294.4% | 6.586351 | 4.979196 | 132.3% |

Q2_K 的 K1 数字从上一正式值 2.962128 降到 2.740874。原因不是噪声：完整 projection preflight 拒绝了此前未闭合的 projected topology，避免非法 shape 进入 emitter，代价是回到较保守但合法的物理组织。SG Q2_K 与双机 TQ2_0 保持过线。

### 7.2 SG2044 过线入口回归

| entry | current GOP/s | source GOP/s | ratio | 说明 |
|---|---:|---:|---:|---|
| Q4_0 prefill | 11.225918 | 7.580451 | 148.1% | 10 reps；新 live set 下 runner 用 MR4 |
| Q4_1 prefill | 6.723931 | 2.535208 | 265.2% | 10 reps；runner 用 MR4 |
| Q5_1 prefill | 6.237136 | 5.950653 | 104.8% | 10 reps；恢复到合法 MR4×NR1 |
| IQ4_NL prefill | 8.629477 | 6.954179 | 124.1% | current CSV |
| Q1_0 prefill | 6.912480 | 4.249822 | 162.7% | current CSV |

Q4_K persistent 当前仍在 reduction layout 处失败。独立 clean-HEAD worktree 得到相同失败，因此它不是本轮 topology 改动引入的回归；本轮没有用旧路径掩盖它。

## 八、Q3_K 为什么仍未过线

本轮对 Q3 做过的正负对照：

- 第一版 wide-lane：widen/scale 后再 reduce，SG 约 1.541 GOP/s；
- reduce-first、MR2：SG 约 3.12；
- MR1×NR1、LMUL=m1：正式 SG 3.599、K1 1.182；
- NR2：SG 单次约 0.978；
- 更大 LMUL：均低于 m1。

当前生成程序仍在 `L.subs` 内重复 decode scale，并产生 spill；GGML donor 在两个 128-element half 之外一次解码完整 16 个 scales。差距包含两层边界：

1. **作者树边界：**把 decoded full-16 scale Value 的 birth/lifetime 移到 `L.subs` 之外，会改变 Value 的 Level 归属，按 spec 2.2 必须由作者树表达，编译器不能暗中提升。
2. **编译器边界：**在不改变作者 Value/Level 的前提下，当前 scaled topology 只覆盖单个 `reduce(dot * scale)` closure；它没有形成跨完整 `L.subs` epoch、包含多个 update site 的 reducer plan。该能力对应 TileLang 的 epoch reducer plan，而不是再加一个 Q3 closure matcher。

因此 Q3 未过线不是“参数没扫够”，也不能由 emitter 改拼写解决。要达到 donor 的 scale lifetime，现有作者树本身需要显式改变；要让同一 Level 内多个 update site 共用 partial plan，physical pass 仍缺少 epoch 级 reducer topology。

## 九、本轮没有闭合的事实

- 高性能 topology 对 `mac_pairs`/`mac_groups(4)` 等价作者写法的稳定性尚未证明；两者都被更早的 non-scalar lane remap 缺口挡住。
- `RVVPartial*` 的 independent lifetime/cohort 尚未进入类型或 operation 语义，标准 CSE 仍不安全。
- `weft-opt` 尚不能解析 physical RISC-V dialect，无法独立重放 canonicalizer/CSE。
- Q3_K 双机仍只有 source 的约一半；具体差距已落到作者 scale birth 与编译器 epoch reducer plan 两项。
- Q4_K persistent 是既有未闭合入口，不是本轮回归。

## 十、可复现命令

```bash
bash examples/run/weft-mul-mat.sh sg2044 iq4_xs prefill 10
bash examples/run/weft-mul-mat.sh k1 iq4_xs prefill 10
bash examples/run/weft-mul-mat.sh sg2044 q3_k prefill 10
bash examples/run/weft-mul-mat.sh k1 q3_k prefill 10
bash examples/run/weft-mul-mat.sh sg2044 q2_k prefill 10
bash examples/run/weft-mul-mat.sh k1 q2_k prefill 10
bash examples/run/weft-mul-mat.sh sg2044 tq2_0 prefill 10
bash examples/run/weft-mul-mat.sh k1 tq2_0 prefill 10
```

`WEFT_KEEP_ARTIFACTS=1` 只用于当轮检查 generated C 与远端汇编；本轮临时 artifacts 在提交前删除。
