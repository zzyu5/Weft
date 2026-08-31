# RISC-V Partial Plan Owner、Q1_0 Signed-mask 与 IQ2_XS Supply 审计

日期：2026-08-31  
实现提交：`24a72067c`（`Close RVV partial plans and fuse signed mask reductions`）

## 1. 本轮结论

本轮没有铺新格式，完成了三个有外部证据的节点：

1. `scalar-add-tree`、`independent`、`level-scaled` 三条旧 partial 路径不再在
   materializer 中重新选择 carrier、combine stages 和 finalization topology；planner 写入完整
   typed plan，materializer 只验证当前 SSA 仍满足 plan 并实例化 operation。
2. Q1_0 K1 的原诊断被纠正。当前改动前的真实汇编没有“32 对 vector spill/reload”；热循环里
   的栈访问是一对 f16 bit-pattern `sh/flh`，按 32 个 outer blocks 动态重复。真正多余的是每个
   32-element issue 的 `vwmul`。新 physical leaf 直接做 bitmask sign merge 和 i8→i16
   reduction，K1 standalone/decode 同步达到 source 的 100.1%/102.2%。
3. SG IQ2_XS 的 scale supply 不是已经被证明的唯一瓶颈。标量 supply 试验未产生稳定的 paired
   收益；真正暴露的是 issue unroll 的 owner 顺序仍有问题：generic unroll 先改树，partial
   planner 才看树，`unroll=4` 会破坏已经成立的 full-product carrier。

完整-product carrier 仍只有 IQ2_XS 一个跨格式正例。旧路径 owner 收口不会凭空让第二个格式
改变 topology；静态横扫没有发现第二个“先切 product、再 slide 拼回”的输入。

## 2. 参考实现给出的职责边界

动手前对照的不是 GPU thread ownership，而是 planner/materializer 的职责分离：

- Triton [`ReduceOpToLLVM.cpp`](../../ref/triton/lib/Conversion/TritonGPUToLLVM/ReduceOpToLLVM.cpp)
  的 `treeReduce` 消费已经给定的 arity，机械生成 combine；reduction carrier 来自前面确定的
  typed layout bases，不在终端 lowering 中重新搜索。
- TileLang [`reducer_plan_materialize.cc`](../../ref/tilelang/src/transform/reducer_plan_materialize.cc)
  在 layout 已冻结后先保存 storage、communication、multiplicity 和 reduction steps，再由
  finalize/materialize 阶段消费显式 plan。

Weft 不能复制 warp、CTA、shuffle 或 XOR butterfly，但可以复制这条 owner 规则：

```text
typed axis/layout/use facts
→ planner 冻结 carrier、slot map、combine stages、final leaf
→ materializer 机械实例化
```

## 3. 旧 partial 路径 owner 收口

### 3.1 Planner 现在写出的事实

新增的 `PartialCombinePlanAttr` 承载：

- `level_scaled`、`independent_vector` 或 `independent_scalar` realization；
- source/reduced/final `PartialSetType`；
- 每层 combine set type 与 arity；
- lhs/rhs/scale part map；
- multiply 与 finalize leaf；
- output parts 与 resource groups。

新增的 `PartialAddTreePlanAttr` 承载跨 leaf 的：

- source 与 normalized set types；
- 每个 leaf 的 split；
- common merged set；
- leaf multiply 与最终 finalize leaf；
- slots、terms 和 resource groups 总数。

定义和 verifier 位于
[`RISCVOps.td`](../include/Weft/Dialect/RISCV/IR/RISCVOps.td) 与
[`RISCVDialect.cpp`](../lib/Dialect/RISCV/IR/RISCVDialect.cpp)。planner/materializer 的实现位于
[`MaterializeRISCVPartialAccumulators.cpp`](../lib/Target/MaterializeRISCVPartialAccumulators.cpp)。

### 3.2 外部不变性检验

在改动前后分别生成同一入口的最终 physical IR，结果为：

| 入口 | 代表路径 | final IR diff lines |
|---|---|---:|
| Q1_0 vec-dot | scalar add/sequential leaf | 0 |
| Q2_K vec-dot | level-scaled | 0 |
| Q3_K vec-dot | independent | 0 |

这说明收口没有利用第二套判定偷偷换 topology。双机 10 次中位数回归为：

| entry | SG2044 GOP/s | K1 GOP/s | correctness |
|---|---:|---:|---|
| Q2_K standalone | 6.578195 | 1.535944 | within tolerance |
| Q6_K standalone | 6.065100 | 2.938433 | within tolerance |

Q2_K 与既有数字基本相同；Q6_K 的多次 SG 探针在约 5.58–6.07 之间波动，但最终 IR 完全未变。
本轮 CSV 写入最后一次 10-repetition 结果，没有把单次更高或更低值挑成 winner。

本节只声明上述三条旧路径完成 owner 收口。`nested`、`layered` 和
`scaled/reduced-scaled` 是已有的独立 typed contracts；本轮没有把“这三条已收口”扩大成
“文件中所有 materialization 分支都已统一”。

## 4. Q1_0 K1：七步诊断与修复

### 4.1 Hot Level 与 work ledger

K=4096 时，作者程序是 32 个 128-element blocks，每个 block 有 4 个 32-element sub-Level，
即每个 output 有 128 个 contraction issues。

改动前 K1 每个 issue 是：

```text
logical u1 mask
→ widen to i8
→ (bit * 2 - 1)
→ vwmul with q8
→ vwredsum to scalar
```

GGML donor
[`quants.c:484-515`](../source/c/ggml/llama.cpp/ggml/src/ggml-cpu/arch/riscv/quants.c#L484)
则是：

```text
vlm mask
→ negate q8
→ vmerge(-q8, q8, mask)
→ vwredsum i8 to i16
```

donor 本身也对每个 32-element issue 做一次 reduction。因此原来的问题不是 P4“过早
reduction”，而是 physical contraction leaf 先制造了一个不需要的 i16 product。

### 4.2 “32 对 spill”诊断纠正

重新生成实现改动前提交 `077d65832` 的 K1 汇编后：

- 没有 vector spill/reload；
- final physical assignment 的 vector peak 是 6 groups；
- 栈上的一对 `sh/flh` 是 f16 bit-pattern conversion，按 32 个 blocks 动态执行，不是 32 个
  独立 vector SSA 被 spill。

因此 Q1_0 不属于“P4 + R 已被证实”。R 不是主因；旧报告中的该表述被本报告取代。

### 4.3 新 typed physical operation

`rvv_signed_bitmask_reduce` 只在以下 typed facts 同时成立时生成：

- data 是同一 reduction axis 上的 signed i8 lane value；
- mask field 是 byte-aligned、`grouped_layered`、`lo_first` 的 logical u1 storage；
- point/origin 明确对应同一 sub-Level；
- lane carrier 覆盖完整 issue；
- `extent × 128 < 32768`，证明直接 i16 reduction 不溢出。

pass 从 canonical `(2*bit-1) * q8` contraction 识别上述 typed 数值/存储关系，写出真实
physical op；emitter 只拼写 `vlm + vneg + vmerge + vwredsum`。它不读取格式名、kernel 名或
target 型号。含该 op 的最终 physical IR 已由 `weft-opt` 成功 parse 和 round-trip。

K1 每个 output 删除 128 个 `vwmul`；最终 IR 的 leaf resource 从原链的 peak 6 groups 降为
3 groups。`vwredsum` 数量不变，因为 donor 的合法程序也需要 128 次。

Q1_0 Encoding 同时补上 GGML record 的 `alignment=2`。它是跨调用 byte layout 事实，不是
backend hint；这使生成 C 不再通过栈临时量做 f16 bit-pattern load。

### 4.4 配对实测

两边使用 Clang 18.1.8、`-O3 -ffp-contract=fast -mabi=lp64d`，10 次 cold median：

| target | entry | 改前 GOP/s | 当前 GOP/s | source GOP/s | 当前/source |
|---|---|---:|---:|---:|---:|
| SG2044 | standalone vec-dot | 4.679408 | 3.993690 | 2.264629 | 176.4% |
| SG2044 | MUL_MAT decode | 4.645915 | 4.670632 | 2.333519 | 200.2% |
| K1 | standalone vec-dot | 2.798853 | 3.775324 | 3.772674 | 100.1% |
| K1 | MUL_MAT decode | 2.806433 | 3.779040 | 3.696397 | 102.2% |

K1 的 standalone/decode 同步提升约 34.9%/34.7%，并同时超过 source。SG 默认 m1 carrier
不满足“完整 32-lane issue”条件，因此不使用该 leaf；它仍走原先两条 16-lane time slices。
强制 SG 使用 m2 会触发 signed-mask leaf，但 standalone 为 3.382428 GOP/s，低于默认 m1 的
3.993690，因此该参数试验没有保留。

Q1_0 prefill 不使用新 signed-mask leaf，主要受 Encoding alignment 变化影响：

| target | 当前 GOP/s | source GOP/s | 当前/source |
|---|---:|---:|---:|
| SG2044 | 8.716957 | 4.249822 | 205.1% |
| K1 | 3.695145 | 3.792784 | 97.4% |

## 5. SG IQ2_XS scale supply：证明与排除

### 5.1 静态占比

standalone 与 MUL_MAT decode 的 contraction core 完全同构。每个 4-way issue 有 40 个 RVV
intrinsic calls，其中：

- regular-repeat index/gather/extract 为 10 个，占 25%；
- 连同后续 scale combine 为 14 个，占 35%；
- decode 额外包含 activation quantization，standalone 没有；
- 两者 shape 与调用粒度不同：standalone N=14336 且 activation 已量化，decode N=4096 且
  quantization 在计时范围内。

这证明 scale supply 有可见成本，但不能把 standalone/decode 的全部差距归给它。

### 5.2 负实验

把 4-replica scale supply 从 vector gather/extract 改为 scalar rematerialization 后，三次
10-repetition invocation 为：

| entry | GOP/s observations |
|---|---|
| standalone | 3.346577 / 3.743597 / 4.480570 |
| MUL_MAT decode | 4.279420 / 2.324921 / 2.323556 |

两列没有同步的稳定收益，因此该规则已撤销。当前 CSV 继续保留可复现的 vector-convert 路径
3.391627/3.121361 GOP/s。

issue unroll 试验同样为负：

- `unroll=2`：standalone 2.958566、decode 0.550740 GOP/s；
- `unroll=4`：generic unroll 先复制 canonical loop，随后 nested partial matcher 失效，最终重新
  出现 sliced `vwmul/vwredsum + slide`，没有形成 donor 的四 issue steady schedule。

后一个结果定位出一个真实 P5 owner 缺口：`NestedPartialPlanAttr.issue_unroll` 已存在，但
emitter 不消费它；通用 `UnrollRISCVLevels` 又运行在 partial planning 之前。当前没有把这个
负结果固化成另一条路径。

## 6. 第二个正例与 SG TQ2_0

对 24 个格式、两台 target 的 final IR 静态扫描中，只有 IQ2_XS 出现过精确的“完整 product
被提前切片，再用 slide 拼回”模式。Q1_0 是不同问题；其他 slide 属于 bit/index/radix 或
layout conversion。因此本轮没有第二个跨格式正例，不能把 full-product carrier 规则描述成
已跨格式验证。

SG TQ2_0 standalone 的默认 binding 不是缺一个 runner meta，而是真实资源非法：

```text
physical program requires 36 vector groups, target budget 32
```

对 LMUL eighths `1/2/4/8/16/32` 的峰值分别为 `132/68/36/36/38/42`，没有合法项。production
MUL_MAT prefill 的另一组 shape/meta 可以生成，但不能反过来证明 standalone 默认 binding 合法。
因此旧的 SG TQ2_0 standalone 性能行已从当前 CSV 删除。

## 7. 当前可复现边界

- Q1_0 standalone 与 MUL_MAT decode 双机均数值正确；K1 两条已超过 source。
- Q1_0 row-dequantize 当前在两台机器都于同一位置失败：logical u1
  `grouped_layered` field 在 emission 前没有 typed physical window/stream operation。旧 CSV
  的两条 row-dequant 数字不是当前编译器可复现结果，已删除。
- SG IQ2_XS 仍为 source 的 97.4%/89.5%（旧当前快照）；scale supply 假设未得到 paired
  实测支持，剩余确定缺口是 issue scheduling/unroll owner 顺序，而不是又一个格式 leaf。
- prediction ledger 没有新增已决条目；仍为 `0 HIT / 1 MISS / 5 pending`。Q1_0 不在原预测表
  中，因此没有事后添加一次 HIT。

## 8. 手工复现

```bash
examples/run/weft-quantized-vec-dot.sh sg2044 q1_0 10
examples/run/weft-mul-mat.sh sg2044 q1_0 decode 10
examples/run/weft-quantized-vec-dot.sh k1 q1_0 10
examples/run/weft-mul-mat.sh k1 q1_0 decode 10

examples/run/weft-quantized-vec-dot.sh sg2044 q2_k 10
examples/run/weft-quantized-vec-dot.sh k1 q2_k 10
examples/run/weft-quantized-vec-dot.sh sg2044 q6_k 10
examples/run/weft-quantized-vec-dot.sh k1 q6_k 10
```

全部成功命令均由当前提交生成 intrinsic C、在对应真机用相同 Clang/flags 编译运行，并报告
`numeric=within-tolerance`。没有建立测试目录、fixture 或性能脚手架。
