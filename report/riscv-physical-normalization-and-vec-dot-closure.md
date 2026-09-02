# RISC-V Physical IR 归一化与 vec-dot 收口

本报告记录本轮一次性事实。设计合同仍以 `doc/` 为准；性能基线来自
`report/baseline/ggml-riscv-kernel-performance.csv`，当前 Weft 数字已写入
`report/weft-kernel-performance.csv`。

## 1. 本轮实际闭合的结构

### 1.1 indexed-entry 的单位在 IR 中闭合

`rvv_indexed_entry_load` 现在接收 typed `entry_offsets` 和
`entry_byte_stride`。`PlanRISCVMemory` 在 IR 中显式生成 index 到 byte offset 的
`shl`/`mul`，并拒绝窄整数无法表示 byte stride 的程序；emitter 不再把逻辑 index
临时乘成 byte offset。descriptor、local value、标量 load 和 RVV gather 共用同一
byte-unit verifier 合同。

这不是从参考实现照搬 index-width 选择。Triton 的普通 gather 在
`ref/triton/lib/Conversion/TritonGPUToLLVM/GatherOpToLLVM.cpp:51-63` 将低层 index
计算统一到 i32；TileLang 的 index-bitwidth pass 主要做越界后的提升。两者都没有
“按 range 自动选 u16 carrier”的机制。因此本轮只闭合单位和 producer use-def，
没有保留一次无收益的 u16 carrier 实验。

### 1.2 layout conversion 不再被 same-width cast 吞掉

same-width cast 若同时跨 layout，materializer 先在 source layout 中建立 typed cast，
再插入真实 `convert_layout`；layout canonicalizer 不再把它错误 rematerialize 成一个
跨 layout 的 cast。`axis_broadcast` 只增加 singleton physical axis 且 lane 数不变时，
emitter 直接保留 source，而不是生成假 broadcast。

这与 Triton `RemoveLayoutConversions.cpp` 的边界一致：producer 重写不能消掉真实的
layout relation；不能重写时 conversion 必须继续作为 SSA operation 存在。

### 1.3 unroll 的展开顺序改成 typed operation-major schedule

partial planner 已经冻结 issue/window/carrier 后，unroll materialization 才按
`issue_iteration` 和 `issue_operation_order` 做依赖拓扑排序。scheduler 只接受 flat、
pure 或 read-only body，并从 SSA 与 memory effect 建依赖；materializer 不重新选择
carrier。适用条件目前收窄到 `window_extent == 1`，因为把该顺序泛化到 IQ1_S 的宽
window 会扩大 live set 并退化性能。

责任划分参照 Triton `PipelineExpander.h` 的 schedule/expansion 分离，以及 TileLang
`pipeline_planning.cc` 与 `inject_pipeline.cc` 的先计划、后机械展开；这里实现的是
issue unroll，不宣称已经形成通用 software pipeline。

### 1.4 P2 只建立了窄的 scalar physical-share owner

同一 block 中，若一个多 consumer scalar 纯表达式只由同一个 encoded field 的
byte-aligned extracts、常量和纯 pointwise op 构成，memory pass 会插入真实
`register_materialize(realization="physical-share")`。不同 field、不同 owner 或带 effect
的表达式不能合并。该改动让共享 lifetime 进入 use-def 和 verifier，但本轮没有测到
动态指令下降，因此不能把它写成性能收益，也不等于通用 multi-consumer placement
已经完成。

### 1.5 IQ2_XXS storage layout 的所有 consumer 已同步

GGML 的一个 IQ2_XXS block 是 `d + 64 q bytes`；每个 32-element group 的前 4 bytes
是 grid index，后 4 bytes 是 little-endian sign/scale metadata。Weft Encoding 现在声明
`q: u8[64]`，vec-dot、row-dequant 和 staged MUL_MAT 的作者树都按同一 8-byte group
读取，不再把 storage 假写成 `u16[32]`。

## 2. 可观察性与主链验证

当前源码生成 24 个 vec-dot、两个 target 共 48 份终端就绪的 RISC-V Physical IR，最近一次完整静态
扫描结果为：

```text
frontend -> terminal-ready RISC-V Physical IR       48/48
weft-opt canonicalize + CSE + terminal verifier     48/48
同一 canonicalize + CSE + verify 序列第二次运行，文本变化  0/48
```

收口后的 IQ2_XXS 两个 target 又独立重放了一次上述序列，结果仍为可 parse、可 verify、
二次文本 diff 为零。`cmake --build build` 会自然重链 `weft-compile` 与 `weft-opt`；本轮
实际输出包含这两个 executable 的 link step，不再存在只更新静态库却拿旧工具测量的口径。

runner 远端最后一条命令不再使用 `exec`，EXIT trap 会删除临时目录；`weft-kernel.sh`
带 phase 的 runtime 不会再双跑。MUL_MAT 的 LMUL/unroll/pipeline 环境覆盖现在逐项替换
默认 binding，不会重复追加 LMUL，也不会因只覆盖 unroll 而丢掉其余默认参数。

## 3. 真机结果

编译器和 GGML wrapper 均使用 Clang 18、`-O3 -ffp-contract=fast` 与相同 ISA/ABI
配置。浮点结果按容差判定；下表均为 10 repetitions。

### 3.1 IQ2_XXS

```text
target   entry                    旧 Weft   当前 Weft   source    当前/source
SG2044   standalone vec-dot       2.847156  3.736277    4.051562   92.22%
SG2044   MUL_MAT decode           2.832527  3.724802    4.035300   92.31%
K1       standalone vec-dot       1.636052  1.688502    1.684037  100.27%
K1       MUL_MAT decode           1.646607  1.689220    1.666187  101.38%
```

单位均为 GOP/s。standalone 与 decode 在两台机器上同步，说明收益在共享 vec-dot
底座，不是 GEMV 外壳。SG2044 的 unroll=2 让 issue 间 load/decode 与 product 使用按
operation-major 顺序出现；unroll=4 的静态资源峰值为 33/32，按合同判非法。K1 维持
unroll=4。

同一 byte layout 的 row-dequant 也在两台机器上数值正确：SG2044
`114.117 MElements/s`，K1 `32.597 MElements/s`。它们只达到各自 source 的 29.6% 和
17.5%；这证明 layout 同步正确，但纯 decode 路径仍是独立的性能缺口。

### 3.2 IQ1_S

```text
target   entry                    当前 Weft   source    当前/source
SG2044   standalone vec-dot       3.630342    2.804481  129.45%
SG2044   MUL_MAT decode           3.650717    2.788436  130.92%
K1       standalone vec-dot       1.974402    2.715809   72.70%
K1       MUL_MAT decode           1.996864    2.677784   74.57%
```

K1 相对 clean base 的约 `2.048 GOP/s` 下降约 3.6%。原因不是旧诊断中的 vector
spill：重新生成的汇编峰值只有 6 vector groups，栈上的 `sh/flh` 是 f16 bit-pattern
转换。当前差距来自修正 same-width cast/layout 后，qh/index 算术不再被非法融合到
scalar carrier；K1 的合法 carrier/memory form 选择仍未打过 source。SG2044 同一棵
作者树已经过线，因此不能用另一棵 VLEN-specific 作者树掩盖。

### 3.3 回归

Q4_K persistent 在当前工作树上数值正确：SG2044 `9.791855 GOP/s`，K1
`3.190268 GOP/s`。另一次受影响回归批次中，Q2_K、Q6_K、TQ2_0、Q5_K、IQ4_XS、
Q4_0、Q4_1、Q5_1、IQ4_NL、Q1_0 在两台机器上均运行且数值在容差内；没有恢复旧
fallback。

## 4. 已执行并撤销的负实验

这些方向的代码均不在最终工作树中。

| 实验 | 静态变化 | 真机结果 | 结论 |
|---|---|---|---|
| IQ1_S operation-local vector partial scratch | 删除 scalar scale extract/VL=1 MAC，增加 stack store/reload | K1 `2.048 -> 2.039` | 指令数减少不足以抵消 local stack roundtrip，撤销 |
| IQ1_S `bsum` pair-fold | 两次 scalar supply 改成 pair-fold leaf | K1 `2.048 -> 2.024` | leaf 仍是两条 strided load，撤销 |
| broad operation-major schedule | 多个 issue 的同类 op 聚集 | IQ1_S K1 `2.048 -> 1.789`，live peak `15 -> 23` | 只保留 window_extent=1 的可证明窄合同 |
| IQ2_XXS scalar-finalized scale topology | 删除 VL=1 vector ALU | SG `3.736 -> 3.711`，K1 `1.689 -> 1.631` | 两台都退化，完整撤销 |
| u32 offset 直接重物化为 u16 | grid gather 由 `vluxei32` 变成 `vluxei16` | SG `3.736 -> 3.735`，K1 `1.689 -> 1.688` | 无收益；参考实现也没有 range-based u16 选择，撤销 |
| IQ2_XXS unroll=4 on SG2044 | 更多 issue 并行 | resource peak `33/32` | 合法性失败，不放宽预算 |

一次 same-field scalar physical-share 实验没有改变最终动态指令和性能；它因 use-def
所有权价值保留，但不计入性能提升。

## 5. 没有掩盖的边界

- IQ2_XXS staged prefill 在当前工作树和 clean HEAD 都停在同一个 RVV widening-dot
  layout 合法性错误：MR/NR free axes 与 reduction carrier 无法闭合。它不是本轮
  `u8[64]` encoding 改动造成的回归，但 staged MUL_MAT consumer 目前仍不能进入终端
  就绪的 RISC-V Physical IR。
- IQ1_S K1 仍只有 source 的约 72.7%；正确 layout conversion 暴露了真实 carrier/
  memory-form 缺口，而不是让旧的非法融合继续跑。
- IQ2_XXS SG2044 仍差 source 约 7.7%；本轮没有证据把剩余差距归给单独的 scale supply，
  因此没有再造一个 pass。
- Q2_K vec-dot 的 55--60% 问题本轮未处理。
- 通用 multi-consumer P2 owner 尚未完成；当前只有同 field scalar expression、typed
  layered window、materialized residency 和 partial supply 等几类窄合同。
- operation-major issue schedule 目前只有 IQ2_XXS 是明确正例；把它推广到宽 window
  的反例已经出现，所以不能宣称跨 topology 泛化。

预测账的已决项仍以 MISS 为主；P1--P5 继续只作为生成工作账的检查维度，不恢复成
未经外部结果支持的预测模型。
