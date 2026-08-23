# RISC-V 发射质量与 K1 真机结果

## 本轮问题

本轮区分两件事：

```text
前序 pass 决定什么机器实现
已经决定的机器实现怎样写成 intrinsic C / local asm
```

检查对象不是 emitter 的文件组织，而是生成 C 经 SG2044 GCC 15 与 K1 Clang 18
之后实际留下的指令、栈对象、spill、`vlenb` 依赖和真机吞吐。

参考 Triton 与 TileLang 后采用的判断是：layout、pipeline、unroll、local operation
必须在 codegen 之前成为明确属性或结构；codegen 可以展开已经选定的结构，但不能从
source closure 重新选一次。Triton 的 convert-layout 与 software-pipeline lowering、
TileLang 的 layout map 与 pipeline order/stage 都遵循这一边界。它们也都在进入最终
后端前生成真实循环/流水结构，而不是把性能意图留成由系统编译器自由解释的提示。

## 暴露出的两个前序决定缺口

### `contract` 没有完整 local operation

`gemv_f32` 的 physical assignment 能完成，但 intrinsic-C emission 会段错误。原因不是
GEMV 名称或 shape，而是普通 `contract` 只得到 `rvv.reduction-product`；只有
`outer_contract` 会得到 lane operand、reduction axis 与 lane memory form。

实际 GEMV 关系是：

```text
W[M,K] × X[K] → Y[M]
lane axis      = M
reduction axis = K
lane operand   = W
memory form    = runtime-strided
```

现在这些事实由 local-operation pass 沿 `contract` operands 与 admit/slice use-def 写入
assignment。生成器不再默认 lhs，也不再读取空字典。修复后的 assignment 中 op17 为：

```text
instruction      = rvv.vfmacc
lane_operand     = lhs
lane_axis        = M
reduction_axis   = K
lane_memory_form = runtime-strided
```

同一 tracing 也用于 `outer_contract`：memory form 从 admitted value 继续追到 slice/view
layout，而不是在没有 encoding facts 的临时 value 上固定得到 `runtime-strided`。

### 浮点 contraction 没覆盖 scalar-vector 与 subtraction

Q4_K 的 RVV 与 IME 在 K1 上产生完全相同的数值偏差，说明问题位于两者共享的浮点
epilogue，而不是 Q4 mapping 或 `vmadot`：

```text
d * scaled - dmin * minimum
acc + ds * difference
```

标量 reference 与 generated C 都使用 `-ffp-contract=fast`。K1 Clang 18 对 reference
实际生成：

```asm
fmsub.s   difference, d, scaled, min_term
fmadd.s   acc, ds, difference, acc
```

旧 generated intrinsic C 固定为独立 `vfmul + vfsub + vfmul + vfadd`，系统编译器无法
跨 intrinsic 调用恢复 contraction。此前 operation pass 只会选择 vector-vector
`mul + add`，因此它缺少的不是 emitter 拼写，而是两类 selected operation：

```text
vfmsac.vv   product - accumulator
vfmacc.vf   accumulator + scalar * vector
```

现在 assignment 分别记录 `instruction` 与 `operand_form`；emitter 只据此拼写
`__riscv_vfmsac_vv_*` 和 `__riscv_vfmacc_vf_*`。在这一变化之后，SG2044、K1 RVV
和 K1 IME 三条路径都与同一 reference 逐位一致。

## 真机结果

所有数字来自同一当前代码状态；数值不一致时没有记录性能。系统 C 编译均使用
`-O3 -ffp-contract=fast`。SG2044 使用 GCC 15.2、VLEN128；K1 使用 Clang 18、
VLEN256。

| kernel | target | 结果 | median | Weft | 同 shape source | 比值 |
|---|---|---:|---:|---:|---:|---:|
| Q4_K vec-dot | SG2044 RVV | bit-exact | 13.864344 ms | 8.470687 GOP/s | 9.728344 GOP/s | 87.07% |
| Q4_K vec-dot | K1 RVV | bit-exact | 42.192988 ms | 2.783413 GOP/s | 2.495786 GOP/s | 111.52% |
| Q4_K vec-dot | K1 IME | bit-exact | 321.734552 ms | 0.365023 GOP/s | 无同 shape IME source | — |
| Q4_K `mac_groups(4)` | SG2044 RVV | bit-exact | 11.930215 ms | 9.843956 GOP/s | 9.728344 GOP/s | 101.19% |
| Q4_K `mac_groups(4)` | K1 RVV | bit-exact | 37.399004 ms | 3.140204 GOP/s | 2.495786 GOP/s | 125.82% |
| F32 GEMV | SG2044 RVV | bit-exact | 764.223475 ms | 0.153673 GOP/s | 无同 shape source | — |
| F32 GEMV | K1 RVV | bit-exact | 100.424059 ms | 1.169446 GOP/s | 无同 shape source | — |

SG2044 Q4_K 曾有一次同一代码状态下的 3.458880 GOP/s 异常低值；立即在系统 load
约 2.2 时复跑五次，中位数恢复为 8.470687 GOP/s，因此 CSV 记录后者，并把 repetitions
改为 5。K1 两条结果连续复跑稳定。

`report/weft-kernel-performance.csv` 已直接更新两条现有 Q4_K vec-dot 行，并追加
`mac_groups(4)`、K1 IME 与两条 F32 GEMV 当前数字。没有同 shape source 的行将 source
字段留空，没有制造速度比。

`mac_groups(4)` 改变了 i16 partial 的逻辑树，却没有改变共同 f32 epilogue。它在两台
机器上不增加任何新 selector 就逐位一致，并分别达到 101.19% 与 125.82%。这是本轮
operation-cluster 选择没有依赖 `mac_pairs` source closure 的外部检验。

## 生成 C 经 K1 Clang 18 后留下什么

对四份当前 generated C 使用相同 K1 编译选项生成汇编：

| generated C | asm 行数 | `vlenb` | unknown-size vector spill/reload | `memcpy` call | fused vector op | `vmadot` |
|---|---:|---:|---:|---:|---:|---:|
| F32 GEMV | 97 | 0 | 0 / 0 | 0 | 1 | 0 |
| Q4_K RVV | 963 | 0 | 0 / 0 | 0 | 2 | 0 |
| Q4_K IME | 1443 | 0 | 0 / 0 | 0 | 2 | 1 |
| IQ2_XXS local-pack | 2572 | 34 | 17 / 17 | 0 | 2 | 0 |

这组结果把三种问题分开了：

1. Q4_K RVV 的 selected representation 能被 Clang 保留下来：没有动态 VLEN spill，
   两个 contraction 也真实落成 `vfmsac` 与 `vfmacc`。
2. IQ2_XXS 的 `vlenb` 不是某个 load helper 的拼写造成的。它来自同一 live region 中
   同时存活的 quantize chunk、decode temporary、两个 accumulator 与 carry；Clang 为
   17 个 unknown-size vector value 成对生成 spill/reload。这是 representation/resource/
   schedule 问题，不能在 emitter 按 IQ2 名称修。
3. K1 IME 没有 vector spill，却仍比 RVV 慢 7.63 倍。当前 helper 的主要成本是四重
   scalar address/decode loop、栈上的 `output[16] / activation[8] / weights[32]`，以及循环
   中唯一一处 `vmadot`。它“使用了 IME 指令”，但没有形成高性能 IME operand layout。

### 没有保留的拼写试验

曾把 selected little-endian f16/f32 load/store 从固定长度 `memcpy` 改成显式 byte
assembly 与 union bit conversion。真机结果为：

```text
SG Q4_K     8.438 → 8.300 GOP/s
K1 Q4_K     2.778 → 2.786 GOP/s
K1 IME      0.365 → 0.362 GOP/s
```

没有稳定收益，因此该改动已撤销。K1 GEMM 中出现的 `call memcpy` 也不是这组 helper：
它是 Clang 把已选 A-panel row-pack loop 识别成行复制后的结果；同一函数没有 unknown-size
vector spill。不能因为看到 `memcpy` 文本就统一禁止它。

## K1 IME 的实际边界

当前 Weft 与 GGML donor 不是同一作者程序。

当前 Weft `q4k_gemv_ime`：

```text
Q4_K 数值树保留 sc/min 两条支路
Q4K_I16 只把 canonical record 做 16 行 byte-major interleave
每个 sub32 调一次 raw q × q8 matrix primitive
scale/min 与 f32 accumulation 仍在外层树中
```

GGML K1 IME Q4_K donor 的 persistent repack：

```text
一个 Q4_K block 拆成 8 个 Q4_1x16 block
repack 时预先形成 fp16 (d * sc) 与 fp16 (-dmin * m)
q payload 重排为可由 IME 连续加载的 row-major nibble tile
leaf 持有多组 accumulator、连续 B window 和跨 K 的寄存器组织
```

这不仅是同一 logical encoding 的另一种 byte order。它改变了中间逻辑值、fp16 舍入
位置、persistent layout 和 ABI。按 spec 2.2，这些内容必须由作者在 std 特化与
persistent packed input 中写出；compiler 不能把当前 Q4_K 树自动变成 donor 的树。

所以本轮没有把 donor loop 包装进 IME emitter。若只优化当前 raw-Q4 tree，首先需要
一种能被 `vmadot` 直接连续消费的作者侧派生 layout；若采用 donor 数值分解，则需要
另一棵明确的 std tree。两者都不是本轮“已决定结构的拼写”可以补出的事实。

## Target 事实是否真正贯穿

标准 RVV Q4_K assignment 在两台机器上的差别目前主要是：

```text
SG2044 VLEN128: 16 个 f32 lane 使用 LMUL=m4
K1     VLEN256: 16 个 f32 lane 使用 LMUL=m2
```

相同 encoding 下 memory form 相同是合理的；但两台 target 的 schedule 仍都是
`unroll=1 / pipeline_depth=1 / sequential-stream`。也就是说，当前 RVV target facts
确实改变了 value representation，却尚未通过 latency/throughput facts改变 local
schedule。K1 并不是只做了 codegen smoke；真机数字已存在，但标准 RVV 的
target-specific schedule 仍未成立。

IME extension 会使同一 matrix-role op 选择另一项 local instruction，说明 extension
capability 能进入选择；不过当前 capability 只描述 M1×N16×K32 与粗粒度资源，没有
描述 accumulator tuple、operand window、decode mapping 和 load schedule。当前低性能
IME helper 正是这些缺失决定的外部证据。

## 本轮闭合与未闭合

闭合的是：

```text
普通 contract 不再因缺 local operation 而崩溃
scalar-vector FMACC 与 product-subtraction FMSAC 成为可观察的 selected decision
Q4_K 在 SG2044/K1 RVV/K1 IME 三条真机路径上逐位一致
K1 不再只有“能生成 C”的证据
```

未闭合的是：

```text
当前 Q4_K IME 作者树/派生 layout 无法形成高性能连续 operand window
IME assignment 没有完整 fragment schedule，helper 仍重建局部循环与地址
IQ2_XXS 的物理 live set 超出 Clang 可无 spill 保存的范围
SG2044 与 K1 的 RVV schedule 尚未随 target latency/resource 发生结构变化
```

其中第一项触及作者树与 persistent layout，不能由 compiler 偷改；后三项属于编译空间
与 local operation contract，不是继续润色 C 变量名可以解决的问题。
