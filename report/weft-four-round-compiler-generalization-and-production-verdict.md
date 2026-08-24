# Weft 四轮编译器泛化推进与 production 判定

## 报告范围

这份报告合并连续四轮工作：

1. 将 Q4_K blocked 结构向陌生量化格式铺开，检验“新增格式只需要 encoding + std”是否成立；
2. 参考 Triton 与 TileLang，重构 RISC-V physical pass，并用结构不同的输入检验 pass 是否真实工作；
3. 分离 physical decision 与 intrinsic-C/local-asm 拼写，在 SG2044 和 K1 上检查最终汇编、spill、IME 与真机性能；
4. 对 103 个 production 项在两台机器上完整重测，给出 206 条当前结果和编译器最终判定。

四轮数字的用途不同，不能混成一张表：

- 第一至三轮的 Q4_K、IQ2_XXS、F32 meta、IME、groups4 和 GEMV 是定向或压力实验；
- 第四轮的 206 条才是统一协议下的 canonical production 结果；
- 当前性能数字以 [`weft-kernel-performance.csv`](weft-kernel-performance.csv) 前 206 个 production 行为准；后 15 行保留定向证据，但不进入 production 分档。

## 总结论

四轮最终没有证明“全格式高性能编译已经完成”，反而把边界厘清了：

1. Q4_K 的高性能路径不能机械复制到 IQ、Q4_0 或全部量化格式。不同数值树需要不同的 reduction-lane、free-axis register repetition 和 value handoff；当前 mapping 还不能统一表达。
2. RISC-V lowering 已从若干 family selector 前进到一条真实的 pass 主干，但多数 pass 在每个具体 candidate 内仍只有一个确定结果；layout conversion、computed spill、普遍 pipeline 和 measured winner 尚未闭合。
3. emitter 可以机械拼写已经闭合的 RVV local operation；但 IME leaf、IQ live set 和 scalar f32 contraction 仍暴露上游决定或语言数值合同缺口。
4. 当前 206 条中只有 204 条 bit-exact；在 152 条手写非标量 source 对照中，通过项中位仅为 16.74%，123 条低于 source 的 50%。
5. spec 2.2 没有被证伪，但也没有被充分证明。大部分慢项尚未同时满足“encoding、作者树、std 与 pass 都完整”这一证伪前提。
6. 当前 Weft 是一套真实编译器主干，不是单纯调用旧 kernel 的 DSL 包装；但从性能泛化证据看，它仍是“少数做深路径 + 大量窄或标量化路径”，还不能称为已经成立的通用高性能算子编译器。

---

# 第一轮：Q4_K blocked 结构向全格式铺开

## 1.1 本轮问题

Q4_K 已经有一条成熟的派生布局路径：

```text
16-output cohort
grouped/layered packed input
integer MAC
scale/min 两条数值支路
跨 K-block accumulator
```

这一轮没有继续优化 Q4_K，而是选择结构差异更大的输入，检验以下命题：

> 如果 backend 已经泛化，那么新增格式应主要是 encoding 声明和 std 数值函数；不应每加入一个格式就修改一次后端。

采用的三个外部输入是：

| 输入 | 压力点 |
|---|---|
| Q4_K | 已成熟的 grouped/layered direct packed MAC |
| IQ2_XXS | grid/sign lookup、8-element codebook reduction、subscale |
| Q4_0 | grouped/layered nibble，加上 `-8 * sum(q8)` correction |

这个选择直接否定了“所有量化格式只是 Q4_K 常数变化”的假设。

## 1.2 参考实现给出的正确抽象位置

对照 Triton 与 TileLang 后，真正缺少的不是更多格式 leaf，而是 operation-local mapping：

```text
被消去的 reduction axis
    → 可以成为 SIMD lane

仍保留的 free axes
    → 可以成为 register repetitions / accumulator tuple

reduce 后的值
    → 必须保留其剩余 free-axis 映射
```

Triton 的 reduce result 会从 parent layout 切片得到新的 encoding；消去轴不再是 result shape 的普通轴，但其物理分布关系不会丢失。layout conflict 也不是通过 whole-kernel matcher 解决，而是显式插入 conversion，再由后续 pass 消除可以消除的 conversion。

当前 Weft 当时只能在 result value 上保存一个仍存在的 `lane_axis`，因此无法同时表示：

- IQ2_XXS：codebook 的 8 个 reduction element 进入 SIMD lane，多个 output 成为 register repetitions；
- Q4_0：K reduction 被消去后，结果仍需保留 M/N 两个 free axes。

## 1.3 IQ2_XXS 作者树与后端压力

本轮加入了 explicit blocked local-pack IQ2_XXS 作者程序，包含：

```text
NC / KC / MC / MR blocking
canonical IQ2_XXS panel 的 kernel-local pack
16-output cohort
Q8_K activation quantize 与 workspace
grid / sign / subscale / 8-element codebook 数值树
跨 K block accumulator
0.125 epilogue
```

它没有调用 GGML 或 materials；blocked std 的 grid/sign ABI 使用 i8 table，以便 lookup 结果再由普通 widen 进入 i32 计算。

为了让该树能够 lowering，本轮实际逼出了这些共享能力：

- local pack 只读取作者显式 `along`，不从 consumer closure 猜 pack axis；
- 区分 widening 与同宽 cast；
- lane tie-break 按真实 cohort 选择，N16 优先于 MR2；
- rank-2 wide value 的另一 rows/cols 轴进入 register repetition；
- scalar-tuple load 使用 per-use dynamic index；
- 合法 vector one-use `mul → add` 由 operation pass 形成 fused cluster，emitter 只拼写 `vfmacc`。

IQ2_XXS blocked tree 在 SG2044 和 K1 都得到 bit-exact 结果，但性能并未自然进入高性能区间：

| target/path | decode GOP/s | source | ratio | prefill GOP/s | source | ratio |
|---|---:|---:|---:|---:|---:|---:|
| SG canonical | 1.911395 | 4.067610 | 46.99% | 1.863710 | 4.239853 | 43.96% |
| SG blocked local-pack | 0.553423 | 4.067610 | 13.61% | 1.309321 | 4.239853 | 30.88% |
| K1 blocked local-pack | 0.509911 | 1.664643 | 30.63% | 1.178748 | 1.705884 | 69.10% |

这些是定向 cold 结果，`repetitions=3`、`warmup=0`；source 数字复用当时 CSV 的同 shape baseline，不是第四轮统一协议重测。

生成代码暴露了具体原因：IQ2 内层每个 8-element grid entry 发出 8 次 byte gather，因为 output N 被放进 SIMD lane。手写高性能 IQ2 的方向相反：8-element codebook reduction 进入 lane，多个 output 进入 register repetitions。

## 1.4 Q4_0 反例

Q4_0 试验复用了 16-output local pack 与 encoded widening MAC，并写出：

```text
raw      = dot(unsigned_q4, signed_q8)
centered = raw - 8 * sum(q8)
```

但 `sum(q8)` 被错误映射成单一 scalar 并广播，没有保留 M free axis，导致数值错误。该入口和失败路径已删除，没有性能数字。

这说明 Q4_0 并不只是 Q4_K 的较小 block：reduce 结果仍带 free axes 时，当前 representation 无法表达其 tuple mapping。

## 1.5 对 dense 资源模型的反证

rank-2 register repetition 被正确计入后，旧 F32 candidate 的资源问题也暴露出来：

- SG2044 `NC=32, MR=4` 的资源峰值达到 66，超过 32 个向量寄存器组；
- 临时缩小到 `NC=16, MR=2` 可以 bit-exact，但仅 0.105278 GOP/s，低于此前 0.234716 GOP/s；
- `acc += outer_contract(...)` 尚未形成真正 tied/in-place accumulator handoff；
- candidate enumeration 没有适应更严格且更真实的资源计数。

## 1.6 第一轮结论

24 格式的实际状态是：

| 状态 | 格式数 |
|---|---:|
| 前序已有高性能终点 | 1：Q4_K |
| 本轮新增双机 bit-exact blocked 路径 | 1：IQ2_XXS，但性能差 |
| 写过后因数值错误删除 | 1：Q4_0 |
| 尚未铺开 | 21 |
| “只改 std 即证明高性能” | 0 |

Q4_K 当时的定向证据仍有效，但只属于明确的作者树和 scope：

| Q4_K path / SG2044 | Weft GOP/s | source | ratio |
|---|---:|---:|---:|
| persistent Q4K_I16 decode | 9.388334 | 9.555431 | 98.25% |
| persistent Q4K_I16 prefill | 11.108464 | 9.720730 | 114.28% |
| canonical local-pack decode | 1.260617 | 9.555431 | 13.19% |
| canonical local-pack prefill | 10.505987 | 9.720730 | 108.08% |

persistent pack 不计入 kernel，local pack 计入；二者也不是同一 ABI。它们不能覆盖 canonical production 行。

第一轮最重要的产出不是新增格式数量，而是把后端缺口定位为：

```text
operation-local eliminated-axis mapping
multi-free-axis register tuple
reduce / contract result handoff
显式 layout conversion
一致的 live-range 与资源计数
```

第一轮结束时实现尚未收敛，没有提交为稳定节点；这些缺口成为第二轮重构的输入。

---

# 第二轮：建立真实工作的 RISC-V physical pass

## 2.1 参考 Triton 与 TileLang 后采用的组织

本轮不再用“文件拆成多个 pass”作为成功判据，而是检查每个 pass 是否持有真实表示、能否在不同输入上得到不同结果。

从 Triton 得到的约束：

- layout propagation 分为 anchor 初始化、沿 users 传播、冲突识别与 conversion insertion；
- conversion 之后再按 dominance 与 use-def 消除冗余；
- allocation 读取 typed size、liveness、offset 与 alias，不用一个静态常数代表资源。

从 TileLang 得到的约束：

- layout 是 logical coordinate 到 physical coordinate 的映射；reshape、merge 和 conflict 必须显式处理；
- pipeline 需要 read/write region、scalar def/use、依赖顺序、stage、条件和 last consumer；
- 单独保存 `pipeline_depth=2` 不等于已经生成 pipeline；
- tuner 只枚举调用者提供的配置，不发明作者 tree。

## 2.2 当前 pass 主干与真实职责

```text
Kernel IR
→ ConstructRISCVProblems
→ AssignRISCVRepresentations
→ SelectRISCVLocalOperations
→ ScheduleRISCVLevels
→ CheckRISCVResources
→ SelectRISCVWinner
→ intrinsic-C / local-asm emission
```

| pass | 本轮真实输入与输出 | 当前结果空间 |
|---|---|---|
| Construct problems | std `auto` 值、Level unroll/pipeline choice | 真正展开笛卡尔积；F32 MR={2,4}、NR={4,8,16,32} 形成 8 candidates |
| Assign representations | producer/user/handoff、axis、memory relation、VLEN | 每个 candidate 一个确定 mapping；不同输入可得到不同 lane/register/stream |
| Select local operations | primitive、typed operands、selected mapping、target facts | 每个 op 一个 realization；memory form、tied/deferred facts可不同 |
| Schedule Levels | Level dependency 与 candidate 的 depth/unroll | 只有 depth 1/2；不是每个 Level 都有第二种结构 |
| Check resources | liveness、register groups、primitive temporary、fragment | legal、reload admitted value 或 invalid；computed value 不能 spill |
| Select winner | 所有 legal candidates 的 resource cost | 静态选择最小 peak；未接 compile-and-measure |
| emitter | frozen assignment | 不再选择；无法投影时明确失败 |

“pass 已存在”仍不等于选择空间完整：

- LMUL 仍是第一个足够合法值，不枚举其它合法 LMUL；
- 多数 operation domain 只有一个 realization；
- matrix fragment 取第一个合法值；
- IME 只有 M1×N16×K32；
- layout conflict 无 transient conversion insertion/elimination；
- pipeline 只有 sequential 与一个局部 double-buffer 形态；
- spill 只能把可重读 admitted value 改为 reload-per-use。

## 2.3 representation 与 memory 的具体变化

`AssignRISCVRepresentations` 开始沿 use-def 读取：

```text
显式 reduce/contract axis relation
→ 相连 memory layout
→ dense 最末轴或 packed 连续性
→ Level cohort
```

结果包括：

- F32 `[M,N]` 从错误的 M-lane 修正为 N-lane；
- lane mapping 传播到 admitted operands、accumulator 和 commit；
- reduction 消去轴记录为 `eliminated_axis`；
- 其它 free axes 记录为 `register_axes/register_extents`。

`SelectRISCVLocalOperations` 再依据 selected lane 与声明布局产生：

```text
unit-stride
runtime-strided
record-address
scalar-address
```

dense outer-contract 的 lane operand、lane axis、reduction axis、memory form 和 temporary budget 也在 emission 前写入 assignment。

emitter 统一使用 `projectPart` / `projectRegisterPart` 投影完整坐标；无法投影时不再猜测，而是明确要求 layout conversion。

## 2.4 tied accumulator 与资源核算

单 consumer 且 mapping 一致的：

```text
outer_contract → add(accumulator, product)
```

可以形成 deferred product 与 accumulator update；`admit → new` 也可形成 tied initialization。dense outer FMA 的 lane operand temporary 被单独计入资源。

这修复了旧资源漏算，但也让过大的 microtile 正确变成非法。例如 SG2044 MR4×NR32 会因 accumulator、lane temporary 和 reserved groups 超预算被删除。

## 2.5 跨输入真机结果

### F32 blocked GEMM

| target | candidate | Weft GOP/s | source | ratio |
|---|---|---:|---:|---:|
| SG2044 | MR4×NR16 | 6.665204 | 7.045805 | 94.60% |
| K1 | MR4×NR32 | 2.479803 | 2.635297 | 94.10% |

这证明同一 std tree 可以因 target facts 得到不同的最佳参数，但也直接反证了静态 winner：

- SG2044 资源检查允许 7/8 candidates，却选择 MR2×NR4；
- 真机更快的是 MR4×NR16；
- K1 真机更快点又变成 MR4×NR32；
- `min(register peak)` 不是性能代价模型。

SG2044 的早期 candidate 扫描为：

| candidate | GOP/s |
|---|---:|
| MR4×NR4 | 5.162587 |
| MR4×NR8 | 6.151110 |
| MR4×NR16 | 6.881550 |
| MR2×NR32 | 5.456639 |

MR4×NR32 因资源超预算没有数字。

### IQ2_XXS local-pack

| target | phase | before | after | change |
|---|---|---:|---:|---:|
| SG2044 | decode | 0.553423 | 0.779411 | +40.8% |
| SG2044 | prefill | 1.309321 | 2.001647 | +52.9% |
| K1 | decode | 0.509911 | 0.510604 | 基本不变 |
| K1 | prefill | 1.178748 | 1.182625 | 基本不变 |

作者 tree 未改。SG 改善说明 use-def mapping 与 fused cluster 有真实作用；K1 基本不变说明 target-specific mapping、schedule 与 system compiler interaction 仍未闭合。

### Q4K_I16 persistent

| phase | before | after | 状态 |
|---|---:|---:|---|
| decode | 9.388334 | 9.389858 | bit-exact，基本不变 |
| prefill | 11.108464 | 10.961397 | bit-exact，小幅波动 |

新 representation 没有破坏已成熟的 grouped/layered 路径。

## 2.6 没有走通的外部输入

- `online_flash_attention`：在 dense `materialize(admit(...))` 处因缺 staging realization 被 operation pass 判非法；
- `top_k_f32`：assignment 完整，但 emitter 尚不能生成 sequential aggregate `new/update/while`；
- 早期 `q4_k_gemv`/groups4 手工 repro：可以生成和运行，但 row 0 出现 `expected=12384.4346, actual=12384.4219`，不能算数值通过。

这些 probe 不属于第四轮 206 个 production 项，第四轮没有重新证明它们已解决。

## 2.7 第二轮结论

本轮证明 representation/memory/resource 主干对 dense、grouped packed、codebook 三类输入产生了真实横向变化；同时也证明以下内容仍未成立：

```text
layout conflict conversion
measured winner
attention staging
ordinary scalar aggregate emission
computed aggregate spill
普遍 loop-local pipeline
```

第二轮形成了稳定代码节点，但不是“完整 physical space 已完成”的节点。

---

# 第三轮：发射质量与 K1 真机

## 3.1 本轮边界

本轮区分：

```text
前序 pass 决定机器实现
已经决定的机器实现如何写成 intrinsic C / local asm
```

检查对象不是 emitter 文件数量，而是生成代码经 SG2044 GCC 15 与 K1 Clang 18 后实际留下的：

```text
指令
栈对象
spill / reload
vlenb dependency
fused operation
IME vmadot
真机吞吐
```

## 3.2 普通 contract 的上游 decision 缺口

F32 GEMV physical assignment 原本可以完成，但 intrinsic-C emission 会崩溃。原因不是 GEMV 名称或固定 shape，而是普通 `contract` 只有 `rvv.reduction-product`，没有：

```text
lane operand
lane axis
reduction axis
lane memory form
```

真实关系为：

```text
W[M,K] × X[K] → Y[M]
lane axis       = M
reduction axis  = K
lane operand    = W
memory form     = runtime-strided
```

修复后，local-operation pass 沿 operands 与 admit/slice use-def 写入这些事实；emitter 不再默认 lhs 或读取空 decision。普通 contract 因而进入与 outer-contract 相同的 operation-decision 主干。

## 3.3 scalar-vector FMACC 与 subtraction FMSAC

Q4_K RVV 和 IME 在 K1 曾产生相同的 f32 epilogue 差异：

```text
d * scaled - dmin * minimum
acc + ds * difference
```

K1 reference 会形成 `fmsub.s` 与 `fmadd.s`；旧 generated C 将其拆成独立 vector intrinsics，系统编译器不能跨 intrinsic call 恢复 contraction。

正确修复位置不是 emitter 猜模式，而是 operation selection：

- product-subtraction 选择 `vfmsac.vv`；
- scalar-vector accumulation 选择 `vfmacc.vf`；
- assignment 记录 `instruction` 与 `operand_form`；
- emitter 只拼写所选 intrinsic。

## 3.4 双机定向真机结果

统一使用 `-O3 -ffp-contract=fast`：SG2044 为 GCC 15.2/VLEN128，K1 为 Clang 18/VLEN256。

| kernel/target | median ms | Weft GOP/s | 同 shape source | ratio |
|---|---:|---:|---:|---:|
| Q4_K vec-dot / SG RVV | 13.864344 | 8.470687 | 9.728344 | 87.07% |
| Q4_K vec-dot / K1 RVV | 42.192988 | 2.783413 | 2.495786 | 111.52% |
| Q4_K vec-dot / K1 IME | 321.734552 | 0.365023 | 无 | — |
| groups4 / SG RVV | 11.930215 | 9.843956 | 9.728344 | 101.19% |
| groups4 / K1 RVV | 37.399004 | 3.140204 | 2.495786 | 125.82% |
| F32 GEMV / SG | 764.223475 | 0.153673 | 无 | — |
| F32 GEMV / K1 | 100.424059 | 1.169446 | 无 | — |

SG Q4_K 曾在同代码状态下出现 3.458880 GOP/s 异常值；机器负载恢复后复跑五次，中位回到 8.470687，定向 CSV 行记录后者并注明 repetitions=5。

`mac_groups(4)` 改变作者树中 i16 partial 的数量和覆盖范围，但复用相同 f32 epilogue与 operation-cluster；没有新增 format selector 即在两台机器逐位一致并达到 101–126%。它是局部 lowering 不依赖 `mac_pairs` 固定 closure 的正向证据。

这些数字属于派生/grouped 作者树，不是第四轮 canonical `quantized_vec_dot_q4_k_q8_k` production 行。

## 3.5 K1 汇编证据

| generated C | asm lines | `vlenb` | vector spill/reload | fused vector op | `vmadot` |
|---|---:|---:|---:|---:|---:|
| F32 GEMV | 97 | 0 | 0 / 0 | 1 | 0 |
| Q4_K RVV | 963 | 0 | 0 / 0 | 2 | 0 |
| Q4_K IME | 1443 | 0 | 0 / 0 | 2 | 1 |
| IQ2_XXS local-pack | 2572 | 34 | 17 / 17 | 2 | 0 |

由此能得到四个具体结论：

1. Q4_K RVV 的 selected representation 被 Clang 保留：没有动态 VLEN spill，`vfmsac` 与 `vfmacc` 真正出现；
2. IQ2_XXS 同一 live region 中同时存活 quantize chunk、decode temporary、两个 accumulator 与 carry，Clang 因而对 17 个 unknown-size vector value做成对 spill/reload；
3. K1 IME 虽没有 vector spill，仍比 RVV 慢 7.63 倍，因为 helper 里有四重 scalar address/decode loop、栈上 `output[16]/activation[8]/weights[32]`，循环中只有一处 `vmadot`；
4. K1 GEMM 中的 `memcpy` 是 Clang 对已选 row-pack loop 的识别，同函数没有 vector spill；不能见到 `memcpy` 就一律禁止。

## 3.6 无收益 emitter 实验

曾将 selected little-endian f16/f32 load/store 从固定长度 `memcpy` 改成显式 byte assembly 与 union conversion：

| target/path | before | after |
|---|---:|---:|
| SG Q4_K | 8.438 | 8.300 |
| K1 Q4_K | 2.778 | 2.786 |
| K1 IME | 0.365 | 0.362 |

没有稳定收益，改动已撤销。这个结果说明主要差距不在 load helper 的表面拼写。

## 3.7 IME 的真实边界

当前 Weft IME tree：

- 保留 Q4_K scale/min 两支路；
- Q4K_I16 仅把 canonical record 做 16 行 byte-major interleave；
- 每个 sub32 调用一次 raw q×q8 matrix primitive；
- scale/min 与 f32 accumulation 仍在外层 DSL tree。

GGML K1 donor 则：

- 把 block 拆成 8 个 Q4_1×16；
- 预先形成 fp16 `d*sc` 与 `-dmin*m`；
- q payload 变成连续 row-major nibble tile；
- leaf 拥有多个 accumulator、连续 B window 与跨 K register organization。

二者不是相同 logical encoding 的简单 byte-order 差异。donor 改变了逻辑中间值、fp16 舍入位置、persistent layout 和 ABI；按 spec 2.2 必须由作者 std 特化明确表达，compiler/emitter 不能自动补。

当前 target capability 也只有 M1×N16×K32 与粗粒度资源，缺少 accumulator tuple、operand window、decode mapping 和 load schedule。因此“能生成 vmadot”不等于 IME 已进入高性能 candidate space。

## 3.8 第三轮结论

本轮闭合了：

- 普通 contract 不再因缺 local operation 崩溃；
- scalar-vector FMACC 与 product-subtraction FMSAC 成为 assignment 中的显式决定；
- Q4_K 定向 RVV/IME 路径在 SG/K1 真机逐位一致；
- K1 不再只有 codegen 证据，而有真实 runtime、asm 和性能证据。

仍未闭合：

- IME 作者树与 persistent layout 无法形成 donor 的连续 operand window；
- IME assignment 没有完整 fragment schedule，helper 仍重建局部 loop/address；
- IQ2_XXS live set 超出 K1 Clang 可以无 spill 保存的范围；
- SG/K1 的 RVV schedule 仍同时是 `unroll=1/pipeline_depth=1/sequential-stream`，没有读取 target latency/throughput 形成不同结构。

---

# 第四轮：双机 production 全量重测

## 4.1 固定语料

每台机器 103 条，双机共 206 条：

| family | 每台条数 | 内容 |
|---|---:|---|
| `MUL_MAT` | 52 | f32、f16、24 种量化格式，各有 decode/prefill |
| quantized vec-dot | 24 | 24 个 weight/activation typed pair |
| activation quantize | 3 | Q8_0、Q8_1、Q8_K |
| row dequantize | 24 | 24 种量化格式 |

## 4.2 硬件与统一协议

| hardware | target | compiler | fixed CPU |
|---|---|---|---:|
| SG2044 | RV64GCV, VLEN128 | GCC 15.2 | 48 |
| K1/X60 | RV64GCV, VLEN256 | Clang 18.1.8 + external assembler | 3 |

Weft 与 GGML wrapper 使用相同：

```text
-O3
-ffp-contract=fast
-march=<target>
-mabi=lp64d
```

K1 额外使用 `-fno-integrated-as`。远端 GGML library 的 `quants.c` 实际编译命令也包含对应 flags，不是只给 wrapper 增加选项。

每项执行：

```text
1 次未计时数值比较
3 次计时
每次前驱逐 64 MiB
固定单核
```

K-family 在 Weft std 名称中用小写 `_k`，GGML runtime 用大写 `_K`。初次 orchestration 的 30 个 source 项大小写错误，随后只补跑对应 source 对照并替换日志；Weft 数字没有替换。

## 4.3 数值结果

```text
SG2044：103 / 103 bit-exact
K1：     101 / 103 bit-exact
合计：   204 / 206 bit-exact
```

两条 FAIL 都是 K1 prefill；CSV 不记录 Weft 性能：

| kernel | mismatch | source throughput |
|---|---|---:|
| Q2_K `MUL_MAT` | output 12288：`-0x1.c72dc4p+0` vs `-0x1.c72dcp+0` | 2.452942 GOP/s |
| IQ4_XS `MUL_MAT` | output 4096：`0x1.1a7c5ap+17` vs `0x1.1a7c5cp+17` | 1.718190 GOP/s |

activation workspace 已先通过 bit-exact，因此它们不是 encoding byte mapping 错误，而是最终 f32 accumulation 的 contraction/舍入边界差异。

当前 scalar `BinaryOp` emitter 会把独立 SSA op 内联成一个 C 表达式，由 K1 Clang 在 `-ffp-contract=fast` 下选择 fusion。曾试过强制所有 scalar f32 op 物化并阻断 contraction：Q2_K 通过，但 IQ4_XS 由 row 1 的 1-ulp 差异变成 row 0 更大的差异，所以实验没有保留。

这暴露了尚未冻结的数值合同：

> 独立 f32 SSA op 是否必然形成舍入边界；若允许 contraction，哪些 op 明确授权。

在合同明确前，不能由 emitter 按格式或 reference assembly 私自选择。

## 4.4 全 production 分布

204 条 bit-exact 项分档，2 条 FAIL 单列：

| range | count |
|---|---:|
| `≥90%` | 16 |
| `70–90%` | 14 |
| `50–70%` | 17 |
| `<50%` | 157 |
| `FAIL` | 2 |

通过项整体中位 Weft/source 为 **23.61%**。

按 target：

| target | exact | median | `≥90%` | `70–90%` | `50–70%` | `<50%` | FAIL |
|---|---:|---:|---:|---:|---:|---:|---:|
| SG2044 | 103 | 30.92% | 9 | 6 | 13 | 75 | 0 |
| K1 | 101 | 19.72% | 7 | 8 | 4 | 82 | 2 |

按 family：

| family | exact | median | `≥90%` | `70–90%` | `50–70%` | `<50%` | FAIL |
|---|---:|---:|---:|---:|---:|---:|---:|
| `MUL_MAT` | 102 | 16.68% | 7 | 3 | 7 | 85 | 2 |
| quantized vec-dot | 48 | 17.16% | 4 | 1 | 4 | 39 | 0 |
| activation quantize | 6 | 75.26% | 1 | 2 | 1 | 2 | 0 |
| row dequantize | 48 | 35.90% | 4 | 8 | 5 | 31 | 0 |

row dequantize 的 source implementation 是 scalar reference，不能据此证明接近手写 intrinsic。

## 4.5 只看手写非标量 source

排除 54 条 scalar source donor 后，剩 152 条：150 条 exact，2 条 FAIL。

| target | exact/total | median | `≥90%` | `70–90%` | `50–70%` | `<50%` | FAIL |
|---|---:|---:|---:|---:|---:|---:|---:|
| SG2044 | 76/76 | 30.92% | 8 | 4 | 12 | 52 | 0 |
| K1 | 74/76 | 10.46% | 1 | 2 | 0 | 71 | 2 |
| 合计 | 150/152 | 16.74% | 9 | 6 | 12 | 123 | 2 |

直接答案是：

- 只有 9/152 条达到 source 的 90%；
- 123/152 条低于 source 的 50%；
- K1 有 71/76 条低于 50%，通过项中位只有 10.46%。

## 4.6 production 归因聚类

### 作者树没有写出 blocked production algorithm

24 个 quantized `MUL_MAT` canonical tree 仍是：

```text
activation quantize
→ for row
→ for column
→ 单输出 vec-dot
```

没有 output cohort、MR×NR accumulator、activation reuse、packed panel 或 prefill staging。F16 tree 也没有 dense blocked staging/convert。两类合计影响 100 个双机 phase 项。

这些结构改变逻辑值集合、accumulator scope 与 Level 归属，按 spec 2.2 必须由 std 作者写出，compiler 不能自动补。

### physical mapping 仍大量 scalar-indexed

K-series、IQ/codebook 和 packed extract 常见：

```text
encoded-access.scalar-indexed
scalar.lookup
pipeline_depth = 1
unroll = 1
```

尚缺：

- reduction 后保留多 free axes 的 register tuple；
- packed/index/codebook mapping 沿 use-def 传播；
- incompatible consumer layout conversion；
- computed aggregate spill；
- 普遍跨 Level iteration pipeline。

SG 最慢十项中的 Q2_K/Q3_K/Q4_K/Q6_K vec-dot 或 `MUL_MAT` 只有约 2.6–3.7%。

### candidate selection 没闭合

F32 使用同一 std tree：

| target | production fixed meta | source | ratio | measured better meta | ratio |
|---|---:|---:|---:|---:|---:|
| SG2044 | 5.410476 | 7.176430 | 75.39% | MR4×NR16：6.665204 | 94.60% |
| K1 | 1.900692 | 2.640165 | 71.99% | MR4×NR32：2.479803 | 94.10% |

resource pass 将 peak register groups 作为 cost，production runner 又固定 MR4×NR4；已有 tuner 能扫描并打印 winner，但 winner 没成为 production artifact 的选择。

### target facts 没贯穿 memory/schedule/lookup

VLEN128/256 会改变 lane 数与 intrinsic type，但大量 kernel 在两台机器仍得到同类 `scalar-indexed + pipeline 1`：

- K1 手写 RVV 对照通过项中位 10.46%；
- K1 IQ2_XS prefill 0.063924 vs 1.809399，只达到 3.53%；
- K1 IQ2_XXS prefill 0.061264 vs 1.707999，只达到 3.59%；
- K1 多数 IQ vec-dot 只有 3.6–6.8%。

### Q8_K absmax 的语言表面缺口

Q8_K activation quantize 当前用一元素 `L.subs` 与一元素 admit 实现有序 scalar absmax scan。普通 scalar loop 不能直接投影 Level-owned block，scalar/wide consumer 之间也没有 layout conversion/reload 表达。

没有为此新增黑盒 primitive或后端注解。更直接写法必须先闭合“普通 scalar control 如何遍历 Level domain并读取同一 block 的 scalar projection”。

性能也仍落后：Q8_K quantize 在 SG 为 source 的 46.43%，K1 为 19.72%；Q8_0/Q8_1 分别达到 SG 66.00%/84.52%、K1 93.63%/89.09%。

---

# 四轮之间的因果关系

## 第一轮否定了什么

第一轮否定：

> 一个高性能 Q4_K backend skeleton 可以只换 encoding 与常数覆盖 24 格式。

IQ2_XXS 与 Q4_0 证明，局部 reduction axis 和剩余 free axes 的机器角色会因数值树不同而改变。后端缺的是 operation-local mapping，不是更多 format selector。

## 第二轮真正建立了什么

第二轮建立了统一 use-def/axis/memory/resource 主干，让 dense、grouped packed、codebook 三类输入真实地产生不同 assignment；但每个 candidate 内仍多是单结果 projection，layout conversion 与 measured winner 缺失。

## 第三轮关闭了哪一层

第三轮把普通 contract、subtraction FMSAC 和 scalar-vector FMACC 的决定前移，证明 emitter 可以机械拼写完整 decision；同时用 K1 asm 证明 IQ spill 与 IME 低性能不是变量名或 memcpy 的表面问题。

## 第四轮如何限制前三轮的正向结论

第三轮 Q4_K 的 8–10 GOP/s 属于派生/grouped 作者树。第四轮 canonical production `quantized_vec_dot_q4_k_q8_k` 在 SG 只有 0.311378 vs 8.941227 GOP/s，即 **3.48%**。

这不是同一条路径退化 28 倍，而是两个不同作者程序与 ABI：

- 高性能路径把 output cohort、派生 interleave 与 grouped MAC 写进 tree/encoding；
- canonical production 输入是一行 canonical View，runtime 在 kernel 外逐输出调用；
- compiler 按 2.2 无权把后者改成前者。

因此定向高性能数字只证明局部 lowering 终点存在，不能证明 canonical std 或全格式泛化已经成立。

---

# 最终三个判定

## 一、离手写 intrinsic 还差多少

在真正拥有手写非标量 source 的 152 条对照中：

```text
≥90%       9
70–90%     6
50–70%    12
<50%     123
FAIL       2
median   16.74%
```

差距大部分是工程性的，但不是“小修 emitter”：

- use-def 驱动的 lane/register mapping 仍窄；
- layout conversion 未实现；
- packed/index/codebook memory form 大量退化；
- computed aggregate 不能 spill；
- schedule 在多数 Level 上仍是 unroll 1/pipeline 1；
- compile-and-measure winner 没进入 production artifact；
- VLEN256 与 IME facts 没贯穿完整 local realization。

另一部分属于作者 tree，而不是 compiler：quant prefill 的 output blocking、activation reuse、persistent packing，以及 F16 staging/convert 当前没有写入 production std。compiler 自动添加会违反 2.2。

scalar f32 contraction/rounding 还是一个真实设计未决项，已经产生两条 K1 FAIL。

## 二、spec 2.2 是否被证伪

没有，但也没有被充分证明。

证伪要求：

```text
encoding 正确
作者 tree 已表达目标高性能算法
std 特化完整
pass 推导与 physical space 完整
```

然后仍必须由 compiler 改变逻辑值集合或 Level 归属才能达到手写性能。

四轮没有出现满足所有条件的反例：

- canonical quant `MUL_MAT` 是作者 tree 未写 blocking/reuse；
- Q4_K 高性能 tree 明确改变 encoding、cohort、persistent layout 与 ABI；
- F32 同 tree 换 measured meta 即从约 72–75% 到约 94%，属于选择问题；
- IQ/K/codebook 仍停在 scalar-indexed 与 pipeline 1，pass 前提不完整；
- 两条 K1 FAIL 连 f32 数值合同都未闭合。

所以当前只能说“未证伪”，不能写成“已由 206 条证明”。

## 三、现在是不是一个算子编译器

功能主链上，它已经是编译器：同一 canonical IR、operation decision、representation、resource 与 emitter 主干确实编译并运行多种输入，且没有调用 GGML/materials 或 legacy fallback。

但性能泛化判据尚未通过。

正向证据：同一组共享 operation-selection 改动后，SG 上多个没有独立 whole-kernel leaf 的 std 路径同时进入合理区间：

- IQ1_S vec-dot 91.93%，MUL_MAT decode/prefill 95.23%/94.82%；
- IQ3_S vec-dot 94.11%，MUL_MAT decode 100.53%；
- IQ3_XXS vec-dot 127.89%，MUL_MAT decode 134.79%。

但这还不是开发后验外推：

1. 这些格式在 backend 演进时已经存在；
2. 同批路径到 K1 后多数退到 4–7%；
3. 第一轮 24 格式 blocked 铺开中，“只改 std 即高性能”的新增格式是 0；
4. canonical Q4_K 只有 3.48%，约 100% 的是另一棵针对性派生树。

因此最终答案是：

> 当前 Weft 不是“几段旧 kernel 外面套 DSL”，因为已有共享 compiler machinery 会跨程序生效；但它也还不是性能上已经成立的通用算子编译器。现在仍拿不出一个可信的、开发后验选择的陌生格式或算子，证明只增加 encoding 与 std 数值函数，就能在 SG2044 和 K1 上都自然取得合理性能。
