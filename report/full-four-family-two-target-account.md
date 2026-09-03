# 四族双机全量性能账

## 1. 结论

本轮在同一份干净 checkout 上运行了 103 个逻辑 case，并分别在 SG2044 与 K1/X60
执行，共 206 个 Weft case。范围包括 24 个 quantized vec-dot、24 个 row-dequant、3 个
activation quantize，以及 26 个 `MUL_MAT` entry 各自的 decode/prefill，共 52 个矩阵 case。

source 的 206 个 case 全部运行成功。Weft 有 197 个 case 得到数值正确的性能结果，9 个
case 失败；失败行没有保留旧吞吐。197 个有效结果中，117 个达到或超过 source，23 个低于
source 的 50%。完整逐行数据位于：

- [`weft-kernel-performance.csv`](weft-kernel-performance.csv)
- [`ggml-riscv-kernel-performance.csv`](baseline/ggml-riscv-kernel-performance.csv)

本轮没有修改 std tree、Canonical IR、RISC-V Physical IR、pass 或 emitter，也没有把诊断
过程中观察到的问题顺手优化掉。

## 2. 测量合同

### 2.1 输入

本轮先统一了 source 与 Weft 的数据生成路径。此前的 quantized vec-dot / `MUL_MAT`
存在不可比较情况：source 从 float 量化，Weft 直接读取随机 packed bytes；IQ source 还可能
使用零填充。直接把任意 bytes 当作 encoded record 也不成立，Q8_K 的 `bsums` 与 payload
会失去语义一致性，TQ radix record 也可能不合法。因此两次使用任意 raw bytes 的预跑结果
全部丢弃，没有进入 CSV。

正式运行使用以下合同：

| family | 输入合同 |
|---|---|
| quantized vec-dot | 两边用相同固定 float 公式生成一个 weight record 和 activation，走相同 GGML quantizer 得到合法 encoded record，再复制到完整 workload |
| quantized `MUL_MAT` | weight 使用相同合法 encoded record；activation 使用相同固定 float 序列；kernel 内 activation quantize 同时计入两边 |
| row-dequant | 两边以相同固定 seed 生成 record，筛到 finite output 后复制；runner 比对 policy、seed、attempt 与 output sample |
| activation quantize | 两边使用相同的 `index % 31` float 序列 |
| dense F16/F32 | 两边使用相同固定 float 输入 |

这里的“相同”由相同生成公式、seed、quantizer 路径及 paired metadata 保证；orchestrator 没有把
一份共享二进制 buffer 同时喂给两个进程。仓库禁止 hash/checksum，因此本轮没有增加 digest
校验。这个边界记录在此，不能把 paired metadata 说成一次独立的逐 byte 证明。

### 2.2 shape、计时与正确性

| family | shape | timed region |
|---|---|---|
| vec-dot | `M=1, N=14336, K=4096` | 完整 projection，activation 已预量化 |
| row-dequant | `N=1024, K=4096` | 完整 tensor dequantize |
| activation quantize | `M=128, K=14336` | 完整 activation quantize |
| `MUL_MAT` decode | `M=1, N=4096, K=4096` | 完整 op graph，含 activation quantize |
| `MUL_MAT` prefill | `M=128, N=4096, K=4096` | 完整 op graph，含 activation quantize |

每项先 warmup 一次；每次计时前遍历 64 MiB eviction buffer；运行 10 次并取 wall-time median。
矩阵和 vec-dot 按 `2MNK` 计算 GOP/s，quantize/dequantize 按 logical elements 计算
MElements/s。浮点结果使用预先固定的 `absolute=1e-4`、`relative=2e-3` 容差并检查明显错误；
离散 Q8 quantize 使用 bit-exact。

### 2.3 toolchain 与 target

source、Weft kernel 与 runtime 都使用 Clang 18.1.8，公共 flags 为
`-O3 -ffp-contract=fast -mabi=lp64d -Wall -Wextra -Werror`，并使用同一 target 的
`-march`：

- SG2044：VLEN128，固定 core 48，标准 RVV；
- K1/X60：VLEN256，固定 core 3，标准 RVV；本轮没有把 IME case 混入 RVV 分布。

run 前后都检查了工作树干净且 checkout identity 未变化。orchestrator 本身没有执行默认 build；
run 后执行 `cmake --build build` 时只重新运行了声明生成步骤，没有重编或重链接
`weft-compile`，所以本轮没有使用旧 compiler executable。把 build 放在 orchestrator 外仍是本次
测量过程与 [`protocol.md`](../doc/experiments/protocol.md) 的一个流程差异。

本轮 raw record 没有保存 OS/kernel、CPU identity、frequency/governor 的完整快照；target、core、
VLEN、toolchain 与 flags 则由 runner 固定。source 与 Weft 在同一 target/core 上逐 case 相邻执行，
因此本报告使用同轮比值定位当前缺口，但不能把这些数字与缺少同样环境事实的其它日期结果直接
拼接。CSV 的 `configuration` 是固定 runner binding 的标签，不是从最终 Physical IR 反推布局的
证明；本报告中的 carrier/memory-form 归因来自当轮临时生成的 IR/C/assembly。

正式协议还规定任一 case 失败时不替换当前 Weft CSV。本轮用户明确要求编译/运行失败也进入
全局账、且不得保留旧数字，因此 CSV 是一份包含 9 个显式 FAIL 的完整审计快照；失败行的
`median_ms`、`throughput` 和 `throughput_unit` 均为空。baseline CSV 中本轮重测的 206 行已替换，
不属于本 manifest 的 38 行保持不动。

## 3. 分布

列顺序为 `>=100% / 90–100% / 70–90% / 50–70% / <50% / FAIL`。

| target | family | >=100% | 90–100% | 70–90% | 50–70% | <50% | FAIL |
|---|---|---:|---:|---:|---:|---:|---:|
| SG2044 | `MUL_MAT` | 23 | 9 | 6 | 2 | 9 | 3 |
| SG2044 | vec-dot | 8 | 6 | 4 | 1 | 5 | 0 |
| SG2044 | row-dequant | 16 | 2 | 3 | 0 | 3 | 0 |
| SG2044 | activation quantize | 2 | 0 | 1 | 0 | 0 | 0 |
| **SG2044 合计** |  | **49** | **17** | **14** | **3** | **17** | **3** |
| K1/X60 | `MUL_MAT` | 27 | 6 | 8 | 1 | 5 | 5 |
| K1/X60 | vec-dot | 15 | 2 | 5 | 0 | 1 | 1 |
| K1/X60 | row-dequant | 24 | 0 | 0 | 0 | 0 | 0 |
| K1/X60 | activation quantize | 2 | 0 | 1 | 0 | 0 | 0 |
| **K1/X60 合计** |  | **68** | **8** | **14** | **1** | **6** | **6** |
| **双机合计** |  | **117** | **25** | **28** | **4** | **23** | **9** |

`MUL_MAT` 再按 phase 拆开：

| target / phase | >=100% | 90–100% | 70–90% | 50–70% | <50% | FAIL |
|---|---:|---:|---:|---:|---:|---:|
| SG decode | 10 | 7 | 3 | 1 | 5 | 0 |
| SG prefill | 13 | 2 | 3 | 1 | 4 | 3 |
| K1 decode | 16 | 2 | 5 | 0 | 2 | 1 |
| K1 prefill | 11 | 4 | 3 | 1 | 3 | 4 |

## 4. 失败项

### 4.1 IQ2 staged prefill：6 个编译失败

`IQ2_S`、`IQ2_XS`、`IQ2_XXS` 的 staged prefill 在两台机器都停在同一条 Physical IR
合法性错误：

```text
selected RVV widening dot has no legal typed operands,
scalar free-axis result, or target widening shape
```

失败点分别是 `python/weft/std/mul_mat.py:226`、`:155`、`:122` 的 shaped `contract`；
`LowerRISCVComposites.cpp:2127-2157` 无法让 MR/NR free axes、reduction lane slices 与 selected
widening carrier 闭合。它们是同一个 P1/R layout-legality 簇，不是三份独立格式错误。当前证据
没有表明作者数值树错误；编译器尚未产出可运行 artifact，因此不记录吞吐。

### 4.2 IQ3_XXS / K1：3 个运行失败

K1 的 standalone vec-dot、`MUL_MAT` decode 和 prefill 都以 exit 135（SIGBUS）结束。相同
entry 在 SG 可执行，因此不是 canonical 数值树普遍不可编译。

生成 C 对 `IQ3_XXS.metadata` 发出 `vle32`；record 大小为 98 bytes、声明 alignment 为 2，
metadata 起始 offset 为 66。连续 record 的奇数项地址只能保证 2-byte alignment。source donor
先用 `memcpy` 把 metadata 放入对齐的本地 `u32` 再加载。因而当前最具体的错误实体是
`metadata` memory edge 的 selected `vle32`，即 P3 alignment/memory-form 合法性缺口。运行日志
没有保存 fault PC，所以报告不把静态地址证明夸大成已捕获的精确 PC；但该 load 已经违反已知
record alignment，必须视为 compiler correctness bug，而不是性能噪声。

## 5. 低于 70% 的完整账

除 row-dequant 外吞吐单位均为 GOP/s；row-dequant 为 MElements/s。

| target | family/phase | kernel | source | Weft | ratio |
|---|---|---|---:|---:|---:|
| K1/X60 | vec-dot/decode | IQ1_M | 1.435 | 0.246 | 17.1% |
| K1/X60 | `MUL_MAT` prefill | IQ1_M | 1.447 | 0.253 | 17.5% |
| K1/X60 | `MUL_MAT` decode | IQ1_M | 1.424 | 0.253 | 17.8% |
| K1/X60 | `MUL_MAT` decode | F32 | 2.012 | 0.564 | 28.1% |
| K1/X60 | `MUL_MAT` prefill | IQ1_S | 2.741 | 0.893 | 32.6% |
| K1/X60 | `MUL_MAT` prefill | F32 | 2.574 | 1.108 | 43.0% |
| K1/X60 | `MUL_MAT` prefill | Q2_K | 2.453 | 1.570 | 64.0% |
| SG2044 | `MUL_MAT` prefill | IQ3_XXS | 2.920 | 0.284 | 9.7% |
| SG2044 | vec-dot/decode | IQ3_XXS | 2.674 | 0.283 | 10.6% |
| SG2044 | `MUL_MAT` decode | IQ3_XXS | 2.618 | 0.283 | 10.8% |
| SG2044 | `MUL_MAT` prefill | IQ1_S | 4.855 | 0.637 | 13.1% |
| SG2044 | vec-dot/decode | IQ1_M | 2.650 | 0.401 | 15.1% |
| SG2044 | `MUL_MAT` decode | IQ1_M | 2.639 | 0.422 | 16.0% |
| SG2044 | `MUL_MAT` prefill | IQ1_M | 2.634 | 0.422 | 16.0% |
| SG2044 | `MUL_MAT` prefill | F32 | 7.433 | 1.529 | 20.6% |
| SG2044 | row-dequant | TQ2_0 | 673.603 | 144.124 | 21.4% |
| SG2044 | row-dequant | Q1_0 | 746.193 | 240.393 | 32.2% |
| SG2044 | vec-dot/decode | Q3_K | 7.490 | 2.797 | 37.3% |
| SG2044 | vec-dot/decode | Q1_0 | 11.448 | 4.300 | 37.6% |
| SG2044 | `MUL_MAT` decode | Q3_K | 7.214 | 2.825 | 39.2% |
| SG2044 | row-dequant | TQ1_0 | 668.583 | 285.809 | 42.7% |
| SG2044 | `MUL_MAT` decode | Q1_0 | 10.372 | 4.522 | 43.6% |
| SG2044 | vec-dot/decode | Q4_K | 10.143 | 4.626 | 45.6% |
| SG2044 | `MUL_MAT` decode | Q4_K | 9.619 | 4.617 | 48.0% |
| SG2044 | vec-dot/decode | TQ1_0 | 8.124 | 4.627 | 57.0% |
| SG2044 | `MUL_MAT` decode | TQ1_0 | 8.015 | 4.639 | 57.9% |
| SG2044 | `MUL_MAT` prefill | Q2_K | 9.658 | 5.734 | 59.4% |

## 6. 原因聚类

P1–P5 在这里仅作为工作账维度，不作为先验预测模型。

### 6.1 IQ1 作者程序：8 个 `<50%` case

IQ1_M 的 standalone、decode、prefill 在两台机器全部只有 15.1%–17.8%。
`vec_dot_iq1_m_q8_k` 的 main product 已写成 `group × scale_part × entry × payload`，但
`vec_dot.py:504-548` 的 correction 仍是 `8 × 4 × 8` 三层普通 Python 标量循环；这条有序程序
不能由 pass 猜回 shaped axis。它同时解释 standalone 与 decode/prefill 同速低下。

IQ1_S 的 standalone/decode 在两台机器为约 73%–75%，但 prefill 降到 SG 13.1%、K1
32.6%。`mul_mat.py:847-857` 仍是 `for row → for column → vec_dot`，没有 output cohort、blocked
accumulator 或 activation 跨输出复用。这是作者树的 outer traversal/lifetime 缺口，不是 compiler
可以从 M=128 自动发明的 GEMM tree。IQ1_M prefill 同时受这条和 scalar correction 影响。

### 6.2 F32 contraction carrier：4 个 case，3 个 `<50%`

F32 的作者树已经有 NC/MC 与 MR/NR axes。当前 Physical IR 却把 M 的 `MR=2` 放入 lane：
accumulator 为 `f32mf2`、有效 VL 只有 2；每个 K iteration 用一次跨 row 的 `vlse32`、两个 scalar
weight load 和两个 `vfmacc.vf`，K=4096 仍骑 issue time。decode tail 进一步退化到 VL=1。

source donor 让 K 骑 contiguous RVV lane，使用 m8 的 `vle32 + vfmacc.vv + vfredusum`。因此
差距落在 P1 carrier 与 P3 memory form：同一 `contract` 的 reduction/free axes 被映射成了错误的
物理组织，不需要改变 canonical Value 或 Level。结果是 SG decode/prefill 87.0%/20.6%，K1
28.1%/43.0%。

### 6.3 SG 单输出 packed contraction：6 个 `<50%` case

这六个 case 是三种不同 physical relation，不能合成一个“量化 vec-dot pass”：

- Q1_0 standalone/decode 为 37.6%/43.6%。每个 128 block 生成两组 16-lane bitmask value，先把
  bit 变成 `2*q-1`，再执行 widened MAC 和 reduction；donor 用 32-lane mask直接选择 `q8/-q8`
  并做 i16 reduction。多出来的是 P1 carrier 切分与 P3 leaf/memory form，不是 GEMV 外壳。
- Q3_K standalone/decode 为 37.3%/39.2%。生成 C 在每组中四次重建同一个
  `packed_plane_merge` scale expression，并把 product 切成 i16m2 后分别 reduction。这里同时有
  P2 computed-scale supply 与 P1/P4 product carrier/收敛位置问题；作者 half/plane/scale axes 已经存在。
- Q4_K standalone/decode 为 45.6%/48.0%。每个 256 block 分成 8 个 group；每组各自 load、两次
  widened MAC 并立即 reduction，min correction 另走 8 次 scalar loop。donor 把 scale/min 形成
  vector supply，保留独立 product 后再收敛。这里是 P3 scale/min window 与 P4 partial topology。

三对 standalone/decode 都同步，说明低速位于共享 vec-dot 底座，不在 `MUL_MAT` decode wrapper。
Q4_K persistent ABI 是另一份作者 artifact 选择；canonical donor 本身已经证明 canonical bytes
不要求 compiler 改 ABI 才能获得高性能。

### 6.4 IQ3_XXS indexed supply：SG 3 个 `<50%` + K1 3 个 FAIL

SG 的 standalone/decode/prefill 分别为 10.6%/10.8%/9.7%，几乎相同，说明 outer traversal
不是主项。每个 256 block 的四个 group 都重新构造 `vid/broadcast/gather` index 链，grid/sign
payload 使用 32-bit indexed gather；donor 将不变 index 放在循环外并使用更窄的 index，同时保持
完整 widened product。该簇是 P2 index rematerialization 与 P3 indexed memory form。K1 又叠加
了 4.2 节的 alignment correctness bug。

### 6.5 SG pure-decode output carrier：3 个 `<50%` case

Q1_0、TQ1_0、TQ2_0 的作者 logical axes 都已显式；TQ1_0 还已经用 `reshape + subview` 写出
`160 + 80 + 16` 三段输出。当前生成 C 仍过早切成很小的 output carrier：

| entry | 可见动态结构（每 record） |
|---|---|
| Q1_0 | 16 次 mask load/merge/slide，32 组 f32 convert/mul/store |
| TQ1_0 | 64 组 convert/mul/store、48 次 slide、64 次 LMUL truncate、10 次 scalar extract |
| TQ2_0 | 64 组 convert/mul/store、48 次 slide、64 次 LMUL truncate |

三者在 K1 分别达到 source 的 3.90×、4.55×、3.48×，在 SG 只有 32.2%、42.7%、21.4%。
这把问题限定为 VLEN128 下 P1 output carrier 与 P4 过早切片，而不是缺作者 axis。三种 storage
relation 不同；当前证据只证明共同症状，尚未证明一个 topology 规则能同时修复三者。

### 6.6 Q2_K prefill residency：2 个 `50–70%` case

Q2_K standalone/decode 已是 SG 97.3%/96.1%、K1 141%/141%，因此 59.4%/64.0% 的 prefill
差距不在 vec-dot leaf。作者 blocked tree 已明确写出 `wp=materialize(W[nc])`、
`xp=materialize(Xq[mc])` 和 MR×NR accumulator；生成 C 却只把 decoded low scales 放进 32-byte
local object，且该 object 在 output-row/cohort 与 K loop 内重新 materialize，W/X payload 仍在
inner sub loop直接加载。缺的是 P2 residency/lifetime：作者已声明的 panel birth 没成为跨多个
output consumer 的 local allocation。

### 6.7 70–90% 档中已有闭合证据的横向项

- Q8_K activation quantize 在 SG/K1 为 71.1%/79.2%。作者树与 donor 都是 256-element block、
  max/min 两次 reduction、16 个 bsum reduction；差异是 P1/P3。Weft 的 max/min 和 quant input
  load 在 SG 为 32/32、K1 为 16/16，donor 分别为 8/8、4/4；SG 另有 4 次 m2→m4 carrier
  repack。数学 reduction 数没有变化。
- K1 IQ4_NL 的 standalone/decode/prefill 为 71.4%/72.8%/71.7%，MXFP4 为
  76.7%/79.4%/80.5%。两者的 donor 每次共同处理两个 32-element blocks；Weft 每次只处理一个，
  每片立即 reduction。MXFP4 还把 codebook vector load 放在 block loop 内，而 donor 只加载一次。
  这是 P1/P4 issue grouping 与 P2 table lifetime，不是 `MUL_MAT` wrapper。
- IQ1_S 的 73%–75% standalone/decode 已在 6.1 定位为 issue-local qh-derived index/metadata 与
  residual indexed supply，即 P2/P3；它与缺 blocked prefill 是两个不同层次的问题。
- SG row IQ2_S、IQ2_XXS、IQ3_S 的 85.2%/77.9%/71.4% 都落在 entry/index/sign memory
  relation。IQ3_S 的具体差异是两个相邻 4-byte entry 被组织成一个 8-lane indexed lookup再用
  slide拆开，而 donor 是两个 unit loads；它与 IQ2 metadata relation相邻但不相同。

另外三项没有足够证据支持新增机制：SG Q4_1 standalone 的 terminal structure 已闭合为两次
`vwmaccsu`、一次 reduction 和一次 scalar extract；SG TQ2_0 已与 donor 的四 plane工作结构一致；
SG Q6_K 仍有两次 `register_to_lane` conversion与一次 split-factor=4 partial repack，但对应
`MUL_MAT` decode 已达到 source 的 112%。这些行保留为可见的剩余开销，不把它们包装成一个
未经第二输入验证的新 pass。

### 6.8 与 Triton / TileLang 的机制对照

这些归因没有从 pass 名称反推：

- Triton `AccelerateMatmul.cpp` 先以 target operation 约束 operand/result encoding，再由后续
  lowering 消费；F32 与 IQ2 暴露的是 Weft 在同一位置没有形成合法或高效的 reduction carrier，
  不是 terminal intrinsic 少换了一个名字。
- Triton `CoalesceUtils.cpp:17-94` 读取 contiguity/divisibility/alignment，`Coalesce.cpp:82-119`
  把选定 layout 写回真实 IR；IQ3_XXS 的 2-byte-aligned edge 被选成 `vle32`，正是这类 fact 没有
  贯穿到 memory-form legality。
- Triton `OptimizeDotOperands.cpp:181-299` 把 local allocation/load 提到共同 base tensor；Q2_K
  prefill 的作者 `materialize` 已经存在，问题是 physical residency 没有覆盖其多个 output consumers。
- TileLang `loop_vectorize.cc:216-266,428-476` 从所有 access 关系和 target width 得到 vector
  carrier；SG pure-decode 的大量 slide/truncate说明 Weft 对已显式 shaped output 仍选了过小 carrier。

这些对照只确定缺口属于 layout、memory、residency 或 carrier pass；它们不授权把 GPU thread
encoding、TIR Ramp 或某个 donor microkernel整体搬进 Weft。

## 7. 聚类后的边界

本轮没有出现“所有 `<50%` 都是同一原因”。最大的共同 failure 是 6 个 IQ2 staged prefill
layout closure；最大的低速作者侧集合是 IQ1 的 8 个 case；最大的单一 compiler relation 是
IQ3_XXS 的 6 个双机 case（3 个低速、3 个运行失败）。F32 的 4 个 case、SG pure-decode 的
3 个 case又分别属于独立 carrier 问题。

没有证据表明 spec 2.2 被证伪：

- IQ1 的 scalar correction 与 prefill outer traversal 确实改变 logical axes/Value lifetime，归作者；
- F32、Q1/Q3/Q4_K、IQ3_XXS、pure-decode carrier、Q8_K quantize 和 Q2_K residency 都可以在
  不改变 canonical Value/Level 的情况下通过另一种 physical carrier、memory form、placement 或
  topology实现，归编译器；
- 没有一个已确认 case 要求 compiler 暗中替换 std function、增加 logical axis 或改 persistent ABI
  才能达到 source。

因此当前全局账显示的是多个真实但彼此不同的作者/physical compiler缺口，而不是职责判据本身
失效。
