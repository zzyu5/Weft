# Production shape 全量性能与归因

## 结论

当前 206 条 production 目标已经全部在真实 shape 上经过同一条 Weft 主链生成、交叉编译并在 SG2044 与 K1 执行；206 条 Weft 结果均与相应 GGML reference 逐位一致，没有数值失败。性能结果并不接近整体闭合：只有 11 条达到 source 的 90% 以上，168 条低于 50%，全量中位比值为 21.24%。

慢项不是一个原因。全量结果把问题分成两类：

1. F16 与 24 种量化 `MUL_MAT` 的 std 树明确按 `row → column → scalar vec-dot` 逐输出执行，没有 output block、多个 accumulator 或跨输出 activation reuse。这是作者树的结构，不是 target 可以暗中补出的物理决定；这 100 条目标需要先由作者重写 std 树。
2. 已经具有合理局部数值树的 K 系列与 IQ/codebook 路径，physical assignment 仍把关键 packed extract 选成 `scalar-indexed`、lookup 选成 `scalar.lookup`，相关 Level 又是 `pipeline_depth=1, unroll=1`。这是编译器的 memory form、value mapping 与 Level scheduling 缺口，不需要改变作者树。

因此，本轮没有证伪 spec 2.2。相反，它把“作者树确实缺结构”和“树不需改变、physical pass 仍太弱”分开了。当前数据也不能宣称 2.2 已被完全验证，因为大量 `MUL_MAT` 仍使用已知不适合高性能 CPU 的 std 树。

完整逐条数据在 [`weft-kernel-performance.csv`](weft-kernel-performance.csv)。该文件现在只含本轮 206 条当前 manifest 结果；旧编译器时期的 `source_*` 历史行已经移除。固定 GGML baseline 文件没有改动。

## 一、覆盖范围

| 能力族 | production shape | 格式/phase | 每台目标 | 两台合计 |
| --- | --- | --- | ---: | ---: |
| `MUL_MAT` | decode `M=1,N=4096,K=4096`；prefill `M=128,N=4096,K=4096` | F32、F16、24 种量化格式，两个 phase | 52 | 104 |
| quantized vec-dot | `M=1,N=14336,K=4096` | 24 个 weight/activation typed pair | 24 | 48 |
| activation quantize | `M=128,K=14336` | Q8_0、Q8_1、Q8_K | 3 | 6 |
| row dequantize | `N=1024,K=4096` | 24 种 encoded row | 24 | 48 |
| 合计 |  |  | 103 | 206 |

两个 target 分别是：

- SG2044：RV64GCV，VLEN=128，固定 CPU 48；
- K1/X60：RV64GCV，VLEN=256，固定 CPU 3。

这 206 条是 103 个逻辑目标在两台机器上的执行结果。当前 manifest 没有单独的 Weft IME realization 行，因此这里不对 IME 性能作结论；K1 行只代表当前 RVV target lowering。

## 二、数值、编译与计时协议

### 2.1 数值边界

每个 Weft runtime 先执行一次未计时调用，再对完整结果作检查：

- `MUL_MAT`：activation workspace 与全部 `M×N` 输出逐位比较；
- vec-dot：全部 `N=14336` 个输出逐位比较；
- activation quantize：完整 encoded workspace 逐字节比较；
- row dequantize：全部 `N×K` 浮点输出逐位比较。

reference 是对应的 GGML quantize、generic vec-dot 或 row-dequant 实现。CSV 中的 `correctness=weft-vs-ggml-reference-bit-exact` 专指这一关系，不表示 Weft 与另一次独立 GGML graph 调用使用了同一份输入字节。

最终结果为：

```text
206 executed
206 Weft-vs-GGML-reference bit-exact
0 numerical FAIL
```

性能输入使用确定性数据。Weft 的 packed 格式使用可产生有限 reference 结果的随机 record 并复制到 production shape；GGML source runtime 对有 quantizer 的格式从确定性浮点行量化，对没有公开 quantizer 的 IQ 格式保留其既有 encoded 输入构造。两边 shape、格式和执行工作量一致，但 IQ/codebook 路径并非逐字节同输入，因此这些行的绝对比值应视为当前部署性能样本，而不是输入分布完全受控的微架构实验。

### 2.2 编译选项

此前“生成 kernel 开 contraction、runtime/reference 关 contraction”的不对称已删除。当前新编译的 generated C、runtime 和 reference wrapper 均显式使用：

```text
-O3 -Wall -Wextra -Werror -ffp-contract=fast -march=<target> -mabi=lp64d
```

C 使用 `-std=c11`，C++ 使用 `-std=c++17`；K1 额外使用 `-fno-integrated-as`。SG2044 使用 GCC/G++ 15.2，K1 使用 Clang/Clang++ 18。没有启用 `-ffast-math` 或 reassociation。生成 C 中用于阻止 system compiler 覆盖 Weft 已选向量形态的 `#pragma GCC unroll 1` 保留。

GGML source kernel 来自两台机器现有的 Release `-O3` library：SG2044 的 `build-gcc15-rv64gcv` 与 K1 的 `build-ime`。K1 的 CMake cache 没有额外写 `-ffp-contract`，使用 Clang 的默认 contraction 语义；本轮显式统一的是这一轮新编译的 Weft、runtime 与 source wrapper，不声称重建了两份 GGML library。

### 2.3 计时范围

每条数据使用：

```text
1 次未计时 correctness/warmup
3 次计时
每次计时前 64 MiB cache eviction
记录中位数
单线程 / 固定 CPU
```

各族的计时范围为：

- `MUL_MAT`：完整 projection，包含 F16 staging 或量化 activation preprocessing；
- vec-dot：activation 已量化，计完整 `N=14336` projection；
- quantize：完整 `M×K` activation；
- dequantize：完整 `N×K` tensor。

SG2044 的长批次中发生了三次整机 shutdown/reboot，连接失败的行没有进入 CSV，而是在机器恢复后重跑。后半段机器上出现了其他常驻 workload；Weft/source 仍按 case 成对运行、固定同一 CPU 并使用相同 eviction，但 SG2044 数据不是独占机器条件下的实验室极限值。K1 批次连续完成。

## 三、总体分布

### 3.1 全量与硬件

| 范围 | 全量 206 | SG2044 103 | K1 103 |
| --- | ---: | ---: | ---: |
| `Weft/source ≥ 90%` | 11 | 5 | 6 |
| `70% ≤ Weft/source < 90%` | 13 | 5 | 8 |
| `50% ≤ Weft/source < 70%` | 14 | 9 | 5 |
| `Weft/source < 50%` | 168 | 84 | 84 |

| 集合 | 最小值 | 中位数 | 最大值 |
| --- | ---: | ---: | ---: |
| 全量 | 0.90% | 21.24% | 257.48% |
| SG2044 | 0.90% | 18.87% | 121.06% |
| K1 | 3.59% | 22.87% | 257.48% |

### 3.2 能力族

| 能力族 | 条数 | ≥90% | 70–90% | 50–70% | <50% | 中位比值 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `MUL_MAT` | 104 | 4 | 2 | 6 | 92 | 14.94% |
| quantized vec-dot | 48 | 3 | 1 | 2 | 42 | 15.27% |
| activation quantize | 6 | 0 | 3 | 1 | 2 | 71.08% |
| row dequantize | 48 | 4 | 7 | 5 | 32 | 36.74% |

最坏项集中在 IQ2_XXS、K 系列以及 prefill `MUL_MAT`：

| 项目 | 目标 | Weft | source | 比值 |
| --- | --- | ---: | ---: | ---: |
| IQ2_XXS `MUL_MAT` prefill | SG2044 | 0.0379 GOP/s | 4.2067 GOP/s | 0.90% |
| IQ2_XXS vec-dot | SG2044 | 0.0731 GOP/s | 4.1298 GOP/s | 1.77% |
| Q4_K vec-dot | SG2044 | 0.3270 GOP/s | 9.2817 GOP/s | 3.52% |
| IQ2_XXS vec-dot | K1 | 0.0613 GOP/s | 1.6826 GOP/s | 3.64% |
| Q4_K vec-dot | K1 | 0.1010 GOP/s | 2.5819 GOP/s | 3.91% |

超过 source 的少数项目不能外推成统一高性能：K1 NVFP4 vec-dot 为 257.48%，但 source 对该格式明确使用 scalar implementation；SG2044 IQ3_XXS vec-dot 为 121.06%，同一 Weft 树在 K1 上只有 6.15%。它们说明单项 source 强弱或 target 映射差异，而不是主链已经普遍胜过手写 RVV。

## 四、归因聚类

### 4.1 聚类一：production `MUL_MAT` std 树没有 block output reuse

当前 F16 与 24 个量化 `MUL_MAT` 函数都由 std 明确写成：

```text
quantize/stage activation
for row in range(M):
    for column in range(N):
        scalar = vec_dot(W[column], Xq[row])
        commit(scalar, Y[row, column])
```

具体树见 `python/weft/std/mul_mat.py`：量化入口的 `row/column` 是普通有序 scalar `for`，F16 的 K 归约也使用 `L.blocks(K, extent=1)` 与 scalar accumulator。它们没有：

- 一次存活多个输出 accumulator；
- N/M 方向的 output Level；
- activation 在多个输出间的显式 reuse；
- prefill M 方向的 block 与 staging reuse。

这不是 emitter 或 physical pass 可以合法补出的结构，因为引入 output block 和多个 accumulator 会改变逻辑值集合及其 Level 归属。数据直接验证了这一点：

| 树 | SG2044 prefill/decode Weft throughput | K1 prefill/decode Weft throughput |
| --- | ---: | ---: |
| 24 个 quant `MUL_MAT` 中位数 | 1.001× | 1.000× |
| F16 | 1.000× | 0.981× |
| 已有 blocked F32 GEMM | 25.835× | 15.748× |

即量化与 F16 prefill 只是把 decode 结构顺序重复 128 次；只有 `python/weft/std/dense.py` 中显式写出 NC/KC/MC、MR/NR accumulator 与 packed A/B 的 F32 树产生了真正的 prefill 复用。F32 prefill 达到 source 的 87.77%（SG2044）与 68.31%（K1），而 F16 prefill只有 5.39% 与 18.00%。

这一聚类影响 F16 + 24 quant 格式 × decode/prefill × 两台 target，共 100 条 `MUL_MAT` 目标。它不是说这 100 条的全部差距都由树造成；每个 scalar vec-dot 自身还继承下面的 physical lowering 缺口。但在作者树改变以前，compiler 无权生成跨输出 microkernel。

结论：这是 std 树问题。本轮没有修改树。

### 4.2 聚类二：packed extract 与 codebook lookup 仍被标量化

SG2044 IQ2_XXS production assignment 是最清楚的代表：

- weight `q` 字段的 `op88/op95/op105/op112` 均为 `encoded-access.scalar-indexed`，对应 access value `v93/v100/v110/v117`；
- Q8_K activation `q` 的 `op165` 也是 `scalar-indexed`，对应 `v174`；
- 两个 codebook consumer `op152/op161` 均为 `scalar.lookup`；
- weight record 与 activation record 虽是共享 live value，但标量 extract/lookup 破坏了 lane 内 decode-to-MAC 链。

这不是 encoding 缺失：assignment 已经记录 `base_family`、field shape、bit offset 和 concrete selector。问题出在 memory-form pass 与 lookup realization 没有把这些 use-specific mapping 投影成 vector indexed/gather 或可复用 decode window。

该问题横跨：

- K 系列的 packed high-bit、scale/min 分支；
- IQ1/IQ2/IQ4 的 codebook、sign 与 delta；
- row dequantize、vec-dot 与 `MUL_MAT` 三个 consumer context。

可观察结果包括 SG2044 IQ2_XXS vec-dot 1.77%、Q2_K 2.89%、Q4_K 3.52%，以及 K1 IQ2_XXS 3.64%、Q4_K 3.91%。同一问题也解释了多数 IQ/K dequantize 为什么明显慢于简单 Q4/Q8 路径。

IQ3_XXS 在 SG2044 达到 121.06%、在 K1 只有 6.15%，说明 `scalar-indexed` 不是单独决定比值的充分条件：source 实现质量与 target codegen 也影响比值。但它仍是 assignment 中明确存在、可横向修复的编译器瓶颈。

结论：这是 compiler memory/value realization 问题，不要求改变 std 树。

### 4.3 聚类三：关键 Level 没有形成局部流水

同一个 IQ2_XXS contraction 的 K-block Level 在 SG2044 被映射为 128 physical lanes，在 K1 被映射为 256 physical lanes，说明 target VLEN 已经影响 lane assignment；但两边 schedule 都是：

```text
loop_structure = sequential-stream
pipeline_depth = 1
unroll = 1
derived_from = no schedulable local cluster in this Level
```

因此 load → packed extract → lookup/decode → widen → accumulate 没有形成跨 record/group 的 prologue、steady state、epilogue，也没有把下一个 group 的 load/decode 与当前 MAC 交错。这个缺口同时落在 K 系列与 codebook 系列，不是 q-format route。

结论：这是 Level scheduler 的横向问题；树已经显式给出可作用的 K-block Level，compiler 不需要改变层归属。

### 4.4 聚类四：Q8_K activation tree 本身选择了顺序 signed-extreme 扫描

Q8_0/Q8_1 activation quantize 已能达到 source 的 66.25–84.25%（SG2044）与 75.92–78.21%（K1）；Q8_K 只有 19.50% 与 12.19%。具体差异不是一个全局 LMUL：

- Q8_0/Q8_1 在 32-element Level 上直接形成 RVV reduce/narrow/store；
- Q8_K std 树在每个 256-element block 内显式写了 `L.subs(kb, extent=1)`，顺序执行 signed-absolute-max、first-wins state 更新；
- assignment 中该 Level 为 `ordered-sequential-level`、physical lanes 1、depth 1；后续 16-element `bsum` Level 才是 RVV lane；最外层 256-element Level仍报告 `no schedulable local cluster`。

这个顺序扫描是作者为了精确匹配 GGML signed-extreme 语义写入的逻辑树。Target 不能把它自行改成另一种 vector argmax/reduction，因为那会改变状态更新结构及可能的 tie 语义。如果作者希望使用一个具有同样 first-wins 可观察语义的显式 structured primitive，需要修改 std 树或语言语义；本轮不代替作者作该决定。

结论：这是另一个明确的 std 树限制，不是 LMUL 调小即可修复。

### 4.5 聚类五：K1 只放大了 lane 数，没有得到匹配的局部实现

K1 的 24 条 vec-dot 中 23 条低于 source 的 50%，SG2044 为 19 条。Assignment 确实随 VLEN 从 128 改到 256 lanes/对应 LMUL，但 packed extract、scalar lookup 和 depth-1 schedule 没有改变实现类别。

这说明 target facts 已进入 assignment，却还没有贯穿到 memory/decode/MAC 的整体局部实现。它不是“再加一个 K1 路径”的理由；正确修复位置仍是共享的 use-specific memory form、value handoff 与 Level schedule。NVFP4 的 2.57× 是 source scalar 基线的例外，不能掩盖这一横向缺口。

## 五、对 spec 2.2 的判断

本轮没有发现下面这种证伪：

> encoding 正确、作者树已经拥有高性能所需的逻辑值和 Level、physical pass 也完整，但仍必须由 compiler 改变逻辑值集合或层归属才能达到手写性能。

实际发现的是：

| 现象 | 归属 | 是否要求 compiler 改树 |
| --- | --- | --- |
| F16/quant `MUL_MAT` 没有 output block 与 multiple accumulators | std 树 | 否；作者应改树 |
| Q8_K 顺序 signed-extreme 扫描 | std 树/显式语义 | 否；作者决定是否换成等价 structured primitive |
| packed extract 为 `scalar-indexed` | compiler memory/value mapping | 否 |
| codebook 为 `scalar.lookup` | compiler local realization | 否 |
| K-block Level depth 1、unroll 1 | compiler scheduler | 否 |
| K1 只改变 lane count，局部实现不变 | compiler target-dependent realization | 否 |

所以，spec 2.2 当前没有被证伪，但不能凭这批结果宣布最终成立：100 条 `MUL_MAT` 目标仍被已知的 std 树结构限制挡在前面，尚未进入“树正确且 compiler 完整”的强检验条件。

## 六、当前真实边界

当前 Weft 已经证明的是：

- 206 条 production target 在真实 shape 上都能通过唯一主链生成并执行；
- encoding、数值树、activation workspace 与完整输出在对应 GGML reference 下逐位正确；
- blocked F32 prefill 在 SG2044 已达到 source 的 87.77%，说明 Level/block/accumulator 模型能承载一条有竞争力的 CPU 树；
- 简单 dequantize 与 Q8_0/Q8_1 quantize 在 K1 上多项达到 75–99%，少数超过 source。

当前没有证明的是：

- production quant/F16 `MUL_MAT` 的高性能树；
- K/IQ/codebook packed extraction 的通用 vector memory realization；
- load/decode/MAC 的真实跨 group software pipeline；
- K1 上与 VLEN256 匹配的普遍 vec-dot 组织；
- IME 在这 206 条 manifest 中的比较结果。

因此目前离手写 intrinsic 的差距主要是两部分：std 尚未写出的 CPU 算法结构，以及 compiler 尚未完成的 packed-memory/decode/pipeline 工程能力。现有数据没有暴露“设计原则必然做不到”的原理性反例；但在上述两类缺口关闭前，也不能把 168 条低于 50% 的结果描述为普通调参问题。
