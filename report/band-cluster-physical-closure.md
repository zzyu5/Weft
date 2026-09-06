# 70-90% 区间的物理闭合与定向重跑

日期：2026-09-06。本文件是一次性结果，不是设计规范。

## 范围与结果

本轮开始时对照表实际有 202 行，其中 70-90% 是 31 行，不是旧上下文中的 27 行。
先重跑这 31 行、90-92% 边界的 6 行和 SG TQ1_0 standalone/decode，共 39 行。
初测 33 行数值通过；IQ3_XXS 的 5 行失败，另有 SG Q1_0 prefill 编译拒绝。
旧表是历史结果，不能替代当前实现的数值验收。

最终重跑集合是初始 39 行、原已过线的 127 行和修复关联的 3 个额外入口，共 169 个唯一条目。
随后对 IQ3_XXS 的等价坐标改写补跑双机 6 条，并对 K1 F32 prefill 复跑一次。
[本轮逐项记录](band-cluster-physical-closure-runs.csv) 每个条目保留最终一次结果；
[对照表](kernel-performance-comparison.csv) 只更新这 169 个对应条目，其余 33 个值不变。
`weft-kernel-performance.csv`、旧报告和旧 CSV 均未修改。

- 169/169 通过现有运行合同的数值验证；没有调整容差、输入策略或计时边界。
- 原已过线的 127/127 仍达到 source 吞吐。
- 原 31 个区间条目最终有 6 个达到 90% 以上，24 个仍在 70-90%，1 个低于 70%。
- 该低于 70% 的条目是 SG IQ1_S prefill：本轮初测已经是 64.60%，最终 64.51%，不是本轮修复造成的新下降。
- 23 个最终结构相关配置通过 final IR 重放及资源闭合前的 layout checkpoint 重放。

CSV 的 `previous_weft/change_percent` 对比本轮开始时的旧表，不等于同源码 A/B。
下文对机制收益采用本轮可执行、数值正确的前后结果。

## 七步工作账与分类

按 `doc/compiler/optimization-principles.md` 第九节，先看 hot Level 的物理轴和资源，
再按 source origin、坐标和 loop trip count 数 supply、memory edge、partial/reduction，
最后比较调度及 donor。参考了 Triton 的
`lib/Conversion/TritonGPUToLLVM/ConvertLayoutOpToLLVM.cpp`、
`lib/Dialect/TritonGPU/Transforms/RemoveLayoutConversions.cpp`，以及 TileLang 的
`src/transform/layout_inference/layout_cost_model.cc`；二者都在用户指定的仓库外目录，未进入构建。

| 集合 | 落到的实体与边界 | 本轮处置 |
| --- | --- | --- |
| IQ3_XXS 双机 standalone/decode/prefill | sign index 的 `[half,entry,code]` 映射；旧拼接把 lane/replica 顺序混在一起。正确投影后还存在两级索引广播 | 修编译器合法性及 rematerialization；再等价合并作者的 entry/code 坐标；6 条实际运行 |
| IQ1_S 双机 standalone/decode/prefill | 同一 `qh` field 的完整窗口与短窗口重复读取；prefill 是另一个动态 `rvv_indexed_entry_load` 关系 | 静态同 block 窗口复用；standalone/decode 命中，prefill 不宣称命中 |
| Q1_0 两机四类入口 | `FuseRISCVBitplanes` 只识别 conversion 包裹的 extract；直接带 RVV layout 的 u1 extract 漏选 | 两种表示共用 bitmask decode；同时拒绝跨 layer 的 storage-window 候选；8 条实际运行 |
| SG dequantize 六种输入 | unit load 的 payload 与 f32 carrier 宽度；IQ2_S/XXS 已有较宽绑定，IQ3_XXS 作者 payload 只有 4 | 扫五种输入的四个 LMUL；只调整 Q3_K、Q5_K、IQ3_S |
| SG Q2_K/Q4_K/Q4_1 成对 contraction | Q2/Q4 的独立 scaled partial 与归约链；Q4_1 已是一份 raw nibble 窗口、两份 activation、一次最终归约 | 有限绑定扫描未得到替代 winner；不能仅凭归约数认定缺少作者 variant |
| SG IQ1_M 成对 contraction | main/correction 两个物理循环仍分别读取同一 activation/qh 区间；grid 供给已按完整 product 形成 | 保留当前程序；跨循环共享的 effect/resource 合同未闭合，不把剩余差距全部归给数值树 |
| K1 IQ4_NL/MXFP4 成对 contraction | 实际汇编已有循环外 codebook load；两份 i16 product 用 `vwadd` 合到 i32 后归约 | 不增加虚假的 codebook-hoist 修复；i16 combine 还需要值域合同，见下文 |
| Q8_K quantize 双机、K1 Q6_K/IQ2_XXS prefill | quantize 的 extrema/量化两段；prefill 的现有输出 tile、partial 与 lookup carrier | 四个 LMUL 的有限扫描无明显收益，保留原绑定 |
| SG IQ2_XS standalone、IQ3_S prefill | 旧区间值与当前运行不一致 | IQ3_S prefill 已到 91% 左右；IQ2_XS 初测的越线值未在最终回归复现，最终记 81.44%，不报成已闭合收益 |

## 实际改动

### 索引布局与读取窗口

`rvvPartToLanePieces` 现在检查轴顺序，而不只检查乘积和 LMUL。
已有较早 lane axis 时，不能把更晚 replica/time axis 当作可直接拼接的后续片段。
最终 layout canonicalization 前先清理死 index/conversion 链，避免死 use 阻止单 use 重物化。
`canReprojectIndex` 对纯 index 链执行最多 32 节点的可投影检查，不复制 memory producer。
terminal 拼接也区分完整 carrier 与 active lanes，不能用 underfilled carrier 的容量代替逻辑长度。

窗口共享在 `MaterializeRISCVReplicaStorageLoads` 内执行：至多检查 32 个已有候选，要求同一
field SSA、record 坐标、常量连续范围和无写入/未知 effect。相同范围共享值，子范围使用已有
`rvv_issue_slice`，不新增更宽读取。SG IQ1_S 的 qh load 每 block 从 9 降到 1，K1 从 5 降到 1。
对应最终 vector register peak 是 SG 24、K1 18。两机、两种 ABI 是实际输入证据，不宣称跨格式泛化。
prefill 的动态 issue/cross-block 读取不在这条规则内。

### IQ3_XXS 等价坐标改写

`entry_code = 2 * entry + code` 将 `[4,2]` 的坐标直接表达为 `[8]`，没有新建 numerical variant。
原索引由 `entry_code // 2` 和 `entry_code % 2` 恢复，weight、sign、activation 的地址逐项相同；
每个 half 仍归约同样的 32 个整数 product，scale、block accumulation 与最终浮点乘法位置不变。
修改在共享的 `examples/kernels/vec_dot/iq3_xxs_q8_k.py`，matrix 入口继续调用它。

SG 每 64-element group 的 C 级 axis broadcast 从 4 个降到 3 个，仍是一个完整 64-lane
product、两个 32-lane partial reduction。索引/数据的轴分别为 `[half,entry_code]` 与
`[half,entry_code,payload]`，lane 因子分别为 `[2,8]` 与 `[2,8,4]`。
SG vector register peak 从修正索引后的 19 增到 21，仍在合法资源内；最终 K1 peak 为 11。
因此这里的收益不能表述为寄存器峰值下降。实际汇编仍有部分常量向量 spill/reload。

| 入口 | 修正索引后的正确版本 GOP/s | 连续轴版本 GOP/s | 最终/source |
| --- | ---: | ---: | ---: |
| SG standalone | 1.729791 | 2.207702 | 82.58% |
| SG decode | 1.818549 | 2.892633 | 110.49% |
| SG prefill | 1.749450 | 2.755449 | 94.37% |
| K1 standalone | 1.298360 | 1.332032 | 92.02% |
| K1 decode | 1.304998 | 1.341974 | 93.44% |
| K1 prefill | 1.305592 | 1.343608 | 91.25% |

这 6 条的 absolute/relative error 均为 0。只修索引时 SG 曾落到 60-69%，该负结果没有被旧表数字覆盖。

### 其它保留的收益

| 入口 | 本轮初测 | 最终 | 最终/source |
| --- | ---: | ---: | ---: |
| SG IQ1_S standalone，GOP/s | 3.557810 | 3.811723 | 78.41% |
| SG IQ1_S decode，GOP/s | 3.603296 | 3.822073 | 78.97% |
| SG Q3_K dequantize，MEl/s | 329.143777 | 345.896219 | 93.15% |
| SG Q5_K dequantize，MEl/s | 347.495309 | 354.617883 | 93.18% |
| SG IQ3_S dequantize，MEl/s | 254.115807 | 262.703723 | 73.34% |

dequantize 的绑定分别是 `lmul_eighths=32,16,16`，位于 kernel 家族旁的 `tuning.json`，
标为 `measured-lmul-subset`，不宣称整个搜索域的最优值。IQ2_S 保留 64，IQ2_XXS 保留 16：
其部分较宽候选生成相同 C，不能为了计时微差换绑定。
Q1_0 prefill 恢复可执行：SG 9.414183 GOP/s，K1 3.662610 GOP/s；不是新增性能提升。

## TQ1_0：不采用新分组

用户引用的 8 个 `vsll/vsra/vand` 不是本轮开始时 SG lane16 artifact 的实际指令账。
当前路径的主要 radix 工作是 wrapped-u8 multiply、widen-by-3、右移、窄化和 product。
VLEN128 donor 在 `quants.c:5975` 起使用 i16 decoded 值及 i16 mul/macc，不能把它描述成同一条 i8 路径。

尝试把前 160 项按 32 lane、中间 80 项和尾部 16 项按 16 lane 累积，最后两次 i32 reduction。
lane 的绝对界限为 640 和 768，总整数绝对界限 32768，均不越界；浮点合并位置未改变。

| 版本 | standalone GOP/s | decode GOP/s | 结论 |
| --- | ---: | ---: | --- |
| 原 lane16 初测 | 5.298726 | 5.358304 | 数值通过 |
| 32+16，m2 | 4.303212 | 4.249680 | 32-lane 中间值被拆分，新增 slide/widening |
| 32+16，m4 | 5.579834 | 4.489552 | 保留宽 carrier，但 decode 出现额外 m4 临时值 spill/reload |
| 最终保留 lane16 | 5.300012 | 5.336275 | source 的 65.24% / 66.57% |

m4 使 radix 分组从每 block 16 组降为 11 组；这证明分组成本并非全部固有，
却没有得到 standalone/decode 同时获益的结果。原型已撤回，未增加第二份算法 source，未改容差。
该结果不证明所有 radix 表示均不可优化，只说明本轮候选不足以替换现有入口。

## 负结果与未闭合实体

- SG Q4_K 的 m1/unroll=8 仍优于小展开候选；m2/m4 与 unroll=2/4/8 的六个宽候选只有
  0.97-1.57 GOP/s，均数值通过但明显退化。VLEN128 donor 本身也使用 16-lane loads，不能把宽 load 当作既定答案。
- SG Q2_K 的 unroll=2/4 与 scalar-load-prime=1 不相容，被明确拒绝。关闭 prime 后三档
  unroll=2/4/8 分别为 4.900909/5.447480/5.845208 GOP/s，仍不替换现有 prime=1、unroll=8。
- Q8_K quantize 的 m4/m8 没有收益；K1 Q6_K/IQ2_XXS prefill 的 LMUL 扫描与现有值接近。
  本轮没有把参数扫描无收益当成“必须新增作者树”的证明。
- SG IQ1_M 每 block 的 main/correction 各有四个 64-element issue：共八个 activation load、
  八个 product、32 个 partial reduction。donor 的两个 i32 accumulator 最终只归约两次，
  但同时把两路计算放在共同的 activation 供应下。当前两个物理循环仍有重复读取；
  未闭合跨循环共享与联合资源前，不把全部差距归给 scale/reduction 数值结构。
- K1 IQ4_NL/MXFP4 的实际汇编已把小 codebook 读取提到 K 循环外，不能用 C 的摆放位置冒充动态 load 次数。
  当前 i8 codebook 参数允许 -128；两份 `(-128)*(-128)` 相加为 32768，不能直接用 i16 combine。
  donor 固定表的值域较窄，使用 i16 `vwmacc`，还按两个 block 展开；这些差异不是无条件可套用的目标优化。
- SG Q4_1 的 raw-window/load/reduction 工作账已与 donor 同形；剩余吞吐差距没有完成因果定位。
  SG IQ1_S prefill 的动态 qh supply 也未被本轮静态窗口规则解决。
- K1 F32 prefill 首测 2.944075，生成 C 未变，同配置复跑 3.831085，旧表 3.838260 GOP/s。
  CSV 使用复跑值；保留首测负结果，不把未复现的下降归因于本轮编译器改动。

## 验收与手动复现

正常 runner 重新将 Python DSL emit 为 canonical IR，经唯一 RISC-V 主链生成 C，
在 SG/core48/VLEN128 与 K1/core3/VLEN256 编译运行，按既有 GGML 合同对照数值。
每次至少十轮，沿用原 cold/eviction 与输入策略。可直接执行：

```bash
WEFT_KEEP_ARTIFACTS=1 examples/run/weft-quantized-vec-dot.sh sg2044 iq1_s 10
WEFT_KEEP_ARTIFACTS=1 examples/run/weft-quantized-vec-dot.sh sg2044 iq3_xxs 10
WEFT_KEEP_ARTIFACTS=1 examples/run/weft-mul-mat.sh sg2044 iq3_xxs prefill 10
WEFT_KEEP_ARTIFACTS=1 examples/run/weft-mul-mat.sh k1 iq3_xxs decode 10
WEFT_KEEP_ARTIFACTS=1 examples/run/weft-mul-mat.sh sg2044 q1_0 prefill 10
WEFT_KEEP_ARTIFACTS=1 examples/run/weft-row-dequantize.sh sg2044 q3_k 10
```

23 个最终配置：IQ1_S/IQ3_XXS 双机三种入口共 12 个，Q1_0 双机四类入口 8 个，
SG 三个调宽的 dequantize 入口。final IR 以
`canonicalize → cse → weft-riscv-canonicalize-layouts → weft-riscv-eliminate-dead-layouts → weft-riscv-verify-final`
执行两遍，启用 `verify-each/verify-roundtrip`，二次文本一致。
同时从 `riscv-layout-input` 重放 layout canonicalization/CSE/dead-layout，再
`--resume-layout-input --emit=intrinsic-c`，不借删除 final resource marker 绕过边界。
重放后生成 C；真机数值验证对应正常唯一主链的运行 artifact，不宣称逐一执行了每份重放 C。

原始证据位于仓库外：

- `/tmp/weft-band-clusters.p7HTN0/`：初始 39 行、工作账、编译对照与 `final-mechanical/results.json`。
- `/tmp/weft-band-final-runs.BxYFeq/verified-final.json`：合并最终补跑后的 169 个条目。
- `/tmp/weft-iq3-flat-runs.t06mR8/`：连续轴版本的双机 6 条与 F32 复跑。
- `/tmp/weft-band-dequant-tune.LTsIlY/`、`/tmp/weft-band-local-tune.cUYiEc/`：有限枚举、拒绝与实测记录。
- `/tmp/weft-tq-wide-runs.o1txGr/`、`/tmp/weft-tq-m4-runs.LksbvI/`：撤回的 TQ 候选；
  `/tmp/weft-tq-radix-wide.UDrgLR/` 保存相应 C/汇编。

本轮提交实现的是三个通用编译器修复、一处等价作者坐标改写和三个历史绑定更新。
没有 whole-kernel lowering、格式/内核/target 名编译器分支、容差扩大或静默 fallback。
未把仍慢的 24 个区间条目及动态 supply/schedule 的开放问题写成已闭合。
