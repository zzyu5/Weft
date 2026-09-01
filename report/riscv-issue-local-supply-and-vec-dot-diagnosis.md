# RISC-V issue-local 供应闭合与 vec-dot 诊断

## 1. 范围与结论

本轮只处理四组已经暴露的事实：IQ1_S / IQ2_XXS 的 issue-local 供应、Q6_K 的
standalone/decode 配对异常，以及 Q2_K 的 prefill/vec-dot 分裂。没有增加格式、没有修改
canonical 数值树，也没有把 kernel/格式名带进 pass 或 emitter。

结果如下：

- 24 格式 × 2 target 的 terminal intrinsic C 静态生成保持 `48/48`；
- IQ1_S 的 standalone/decode 在 SG2044 分别从 `2.295/2.310` 提到
  `3.178/3.190 GOP/s`，超过 source；K1 分别从 `1.272/1.275` 提到
  `1.988/2.016 GOP/s`，仍是 source 的 `73.2%/75.3%`；
- IQ2_XXS 在 SG2044 保持原形态和约 `70%` source；K1 的 standalone/decode
  提到 `1.636/1.647 GOP/s`，达到 source 的 `97.2%/98.8%`；
- Q6_K 当前 standalone/decode 在 SG2044 相差 `6.3%`、K1 相差 `0.2%`，原来的
  `16.2%` 异常不能复现，属于旧 CSV 快照污染；
- Q2_K 的 standalone/decode 仍只有 source 的 `55.5%–62.9%`。prefill 的过线机制来自
  blocked output cohort 和 local materialization，M=1 作者程序没有这些实体；这不是一个
  “已有 compiler pass 在 M=1 没触发”的同树退化。

## 2. 第九节七步诊断

### 2.1 Final IR mapping 与资源

IQ1_S 和 IQ2_XXS 的 canonical tree 已显式保留 `group × entry × payload`，共同 reduction
carrier 已在前一轮冻结。本轮没有改变 P1 carrier、partial combine topology 或 Level 归属。

新增的 physical relation 是 field-to-register supply：

```text
typed Field + affine logical index + consumer layout
    → rvv.replica_storage_load
      { unit/strided projection, source windows, part map, exact leaf }
```

`RVVReplicaStorageLoadOp` 的 verifier 同时检查 field/storage geometry、logical span、lane extent、
projection stride、record coordinates、part map、access form、SEW/LMUL/`vl` 与 exact leaf。
terminal emitter 只把 `unit` 写成 `vle`、把 `strided` 写成 `vlse`；缺失或不一致的 plan/leaf
直接失败。

partial materializer 原先会为 issue slice 重新构造一个 natural plan。本轮把它改成只投影
已有 unit plan 和 leaf，并按投影后的 result type重算资源合同；memory form的唯一选择点仍是
replica-storage-load materialization。

### 2.2 动态工作账

IQ1_S 在每个 K block 的主要变化是：

| target | 旧 grid/index gather | 当前 grid gather | 旧 activation indexed supply | 当前 activation supply | bsum supply |
|---|---:|---:|---:|---:|---:|
| SG2044 | 16 | 4 | 16 | 4 次连续 `vle8` | 16 个标量/索引 load → 2 次 `vlse16` |
| K1 | 8 | 2 | 8 | 2 次连续 `vle8` | 16 个标量/索引 load → 2 次 `vlse16` |

这组计数说明收益不是给现有工作换名字：SG2044 每 block 删除了 12 次 grid gather、12 次
activation indexed supply，并把 16 个 bsum 元素的离散供应变成两个 8-lane strided load。

IQ2_XXS 的 strict SSA/read identity 计数没有可合并的相同 logical offset；它不是 CSE 问题。
K1 上 typed affine relation产生 2 次 `vlse16`，替换 issue 内的离散 scale supply；SG2044 的
两-lane stride-4 变体实测为负，因此固定 target priority 保留原形态。

### 2.3 供应 identity 与 consumer

IQ1_S 当前仍不是完整 P2 闭合：qh raw field虽已在外层存在，但 qh-derived index/metadata
仍随 issue 重构；current C 中 qh 的标量供应没有减少。能证明复用或规则访存的 q/activation、
grid index 和 bsum edge 才被改写，不能把 storage proposal 扩成整个 contraction carrier。

IQ2_XXS 不同 issue 的 scale offset是不同值，不能忽略 offset做 read-CSE。可用事实是
“这些不同值形成固定 stride 的同一 lane window”，因此归 P3 memory form，而不是 P2
same-value reuse。

### 2.4 Memory form 与地址

`ShapedAffineIndex` 现在保存动态 scalar base 的整数 coefficient。pass据此证明：

- lane axis 的 affine coefficient为 1 时可形成 unit window；
- 单 lane axis、正 fixed stride、完整 window/span/alignment 闭合时可形成 strided window；
- grouped/layered mapping 不能伪装成普通 strided load；
- dynamic base 的 known multiple必须满足整个 physical window 的 alignment。

strided选择使用 target facts 和固定优先级，不由 emitter判断。最终 48-entry scan 中仅有：

```text
SG2044  IQ1_S    2 × vlse
SG2044  IQ2_XXS  0 × vlse
K1      IQ1_S    2 × vlse
K1      IQ2_XXS  2 × vlse
```

### 2.5 Partial / MAC / reduction

本轮没有改 partial carrier、MAC 或 reduction topology。IQ1_S 的 current C 仍按 SG 4 issue、
K1 2 issue进行 product/reduction；IQ2_XXS 继续消费前一轮冻结的 common reduction carrier。
因此吞吐变化可以归因于 P3 supply 与已绑定的 issue unroll，而不是 P4 reduction 重排。

### 2.6 Pipeline

所有受测 candidate 都是 `pipeline=1`。本轮没有生成 prologue/steady/epilogue，也没有把
load 改写收益归给 P5。

### 2.7 Donor 与参考编译器对照

GGML donor 的共性不是某个格式名，而是按实际 storage relation成段供应 q/activation/bsum，
避免按 scalar consumer重复发起 indexed load。本轮只在 typed affine geometry能证明同一关系时
生成 unit/strided load。

对应的参考机制是：

- Triton 的 `multiple_of` / `max_contiguous` 为 vectorized memory access提供 alignment 与
  contiguous-run facts：`ref/triton/python/triton/language/core.py:3190-3231`；
- Triton Gluon 的 `BlockedLayout` 承担 coalesced global-memory layout，冲突通过真实
  `convert_layout` 表示：`ref/triton/python/tutorials/gluon/02-layouts.py:129-138` 与
  `09-tma-gather-scatter.py:248-277`；
- TileLang 的 `VectorizePlanner::Plan` 同时读取 buffer stride、loop index与 memory/local/
  broadcast constraints：`ref/tilelang/src/transform/loop_vectorize.cc:51-79,210-266,307-379`。

Weft复用的是“typed geometry决定memory form、冲突留在IR、terminal只拼写”的机制；没有
复制 thread/warp ownership或shared-memory transpose，因为 Weft 的 logical value由
time/lane/register-replica/fragment/local-storage分解，不先属于一个SIMT owner。

## 3. 实现边界

pass主链在 partial materialization之后执行：

```text
CanonicalizeRISCVLayouts
→ PlanRISCVMemory
→ MaterializeRISCVReplicaStorageLoads
→ HoistRISCVLoopInvariants
→ SelectRISCVOperations
→ FinalizeRISCVLeaves
```

这里没有第二次运行完整 `ShareRISCVLayeredWindows`。完整 Share 的负实验会重新触碰已经闭合的
layered/projected geometry，并使 14 个静态入口失败；窄 pass只处理刚产生的 natural/strided
replica edge。

普通 `cmake --build build` 已验证会重建 target library并重新链接
`weft-compile`、`weft-opt`，本轮数据不是旧 executable 产物。

## 4. 双机结果

所有条目均为同一 Clang 18、`-O3 -ffp-contract=fast` 口径，10 repetitions，数值为
`within-tolerance`。

| kernel | target | 旧 Weft | 当前 Weft | source | 当前/source |
|---|---|---:|---:|---:|---:|
| IQ1_S standalone | SG2044 | 2.295 | 3.178 | 2.804 | 113.3% |
| IQ1_S decode | SG2044 | 2.310 | 3.190 | 2.788 | 114.4% |
| IQ1_S standalone | K1 | 1.272 | 1.988 | 2.716 | 73.2% |
| IQ1_S decode | K1 | 1.275 | 2.016 | 2.678 | 75.3% |
| IQ2_XXS standalone | SG2044 | 2.835 | 2.847 | 4.052 | 70.3% |
| IQ2_XXS decode | SG2044 | 2.831 | 2.833 | 4.035 | 70.2% |
| IQ2_XXS standalone | K1 | 1.456 | 1.636 | 1.684 | 97.2% |
| IQ2_XXS decode | K1 | 1.468 | 1.647 | 1.666 | 98.8% |
| Q6_K standalone | SG2044 | 5.527 | 5.193 | 4.895 | 106.1% |
| Q6_K decode | SG2044 | 4.630 | 4.866 | 4.851 | 100.3% |
| Q6_K standalone | K1 | 3.135 | 3.132 | 2.384 | 131.4% |
| Q6_K decode | K1 | 3.115 | 3.126 | 2.333 | 134.0% |
| Q2_K standalone | SG2044 | 6.578 | 5.578 | 9.308 | 59.9% |
| Q2_K decode | SG2044 | 6.611 | 5.547 | 8.815 | 62.9% |
| Q2_K standalone | K1 | 1.536 | 1.357 | 2.446 | 55.5% |
| Q2_K decode | K1 | 1.531 | 1.337 | 2.404 | 55.6% |

IQ2_S 作为 broad-anchor 回归也重新实测：SG standalone/decode 为
`2.277/2.285 GOP/s`，K1 为 `1.424/1.436 GOP/s`，与 clean revision一致，没有恢复早先的
storage-owner退化。

## 5. Q6_K 配对异常

当前同一 executable重跑后：

| target | standalone | decode | decode/standalone - 1 |
|---|---:|---:|---:|
| SG2044 | 5.193 | 4.866 | -6.3% |
| K1 | 3.132 | 3.126 | -0.2% |

Q6_K 在两台机器上都回到 ±10% 范围。final IR 的 vec-dot机制没有出现 decode-only op，
因此 Q6_K 不构成 “GEMV wrapper 有独立瓶颈” 的反例；本轮没有据此声称其余 23 对全部满足
同一阈值。

## 6. Q2_K 诊断

当前 prefill 与 standalone/decode 的 RISC-V IR差异是作者程序可见的：

- prefill 是 `outer_contract`，topology带 `outputReplicas=2`；
- prefill有 `rvv_local_materialize`、`local_alloc/bind/load` 和
  `rvv_assemble_replicas`；
- standalone/decode是单输出 contraction，`outputReplicas=1`，local storage为 0；
- 三者当前 partial topology都为 `sequential_per_stream`。

因此 prefill 的 `104.7%/126.8%` 与 standalone/decode 的 `55.5%–62.9%` 不是“同一物理
program只有 M=1 时 pass没触发”。blocked output cohort与materialize的存在由作者 tree决定，
compiler不能把它们加进单输出 vec-dot。Q2_K 的未闭合点是现有单输出数值树在 donor 对应的
storage/partial leaf上仍多做工作；本轮没有用 prefill机制或第二棵树掩盖这个事实。

## 7. 负实验与未闭合事实

- 忽略 logical offset做 read-CSE：静态可删除 op数为 0；不同 offset 读不同数据，未实现。
- 在 SG2044 对 IQ2_XXS 的两-lane stride-4 supply强制 `vlse16`：standalone 从
  `2.835` 降到约 `2.75 GOP/s`，未保留；K1 同一 physical relation为正，保留。
- 第二次运行完整 Share：14 个 entry静态失败，未保留；改为窄 replica-load pass后恢复
  `48/48`。
- 放宽 storage anchor：曾把 IQ1_S 提到约 93.5% source，但使 Q6_K、IQ2_S 错误继承
  contraction carrier，未恢复。
- IQ1_S 的 qh/index issue-local reconstruction仍存在；所以 K1 的 73–75% 不能归结为已完成
  的通用 P2 owner。
- IQ2_XXS 的 SG2044 70%仍是 typed local-pack/byte geometry缺口；同值CSE和两-lane strided
  load都已被数据排除。
- Q2_K 单输出 contraction仍是独立未闭合项，本轮只完成了 final IR层面的确定诊断。

## 8. 可证伪预测账

原预测的措辞冻结，结果只追加：

| 格式/入口 | 作者前置条件 | 原主要预测 | 对答案 |
|---|---|---|---|
| IQ1_S | grid/sign entry与lane成为shaped axes | 先P2，再P4；P3次之 | **MISS**：首个跨target收益来自P3 natural/strided supply与参数unroll；qh/index P2仍未闭合，P4不是本轮主因 |
| IQ1_M | 同上 | 先P2，再P4；P3次之 | pending |
| IQ2_S / IQ2_XS / IQ2_XXS | entry、payload、group与output cohort显式存在 | P2供应与P4 partial topology；indexed-entry只解决P3一部分 | **MISS**：首要缺口实际是P1 issue/product carrier与P3 indexed-entry；IQ2_XXS K1的strided收益仍只计P3证据，不改写原首因 |
| IQ3_S / IQ3_XXS | grid/sign与high-bit plane为shaped values | P3，然后P2/P4 | pending |
| TQ1_0 | shaped radix-3 tree | P4为主、P3为次，prefill再看P5 | pending |
| Q4_K canonical | 先确定canonical ABI的blocked/local-materialize tree | 作者边界优先；tree成立后P2/P3/P5 | pending |
| Q5_K dequant | shaped sub-axis已足够 | P3 joined geometry与P2 raw-window supply | pending |

累计为 `0 HIT / 2 MISS / 5 pending`。两个已决项的主要分类都错，已经达到“已决项过半错误”
的证伪条件。因此 P1–P5 目前只能作为生成工作账的检查维度，不能再称为已经验证的性能预测
模型。第九节“先数再改”的诊断流程不依赖这些预测命中，仍由本轮的动态计数和负实验支持。

## 9. Repro

```bash
cmake --build build
./examples/run/weft-quantized-vec-dot.sh sg2044 iq1_s 10
./examples/run/weft-mul-mat.sh sg2044 iq1_s decode 10
./examples/run/weft-quantized-vec-dot.sh k1 iq2_xxs 10
./examples/run/weft-mul-mat.sh k1 iq2_xxs decode 10
./examples/run/weft-quantized-vec-dot.sh sg2044 q6_k 10
./examples/run/weft-mul-mat.sh sg2044 q6_k decode 10
./examples/run/weft-quantized-vec-dot.sh k1 q2_k 10
./examples/run/weft-mul-mat.sh k1 q2_k decode 10
```

runner使用当前 `build/tools/weft-compile/weft-compile`，生成 intrinsic C，在对应目标机用同一
Clang/flags编译，并与 GGML reference在同一进程内做数值对照后计时。
