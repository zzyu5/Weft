# 成对 natural 读取闭合与剩余缺口

本轮从上一轮 30 条待定继续推进；执行依据仍是 [optimization-principles 第九节七步](../doc/compiler/optimization-principles.md#9-可复用诊断动作)。报告是一次性结果快照，不替代设计规范。

## 结果

- 初始 30 条全部先重跑，数值通过。IQ2_XS SG standalone 在改动前重测已过线，最终默认入口复跑为 **1.006196**；这不是本轮代码优化新增的达线收益。
- 保留一个通用编译器能力：同一 natural field 的两路完整交错读取选择闭合的 `rvv_segment_pair_load`，不改变作者树或后续 widening/add/reduction。它在三个格式、双机 standalone/decode 的 **12 个真实输入**上生效；另 6 个 prefill 回归未命中这个一维规则。
- SG Q4_K standalone/decode 从本轮基线 **0.786178 / 0.798548** 到最终复跑 **0.882407 / 0.944265**，仍未达 0.95。IQ1_S 的四个 standalone/decode 也有改善，但仍未达线。
- 当前对照表有 **29 条待定**、173 条已有数字 >= 0.95；本轮没有把代码优化后仍低于线的条目算作完成达线。逐条状态见 [本轮待定 CSV](ratio95-segment-pair-pending.csv)，最新数字见 [对照表](kernel-performance-comparison.csv)。

本轮只实际运行 42 个不同的表内输入，共 52 条默认配置运行记录，另有 8 个显式 unroll 调优候选；没有运行 202 条全量。下文的 184 是生成 C 的影响检查范围，**不是硬件重跑项数**。

## 工作账与实现

最初将 IQ1 的问题笼统聚为“兄弟供应重复”并不足以动手。进一步定位到一个与 Q4_K、Q5_K 共用的具体关系：

```text
same encoded read / natural field
    even[i] = field[base + 2*i]
    odd[i]  = field[base + 2*i + 1]
        ↓ 原先两条独立 strided edge
        ↓ 现在一个 vlseg2，两个原 layout 的结果
    原有 widen / add / scale / reduction
```

这里偶、奇位置不是相同地址，不能做 load CSE；两路的并集恰好是连续窗口，因此属于 memory form 选择。Q4_K 的 `bsum` 源为 16 个 i16，SG 原先每路各两个 VL4 part，总共 4 条 `vlse16`；现在是 2 条 `vlseg2e16`。IQ1_S 双机以及 Q4_K/Q5_K K1 为 2 条变 1 条。读取字节集合、实际元素类型和整数结合边界没有变化。

具体落点：

- `ShareRISCVLayeredWindows.cpp::segmentPairBase/materializeSegmentPairs`，由最终的 `MaterializeRISCVReplicaStorageLoads` 调用。最多向后检查 32 个 load，核对同一 owner/name/typed field geometry、静态 base 差 1、step=2、完整 part map，以及中间 read-only/pure effect。
- `RISCVOps.td::RVVSegmentPairLoadOp` 与 `RISCVDialect.cpp::RVVSegmentPairLoadOp::verify`：双结果、完整一维相同 layout、静态范围、element-byte alignment、segmentFields=2、NF×LMUL<=8、exact leaf 和临时资源都在 Physical IR 闭合。
- `RISCVIntrinsicC.cpp::compileRVVSegmentPairLoad` 只拼写已选 `vlseg2` 和 tuple component binding，不改 layout、不选择其它 load、不发明 outer traversal。
- `VerifyFinalRISCV` 纳入新 terminal op；资源 pass 按两个 SSA 结果和所有静态 tuple parts 的显式临时占用重新核算。实际汇编中 tuple component binding 没有增加搬运指令，但没有据此把临时资源声明为零。

规则没有 hardcode `bsum`、格式、kernel 或 target 名称；所有名称只用于本文定位输入。多轴、tail、动态 base、跨写入区间不属于这个已实现关系。新内存合同已同步到 [Physical IR](../doc/compiler/riscv-ir.md) 和 [passes](../doc/compiler/passes.md)。

参考 /home/kingdom/phdworks/ref/triton 的 `Coalesce.cpp:82-119`、/home/kingdom/phdworks/ref/tilelang 的 `loop_vectorize.cc:216-266` 核对“从访问与 consumer 约束选择表示”的职责；没有引入 GPU thread 模型或 whole-kernel 模板。

## 所有命中条目

下面均为 standalone 或矩阵 decode。对本轮原始低于线集合使用首轮实测；已过线回归项的“之前”明确标为轮初表值，不冒称同轮 A/B。最后一列是每个热 record 的 segment load 数；对应 strided bsum load 均已消失。

| Kernel | Target | 之前来源 | 之前 ratio | 当前 ratio | segment loads |
| --- | --- | --- | ---: | ---: | ---: |
| iq1_s_q8_k | SG2044 | 本轮首跑 | 0.783627 | 0.805852 | 1 |
| mul_mat_q4_k | SG2044 | 本轮首跑 | 0.798548 | 0.944265 | 2 |
| q4_k_q8_k | SG2044 | 本轮首跑 | 0.786178 | 0.882407 | 2 |
| mul_mat_iq1_s | SG2044 | 本轮首跑 | 0.790392 | 0.811565 | 1 |
| q5_k_q8_k | SG2044 | 轮初表值 | 4.044228 | 4.371549 | 2 |
| mul_mat_q5_k | SG2044 | 轮初表值 | 4.099084 | 4.408306 | 2 |
| iq1_s_q8_k | K1/X60 | 本轮首跑 | 0.779498 | 0.791132 | 1 |
| mul_mat_iq1_s | K1/X60 | 本轮首跑 | 0.798248 | 0.808033 | 1 |
| q4_k_q8_k | K1/X60 | 轮初表值 | 1.353970 | 1.396533 | 1 |
| q5_k_q8_k | K1/X60 | 轮初表值 | 1.357253 | 1.381305 | 1 |
| mul_mat_q5_k | K1/X60 | 轮初表值 | 1.368556 | 1.392568 | 1 |
| mul_mat_q4_k | K1/X60 | 轮初表值 | 1.378138 | 1.436148 | 1 |

6 个 prefill 回归未生成新 op，数字整体接近原值：SG IQ1_S/Q4_K/Q5_K 为 **0.645977 / 0.959581 / 1.827887**，K1 为 **1.004122 / 1.608617 / 2.042186**。它们不是这条规则的泛化正例。SG IQ1_S 和 Q5_K prefill、K1 Q4_K prefill 有轻微下降，也如实写回对照表。

## 不保留的尝试

### Q4_K unroll

固定其余 source/physical binding，仅枚举现有合法域的 unroll=1/2/4/8，standalone 与 decode 各 4 个候选。所有候选数值通过，两个入口都仍是原来的 8 最快，因此没有改 tuning.json。

| unroll | SG standalone GOP/s | SG decode GOP/s |
| ---: | ---: | ---: |
| 1 | 5.632108 | 5.618177 |
| 2 | 8.029827 | 8.262302 |
| 4 | 8.424326 | 8.462641 |
| 8 | 8.875111 | 8.991303 |

调优后另用普通入口各复跑一次，表中采用最终复跑值，不采用过程中的最高单次值。搜索预算各 4，完整候选与拒绝/数值记录由既有 tuner 保存，没有新增覆盖合法性的成本分数。

### Field 身份比较

`sharesConcreteFieldSupply` 的对象身份判断看起来可能把等价 field 误判成不同供应，因此做了两版窄原型：

1. 同一 owner/name、完整 result type/access 相等；
2. 同一 owner/name、logical element/shape/axes 相等，排除物理表示的差别。

在 184 份可重生成程序上，两版均 **184 编译通过、生成 C 变化 0**，没有实际收益，全部撤回。没有提交这两版代码，也没有为相同的 C 再做一轮全量硬件测量。final IR 中看到重复 FieldOp，并不能直接证明这个更早的调度判断就是瓶颈。

对 IQ2_XXS 再追到真实的 issue 路径：

- 当前输入全是 `within-record`，`VectorizeRISCVRecordLoops` 的 across-records RawKey 不是它们正在执行的 owner。
- `word1` 是同时供 sign 和 scale 的 shaped u32 值，但 `RegisterMaterializeOp::verify`（`RISCVDialect.cpp:5284`）的 `physical-share` 只接受 scalar carrier；不能直接给 shaped word1 加标记。
- `cloneIssueWindow` 在计划类型改变时重建 producer；deferred scale 在 `MaterializeRISCVPartialAccumulators.cpp:6119` 使用独立 clone map。现有 materialize 分支也会递归 clone input，不是任意 shaped 供应的冻结边界。
- 因而仍缺“同一 shaped metadata 的一次供应 + scalar/vector issue 投影 + 完整 lifetime/resource”关系。sign 两路 u16 gather 与 scale 的 scalar 重读尚未消除，SG standalone/decode 当前仍约 **0.408704 / 0.414959**。没有把这项诊断写成已解决，也没有默认改作者数值树。

## 验收与复现

保留实现的 18 个 cohort 输入均按原有外部 ggml oracle 数值通过，原 tolerances、shape、单核、10 次冷态 median、64 MiB eviction 和相同 Clang/数值 flags 未变。没有用生成 IR/C 或编译成功代替运行。

18/18 完成 parse/verify、清 final resource marker 和三项资源统计、包含 rematerialization 的收尾重放、再次 CSE 后资源重算及第二轮原始文本 diff=0。资源修正后生成的 18 份 C 与已执行 C 直接比较全部相同。最后 4 个普通入口复跑另作同一机械重放，详见各自 results.json；这些集合有重复，不相加宣称独立覆盖量。

从仓库根目录使用既有 rvv/k1 SSH 与工具链配置运行：

```bash
WEFT_KEEP_ARTIFACTS=1 examples/run/weft-quantized-vec-dot.sh sg2044 q4_k 10
WEFT_KEEP_ARTIFACTS=1 examples/run/weft-mul-mat.sh sg2044 q4_k decode 10
WEFT_KEEP_ARTIFACTS=1 examples/run/weft-quantized-vec-dot.sh k1 iq1_s 10
WEFT_KEEP_ARTIFACTS=1 examples/run/weft-mul-mat.sh k1 q5_k decode 10
```

不要设置额外 `WEFT_AUTO_*`、`WEFT_META_*` 或 `WEFT_TUNE_SELECTION`。待定 CSV 的每一行另附原生 repro 命令。它的 `source` 路径相对仓库根目录，`evidence` 中仓库相对路径指历史快照，其余为本轮绝对 artifact 路径。

证据在仓库外 `/tmp/weft-ratio95-supply.ShJU9a`：

- `selection.json`、`baseline-{sg2044,k1}.jsonl`：轮初选择与 30 条首跑。
- `segment-pair-{sg2044,k1}.jsonl`、`segment-final-sg2044.jsonl`：18 个 cohort 与最后 4 个普通入口复跑，包含配置、数值、计时和本地/远端 artifact。
- `SG2044-q4_k-segment.s`、`K1-iq1_s-segment.s`：真实同 flags 汇编，分别在 177/179、163 行出现 segment load。
- `segment-pair-mechanical/results.json`、`segment-final-mechanical/results.json`：完整命令、验证结果与二次文本比较。
- `tune-segment-{vec-dot,mul-mat}-sg2044/`：两个有界搜索的 search/candidate/selected 与原始日志。
- `field-identity-before/`、`field-identity-after/`、`field-logical-identity-after/`：184 份 C 的直接影响比较，未使用 hash/checksum。

保留节点已提交：`becf8a6e3` 刷新待定基线，`d9d2ee2cf` 闭合成对 segment load。其余缺口按 [29 条待定清单](ratio95-segment-pair-pending.csv) 保留；本轮未修改的类别只重跑并继承上一轮定位，没有包装成新的归因或理论性能下界。
