# Q2_K 物理 topology 收口与横向证据

## 结论

当前唯一代码状态下，Q2_K production prefill 在两台机器上都超过同目标 source：

| target | Weft | source | Weft/source | correctness |
|---|---:|---:|---:|---|
| SG2044 / VLEN128 | 10.147413 GOP/s | 9.462195 GOP/s | 107.2% | within tolerance |
| K1 / VLEN256 | 2.962105 GOP/s | 2.410350 GOP/s | 122.9% | within tolerance |

两项结果均为 `M=128,N=4096,K=4096`、10 repetitions、Clang 18、相同 runner 计时范围。当前数字已经写入 `report/weft-kernel-performance.csv`。

本轮没有修改 Q2_K 作者树。regular-repeat load/gather、independent partial set 和 pairwise final topology 继续由现有 typed physical IR 承载。

## 先数后改：横向验证发现的错误

在修改代码前，对 27 个 quantized prefill entry 的最终 RISC-V IR 做了静态扫描：

- `rvv_regular_repeat_index/gather` 只出现在 K1 Q2_K；SG2044 没有 entry 触发。
- SG Q2_K 形成两组 scaled partial topology。
- K1 Q2_K 形成两组 `partial_set -> partial_combine -> partial_finalize`。
- K1 TQ2_0 也被 independent-partial 规则捕获，形成 8 个 set、16 个 combine、8 个 finalize。
- Q5_K、IQ4_XS、IQ4_NL 都没有形成 regular-repeat 或 independent-partial topology。

K1 TQ2_0 的生成程序给出了反例。每个 4x2 output tile、每个 K block 中，它执行：

- 8 个 independent partial set；
- 16 个 pairwise combine；
- 8 个 final reduction；
- 对应生成 C 中共 64 次 widened multiply、48 次 vector add、16 次 vector reduction。

这不是 TQ2_0 的作者数值树要求。它的 `RVVWidenDotOp` 已带 `stream_reduction = "fused"`，并且 use-def 中有唯一 typed layered root。该 dot 应由 fused layered-stream topology 物化，而不是把顺序 stream slots 解释为 independent partial slots。

真实运行确认了静态计数：在本轮修改前的当前 HEAD 上，K1 TQ2_0 从历史正式值 6.418805 GOP/s 退化到 2.132298 GOP/s。这个负结果说明 `slots >= 4 && power-of-two && resource-fit` 本身不足以定义 independent partial。

## 修改

`lib/Target/MaterializeRISCVPartialAccumulators.cpp` 中，independent-partial materialization 现在跳过 `stream_reduction = "fused"` 的 dot，把它交给紧随其后的 fused layered-stream materialization。

这项优先级来自已有 typed contract：

- `stream_reduction` 只允许 `fused` 或 `per_stream`；
- fused 路径继续验证 reduction axis、time/lane factors、唯一 layered root、field/origin、group/layer geometry、tail 和 resource facts；
- independent 路径仍处理没有 fused storage ownership 的合法 partial set。

修改不读取格式、kernel 或 target 名，也不改变作者树。它解决的是两个已有 physical topology 谁拥有该 dot，而不是增加第三条 lowering。

参考机制：

- Triton `ReduceOpToLLVM.cpp` 根据值的 typed layout 组织 thread-local、lane 和 tree reduction，而不是仅按 reduction extent 选择一种树。
- TileLang `reducer_plan_materialize.cc` 要求多个 update site 得到 structurally equal partial plan；不满足窄计划条件时不会把任意 slots 强行解释成同一种 partial topology。

## 修改后的程序形态

K1 TQ2_0 的最终 RISC-V IR 从：

```text
8 partial_set
16 partial_combine
8 partial_finalize
```

变成：

```text
1 layered_storage_load
4 layered_storage_decode
4 widen_accumulate
1 finalize_widen_dot
```

K1 TQ2_0 的 10-repetition 结果恢复并超过原正式值：

| target | current Weft | source | Weft/source | correctness |
|---|---:|---:|---:|---|
| SG2044 | 20.495379 GOP/s | 6.982594 GOP/s | 293.5% | exact for this input |
| K1 | 6.657391 GOP/s | 4.979196 GOP/s | 133.7% | exact for this input |

因此本轮同时得到一条正证据和一条负证据：typed topology priority 能恢复 TQ2_0；但“有 4 个以上 time slots”不能作为 independent-partial 的充分条件。

## 横向结果

以下是修改后 3-repetition 的受影响样本检查；它们不替代 CSV 中的正式 10-repetition 记录。

| format | SG2044 | K1 | source SG / K1 | observation |
|---|---:|---:|---:|---|
| TQ2_0 | 20.414291 | 6.696232 | 6.982594 / 4.979196 | fused layered topology 生效 |
| Q5_K | 7.096138 | 2.672509 | 2.702302 / 1.250635 | 没有进入本轮 topology，性能保持 |
| IQ4_XS | 3.487009 | 1.275652 | 5.894772 / 1.718922 | 没有进入本轮 topology，仍低于 source |
| IQ4_NL | 8.630551 | 2.461375 | 6.954179 / 2.869061 | 没有进入本轮 topology；SG 过线、K1 未过线 |

没有生效的具体边界：

- Q5_K 同时有 q/qh 两个 layered roots，scale/min 又是 joined relation；当前 topology 不拥有联合多 root window。
- IQ4_XS 的 scale relation仍是 typed sub projection 之外的 scalar index/shift，lookup result 也不是 partial root。
- IQ4_NL 有 layered payload，但 codebook lookup result 不是 storage/partial window root。

regular-repeat 的全仓扫描没有找到第二个现有 entry：当前只有 K1 Q2_K 满足 natural indexed field、regular selector、stride 1、repeat/time/lane closure 等完整条件。代码规则不含格式名，但现有语料不能提供跨格式的正向运行证据；这一点不能用“实现是通用的”替代。

## SG2044 回归

本轮修改后，既有六条过线路径均完成 3-repetition 回归：

| entry | Weft GOP/s | source GOP/s | Weft/source |
|---|---:|---:|---:|
| Q4_0 | 8.431264 | 7.580451 | 111.2% |
| Q4_1 | 6.658134 | 2.535208 | 262.6% |
| Q5_1 | 6.411515 | 5.950653 | 107.7% |
| IQ4_NL | 8.630551 | 6.954179 | 124.1% |
| Q1_0 | 6.933841 | 4.249822 | 163.2% |
| Q4_K persistent | 10.147606 | 9.699175 | 104.6% |

所有回归均为 `within-tolerance`；没有恢复旧 path 或 emitter-side 决策。

## 可复现命令

```bash
bash examples/run/weft-mul-mat.sh sg2044 q2_k prefill 10
bash examples/run/weft-mul-mat.sh k1 q2_k prefill 10
bash examples/run/weft-mul-mat.sh sg2044 tq2_0 prefill 10
bash examples/run/weft-mul-mat.sh k1 tq2_0 prefill 10
```

`WEFT_KEEP_ARTIFACTS=1` 可临时保留 generated C 与远端汇编；本轮分析使用的临时 artifacts 在提交前全部删除。
