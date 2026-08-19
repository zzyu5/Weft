# Weft baseline Kernel 全量真机重测

## 结论

本轮只运行存在固定 source baseline、并且已有对应 Weft DSL kernel 的项目。所有项目均经过当前
唯一主链：

~~~text
DSL kernel
→ canonical Kernel IR
→ MLIR RISC-V artifact pass
→ intrinsic C / local IME asm
→ target system compiler
→ SG2044 或 K1 真机执行
~~~

最终结果：

| 目标 | 条目 | 结果 |
|---|---:|---:|
| SG2044 / RVV VLEN128 | 121 | 121 PASS |
| K1/X60 / RVV VLEN256 | 123 | 123 PASS |
| K1/X60 / IME1 VLEN256 | 6 | 6 PASS |
| 总计 | 250 | 250 PASS |

没有 compile failure、unsupported、runtime failure 或 runtime 数值检查失败。固定 GGML baseline
source、baseline Markdown 和 baseline CSV 均未修改；本轮只更新当前 Weft 性能数字表
report/weft-kernel-performance.csv 的 250 条对应记录。

## 一、运行集合

全量集合不是所有 examples，而是以下能够和 source baseline 对应的项目：

| 类别 | 条目数 | 覆盖 |
|---|---:|---|
| source_mul_mat | 104 | 两台机器，26 种 dense/quant format，decode 与 prefill |
| source_mul_mat_ime | 6 | K1 IME，Q4_0/Q4_1/Q4_K，decode 与 prefill |
| source_vec_dot | 48 | 两台机器，24 种量化 row-dot |
| source_quantization | 6 | 两台机器，Q8_0/Q8_1/Q8_K activation quantize |
| source_dequantization | 48 | 两台机器，24 种 row dequantize |
| source_forward | 38 | pointwise、normalization、layout、gather、RoPE、attention |
| 总计 | 250 | SG 121 + K1 RVV 123 + K1 IME 6 |

每条记录沿用性能 CSV 已有的：

- algorithm、shape、dtype/quant format 与 phase；
- target profile；
- repetitions、warmup 与 64 MiB eviction 配置；
- correctness scope；
- preprocessing 是否进入 timed region；
- throughput 单位。

本轮没有为了缩短时间统一降低 repetitions。普通 forward、quantize 和 dequantize runtime 继续
使用各自固定的 10 次采样；MUL_MAT、vec-dot 与 IME 使用 CSV 已记录的 3 次或 10 次协议；IME
仍执行 3 次 warmup。

## 二、运行方式

仓库没有保存全量 batch runner。本轮在仓库外临时组合现有单项入口
examples/run/weft.sh：

- SG2044 与 K1 两台机器并行运行；
- 同一台机器上的条目严格串行；
- K1 IME 在 K1 RVV 全部结束后执行；
- 每条日志绑定现有性能 CSV 的唯一行；
- 临时 orchestration 和日志未进入仓库。

IQ entry 的大小写差异只在临时 manifest 中映射：

~~~text
iq1_m_q8_K → iq1_M_q8_K
iq2_s_q8_K → iq2_S_q8_K
iq3_s_q8_K → iq3_S_q8_K
~~~

没有向 DSL、compiler 或正式 runner 添加 alias、compatibility layer 或 fallback。

## 三、正确性结果

250 条 runtime 均以退出状态 0 完成，并通过各 runtime 已有的数值对照。

整体日志中：

- 6 条 IME activation_code_mismatches 全部为 0；
- 最大 absolute error 为 0.0156254768，来自 K1 IME Q4_0 decode；
- 最大 relative error 为 0.0318279199，来自 SG/K1 FlashAttention；
- Q5_K vec-dot 最大 absolute error 为 0.00146484375；
- 其余误差按每个 runtime 的完整或 sampled correctness scope 记录在性能 CSV。

这些数值只是如实记录每项 runtime 的现有数值合同，不引入新的全局误差阈值或验收状态机。

## 四、性能 CSV 更新

report/weft-kernel-performance.csv 的第 151–400 行已由本轮真机日志覆盖。保持不变的字段包括：

- category、kernel、phase；
- hardware、target、implementation；
- shape、scope；
- repetitions、warmup、eviction_mib；
- correctness_scope；
- preprocess_in_timed_region；
- throughput_unit。

由真实日志更新的字段是：

- activation_code_mismatches；
- activation_scale_max_absolute_error；
- max_absolute_error；
- max_relative_error；
- median_ms；
- throughput。

第 2–150 行的历史 compiler-pressure 数字没有参与本轮运行，也没有修改。

## 五、和固定 source baseline 的比较口径

Weft 表共有 250 条本轮结果，而固定 source baseline 是 244 条硬件记录。差异来自 K1 Q4：

- source baseline 的 Q4_0/Q4_1/Q4_K decode/prefill 使用 IME1，共 6 条；
- Weft 同时记录 6 条标准 RVV realization 和 6 条 IME realization。

因此性能汇总使用：

- 238 条同硬件、算法、shape、phase，且不跨 RVV/IME target class 的普通记录；
- 6 条 Weft IME 对 source IME；
- 合计 244 条直接可比记录。

这里的“直接可比”不要求 intrinsic/asm implementation 字符串相同：例如 source scalar
dequantize 与 Weft RVV dequantize 正是同一算法在同一硬件上的实现竞争。它要求 target class
可比，因此另外 6 条 K1 RVV Q4 虽保留在 Weft CSV，但 source 只有 IME 实现，没有纳入聚合。
跨 SG2044 与 K1 的绝对时间也不互相比较。

## 六、总体性能结果

244 条直接可比记录：

| 指标 | 结果 |
|---|---:|
| Weft 时间不高于 source baseline | 113 |
| Weft 时间高于 source baseline | 131 |
| 全部记录几何平均速度比 | 1.023× |

这里速度比定义为：

~~~text
source baseline median_ms / Weft median_ms
~~~

大于 1 表示 Weft 更快。总体 1.023× 只说明完整集合的几何平均接近并略高于 baseline，不能掩盖
不同 compiler capability 之间很明显的不均衡。

## 七、按共享能力分类

| 类别 | 目标 | 可比项 | Weft 更快或相等 | 几何平均速度比 |
|---|---|---:|---:|---:|
| MUL_MAT | SG2044 RVV | 52 | 16 | 0.820× |
| MUL_MAT | K1 RVV | 46 | 16 | 0.774× |
| MUL_MAT | K1 IME | 6 | 0 | 0.781× |
| vec-dot | SG2044 RVV | 24 | 10 | 0.944× |
| vec-dot | K1 RVV | 24 | 6 | 0.818× |
| activation quantize | SG2044 RVV | 3 | 1 | 0.933× |
| activation quantize | K1 RVV | 3 | 2 | 0.974× |
| row dequantize | SG2044 RVV | 24 | 15 | 1.273× |
| row dequantize | K1 RVV | 24 | 24 | 2.539× |
| forward | SG2044 RVV | 18 | 10 | 1.218× |
| forward | K1 RVV | 20 | 13 | 1.222× |

结论很清楚：

- row dequantize 是当前最成熟的共享能力，尤其 K1 的 24 项全部快于 source baseline；
- forward 整体优于 baseline，但内部仍有明显例外；
- activation quantize 已接近 baseline；
- dense/quantized MUL_MAT 仍是最大的总体性能负担；
- vec-dot 在 SG 接近 baseline，在 K1 仍明显偏弱；
- IME 路径全部正确，但当前 6 项均未达到 source IME 时间。

## 八、代表性领先项

| Kernel | 目标 | Weft ms | baseline ms | 速度比 |
|---|---|---:|---:|---:|
| dequantize_tq1_0 | K1 | 4.245544 | 37.953 | 8.939× |
| mul_mat_q8_0 decode | SG2044 | 7.209649 | 60.024 | 8.326× |
| dequantize_tq2_0 | K1 | 3.392670 | 24.113 | 7.107× |
| sum_rows | SG2044 | 0.314601 | 2.145 | 6.818× |
| q8_0_q8_0 vec-dot | SG2044 | 31.506356 | 212.232 | 6.736× |
| dequantize_q2_K | K1 | 5.630491 | 31.204 | 5.542× |
| dequantize_q4_1 | K1 | 5.327934 | 28.075 | 5.269× |
| dequantize_q4_0 | K1 | 4.771162 | 24.574 | 5.151× |

这些结果来自不同的共享能力族，不能用于声称整个 compiler 已经全面领先 baseline。

## 九、主要性能缺口

| Kernel | 目标 | Weft ms | baseline ms | 速度比 |
|---|---|---:|---:|---:|
| mul_mat_q2_K prefill | SG2044 | 3479.495057 | 453.908 | 0.130× |
| mul_mat_q2_K decode | SG2044 | 27.203569 | 3.806 | 0.140× |
| q2_K_q8_K vec-dot | SG2044 | 73.871836 | 12.617 | 0.171× |
| contiguous transpose | K1 | 21.110746 | 4.251 | 0.201× |
| F16 MUL_MAT prefill | K1 | 3819.173396 | 782.657 | 0.205× |
| dequantize_iq1_m | SG2044 | 27.691601 | 5.987 | 0.216× |
| F16 MUL_MAT decode | K1 | 40.513012 | 9.785 | 0.242× |
| q2_K_q8_K vec-dot | K1 | 164.102008 | 48.013 | 0.293× |
| mul_mat_q2_K decode | K1 | 46.917902 | 13.958 | 0.297× |
| mul_mat_q2_K prefill | K1 | 5985.944128 | 1781.885 | 0.298× |

这些不是十条独立 kernel bug。它们集中暴露：

- Q2 packed decode、widen 与 dot 的组织仍然低效；
- F16 matmul 在 K1 的 lane/register mapping、load reuse 或 widening microkernel 仍不成熟；
- K1 transpose 的 indexed/strided memory schedule 明显不足；
- SG IQ1_M dequantize 的 codebook/decode realization 没有复用 baseline 的有效组织；
- dense prefill 的 microtile、multiple accumulators 与 load/compute overlap 仍不足。

本轮只负责完成全量测试和记录，没有为这些数字增加 kernel-specific fast path。

## 十、K1 IME 结果

| Kernel | Phase | Weft ms | source IME ms | 速度比 |
|---|---|---:|---:|---:|
| Q4_0 | decode | 3.192315 | 2.941 | 0.921× |
| Q4_0 | prefill | 233.693783 | 151.069 | 0.646× |
| Q4_1 | decode | 4.001947 | 3.519 | 0.879× |
| Q4_1 | prefill | 245.594566 | 174.724 | 0.711× |
| Q4_K | decode | 4.006324 | 3.436 | 0.858× |
| Q4_K | prefill | 245.848925 | 174.758 | 0.711× |

六项几何平均速度比为 0.781×。Decode 已较接近 source leaf；prefill 的 fragment reuse、packing
与局部流水仍有较大差距。六项 activation code mismatch 均为 0，说明当前差距不是错误输入
packing 或静默 RVV fallback。

## 十一、本轮得到的事实

1. 新 MLIR compiler invocation 不只对一个 GEMM repro 生效，而是完整承载了 250 条已有
   baseline realization。
2. 250 条均通过同一 DSL → Kernel IR → RISC-V artifact 主链，没有旧 compiler 或 source
   runtime fallback。
3. 固定 baseline 覆盖的主要正确性能力在 SG2044 RVV、K1 RVV 与 K1 IME 上全部保持。
4. 当前性能总体接近 baseline，但能力成熟度不均衡；不能用总体几何平均掩盖 dense、Q2、K1
   F16、K1 transpose 与 IME prefill 的明确缺口。
5. 这些缺口已经能够按共享 physical capability 归因，不需要恢复 whole-kernel route。
