# GGML RISC-V Kernel Baseline

本文件是后续 Weft 单 kernel 性能对照的唯一 GGML baseline 口径。完整模型
`llama-bench` 仍是独立的 e2e baseline，不能与这里的单 kernel 时间共用分母。

## 最终集合

当前 `ssh rvv` 上实际使用的是标准 RVV、VLEN=128 的 GGML CPU build。按 production
primitive entry 去重，不重复计算同一 entry 内部的 VLEN body、generic fallback、reference
implementation 或 graph traversal，最终 baseline 是 **69 个可运行 kernel**：

| family | 数量 | RVV intrinsic | 当前标量/库实现 | 当前手写汇编 |
|---|---:|---:|---:|---:|
| quantized vec-dot | 24 | 23 | 1 | 0 |
| activation row quantize | 3 | 3 | 0 | 0 |
| row dequantize | 24 | 0 | 24 | 0 |
| LLM forward | 18 | 3 | 15 | 0 |
| **总计** | **69** | **29** | **40** | **0** |

69/69 均已在目标机完成动态链接、真实调用并返回有限输出；没有用 scalar stub 冒充
RVV 成功，也没有把未编译源码写成可运行 baseline。

### 实现类别

- 当前实际 RVV intrinsic：23 个 RISC-V vec-dot、`quantize_row_q8_0/q8_1/q8_K`、
  `scale`、`silu`、`softmax`。
- 当前仅标量或标准库路径：`nvfp4_q8_0`、全部 24 个 row-dequant、其余 15 个
  forward entry。其中 `cpy/cont` 主要是 scalar/`memcpy`，FlashAttention 是当前 generic
  mixed path。
- 当前实际手写汇编：**0 个**。

源码中确实存在手写汇编，但它们不是这台机器当前选择的 baseline：

- `q2_K/q3_K/q4_K/q6_K` 有 XTheadVector inline-asm alternate body；当前 build 使用标准
  `__riscv_v`，四项实际走 RVV intrinsic。
- SpacemiT IME1/IME2 source 含手写 custom-instruction asm，但当前 build 没有启用
  `GGML_CPU_RISCV64_SPACEMIT`，动态库中没有对应符号。
- RISC-V repack source 有 13 个 RVV quantize/GEMV/GEMM entry；VLEN128 的 production
  selector 对 16×1 路径明确不选择，q4_0 8×8 又要求至少 VLEN256，因此当前可达数为 0。
  这些 source-only entry 和 48 个 generic fallback 不进入 69 项 baseline。

## Cold-memory 口径

这里的“冷启动”沿用项目原有含义：**cache-cold 的稳定 kernel invocation**，不是进程
第一次调用、动态库加载、SSH、编译或 codegen 时间。

- 目标：`ssh rvv`，标准 RVV，VLEN=128，固定 CPU 8，单线程；
- runtime 和动态链接在计时前完成；
- 每个 kernel 先完整 warmup 一次；
- 每次计时前顺序读写 64 MiB eviction buffer；eviction 本身不计时；
- 每项运行 10 次，表中为 wall-time median；
- 输入数值确定性构造，shape、block layout、buffer footprint 和调用次数使用完整模型级
  尺寸，不使用 16×16、17×18×19 等 toy case；
- IQ1/IQ2/IQ3 没有普通 row quantizer，使用合法的全零 quantized block；这能真实执行
  同一 RVV entry 和完整矩阵调用次数，但不表示真实 GGUF 数值分布。

已知远端两个长期高负载进程位于 CPU 37 和 56，不与固定 CPU 8 共核。`cont` 的一次
首轮 sweep 出现外部干扰；随后两次独立 cold median 为 13.476 ms 和 13.532 ms，表中采用
后者。其余抽查离群项均稳定复现。

## Quantized vec-dot

统一 shape：DeepSeek-R1-Distill-Llama-8B FFN-up decode，`M=1, N=14336, K=4096`。
时间覆盖完整 `N` 个输出 dot；吞吐按 `2MNK` 计算。

| kernel | 当前实现 | cold median (ms) | GOP/s |
|---|---|---:|---:|
| `q1_0_q8_0` | RVV intrinsic | 51.944 | 2.261 |
| `q4_0_q8_0` | RVV intrinsic | 47.494 | 2.473 |
| `q4_1_q8_1` | RVV intrinsic | 46.328 | 2.535 |
| `q5_0_q8_0` | RVV intrinsic | 18.469 | 6.359 |
| `q5_1_q8_1` | RVV intrinsic | 19.979 | 5.878 |
| `q8_0_q8_0` | RVV intrinsic | 212.518 | 0.553 |
| `q2_K_q8_K` | RVV intrinsic | 12.622 | 9.305 |
| `q3_K_q8_K` | RVV intrinsic | 15.901 | 7.386 |
| `q4_K_q8_K` | RVV intrinsic | 11.781 | 9.968 |
| `q5_K_q8_K` | RVV intrinsic | 90.311 | 1.300 |
| `q6_K_q8_K` | RVV intrinsic | 23.592 | 4.978 |
| `iq1_s_q8_K` | RVV intrinsic | 41.872 | 2.805 |
| `iq1_m_q8_K` | RVV intrinsic | 25.361 | 4.631 |
| `iq2_s_q8_K` | RVV intrinsic | 48.922 | 2.401 |
| `iq2_xs_q8_K` | RVV intrinsic | 33.476 | 3.508 |
| `iq2_xxs_q8_K` | RVV intrinsic | 28.693 | 4.093 |
| `iq3_s_q8_K` | RVV intrinsic | 143.806 | 0.817 |
| `iq3_xxs_q8_K` | RVV intrinsic | 76.175 | 1.542 |
| `iq4_nl_q8_0` | RVV intrinsic | 16.721 | 7.023 |
| `iq4_xs_q8_K` | RVV intrinsic | 43.325 | 2.711 |
| `tq1_0_q8_K` | RVV intrinsic | 18.880 | 6.220 |
| `tq2_0_q8_K` | RVV intrinsic | 17.648 | 6.655 |
| `mxfp4_q8_0` | RVV intrinsic | 17.626 | 6.663 |
| `nvfp4_q8_0` | scalar | 310.581 | 0.378 |

## Activation quantize

统一 shape：DeepSeek 8B FFN-down prefill activation，`M=128, K=14336`。

| kernel | 当前实现 | cold median (ms) | MElements/s |
|---|---|---:|---:|
| `quantize_row_q8_0` | RVV intrinsic | 3.294 | 557.016 |
| `quantize_row_q8_1` | RVV intrinsic | 4.500 | 407.795 |
| `quantize_row_q8_K` | RVV intrinsic | 3.969 | 462.372 |

只纳入这三个 inference activation quantizer。其余 16 个 scalar `from_float` wrapper 主要
服务模型量化、转换或 reference 路径，不是 quantized matmul 的 activation hot path，因此
不重复扩张 baseline。

## Row dequantize

统一 shape：DeepSeek 8B attention-K tensor，`N=1024, K=4096`。24 项均是 GGML 当前
trait-visible scalar implementation。

| kernel | cold median (ms) | MElements/s |
|---|---:|---:|
| `dequantize_row_q1_0` | 9.303 | 450.833 |
| `dequantize_row_q4_0` | 4.311 | 973.028 |
| `dequantize_row_q4_1` | 4.809 | 872.142 |
| `dequantize_row_q5_0` | 4.507 | 930.618 |
| `dequantize_row_q5_1` | 4.778 | 877.833 |
| `dequantize_row_q8_0` | 5.758 | 728.391 |
| `dequantize_row_mxfp4` | 24.627 | 170.317 |
| `dequantize_row_nvfp4` | 23.008 | 182.294 |
| `dequantize_row_q2_K` | 19.512 | 214.955 |
| `dequantize_row_q3_K` | 15.874 | 264.216 |
| `dequantize_row_q4_K` | 5.817 | 720.997 |
| `dequantize_row_q5_K` | 9.496 | 441.706 |
| `dequantize_row_q6_K` | 14.065 | 298.217 |
| `dequantize_row_tq1_0` | 9.283 | 451.844 |
| `dequantize_row_tq2_0` | 5.718 | 733.562 |
| `dequantize_row_iq2_xxs` | 10.891 | 385.108 |
| `dequantize_row_iq2_xs` | 9.421 | 445.213 |
| `dequantize_row_iq2_s` | 9.151 | 458.356 |
| `dequantize_row_iq3_xxs` | 14.454 | 290.178 |
| `dequantize_row_iq3_s` | 10.179 | 412.059 |
| `dequantize_row_iq1_s` | 5.919 | 708.580 |
| `dequantize_row_iq1_m` | 5.927 | 707.662 |
| `dequantize_row_iq4_nl` | 25.707 | 163.160 |
| `dequantize_row_iq4_xs` | 26.090 | 160.764 |

`q8_1` 没有 production dequant entry；`q8_K` 虽有 helper definition，但没有挂到当前
production `to_float` trait，因此二者不计入这 24 项。

## LLM forward

这些 entry 覆盖 dense decoder、常见 GELU 模型、MoE row/repeat/concat 辅助路径，以及可选
FlashAttention。所有 shape 都来自 DeepSeek/Llama 8B 级模型，而非统一小向量。

| kernel | 当前实现 | 模型级 shape | cold median (ms) |
|---|---|---|---:|
| `add` | scalar | hidden `[128,4096]` | 0.783 |
| `sub` | scalar | hidden `[128,4096]` | 0.791 |
| `mul` | scalar | hidden `[128,4096]` | 0.787 |
| `div` | scalar | hidden `[128,4096]` | 0.759 |
| `scale` | RVV intrinsic | hidden `[128,4096]` | 0.982 |
| `cpy` | scalar/`memcpy` | hidden `[128,4096]` | 0.730 |
| `cont` | scalar/`memcpy` | transpose of hidden `[128,4096]` | 13.532 |
| `gelu` | scalar | FFN `[128,14336]` | 5.133 |
| `silu` | RVV intrinsic | FFN `[128,14336]` | 4.110 |
| `norm` | scalar | hidden `[128,4096]` | 3.503 |
| `rms_norm` | scalar | hidden `[128,4096]` | 2.635 |
| `softmax` | RVV intrinsic | heads=32, Q=128, K=128 | 4.203 |
| `rope` | scalar | head_dim=128, heads=32, tokens=128 | 1.311 |
| `get_rows` | scalar Q4_K dequant | vocab=128256, hidden=4096, tokens=128 | 0.823 |
| `repeat` | scalar | norm `[4096]` → hidden `[128,4096]` | 0.741 |
| `sum_rows` | scalar | hidden `[128,4096]` | 2.144 |
| `concat` | scalar | hidden `[64+64,4096]` | 1.013 |
| `flash_attn` | generic mixed | Q=128, KV=128, H=32, Hkv=8, D=128 | 18.196 |

当前 build 没有编译 SpacemiT RVV forward traits，所以即使 source 中存在对应 target kernel，
这里仍按动态库实际执行的 generic 路径分类。

## 运行入口

每次只运行一个明确 kernel：

```bash
./examples/run/ggml-kernel.sh \
  <vec_dot|quantize|dequantize|forward> \
  <kernel> \
  10
```

四个 family 各自拥有独立 runtime；runner 只负责上传、按目标 build flags 编译、固定 CPU
并执行，不包含 kernel 语义分支：

```text
examples/repro/ggml/vec_dot_runtime.cpp
examples/repro/ggml/quantize_runtime.cpp
examples/repro/ggml/dequantize_runtime.cpp
examples/repro/ggml/forward_runtime.cpp
```

未来 Weft intrinsic C 只能在相同 shape、layout、线程数和 cold-memory 口径下与对应行做
比值。需要完整模型结论时，另用 `examples/run/ggml.sh` 的 prompt/decode token/s；不得拿
本表 kernel ms 推导模型 token/s。
