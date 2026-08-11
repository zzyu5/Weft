# GGML RISC-V Kernel Baseline

## Baseline 结构

全部数据按同一层次组织：先区分 production graph 与 standalone helper，再按算子语义分类；同一逻辑算子在两台硬件都运行时放在同一行横向比较。

| 类别 | 逻辑 case | 硬件实测行 |
|---|---:|---:|
| Production `MUL_MAT`：26 formats × decode/prefill | 52 | 104 |
| Quantized vec-dot projection helper | 24 | 48 |
| Activation quantize helper | 3 | 6 |
| Row dequantize helper | 24 | 48 |
| Forward primitive | 20 | 38 |
| **总计** | **123** | **244** |

## 硬件与测量

| 硬件 | 执行核心 | ISA 路径 | VLEN |
|---|---:|---|---:|
| SG2044 | 48 | 标准 RVV | 128 bit |
| K1/X60 | 3 | 标准 RVV、SpacemiT RVV、IME1 | 256 bit |

所有性能行均为单线程；计时前 warmup 一次，每次计时前遍历 64 MiB eviction buffer，记录 10 次 kernel invocation 的 wall-time median。编译、动态链接、tensor 初始化、weight quantize/repack、warmup 和 eviction 不计入时间。

时间统一使用毫秒。矩阵乘与 vec-dot 吞吐按 `2MNK` 计算为 GOP/s；quantize/dequantize 吞吐为 MElements/s。

## 1. Matrix multiplication

Production 表测量真实 `GGML_OP_MUL_MAT` graph，shape 统一为 Llama-8B hidden projection：`N=4096, K=4096`。decode 为 `M=1`，prefill 为 `M=128`。K1 的 `Q4_0/Q4_1/Q4_K` 使用 production SpacemiT repack + IME1；其余格式使用对应普通 RVV 或 scalar path。SG2044 的 VLEN128 build 不选择 packed `16x1` repack，prefill 仍是完整矩阵乘 graph，但核心 lowering 是逐行调用对应 vec-dot。

### Decode：M=1

| weight format | SG2044 implementation | SG2044 ms | SG2044 GOP/s | K1/X60 implementation | K1/X60 ms | K1/X60 GOP/s |
|---|---|---:|---:|---|---:|---:|
| `f32` | RVV F32 vec-dot | 20.612 | 1.628 | RVV F32 vec-dot | 16.715 | 2.007 |
| `f16` | RVV F16 vec-dot | 7.387 | 4.542 | RVV F16 vec-dot | 9.785 | 3.429 |
| `q1_0` | RVV quantized vec-dot | 14.379 | 2.334 | RVV quantized vec-dot | 9.078 | 3.696 |
| `q4_0` | RVV quantized vec-dot | 13.661 | 2.456 | IME1 asm | 2.941 | 11.409 |
| `q4_1` | RVV quantized vec-dot | 13.336 | 2.516 | IME1 asm | 3.519 | 9.535 |
| `q5_0` | RVV quantized vec-dot | 5.444 | 6.164 | RVV quantized vec-dot | 18.758 | 1.789 |
| `q5_1` | RVV quantized vec-dot | 5.797 | 5.789 | RVV quantized vec-dot | 19.692 | 1.704 |
| `q8_0` | RVV quantized vec-dot | 60.024 | 0.559 | RVV quantized vec-dot | 16.003 | 2.097 |
| `q2_K` | RVV quantized vec-dot | 3.806 | 8.815 | RVV quantized vec-dot | 13.958 | 2.404 |
| `q3_K` | RVV quantized vec-dot | 4.665 | 7.193 | RVV quantized vec-dot | 14.014 | 2.394 |
| `q4_K` | RVV quantized vec-dot | 3.760 | 8.923 | IME1 asm | 3.436 | 9.766 |
| `q5_K` | RVV quantized vec-dot | 25.712 | 1.305 | RVV quantized vec-dot | 27.260 | 1.231 |
| `q6_K` | RVV quantized vec-dot | 6.917 | 4.851 | RVV quantized vec-dot | 14.384 | 2.333 |
| `iq1_s` | RVV quantized vec-dot | 12.033 | 2.788 | RVV quantized vec-dot | 12.531 | 2.678 |
| `iq1_m` | RVV quantized vec-dot | 7.389 | 4.541 | RVV quantized vec-dot | 23.535 | 1.426 |
| `iq2_s` | RVV quantized vec-dot | 16.593 | 2.022 | RVV quantized vec-dot | 23.214 | 1.445 |
| `iq2_xs` | RVV quantized vec-dot | 9.616 | 3.490 | RVV quantized vec-dot | 19.069 | 1.760 |
| `iq2_xxs` | RVV quantized vec-dot | 8.315 | 4.035 | RVV quantized vec-dot | 20.138 | 1.666 |
| `iq3_s` | RVV quantized vec-dot | 40.956 | 0.819 | RVV quantized vec-dot | 20.438 | 1.642 |
| `iq3_xxs` | RVV quantized vec-dot | 24.555 | 1.367 | RVV quantized vec-dot | 21.235 | 1.580 |
| `iq4_nl` | RVV quantized vec-dot | 4.887 | 6.867 | RVV quantized vec-dot | 12.116 | 2.769 |
| `iq4_xs` | RVV quantized vec-dot | 12.412 | 2.703 | RVV quantized vec-dot | 19.812 | 1.694 |
| `tq1_0` | RVV quantized vec-dot | 5.457 | 6.149 | RVV quantized vec-dot | 10.046 | 3.340 |
| `tq2_0` | RVV quantized vec-dot | 5.596 | 5.996 | RVV quantized vec-dot | 7.020 | 4.780 |
| `mxfp4` | RVV quantized vec-dot | 5.156 | 6.508 | RVV quantized vec-dot | 13.191 | 2.544 |
| `nvfp4` | scalar vec-dot | 88.900 | 0.377 | scalar vec-dot | 214.137 | 0.157 |

### Prefill：M=128

| weight format | SG2044 implementation | SG2044 ms | SG2044 GOP/s | K1/X60 implementation | K1/X60 ms | K1/X60 GOP/s |
|---|---|---:|---:|---|---:|---:|
| `f32` | RVV F32 vec-dot | 574.193 | 7.480 | RVV F32 vec-dot | 1606.776 | 2.673 |
| `f16` | RVV F16 vec-dot | 299.138 | 14.358 | RVV F16 vec-dot | 782.657 | 5.488 |
| `q1_0` | RVV quantized vec-dot | 1010.623 | 4.250 | RVV quantized vec-dot | 1132.405 | 3.793 |
| `q4_0` | RVV quantized vec-dot | 1734.067 | 2.477 | IME1 asm | 151.069 | 28.431 |
| `q4_1` | RVV quantized vec-dot | 1694.128 | 2.535 | IME1 asm | 174.724 | 24.581 |
| `q5_0` | RVV quantized vec-dot | 680.931 | 6.307 | RVV quantized vec-dot | 2321.729 | 1.850 |
| `q5_1` | RVV quantized vec-dot | 721.764 | 5.951 | RVV quantized vec-dot | 2429.874 | 1.768 |
| `q8_0` | RVV quantized vec-dot | 2093.162 | 2.052 | RVV quantized vec-dot | 1926.148 | 2.230 |
| `q2_K` | RVV quantized vec-dot | 453.908 | 9.462 | RVV quantized vec-dot | 1781.885 | 2.410 |
| `q3_K` | RVV quantized vec-dot | 594.975 | 7.219 | RVV quantized vec-dot | 1753.587 | 2.449 |
| `q4_K` | RVV quantized vec-dot | 444.056 | 9.672 | IME1 asm | 174.758 | 24.577 |
| `q5_K` | RVV quantized vec-dot | 1589.374 | 2.702 | RVV quantized vec-dot | 3434.231 | 1.251 |
| `q6_K` | RVV quantized vec-dot | 580.638 | 7.397 | RVV quantized vec-dot | 1711.685 | 2.509 |
| `iq1_s` | RVV quantized vec-dot | 1529.498 | 2.808 | RVV quantized vec-dot | 1563.161 | 2.748 |
| `iq1_m` | RVV quantized vec-dot | 941.799 | 4.560 | RVV quantized vec-dot | 2962.101 | 1.450 |
| `iq2_s` | RVV quantized vec-dot | 1531.405 | 2.805 | RVV quantized vec-dot | 2910.575 | 1.476 |
| `iq2_xs` | RVV quantized vec-dot | 1050.595 | 4.088 | RVV quantized vec-dot | 2374.329 | 1.809 |
| `iq2_xxs` | RVV quantized vec-dot | 1022.966 | 4.199 | RVV quantized vec-dot | 2516.053 | 1.707 |
| `iq3_s` | RVV quantized vec-dot | 3456.974 | 1.242 | RVV quantized vec-dot | 2551.970 | 1.683 |
| `iq3_xxs` | RVV quantized vec-dot | 1717.850 | 2.500 | RVV quantized vec-dot | 2649.375 | 1.621 |
| `iq4_nl` | RVV quantized vec-dot | 617.610 | 6.954 | RVV quantized vec-dot | 1496.994 | 2.869 |
| `iq4_xs` | RVV quantized vec-dot | 728.606 | 5.895 | RVV quantized vec-dot | 2498.640 | 1.719 |
| `tq1_0` | RVV quantized vec-dot | 694.845 | 6.181 | RVV quantized vec-dot | 1242.588 | 3.456 |
| `tq2_0` | RVV quantized vec-dot | 615.096 | 6.983 | RVV quantized vec-dot | 862.583 | 4.979 |
| `mxfp4` | RVV quantized vec-dot | 650.432 | 6.603 | RVV quantized vec-dot | 1634.120 | 2.628 |
| `nvfp4` | scalar vec-dot | 11341.923 | 0.379 | scalar vec-dot | 27330.694 | 0.157 |

### Quantized vec-dot projection helper

这组直接调用 GGML vec-dot helper，shape 为 DeepSeek-R1-Distill-Llama-8B FFN-up decode：`M=1, N=14336, K=4096`。

| kernel | SG2044 implementation | SG2044 ms | SG2044 GOP/s | K1/X60 implementation | K1/X60 ms | K1/X60 GOP/s |
|---|---|---:|---:|---|---:|---:|
| `q1_0_q8_0` | RVV intrinsic | 51.859 | 2.265 | RVV intrinsic | 31.129 | 3.773 |
| `q4_0_q8_0` | RVV intrinsic | 47.504 | 2.472 | RVV intrinsic | 52.794 | 2.225 |
| `q4_1_q8_1` | RVV intrinsic | 46.365 | 2.533 | RVV intrinsic | 52.854 | 2.222 |
| `q5_0_q8_0` | RVV intrinsic | 18.514 | 6.343 | RVV intrinsic | 64.756 | 1.814 |
| `q5_1_q8_1` | RVV intrinsic | 20.121 | 5.837 | RVV intrinsic | 68.025 | 1.726 |
| `q8_0_q8_0` | RVV intrinsic | 212.232 | 0.553 | RVV intrinsic | 55.091 | 2.132 |
| `q2_K_q8_K` | RVV intrinsic | 12.617 | 9.308 | RVV intrinsic | 48.013 | 2.446 |
| `q3_K_q8_K` | RVV intrinsic | 15.894 | 7.389 | RVV intrinsic | 48.107 | 2.441 |
| `q4_K_q8_K` | RVV intrinsic | 13.009 | 9.028 | RVV intrinsic | 45.462 | 2.583 |
| `q5_K_q8_K` | RVV intrinsic | 90.167 | 1.302 | RVV intrinsic | 94.593 | 1.242 |
| `q6_K_q8_K` | RVV intrinsic | 23.991 | 4.895 | RVV intrinsic | 49.253 | 2.384 |
| `iq1_s_q8_K` | RVV intrinsic | 41.876 | 2.804 | RVV intrinsic | 43.243 | 2.716 |
| `iq1_m_q8_K` | RVV intrinsic | 25.578 | 4.591 | RVV intrinsic | 81.768 | 1.436 |
| `iq2_s_q8_K` | RVV intrinsic | 49.069 | 2.393 | RVV intrinsic | 80.397 | 1.461 |
| `iq2_xs_q8_K` | RVV intrinsic | 33.720 | 3.483 | RVV intrinsic | 65.832 | 1.784 |
| `iq2_xxs_q8_K` | RVV intrinsic | 28.986 | 4.052 | RVV intrinsic | 69.737 | 1.684 |
| `iq3_s_q8_K` | RVV intrinsic | 142.783 | 0.823 | RVV intrinsic | 70.562 | 1.664 |
| `iq3_xxs_q8_K` | RVV intrinsic | 76.014 | 1.545 | RVV intrinsic | 73.452 | 1.599 |
| `iq4_nl_q8_0` | RVV intrinsic | 16.924 | 6.939 | RVV intrinsic | 41.497 | 2.830 |
| `iq4_xs_q8_K` | RVV intrinsic | 43.546 | 2.697 | RVV intrinsic | 68.559 | 1.713 |
| `tq1_0_q8_K` | RVV intrinsic | 18.967 | 6.192 | RVV intrinsic | 34.457 | 3.408 |
| `tq2_0_q8_K` | RVV intrinsic | 17.677 | 6.644 | RVV intrinsic | 23.848 | 4.925 |
| `mxfp4_q8_0` | RVV intrinsic | 17.522 | 6.702 | RVV intrinsic | 45.392 | 2.587 |
| `nvfp4_q8_0` | scalar | 310.850 | 0.378 | scalar | 748.241 | 0.157 |

## 2. Activation quantization

Standalone activation quantize helper，shape 为 `M=128, K=14336`。

| kernel | SG2044 implementation | SG2044 ms | SG2044 MElements/s | K1/X60 implementation | K1/X60 ms | K1/X60 MElements/s |
|---|---|---:|---:|---|---:|---:|
| `quantize_row_q8_0` | RVV intrinsic | 3.292 | 557.451 | RVV intrinsic | 5.560 | 330.016 |
| `quantize_row_q8_1` | RVV intrinsic | 4.503 | 407.501 | RVV intrinsic | 6.395 | 286.934 |
| `quantize_row_q8_K` | RVV intrinsic | 3.975 | 461.632 | RVV intrinsic | 5.112 | 358.931 |

## 3. Row dequantization

Standalone row dequantize helper，shape 为 `N=1024, K=4096`。

| kernel | SG2044 implementation | SG2044 ms | SG2044 MElements/s | K1/X60 implementation | K1/X60 ms | K1/X60 MElements/s |
|---|---|---:|---:|---|---:|---:|
| `dequantize_row_q1_0` | scalar | 9.309 | 450.553 | scalar | 28.287 | 148.276 |
| `dequantize_row_q4_0` | scalar | 4.602 | 911.391 | scalar | 24.574 | 170.684 |
| `dequantize_row_q4_1` | scalar | 4.713 | 889.906 | scalar | 28.075 | 149.394 |
| `dequantize_row_q5_0` | scalar | 4.734 | 885.910 | scalar | 26.209 | 160.035 |
| `dequantize_row_q5_1` | scalar | 4.806 | 872.782 | scalar | 28.521 | 147.062 |
| `dequantize_row_q8_0` | scalar | 5.788 | 724.673 | scalar | 22.807 | 183.901 |
| `dequantize_row_mxfp4` | scalar | 24.783 | 169.239 | scalar | 25.102 | 167.090 |
| `dequantize_row_nvfp4` | scalar | 21.732 | 193.004 | scalar | 41.578 | 100.879 |
| `dequantize_row_q2_K` | scalar | 19.897 | 210.802 | scalar | 31.204 | 134.414 |
| `dequantize_row_q3_K` | scalar | 15.998 | 262.184 | scalar | 36.797 | 113.985 |
| `dequantize_row_q4_K` | scalar | 5.837 | 718.606 | scalar | 25.133 | 166.886 |
| `dequantize_row_q5_K` | scalar | 9.509 | 441.095 | scalar | 37.372 | 112.231 |
| `dequantize_row_q6_K` | scalar | 14.151 | 296.390 | scalar | 41.274 | 101.620 |
| `dequantize_row_tq1_0` | scalar | 9.322 | 449.957 | scalar | 37.953 | 110.512 |
| `dequantize_row_tq2_0` | scalar | 5.723 | 732.826 | scalar | 24.113 | 173.940 |
| `dequantize_row_iq2_xxs` | scalar | 10.892 | 385.071 | scalar | 22.497 | 186.441 |
| `dequantize_row_iq2_xs` | scalar | 8.070 | 519.730 | scalar | 25.511 | 164.411 |
| `dequantize_row_iq2_s` | scalar | 10.032 | 418.088 | scalar | 23.349 | 179.633 |
| `dequantize_row_iq3_xxs` | scalar | 13.421 | 312.528 | scalar | 24.100 | 174.039 |
| `dequantize_row_iq3_s` | scalar | 10.283 | 407.880 | scalar | 23.352 | 179.614 |
| `dequantize_row_iq1_s` | scalar | 6.107 | 686.765 | scalar | 32.194 | 130.282 |
| `dequantize_row_iq1_m` | scalar | 5.987 | 700.561 | scalar | 34.176 | 122.728 |
| `dequantize_row_iq4_nl` | scalar | 25.982 | 161.431 | scalar | 27.059 | 155.008 |
| `dequantize_row_iq4_xs` | scalar | 26.202 | 160.073 | scalar | 26.536 | 158.062 |

## 4. Elementwise and activation

| kernel | model shape | SG2044 implementation | SG2044 ms | K1/X60 implementation | K1/X60 ms |
|---|---|---|---:|---|---:|
| `add` | hidden[128,4096] | scalar | 0.799 | SpacemiT RVV intrinsic | 1.556 |
| `sub` | hidden[128,4096] | scalar | 0.797 | SpacemiT RVV intrinsic | 1.569 |
| `mul` | hidden[128,4096] | scalar | 0.798 | SpacemiT RVV intrinsic | 1.595 |
| `div` | hidden[128,4096] | scalar | 0.800 | SpacemiT RVV intrinsic | 1.992 |
| `scale` | hidden[128,4096] | RVV intrinsic | 0.983 | RVV intrinsic | 1.751 |
| `gelu` | ffn[128,14336] | scalar | 5.114 | scalar | 37.761 |
| `silu` | ffn[128,14336] | RVV intrinsic | 4.113 | RVV intrinsic | 13.233 |

## 5. Normalization and reduction

| kernel | model shape | SG2044 implementation | SG2044 ms | K1/X60 implementation | K1/X60 ms |
|---|---|---|---:|---|---:|
| `norm` | hidden[128,4096] | scalar | 3.507 | SpacemiT RVV intrinsic | 1.688 |
| `rms_norm` | hidden[128,4096] | scalar | 2.637 | SpacemiT RVV intrinsic | 1.699 |
| `softmax` | attention[heads=32,Q=128,K=128] | RVV intrinsic | 4.170 | RVV intrinsic | 8.349 |
| `sum_rows` | hidden[128,4096] | scalar | 2.145 | SpacemiT RVV intrinsic | 0.606 |

## 6. Layout, memory, and indexing

| kernel | model shape | SG2044 implementation | SG2044 ms | K1/X60 implementation | K1/X60 ms |
|---|---|---|---:|---|---:|
| `cpy` | hidden[128,4096] | scalar/memcpy | 0.726 | scalar/memcpy | 1.030 |
| `cont` | transpose(hidden[128,4096]) | scalar/memcpy | 5.574 | RVV inline asm | 4.251 |
| `get_rows` | embedding[vocab=128256,hidden=4096],tokens=128 | scalar | 0.825 | scalar | 3.362 |
| `get_rows_f32` | TinyLlama embedding[vocab=32000,hidden=2048],tokens=128 | — | — | RVV inline asm | 0.691 |
| `repeat` | norm_weight[4096]->hidden[128,4096] | scalar | 0.747 | SpacemiT RVV intrinsic | 0.365 |
| `concat` | hidden[64+64,4096] | scalar | 1.008 | scalar | 2.018 |
| `concat_dim0` | hidden[128,2048+2048] | — | — | SpacemiT scalar | 3.924 |

## 7. Attention and position

| kernel | model shape | SG2044 implementation | SG2044 ms | K1/X60 implementation | K1/X60 ms |
|---|---|---|---:|---|---:|
| `rope` | q[head_dim=128,heads=32,tokens=128] | scalar | 1.253 | scalar | 5.237 |
| `flash_attn` | Q=128,KV=128,H=32,Hkv=8,D=128 | generic mixed | 21.667 | generic mixed | 44.422 |
