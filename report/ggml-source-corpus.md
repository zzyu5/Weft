# GGML source corpus 与 RVV 执行结果

日期：2026-08-10

## 边界

`source/` 现在只保存从 `ggml-org/llama.cpp` 的 GGML CPU 实现抽取出来、能够独立编译的
C/C++ kernel source；项目自身的 Python DSL kernel 位于 `examples/kernels/`，不再混入
upstream source corpus。

每个 GGML source 旁边只有一个同名 `*_runtime`。runtime 负责构造固定输入、调用 source
并进行数值对照，不复制 kernel 算法。统一入口 `examples/run/source.sh` 只做显式的
kernel-to-source 映射、传输、远端编译和执行；未知名称直接报错。

这份结果只证明 GGML source corpus 能够在真实 RISC-V 主机执行，不代表 Weft 已经从 DSL
lower 出这些实现。

## 十个 baseline

| Kernel | GGML source | 执行命令 | `ssh rvv` 结果 |
| --- | --- | --- | --- |
| `add_bias` | `source/c/ggml/elementwise/add/add.cpp` | `./examples/run/source.sh add_bias` | `PASS ggml_add_f32` |
| `rms_norm` | `source/c/ggml/normalization/rms_norm/rms_norm.cpp` | `./examples/run/source.sh rms_norm` | `PASS ggml_rms_norm_f32` |
| `online_softmax` | `source/c/ggml/normalization/online_softmax/online_softmax.cpp` | `./examples/run/source.sh online_softmax` | `PASS ggml_online_softmax_f32` |
| `blocked_gemm` | `source/c/ggml/gemm/blocked/mul_mat.cpp` | `./examples/run/source.sh blocked_gemm` | `PASS ggml_mul_mat_f16_f16` |
| `q4_0_q8_0` | `source/c/ggml/quantization/block_dot/q4_0_q8_0/q4_0_q8_0.c` | `./examples/run/source.sh q4_0_q8_0` | `PASS q4_0_q8_0_block_dot` |
| `q4_1_q8_1` | `source/c/ggml/quantization/block_dot/q4_1_q8_1/q4_1_q8_1.c` | `./examples/run/source.sh q4_1_q8_1` | `PASS q4_1_q8_1_block_dot` |
| `q5_0_q8_0` | `source/c/ggml/quantization/block_dot/q5_0_q8_0/q5_0_q8_0.c` | `./examples/run/source.sh q5_0_q8_0` | `PASS q5_0_q8_0_block_dot` |
| `q5_1_q8_1` | `source/c/ggml/quantization/block_dot/q5_1_q8_1/q5_1_q8_1.c` | `./examples/run/source.sh q5_1_q8_1` | `PASS q5_1_q8_1_block_dot` |
| `q8_0_q8_0` | `source/c/ggml/quantization/block_dot/q8_0_q8_0/q8_0_q8_0.c` | `./examples/run/source.sh q8_0_q8_0` | `PASS q8_0_q8_0_block_dot` |
| `q4_K_q8_K` | `source/c/ggml/quantization/block_dot/q4_K_q8_K/q4_K_q8_K.c` | `./examples/run/source.sh q4_K_q8_K` | `PASS q4_K_q8_K_block_dot` |

## 实际执行方式

runner 将 `source/c/ggml/` 复制到 `ssh rvv` 上独立创建的 `/tmp/ggml-source.*` 目录。
四个 C++ kernel 使用 `g++ -O2 -std=c++17 -Wall -Wextra -Werror`，六个 quant kernel
使用 `clang -O2 -std=c11 -Wall -Wextra -Werror`；执行结束后删除该次临时目录。

本次十项均在 `riscv64` 主机逐项重新编译并执行成功，远端没有残留此次 runner 创建的
临时目录。
