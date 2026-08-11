# GGML RISC-V Kernel Baseline

本文件是后续 Weft 单 kernel 性能对照的 GGML baseline。这里比较的是模型级 shape 下的
单次 kernel graph invocation；完整模型 `llama-bench` 是独立的 e2e baseline，二者不能
共用分母。

## 结论与计数口径

GGML 的 RISC-V 路径不只有普通 RVV intrinsic。源码中同时存在标准 RVV intrinsic、
SpacemiT 手写 RVV inline asm、XTheadVector inline asm、IME1 custom-instruction asm 和
IME2 custom-instruction asm。它们依赖不同 ISA 与硬件，不能揉成一个脱离 target 的总数。

当前可复现的 runtime baseline 分成三个 target profile：

| target profile | 可运行 production entry | 实测 case | 当前结果 |
|---|---:|---:|---|
| SG2044，标准 RVV VLEN128 | 69 | 69 | 29 RVV intrinsic，40 scalar/library/generic，69/69 运行通过 |
| K1/X60，SpacemiT RVV VLEN256 | 11 | 11 | 8 RVV intrinsic，2 手写 RVV inline asm，1 scalar，11/11 运行通过 |
| K1/X60，IME1 | 3 个量化 weight-format entry | 6 个 decode/prefill case | production repack + activation quantize + IME1 GEMM，6/6 运行通过 |

这三个 profile **不能相加成一个“83 个 kernel”**：第二组与第一组有逻辑算子重叠，第三组
又是在不同 target 上对三个量化格式各测 decode/prefill。后续 Weft 对表时必须先选择 target
profile，再在相同 shape、layout、线程数和 cold-memory 口径下比较。

另外两组源码路径已经纳入清单，但当前没有可声称成功的性能数字：

- XTheadVector：4 个量化 vec-dot alternate entry；当前 SG2044 build 选择标准 RVV，K1
  runner 也不是 XTheadVector build。
- IME2：8 个 active optimized dispatch family；目前可访问的 K1/X60 只报告
  `use_ime1=1,use_ime2=0`，没有 IME2 core/build，因此不伪造 cold time。

## 手写汇编清单

### SpacemiT RVV inline asm

`source/c/ggml/llama.cpp/ggml/src/ggml-cpu/spacemit/rvv_kernels.cpp` 中有三个手写
RVV helper、五个 `__asm__ volatile` block：

- `rvv_transposed_s32_mn_to_nm`：e32 transpose；当前由 `cont` runtime 真实触发。
- `rvv_transposed_s16_mn_to_nm`：e16 transpose；源码存在，但当前 K1 VLEN256 profile 没有
  一条无缺陷的 production F16 graph 可以作为 baseline。
- `memcpy1d`：按 VLEN 选择的手写 vector copy；当前由 F32 `get_rows` runtime 真实触发。

因此，K1 profile 中可运行并计时的手写 RVV GGML entry 是 **2 个**：`cont` 和
`get_rows_f32`。这不是“源码里有汇编但 benchmark 实际没走”的静态计数。

### XTheadVector inline asm

`source/c/ggml/llama.cpp/ggml/src/ggml-cpu/arch/riscv/quants.c` 中有四个
`__riscv_xtheadvector` alternate entry：

- `ggml_vec_dot_q2_K_q8_K_xtheadvector`
- `ggml_vec_dot_q3_K_q8_K_xtheadvector`
- `ggml_vec_dot_q4_K_q8_K_xtheadvector`
- `ggml_vec_dot_q6_K_q8_K_xtheadvector`

它们使用 `th.*` inline asm。SG2044 的 69 项 profile 对应标准 RVV build，这四项实际走
RVV intrinsic，而不是这些 alternate body。

### IME1 asm

`source/c/ggml/llama.cpp/ggml/src/ggml-cpu/spacemit/ime1_kernels.cpp` 有六个 inline-asm
block，production path 对外体现为三个手写 asm family：

- `ime1::quantize_a_row_i8`
- `ime1::quantize_a_4row_i8`
- `ime1::gemm_kernel_i8i4`，内部按 M=1/M=4 与 zero-point 形式选择实现

GGML 对 `Q4_0/Q4_1/Q4_K` 分别建立 repack trait；这三个格式都复用 i8×i4 IME1 GEMM
family，而不是三个互不相关的手写 kernel。decode 会覆盖单行 activation quantizer，prefill
会覆盖四行 quantizer。

### IME2 asm

`source/c/ggml/llama.cpp/ggml/src/ggml-cpu/spacemit/ime2_kernels.cpp` 有 27 个 asm block。
当前 wrapper 实际选择 optimized body 的八个 family 是：

- `gemm_kernel_i8i2k`
- `gemm_kernel_i8i3k`
- `gemm_kernel_i8i4`
- `gemm_kernel_i8i4_hp`
- `gemm_kernel_i8i8`
- `gemm_kernel_i8i5`
- `moe_m2_gemm_kernel_i8i4`
- `moe_m2_gemm_kernel_i8i5`

`gemm_kernel_i8mxfp4` 当前 wrapper 明确选择 reference body；
`moe_m2_gemm_kernel_i8mxfp4` 的 optimized 调用被注释，函数直接返回。因此二者不能算作
active IME2 asm baseline。

## Cold-memory 口径

这里的“冷启动”是 **cache-cold 的稳定 kernel invocation**，不是进程首次调用、动态库
加载、SSH、编译、模型载入、weight repack 或 codegen 时间。

共同口径：

- runtime 和动态链接在计时前完成；
- 每个 case 先完整 warmup 一次；
- 每次计时前读写 eviction buffer，eviction 本身不计时；
- 每项运行 10 次，表中为 wall-time median；
- 输入使用真实模型级 shape，不使用 toy matrix；
- 单线程并固定到一个目标 core。

目标差异：

| profile | host/core | VLEN | eviction | build |
|---|---|---:|---:|---|
| 标准 RVV | `ssh rvv` / CPU 8 | 128 bit | 64 MiB | SG2044 `rv64gcv` GGML CPU build |
| SpacemiT RVV | `ssh k1` / CPU 3 | 256 bit | 64 MiB | X60 SpacemiT GGML build |
| IME1 | `ssh k1` / CPU 3 | 256 bit | 32 MiB | X60 IME1-enabled SpacemiT GGML build |

K1 的 IME 只存在于 core 0--3，因此固定 core 3。当前机器没有 `/dev/tcm_sync_mem`，vendor
runtime 报告 TCM allocation 不可用并回退 heap；下面的 IME1 数字仍是真实 IME1 指令执行，
但不是 TCM-enabled 结果。

## K1/X60 SpacemiT RVV

这 11 个 entry 通过 GGML graph 和 SpacemiT `tensor_traits_common` dispatch，而不是直接调用
helper。所有 case 均在 VLEN256、core 3、单线程、64 MiB eviction、10 次 median 下运行成功。

| kernel | 实际实现 | 模型级 shape | cold median (ms) |
|---|---|---|---:|
| `norm` | SpacemiT RVV intrinsic | hidden `[128,4096]` | 1.703 |
| `rms_norm` | SpacemiT RVV intrinsic | hidden `[128,4096]` | 1.768 |
| `add` | SpacemiT RVV intrinsic | hidden `[128,4096]` | 1.578 |
| `sub` | SpacemiT RVV intrinsic | hidden `[128,4096]` | 1.533 |
| `mul` | SpacemiT RVV intrinsic | hidden `[128,4096]` | 1.559 |
| `div` | SpacemiT RVV intrinsic | hidden `[128,4096]` | 2.019 |
| `repeat` | SpacemiT RVV intrinsic | norm `[4096]` -> hidden `[128,4096]` | 0.352 |
| `sum_rows` | SpacemiT RVV intrinsic | hidden `[128,4096]` | 0.598 |
| `cont` | **handwritten RVV e32 transpose asm** | transpose of hidden `[128,4096]` | 4.307 |
| `get_rows_f32` | **handwritten RVV copy asm** | TinyLlama vocab=32000, hidden=2048, tokens=128 | 0.682 |
| `concat_dim0` | SpacemiT scalar target body | hidden `[128,2048+2048]` | 3.963 |

SpacemiT source 中还有 VLEN1024-only FlashAttention target；K1 是 VLEN256，production dispatch
会落到 generic path，所以它不计入这 11 项 target baseline。

### 未纳入可运行集合的 vendor defect

两条源码路径存在，但不能写成成功 baseline：

- `forward_cpy_with_permute` 把 `n_dst_stride` 计算成 `n_src_stride * m`，而同构的 CONT
  路径使用正确的 `dst->nb[1]`。模型级 F32 transpose-copy 会写出目标 buffer 并以 exit 255
  结束。
- `permute_transpose_impl` 的 `sizeof(int16_t)` 分支错误调用
  `rvv_transposed_s32_mn_to_nm`，会用 e32 load/store 处理 F16。此前一次“能返回有限 float”
  不能证明正确，因为两个 half 被当作一个 float 读取。

runner 不再暴露这两个失败 case。修复 vendor backend 之前，它们只属于 source defect，不属于
可运行性能基线。

## K1/X60 IME1

IME1 runtime 没有直接调用 `ime1_kernels.cpp` 的函数。它使用
`ggml_backend_cpu_riscv64_spacemit_buffer_type()` 分配 weight，通过 `set_tensor` 执行 vendor
repack，要求 `weight->extra` 非空，再建立单一 `GGML_OP_MUL_MAT` graph。当前 build 只启用
IME1，runtime banner 为 `use_ime1=1,use_ime2=0`；因此 trait 选择会进入 IME1 activation
quantizer 与 `gemm_kernel_i8i4`。

统一权重 shape 来自 DeepSeek-R1-Distill-Llama-8B FFN-up：`N=14336, K=4096`。decode
使用 `M=1`，prefill 使用 `M=128`。计时范围是 production activation quantize + IME1 GEMM，
不含 weight repack。

| weight format | phase | M | cold median (ms) | GOP/s |
|---|---|---:|---:|---:|
| `Q4_0` | decode | 1 | 9.666 | 12.150 |
| `Q4_0` | prefill | 128 | 527.101 | 28.519 |
| `Q4_1` | decode | 1 | 11.499 | 10.214 |
| `Q4_1` | prefill | 128 | 603.625 | 24.904 |
| `Q4_K` | decode | 1 | 11.450 | 10.257 |
| `Q4_K` | prefill | 128 | 603.943 | 24.890 |

6/6 case 均完成 dynamic link、repack、warmup、10 次 cold invocation，并返回有限 F32 输出。

## SG2044 标准 RVV profile

这一节只描述 `ssh rvv` 的标准 RVV VLEN128 build，不能代表整个 GGML RISC-V source。
按 production primitive entry 去重，共 **69 个可运行 kernel**：

| family | 数量 | RVV intrinsic | 标量/库/generic | 手写汇编 |
|---|---:|---:|---:|---:|
| quantized vec-dot | 24 | 23 | 1 | 0 |
| activation row quantize | 3 | 3 | 0 | 0 |
| row dequantize | 24 | 0 | 24 | 0 |
| LLM forward | 18 | 3 | 15 | 0 |
| **总计** | **69** | **29** | **40** | **0** |

69/69 均完成实际动态链接、真实调用并返回有限输出。IQ1/IQ2/IQ3 没有普通 row quantizer，
runtime 使用合法的全零 quantized block；它会执行同一 RVV entry 与完整矩阵调用次数，但不
代表真实 GGUF 数值分布。

### Quantized vec-dot

统一 shape：DeepSeek-R1-Distill-Llama-8B FFN-up decode，`M=1, N=14336, K=4096`。

| kernel | 实现 | cold median (ms) | GOP/s |
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

### Activation quantize

统一 shape：DeepSeek 8B FFN-down prefill activation，`M=128, K=14336`。

| kernel | 实现 | cold median (ms) | MElements/s |
|---|---|---:|---:|
| `quantize_row_q8_0` | RVV intrinsic | 3.294 | 557.016 |
| `quantize_row_q8_1` | RVV intrinsic | 4.500 | 407.795 |
| `quantize_row_q8_K` | RVV intrinsic | 3.969 | 462.372 |

### Row dequantize

统一 shape：DeepSeek 8B attention-K tensor，`N=1024, K=4096`。24 项均是当前
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

`q8_1` 没有 production dequant entry；`q8_K` helper 没有挂到当前 production
`to_float` trait，二者不计入这 24 项。

### LLM forward

| kernel | 实现 | 模型级 shape | cold median (ms) |
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
| `repeat` | scalar | norm `[4096]` -> hidden `[128,4096]` | 0.741 |
| `sum_rows` | scalar | hidden `[128,4096]` | 2.144 |
| `concat` | scalar | hidden `[64+64,4096]` | 1.013 |
| `flash_attn` | generic mixed | Q=128, KV=128, H=32, Hkv=8, D=128 | 18.196 |

## 运行入口

每次只运行一个明确 kernel：

```bash
./examples/run/ggml-kernel.sh \
  <vec_dot|quantize|dequantize|forward|spacemit_rvv|ime1> \
  <kernel> \
  10
```

SpacemiT RVV 的当前可运行 kernel 参数：

```text
norm rms_norm add sub mul div repeat sum_rows cont get_rows_f32 concat_dim0
```

IME1 的 kernel 参数：

```text
q4_0_decode q4_0_prefill
q4_1_decode q4_1_prefill
q4_K_decode q4_K_prefill
```

runtime 文件：

```text
examples/repro/ggml/vec_dot_runtime.cpp
examples/repro/ggml/quantize_runtime.cpp
examples/repro/ggml/dequantize_runtime.cpp
examples/repro/ggml/forward_runtime.cpp
examples/repro/ggml/ime1_runtime.cpp
```

runner 只负责选择目标 GGML build、上传对应 runtime、固定 CPU 并执行。IME2 没有 runner，
因为当前没有可访问的 IME2 target；未实现就保持不可运行，不以 IME1、普通 RVV 或 reference
body 兜底冒充 IME2。
