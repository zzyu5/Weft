# RISC-V LLM kernel source corpus

## 结论

原先按十个 Weft repro 人工摘录的 `source/c/ggml` 已删除。它把 generic C 循环写成新的
standalone 函数，既不是 GGML 手写 RISC-V source，也不能用于性能对比。

现行 corpus 改为两种来源：

```text
source/c/
├── ggml/
│   └── llama.cpp/ggml/       原样的 GGML CPU、arch/riscv 与 Spacemit source
└── tianchenrv/
    └── rvv/                  materials 中按 target 选定的最终高性能 kernel source
```

`examples/run/source.sh` 随旧摘录一并删除。新的 baseline/generated 性能 runner 必须直接
调用这里的真实 entry，并在同一输入、ABI、layout、编译器与计时器中比较；本次整理不保留
旧 runner 作为兼容入口。

## GGML RISC-V 覆盖

GGML 原始 source 现在完整保留以下 LLM inference 路径，不再局限于六个 block-dot：

| 族 | 覆盖 | 选定 source |
|---|---|---|
| quantize | `q8_0`、`q8_1`、`q8_K` | `arch/riscv/quants.c` 的 RVV entry |
| vec-dot | 24 个格式族 | 23 个格式使用 `arch/riscv/quants.c` 的 RVV/VLEN-specific entry；`nvfp4` 使用 generic scalar entry |
| repack | q8_0 4×8 activation repack | `arch/riscv/repack.cpp` |
| quantized GEMV | q4_0 8×8/16×1、q4_K、q2_K、q8_0、IQ4_NL | `arch/riscv/repack.cpp` |
| quantized GEMM | q4_0 8×8/16×1、q4_K、q2_K、q8_0、IQ4_NL | `arch/riscv/repack.cpp` |
| dequantize | 25 个 row-dequant 格式 | `src/ggml-quants.c`；没有专用 RVV 的格式保留实际 scalar/autovec 路径 |
| forward math | add/sub/mul/div、scale、copy、GELU、SiLU、softmax、RoPE、Norm、RMSNorm 等 | `vec.*`、`binary-ops.cpp`、`unary-ops.cpp`、`ops.cpp` |
| Spacemit RVV | FlashAttention、Norm、RMSNorm、binary、permuted copy/cont、repeat、sum-rows、get-rows、concat | `spacemit/rvv_kernels.cpp` 与 `ime.cpp` |
| Spacemit IME | q4_0、q4_K、q8_0 quantized GEMM；IME1/IME2 | `spacemit/ime1_kernels.cpp`、`ime2_kernels.cpp`、`repack.cpp` |

vec-dot 的 24 个格式族为：

```text
q1_0 q2_K q3_K q4_0 q4_1 q4_K q5_0 q5_1 q5_K q6_K q8_0
iq1_m iq1_s iq2_s iq2_xs iq2_xxs iq3_s iq3_xxs iq4_nl iq4_xs
mxfp4 nvfp4 tq1_0 tq2_0
```

这不是 24 份摘录文件。上游在一个 translation unit 内共享 block ABI、lookup tables、static
helper 与 VLEN dispatch；保留完整原文件才能忠实保存真实实现。具体 wrapper 根据
VLEN128/VLEN256/VLEN512/VLEN1024、Zvfh 或 XTheadVector 选择专用 body。

## materials 中保留的最终 source

`materials/` 中约百个“kernel 单元”实际混合了语义算子、dtype/layout、目标板、schedule、
失败候选、driver 和 generated artifact。整理后只留下 26 个 kernel source：

| 目录 | 数量 | 唯一选择 |
|---|---:|---|
| `rvv/forward/vlen256/` + `rvv/forward/vla/` | 4 | VLEN256 add/mul/copy；VLA RMSNorm |
| `rvv/gemm/flat/vlen128/` | 3 | q4_1、q5_0、q5_1 packed GEMM |
| `rvv/gemv/vlen256/` | 2 | q5_0、q5_1 deployed GEVM |
| `rvv/gemm/kquant/vlen128/` | 5 | q2_K/q4_K/q5_K tiled；q3_K/q6_K plain |
| `rvv/gemm/kquant/vlen256/` | 5 | q2_K/q4_K/q6_K unrolled；q3_K/q5_K rolled |
| `rvv/gemm/iq/vlen256/` | 7 | IQ1_M/IQ1_S、IQ2_S/IQ2_XS/IQ2_XXS、IQ3_S/IQ3_XXS grid4 |

这些是按相同实验族内部的最终选择做的去重。没有迁入：

- 旧 scalar/RVV/IME selector 与 whole-kernel route；
- 被更快 schedule 替代的同 ABI 候选；
- `raw/stock_*` 对手副本、objdump、日志、CSV、driver、fixture 与 MLIR；
- 只有 cross-op 数字、layout 不同或数值错误的所谓 winner；
- materials 中复制自 GGML 手写 vec-dot 的版本——GGML 原文件已经是唯一 source。

## 选择边界

“最高性能”按 `语义 + dtype + block/packed layout + decode/prefill phase + target capability`
定义，而不是只按 kernel 名定义。VLEN128、VLEN256 和 IME 是不同 target 域；同一语义在这些
域中允许各保留一个 realization。不能用 VLEN256 的 source 替代 VLEN128，也不能把 GEMM
对 block-dot 的 cross-op 数字当作同 ABI 胜负。

GGML source 是外部 as-shipped baseline。`source/c/tianchenrv` 是历史优化供体，不能冒充
外部 baseline，也不进入现行 compiler。后续性能对比的对象应是“新 Weft generated artifact
vs 真实 GGML entry”；历史 source 只帮助确认已经探索过的高性能结构与 target-specific
realization。

## 当前 artifact 边界

本次完成的是 source corpus 整理与旧路径删除。GGML 文件保持原始相对 include 结构；现有
十个 Weft 手工数值 repro 的 runtime 已移到 `examples/repro/weft/`，不再从 `source/` 借用
算法或 reference symbol。

完整 corpus 的统一 runtime、同 harness 性能计时与逐 entry `ssh rvv` 执行尚未建立，因此
这里不报告新的性能数值，也不把 materials 的历史数据写成当前结果。
