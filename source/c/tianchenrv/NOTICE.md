# TianchenRV 选定 RISC-V kernel source

本目录只保存从 `materials/experiments/` 抽取的、在对应 target 域中最终被选中的 kernel
source。函数体保持历史最终 artifact 原样；driver、raw log、MLIR、selector、旧 route、失败
候选和被更快版本替代的 schedule 不进入这里。

当前选定内容：

- `rvv/forward/vlen256/`：VLEN256 的 add、mul、copy；`rvv/forward/vla/` 保存
  VLEN128/VLEN256 都成立的一遍式 RMSNorm；
- `rvv/gemm/flat/vlen128/`：GGML 当前 RISC-V repack 文件没有覆盖的 q4_1、q5_0、q5_1
  packed GEMM；
- `rvv/gemv/vlen256/`：q5_0、q5_1 decode 的最终 deployed GEVM；
- `rvv/gemm/kquant/vlen128/`：q2_K/q4_K/q5_K 的 tiled realization 与 q3_K/q6_K 的
  plain realization；
- `rvv/gemm/kquant/vlen256/`：q2_K/q4_K/q6_K unrolled、q3_K/q5_K rolled；
- `rvv/gemm/iq/vlen256/`：IQ1/IQ2/IQ3 的 grid4 packed GEMM realization。

这些文件是优化知识与后续 DSL 对照 source，不是现行 compiler 的输入、provider、fallback，
也不是外部 GGML baseline。历史测量只用于从同一实验族中删除更慢变体；当前性能结论必须
由新的统一 repro 重新测得。若新的同 target、同 ABI、同 layout 测量改变 winner，就直接替换
这里的文件，不并存旧实现。
