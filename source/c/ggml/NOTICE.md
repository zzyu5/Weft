# GGML RISC-V source corpus

`llama.cpp/ggml/` 按上游相对路径保存 GGML CPU 中与 RISC-V LLM inference 有关的原始
source。文件内容不做 standalone 改写，不把函数摘录后改名，也不删除上游内部的 target
dispatch。许可见同目录 `LICENSE`。

保留范围：

- `src/ggml-cpu/arch/riscv/`：RVV quantize、24 族 quantized vec-dot、repack、GEMV、GEMM
  与 RISC-V feature probe；
- `src/ggml-cpu/spacemit/`：Spacemit RVV forward kernels、FlashAttention、IME1/IME2
  quantized GEMM、repack 与 target glue；
- `src/ggml-cpu/` 顶层文件：没有专用 RISC-V 实现时实际执行的 generic scalar/autovec
  operator，以及 RISC-V source 所依赖的 traits、block ABI 和 common lowering helpers；
- `src/ggml-quants.c`：row dequantize 与 reference quantization；
- `include/` 与相邻顶层 source：保持上述文件的 GGML ABI 和 include 关系。

这里不保存第二份“适配后的 GGML kernel”。一个 kernel 的上游 RVV、VLEN-specific、Zvfh、
IME 或 generic scalar realization 由原始 source 中的 target 条件确定。benchmark runner 必须
在调用前固定 target capability；不能把上游 generic branch 当成 Weft 的隐式 fallback。

历史 TianchenRV 实验中实测更优、且不属于上游 GGML 的 source 单独放在
`../tianchenrv/`，二者不能混作同一个 baseline。
