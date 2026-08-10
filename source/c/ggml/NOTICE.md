# GGML source corpus

本目录保存从 [ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp) 抽取的、可独立执行
的 CPU kernel source。代码遵循同目录 [`LICENSE`](LICENSE) 的 MIT 许可。

来源边界：

- `elementwise/add/`：`ggml/src/ggml-cpu/binary-ops.cpp` 的 F32 add/broadcast 内核；
- `normalization/rms_norm/`：`ggml/src/ggml-cpu/ops.cpp` 的 RMSNorm + MUL 数值循环；
- `normalization/online_softmax/`：`ops.cpp` flash-attention 中的 online maximum/sum 更新；
- `gemm/blocked/`：`ggml/src/ggml-cpu/ggml-cpu.c` 的 16×16 mul-mat block traversal；
- `quantization/block_dot/`：`ggml-common.h`、`ggml-cpu/quants.c` 与 `ggml-quants.c`
  的 q4/q5/q8/K-quant block ABI、decode 与 generic dot 语义。

每个 source 的相邻 `*_runtime` 只提供固定输入、调用和数值对照，不复制 kernel 算法。
Tensor graph、threadpool、dispatcher、llamafile 与其他可选依赖没有进入这个独立 corpus。
