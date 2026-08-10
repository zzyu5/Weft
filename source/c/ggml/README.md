# GGML-derived C source

本目录保存六个量化 block-dot baseline 所需的最小、独立 C reference。代码与 block ABI
来自 [ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp)，遵循同目录 `LICENSE` 的
MIT 许可。

抽取依据是上游仓库中的：

- `ggml/src/ggml-common.h`：q4/q5/q8 与 K-quant block layout；
- `ggml/src/ggml-cpu/quants.c`：通用标量 block-dot 语义；
- `ggml/src/ggml-quants.c`：解包语义，用于相邻 runtime 的数值对照。

这里没有导入 GGML runtime、tensor/graph 层、CPU dispatcher 或 RISC-V intrinsic 文件。
每个 `reference.c` 只保留一个格式对的算法；签名被缩成“typed blocks + block count”，但
packed bytes、scale/min 公式和 block 粒度保持 GGML 语义。相邻 `runtime.c` 是手工 repro，
不构成新的测试框架。

这些 C 文件是 baseline 与语义 donor，不是 Weft provider。后续 RVV provider 必须按
canonical primitive 组合实现，不能调用这些 reference，也不能按目录名注册整 kernel
route。
