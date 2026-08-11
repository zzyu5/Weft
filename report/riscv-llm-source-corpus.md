# GGML RISC-V LLM source corpus

## 结论

`source/` 现在只包含 llama.cpp 的真实 GGML CPU/RISC-V source。上一轮误放入的
`source/c/tianchenrv/` 已整棵删除；历史 TianchenRV 生成 kernel 只用于定位真实 GGML
opponent，不再作为 source、baseline 或 donor 副本存在。

旧的十项 standalone 摘录、scalar block-dot、副本 runtime 和 `examples/run/source.sh` 也已
删除。所有空目录已清理。

## 最终目录

```text
source/c/ggml/
├── LICENSE
├── NOTICE.md
└── llama.cpp/ggml/
    ├── include/                  GGML CPU 必需 public ABI
    └── src/
        ├── ggml.c / common / quants / threading
        └── ggml-cpu/
            ├── generic CPU op、traits、vec、quants、repack
            ├── arch/riscv/       真实 RVV quantize、vec-dot、repack、GEMV、GEMM
            └── spacemit/         真实 RVV forward、FlashAttention、IME1/IME2
```

共 51 个 llama.cpp source/header 文件。没有其他架构 backend、backend loader、optimizer、
GGUF 工具、build artifact、generated source 或空目录。保留文件与 `ssh rvv` 上实际用于
`llama-bench` 的 upstream-derived checkout 逐文件比较一致；远端 checkout 里的实验 CMake
修改没有进入 corpus。

## 量化路径

真实 production path 是：

```text
GGUF tensor type
  → ggml-cpu.c::type_traits_cpu
  → GGML_OP_MUL_MAT / GGML_OP_MUL_MAT_ID
  → target packed GEMV/GEMM（存在且 target 合法时）
  → generic mul-mat traversal + format-specific ggml_vec_dot_*
```

### RISC-V vec-dot

`arch/riscv/quants.c` 包含 23 个 RISC-V entry：

```text
q1_0
q4_0 q4_1 q5_0 q5_1 q8_0
q2_K q3_K q4_K q5_K q6_K
iq1_s iq1_m iq2_s iq2_xs iq2_xxs iq3_s iq3_xxs iq4_nl iq4_xs
tq1_0 tq2_0 mxfp4
```

它们使用直接 RVV intrinsic、VLEN-specific body 或 XTheadVector 实现。`nvfp4` 是当前唯一
没有 RISC-V 专用 entry 的已登记格式，真实路径位于 `ggml-cpu/quants.c` 的 generic scalar
implementation；没有为它伪造 RVV source。

### Quantize、dequantize 与 packed matmul

- RISC-V activation quantize：`q8_0`、`q8_1`、`q8_K`；
- row-dequant/reference quantize：`ggml-quants.c` 中 GGML 实际登记的全部格式；
- RISC-V packed GEMV/GEMM：q4_0 8×8/16×1、q4_K 16×1、q2_K 16×1、IQ4_NL
  16×1、q8_0 16×1；
- q8_0 4×8 activation repack。

q4_1、q5_0、q5_1、q3_K、q5_K、q6_K 以及多数 IQ/TQ 格式当前没有同名 RISC-V
`ggml_gemm_*` entry。它们在完整模型中的真实 opponent 是 generic mul-mat traversal 加对应
vec-dot，不能根据 TianchenRV 历史 kernel 名虚构一个 GGML GEMM。

## 非量化路径

保留的 generic/target source 覆盖：add/sub/mul/div、scale、copy、GELU、SiLU、softmax、
RoPE、Norm、RMSNorm、FlashAttention、repeat、sum-rows、get-rows、concat，以及 Spacemit
IME1/IME2 quantized GEMM。没有专用 RVV 的算子保留 llama.cpp 实际 scalar/autovec 路径，
而不是为了目录整齐写一个假的 intrinsic kernel。

## 模型级执行

统一入口：

```bash
./examples/run/ggml.sh \
  <remote-model.gguf> \
  <prompt-tokens> \
  <generation-tokens> \
  <threads> \
  <repetitions>
```

它在 `ssh rvv` 上直接运行真实 llama.cpp `llama-bench`，固定 `n_batch=2048`、
`n_ubatch=512`，同时得到 prefill 与 decode。runner 不创建小数组，不复制 kernel，不读取
TianchenRV 产物。

### 标准模型级结果

真实 TinyLlama Q4_0 模型：1,100,048,384 参数，模型数据 635,990,016 bytes；4 threads，
`pp128/tg32`，各 3 次：

| Phase | Tokens | 平均吞吐 |
|---|---:|---:|
| prompt/prefill | 128 | 5.244427 token/s |
| generation/decode | 32 | 4.781877 token/s |

这条命令实际执行成功：

```bash
./examples/run/ggml.sh \
  /home/ubuntu/tcrv-llamacpp/models/tinyllama-q4_0.gguf \
  128 32 4 3
```

### 8B K-quant 与格式覆盖

DeepSeek-R1-Distill-Llama 8B Q4_K_M 模型具有 8,030,261,312 参数。真实 GGUF metadata 为
`n_embd=4096`、`n_ff=14336`、32 layers；292 个 tensor 中包含 193 个 Q4_K、33 个 Q6_K
和 66 个 F32 tensor。`pp8/tg2` 完整模型运行通过，分别为 2.426912 与 1.078821 token/s。

远端现有真实模型还逐一完成了最短完整 dispatch：

```text
DeepSeek 8B Q2_K
DeepSeek 8B Q4_1
DeepSeek 8B Q5_0
DeepSeek 8B Q5_1
DeepSeek 8B Q5_K_M
DeepSeek 8B Q6_K
DeepSeek 8B IQ4_NL
TinyLlama 1.1B Q8_0
```

每个模型都完成 prompt 与 generation 两个 phase；这些一次运行只证明 model load、graph
dispatch 与对应 quant source 路径可执行，不作为性能 headline。

## 边界

`llama-bench` 给出完整 GGUF 模型基线。未来 Weft generated intrinsic C 与 GGML 的直接
kernel ratio 必须另用同一真实 tensor、shape、layout 和计时器；完整模型 token/s 与单 kernel
时间不能混成同一个分母。现有 `examples/repro/weft/` 中的小型数值 repro 仍只证明 compiler
artifact 可执行，不再被称为 GGML source baseline。
