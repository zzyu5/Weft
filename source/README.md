# Weft source corpus

`source/` 是端到端编译使用的输入语料库，不是测试目录，也不参与根 CMake 的默认构建。
目录按 source language、upstream/project、operator 和 variant 组织；同一变体的 source 与
runtime 相邻。

```text
source/
├── weft/weft/<operator>/<variant>/
└── c/
    ├── weft/<operator>/<variant>/
    └── ggml/quantization/block_dot/<format-pair>/
```

## 固定 baseline

| # | Baseline | 主要覆盖 |
|---:|---|---|
| 1 | `add_bias_f32` | VLA、load/store、pointwise |
| 2 | `rms_norm_f32` | reduction、两遍遍历、math |
| 3 | `online_softmax_f32` | summary algebra、两遍遍历 |
| 4 | `blocked_gemm_f16_f32` | logical block、contract |
| 5 | `q4_0_q8_0_block_dot` | signed nibble decode、scale |
| 6 | `q4_1_q8_1_block_dot` | affine nibble decode、min fold |
| 7 | `q5_0_q8_0_block_dot` | split high-bit decode、scale |
| 8 | `q5_1_q8_1_block_dot` | affine 5-bit decode、min fold |
| 9 | `q8_0_q8_0_block_dot` | int8 dot、block scale |
| 10 | `q4_K_q8_K_block_dot` | 256-wide super-block scale/min |

GGML-derived source 的许可和抽取边界见 [`c/ggml/README.md`](c/ggml/README.md)。手工入口：

```bash
examples/repro/canonical/run.sh
examples/repro/source/run_c_baselines.sh
```
