# Weft source corpus

`source/` 是当前端到端工作的输入语料库，不是测试目录，也不参与根 CMake 的默认构建。
目录第一层按 source language、第二层按 upstream/project 组织；同一变体的 source 与手工
runtime 相邻。这一布局沿用 IntentDSL 中真正有用的部分，但不复制它的 GPU runtime 模型。

```text
source/
├── weft/weft/<operator>/<variant>/
└── c/
    ├── weft/<operator>/<variant>/
    └── ggml/quantization/block_dot/<format-pair>/
```

## 本轮固定的十个 baseline

这十项是下一条 Scalar → RVV 端到端链路的固定 source corpus。名称描述算法语义，不是
whole-kernel provider route。

| # | Baseline | 主要覆盖 | Weft source | C reference/runtime |
|---:|---|---|---|---|
| 1 | `add_bias_f32` | VLA、load/store、pointwise | ready | ready |
| 2 | `rms_norm_f32` | reduction、两遍遍历、math | ready | ready |
| 3 | `online_softmax_f32` | summary algebra、两遍遍历 | ready | ready |
| 4 | `blocked_gemm_f16_f32` | logical block、contract | ready | ready |
| 5 | `q4_0_q8_0_block_dot` | signed nibble decode、scale | pending packed semantics | ready |
| 6 | `q4_1_q8_1_block_dot` | affine nibble decode、min fold | pending packed semantics | ready |
| 7 | `q5_0_q8_0_block_dot` | split high-bit decode、scale | pending packed semantics | ready |
| 8 | `q5_1_q8_1_block_dot` | affine 5-bit decode、min fold | pending packed semantics | ready |
| 9 | `q8_0_q8_0_block_dot` | int8 dot、block scale | pending quant contract | ready |
| 10 | `q4_K_q8_K_block_dot` | 256-wide super-block scale/min | pending packed semantics | ready |

前四项给出非量化主干的最小代表集；后六项恢复旧项目真正关心的 GGML block-dot 覆盖。
不再增加十一个“顺手”的算子，也不把 q-format 名称变成 provider selector。

## 当前边界

- `source/weft/weft/` 中既包含十项 baseline 已能忠实表达的四份 source，也保留最终规范的
  三份额外语言 acceptance source；后者不计入十项 baseline。
- 六项量化 reference 使用 GGML 的真实 packed block ABI 与通用标量算法。其 Weft source
  只有在 canonical 层能表达对应 decode/scale/min 语义后才加入；禁止用未定义的泛化 op、
  解包后的假输入或旧 q-format 整核 emitter 冒充。
- `source/c/ggml/LICENSE` 与 `source/c/ggml/README.md` 记录上游许可及抽取边界。
- `examples/repro/` 只提供手工运行入口；它不建立测试框架、case matrix 或新的验证合同。

手工检查 canonical source：

```bash
examples/repro/canonical/run.sh
```

手工编译并运行十份 C reference/runtime：

```bash
examples/repro/source/run_c_baselines.sh
```
