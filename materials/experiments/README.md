# 历史实验材料索引

本文件说明 `materials/experiments/` 的历史组织方式，以及它与旧测量数据的关系。
它不是设计权威；Weft 语言与编译器的唯一规范入口是
[`doc/index.md`](../../doc/index.md)。

## 这次重组做了什么

项目历史上，实验/测量数据按"战役代号"组织（`r51c-grid-family-flip`、
`g8-stage3-attack`、`t4b-m2-dispatch` 这类命名——代号本身不透露测的是什么算子、
什么格式、属于哪类实验），且散布在 `active/`、`archive/`、`sealed/`、`runs/`
四个并列目录下，同一个格式（如 q4_K）的历史记录可能同时出现在四个目录的十几个
不同战役代号文件夹里。

本次重组废除了"战役代号"这条组织轴，改为两条更直接的轴：

1. **实验类型**——这个实验测的是什么性质的东西；
2. **kernel/格式**——在这个类型下，这份记录是关于哪个具体算子/量化格式的。

旧的战役代号目录被拆开：能精确判断"这份文件属于哪个具体格式"的（多数情况下
文件名本身就带格式前缀，如 `iq2_xxs_dequant.c`、`q4_0_repack.kernel.c`），拆到
对应格式的文件夹下；判断不了、或者内容本身就是跨格式的批处理产物/基建脚本的
（如 `run_board.sh`、`FINDING.md` 这类整批共用的驱动/结论文件），归入该类型下的
`_cross-format/`——这不是偷懒兜底，是如实反映这份记录本来就不属于单一格式。

## 目录结构

```text
experiments/
├── coverage/     单算子/单格式覆盖率攻坚：vec_dot / gemm / gemv / dequant 正确性与性能
├── e2e/          端到端全模型 prefill/decode 分相测量
├── cold-start/   K1 / IME 硬件冷启动 kernel-sym 普查（对手身份核实、hand-brick vs 生成代码对比）
├── ime/          IME 矩阵扩展专项（性能桥接、paradigm 消融、与 RVV 资源关系）
├── selector/     选择器 / dispatch 消融与验证（哪个候选被选中、为什么）
├── repack/       repack layout 相关的构造与性能记录
├── scripts/      跨 kernel 可复用的通用测试驱动骨架（非一次性战役脚本）
├── _shared/      跨 kernel 的权威表格与公共分析资产（见下）
└── misc/         不属于以上任何类型的独立分析文档、原始 run 现场留痕
```

`coverage/`、`e2e/`、`cold-start/`、`ime/`、`selector/`、`repack/` 六个类型目录下，
按具体格式（`q4_0`、`q4_K`、`iq2_xxs`……）或跨格式（`_cross-format/`）建子文件夹；
子文件夹内部是若干个 `<来源战役代号>-history/` 目录——保留原始产物与来源标记，
不重写、不删除、不去重覆盖，只是换了摆放位置。

`_shared/` 下是明确跨 kernel、不属于单一格式的权威资产：

- `master/` —— 唯一权威结果主表（`T3_master_rebuild.csv`），是历史上所有测量
  结论汇总后的当前状态；
- `result-tables/`、`visibility/`、`roofline/`、`cert-status/`、`opponent-facts/`、
  `csv-templates/` —— 跨 kernel 的公共分析表、六态普查、物理墙标定、对手身份
  溯源、CSV 表头模板。

`scripts/` 下是已确认可复用、不绑定某一次具体战役的通用测试驱动骨架
（`dequant-row-drivers`、`kquant-vecdot-harness`、`product-reduce-harness`、
`e2e-harness-protocol`）。

## 诚实边界：这不是"最新记录 + 干净脚本 + 历史"三层

用户原本设想的结构是每个 kernel 文件夹下都有干净的 `{script, latest, history}`
三层。这次重组**没有做到这一层精细度**，原因：绝大多数历史记录是"一次性战役
产物"——脚本、原始日志、结论文档是绑在同一次具体测量里的，脚本本身往往写死了
特定路径或特定实验条件，直接抽出来当作"当前脚本"会造成误导（让人以为这是随时
能重新跑的通用工具，但实际上只对应那一次具体环境）。

因此本次重组的实际颗粒度是：

- 已经能确认"这是通用、可复用、不绑定一次性环境"的脚本 —— 已经抽出放进
  `scripts/`；
- 其余全部保留在各自 `<战役代号>-history/` 目录里，作为历史记录看待，不冒充
  "当前可用脚本"。

## 找回的官方 harness 脚本：只是留档，不是当前工具

上一轮工具链清理（归档 `tools/bench/`）时，连同官方 bench runner 一起把它调用的
五个 per-op harness 脚本（`cells/dequantize_row.sh`、`cells/vec_dot.sh`、
`cells/product_reduce.sh`、`cells/scalar_vec_dot.sh`、`cells/gemm_tile.sh`）和
它们共用的产物导出器 `export_current_artifact.py` 一并归档了，导致对应实验数据
下只剩"结果 + driver 源码"，没有"当年这些结果是怎么跑出来的"这一份记录。

这次把这五个脚本 + `export_current_artifact.py` + 说明其调用契约的
`cells/README.md`（现改名 `scripts/cells-README.md`）从 `_attic/` 找回来，按脚本
本身依赖的 driver 资产所在位置放回对应实验数据目录，不是集中放一处：

| 脚本 | 放回位置 |
|---|---|
| `dequantize_row.sh` | `scripts/dequant-row-drivers/` |
| `vec_dot.sh` | `scripts/kquant-vecdot-harness/` |
| `product_reduce.sh` | `scripts/product-reduce-harness/` |
| `scalar_vec_dot.sh` | `coverage/_cross-format/g8-A3-xscalar-rv64gc-history/` |
| `gemm_tile.sh` | `coverage/_cross-format/g8-P2-grid4-history/` |
| `export_current_artifact.py`、`cells-README.md` | `scripts/`（五个 harness 共用） |

**这纯粹是留档，不是恢复"当前可用工具"**——找回来的目的只是保留"这批历史测量
数据当年是怎么跑出来的"这条记录，跟当前 worker-local Weft kernel DSL/compiler
主干已经没有直接关系，不代表这些脚本现在还能跑、也不会为了让它们能跑
而去修复任何依赖（旧硬件环境路径、旧 `test/` fixture 位置等）。

## 已知问题：`_cross-format/` 体量偏大

`coverage/_cross-format/` 目前有 808 个文件，是全部类型目录中最大的一块——这
如实反映了历史测量数据本身有大量跨格式批处理产物（例如 `g8-stage3-attack` 系列
按"批次编号"而非"算子格式"组织实验，一个批次里混着好几个格式的驱动+日志）。
这不是分类工作没做完，而是这批历史数据本身的组织方式导致按格式切分后仍有大量
真正跨格式的内容。如果未来需要更细的颗粒度，需要重新审视这批 `_cross-format/`
内容、按需再往下拆，而不是本次重组能一次性解决的。

## 与旧目录的对应关系（供追溯）

旧的 `active/`、`archive/`、`sealed/`、`runs/`、`_templates/` 五个目录已经不存在，
内容全部按上述规则分散进新结构。若需要按旧战役代号反查具体内容去了哪里，可以
全局搜索该代号字符串（重组时保留了每份记录来源的 `<战役代号>-history/` 命名，
没有抹掉这条线索）。

上一版本的完整目录快照见 `../archive-local/2026-08-07-tooling-cleanup/`（git-ignored，
仅本地保留供追溯，不是当前权威）。
