# Weft 文档索引

本目录是 Weft 项目唯一现行的方向性文档集合。它不带历史版本、不复述旧
架构（旧 V2/两柱六律/GPU 第二 execution paradigm 等，已整体归档到
`_attic/2026-08-07-riscv-quant-v2-archive/`）。

- [`总纲.md`](总纲.md) — 项目定位、研究问题、两条不可动摇的本质、系统
  边界、总体数据流、当前状态。**先读这份。**
- [`weft-intent-riscv-rearchitecture.md`](weft-intent-riscv-rearchitecture.md) —
  standalone Weft DSL 与嵌入式后端的架构裁决、RISC-V blocked/VLA 抽象、
  Intent→Weft bridge、Scalar/RVV/IME lowering、旧代码处置和新主干落点。
  实现新主干时以这份文档为准。
- [`WEFT_CURRENT_STATE.md`](WEFT_CURRENT_STATE.md) — 当前新主干已经真实成立的
  compiler slice、证据边界、layout/owner 成熟度和下一阶段承重问题。
- [`modules/`](modules/) — 模块级设计文档，按数据流顺序编号（A–H）。
  每个模块文档写清楚：职责、与旧架构的关系（复用/推倒/未定）、待裁问题。
  这是**模块化分解**，不是排期——不代表 A 先做完才能做 B。

## 模块列表

| 模块 | 名称 | 与旧架构关系（概览） |
|---|---|---|
| A | [front-end 契约](modules/A-front-end-契约.md) | 重写：独立 Weft Python eDSL + Kernel MLIR |
| B | [RISC-V domain/capability](modules/B-riscv-domain-capability.md) | 部分复用：粒度从整 kernel 变为按原语实例 |
| C | [owner 选择/legality](modules/C-owner-选择与legality.md) | 大改：按具体格式手写 → 按结构化原语种类泛化 |
| D | [Physical Plan 推导](modules/D-physical-plan-推导.md) | 全新：核心研究贡献，无现有代码对应 |
| E | [组合/嵌套一致性](modules/E-组合与嵌套.md) | 全新：旧架构从未处理嵌套决策依赖 |
| F | [typed body 构造](modules/F-typed-body构造.md) | 瞬态 lowering：只抽取通用 owner-local primitive |
| G | [artifact/发射](modules/G-artifact与发射.md) | 复用程度最高：artifact-neutral lifecycle 已就位 |
| H | [正确性/证据](modules/H-正确性与证据.md) | 一条真实 repro；不迁测试树和覆盖账本 |

## 旧文档去向

旧 `.trellis/spec/`、旧 `.trellis/tasks/`、旧 `docs/method/`、
旧 `docs/reports/`、`.trellis/事故档案/`、旧 workspace journal，已整体
移入 `_attic/2026-08-07-riscv-quant-v2-archive/`（该目录被 `.gitignore`
忽略，仅本地保留供查阅，不作为现行权威）。
