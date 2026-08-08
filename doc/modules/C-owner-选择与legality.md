# 模块 C — Owner 选择 / Legality

## 职责

对每个结构化原语实例（一个 `contract`、一个 `reduce`……），给定其 typed
representation facts `g`、RISC-V domain 内某 owner 的能力投影 `c_o`、静态
上下文 `ω`，构造候选 owner + legality 判定 + analytic prior。

这是旧架构"每个具体格式手写一份候选表"（`RepackAccumulatorLMULFormulaResult`
等）要泛化的地方：泛化的方向是**按结构化原语种类**（`contract`/`reduce`/
`scan`/...）通用运作，而不是按具体量化格式（q2_K/q4_K/...）各写一份。

## 核心纪律（继承自旧两柱柱二，不动摇）

- candidate 由解析式 `A(g,c,ω)` 构造，不由 measurement 生成；
- legality 恒定先于 selection——不是 cost-based 全局 argmin；
- qualified measurement 只能在合法候选集内做残差修正，不创造候选、不让
  非法候选合法。

## 待裁问题

- 候选构造本身要不要用声明式规则表（IRDL 风格 owner-local 类型投影 +
  Datalog 风格候选/legality 推导），还是继续用 owner-local 手写 C++（两者
  的权衡见 `.trellis/tasks/`归档前的 brainstorm research，已随旧 task 树
  一并归档，需要时可从 `_attic/2026-08-07-riscv-quant-v2-archive/` 找回
  参考，但不作为现行结论）；
- 是否所有结构化原语种类共享同一套候选构造/legality 骨架，还是
  `contract` 与 `reduce`/`scan` 因为语义差异太大需要各自的骨架。
