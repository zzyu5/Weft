# 模块 F — Typed Extension Body 构造

## 职责

把模块 D/E 产出的 Physical Plan，构造成具体 owner 的 typed execution body
（`weft_rvv`/`weft_ime`/`weft_scalar` 等）。

## 与旧架构的关系

旧架构在这一层（typed body 的结构、dialect 设计、selected-body realization
边界）已经有相当成熟的基础设施（`weft_rvv` vector dtype/SEW/LMUL/policy/
AVL/VL/setvl/mask/tail 等结构），大概率是**复用为主**的模块——旧架构真正
需要推倒重来的是"怎么决定填进 typed body 的具体参数"（模块 C/D 的职责），
不是"typed body 本身怎么表示"。

## 待裁问题

- 旧 `lib/Plugin/RVV/` 下的具体 dialect/ops 定义在多大程度上可以直接保留，
  多大程度上需要因为模块 A 的输入契约变化（开放结构组合空间 vs 封闭问题
  目录）而调整——这需要在模块 A/D 的设计成熟后才能回答，本文档不预判。
