# 模块 F — Typed Extension Body 构造

## 职责

把 canonical Weft Kernel IR + selected RISC-V execution 机械降低为具体 owner 的内部
表示和目标源码。typed execution body（`weft_rvv`/`weft_ime`/
`weft_scalar` 等）若继续存在，只是单向生成的瞬态 lowering IR，不是第四个
正式表示层，也不能持有新的 selection。

## 与旧架构的关系

旧架构中的通用 RVV vector/mask/index type、SEW/LMUL/policy、AVL/VL、setvl、
memory、widening 和 reduction primitive，以及 IME 的底层 MMA emission，都是
可抽取资产。大量 format-specific pre-realized body、selected-variant provenance
和 route metadata 不进入新主干。

复用单位是 owner-local primitive 与 lowering 知识，不是整套 typed-body 架构。
任何保留的 typed IR 都必须能够从 Weft Kernel IR、selected execution 和静态 target
表机械重建。

## 落地边界

- 第一条真实 repro 需要哪些最小 RVV primitive，应按调用链逐件抽取；禁止先把
  整个旧 dialect 移入新主干后再清理。
