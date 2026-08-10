# 模块 E — 组合 / 嵌套一致性

## 职责

处理 Weft blocked loop、block tensor、reduce/dot 和 owner group 嵌套时的资源
一致性。这是全新难点，旧架构（全部是"叶子"问题类型，从未处理过一般 nested
blocked program）没有对应设施。

典型场景（如 FlashAttention）：外层 task/block tile，与内层 state-carry loop
中嵌套的 `dot` 的 LMUL/资源决策不是独立的——外层 tile 选大了，会挤爆
内层 `dot` 所需的寄存器/scratch
预算。模块 D 的推导公式如果只按"单层原语独立求解"设计，无法处理这种
跨层资源竞争。

## 待裁问题

- 跨层一致性校验/联合求解的具体机制（自顶向下传递资源预算约束？自底向上
  先求内层需求再约束外层候选集？两者都需要，还是有更好的第三种方式？）；
- Weft compile-time meta-parameter、runtime scalar 和 block-tensor extent 如何
  在类型上区分，避免 tuner 参数进入 runtime algorithm 或 C ABI。
