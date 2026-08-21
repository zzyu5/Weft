# Weft 文档入口

[`report/weft-spec.md`](../report/weft-spec.md) 是语言与编译器设计的唯一规范。本目录只说明当前前端代码怎样落实该规范，不另立一份设计。

- [语言模型](language/model.md)：kernel、View、Value、Level 与普通控制流。
- [Encoding](language/encoding.md)：基础布局与派生编码族。
- [Level 与控制流](language/level-and-control.md)：层归属、births 与 handoff。
- [函数、基本 op 与 engine role](language/functions-and-ops.md)：同语言标准库和硬绑定。
- [Canonical Kernel IR](compiler/kernel-ir.md)：当前 MLIR 类型、operation 与 verifier 边界。
- [RISC-V 编译空间](compiler/riscv-planning.md)：实体级瞬态属性、前向 pass 序列与候选枚举。
- [前端 examples](examples.md)：五个手工复现入口分别约束什么。

旧的 VLA、block command、内置 matmul、量化格式 op、pipeline permission 和 worker/program-tile 根模型已经不属于当前前端。
