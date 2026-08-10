# Verifier、作者义务与禁止退化

## 24. Verifier 义务

Canonical verifier 必须检查：

- kernel ABI 与 type legality；
- pointer arithmetic type；
- VLA region lexical nesting；
- VLA value 不非法逃逸；
- VLA region 中无任意外层 state mutation；
- block shape 与 broadcast；
- dynamic extent identity；
- contract paired axes 与 output shape；
- reduce/scan/summary state type；
- custom lift/merge/finalize purity；
- masked value 只进入 validity-aware consumer；
- meta parameter 不进入非法 runtime role；
- extension primitive dialect 是否注册；
- effect 与 atomic/fence 基本规则。

Selected verifier 必须检查：

- 所有物理 plan 引用真实 canonical anchor；
- provider 与 target capability 匹配；
- LMUL、register、fragment、scratch legality；
- selected record 未复制或改变 canonical algorithm；
- 每个需要 realization 的 primitive 有唯一计划；
- local fusion envelope 合法；
- multiversion predicate 具有 fallback 或明确 no-match 行为。

---

## 25. 作者义务

下列性质通常不能完全静态证明，由作者承担：

- `noalias`、alignment 与 bounds assertion 真实成立；
- VLA non-atomic iteration effect independence；
- custom summary `merge` 的 identity / associativity / commutativity 声明；
- runtime work slices 之间无未同步数据竞争；
- 外部 runtime 传入的 work descriptor 合法；
- extension-specific semantic primitive 的参数满足其 source contract。

这些义务必须由文档和 diagnostics 明确，不建立额外审批系统。

---

## 26. 禁止退化方向

出现以下任一情况，说明实现偏离本规范：

1. 在 canonical language 中重新引入 `program_id` / grid；
2. Weft runtime 自行创建线程或把 OpenMP 变成语言语义；
3. 把整个 worker kernel 当成一个静态 Triton-style tile；
4. 暴露 `vl`、VLEN、LMUL、lane ID 或 vector register number 给普通 source；
5. 把 logical block 等同于 register tile 或 IME fragment；
6. 从普通 multiply/add graph 自动发现并替换成 contract；
7. 根据 GEMM、Softmax、q4_K 等名字选择整段实现；
8. 新增扩展时复制整算子 kernel 模板；
9. provider 修改 source outer loop、staging、ABI 或 logical predicate；
10. selected IR 与 canonical IR 同时拥有算法真理；
11. tuner 创造 candidate 或让非法 candidate 合法；
12. 把 physical tail 存成 canonical logical mask；
13. 用普通 carried loop 假装 summary fold，同时期待 compiler 猜出 merge；
14. 把所有 state construct 压成一个无 observable distinction 的 op；
15. 把 numerical semantics 降成“测试时用容差”，却不给 compiler 合法 reassociation 权；
16. 让 Python 成为运行时依赖或唯一可生成 IR 的入口；
17. 将 extension evidence、approval 或 certification 变成 DSL 核心抽象。

---
