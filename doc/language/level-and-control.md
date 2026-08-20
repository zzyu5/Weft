# Level、值生命周期与普通控制流

## Level 不是命名循环

```python
with L.rows(M, group=16) as mb:
    acc = new(f32, [16], init=0)
    panel = materialize(pack(A[mb, :], along="k")) @ transfer
    ...
```

一个 canonical `weft_kernel.level` 固定拥有三个区域：

1. `state_births`：本层的 `new`；
2. `staged_births`：本层的 `materialize`；
3. `body`：该层的一段完整程序。

Level 的 domain operand 类型同时保存逻辑轴、与父层的 relation、extent、partition、multiplicity 和 tail。body 的第一个参数是当前 domain point，随后依次是外层 carried 值、state births 与 staged births。

body 必须以 `weft_kernel.handoff` 结束。源码中没有独立 handoff 语法：`acc += x`、`o = o * alpha + update` 都是普通 SSA 赋值；前端根据普通 def-use 把本层更新后的外层值放进 handoff。因而 IR 同时保留了普通 SSA 数值关系和“在哪一层交回”的归属。

`state_names`、`staged_names` 及三个 region signature 共同定义归属；它们不是附加在 `for` 上、可被忽略的 metadata。verifier 会逐项检查 births 的名称、yield、body 参数和 handoff/result 类型。

## staged 与 state

- `new` 的值可以被后代层更新，并由嵌套 level 的 handoff 沿普通 SSA 链交回。
- `materialize` 的值进入 staged region，传给 body 后为只读；对它重新赋值会在前端报错。
- `admit` 在哪一层出现，就表示该 View 区域在哪一层被供应一次；这是复用语义本身。

## 普通控制流

普通 `range`、`if`、`while` 分别生成 `weft_kernel.for/if/while`，用 `yield` 或 `condition` 显式携带 SSA 值。它们没有 `ordered`、`parallel`、`VLA` 或 `pipeline` 属性，默认就是有序标量控制。

Top-K 使用普通控制流且不生成任何 Level；这不是缺少优化标记，而是该算法树没有稳定的层级数值合成。
