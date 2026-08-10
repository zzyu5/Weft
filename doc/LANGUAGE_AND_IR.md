# 语言与 IR 工程合同

> 本文把最终规范中的语言语义映射为实现责任，不新增语法或 IR 语义。概念 op 名称不是
> 对当前 TableGen schema 的认可；最终字段仍以 `WEFT_FINAL_SPEC.md` 为准。

## 1. 前端只做一次语义物化

Python eDSL 是参考 frontend。它负责读取受限 Python AST、解析静态 binding、执行语言
类型检查，并直接生成 canonical Weft Kernel MLIR。正确的数据流是：

```text
Python source + constexpr bindings
        │
        ├─ source capture / AST / locations       （瞬态）
        ├─ expression typing / symbol environment （瞬态）
        └─ canonical Weft Kernel MLIR             （唯一结果）
```

前端可以拥有通用 `Value/Operation/Region` assembly builder，但该 builder 只能机械打印
MLIR，不能拥有一份独立 op schema、长期 typed Python IR、独立 verifier 或优化 pipeline。
语义 verifier 必须落在 canonical dialect 上，所有其他前端也复用这一验证入口。

`@weft.kernel` 定义 AOT kernel，不允许把 kernel 当普通 Python 函数 eager 执行。Python
对象的职责止于构建期；最终 artifact 不依赖 Python runtime。

## 2. Source 能表达什么

### 2.1 ABI 与普通控制

Kernel 参数是普通 scalar、pointer、descriptor 和 `constexpr` meta-parameter。Pointer
qualifier、alignment、address space、restrict/alias 等必须通过 typed ABI contract 表达，
不能由 emitter 根据参数名猜测。

普通 `if`、scalar `W.range`、`while`、early exit 和 pure helper 属于 worker-local scalar
control。普通 sequential carry 也使用 scalar control；它不是 `W.vla` 的隐含能力。

### 2.2 VLA region

```python
with W.vla(begin, end) as i:
    ...
```

这段 source 只承诺逻辑区间 `[begin, end)` 覆盖一次。Canonical IR 必须以 structured
region 保存 begin/end、induction coordinate、region body、yield/state algebra 边界与 source
location。它不得保存 strip 切法或一次迭代的 `vl`。

Verifier 至少要阻止：

- 同一 lexical scope 出现两个活跃 VLA axes；
- VLA coordinate 或 physical strip state 逃逸到不合法 scope；
- 任意 loop-carried vector state 绕过 state algebra；
- contraction 收缩 VLA axis；
- effect 依赖跨 strip 顺序却没有声明 sequential semantics。

### 2.3 Logical block

`W.block_axis(extent, offset=0)` 建立静态 logical block axis。`full`、`zeros`、
`expand_dims`、`broadcast_to`、`reshape`、`transpose` 等操作产生或变换 logical block
values。Block shape、axis relation 和 source order 属于 canonical 事实；layout、microtile、
packing 与 fragment 不属于。

### 2.4 Memory、predicate 与 masked value

规范性 API 形状包括：

```python
W.load(ptr, where=True, other=W.invalid, alignment=None)
W.store(ptr, value, where=True, alignment=None)
W.prefetch(ptr, where=True, locality="default")
W.atomic_add(ptr, value, where=True, order="relaxed")
W.fence(order="acq_rel")
```

当 `other` 为 `W.invalid` 时，load 结果必须在 IR 类型或等价的一等语义中携带 validity。
Pointwise propagation、`W.valid`、`W.fill` 和 `W.select` 不能提前把 logical validity
坍缩成 RVV mask。Consumer 根据自身语义解释 invalid：

- store：无效元素不产生写 effect；
- reduce/scan：使用已声明 identity 与 algebra；
- contract：分别遵守 lhs/rhs predicate 和 accumulation contract；
- 显式 `fill`：作者选择何时物化普通值。

### 2.5 Structured compute 与数值属性

`contract`、`reduce`、`scan`、`summary_fold` 及 extension primitive 必须是 canonical
structured operations。它们持有 source-observable 的 axes、combiner、identity、output
order、dtype、accumulation、rounding、math mode 与 reproducibility contract；不得只用一个
`ordered: bool` 代替完整数值语义。

`contract` 的 operands 是 region values，其 outer traversal 和 algorithm/cache blocking
由 source 决定。Provider 只能实现这个局部 primitive，不能生成新的 outer loop、改变
operand relation 或把整段 kernel 替换为模板。

## 3. Canonical Kernel IR

### 3.1 保存的事实

Canonical IR 保存：

- kernel ABI、source locations、scalar/pointer/constexpr types；
- scalar control flow、VLA regions、logical block axes/values；
- pointer/index expressions、logical predicates、masked values；
- memory/effect operations；
- pointwise operations、state algebra、contract 与 typed extension primitives；
- source meta-parameters 与 observable numerical policy。

概念 op 族包括 `kernel`、`range`、`vla`、`block_axis`、memory/effect、mask/fill、
reduce/scan/summary_fold、contract、shape transforms 和 `meta_value`。Extension semantic op
位于 sibling dialect；正式输入必须能加载、parse、verify 其扩展 dialect。

### 3.2 明确禁止的事实

Canonical IR 不保存 exact `vl`、LMUL、register number/microtile、provider ID、IME
fragment、instruction spelling、measurement、thread count 或 launch policy。也不保存
`task_id/grid_rank`，因为这些概念根本不存在于语言中。

### 3.3 Identity 与引用

Selected records 需要稳定引用 canonical anchor，但 anchor 只是 IR identity，不是算法的
第二份名字系统。不得因 selected IR 需要引用就给每个节点复制一份 shape、axis 或 route
metadata。Clone、inlining 和 canonicalization 必须有明确的 anchor 更新规则。

## 4. Selected Execution IR

Selected IR 只记录“存在多个合法方案，而本次选了哪一个”的事实，例如：

- source meta binding；
- VLA region 由 Scalar 还是 RVV realization；
- SEW、LMUL、repeat、unroll、memory strategy；
- reduce/scan/summary strategy；
- contract provider、register microtile、fragment、packing、scratch；
- local fusion 与 math primitive strategy；
- specialization identity。

每条 plan 记录必须引用 canonical anchor。Selected IR 禁止复制 source bounds、pointer
expression、logical shape、contract axes、logical predicate、reduce combiner、state identity、
ABI 或 output order。需要这些事实时从 canonical IR 重算。

Provider 可以定义 sibling selected op，但它必须实现统一 selected-provider interface，
并服从同一条“不复制算法”约束。Selected verifier 验证引用存在、provider record 完整、
target legality 与跨 record compatibility；它不重新验证一份复制的算法。

## 5. 瞬态 lowering

Provider-local typed IR、EmitC、LLVM IR、RVV intrinsic graph 或 IME fragment lowering 可以
存在，但必须满足：

```text
canonical + selected + target facts  ->  transient lowering  ->  artifact
```

它不能接受 source 作为 front door，不能增加选择，不能反向修改 canonical 语义，不能被
缓存成第四份 authority。Emitter 遇到 selected record 缺失或非法时必须失败，不能从
kernel shape、名字或旧 route 补猜一个方案。

## 6. Verifier 分工

| Verifier | 验证什么 | 不验证什么 |
|---|---|---|
| Kernel verifier | source/canonical 语义、types、regions、effect、state algebra、numerics | target 是否有某条指令 |
| Target profile verifier | architectural facts 内部一致性 | 某个 kernel 是否快 |
| Provider legality | primitive + operands + target + local config 是否可实现 | whole kernel 属于哪类 |
| Selected verifier | anchors、选中 record、跨 record 兼容与完整性 | 复制的算法 schema |
| Artifact handoff | ABI、symbol、selected provenance 与生成完整性 | 重新做 provider selection |

“某层需要 verifier 保证它和上一层复制得一致”通常说明该层不应持久化。优先删除重复
事实，让它成为可重算分析，而不是继续增加同步验证器。
