# 局部计算命令

局部命令是作者对编译器的语义授权：用户明确说明“这里是 matmul/reduce/quant relation”，
编译器不用从普通 SSA 图猜；同时命令不取得外围 traversal、storage、state 或 kernel ABI。

## 普通 SSA 结果

所有有结果的 command 都产生普通 scalar/engine SSA value。合法程序包括：

```text
gemm → pointwise → store
gemm → two consumers
vdot → state update
reduce → branch merge
lookup/decode → quant compute
structured result → another compatible command
```

Direct-store、one-use、source adjacency 或固定 producer closure 都不是语言合同。

## `W.vdot`

当前 `W.vdot(lhs, rhs, init, acc_dtype, order, math)` 表达显式共享 reduction axis 的局部 vector
dot 关系，覆盖 canonical 支持的 `[R,K]×[K]`、`[VLA,K]×[K]` 与 `[R,K]×[VLA,K]`。它不创建
外围 reduction loop，也不决定 RVV lane 或 LMUL。

## `W.gemm`

`W.gemm(lhs, rhs, init, acc_dtype, order, math)` 表达局部
`[M,K] × [K,N] + init → [M,N]`。M/N/K 是 `W.axis` 建立并由 operand 共享的逻辑轴。

作者拥有：

- M/N/K 的 outer block traversal；
- BM/BN/BK 与循环顺序；
- accumulator 跨 K-loop 的 lifetime；
- visible staging、workspace 与 persistent packing；
- epilogue 与 store。

Target 拥有 command 内的 lane 方向、register microtile、multiple accumulators、K-unroll、load
reuse、primitive-private packing、local pipeline 与 RVV/IME realization。

`W.gemm` 不是预编译函数。Canonical `weft_kernel.matmul` 保留 operands、axes、init 与数值
属性，target 为当前上下文生成实现；同一命令在 GEMM、Conv、MoE、Attention 和 OutProd 的
外围程序中使用同一编译规则。

## Accumulator

`W.accumulator(domain, dtype, init=...)` 创建普通初始 engine value。它不创建特殊寄存器对象；
是否跨 loop 保存由普通 carry 决定，物理 placement 由 lifetime、consumer 和资源共同决定。

## Reduce、scan 与 state

`W.reduce` 消去指定逻辑轴；`W.scan` 保留顺序前缀关系；argmax、online summary 与 ordered
state 各自保留完整、不可互猜的可观察语义。Target 只能在语义允许时选择 lane collective、
sequential update 或局部 extension realization，不能把 ordered state 改成可重结合 reduction。

## Lookup 与 decode

Lookup/decode 明确 packed/codebook 数值关系和 predicate。它们的结果继续携带逻辑轴，可进入
pointwise、state、dot 或 store。Target 可融合局部 decode 与 compute，但不能改变 persistent
format 或接管外围 loop。

## `W.quant.*`

量化命令放在一个 namespace 下，表示它们是同一 DSL 的 typed packed-compute commands，而非
18 个平级语言根构造。每个 command 只保存真正不同的局部数值语义，例如 nibble/high-bit、
group scale/minimum、codebook index 或 correction。

编译器共享：packed-axis 分解、semantic lane、decode/widen、gather、cross-output reuse、
accumulator organization、resource 和 pipeline。最低层 leaf 只保留不可进一步分解的 RVV
intrinsic 或 IME asm。

当前不提供语义模糊的通用 `qgemm`。只有当多个真实格式能共享一个完整、可观察且不丢信息的
typed schema 时，才可以增加通用 command；不能为了 API 简短把格式语义藏回后端分支。

## 数值参数

`acc_dtype`、`order` 与 `math` 是语义自由度，不是机器调参。默认值使常见调用只需写：

```python
acc = W.gemm(a_block, b_block, init=acc, acc_dtype=W.f32)
```

只有算法需要不同顺序或数学承诺时才显式传入；LMUL、microtile、fragment、pipeline depth 和
target 名永远不作为 command 参数。

## Extension command 的门槛

新增 command 必须同时满足：

1. 有局部、完整、应用可观察的语义；
2. 不能从现有 command 与普通 SSA 无歧义表达；
3. 不拥有 outer control、blocking、persistent storage 或 ABI；
4. 有 canonical op/type、verifier 和真实 target artifact；
5. 未实现 target 明确 unsupported，不走 fallback。
