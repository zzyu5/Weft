# Transpose、RoPE 与 Online Attention Composition

Attention相关kernel展示的是多个canonical construct如何组合，不是让Weft导入或融合一个
attention graph。每个worker-local entry都显式拥有自己的loop、pointer、workspace与state。

## Contiguous transpose

```python
@weft.kernel
def transpose_f32(
    input: W.ptr[W.f32, W.readonly],
    destination: W.ptr[W.f32, W.writeonly],
    rows: W.index,
    columns: W.index,
    input_stride: W.index,
    destination_stride: W.index,
) -> None:
    for row in W.range(0, rows):
        with W.vla(0, columns) as column:
            value = W.load(input + row * input_stride + column)
            W.store(destination + column * destination_stride + row, value)
```

这里的permutation是显式pointer relation，不是block shape-transform primitive。Target可以选择unit-stride
load加strided/indexed store，但不能改变row/column mapping。

## RoPE NeoX

RoPE DSL kernel 显式分成angle preparation与rotation：

```text
for position:
  theta = DSL-defined initial state
  for pair:
    cache cos(theta), sin(theta)
    theta = theta * theta_scale

  for head:
    VLA over half dimension
      load first/second half
      apply explicit rotation formula
      store both halves
```

`theta` 是ordered sequential carry；angle cache是 DSL-declared worker-local workspace，其
shape与lifetime进入entry storage declaration。Target可以为
sin/cos与VLA arithmetic选择不同realization，但不能把carry猜成scan，也不能自动创建或删除
cache stage。

## Causal masked memory

```python
for row in W.range(0, heads * queries):
    query = row % queries
    with W.vla(0, keys) as column:
        masked = column > n_past + query
        W.store(scores + row * stride + column,
                W.neg_inf(W.f32), where=masked)
```

Row/query映射、causal predicate、logical key domain与predicated store effect均在 DSL kernel 中。
Target为predicate选择mask realization，为store选择unit-stride memory与LMUL；physical tail
不能被并入或替代causal predicate。

## Online FlashAttention

Worker-local online attention保留完整algorithm skeleton：

```text
for query_head:
  map to explicit KV head
  for query:
    stage query F32 -> F16 workspace
    initialize maximum, total and output accumulator
    for key within causal bound:
      score = load F16 query/key, widen to F32, multiply + F32 reduce
      compute new maximum and rescale old state
      update total and value accumulator
    normalize accumulator to F32 output
```

Query staging、causal bound、GQA mapping、key order与online `(maximum,total)` update都是
author-observable。普通scalar carry保持key iteration order；它不是局部typed summary，因为每步
还更新一个value accumulator和workspace-visible state。Query与accumulator workspace均由caller
按generated header分配，由当前worker在一次invocation内独占；它们可跨key loop复用，不是
target可偷造的primitive temporary。

Target lowering可以分别实现并联合安排以下局部运算：

- F32→F16 query staging；
- widening F16 dot reduction；
- weighted F16 accumulator update；
- F16→F32 normalization；
- unit/strided memory与局部pure fusion。

这些realization可以共享register/load schedule，但不能合并成按`flash_attention` symbol选择的
whole-kernel emitter，也不能从一个framework attention graph发明上述loop与state。

Transpose、RoPE与FlashAttention在application中可以相邻调用，但Weft core不执行跨entry
graph fusion。若上游要构造新的fused worker-local algorithm，必须生成一份新的完整canonical
Kernel IR，而不是要求target猜测。
