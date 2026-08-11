# Predicate Reduction

## Predicate 与 reduce identity

```python
with W.vla(begin, end) as i:
    valid = token_index(i) < sequence_length
    value = W.load(x + address(i), where=valid)
    maximum = W.reduce(
        value,
        op="max",
        identity=W.neg_inf(W.f32),
        order="relaxed",
    )
```

`valid=false` 的位置不贡献，语义等同于 `-inf`，而不是无条件填 `0`。

## Inclusive prefix scan

```python
with W.vla(row_begin, row_end) as i:
    value = W.load(source + i)
    prefix = W.scan(value, op="add", identity=W.f32(0.0),
                    inclusive=True, order="ordered", acc_dtype=W.f32)
    W.store(destination + i, prefix)
```

每个logical position的prefix都可观察。Target可以在一个RVV strip内用slide/add tree，并把
最后一个active lane显式传为下一strip carry；它不能只生成final reduction，也不能从store
use猜测scan。

## Coordinate-carrying argmax summary

First-index argmax通过显式coordinate形成 `(maximum, index)` state：

```python
state = W.summary_fold(
    value,
    identity=W.tuple(W.neg_inf(W.f32), W.index(0)),
    lift=argmax_lift,
    merge=argmax_merge,
    coordinate=i,
    order="preserve",
)
```

Lift把value和logical index组成state，merge在值相等时选择较小index。RVV max reduction、
equal mask与first-set-lane只是该summary algebra的target realization；index不是由physical
lane number补出。
