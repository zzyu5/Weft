# Predicate Reduction

### 23.3 Predicate 与 reduce identity

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
