# Extension Primitive

### 23.6 语义不同的扩展 primitive

假设某扩展提供带 block scale、固定 saturation 与特定 accumulator 语义的 dot，它不能作为普通 `W.contract` 的无差别 lowering。Source 使用显式 primitive：

```python
acc = W.block_scaled_contract(
    packed_a,
    packed_b,
    scale_a=scale_a,
    scale_b=scale_b,
    init=acc,
    rounding="rne",
    saturation=True,
)
```

不同硬件仍可以为该同一 semantic primitive 提供多个 provider。

---
