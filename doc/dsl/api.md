# Python DSL API 索引

本文件汇总当前 canonical source surface。详细语义由同目录其他文档定义；目标不支持某个
合法 primitive 时必须明确报 unsupported，不能静默换成另一条算法路径。

## Kernel、类型与 qualifier

```python
@weft.kernel
def kernel(...) -> None | scalar: ...

W.i1
W.i8, W.i16, W.i32, W.i64
W.u8, W.u16, W.u32, W.u64
W.f16, W.bf16, W.f32, W.f64
W.index
W.ptr[T, qualifiers...]
W.constexpr[T]

W.readonly
W.writeonly
W.noalias
W.restrict
W.aligned(bytes)
W.address_space(name)
```

## Control 与 helper

```python
for i in W.range(begin, end, step=1): ...
if scalar_i1: ...
while scalar_i1: ...
W.select(predicate, true_value, false_value)

@W.pure
def pure_helper(...):
    return value

@W.helper(effects=("read", "write", "atomic", "fence"))
def effectful_helper(...):
    return value
```

`for ... else` 与 `while ... else` 不属于语言。Integer/index value支持 `&`、`|`、`^`、
`<<` 与 `>>`。

## VLA、predicate 与 memory

```python
with W.vla(begin, end) as i:
    ...

W.load(ptr, where=True, other=W.invalid, alignment=None)
W.store(ptr, value, where=True, alignment=None)
W.prefetch(ptr, where=True, locality="default")
W.atomic_add(ptr, value, where=True, order="relaxed")
W.fence(order="acq_rel")

W.valid(masked_value)
W.fill(masked_value, fill_value)
W.select(predicate, true_value, false_value)
```

## Logical block

```python
W.block_axis(extent, offset=0)
W.full(shape, value, dtype=None)
W.zeros(shape, dtype)
W.expand_dims(value, axis)
W.broadcast_to(value, shape)
W.reshape(value, shape)
W.transpose(value, permutation)
```

Python `value[:, None]` / `value[None, :]` 是 `expand_dims` 的 source sugar。

## State algebra

```python
W.reduce(value, op=..., identity=..., where=True,
         axis=None, order="relaxed", acc_dtype=...)

W.scan(value, op=..., identity=..., inclusive=True, where=True,
       segment_start=None, order="ordered", acc_dtype=...)

W.summary_fold(value, identity=..., lift=pure_helper,
               merge=pure_helper, finalize=None,
               where=True, coordinate=None, order="preserve")
```

普通 sequential carry使用 scalar `W.range` / Python `while`，不会被自动提升为 reduce、scan
或 summary fold。

## Structured compute 与 data relation

```python
W.contract(lhs, rhs, init=...,
           lhs_axes=(...), rhs_axes=(...), output_order=None,
           acc_dtype=..., out_dtype=...,
           where_lhs=True, where_rhs=True,
           order="relaxed", math="native")

W.dot(lhs, rhs, ...)          # rank-1/rank-2 contract sugar
W.permute(value, permutation)
W.lookup(table, indices, where=True)
W.decode(codes, table, where=True, out_dtype=...)
```

## Pointwise、special value 与 conversion

```python
W.maximum(a, b)
W.minimum(a, b)
W.exp(x, math="native")
W.exp2(x, math="native")
W.log(x, math="native")
W.sin(x, math="native")
W.cos(x, math="native")
W.rsqrt(x, math="native")
W.neg_inf(dtype)
W.tuple(a, b, ...)

W.cast(value, dtype)
W.widen(value, dtype)
W.narrow(value, dtype, rounding="rne", saturation=False)
W.bitcast(value, dtype)
```

普通 Python arithmetic、comparison 与 integer bitwise operator直接形成 canonical unary/
binary/compare op。

## 当前 extension source surface

```python
W.affine_i4_i8_contract(
    activation, packed_weight,
    activation_scale=..., weight_scale=...,
    weight_zero_point=..., init=...,
)

W.symmetric_i4_i8_contract(
    activation, packed_weight,
    activation_scale=..., weight_scale=..., init=...,
)

W.grouped_affine_i4_i8_dot(
    packed_weight, scale_min, activation, activation_sum_bytes,
    dot_scale, minimum_scale, init,
)
```

它们在 Python source 中通过 `W` 暴露，在 canonical IR 中属于 sibling `weft_ext` dialect。
Extension 必须保持 local、typed、可组合；完整 kernel、persistent layout和outer traversal不能
进入 extension op。
