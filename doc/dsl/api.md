# Python DSL API 索引

本文件汇总当前 Python DSL。详细语义由同目录其他文档定义；目标不支持某个
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
W.external
W.workspace
W.persistent(format)

W.storage(pointer, shape)
```

External 是默认 storage class。Persistent/workspace pointer 必须在 entry body各有一个
`W.storage`；workspace 同时要求 `W.noalias`。`W.storage` 描述 caller-provided object，不执行
allocation。

## Control 与 helper

```python
for i in W.range(begin, end, step=1): ...
if scalar_i1: ...
while scalar_i1: ...
W.select(predicate, true_value, false_value)

@W.pure
def pure_helper(...):
    return value

@W.helper(effects=("read", "write"))
def effectful_helper(...):
    return value
```

`for ... else` 与 `while ... else` 不属于语言。Integer/index value支持 `&`、`|`、`^`、
`<<` 与 `>>`。Helper参数和返回值不写Python type annotation；其typed schema来自每个调用点，
只有kernel entry annotation定义 C ABI。

## VLA、predicate 与 memory

```python
with W.vla(begin, end) as i:
    ...

W.load(ptr, where=True, other=W.invalid, alignment=None)
W.load_f16_le(byte_ptr)
W.store(ptr, value, where=True, alignment=None)
W.select(predicate, true_value, false_value)
```

## Logical block

```python
axis = W.block(extent, offset=0)
W.full((axis0,), value, dtype=W.f32)
W.full((axis0, axis1), value, dtype=W.f32)
W.zeros((axis0,), W.f32)
W.zeros((axis0, axis1), W.f32)
```

每次`W.block`创建唯一 DSL axis identity；constructor的shape entry必须是direct block value，
不接受裸整数。Python `value[:, None]` / `value[None, :]` 构造identity为0的singleton logical axis。当前 DSL 没有
任意broadcast、reshape或transpose op；坐标重排必须由作者写成显式pointer/index relation。
当前`W.full/W.zeros`只创建rank-one或rank-two f32 block。

## State algebra

```python
W.reduce(value, op=..., identity=..., where=True,
         axis=None, order="relaxed", acc_dtype=...)

W.scan(value, op=..., identity=..., inclusive=True, where=True,
       segment_start=None, order="ordered", acc_dtype=...)

W.argmax(value, coordinate, tie="lowest_coordinate", order="relaxed")
W.online_softmax_summary(value, math="native", order="preserve")
W.sort_indices(input_ptr, output_ptr, scratch_workspace_ptr, extent,
               order="ascending", nan="last", tie="index_ascending")
```

`sort_indices`的scratch必须是rank-one `u32` workspace，其`W.storage` extent与排序extent相同；
它是caller提供的算法workspace，不是target隐藏的stack ABI。

普通 sequential carry使用 scalar `W.range` / Python `while`，不会被自动提升为 reduce、scan
或typed summary。

## Structured compute 与 data relation

```python
W.dot(lhs, rhs, init=..., acc_dtype=...,
      order="relaxed", math="native")
W.matmul(lhs, rhs, init=..., acc_dtype=...,
         order="relaxed", math="native")
W.lookup(table, indices, where=True)
W.decode(codes, table, where=True, out_dtype=...)
```

当前形态为：`dot`使用f32 multiplicand与f32 accumulator；`matmul`使用
f16 `[M,K] × [K,N]`与f32 accumulator；`lookup`是all-active VLA u8 index查询
`block<16xf32>`；`decode`是all-active `block<16xu8>`通过`block<16xi8>`表得到i8 block。

## Pointwise、special value 与 conversion

```python
W.maximum(a, b)
W.minimum(a, b)
W.exp(x, math="native")
W.log(x, math="native")
W.floor(x)
W.sin(x, math="native")
W.cos(x, math="native")
W.sqrt(x, math="native")
W.rsqrt(x, math="native")
W.neg_inf(dtype)
W.tuple(a, b, ...)

W.cast(value, dtype)
W.narrow(value, W.i8, rounding="rne", saturation=True)
W.bitcast(value, dtype)
```

普通 Python arithmetic、comparison 与 integer bitwise operator直接形成 canonical unary/
binary/compare op。

当前`W.narrow`只接受VLA f32→i8，并要求显式RNE与saturation；它不隐含scale或zero point。

`W.tuple`只接受scalar field；block state直接作为普通SSA value通过`for/while/if` carry。

## 当前 extension 运算

```python
W.affine_i4_i8_dot(
    activation, packed_base,
    activation_scale=..., init=...,
)

W.symmetric_i4_i8_dot(
    activation, packed_base,
    activation_scale=..., init=...,
)

W.grouped_affine_i4_i8_dot(
    packed_weight, scale_min, activation, activation_sum_bytes,
    dot_scale, minimum_scale, init,
)

W.sign_bit_i8_dot(
    sign_bits, activation, activation_scale, sign_scale, init,
)

W.e2m1_e8m0_i8_dot(
    packed_codes, exponent, activation, activation_scale, init,
)

W.packed_i4_i8_dot(
    packed_codes, activation,
    zero_point, dot_scale, additive_bias, init,
)

W.packed_i5_i8_dot(
    low_bits, high_bits, activation,
    zero_point, dot_scale, additive_bias, init,
)

W.base3_ternary_i8_dot(
    codes, high_digits, activation,
    weight_scale, activation_scale, init,
)

W.packed_i2_ternary_i8_dot(
    codes, activation,
    weight_scale, activation_scale, init,
)

W.signed_codebook_i8_dot(
    codes, sign_metadata, activation,
    grid_table, sign_table, dot_scale, init,
)

W.iq2_s_i8_dot(...)
W.iq3_s_i8_dot(...)
W.iq1_m_i8_dot(...)
W.q6_k_i8_dot(...)
```

它们在 Python DSL 中通过 `W` 暴露，在 Kernel IR 中属于 sibling `weft_ext` dialect。
Extension 必须保持 local、typed、可组合；完整 kernel、persistent format和outer traversal不能
进入 extension op。
