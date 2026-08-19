# 公开 DSL API

本页只列当前 Python frontend、canonical Kernel IR 和 RISC-V lowering 同时存在的入口。未列出的
名字不是兼容 API；旧 `W.block/W.storage/W.dot/W.matmul` 与顶层量化命令已经删除。

## 定义

- `@weft.kernel`
- `@W.pure`
- `@W.helper(effects=(...))`

## Scalar 与 type

- `W.i1`, `W.i8/i16/i32/i64`, `W.u8/u16/u32/u64`
- `W.f16`, `W.bf16`, `W.f32`, `W.f64`, `W.index`
- `W.ptr[dtype, qualifiers...]`
- `W.constexpr[dtype]`
- pointer qualifier：`readonly`, `writeonly`, `noalias`, `restrict`, `aligned(n)`,
  `external`, `workspace`, `persistent(format)`

## Control 与域

```python
W.range(begin, end, step=1)
W.blocks(begin, end, block)
W.pipeline(W.range(...))
W.pipeline(W.blocks(...))
W.vla(begin, end)
W.axis(extent, offset=0)
```

`range/blocks/pipeline` 仅用于 `for`；`vla` 仅用于 `with`。

## Storage 与 memory

```python
W.buffer(ptr, shape)
W.load(ptr, where=True, other=W.invalid, alignment=None)
W.store(ptr, value, where=True, alignment=None)
W.transfer(source, destination, alignment=None)
W.load_f16_le(base)
```

## Engine value 构造与 pointwise

```python
W.full(domain, value, dtype=None)
W.zeros(domain, dtype)
W.accumulator(domain, dtype, init=0.0)
W.select(predicate, a, b)
W.cast(value, dtype)
W.bitcast(value, dtype)
W.narrow(value, dtype, saturation=...)
W.maximum(a, b)
W.minimum(a, b)
W.exp/tanh/log/sin/cos/floor/sqrt/rsqrt(value)
W.neg_inf(dtype)
```

Python arithmetic/comparison/bitwise operators在类型和逻辑轴相容时映射为 canonical pointwise op。
当前 `full/zeros/accumulator` 只构造 rank-one/rank-two f32 engine value；其他 dtype/rank 明确
unsupported。

## Collective 与 state

```python
W.vdot(lhs, rhs, *, init, acc_dtype, order="relaxed", math="native")
W.gemm(lhs, rhs, *, init, acc_dtype, order="relaxed", math="native")
W.reduce(value, op="add", identity=..., where=True, axis=None,
         order="relaxed", acc_dtype=None)
W.scan(value, op="add", identity=..., inclusive=True, where=True,
       segment_start=None, order="ordered", acc_dtype=None)
W.argmax(...)
W.online_softmax_summary(...)
W.sort_indices(...)
W.tuple(...)
```

具体 reduction/state 参数由对应 kernel 文档与 frontend 诊断约束；这些 command 不能互相猜测。

## Lookup 与 decode

```python
W.lookup(table, indices, where=True)
W.decode(codes, table, where=True, out_dtype=...)
```

## Typed quant commands

以下名字只存在于 `W.quant` namespace：

```text
affine_i4_i8_dot
symmetric_i4_i8_dot
grouped_affine_i4_i8_dot
sign_bit_i8_dot
e2m1_e8m0_i8_dot
packed_i4_i8_dot
packed_i5_i8_dot
base3_ternary_i8_dot
packed_i2_ternary_i8_dot
signed_codebook_i8_dot
packed_u9_u7_codebook_i8_dot
packed_u11_grid_delta_i8_dot
nibble_codebook_i8_dot
packed_i3_grouped_i8_dot
iq2_s_i8_dot
iq3_s_i8_dot
iq1_m_i8_dot
q6_k_i8_dot
```

调用形式为 `W.quant.<name>(...)`。每个命令的 typed operand domain、persistent format 与数值
合同见 [量化局部计算](../kernels/quantized-local.md) 和
[扩展 primitive](../kernels/extensions.md)。

## 不存在的 API

当前没有 source-visible engine selector、RVV/IME 名称、LMUL、vector type、fragment、pipeline
stage、`W.tune`、隐式 allocation、一般化 contraction 或通用 `qgemm`。需要这些名字才能工作
的代码不属于当前 Weft DSL。
