# Packed Quantization 与 Irregular Access

Packed quantized kernel必须把persistent storage、decode relation、scale/zero-point与outer
traversal写在source中。Target可以融合局部decode与compute，但不能从byte pointer、shape或
q-format名字猜出这些语义。

## Q4_K GetRows

GetRows的算法边界是：

```text
for token:
  row = row_indices[token]
  row_base = packed_table + row * packed_row_stride
  for quant_block:
    load explicit scale/minimum bytes
    decode low/high nibbles
    store dequantized logical values
```

Token-to-row relation、packed row stride、block size、scale/minimum decode和output位置都是
source-visible。Logical `block_axis(32)` 可以描述一次局部nibble group：

```python
member = W.block_axis(32)
packed = W.load(
    packed_block + W.index(16) + group_pair * W.index(32) + member,
    other=W.u8(0),
)
low_code = packed & W.u8(15)
high_code = packed >> W.u8(4)
low_value = (
    block_scale * W.cast(low_scale, W.f32) * W.cast(low_code, W.f32)
    - block_minimum * W.cast(low_minimum, W.f32)
)
high_value = (
    block_scale * W.cast(high_scale, W.f32) * W.cast(high_code, W.f32)
    - block_minimum * W.cast(high_minimum, W.f32)
)
W.store(output_group + member, low_value)
W.store(high_output_group + member, high_value)
```

Target可以选择indexed/unit-stride RVV、register unpack和local producer fusion；它不能把任意
load/bitwise graph重新分类成Q4_K，也不能改变row lookup或persistent format。

`W.lookup` 与 `W.decode` 是更一般的canonical semantic anchors。Source只有在table/index或
codebook relation本身就是算法语义时才使用它们；普通pointer arithmetic不会被compiler
自动提升成这两个op。

## Activation quantize + affine contract

Quantized projection通常包含两个不同层次：

```text
source algorithm
  activation max-reduce
  explicit scale computation
  W.narrow(..., rounding="rne", saturation=True)
  persistent scale/code scratch store
  outer row / N / K traversal

local semantic primitive
  W.affine_i4_i8_contract(...)
```

这一source variant要求 `inner % 32 == 0`，且每个activation block的maximum非零。若算法要
定义partial K32或all-zero block，必须在DSL中显式写出predicate/scale规则；target不能增加
默认值或防御性branch。下面是省略entry signature与byte-assembly helper的source excerpt：

```python
for block in W.range(0, k_blocks):
    with W.vla(0, 32) as lane:
        value = W.load(x + block * W.index(32) + lane)
        maximum = W.reduce(W.maximum(value, -value), op="max",
                           identity=W.f32(0.0), acc_dtype=W.f32)

    scale = maximum / W.f32(127.0)
    with W.vla(0, 32) as lane:
        value = W.load(x + block * W.index(32) + lane)
        code = W.narrow(value / scale, W.i8,
                        rounding="rne", saturation=True)
        W.store(code_scratch + block * W.index(32) + lane, code)

for column_begin in W.range(0, columns, W.index(16)):
    acc = W.zeros((16,), dtype=W.f32)
    for block in W.range(0, k_blocks):
        k = W.block_axis(32)
        activation_codes = W.load(
            code_scratch + block * W.index(32) + k,
            other=W.i8(0),
        )
        column = W.block_axis(16)
        packed_byte = W.block_axis(16)
        packed_block = (column_begin // W.index(16)) * k_blocks + block
        packed_base = packed_weight + packed_block * W.index(304)
        weight_scale = load_f16_le(packed_base + column * W.index(2))
        weight_zero_point = W.load(
            packed_base + W.index(32) + column, other=W.u8(0)
        )
        packed_codes = W.load(
            packed_base
            + W.index(48)
            + (packed_byte[None, :] // W.index(8)) * W.index(128)
            + column[:, None] * W.index(8)
            + packed_byte[None, :] % W.index(8),
            other=W.u8(0),
        )
        acc = W.affine_i4_i8_contract(
            activation_codes,
            packed_codes,
            activation_scale=W.load(scale_scratch + block, other=W.f32(0.0)),
            weight_scale=weight_scale,
            weight_zero_point=weight_zero_point,
            init=acc,
        )
    W.store(output_row + column_begin + W.block_axis(16), acc)
```

Target可以把local primitive映射到RVV dot或IME fragment，也可以在primitive envelope内做
短生命周期repack；它不能自动创造activation quantization pass、scratch ABI、N16/K32
outer loop或persistent weight layout。

## Grouped block dot

若persistent Q4_K与Q8_K block已经由caller提供，source显式遍历block并加载四个semantic
operand，再调用 `W.grouped_affine_i4_i8_dot`。Activation quantization是否计入kernel由source
ABI决定，target不能在两种scope之间自动切换。

这三个模式共享packed decode与local contraction machinery，但不形成按format分派的
whole-kernel route。
