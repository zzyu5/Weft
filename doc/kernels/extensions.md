# Typed Local Extension Primitive

Extension用于普通Kernel op无法完整表达的可观察局部语义，不用于给完整kernel命名。Python
surface当前通过 `W.*` 暴露，canonical operation属于 `weft_ext` sibling dialect。

## Affine i4×i8 block contraction

```python
result = W.affine_i4_i8_contract(
    activation,             # block<32, i8>
    packed_weight,          # block<16x16, u8>
    activation_scale=...,   # scalar f32
    weight_scale=...,       # block<16, f32>
    weight_zero_point=...,  # block<16, u8>
    init=...,               # block<16, f32>
)
```

`packed_weight[n,p]` 的low/high nibble表示两个unsigned-i4 code。每个code减去
`weight_zero_point[n]`，与32-element signed-i8 activation做dot，再乘显式
`activation_scale * weight_scale[n]`，最后加 `init[n]`。Result是
`block<16,f32>`。

Primitive只拥有这16个local affine dot的numerical relation。Activation quantization、
persistent packed-weight address、outer M/N/K loop、N16/K32 traversal和public ABI必须在
Kernel IR中显式存在。IME1或其他matrix fragment只是同一语义的target realization。

## Grouped affine i4×i8 block dot

```python
result = W.grouped_affine_i4_i8_dot(
    packed_weight,          # block<128, u8>
    scale_min,              # block<12, u8>
    activation,             # block<256, i8>
    activation_sum_bytes,   # block<32, u8>
    dot_scale,              # scalar f32
    minimum_scale,          # scalar f32
    init,                   # scalar f32
)
```

该primitive定义一个256-element packed-i4×signed-i8 dot：128-byte weight由四组
low/high-nibble pair组成；八个logical 32-element group使用12-byte 6-bit scale/minimum
table；`activation_sum_bytes` 是十六个little-endian signed-i16 activation sum的32-byte
encoding，用于affine minimum correction。`dot_scale`、`minimum_scale`和`init`都是显式
numerical operand。

Persistent 144/292-byte block storage、pointer field offset和outer reduction loop由source拥有；
primitive不保存这些pointer relation。VLEN、LMUL、register organization与RVV asm spelling
属于target。

## Extension 判据

增加extension op必须同时满足：

- 存在普通canonical primitive无法无差别表达的observable local semantics；
- operands/results足以独立定义该语义，不依赖kernel symbol或whole-kernel shape；
- public ABI、persistent layout、outer loop、staging与cross-primitive state仍在Kernel IR；
- target可以为同一op提供多个local realization，不需要复制完整operator emitter。

若新硬件只更快实现已有 `W.contract`、`W.reduce` 或memory semantics，应只增加target-local
realization，不增加extension op。
