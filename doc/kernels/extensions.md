# Typed Local Extension Primitive

Extension用于普通Kernel op无法完整表达的可观察局部语义，不用于给完整kernel命名。Python
DSL 通过 `W.*` 暴露，Kernel IR operation属于 `weft_ext` sibling dialect。

## Affine i4×i8 block dot

```python
result = W.affine_i4_i8_dot(
    activation,             # block<32, i8> 或 interleaved block<128, i8>
    packed_base,            # scalar ptr<u8>
    activation_scale=...,   # scalar f32 或 block<4, f32>
    init=...,               # block<16, f32> 或 block<4,16, f32>
)
```

`packed_base` 指向一个局部 304-byte N16×K32 field：byte 0 开始是十六个 little-endian f16
scale，byte 32 开始是十六个 u8 zero point，byte 48 开始是 two-half packed nibble。每个 code
减去本列 zero point，与32-element signed-i8 activation做dot，再乘显式 activation scale 和
本列 f16 scale，最后加 `init[n]`。M1形式返回 `block<16,f32>`；M4形式用四行显式
activation scale和四路交错K32 code计算四组相同的N16关系，返回普通
`block<4,16,f32>`。

Primitive只拥有这16个local affine dot和上述局部byte relation。Activation quantization、
persistent packed-weight base、outer M/N/K loop、N16/K32 traversal和 C ABI必须在
Kernel IR中显式存在。IME1或其他matrix fragment只是同一语义的target realization。

## Symmetric i4×i8 block dot

```python
result = W.symmetric_i4_i8_dot(
    activation,             # block<32, i8> 或 interleaved block<128, i8>
    packed_base,            # scalar ptr<u8>
    activation_scale=...,   # scalar f32 或 block<4, f32>
    init=...,               # block<16, f32> 或 block<4,16, f32>
)
```

该primitive同样产生一行或四行的16个local dot。`packed_base`指向一个局部288-byte
N16×K32 field：
byte 0 开始是十六个 little-endian f16 scale，byte 32 开始是 two-half packed nibble。Nibble
数值是 `code - 8`；结果乘 activation scale 和本列 f16 scale 后加到 `init[n]`。

Q4_0 persistent block使用 `q4_0_n16_k32_288b` identity；affine Q4_1/Q4_K packed block使用
`affine_i4_n16_k32_304b` identity。当前block地址、K recurrence以及M4 activation staging仍由
DSL kernel表达。Target按row tile与target facts选择local N16×K32或M4N16×K32 RVV/IME1
realization，不拥有activation quantize、outer loop或kernel ABI。

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

External或persistent packed-block storage、pointer field offset和outer reduction loop由作者
拥有；具体byte extent与storage identity由外围DSL和`W.storage`声明，primitive只验证收到的
四个local block operands。VLEN、LMUL、register organization与RVV asm spelling属于target。

## Sign-bit × signed-i8 local dot

```python
result = W.sign_bit_i8_dot(
    sign_bits,        # block<4, u8>
    activation,       # block<32, i8>
    activation_scale, # scalar f32
    sign_scale,       # scalar f32
    init,             # scalar f32
)
```

对logical lane `p`，little-endian bit
`(sign_bits[p / 8] >> (p % 8)) & 1` 选择 `+activation[p]` 或
`-activation[p]`。32项整数和乘显式 `activation_scale * sign_scale` 后加到 `init`。

Primitive不拥有Q1 packed input block、128→4×32 sub-block关系或outer row/block traversal。
当前VLEN128 realization使用mask load、signed widening、sign merge与widening reduction；这些
都是target physical facts，不改变local numerical semantics。

## E2M1/E8M0 × signed-i8 local dot

```python
result = W.e2m1_e8m0_i8_dot(
    packed_codes,     # block<16, u8>
    exponent,         # scalar u8 E8M0 code
    activation,       # block<32, i8>
    activation_scale, # scalar f32
    init,             # scalar f32
)
```

每个packed byte的low nibble编码logical element `0..15`，high nibble编码 `16..31`。E2M1
codebook为 `[0,.5,1,1.5,2,3,4,6,0,-.5,-1,-1.5,-2,-3,-4,-6]`；E8M0 exponent定义
block scale，code zero采用f32 bit pattern `0x00400000`。Decoded dot乘
`activation_scale` 后加到 `init`。

External 17-byte MXFP4 packed input block、34-byte Q8_0 block和outer reduction由作者拥有。当前
VLEN128 realization选择nibble拼接、table gather、widening multiply与i32 reduction。

## Codebook / grouped integer local dot

`W.iq2_s_i8_dot`、`W.iq3_s_i8_dot`、`W.iq1_m_i8_dot` 与 `W.q6_k_i8_dot` 分别保留各自
code/high-bit/sign或delta/group-scale的可观察语义，输入activation、scale与init也都是显式
operand。它们共同产生一个256-element local integer dot，但persistent block stride、outer
row/block traversal和activation workspace仍在普通Kernel IR中。

Target从typed operands与target profile选择semantic lanes、byte vector shape、reduction segments
和resource budget。IQ2_S与Q6_K的fixed realization进一步选成明确的32-lane或64-lane leaf；
scalable realization、IQ3_S与IQ1_M各自选择对应local leaf。Leaf只拼写当前decode/widen/reduce
序列，不读取kernel名、格式名或外围source closure。

## Extension 判据

增加extension op必须同时满足：

- 存在普通canonical primitive无法无差别表达的observable local semantics；
- operands/results足以独立定义该语义，不依赖kernel symbol或whole-kernel shape；
- C ABI、persistent format、outer loop、staging与cross-primitive state仍在Kernel IR；
- target可以为同一op提供多个local realization，不需要复制完整operator emitter。

若新硬件只更快实现已有 `W.dot`、`W.matmul`、`W.reduce` 或memory semantics，应只增加target-local
realization，不增加extension op。
