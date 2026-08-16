# Packed Quantization 与 Irregular Access

Packed quantized kernel必须把storage ownership、decode relation、scale/zero-point与outer
traversal写在 DSL kernel 中。Target可以融合局部decode与compute，但不能从byte pointer、shape或
q-format名字猜出这些语义。

普通packed input仍可声明为external；只有caller按显式format identity构建并跨调用复用的
target-compatible object才声明`W.persistent(format)`，并用`W.storage`给出shape。Packed byte
内容本身不会让target自动授予persistent身份。

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
author-visible。Logical `block(32)` 可以描述一次局部nibble group：

```python
member = W.block(32)
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

## Plain F32 indexed GetRows

非量化GetRows使用同一条memory machinery而没有decode：DSL kernel 先把显式u32 row index cast为
logical index，再形成 `table + row * table_stride + hidden`；每个token的hidden dimension是
独立VLA。Target只根据pointer relation选择scalar indexed row base与unit-stride RVV copy，
不能从embedding shape或entry名发明row lookup。

`W.lookup` 与 `W.decode` 是更一般的 Kernel IR semantic anchors。DSL kernel 只有在table/index或
codebook relation本身就是算法语义时才使用它们；普通pointer arithmetic不会被compiler
自动提升成这两个op。

## Q8_0 activation quantization

Q8_0 DSL kernel 显式拥有每个32-element block的全部observable relation：

```text
maximum = reduce(max(abs(x)))
scale = maximum / 127
inverse = 0 if maximum == 0 else 1 / scale
store little-endian f16(scale) at byte 0
store 32 RNE+saturating i8 codes at byte 2
block stride = 34 bytes
```

两个VLA分别承担max reduction与narrow/store。Target为每个VLA、reduce和narrow独立选择LMUL，
并用F32→I16→I8 narrowing intrinsic实现显式RNE/saturation；34-byte layout与zero-maximum
branch不属于target推断。

## IQ4_NL codebook decode

IQ4_NL block是little-endian F16 scale加16个packed bytes，总计18 bytes。DSL kernel 把固定16-entry
signed-i8 codebook作为显式readonly pointer operand，拆出low/high nibble并调用
`W.decode(code, table, out_dtype=W.i8)`，随后signed widen到F32、乘scale并写32个logical值。

当前block-local decision只观察decode的table/code/result性质，选择一次16-lane
`vrgather` realization；emitter从decision取得E8M1/E32M4 physical shape。它不检查kernel名，
也不从packed byte pattern猜IQ4_NL。

## Q1_0 × Q8_0 row dot

Q1_0 packed input block覆盖128个logical weight，总计18 bytes：little-endian F16 scale加16个
sign bytes。一个Q1 block显式对应四个34-byte Q8_0 block；DSL kernel 负责：

```text
weight row stride = (K / 128) * 18
weight block      = row_base + block * 18
for sub_block in 0..4:
  sign bytes      = weight block + 2 + sub_block * 4
  activation      = q8 base + (block * 4 + sub_block) * 34
  result          = sign_bit_i8_dot(..., result)
```

`sign_bit_i8_dot`只定义当前32 elements的bit-to-sign relation。128-element grouping、四个独立
Q8 scale、outer block reduction与row output不会被target从format名或byte stride推断。

## Q4_0 / Q4_1 / Q5_0 / Q5_1 × Q8 local dot

Q4_0与Q4_1分别使用18-byte和20-byte packed weight block。DSL kernel显式形成block地址，加载
16-byte nibble field与32-element signed-i8 activation，然后调用`W.packed_i4_i8_dot`。Q4_0
传入zero point 8；Q4_1传入zero point 0和weight minimum×activation sum correction。

Q5_0与Q5_1分别使用22-byte和24-byte packed weight block。DSL kernel显式形成block地址，
加载16-byte low-nibble field、4-byte high-bit field与32-element signed-i8 activation，然后调用
`W.packed_i5_i8_dot`。Q5_0显式传入zero point 16；Q5_1显式传入zero point 0以及由weight/activation
offset形成的additive bias。

Primitive只拥有当前32-element unsigned-i4/i5 decode与signed-i8 dot。Weight/Q8 block stride、
scale/bias来源、outer block traversal和row accumulator由DSL kernel拥有。Target根据相同typed
operand关系选择VLEN128或VLEN256 local leaf，并决定nibble拼接、high-bit merge、widening multiply
与reduction；不会从22/24-byte stride或kernel名恢复Q5 route。

## TQ1_0 / TQ2_0 × Q8_K row dot

两种row dot都由DSL kernel显式遍历256-element block、形成weight/Q8_K地址、加载两个scale并
carry scalar accumulator。区别只落在两个局部semantic primitive：TQ1_0调用
`W.base3_ternary_i8_dot`消费48-byte base-3 codes与4-byte high digits；TQ2_0调用
`W.packed_i2_ternary_i8_dot`消费64-byte packed two-bit fields。

54/66-byte weight stride、292-byte Q8_K stride、scale field地址和outer row/block traversal不进入
primitive。Target共享byte/widen/reduction shape与resource选择，但为两种不同数值关系选择各自的
VLEN128/VLEN256 local leaf；这不是按TQ格式接管完整row dot。

## MXFP4 × Q8_0 row dot

MXFP4 packed input block覆盖32个logical weight，总计17 bytes：一个E8M0 exponent byte加16个
packed E2M1 bytes。DSL kernel 显式形成17/34-byte block地址，分别load exponent、packed codes和
Q8_0 scale/codes，然后调用一次 `e2m1_e8m0_i8_dot`。

E2M1 codebook与E8M0 scale是local primitive的可观察数值语义；row stride、block traversal、
activation layout和 C ABI仍是Kernel IR事实。Target可以选择table/register organization，
不能恢复whole-kernel `mxfp4` route。

## Activation quantize + affine local dot

Quantized projection通常包含两个不同层次：

```text
DSL algorithm
  activation max-reduce
  explicit scale computation
  W.narrow(..., rounding="rne", saturation=True)
  worker-local scale/code workspace store
  outer row / N / K traversal

local semantic primitive
  W.affine_i4_i8_dot(...)
```

这一 DSL variant要求 `inner % 32 == 0`，且每个activation block的maximum非零。若算法要
定义partial K32或all-zero block，必须在DSL中显式写出predicate/scale规则；target不能增加
默认值或防御性branch。下面是省略entry signature的 DSL excerpt：

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
    column = W.block(16)
    acc = W.zeros((column,), dtype=W.f32)
    for block in W.range(0, k_blocks):
        k = W.block(32)
        activation_codes = W.load(
            code_scratch + block * W.index(32) + k,
            other=W.i8(0),
        )
        packed_block = (column_begin // W.index(16)) * k_blocks + block
        packed_base = packed_weight + packed_block * W.index(304)
        acc = W.affine_i4_i8_dot(
            activation_codes,
            packed_base,
            activation_scale=W.load(scale_scratch + block, other=W.f32(0.0)),
            init=acc,
        )
    W.store(output_row + column_begin + column, acc)
```

`scale_scratch`与`code_scratch`是 DSL-declared workspace pointer：caller分配，shape进入 C
header，并由当前worker跨quantization与dot loops复用。Target可以把local primitive映射到RVV
dot或IME fragment，也可以在primitive envelope内做短生命周期repack；它不能自动创造
activation quantization pass、workspace ABI、N16/K32 outer loop或persistent weight format。

## Activation quantize + symmetric Q4_0 local dot

Q4_0 variant保留相同的activation max-reduce、scale/code scratch与 DSL N/K recurrence，但
persistent N16×K32 block是32-byte F16 scale field加256-byte packed nibble field，总计288
bytes。每个K32 iteration调用：

```python
acc = W.symmetric_i4_i8_dot(
    activation_codes,
    packed_base,
    activation_scale=scale,
    init=acc,
)
```

该 primitive定义 little-endian scale、`code - 8` 与每行16个local dot，persistent storage
identity为 `q4_0_n16_k32_288b`。Target由typed activation/init shape和target facts选择
N16×K32或M4N16×K32的RVV/IME1 realization；作者的K loop、accumulator state、column loop、
288-byte block address和最终store仍出现在生成C中。Leaf不遍历完整K、不量化activation、
不选择persistent format，也不拥有 C ABI。

## Grouped block dot

若Q4_K与Q8_K packed block已经由caller提供，DSL kernel显式声明它们是普通external还是
persistent storage，并显式遍历block、形成field offset、加载四个semantic operand，再调用
`W.grouped_affine_i4_i8_dot`。Primitive verifier只检查local operands的type/shape；storage identity、
packed byte extent和outer pointer relation由外围DSL与`W.storage`负责。Activation quantization是否
计入kernel由DSL ABI决定，target不能在两种scope之间自动切换。

Target为这一局部语义选择完整的operand shape、decode/reduction组织和exact leaf。VLEN128
realization使用固定宽局部RVV/asm序列；VLEN256 realization用32-lane nibble decode、signed
widening multiply和i32 reduction，metadata只解包当前八组scale/minimum。显式VLEN大于128且
满足shape/resource条件的其他target使用scalable leaf。选择在physical planning中完成，
intrinsic C生成不再读取VLEN二次分派。

这些模式与IQ2/IQ3/IQ1/Q6 codebook dot共享vector shape、indexed gather、widening、reduction
和resource machinery。IQ2_XXS与IQ3_XXS把各自packed layout和scale extraction留在DSL kernel，
并通过同一个`W.signed_codebook_i8_dot`表达32-element signed-table lookup与integer dot。每个
extension primitive仍保留自身完整的局部数值关系；它们不形成按format分派的whole-kernel route。
