# Encoding 与 View

Encoding 只描述逻辑字段坐标到 storage unit/bit range 的映射。它不携带字段不变量，不提供解码等价关系，也不授权编译器重建、删除或替换字段。

## 基础编码

```python
@weft.encoding
class Example:
    layout = bitorder.lsb_first, byteorder.little
    alignment = 16
    header: u16
    reserved = padding(2, value=0)
    q: u4[32] @ grouped(32) @ layered(16, lo_first)
```

字段按源码顺序占据 storage span；只有显式属于同一个 `joined` 组合的字段可以共享 span。字段内部不再默认使用 `bit_offset + i * width`。作者组合：

- `natural`：连续自然元素；
- `grouped(N)`：每 N 个逻辑元素重复一次布局；
- `layered(P, order)`：组内每 P 个元素一层，同一位置共享一个 storage unit 的不同 bit range；
- `joined(group, fields, low_bits, order)`：多个同形逻辑字段共享规则化 storage，后半字段 bits 分层回填；
- 后续由真实消费者闭合的 `bit_planes`：独立 storage plane 的纯 bit 重组。

canonical declaration 明确保留：

- 字节内 bit order；
- byte order；
- 每个字段的 dtype、shape、storage span 与结构化 layout expression；
- encoding alignment；
- padding 的 bit offset、宽度和填充值；
- 整个对象的 storage bits 与 layout identity。

Q4_0 的 q 使用 `grouped(32) @ layered(16, lo_first)`；Q4_K 的 q 使用 `grouped(64) @ layered(32, lo_first)`。二者 storage 大小相同于线性 nibble array，但逻辑索引映射不同。Q8 字段保持 `natural`。

Q4_K 的 `sc/m` 使用相同的 `joined(4, 2, 4, lo_first)` 并共享 12 bytes。该词只规定 bit 的规则化重组，不包含 scale/min 数学；禁止把它退化成两个连续 `u6[8]` bit streams。

`View[Encoding, shape]` 是使用该编码的内存对象；`View[f32, (M, N)]` 由前端生成固定 identity 的 dense encoding 特例。

## 派生编码

```python
@weft.derive
def Q4K_I16(W: View[Q4_K, (M, K)]) -> View[Q4K_I16, (M, K)]:
    return interleave(W, rows=16)
```

`derive` 的 region 是 build/load 阶段的类型生成程序，不是执行 kernel 的外层循环。canonical IR 区分：

- `derived_family`：如 `Q4K_I16<rows=16>`，只有 builder 参数，没有固定物理 layout identity；
- `derived_instance`：artifact 选择完成后的具体布局，必须带非空 identity。

本轮前端生成并验证抽象派生族及其 builder；具体实例由后续 artifact 构造阶段产生，不能在 kernel lowering 时偷偷换布局。
