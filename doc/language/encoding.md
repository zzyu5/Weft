# Encoding 与 View

Encoding 只描述内存位布局。它不携带字段不变量，不提供解码等价关系，也不授权编译器重建、删除或替换字段。

## 基础编码

```python
@weft.encoding
class Example:
    layout = bitorder.lsb_first, byteorder.little
    alignment = 16
    header: u16
    reserved = padding(2, value=0)
    lo: u4[16] @ nibble(lo_first)
```

字段按源码顺序布局。canonical declaration 明确保留：

- 字节内 bit order；
- byte order；
- 每个字段的 dtype、shape、packing、bit offset 与 storage bits；
- encoding alignment；
- padding 的 bit offset、宽度和填充值；
- 整个对象的 storage bits 与 layout identity。

`nibble(lo_first/hi_first)` 只适用于四位字段。相邻字段若使用相同的 `packed(n)`，它们必须合计恰好填满 `n` 字节；因此 Q4_K 的 `sc` 与 `m` 共同占据连续的 12 字节，而不是依赖 C bit-field ABI。

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
