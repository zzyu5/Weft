# 数值语义

## Numerical semantics

### 最小数值接口

Weft 不建立全局数值规则层。只有真正影响 observable semantics 与高性能 lowering 的属性进入 op：

- input / output dtype；
- accumulator dtype；
- `order` / reassociation policy；
- `math` mode；
- rounding；
- saturation。

### Reduction order

`order` 至少支持：

- `"ordered"`：保持 DSL kernel 定义的逐元素顺序，不允许改变浮点 parenthesization；
- `"preserve"`：允许对连续分区重新结合，但保持逻辑分区顺序；
- `"relaxed"`：允许目标相关的 tree、lane 与 strip 重组。

Built-in floating reduce 的默认值是 `"relaxed"`，以允许高性能 VLA reduction。需要严格复现时，作者必须显式选择 `"ordered"`。

### 跨 VLEN 可复现性

对于 `order="relaxed"` 的 floating reduce、typed summary、dot或matmul：

- 同一 DSL kernel 在不同 VLEN、LMUL 或 target realization 上可以采用不同 parenthesization；
- 结果低位可以不同；
- Weft 不承诺 bitwise reproducibility；
- 该差异是语言允许的实现自由，而不只是测试策略。

### Math mode

`math` 至少支持：

- `"strict"`：禁止未授权近似和乘加融合；
- `"native"`：允许目标原生精度与已定义的 fused instruction；
- `"fast"`：允许显式文档化的近似数学 realization。

默认 `"native"`。

Target必须读取`math`形成合法实现；不能把三种值当作无作用annotation。当前RISC-V target对
VLA `exp/tanh`没有`strict`实现时直接拒绝，不会静默使用近似实现。

近似 `exp`、`rsqrt`、activation 等 target realization 必须满足相应 primitive 定义的 value semantics；误差测试属于实现质量，不进入语言语义。

`W.floor(x)` 是显式逐元素语义，返回不大于 `x` 的最大整数值并保持floating dtype。它不
隐含坐标clamp、整数cast或vector realization；这些关系必须由 DSL kernel 分别写出，target只为
已有scalar/block/region value选择合法实现。

### Conversion、narrow 与 quantization

三种 conversion 必须区分：

- `W.cast(value, dtype)`：普通逐元素数值转换，包括 widening；
- `W.narrow(value, dtype, rounding="rne", saturation=False)`：显式窄化、rounding 与 saturation；
- `W.bitcast(value, dtype)`：等 bit width、逐元素保留 bit pattern。

`W.narrow` 不隐含 scale、zero-point、codebook 或 packed storage。完整 quantization relation必须
由 DSL kernel 中的显式 scale/math与 `W.narrow` 共同表达，或由一个 typed local quantization/
extension primitive表达。Persistent packed format与写入位置仍属于 DSL algorithm。

如果某个 RISC-V 扩展改变 observable scale、zero-point、codebook、rounding、saturation或
accumulation语义，必须增加新的局部 primitive或显式属性，不能伪装成普通 cast/dot
的无差别 lowering。

`W.bitcast(value, dtype)` 只允许相同固定 bit width 的 scalar element type，并逐元素保留
bit pattern。它不把多个 byte lane重组为宽 element，也不隐含端序 assembly。

`&`、`|`、`^`、`<<` 与 `>>` 只接受 integer/index element。右移遵守 operand 的有符号性：
signed integer 使用算术右移，unsigned integer 使用逻辑右移。
