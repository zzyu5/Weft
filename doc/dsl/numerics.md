# 数值语义

## 14. Numerical semantics

### 14.1 最小数值接口

Weft 不建立全局 numerical contract framework。只有真正影响 observable semantics 与高性能 lowering 的属性进入 op：

- input / output dtype；
- accumulator dtype；
- `order` / reassociation policy；
- `math` mode；
- rounding；
- saturation；
- exceptional-value policy（仅相关 op）。

### 14.2 Reduction order

`order` 至少支持：

- `"ordered"`：保持 source 定义的逐元素顺序，不允许改变浮点 parenthesization；
- `"preserve"`：允许对连续分区重新结合，但保持逻辑分区顺序；
- `"relaxed"`：允许目标相关的 tree、lane 与 strip 重组。

Built-in floating reduce 的默认值是 `"relaxed"`，以允许高性能 VLA reduction。需要严格复现时，作者必须显式选择 `"ordered"`。

### 14.3 跨 VLEN 可复现性

对于 `order="relaxed"` 的 floating reduce、summary fold 或 contract：

- 同一 source 在不同 VLEN、LMUL 或 provider 上可以采用不同 parenthesization；
- 结果低位可以不同；
- Weft 不承诺 bitwise reproducibility；
- 该差异是语言允许的实现自由，而不只是测试策略。

### 14.4 Math mode

`math` 至少支持：

- `"strict"`：禁止未授权近似和 contraction；
- `"native"`：允许目标原生精度与已定义的 fused instruction；
- `"fast"`：允许显式文档化的近似数学 realization。

默认 `"native"`。

近似 `exp`、`rsqrt`、activation 等 provider 必须满足相应 primitive 定义的 value semantics；误差测试属于实现质量，不进入 source 认证流程。

### 14.5 Quantization

窄化、量化与饱和 op 必须显式定义：

- source / destination dtype；
- scale / zero-point / codebook relation；
- rounding mode；
- saturation / wrap policy。

如果某个 RISC-V 扩展改变这些 observable semantics，必须增加新的局部 primitive 或显式属性，不能伪装成普通 cast/contract 的无差别 lowering。

---
