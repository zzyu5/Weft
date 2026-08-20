# 函数、基本 op 与 engine role

## 普通函数与标准库

普通 Python 函数在 kernel lowering 时 inline。`python/weft/std/` 只保存这类同语言函数；没有内置 matmul，也没有 canonical `matmul` operation。

同名实现可由 `weft.overloads(name, ...)` 组成重载集合。选择读取实参的 View encoding、rank、scalar dtype，以及显式 engine 参数；只按参数个数不能形成唯一选择时直接报错。重载选择的是作者写好的树，不是后端结构搜索。

## 当前基本 op

```text
mac_pairs / mac_groups   固定相邻分组的有限位宽乘加
widen                    保持逻辑元素，提升位宽或转成不更窄的浮点
reduce / fold2 / dot     明确的局部归约关系
contract                 指定轴缩并
outer_contract           外积式缩并
lookup                    显式查表
pack                      invocation 内的短生命周期重排
interleave                derive builder 内的持久编码生成操作
```

GEMM 是包含 `outer_contract` 的普通函数；GEMV 是包含 `contract` 的退化树。任何 operation 都不拥有外围 traversal、完整 operator ABI 或另一套 kernel。

`maximum`、`minimum`、`exp` 和 Python 有限位宽算术生成普通 unary/binary/compare op。`contract(acc=f32)` 的累加类型进入 canonical op 与 result type，不是被接受后忽略的注解。

`i32(value)` 等 dtype constructor 是显式有限位宽转换，生成保持逻辑 domain 的 `cast`；它不会被当成 widening 或由 store 悄悄猜出。

## Engine role 是硬绑定

`@ scalar / @ wide / @ matrix / @ transfer` 可选地绑定一个明确 basic op。绑定只作用于表达式根，不会泄漏到它的 operands；给 inline 函数调用绑定时，绑定其返回的根 operation。

当前结构兼容关系是：

- `transfer`：`admit`、`commit`、`materialize`、显式 stage handoff、`pack`、`interleave`；
- `matrix`：`dot`、`contract`、`outer_contract`；
- `wide`：pointwise、compare、分组乘加、widen、reduce、fold2、dot/contract 与 lookup；
- `scalar`：标量 pointwise、compare、cast 与 lookup；它不能绑定分组乘加、归约或 contraction。

缺省 role 允许后续 target 求解合法 realization；一旦写出 role，verifier 与 target 都必须遵守，不允许把 `@wide` 改投 matrix。
