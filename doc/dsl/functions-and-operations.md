# 函数、基本 Operation 与编译配置

## 1. 一门语言、一个用户层级

Weft 的 std 函数与应用 kernel 使用同一门语言：

```text
没有 privileged library IR
没有 opaque whole-kernel template
没有“普通用户 DSL / 专家 DSL”两套语义
```

std helper 是可读、可复制、可修改的普通函数。它们在 canonical Kernel IR 形成前 inline；函数内部的 Level、Value、Encoding access 和 operation 与调用者组成同一棵 SSA 程序。`@weft.derive` builder 在 artifact phase 单独求值并产生派生 Encoding，不进入 invocation kernel 的 SSA 树。

库可以维护多个 overload/特化：

```python
gemv(f32, f32)
gemv(Q4K_I[16], Q8_K)
gemv_panelized(Q4K_I[16], Q8_K)
```

前两者因 operand Encoding 不同而是不同程序；第三者若增加 `materialize` 或改变 Level，则也是另一棵作者树。前端根据函数调用、参数类型和静态 source 参数唯一解析 overload。build config 不能替作者在多棵 std tree 中搜索算法结构。

## 2. 没有内置 GEMM

语言核心没有 whole-kernel `matmul` primitive。GEMM、GEMV、quantized vec-dot、attention 和 sort 都是 std 中的普通函数。

GEMM 使用：

```text
Level hierarchy
materialize / admit / commit
new accumulator
outer_contract
```

GEMV 使用同一组构造，去掉 N/column Level 并把 accumulator 降为一维。`outer_contract` 只定义局部 operand/result 数值关系，不拥有 M/N/K outer traversal、blocking、staging、persistent Encoding、accumulator lifetime 或 kernel ABI。

## 3. `auto`

```python
with L.tiles(N, extent=auto("NC")) as nc:
with L.rows(mc, group=auto("MR")) as mb:
```

`auto` 声明由构建过程实例化的有限 source 参数域。`auto("MR")` 在所在函数中引入同名静态符号；同一作用域内同名 `auto` 必须绑定同一个值。std overload 或显式 build config 必须提供非空有限值域。

每组绑定形成一份固定 source candidate。其 Level tree、logical values、births 和 handoff 不再变化；target 只物理化这棵树。资源不足时 spill 或拒绝，不回头修改 source 参数。

target 还可以为已经确定的物理结构公开 LMUL、physical microtile extent、unroll、pipeline depth 和 buffer count 等有限 physical parameters。source `auto` 可以改变作者程序的 cohort 或 Level extent；physical parameter 只实例化同一 source tree。二者可以由同一构建工具实测，但不能混成一类参数。

## 4. 基本 operation 的要求

每个基本 operation 必须唯一规定：

- operand/result 的 element type；
- result logical shape 和 axis relation；
- 被消去或保留的 axes；
- 数值顺序、结合和精度语义；
- overflow/wrap/saturate 行为；
- memory/effect 行为。

operation 不声明 source engine role。target operation contract 另行判断该 canonical operation 能否由 scalar、RVV、IME 或 transfer engine 实现。编译器也不能从普通 SSA closure 识别出未写明的 dot、online softmax、codebook axis 或其它 primitive。

## 5. Pointwise 与类型 operation

```text
unary        neg / abs / exp / ...
binary       add / sub / mul / div / maximum / minimum / bitwise
compare      typed comparison
cast         数值转换或明确 reinterpret
widen        保持逻辑元素与 axes，提升 element width
narrow       保持逻辑元素与 axes，缩窄 element width
```

Pointwise operands 按 axis identity 对齐。缺少某轴的 scalar/singleton operand 可以 broadcast；两个同名 axis extent 不一致是类型错误。`widen` 和 `narrow` 的位置是 canonical 数值语义，target 不能把转换移过会改变 overflow 或舍入的位置。

浮点 `maximum/minimum` 使用 IEEE 754 `maximumNumber/minimumNumber` 语义：单个 NaN 与数字比较时返回数字，两个 NaN 返回 NaN；`maximum(+0,-0)=+0`，`minimum(+0,-0)=-0`。整数版本使用相应的有符号/无符号全序。

## 6. Grouped multiply-accumulate

```python
mac_pairs(a, b, into=i16)          # [N] × [N] -> [N/2]
mac_groups(a, b, n=4, into=i16)   # [N] × [N] -> [N/n]
```

grouped result 继承被分组 axis 的 identity，只把 extent 除以 group width；后续 `reduce(..., axis="k")` 仍引用同一 logical axis。operand extent 必须能被 group width 整除。group width 与 `into` 改变 result shape、partial 数量和 overflow 行为，属于作者选择；RVV widening MAC、普通 multiply+reduce 或 extension instruction 是同一 operation 的物理 realization。

## 7. Reduction 与 contraction

### 7.1 `reduce`

```python
reduce(value, op="add", axis="k")
```

`reduce` 消去指定 logical axes，保留全部 free axes。logical reduction domain 必须非空；physical tail 只补 identity，不增加 logical elements。

integer `add` 使用声明的 accumulator overflow 规则；floating `add` 在 closed operation 内允许改变括号次序，但不允许移出 operation/Level boundary；`max/min` 使用与 pointwise 相同的 NaN 和 signed-zero 规则。每个新增 reduction operation 必须定义 identity、空域政策和结合自由度。

源程序只有一个 reduce result；target 可以在 closed operation 内使用多个 physical partial、lane collective 或树归约。这些 temporary 不成为 canonical Value。

### 7.2 `fold2`

```python
fold2(x)       # T[N] -> i32[N/2]
```

相邻两项相加并提升到 `i32`。输入最后一轴必须能被 2 整除。它改变 logical shape，因此是显式 operation。

### 7.3 `dot`、`contract` 与 `outer_contract`

```python
dot(a, b)
contract(a, b, over="k", acc=f32)
outer_contract(a, b, over="k", acc=i32)
```

`dot` 逐元素相乘并消去 operands 的最后一个共同 axis。`contract` 在指定 reduction axes 上缩并；`outer_contract` 显式保留两侧不同的 free axes，形成 outer-product result。它们只拥有局部数值关系，不拥有外围 traversal、staging、state 或 ABI。

## 8. Lookup 与显式 index domain

```python
indices = iota(8, dtype=u32)
values = lookup(codebook, base + indices)
```

`lookup` 对每个 index coordinate 读取 table entry；result shape/axes 与 indices 完全相同。越界 policy、mask 与 fill 必须由调用显式给出。八个独立 scalar lookup 不等于一条 `[8]` lookup；target 只能为已经存在的 shaped index 选择 unit/indexed/gather 或专用 table realization。

## 9. Logical projection、reshape 与 transpose

对 shaped Value 的 slice/projection 必须显式保存输入输出 axis relation：

```python
panel = materialize(admit(B[kc, nc]))
b = panel[kb, nb]
```

`reshape`、`transpose` 和显式 index operation 若改变 logical axes、coordinates 或遍历关系，必须作为 canonical operation 写入程序，并完整定义坐标映射。它们不是 physical layout hint。

invocation-local pack 不是 core operation。target 可以根据 producer Encoding、all consumers、axis mapping、widening、reuse、pipeline 与 resources，为同一 canonical Value/use edge 选择 register window、local-storage panel、RVV tuple、IME operand 或其它 local pack。这个选择不得改变 Value 的 shape/axes、Level birth、effects 或 pin ABI。

```python
interleave(view, rows=16)
```

`interleave` 只允许出现在 derived Encoding builder 中，生成跨调用可见的 layout family。它改变 artifact bytes 与 ABI，因此属于 source semantics，而不是 invocation-local target 优化。

## 10. Build config 与 target requirement

core DSL 不提供 `@wide`、`@matrix`、`@transfer` 或通用 `stage_handoff`。`scalar/wide/matrix/transfer` 是 target physical machine 的 engine 类别。

如果 build 必须实际使用某项 target extension，可以声明硬 requirement：

```text
target = K1
require = uses_extension(IME)
```

requirement 是对 target physical program 的谓词。它先把不能满足要求的 physical structures 判为非法，再验证 selected program；不满足就拒绝 build，不静默 fallback。它不改变 canonical values，也不授权编译器改 source tree。普通 use-def 表达 value consumer，`materialize` 表达 logical staged boundary；register↔fragment 或 wide↔matrix 的交接由 physical conversion 表达。

Weft core DSL 当前不定义 soft-hint 表面。target schedule preference、cache policy 或 instruction preference若存在，应作为 build/target metadata附着到明确实体，不得进入 numerical operation 或 Value identity。

## 11. 函数边界

普通 inline 函数可以接收/返回 shaped Value、包含 Level 和普通控制、调用其它 std 函数、形成多个 consumer，并由类型和显式静态 source 参数重载。

函数不能隐藏 caller-visible persistent artifact、workspace 或 effect。需要它们时必须出现在 kernel 签名、View 类型或 derive builder 中。
