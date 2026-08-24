# 函数、基本 Operation 与 Engine Role

## 1. 一门语言、一个用户层级

Weft 的 std 函数与应用 kernel 使用同一门语言：

```text
没有 privileged library IR
没有 opaque whole-kernel template
没有“普通用户 DSL / 专家 DSL”两套语义
```

std 中的 kernel helper 是可读、可复制、可修改的普通函数。它们的调用在 canonical Kernel IR形成前 inline；函数内部的 Level、Value、Encoding access 和 operation 与调用者组成一棵 SSA 程序。`@weft.derive` builder 是例外：它在 artifact phase 单独求值并产生派生 Encoding，不进入 invocation kernel 的 SSA 树。

语言地位相同不等于库中只有一个实现。库作者可以维护多个 overload/特化。下面代码只表示签名关系，不要求字面采用 Python 的运行时重载语法：

```python
gemv(f32, f32)                         # dense wide tree
gemv(Q4K_I[16], Q8_K)                  # packed wide tree
gemv(Q4K_I[16], Q8_K, engine=matrix)   # matrix-role tree
```

这些实现都由人写。前端根据类型、显式 config 和 overload 规则选中；若实现公开了 source `auto` 参数，构建过程只枚举作者声明的有限参数绑定。编译器不会从普通 multiply/add 图发明另一棵 std tree，也不会生成多棵结构后竞赛。

## 2. 没有内置 GEMM

语言核心没有 `matmul` kernel primitive。GEMM、GEMV、quantized vec-dot、attention 和 sort 都是 std 中的普通函数。

GEMM 使用：

```text
Level hierarchy
pack/materialize/admit/commit
new accumulator
outer_contract
```

GEMV 使用同一组构造，去掉 N/column Level 并把 accumulator 降为一维。

这条规则避免 `matmul` primitive 暗中拥有：

- M/N/K outer traversal；
- cache blocking；
- accumulator 作用域；
- staging 与 persistent packing；
- kernel ABI。

`outer_contract` 只定义局部数值关系，不接管完整算子。

## 3. `auto`

```python
with L.tiles(N, extent=auto("NC")) as nc:
with L.rows(mc, group=auto("MR")) as mb:
```

`auto` 声明一个由构建/调优过程实例化的源参数。它不是“允许后端随便修改”的 permission bit，也不是 target 结构搜索入口。

`auto("MR")` 在包含它的 std/kernel 函数中引入同名静态符号 `MR`；同一作用域内重复出现的同名 `auto` 必须绑定同一个值。每个名字都必须由 std overload 或显式 build config 给出非空、有限的合法值域；缺少值域或跨作用域重名歧义是前端错误。

每组具体绑定：

```text
NC=64, KC=256, MR=4, NR=16
```

形成一份已实例化 candidate。对该 candidate：

- Level tree、逻辑 value、births 和 handoff 固定；
- 编译器只物理化该树；
- 资源不足时 spill 或拒绝；
- 不回头修改 `MR/NR/KC`。

调优器可以枚举作者为 `auto` 声明的有限取值并实测完整 candidate。合法取值域由 std overload 或显式 build config 提供，不由 target pass 根据 kernel shape 发明。它不搜索未声明的 Level topology，也不与 target compiler 自己生成的其它结构比较。

target lowering 还可以为已经确定的物理结构公开另一组 finite physical parameters，例如 LMUL、schema 内 physical microtile extent、unroll、pipeline depth 和 buffer count。它们不进入 DSL 或 canonical IR；构建期 tuner 可以实测这些参数绑定。source `auto` 与 physical parameters 可以由同一工具测量，但前者实例化作者程序，后者只实例化同一 source tree 的机器表示。

## 4. 基本 op 的要求

每个基本 op 必须唯一规定：

- operand/result 的 element type；
- result logical shape 和 axis relation；
- 被消去或保留的轴；
- 数值顺序、结合和精度语义；
- overflow/wrap/saturate 行为；
- 允许的 engine role；
- memory/effect 行为。

编译器不能从普通 SSA closure 识别一个基本 op。需要 primitive 语义时，作者显式调用它。

## 5. Pointwise 与类型 operation

```text
unary        neg / abs / exp / ...
binary       add / sub / mul / div / maximum / minimum / bitwise
compare      typed comparison
cast         数值转换或明确 reinterpret
widen        保持逻辑元素与轴，提升 element width
narrow       保持逻辑元素与轴，缩窄 element width
```

Pointwise operand 按 axis identity 对齐。缺少某轴的 scalar/singleton operand可以 broadcast；两个同名轴 extent不一致是类型错误，而不是后端选择。

`widen` 和 `narrow` 的位置是 canonical 数值语义。编译器可以改变其物理指令，但不能把转换移过会改变 overflow或舍入的位置。

浮点 `maximum/minimum` 使用 IEEE 754 `maximumNumber/minimumNumber` 语义：单个 NaN 与数字比较时返回数字，两个 NaN 返回 NaN；`maximum(+0,-0)=+0`，`minimum(+0,-0)=-0`。整数版本使用通常的有符号/无符号全序。目标不能用另一种 NaN 或 signed-zero 规则替代。

## 6. Grouped multiply-accumulate

```python
mac_pairs(a, b, into=i16)
```

每两个相邻逻辑元素的乘积相加，产生一个 `into` 类型结果：

```text
[n] × [n] → [n/2]
```

```python
mac_groups(a, b, n=4, into=i16)
```

每 n 个相邻元素的乘积相加：

```text
[N] × [N] → [N/n]
```

grouped result 继承被分组轴的 identity，只把该轴 extent 除以 group width；因此后续 `reduce(..., axis="k")` 仍引用同一逻辑轴。operand extent 必须能被 group width 整除，否则 verifier 拒绝。group width 和 `into` 改变 canonical result shape、partial 数量与 overflow 行为，属于作者选择。RVV widening MAC、普通 multiply+reduce 或专用 extension 是同一 primitive 的物理 realization。

## 7. Reduction 与 contraction

### 7.1 `reduce`

```python
reduce(value, op="add", axis="k")
```

消去指定逻辑轴，保留其它 free axes。logical reduction axis 必须非空；physical tail 只补该 op 的 identity，不增加 logical elements。

当前基本 reduction 的语义是：integer `add` 使用声明的 accumulator overflow 规则，identity 为 0；floating `add` 在该 closed primitive 内允许改变括号次序，但不允许把 operation 移出 reduction/Level 边界，identity 为 `+0`；`max/min` 使用与 pointwise `maximum/minimum` 相同的 NaN 与 signed-zero 规则，identity 分别为该类型的负/正无穷或最小/最大有限值。每个新增 reduction op 都必须定义自己的 identity、空域政策和结合自由度，不能继承一个全局 `ordered` 或 `fast-math` 开关。

源程序只有一个 reduce result；编译器可以在 closed primitive 内部使用多个物理 partial、lane collective 或树归约。这些物理 temporary 不成为 canonical logical values。

### 7.2 `fold2`

```python
fold2(x)
```

相邻两项相加，并把结果提升到 `i32`。输入最后轴必须能被 2 整除，否则 verifier 拒绝：

```text
T[n] → i32[n/2]
```

它改变逻辑 value shape，因此是显式 op。Q4_K 用它处理 Q8_K `bsum` 粒度 16 与 scale/min 粒度 32 的错配。

### 7.3 `dot`

```python
dot(a, b)
```

逐元素相乘并消去 operands 的最后一个共同轴；其它 free axes 按 identity 保留。需要显式选择其它 reduction axes 时使用 `contract(..., over=...)`。`dot` 不拥有外围 traversal、staging 或 accumulator lifetime。

### 7.4 `contract` / `outer_contract`

```python
contract(a, b, over="k", acc=f32)
outer_contract(a, b, over="k", acc=i32)
```

`contract` 定义两个 shaped values 在指定 reduction axes 上的缩并；`outer_contract` 显式保留两侧不同的 free axes，形成 outer-product式 result。

它们只拥有局部 operand/result numerical relation。M/N/K Level、packing、state和commit属于调用者程序。

## 8. Lookup 与显式 index domain

```python
indices = iota(8, dtype=u32)
values = lookup(codebook, base + indices)
```

`lookup` 接收只读 table/View 与 unsigned logical indices。对每个 index logical coordinate，operation 读取对应 table entry；结果的 shape 和 axes 必须与 indices 完全相同，table entry axis 被索引消去，不会凭字段名进入 result。越界 policy、mask 与 fill 必须由 operation 调用显式给出，不能由 target 猜测。

八个独立 scalar lookup 不等于一个 `[8]` lookup。目标编译器只能为显式 shaped index选择 unit/indexed/gather或专用 table realization。

例如 indices 是 `[M,G]`，则 lookup result 也是 `[M,G]`；后续沿 G reduce 得到 `[M]`。target可以把 G 映射到 lane、把 M 映射到 register replica，但不能消掉 M，也不能把八个独立 scalar lookup 恢复成 G axis。

## 9. Pack 与 interleave

```python
pack(view, along="k")
```

在 invocation 内建立 ephemeral pack 请求；result 的 logical values、shape 与 axes 和源 View 对应。作者决定：

- 是否建立 pack；
- `along` 指定哪条 logical axis 应成为局部连续供应方向；
- pack 所在的 Level、物化次数和可见 lifetime；
- 包围该 operation 的 engine role。

作者不规定 physical pack schema 或 schema 内 extents。target compiler 根据 `along`、producer storage mapping、所有 consumer representation、widening、decode/compute fusion、pipeline buffer、资源和跨 engine handoff，按 target 的固定规则与优先级选择 schema；tuner只实例化 schema 内有限数值参数。该表示可以形成 vector window、register tuple、fragment operand 或 invocation-local storage，但不能改变 logical shape、Level 归属或 pin ABI。

`materialize(pack(...))` 可以供声明 Level 的多个后代 operation 复用，不能越过 pin boundary。具体 register/cache/stack representation 不进入 DSL。

```python
interleave(view, rows=16)
```

只允许出现在 derived encoding builder中，生成跨调用可见的派生layout family。它改变artifact，不能作为普通kernel内的隐式优化。

### 9.1 `handoff`

```python
handoff(value)
```

`handoff` 保持 Value 的 element type、shape 和 axes不变，只把一个已 materialize 的数值边界显式交给另一 engine role 的 operation；canonical IR 将它保存为 `stage_handoff`。它不执行 decode、copy 或同步，也不改变 Encoding。

## 10. Engine role

四个角色是语言级硬约束：

```text
scalar      标量数值、地址与有序控制能力
wide        宽向量能力
matrix      矩阵/张量扩展能力
transfer    数据搬运能力
```

使用方式：

```python
p = mac_pairs(a, b, into=i16) @ wide
panel = materialize(pack(B, along="k")) @ transfer
acc += contract(a, b, over="k") @ matrix
```

role 不是 hint：

- 没有 role annotation 表示作者没有增加 engine 约束；目标只能在该 op 语义本来允许的角色中选择；
- `@wide` 不能被目标静默落到 matrix engine；
- `@matrix` 不能退回 wide 以假装支持；
- `@transfer` 不要求独立 DMA，普通 load/store也可以实现该能力；
- target没有合法 realization时，当前candidate明确不支持。

因此“未标注”和“标注后允许后端忽略”不是同一回事。前者保留 op 定义本来拥有的合法角色集合；后者不存在。

### 10.1 使用 IME

应用作者不调用 IME intrinsic，也不在 DSL 中写 fragment shape。库提供一棵使用 `@matrix` 的 std tree：

```python
acc += contract(panel, activation, over="k", acc=i32) @ matrix
```

调用者通过参数类型、derived encoding和显式engine config选择这棵tree。目标编译器在 `matrix` role 内选择IME fragment、local pack、指令与handoff；若operand/layout不满足，candidate失败。只有未绑定 role 且 operation 本身同时允许多个 role 时，target 才可以按固定结构优先级在 wide 与 matrix realization 之间选择。

同名的 `@wide` tree与`@matrix` tree是两个作者程序，不是后端把一个role自动换成另一个。

## 11. 函数边界

普通 inline函数可以：

- 接收/返回 shaped Value；
- 包含Level和普通控制；
- 调用其它std函数；
- 形成多个consumer；
- 由类型和显式静态参数重载。

函数不能隐藏caller可观察的 persistent artifact、workspace或effect。需要它们时必须出现在kernel签名、View类型或derive builder中。
