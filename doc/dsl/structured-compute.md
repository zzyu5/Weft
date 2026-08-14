# Dot、Matmul 与 State Algebra

## Dot 与 matmul 授权点

Weft不提供任意轴tensor contraction。现有真实kernel只需要两种明确的局部乘加域：

```python
value = W.dot(lhs, rhs, init=acc, acc_dtype=W.f32,
              order="relaxed", math="native")

value = W.matmul(lhs, rhs, init=acc, acc_dtype=W.f32,
                 order="relaxed", math="native")
```

`W.dot`固定收缩双方最后一个logical block axis，当前合法shape为：

```text
[R,K]   x [K]     -> [R]
[VLA,K] x [K]     -> [VLA]
[R,K]   x [VLA,K] -> [VLA,R]
```

`W.matmul`固定表达local block relation：

```text
[M,K] x [K,N] -> [M,N]
```

VLA axis始终是free/batch axis，不能被dot或matmul缩并。跨VLA axis的聚合必须使用reduce、
scan或显式typed summary primitive。`init`与`acc_dtype`均为必填source semantics；init必须是scalar或与结果
同shape的accumulator value。Operand validity由masked value本身携带，不另设第二套where轴
描述。结果element type等于accumulator dtype。

普通scalar loop中的multiply/add是作者写下的有序carry，不会被自动识别成dot或matmul。
只有显式primitive才授权target重组其局部K domain。

`dot`/`matmul`的result是普通SSA value，而不是terminal microkernel result。它可以进入
pointwise、多个consumer、state update、loop carry、memory store或另一个合法primitive；`init`
也可以是跨作者循环存活的block accumulator。Realizer必须为这些use建立明确physical handoff，
不能以“必须紧接一个store”等精确closure作为该primitive的语义入口。

作者拥有：

- outer traversal与cache blocking；
- staging、recomputation和persistent packing；
- operand block、pointer/index、logical predicate与memory effect；
- source-visible accumulator及跨block carry；
- numerical order与math policy。

Target只在当前dot/matmul内部决定：

- LMUL与register microtile；
- multiple accumulators与K-unroll；
- load schedule、prefetch与local pipeline；
- RVV或矩阵extension realization；
- 不越过primitive边界的短生命周期packing/temporary。

Target不得创建source中不存在的outer loop，改变blocking/staging/persistent format，或从kernel
名、格式名、完整loop nest与普通SSA graph选择实现。

### Blocked GEMM 的授权边界

Blocked GEMM 的 M/N/K loop、BM/BN/BK、operand block、accumulator lifetime、staging、packing和
store都由作者显式写出。同一个worker可以持有accumulator跨K loop，并在外围control中顺序处理
多个M/N block。`W.matmul`只授权当前local block product的物理重组；它不能替作者创建outer
K loop、persistent repack或另一种GEMM traversal。这一持续worker-owned lifetime是Weft与典型
Triton program/CTA-owned result tile合同的核心差异。

## State algebra

Weft保留四种不同的state语义，不因底层实现可复用而合并。

### Reduce

```python
result = W.reduce(value, op="add", identity=0.0, where=True,
                  axis=None, order="relaxed", acc_dtype=W.f32)
```

Reduce只观察最终聚合状态。VLA中`axis=None`消去active VLA axis；block value必须显式选择
block axis。Masked或`where=false`元素等价于identity。

### Scan

```python
prefix = W.scan(value, op="add", identity=0.0, inclusive=True,
                where=True, segment_start=None,
                order="ordered", acc_dtype=W.f32)
```

Scan为每个logical position产生prefix；output order可观察。`segment_start`是独立语义，不能
从logical mask猜测。

### Typed summary

固定且可观察的summary语义必须使用显式typed primitive：

```python
value, coordinate = W.argmax(x, i,
                             tie="lowest_coordinate", order="relaxed")

maximum, scaled_sum = W.online_softmax_summary(
    x, math="native", order="preserve"
)
```

`argmax`只定义maximum、显式coordinate与最低coordinate tie；`online_softmax_summary`只定义
稳定的`(maximum, scaled_sum)`合并代数。二者不拥有surrounding traversal、memory、normalize
consumer或kernel ABI。Target不得从helper closure或普通SSA graph猜出这些primitive。

Weft不提供任意`lift/merge/finalize` summary fold；现有真实kernel没有证明这种泛化是核心
语言所需。新的summary只有在出现独立、完整且局部的可观察语义时才增加typed primitive。

### Sequential carry

```python
state = init
for i in W.range(begin, end):
    state = step(state, i)
```

普通loop carry按logical iteration order执行。Compiler不得将其替换为reduce、scan、typed summary
或dot/matmul，也不得根据代码形状猜测结合律。
