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
scan或summary fold。`init`与`acc_dtype`均为必填source semantics；init必须是scalar或与结果
同shape的accumulator value。Operand validity由masked value本身携带，不另设第二套where轴
描述。结果element type等于accumulator dtype。

普通scalar loop中的multiply/add是作者写下的有序carry，不会被自动识别成dot或matmul。
只有显式primitive才授权target重组其局部K domain。

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
- 不越过primitive边界的短生命周期packing/scratch。

Target不得创建source中不存在的outer loop，改变blocking/staging/persistent layout，或从kernel
名、格式名、完整loop nest与普通SSA graph选择实现。

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

### Summary fold

```python
state = W.summary_fold(value, identity=identity,
                       lift=lift, merge=merge, finalize=finalize,
                       where=True, coordinate=None, order="preserve")
```

其语义为：

```text
lift(element[, coordinate]) -> state
merge(state_a, state_b)     -> state
finalize(state)             -> result
```

需要逻辑位置时，作者通过`coordinate=`显式传入与value同domain的坐标；target不能以lane ID、
pointer use或strip位置替代。

固定且可观察的summary语义使用typed primitive：

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

自定义`lift`、`merge`与`finalize`必须是pure helper，state type闭合。作者负责identity与声明
order下所需的结合性质；compiler检查类型、capture与effect purity，不建立证明系统。

### Sequential carry

```python
state = init
for i in W.range(begin, end):
    state = step(state, i)
```

普通loop carry按logical iteration order执行。Compiler不得将其替换为reduce、scan、summary
或dot/matmul，也不得根据代码形状猜测结合律。
