# Canonical Weft Kernel IR

## 唯一算法表示

Canonical Kernel IR 保存完整 worker-local algorithm：

- kernel symbol、普通 C ABI、argument name/kind与scalar return；
- scalar、typed pointer与specialization-time constexpr parameter；
- scalar `if` / `for` / `while` control与carried state；
- 一个 active VLA axis、logical block axis与region value；
- pointer/index、logical predicate、masked value与memory effect；
- pointwise、conversion与special value；
- reduce、scan、summary fold与sequential carry的observable distinction；
- contract、lookup、decode、permute与typed local extension primitive；
- source meta value、source location与numerical attributes。

它不得保存 exact `vl`、LMUL、register number、register microtile、target realization ID、
IME fragment、instruction spelling、build measurement、thread count或launch policy。

## 持久类型

```text
!weft_kernel.ptr<T, address-space, access, noalias, alignment, restrict>
!weft_kernel.constexpr<T>
!weft_kernel.block<[D0, D1, ...], T>
!weft_kernel.region<[-1, D0, D1, ...], T>
!weft_kernel.masked<ValueType>
!weft_kernel.tuple<[T0, T1, ...]>
```

- `ptr` 保存作者声明的element、address space、access与alias/alignment facts；
- `constexpr` 只用于entry ABI，body通过 `meta_value` 读取已绑定值；
- `block` 是零个或多个logical block axes，不是register microtile；
- `region` 的首个 `-1` 表示当前active VLA axis；
- `masked` 携带first-class logical validity，不能当普通value逃逸；
- `tuple` 是reduce/summary/control carried state使用的闭合heterogeneous value。

Dynamic logical block extent在shape中写为 `-1`，真实extent仍是对应op的显式operand。两个
dynamic extent不能仅因都打印为 `-1` 就被视为同一logical identity。

## Canonical op families

下面是可parse的核心mnemonic，而不是按kernel类型划分的catalog：

```text
entry/control
  weft_kernel.kernel / return / yield / condition
  weft_kernel.if / for / while / vla

value/domain
  weft_kernel.constant / meta_value / block_axis / full
  weft_kernel.expand_dims / broadcast_to / reshape / transpose
  weft_kernel.tuple / tuple_get / special_value / invalid

scalar, block and region compute
  weft_kernel.ptr_add / unary / binary / compare / select
  weft_kernel.cast / bitcast / widen / narrow

memory and validity
  weft_kernel.load / store / prefetch / atomic_add / fence
  weft_kernel.valid / fill

state and structured compute
  weft_kernel.reduce / scan / summary_fold / contract
  weft_kernel.permute / lookup / decode
```

不存在 `weft_kernel.range`、`weft_kernel.mask` 或generic `weft_kernel.atomic` op：Python
`W.range` 生成 `weft_kernel.for`；logical validity由 `masked` type、`valid`和`fill`表达；
当前atomic语义是显式 `atomic_add`。

## Region 与 entry 边界

- 一个 module可以包含多个kernel symbol，但每个 `weft_kernel.kernel` 是独立worker-local
  entry；CLI选择symbol不参与target realization选择。
- entry argument kind只能是 `pointer`、`scalar` 或 `constexpr`；return为none或一个scalar。
- VLA不能嵌套；active VLA coordinate是 `region<[-1], index>`，保留VLA axis的value不能逃出
  lexical region。
- `summary_fold` 的lift/merge/finalize region必须闭合、pure并以 `yield` 终结。
- `contract` 的paired axes、where footprint、init/result shape与dtype必须由IR显式保存并局部
  verify。

## Extension dialect

新的可观察局部语义可以放在编译时已链接的sibling dialect。当前正式输入注册：

```text
weft_kernel
weft_ext
```

`weft_ext` 当前包含 `affine_i4_i8_contract` 与 `grouped_affine_i4_i8_dot`。Extension op必须
能随module独立parse/verify；它只表达typed local numerical relation，不能持有public ABI、
persistent pointer layout、outer traversal或target fragment。

## Python frontend

Python frontend直接从 `@weft.kernel` source构造上述IR，没有长期typed Python IR、selection
IR或第二份algorithm authority。`@W.helper` 默认inline；只有需要一等region semantics的
helper（例如summary lift/merge/finalize）进入canonical region。
