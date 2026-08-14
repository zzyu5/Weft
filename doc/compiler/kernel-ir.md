# Canonical Weft Kernel IR

## 唯一算法表示

Canonical Kernel IR 保存完整 worker-local algorithm：

- kernel symbol、普通 C ABI、argument name/kind与scalar return；
- scalar、typed pointer与specialization-time constexpr parameter；
- external/persistent/workspace ownership、persistent format identity与source-visible storage shape；
- scalar `if` / `for` / `while` control与carried state；
- 一个 active VLA axis、logical block axis与region value；
- pointer/index、logical predicate、masked value与memory effect；
- pointwise、conversion与special value；
- reduce、scan、typed summary与sequential carry的observable distinction；
- dot、matmul、lookup、decode与typed local extension primitive；
- source meta value、source location与numerical attributes。

它不得保存 exact `vl`、LMUL、register number、register microtile、target realization ID、
IME fragment、instruction spelling、build measurement、thread count或launch policy。

## 持久类型

```text
!weft_kernel.ptr<T, address-space, access, noalias, alignment, restrict,
                 storage-class, storage-format>
!weft_kernel.constexpr<T>
!weft_kernel.block<[D0, D1, ...], T>
!weft_kernel.region<[-1, D0, D1, ...], T>
!weft_kernel.masked<ValueType>
!weft_kernel.tuple<[T0, T1, ...]>
```

- `ptr` 保存作者声明的element、address space、access、alias/alignment、external/persistent/
  workspace class与persistent format identity；
- `constexpr` 只用于entry ABI，body通过 `meta_value` 读取已绑定值；
- `block` 是零个或多个logical block axes，不是register microtile；
- `region` 的首个 `-1` 表示当前active VLA axis；
- `masked` 携带first-class logical validity，不能当普通value逃逸；
- `tuple` 是reduce/summary/control carried state使用的闭合heterogeneous value。

Persistent/workspace pointer还必须由entry block中唯一的`weft_kernel.storage`绑定shape与显式
extent operands；external pointer禁止该op。Source-visible三类storage都是caller-provided entry
pointer。Compiler-private primitive temporary不进入canonical type/op/ABI。

这些类型组成唯一的canonical value system。Sibling extension dialect不得重新定义block、region、
masked validity、extent identity或state容器；extension operand必须直接使用上述类型和相同的
use-def/effect规则。`weft_ext`是局部semantic primitive的命名空间，不是第二份execution IR。

Dynamic logical block extent在shape中写为 `-1`，真实extent仍是对应op的显式operand。两个
dynamic extent不能仅因都打印为 `-1` 就被视为同一logical identity。

## Canonical op families

下面是可parse的核心mnemonic，而不是按kernel类型划分的catalog：

```text
entry/control
  weft_kernel.kernel / return / yield / condition
  weft_kernel.if / for / while / vla

storage
  weft_kernel.storage

value/domain
  weft_kernel.constant / meta_value / block_axis / full
  weft_kernel.expand_dims
  weft_kernel.tuple / tuple_get / special_value / invalid

scalar, block and region compute
  weft_kernel.ptr_add / unary / binary / compare / select
  weft_kernel.cast / bitcast / narrow

memory and validity
  weft_kernel.load / store

state and structured compute
  weft_kernel.reduce / scan / argmax / online_softmax_summary / dot / matmul
  weft_kernel.sort_indices / lookup / decode
```

`expand_dims`只承载Python singleton-axis slicing的canonical view，不是public任意shape-transform
入口。不存在 `weft_kernel.range`、`weft_kernel.mask`、generic atomic或source prefetch op：Python
`W.range`生成`weft_kernel.for`；logical validity由`masked` type传播，ordinary consumer需要
filled value时必须由source在load处提供`other`。Physical prefetch只属于Realizer
schedule；当前VLA memory effect要求lane independence。

## Region 与 entry 边界

- 一个 module可以包含多个kernel symbol，但每个 `weft_kernel.kernel` 是独立worker-local
  entry；CLI选择symbol不参与target realization选择。
- entry argument kind只能是 `pointer`、`scalar` 或 `constexpr`；return为none或一个scalar。
- persistent/workspace pointer的storage contract必须直接位于entry block，shape必须为正并可由
  constant、bound meta或runtime index ABI事实求得；workspace必须noalias。
- primitive对workspace的访问extent必须与storage extent具有可证明identity；例如
  `sort_indices`的rank-one u32 scratch必须与排序extent相同。
- 第二个active VLA region不能嵌套；普通scalar `for` / `while` / `if`可以位于VLA body中。
  Active VLA coordinate是 `region<[-1], index>`，保留VLA axis的value不能逃出lexical region。
- `argmax`与`online_softmax_summary`保存完整局部summary语义；target不得从普通SSA graph猜出。
- `dot`固定收缩双方最后一个logical block axis；`matmul`固定表达`[M,K] x [K,N]`。
  两者的init/result shape、accumulator dtype与numerical policy必须由IR显式保存并局部verify。

因此canonical kernel可以统一描述为：

```text
ordered control regions
+ zero or one active VLA axis per lexical scope
+ scalar / block / region SSA values with explicit extent and validity
+ pointer/index relations and memory effects
+ explicit local semantic primitives
```

Reduce、scan、summary、dot/matmul与extension op不是独立执行路径；它们是在上述region/value/effect
模型中授予某一局部domain重组权的anchor。其operand与result仍是普通SSA value，可以multi-use、
进入pointwise/state/memory或作为control carry。普通SSA或loop carry本身没有局部domain重组授权。

## Extension dialect

新的可观察局部语义可以放在编译时已链接的sibling dialect。当前正式输入注册：

```text
weft_kernel
weft_ext
```

`weft_ext` 当前包含 `affine_i4_i8_contract`、`symmetric_i4_i8_contract`、
`grouped_affine_i4_i8_dot`、`sign_bit_i8_dot`、`e2m1_e8m0_i8_dot`、`iq2_s_i8_dot`、
`iq3_s_i8_dot`、`iq1_m_i8_dot`与`q6_k_i8_dot`。Extension op必须
能随module独立parse/verify；它只表达typed local numerical relation，不能持有public ABI、
persistent pointer format、outer traversal或target fragment。它必须复用`weft_kernel`的logical
value queries与validity规则；若primitive没有定义masked语义，source必须在operand producer处
构造普通filled value，extension verifier不能静默unwrap。

## Python frontend

Python frontend直接从 `@weft.kernel` source构造上述IR，没有长期typed Python IR、selection
IR或第二份algorithm authority。`@W.helper` 默认inline；只有需要一等region semantics的
helper（例如summary lift/merge/finalize）进入canonical region。
