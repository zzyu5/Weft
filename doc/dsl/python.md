# Python eDSL 与控制流

## Python DSL 的地位

### 参考前端，不是运行时发射器

Weft Python DSL 是规范化参考前端：

```text
Weft Python DSL kernel
    → canonical Weft Kernel IR
    → RISC-V target lowering
    → intrinsic C / necessary inline asm
    → system C compiler / archiver
    → object / static library / generated C header
```

Python 不参与生成物运行。运行期不得依赖 Python interpreter、LLVM JIT 或 Weft Python package。

其他前端可以直接生成同一个 canonical Weft Kernel IR，不经过 Python。

### Kernel 定义

```python
import weft
import weft.language as W

@weft.kernel
def saxpy(
    x: W.ptr[W.f32],
    y: W.ptr[W.f32],
    a: W.f32,
    begin: W.index,
    end: W.index,
) -> None:
    with W.vla(begin, end) as i:
        xv = W.load(x + i)
        yv = W.load(y + i)
        W.store(y + i, a * xv + yv)
```

`@weft.kernel` 定义一个 worker-local entry。它不是 Python callable 的 eager 执行语义。
Entry 可以返回 `None` 或一个 scalar value；返回类型是 C ABI 的一部分，不能由
emitter 根据 kernel 名或 use context 改写。

### Kernel 参数类型

核心参数类型：

```python
W.i1
W.i8, W.i16, W.i32, W.i64
W.u8, W.u16, W.u32, W.u64
W.f16, W.bf16, W.f32, W.f64
W.index
W.ptr[T]
W.constexpr[T]
```

实现可以增加扩展 scalar type，但不得改变核心类型语义。

`W.index` 是地址和逻辑坐标使用的整数类型，不等于 physical lane index。

`W.constexpr[T]` 是 specialization-time meta-parameter。Kernel ABI 与 canonical IR 使用
`!weft_kernel.constexpr<T>` 与普通 runtime scalar 区分；kernel body 通过
`weft_kernel.meta_value` 读取其已绑定值。该值在 target lowering 前必须绑定，不会成为
生成 entry 的 runtime 参数。

### Pointer qualifier

Pointer 参数可以声明：

- storage class：external、persistent(format) 或 workspace；
- readonly / writeonly；
- noalias；
- minimum alignment；
- restrict-like ownership facts。

这些 qualifier 是作者承诺。编译器可以用其进行 vector memory、hoist、physical prefetch 和
local fusion；运行时违反承诺属于调用方错误。External 是默认 storage class；persistent 与
workspace 必须在 entry body 使用 `W.storage` 声明 shape，workspace 还必须声明 `W.noalias`。

概念语法：

```python
x: W.ptr[W.f32, W.readonly, W.noalias, W.aligned(64)]
packed: W.ptr[W.u8, W.readonly, W.persistent("q4_k_n16_k32_304b")]
scratch: W.ptr[W.f32, W.workspace, W.noalias]

W.storage(packed, (n_blocks, k_blocks, 304))
W.storage(scratch, (rows, columns))
```

`W.storage` 必须直接出现在 kernel entry body；它不是 allocation。所有 author-visible object 都
由 caller 分配并作为普通 pointer 参数传入。完整 ownership/lifetime 说明见
[Storage ownership 与 lifetime](storage-and-lifetime.md)。

### Compile-time meta-parameter

```python
@weft.kernel
def gemm_worker(..., BM: W.constexpr[W.index], BN: W.constexpr[W.index], BK: W.constexpr[W.index]):
    ...
```

DSL kernel 声明 meta-parameter 的语义位置与使用位置。候选值集合由外部 build loop 提供，
不写进 kernel body。基于 meta value 的 loop bound、shape 或 branch 是 specialization-time
结构；lowering 不得把未绑定 meta value 留成 runtime data-dependent 输入。


## Python 控制流

### Scalar range

```python
for row in W.range(row_begin, row_end):
    ...
```

`W.range` 是普通有序 scalar loop。作者写出的 carried scalar / block state 必须保持逻辑迭代顺序，除非它被显式改写成 reduce、scan 或typed summary primitive。

### Scalar condition

Python `if` 条件必须是 scalar `i1`。分支后继续使用的赋值必须在两个分支中都定义，且
两侧结果类型相同；frontend 将它们显式变成 `weft_kernel.if` 的 region results。

对 VLA / block predicate 必须使用：

```python
W.select(predicate, true_value, false_value)
```

或 validity-aware memory / structured primitive，不能把 vector predicate 当作 Python 控制流。

### While 与 early exit

受限 Python `while` 表达有序状态机，并 lowering 为 `weft_kernel.while`。条件必须是 scalar
`i1`，carried state 默认不可重排、不可跨 iteration 并行；`while ... else` 不属于语言。

### Helper function

纯 helper 使用：

```python
@W.pure
def merge(a, b):
    ...
    return value
```

Effectful helper 使用：

```python
@W.helper(effects=("read",))
def load_pair_sum(ptr):
    return W.load(ptr) + W.load(ptr + W.index(1))
```

允许声明的 effect 是 `read` 与 `write`。Helper 必须以一个 value return
结束，并且不接受Python参数/返回annotation；typed schema来自每个调用点。Frontend始终将helper
inline到caller。Typed summary使用自己的显式canonical primitive，不存在第二条helper-region语义。
Helper不是第二份IR，也不是运行时Python call。

任意 Python reflection、动态对象、文件 I/O、异常、generator 和运行时 monkey-patching 不属于 kernel language。
