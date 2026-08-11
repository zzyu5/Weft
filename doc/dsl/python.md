# Python eDSL 与控制流

## 6. Python DSL 的地位

### 6.1 参考前端，不是运行时发射器

Weft Python DSL 是规范化参考 source frontend：

```text
Python Weft source
    → canonical Weft Kernel IR
    → RISC-V target lowering
    → intrinsic C / necessary inline asm
    → object / static library / header
```

Python 不参与生成物运行。运行期不得依赖 Python interpreter、LLVM JIT 或 Weft Python package。

其他前端可以直接生成同一个 canonical Weft Kernel IR，不经过 Python。

### 6.2 Kernel 定义

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
Entry 可以返回 `None` 或一个 scalar value；返回类型是 public C ABI 的一部分，不能由
emitter 根据 kernel 名或 use context 改写。

### 6.3 Kernel 参数类型

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

`W.constexpr[T]` 是 specialization-time meta-parameter。它与普通 runtime scalar 必须在类型上区分。

### 6.4 Pointer qualifier

Pointer 参数可以声明：

- address space；
- readonly / writeonly；
- noalias；
- minimum alignment；
- restrict-like ownership facts。

这些 qualifier 是作者承诺。编译器可以用其进行 vector memory、hoist、prefetch 和 local fusion；运行时违反承诺属于调用方错误。

概念语法：

```python
x: W.ptr[W.f32, W.readonly, W.noalias, W.aligned(64)]
```

具体 Python typing 形式可以调整，但 canonical IR 必须保存相同事实。

### 6.5 Compile-time meta-parameter

```python
@weft.kernel
def gemm_worker(..., BM: W.constexpr[W.index], BN: W.constexpr[W.index], BK: W.constexpr[W.index]):
    ...
```

Source 只声明 meta-parameter 的语义位置。候选值集合由 build specification 提供，不要求写进 kernel body。

禁止在 runtime data-dependent branch 中把 `W.constexpr` 当作普通运行时输入。

---

## 7. Python 控制流

### 7.1 Scalar range

```python
for row in W.range(row_begin, row_end):
    ...
```

`W.range` 是普通有序 scalar loop。作者写出的 carried scalar / block state 必须保持逻辑迭代顺序，除非它被显式改写成 reduce、scan 或 summary fold。

### 7.2 Scalar condition

Python `if` 条件必须是 scalar `i1`。

对 VLA / block predicate 必须使用：

```python
W.select(predicate, true_value, false_value)
```

或 validity-aware memory / structured primitive，不能把 vector predicate 当作 Python 控制流。

### 7.3 While 与 early exit

`W.while_` 或受限 Python `while` 可以表达有序状态机。其 carried state 默认不可重排、不可跨 iteration 并行。

### 7.4 Helper function

纯 helper 可以使用：

```python
@W.pure
def merge(a, b):
    ...
```

Effectful helper 必须显式声明 effect，并遵守 kernel ABI / region 限制。

任意 Python reflection、动态对象、文件 I/O、异常、generator 和运行时 monkey-patching 不属于 kernel language。

---
