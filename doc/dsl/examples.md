# DSL 示例

这些示例展示作者树的语义边界。它们不规定目标 LMUL、physical microtile、pack schema/extent、fragment、load form或pipeline。为突出关系，片段省略 import、decorator 与函数签名中可由上下文确定的静态符号声明；未定义的大写符号均为静态 shape/参数，不是运行时隐式变量。

## 1. Q4_K × Q8_K multi-output vec-dot

下面的程序在一个 16-output cohort 中计算 16 条共享同一 activation block 的 vec-dot，因此完整函数形状是 GEMV。局部数值树仍是 Q4_K × Q8_K vec-dot；scalar 版本去掉 output free axis，并直接接收 canonical `Q4_K`。二者不是由后端在同一 ABI 下互换的实现。

### 1.1 Encoding

```python
@weft.encoding
class Q4_K:
    layout = bitorder.lsb_first, byteorder.little
    elements = 256
    alignment = 1
    d:    f16
    dmin: f16
    sc:   u6[8]   @ joined(4, 2, 4, lo_first)
    m:    u6[8]   @ joined(4, 2, 4, lo_first)
    q:    u4[256] @ grouped(64) @ layered(32, lo_first)

@weft.encoding
class Q8_K:
    layout = bitorder.lsb_first, byteorder.little
    elements = 256
    alignment = 1
    ds:   f32
    q:    i8[256]
    bsum: i16[16]

@weft.derive
def Q4K_I(
    W: View[Q4_K, (N, K)], *, rows: static[int]
) -> View[Q4K_I[rows], (N, K)]:
    return interleave(W, rows=rows)
```

`Q4K_I[rows]` 是可由 std 函数静态生成的派生 family；导出 artifact 时 `rows=16` 被实例化并形成固定 layout identity。这里的 kernel ABI 因而接收 `Q4K_I[16]`，不是 canonical `Q4_K`，也不是仍可变化的抽象 family。

### 1.2 Kernel

```python
def q4k_gemv(
    W: View[Q4K_I[16], (N, K)],
    X: View[Q8_K, (K,)],
    Y: View[f32, (N,)],
):
    # static preconditions: N % 16 == 0, K % 256 == 0
    with L.rows(N, group=16) as nb:
        f32_acc = new(f32, [16], init=0)

        with L.blocks(K, extent=256) as kb:
            w = admit(W[nb, kb]) @ transfer
            x = admit(X[kb]) @ transfer
            i32_acc = new(i32, [16], init=0)

            with L.subs(extent=32) as s:
                # [16,32] × [32] -> [16,16] i16 partials
                p16 = mac_pairs(w.q[s], x.q[s], into=i16) @ wide
                # 消去 element-pair 轴，保留 16-output free axis
                sub_sum = reduce(widen(p16, i32), op="add", axis="k") @ wide
                i32_acc += sub_sum * i32(w.sc[s]) @ wide

            # bsum 粒度 16，m 粒度 32
            mins = fold2(x.bsum)
            min_term = dot(widen(w.m, i32), mins) @ wide

            # ds 属于当前 Q8_K block，必须在 block handoff 处参与
            f32_acc += f32(x.ds) * (
                f32(w.d) * f32(i32_acc)
                - f32(w.dmin) * f32(min_term)
            ) @ wide

        commit(f32_acc, Y[nb])
```

### 1.3 作者写下的知识

| 知识 | 代码位置 |
|---|---|
| min 支路与 q-product 支路分开 | `i32_acc` 与 `min_term` |
| min 支路读取预存 bsum | `fold2(x.bsum)` |
| 16/32 粒度错配 | `fold2` |
| element/sub/block 三层频率 | `mac_pairs`、`L.subs`、`L.blocks` |
| i16→i32→f32 位宽层次 | `into`、`widen`、显式 cast |
| 同一 Level 共同产生 16 个输出 | `L.rows(..., group=16)` |
| 持久跨行布局 | `Q4K_I[16]` derive 与 kernel 参数类型 |

目标编译器可以改变 `p16` 的物理 lane/register representation、unpack、归约树和指令；不能删除 min 支路、移动 `ds`、把 16-output cohort改成四个独立Level，或把 persistent layout换成另一个ABI。

### 1.4 三种容易混淆的程序

下列程序不是同一路径的三个后端选项：

```text
single vec-dot
    Q4_K[K] × Q8_K[K] → scalar

persistent blocked MUL_MAT/GEMV
    View[Q4K_I[16], (N,K)]，跨调用已重排，多输出cohort在kernel内

canonical local-pack MUL_MAT
    Q4_K[N,K]，每次invocation内materialize(pack(...))
```

它们的ABI、物化次数和Level树不同，必须分别由作者函数表达。

## 2. Blocked MUL_MAT

```python
def mul_mat(
    A: View[f32, (M, K)],
    B: View[f32, (K, N)],
    C: View[f32, (M, N)],
):
    with L.tiles(N, extent=auto("NC")) as nc:
        with L.tiles(K, extent=auto("KC")) as kc:
            # 每个(NC,KC) panel一次，被所有MC复用
            Bp = materialize(pack(B[kc, nc], along="k")) @ transfer

            with L.tiles(M, extent=auto("MC")) as mc:
                # 每个(MC,KC) panel一次
                Ap = materialize(pack(A[mc, kc], along="k")) @ transfer

                with L.rows(mc, group=auto("MR")) as mb:
                    with L.cols(nc, group=auto("NR")) as nb:
                        # acc位于KC内部：每个KC从C读入并写回
                        acc = new(f32, [MR, NR], init=admit(C[mb, nb]))

                        with L.blocks(kc, extent=auto("KB")) as kb:
                            a = admit(Ap[mb, kb]) @ transfer
                            b = admit(Bp[kb, nb]) @ transfer
                            acc += outer_contract(a, b, over="k") @ wide

                        commit(acc, C[mb, nb])
```

### 2.1 KC 不能省略

`Bp` 在 KC 层诞生，作用域是 `KC × NC`，被该 KC 下所有 MC 复用。没有 KC 层时，无法仅靠注释表达 panel size、物化频率和C的分段累加。

`along="k"` 规定 K 是 local pack 的连续供应方向；它不固定 vector window、register tuple、fragment operand 或 local-storage tile 的具体形状。后者由 target compiler 根据 A、B operands 的 producer mapping、所有 consumer、资源与 engine role 选择。

### 2.2 Accumulator 作用域是数值决策

上面的 accumulator 位于 KC 内：

```text
每个KC：read C → accumulate current KC → write C
```

若把 accumulator 移到 KC 外：

```text
read/init once → accumulate entire K → write once
```

两者的 C traffic、state lifetime 和浮点累加顺序不同，是两棵作者程序。编译器不能根据cache或寄存器压力在两者之间切换。

量化 MUL_MAT 可以复用 NC/KC/MC/MR/NR 外层骨架，但内层量化树、activation quantize、workspace和persistent/local packing必须由对应std函数显式写出；它们不是dense `outer_contract` 的隐式后端模式。

## 3. GEMV 是 blocked MUL_MAT 的退化

```python
def gemv(
    W: View[f32, (M, K)],
    X: View[f32, (K,)],
    Y: View[f32, (M,)],
):
    with L.rows(M, group=auto("MR")) as mb:
        acc = new(f32, [MR], init=0)
        with L.blocks(K, extent=auto("KB")) as kb:
            # X block被当前MR行共同使用
            x = admit(X[kb]) @ transfer
            acc += contract(admit(W[mb, kb]), x, over="k") @ wide
        commit(acc, Y[mb])
```

相对 GEMM：

- 没有 N/column Level；
- accumulator 从 `[MR,NR]` 降为 `[MR]`；
- X block 在 K Level 供应一次，被 MR 行共享；
- 没有因为名字叫GEMV而引入新primitive。

若作者改成 `group=1` 并把 `admit(X)` 放进每行内部，就是朴素row-dot：X被读取M次。编译器不能把后者识别成前者。

## 4. Online attention 与非归约 handoff

```python
def rowmax(x):
    return reduce(x, op="max", axis="tk")

def rowsum(x):
    return reduce(x, op="add", axis="tk")

def flash_attention(
    Q: View[f16, (Tq, D)],
    K: View[f16, (Tk, D)],
    V: View[f16, (Tk, D)],
    O: View[f16, (Tq, D)],
):
    with L.rows(Tq, group=auto("BQ")) as qb:
        q = materialize(admit(Q[qb, :])) @ transfer
        m = new(f32, [BQ], init=-inf)
        l = new(f32, [BQ], init=0)
        o = new(f32, [BQ, D], init=0)

        with L.blocks(Tk, extent=auto("BK")) as kb:
            k = admit(K[kb, :]) @ transfer
            v = admit(V[kb, :]) @ transfer

            s = contract(q, k, over="d", acc=f32) @ wide
            m_new = maximum(m, rowmax(s)) @ wide
            p = exp(s - m_new) @ wide
            alpha = exp(m - m_new) @ wide

            l = l * alpha + rowsum(p) @ wide
            o = o * alpha + contract(p, v, over="tk", acc=f32) @ wide
            m = m_new

        out = narrow(o / l, f16) @ wide
        commit(out, O[qb, :])
```

这份程序确立：

- `q` 是 qb 层 staged value；
- `m/l/o` 是 qb 层 state；
- Tk block顺序来自 old state→new state的SSA依赖；
- `o = o * alpha + contribution` 是先重标定再合并，不是普通 `+=`；
- 一个 Level 可包含两个 contract、多个reduce/pointwise和三个state update。

编译器不能把普通 softmax换成online算法，也不能把该handoff简化成未重标定的累加。

## 5. Top-K 明确不用 Level

```python
def topk(X: View[f32, (N,)], out: View[i32, (K_,)]):
    # static precondition: 1 <= K_ <= N
    heap = new(f32, [K_], init=-inf)
    idx = new(i32, [K_], init=-1)

    for i in range(N):
        score = X[i]
        if score > heap[K_ - 1]:
            pos = K_ - 1
            j = K_ - 1
            while j > 0:
                if score > heap[j - 1]:
                    pos = j - 1
                j -= 1

            for j in range(K_ - 1, pos, -1):
                heap[j] = heap[j - 1]
                idx[j] = idx[j - 1]

            heap[pos] = score
            idx[pos] = i32(i)

    commit(idx, out)
```

这个算法虽然有静态的 N/K 边界，但作者程序是一条数据依赖的有序 insertion control；它没有声明 shaped domain/cohort，也不需要跨Level birth或typed handoff，因此普通`for/while/if`就是完整表达。相等分数不触发插入，所以当前程序的 tie policy 是保留先出现的元素；NaN 比较为 false，因此 NaN 不进入结果，全 NaN 输入保留初始化的 `-1` indices。

编译器不会把循环自动向量化。宽Top-K必须由作者或std另写一棵包含显式shaped axis/Level的程序，不能从这份标量tree恢复。
