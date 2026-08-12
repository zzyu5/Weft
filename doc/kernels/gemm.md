# Worker-local Blocked GEMM

## Worker-local blocked GEMM

```python
@weft.kernel
def gemm_worker(
    a: W.ptr[W.f16],
    b: W.ptr[W.f16],
    c: W.ptr[W.f32],
    m_begin: W.index,
    m_end: W.index,
    n: W.index,
    k: W.index,
    lda: W.index,
    ldb: W.index,
    ldc: W.index,
    BM: W.constexpr[W.index],
    BN: W.constexpr[W.index],
    BK: W.constexpr[W.index],
) -> None:
    for m0 in W.range(m_begin, m_end, BM):
        for n0 in W.range(0, n, BN):
            acc = W.zeros((BM, BN), dtype=W.f32)

            for k0 in W.range(0, k, BK):
                mi = W.block_axis(BM)
                ni = W.block_axis(BN)
                ki = W.block_axis(BK)

                m_idx = m0 + mi[:, None]
                n_idx = n0 + ni[None, :]
                k_lhs = k0 + ki[None, :]
                k_rhs = k0 + ki[:, None]

                a_valid = (m_idx < m_end) & (k_lhs < k)
                b_valid = (k_rhs < k) & (n_idx < n)

                a_blk = W.load(a + m_idx * lda + k_lhs, where=a_valid)
                b_blk = W.load(b + n_idx * ldb + k_rhs, where=b_valid)

                acc = W.contract(
                    a_blk,
                    b_blk,
                    init=acc,
                    lhs_axes=(1,),
                    rhs_axes=(0,),
                    acc_dtype=W.f32,
                    order="relaxed",
                    math="native",
                )

            mi = W.block_axis(BM)
            ni = W.block_axis(BN)
            m_idx = m0 + mi[:, None]
            n_idx = n0 + ni[None, :]
            out_valid = (m_idx < m_end) & (n_idx < n)
            W.store(c + m_idx * ldc + n_idx, acc, where=out_valid)
```

这里：

- `m0/n0/k0`、BM/BN/BK 与 staging skeleton 归作者；
- `a` 的logical layout是 `[M,K]`，`b` 是供dot使用的 `[N,K]` row-major persistent layout；
- block values 与 contraction axes 归 canonical semantics；
- `mr×nr`、LMUL、RVV microkernel 或 IME fragment 归 target lowering 与构建期 tuning；
- 外部 runtime 决定每个 worker 的 `[m_begin,m_end)`。

## F32 row microtile contract

当前F32 workload保留另一种自然source结构：作者按6行遍历M、按单列遍历N，并用一个动态K
block表达6个row dot：

```python
for row in W.range(m_begin, m_end, 6):
    for column in W.range(0, n):
        row_lane = W.block_axis(6)
        inner = W.block_axis(k)
        lhs = W.load(a + (row + row_lane[:, None]) * lda + inner[None, :],
                     where=row + row_lane[:, None] < m_end,
                     other=W.f32(0.0))
        rhs = W.load(b + column * ldb + inner, other=W.f32(0.0))
        value = W.contract(lhs, rhs, init=W.zeros((6,), dtype=W.f32),
                           lhs_axes=(1,), rhs_axes=(0,),
                           acc_dtype=W.f32, order="relaxed", math="native")
        W.store(c + (row + row_lane) * ldc + column, value,
                where=row + row_lane < m_end)
```

Target从contract、block axis、typed operand以及pointer/access projection选择F32 RVV row
microtile与LMUL；当前row6选择LMUL4，row8选择LMUL1，同一个K vector供各row accumulator
复用。该选择不要求enclosing loop
形成固定row/column closure，所以同一local contract可以位于expert grouping等其他source
context中。`row step=6` 是当前source的cache/register blocking选择；它不是`gemm_f32`
kernel类别，也没有把N/K loop、grouping或matrix layout从target反推回IR。
