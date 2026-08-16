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
            mi = W.block(BM)
            ni = W.block(BN)
            acc = W.zeros((mi, ni), dtype=W.f32)

            for k0 in W.range(0, k, BK):
                ki = W.block(BK)

                m_idx = m0 + mi[:, None]
                n_idx = n0 + ni[None, :]
                k_lhs = k0 + ki[None, :]
                k_rhs = k0 + ki[:, None]

                a_valid = (m_idx < m_end) & (k_lhs < k)
                b_valid = (k_rhs < k) & (n_idx < n)

                a_blk = W.load(a + m_idx * lda + k_lhs, where=a_valid)
                b_blk = W.load(b + n_idx * ldb + k_rhs, where=b_valid)

                acc = W.matmul(
                    a_blk,
                    b_blk,
                    init=acc,
                    acc_dtype=W.f32,
                    order="relaxed",
                    math="native",
                )

            acc = acc + W.f32(0.0)
            m_idx = m0 + mi[:, None]
            n_idx = n0 + ni[None, :]
            out_valid = (m_idx < m_end) & (n_idx < n)
            W.store(c + m_idx * ldc + n_idx, acc, where=out_valid)
```

这里：

- `m0/n0/k0`、BM/BN/BK 与 staging skeleton 归作者；
- `mi/ni/ki`是三个不同 DSL axis；A/B显式共享同一个`ki`，accumulator显式拥有`[mi,ni]`；
- `a` 的logical relation是 `[M,K]`，`b` 是供dot使用的 `[N,K]` row-major relation；只有 DSL kernel
  使用`W.persistent(format)`显式声明的caller-provided packed object才具有persistent身份；
- block values 与matmul relation归canonical semantics；
- `mr×nr`、LMUL、RVV microkernel 或 IME fragment 归 target lowering 与构建期 tuning；
- 外部 runtime 决定每个 worker 的 `[m_begin,m_end)`。

## F32 row microtile dot

当前F32 workload保留另一种自然 DSL 结构：作者按6行遍历M、按单列遍历N，并用一个动态K
block表达6个row dot：

```python
for row in W.range(m_begin, m_end, 6):
    for column in W.range(0, n):
        row_lane = W.block(6)
        inner = W.block(k)
        lhs = W.load(a + (row + row_lane[:, None]) * lda + inner[None, :],
                     where=row + row_lane[:, None] < m_end,
                     other=W.f32(0.0))
        rhs = W.load(b + column * ldb + inner, other=W.f32(0.0))
        value = W.dot(lhs, rhs, init=W.zeros((row_lane,), dtype=W.f32),
                      acc_dtype=W.f32, order="relaxed", math="native")
        shifted = value + W.f32(0.0)
        scaled = value * W.f32(1.0)
        W.store(c + (row + row_lane) * ldc + column,
                W.maximum(shifted, scaled),
                where=row + row_lane < m_end)
```

Target先从dot的typed axis facts确定structure：lhs拥有VLA free axis时选择VLA vector-dot，rhs
拥有VLA free axis时选择VLA microtile，两侧均无VLA free axis时选择local-row microkernel；这不是
由example或外围loop名称给出的family标签。随后在该structure内构造F32 RVV LMUL/K-unroll实例。
每个row accumulator、展开后的streamed operands、predicate、state与per-consumer handoff都进入
register-group预算；不同VLEN、row tile和K relation因此可以选择不同LMUL或unroll，同一个K
vector仍供各row accumulator复用。该选择不要求enclosing loop
形成固定row/column producer shape，所以同一local dot可以位于expert grouping等其他 DSL
context中。`row step=6` 是当前 DSL kernel 的cache/register blocking选择；它不是`gemm_f32`
kernel类别，也没有把N/K loop、grouping或matrix layout从target反推回IR。
