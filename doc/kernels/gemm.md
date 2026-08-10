# Worker-local Blocked GEMM

### 23.5 Worker-local blocked GEMM

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
                b_blk = W.load(b + k_rhs * ldb + n_idx, where=b_valid)

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
- block values 与 contraction axes 归 canonical semantics；
- `mr×nr`、LMUL、RVV microkernel 或 IME fragment 归 provider/compiler/tuner；
- 外部 runtime 决定每个 worker 的 `[m_begin,m_end)`。
