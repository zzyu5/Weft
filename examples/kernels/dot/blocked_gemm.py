import weft
import weft.language as W


@weft.kernel
def gemm_worker(
    a: W.ptr[W.f16, W.readonly, W.noalias],
    b: W.ptr[W.f16, W.readonly, W.noalias],
    c: W.ptr[W.f32, W.writeonly, W.noalias],
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

            k0 = W.index(0)
            while k0 < k:
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
                k0 = k0 + BK

            acc = acc + W.f32(0.0)
            m_idx = m0 + mi[:, None]
            n_idx = n0 + ni[None, :]
            out_valid = (m_idx < m_end) & (n_idx < n)
            W.store(c + m_idx * ldc + n_idx, acc, where=out_valid)
