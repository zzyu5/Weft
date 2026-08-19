import weft
import weft.language as W


@weft.kernel
def gemm_worker(
    a: W.ptr[W.f32, W.readonly, W.noalias],
    b: W.ptr[W.f16, W.readonly, W.noalias],
    c: W.ptr[W.f32, W.writeonly, W.noalias],
    a_f16: W.ptr[W.f16, W.workspace, W.noalias],
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
    W.buffer(a_f16, shape=(m_end, k))
    for row in W.range(m_begin, m_end):
        with W.vla(0, k) as inner:
            value = W.load(a + row * lda + inner)
            W.store(a_f16 + row * lda + inner, W.cast(value, W.f16))

    for m0 in W.blocks(m_begin, m_end, BM):
        for n0 in W.blocks(0, n, BN):
            mi = W.axis(BM)
            ni = W.axis(BN)
            acc = W.accumulator((mi, ni), W.f32, init=0.0)

            for k0 in W.pipeline(W.blocks(0, k, BK)):
                ki = W.axis(BK)

                m_idx = m0 + mi[:, None]
                n_idx = n0 + ni[None, :]
                k_lhs = k0 + ki[None, :]
                k_rhs = k0 + ki[:, None]

                a_valid = (m_idx < m_end) & (k_lhs < k)
                b_valid = (k_rhs < k) & (n_idx < n)

                a_blk = W.load(a_f16 + m_idx * lda + k_lhs, where=a_valid)
                b_blk = W.load(b + n_idx * ldb + k_rhs, where=b_valid)

                acc = W.gemm(
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
