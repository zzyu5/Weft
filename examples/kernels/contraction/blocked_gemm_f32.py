import weft
import weft.language as W


@weft.kernel
def gemm_f32_worker(
    a: W.ptr[W.f32, W.readonly, W.noalias],
    b: W.ptr[W.f32, W.readonly, W.noalias],
    c: W.ptr[W.f32, W.writeonly, W.noalias],
    m_begin: W.index,
    m_end: W.index,
    n: W.index,
    k: W.index,
    lda: W.index,
    ldb: W.index,
    ldc: W.index,
) -> None:
    for row in W.range(m_begin, m_end, 6):
        for column in W.range(0, n):
            row_lane = W.block_axis(6)
            inner = W.block_axis(k)
            row_index = row + row_lane[:, None]
            inner_index = inner[None, :]
            row_valid = row_index < m_end
            lhs = W.load(
                a + row_index * lda + inner_index,
                where=row_valid,
                other=W.f32(0.0),
            )
            rhs = W.load(b + column * ldb + inner, other=W.f32(0.0))
            value = W.dot(
                lhs,
                rhs,
                init=W.zeros((6,), dtype=W.f32),
                acc_dtype=W.f32,
                order="relaxed",
                math="native",
            )
            W.store(
                c + (row + row_lane) * ldc + column,
                value,
                where=row + row_lane < m_end,
            )
