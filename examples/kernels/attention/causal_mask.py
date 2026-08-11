import weft
import weft.language as W


@weft.kernel
def causal_mask_f32(
    scores: W.ptr[W.f32],
    row_begin: W.index,
    row_end: W.index,
    cols: W.index,
    stride: W.index,
    queries: W.index,
    n_past: W.index,
) -> None:
    for row in W.range(row_begin, row_end):
        query = row % queries
        with W.vla(0, cols) as col:
            masked = col > n_past + query
            W.store(
                scores + row * stride + col,
                W.neg_inf(W.f32),
                where=masked,
            )
