import weft
import weft.language as W


@W.pure
def _winner(value, coordinate):
    return W.tuple(value, coordinate)


@W.pure
def _merge_winner(a, b):
    value_a, index_a = a
    value_b, index_b = b
    take_b = (value_b > value_a) | (
        (value_b == value_a) & (index_b < index_a)
    )
    return W.tuple(
        W.select(take_b, value_b, value_a),
        W.select(take_b, index_b, index_a),
    )


@weft.kernel
def top_k_f32(
    scores: W.ptr[W.f32, W.readonly, W.noalias],
    indices: W.ptr[W.u32, W.noalias],
    row_begin: W.index,
    row_end: W.index,
    columns: W.index,
    k: W.index,
    score_stride: W.index,
    index_stride: W.index,
) -> None:
    for row in W.range(row_begin, row_end):
        for rank in W.range(0, k):
            with W.vla(0, columns) as column:
                eligible = column >= W.index(0)
                for previous_rank in W.range(0, rank):
                    previous = W.cast(
                        W.load(
                            indices + row * index_stride + previous_rank,
                            other=W.u32(0),
                        ),
                        W.index,
                    )
                    eligible = eligible & (column != previous)
                candidate = W.select(
                    eligible,
                    W.load(scores + row * score_stride + column),
                    W.neg_inf(W.f32),
                )
                state = W.summary_fold(
                    candidate,
                    identity=W.tuple(W.neg_inf(W.f32), W.index(0)),
                    lift=_winner,
                    merge=_merge_winner,
                    coordinate=column,
                    order="relaxed",
                )
            _, selected = state
            W.store(
                indices + row * index_stride + rank,
                W.cast(selected, W.u32),
            )


@weft.kernel
def top_k_f32_equivalent(
    scores: W.ptr[W.f32, W.readonly, W.noalias],
    indices: W.ptr[W.u32, W.noalias],
    row_begin: W.index,
    row_end: W.index,
    columns: W.index,
    k: W.index,
    score_stride: W.index,
    index_stride: W.index,
) -> None:
    for row in W.range(row_begin, row_end):
        score_row = scores + row * score_stride
        index_row = indices + row * index_stride
        for rank in W.range(0, k):
            with W.vla(0, columns) as column:
                eligible = column >= W.index(0)
                for previous_rank in W.range(0, rank):
                    previous = W.cast(
                        W.load(index_row + previous_rank, other=W.u32(0)),
                        W.index,
                    )
                    eligible = (previous != column) & eligible
                candidate = W.select(
                    eligible,
                    W.load(score_row + column),
                    W.neg_inf(W.f32),
                )
                state = W.summary_fold(
                    candidate,
                    identity=W.tuple(W.neg_inf(W.f32), W.index(0)),
                    lift=_winner,
                    merge=_merge_winner,
                    coordinate=column,
                    order="relaxed",
                )
            _, selected = state
            W.store(index_row + rank, W.cast(selected, W.u32))
