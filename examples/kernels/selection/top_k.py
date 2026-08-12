import weft
import weft.language as W

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
                state = W.argmax(
                    candidate,
                    column,
                    tie="lowest_coordinate",
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
                state = W.argmax(
                    candidate,
                    column,
                    tie="lowest_coordinate",
                    order="relaxed",
                )
            _, selected = state
            W.store(index_row + rank, W.cast(selected, W.u32))
