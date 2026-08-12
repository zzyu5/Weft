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
def top_p_nucleus_f32(
    probabilities: W.ptr[W.f32, W.readonly, W.noalias],
    sorted_indices: W.ptr[W.u32, W.noalias],
    sorted_probabilities: W.ptr[W.f32, W.noalias],
    uniforms: W.ptr[W.f32, W.readonly, W.noalias],
    nucleus_count: W.ptr[W.u32, W.writeonly, W.noalias],
    sampled_tokens: W.ptr[W.u32, W.writeonly, W.noalias],
    rows: W.index,
    vocabulary: W.index,
    threshold: W.f32,
) -> None:
    for row in W.range(0, rows):
        row_offset = row * vocabulary
        for rank in W.range(0, vocabulary):
            with W.vla(0, vocabulary) as token:
                eligible = token >= W.index(0)
                for previous_rank in W.range(0, rank):
                    previous = W.cast(
                        W.load(
                            sorted_indices + row_offset + previous_rank,
                            other=W.u32(0),
                        ),
                        W.index,
                    )
                    eligible = eligible & (token != previous)
                candidate = W.select(
                    eligible,
                    W.load(probabilities + row_offset + token),
                    W.neg_inf(W.f32),
                )
                state = W.summary_fold(
                    candidate,
                    identity=W.tuple(W.neg_inf(W.f32), W.index(0)),
                    lift=_winner,
                    merge=_merge_winner,
                    coordinate=token,
                    order="relaxed",
                )
            selected_probability, selected_index = state
            W.store(
                sorted_indices + row_offset + rank,
                W.cast(selected_index, W.u32),
            )
            W.store(
                sorted_probabilities + row_offset + rank,
                selected_probability,
            )

        with W.vla(0, vocabulary) as rank:
            prefix = W.scan(
                W.load(sorted_probabilities + row_offset + rank),
                op="add",
                identity=W.f32(0.0),
                inclusive=True,
                order="ordered",
                acc_dtype=W.f32,
            )
            W.store(sorted_probabilities + row_offset + rank, prefix)

        cutoff = W.index(0)
        found = W.i1(False)
        for rank in W.range(0, vocabulary):
            cumulative = W.load(
                sorted_probabilities + row_offset + rank,
                other=W.f32(0.0),
            )
            if (found == W.i1(False)) & (cumulative >= threshold):
                cutoff = rank + 1
                found = W.i1(True)
        if found == W.i1(False):
            cutoff = vocabulary
        W.store(nucleus_count + row, W.cast(cutoff, W.u32))

        last_nucleus_rank = cutoff - W.index(1)
        nucleus_mass = W.load(
            sorted_probabilities + row_offset + last_nucleus_rank,
            other=W.f32(0.0),
        )
        draw = W.load(uniforms + row, other=W.f32(0.0)) * nucleus_mass
        sampled_rank = W.index(0)
        sampled = W.i1(False)
        for rank in W.range(0, cutoff):
            cumulative = W.load(
                sorted_probabilities + row_offset + rank,
                other=W.f32(0.0),
            )
            if (sampled == W.i1(False)) & (cumulative >= draw):
                sampled_rank = rank
                sampled = W.i1(True)
        sampled_token = W.load(
            sorted_indices + row_offset + sampled_rank,
            other=W.u32(0),
        )
        W.store(sampled_tokens + row, sampled_token)
