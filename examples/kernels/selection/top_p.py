import weft
import weft.language as W

@weft.kernel
def top_p_nucleus_f32(
    probabilities: W.ptr[W.f32, W.readonly, W.noalias],
    sorted_indices: W.ptr[W.u32, W.workspace, W.noalias],
    sorted_probabilities: W.ptr[W.f32, W.workspace, W.noalias],
    uniforms: W.ptr[W.f32, W.readonly, W.noalias],
    nucleus_count: W.ptr[W.u32, W.writeonly, W.noalias],
    sampled_tokens: W.ptr[W.u32, W.writeonly, W.noalias],
    rows: W.index,
    vocabulary: W.index,
    threshold: W.f32,
) -> None:
    W.storage(sorted_indices, (rows, vocabulary))
    W.storage(sorted_probabilities, (rows, vocabulary))

    for row in W.range(0, rows):
        row_offset = row * vocabulary
        W.sort_indices(
            probabilities + row_offset,
            sorted_indices + row_offset,
            vocabulary,
            order="descending",
            nan="last",
            tie="index_ascending",
        )
        for rank in W.range(0, vocabulary):
            token = W.cast(
                W.load(
                    sorted_indices + row_offset + rank,
                    other=W.u32(0),
                ),
                W.index,
            )
            W.store(
                sorted_probabilities + row_offset + rank,
                W.load(
                    probabilities + row_offset + token,
                    other=W.f32(0.0),
                ),
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
