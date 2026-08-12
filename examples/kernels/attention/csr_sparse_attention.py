import weft
import weft.language as W


@weft.kernel
def csr_sparse_attention_f32(
    query: W.ptr[W.f32, W.readonly, W.noalias],
    key: W.ptr[W.f32, W.readonly, W.noalias],
    value: W.ptr[W.f32, W.readonly, W.noalias],
    row_offsets: W.ptr[W.u32, W.readonly, W.noalias],
    key_indices: W.ptr[W.u32, W.readonly, W.noalias],
    output: W.ptr[W.f32, W.writeonly, W.noalias],
    accumulator_scratch: W.ptr[W.f32, W.noalias],
    row_begin: W.index,
    row_end: W.index,
    head_dimension: W.index,
    scale: W.f32,
) -> None:
    for row in W.range(row_begin, row_end):
        query_row = query + row * head_dimension
        output_row = output + row * head_dimension
        with W.vla(0, head_dimension) as dimension:
            W.store(accumulator_scratch + dimension, W.f32(0.0))

        maximum = W.neg_inf(W.f32)
        total = W.f32(0.0)
        begin = W.cast(W.load(row_offsets + row, other=W.u32(0)), W.index)
        end = W.cast(W.load(row_offsets + row + 1, other=W.u32(0)), W.index)
        for edge in W.range(begin, end):
            key_index = W.cast(
                W.load(key_indices + edge, other=W.u32(0)), W.index
            )
            key_row = key_index * head_dimension
            with W.vla(0, head_dimension) as dimension:
                score = W.reduce(
                    W.load(query_row + dimension, other=W.f32(0.0))
                    * W.load(key + key_row + dimension, other=W.f32(0.0)),
                    op="add",
                    identity=W.f32(0.0),
                    axis=None,
                    order="relaxed",
                    acc_dtype=W.f32,
                )
            score = score * scale
            if score > maximum:
                old_weight = W.exp(maximum - score, math="fast")
                with W.vla(0, head_dimension) as dimension:
                    W.store(
                        accumulator_scratch + dimension,
                        W.load(
                            accumulator_scratch + dimension,
                            other=W.f32(0.0),
                        )
                        * old_weight
                        + W.load(
                            value + key_row + dimension,
                            other=W.f32(0.0),
                        ),
                    )
                total = total * old_weight + W.f32(1.0)
                maximum = score
            else:
                new_weight = W.exp(score - maximum, math="fast")
                with W.vla(0, head_dimension) as dimension:
                    W.store(
                        accumulator_scratch + dimension,
                        W.load(
                            accumulator_scratch + dimension,
                            other=W.f32(0.0),
                        )
                        + W.load(
                            value + key_row + dimension,
                            other=W.f32(0.0),
                        )
                        * new_weight,
                    )
                total = total + new_weight

        with W.vla(0, head_dimension) as dimension:
            W.store(
                output_row + dimension,
                W.load(accumulator_scratch + dimension, other=W.f32(0.0))
                / total,
            )
