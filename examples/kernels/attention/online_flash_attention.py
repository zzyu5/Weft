import weft
import weft.language as W


@weft.kernel
def online_flash_attention_f32_f16(
    query: W.ptr[W.f32, W.readonly, W.noalias],
    key: W.ptr[W.f16, W.readonly, W.noalias],
    value: W.ptr[W.f16, W.readonly, W.noalias],
    mask: W.ptr[W.f16, W.readonly, W.noalias],
    output: W.ptr[W.f32, W.writeonly, W.noalias],
    query_scratch: W.ptr[W.f16, W.noalias],
    accumulator_scratch: W.ptr[W.f16, W.noalias],
    head_begin: W.index,
    head_end: W.index,
    queries: W.index,
    keys: W.index,
    head_dimension: W.index,
    group_size: W.index,
    scale: W.f32,
) -> None:
    for query_head in W.range(head_begin, head_end):
        key_value_head = query_head // group_size
        query_head_offset = query_head * queries * head_dimension
        key_value_head_offset = key_value_head * keys * head_dimension

        for query_index in W.range(0, queries):
            query_row = query_head_offset + query_index * head_dimension

            with W.vla(0, head_dimension) as dimension:
                query_element = W.load(query + query_row + dimension)
                W.store(
                    query_scratch + dimension,
                    W.cast(query_element, W.f16),
                )

            with W.vla(0, head_dimension) as dimension:
                W.store(accumulator_scratch + dimension, W.f16(0.0))

            maximum = W.neg_inf(W.f32)
            total = W.f32(0.0)
            for key_index in W.range(0, keys):
                mask_element = W.cast(
                    W.load(
                        mask + query_index * keys + key_index,
                        other=W.f16(0.0),
                    ),
                    W.f32,
                )
                if mask_element != W.neg_inf(W.f32):
                    key_row = (
                        key_value_head_offset + key_index * head_dimension
                    )
                    with W.vla(0, head_dimension) as dimension:
                        query_element = W.cast(
                            W.load(query_scratch + dimension), W.f32
                        )
                        key_element = W.cast(
                            W.load(key + key_row + dimension), W.f32
                        )
                        score = W.reduce(
                            query_element * key_element,
                            identity=W.f32(0.0),
                            acc_dtype=W.f32,
                            order="relaxed",
                        )

                    score = score * scale + mask_element
                    if score > maximum:
                        old_weight = W.exp(maximum - score, math="fast")
                        old_weight_f16 = W.cast(old_weight, W.f16)
                        with W.vla(0, head_dimension) as dimension:
                            old_output = W.load(
                                accumulator_scratch + dimension
                            )
                            value_element = W.load(
                                value + key_row + dimension
                            )
                            W.store(
                                accumulator_scratch + dimension,
                                old_output * old_weight_f16 + value_element,
                            )
                        total = total * old_weight + W.f32(1.0)
                        maximum = score
                    else:
                        new_weight = W.exp(score - maximum, math="fast")
                        new_weight_f16 = W.cast(new_weight, W.f16)
                        with W.vla(0, head_dimension) as dimension:
                            old_output = W.load(
                                accumulator_scratch + dimension
                            )
                            value_element = W.load(
                                value + key_row + dimension
                            )
                            W.store(
                                accumulator_scratch + dimension,
                                old_output + value_element * new_weight_f16,
                            )
                        total = total + new_weight

            with W.vla(0, head_dimension) as dimension:
                accumulated = W.cast(
                    W.load(accumulator_scratch + dimension), W.f32
                )
                W.store(
                    output + query_row + dimension,
                    accumulated / total,
                )
