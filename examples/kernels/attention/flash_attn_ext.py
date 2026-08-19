import weft
import weft.language as W


@weft.kernel
def flash_attn_ext_f32_f16(
    query: W.ptr[W.f32, W.readonly, W.noalias],
    key: W.ptr[W.f16, W.readonly, W.noalias],
    value: W.ptr[W.f16, W.readonly, W.noalias],
    mask: W.ptr[W.f16, W.readonly, W.noalias],
    sinks: W.ptr[W.f32, W.readonly, W.noalias],
    output: W.ptr[W.f32, W.writeonly, W.noalias],
    query_scratch: W.ptr[W.f32, W.workspace, W.noalias],
    accumulator_scratch: W.ptr[W.f32, W.workspace, W.noalias],
    outer_begin: W.index,
    outer_end: W.index,
    outer_count: W.index,
    key_outer_count: W.index,
    value_outer_count: W.index,
    mask_outer_count: W.index,
    query_heads: W.index,
    key_heads: W.index,
    value_heads: W.index,
    mask_heads: W.index,
    queries: W.index,
    keys: W.index,
    key_dimension: W.index,
    value_dimension: W.index,
    query_outer_stride: W.index,
    query_head_stride: W.index,
    query_row_stride: W.index,
    key_outer_stride: W.index,
    key_head_stride: W.index,
    key_row_stride: W.index,
    value_outer_stride: W.index,
    value_head_stride: W.index,
    value_row_stride: W.index,
    mask_outer_stride: W.index,
    mask_head_stride: W.index,
    mask_row_stride: W.index,
    output_outer_stride: W.index,
    output_query_stride: W.index,
    output_head_stride: W.index,
    scale: W.f32,
    maximum_bias: W.f32,
    logit_softcap: W.f32,
) -> None:
    W.buffer(query_scratch, (key_dimension,))
    W.buffer(accumulator_scratch, (value_dimension,))

    head_power_of_two = W.index(1)
    while head_power_of_two * W.index(2) <= query_heads:
        head_power_of_two = head_power_of_two * W.index(2)
    logarithm_two = W.log(W.f32(2.0))
    slope_base = W.exp(
        -maximum_bias
        * logarithm_two
        / W.cast(head_power_of_two, W.f32)
    )
    extra_slope_base = W.exp(
        -(maximum_bias / W.f32(2.0))
        * logarithm_two
        / W.cast(head_power_of_two, W.f32)
    )

    key_outer_group = outer_count / key_outer_count
    value_outer_group = outer_count / value_outer_count
    key_head_group = query_heads / key_heads
    value_head_group = query_heads / value_heads

    for outer in W.range(outer_begin, outer_end):
        key_outer = outer / key_outer_group
        value_outer = outer / value_outer_group
        mask_outer = outer % mask_outer_count
        for query_head in W.range(0, query_heads):
            key_head = query_head / key_head_group
            value_head = query_head / value_head_group
            mask_head = query_head % mask_heads
            if query_head < head_power_of_two:
                slope = W.exp(
                    W.log(slope_base)
                    * W.cast(query_head + W.index(1), W.f32)
                )
            else:
                slope = W.exp(
                    W.log(extra_slope_base)
                    * W.cast(
                        W.index(2) * (query_head - head_power_of_two)
                        + W.index(1),
                        W.f32,
                    )
                )

            for query_index in W.range(0, queries):
                query_row = (
                    outer * query_outer_stride
                    + query_head * query_head_stride
                    + query_index * query_row_stride
                )
                with W.vla(0, key_dimension) as dimension:
                    W.store(
                        query_scratch + dimension,
                        W.load(query + query_row + dimension),
                    )
                with W.vla(0, value_dimension) as dimension:
                    W.store(accumulator_scratch + dimension, W.f32(0.0))

                maximum = W.neg_inf(W.f32)
                total = W.f32(0.0)
                for key_index in W.range(0, keys):
                    mask_value = W.cast(
                        W.load(
                            mask
                            + mask_outer * mask_outer_stride
                            + mask_head * mask_head_stride
                            + query_index * mask_row_stride
                            + key_index,
                            other=W.f16(0.0),
                        ),
                        W.f32,
                    )
                    if mask_value != W.neg_inf(W.f32):
                        key_row = (
                            key_outer * key_outer_stride
                            + key_head * key_head_stride
                            + key_index * key_row_stride
                        )
                        with W.vla(0, key_dimension) as dimension:
                            query_element = W.load(query_scratch + dimension)
                            key_element = W.cast(
                                W.load(key + key_row + dimension), W.f32
                            )
                            score = W.reduce(
                                query_element * key_element,
                                op="add",
                                identity=W.f32(0.0),
                                order="relaxed",
                                acc_dtype=W.f32,
                            )

                        score = (
                            logit_softcap
                            * W.tanh(score * scale / logit_softcap)
                            + mask_value * slope
                        )
                        if score > maximum:
                            old_weight = W.exp(maximum - score, math="fast")
                            with W.vla(0, value_dimension) as dimension:
                                old_output = W.load(
                                    accumulator_scratch + dimension
                                )
                                value_element = W.cast(
                                    W.load(
                                        value
                                        + value_outer * value_outer_stride
                                        + value_head * value_head_stride
                                        + key_index * value_row_stride
                                        + dimension
                                    ),
                                    W.f32,
                                )
                                W.store(
                                    accumulator_scratch + dimension,
                                    old_output * old_weight + value_element,
                                )
                            total = total * old_weight + W.f32(1.0)
                            maximum = score
                        else:
                            new_weight = W.exp(score - maximum, math="fast")
                            with W.vla(0, value_dimension) as dimension:
                                old_output = W.load(
                                    accumulator_scratch + dimension
                                )
                                value_element = W.cast(
                                    W.load(
                                        value
                                        + value_outer * value_outer_stride
                                        + value_head * value_head_stride
                                        + key_index * value_row_stride
                                        + dimension
                                    ),
                                    W.f32,
                                )
                                W.store(
                                    accumulator_scratch + dimension,
                                    old_output + value_element * new_weight,
                                )
                            total = total + new_weight

                sink_score = W.load(sinks + query_head)
                if sink_score > maximum:
                    sink_weight = W.exp(maximum - sink_score, math="fast")
                    with W.vla(0, value_dimension) as dimension:
                        W.store(
                            accumulator_scratch + dimension,
                            W.load(accumulator_scratch + dimension) * sink_weight,
                        )
                    total = total * sink_weight + W.f32(1.0)
                    maximum = sink_score
                else:
                    total = total + W.exp(
                        sink_score - maximum, math="fast"
                    )

                output_row = (
                    outer * output_outer_stride
                    + query_index * output_query_stride
                    + query_head * output_head_stride
                )
                with W.vla(0, value_dimension) as dimension:
                    W.store(
                        output + output_row + dimension,
                        W.load(accumulator_scratch + dimension) / total,
                    )
