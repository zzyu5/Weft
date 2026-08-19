import weft
import weft.language as W


@weft.kernel
def rope_neox_f32(
    source: W.ptr[W.f32, W.readonly, W.noalias],
    positions: W.ptr[W.i32, W.readonly, W.noalias],
    destination: W.ptr[W.f32, W.writeonly, W.noalias],
    angle_cache: W.ptr[W.f32, W.workspace, W.noalias],
    tokens: W.index,
    heads: W.index,
    half_dimension: W.index,
    head_stride: W.index,
    token_stride: W.index,
    theta_scale: W.f32,
) -> None:
    W.buffer(angle_cache, (2, half_dimension))

    for token in W.range(0, tokens):
        theta = W.cast(
            W.load(positions + token, other=W.i32(0)), W.f32
        )
        for pair in W.range(0, half_dimension):
            cosine = W.cos(theta)
            sine = W.sin(theta)
            W.store(angle_cache + pair, cosine)
            W.store(angle_cache + half_dimension + pair, sine)
            theta = theta * theta_scale

        token_source = source + token * token_stride
        token_destination = destination + token * token_stride
        for head in W.range(0, heads):
            head_source = token_source + head * head_stride
            head_destination = token_destination + head * head_stride
            with W.vla(0, half_dimension) as pair:
                first_address = head_source + pair
                second_address = head_source + half_dimension + pair
                first = W.load(first_address)
                second = W.load(second_address)
                cosine = W.load(angle_cache + pair)
                sine = W.load(angle_cache + half_dimension + pair)
                rotated_first = first * cosine - second * sine
                rotated_second = first * sine + second * cosine

                output_first = head_destination + pair
                W.store(output_first, rotated_first)
                W.store(output_first + half_dimension, rotated_second)
