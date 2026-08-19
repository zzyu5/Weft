import weft
import weft.language as W


@weft.kernel
def qwen3vl_mrope_f32(
    source: W.ptr[W.f32, W.readonly, W.noalias],
    positions: W.ptr[W.i32, W.readonly, W.noalias],
    destination: W.ptr[W.f32, W.writeonly, W.noalias],
    angle_cache: W.ptr[W.f32, W.workspace, W.noalias],
    tokens: W.index,
    heads: W.index,
    head_dimension: W.index,
    rotary_dimension: W.index,
    head_stride: W.index,
    token_stride: W.index,
    section_t: W.index,
    section_h: W.index,
    section_w: W.index,
    section_e: W.index,
    theta_scale: W.f32,
) -> None:
    half_dimension = rotary_dimension / W.index(2)
    section_total = section_t + section_h + section_w + section_e
    W.buffer(angle_cache, (2, half_dimension))

    for token in W.range(0, tokens):
        theta_t = W.cast(
            W.load(positions + token, other=W.i32(0)), W.f32
        )
        theta_h = W.cast(
            W.load(positions + tokens + token, other=W.i32(0)), W.f32
        )
        theta_w = W.cast(
            W.load(positions + tokens * W.index(2) + token, other=W.i32(0)),
            W.f32,
        )
        theta_e = W.cast(
            W.load(positions + tokens * W.index(3) + token, other=W.i32(0)),
            W.f32,
        )

        for pair in W.range(0, half_dimension):
            sector = pair % section_total
            sector_kind = sector % W.index(3)
            theta = theta_e
            if (sector_kind == W.index(0)) & (
                sector < section_t * W.index(3)
            ):
                theta = theta_t
            if (sector_kind == W.index(1)) & (
                sector < section_h * W.index(3)
            ):
                theta = theta_h
            if (sector_kind == W.index(2)) & (
                sector < section_w * W.index(3)
            ):
                theta = theta_w

            W.store(angle_cache + pair, W.cos(theta))
            W.store(angle_cache + half_dimension + pair, W.sin(theta))
            theta_t = theta_t * theta_scale
            theta_h = theta_h * theta_scale
            theta_w = theta_w * theta_scale
            theta_e = theta_e * theta_scale

        token_source = source + token * token_stride
        token_destination = destination + token * token_stride
        for head in W.range(0, heads):
            head_source = token_source + head * head_stride
            head_destination = token_destination + head * head_stride
            with W.vla(0, half_dimension) as pair:
                first = W.load(head_source + pair)
                second = W.load(head_source + half_dimension + pair)
                cosine = W.load(angle_cache + pair)
                sine = W.load(angle_cache + half_dimension + pair)
                W.store(head_destination + pair, first * cosine - second * sine)
                W.store(
                    head_destination + half_dimension + pair,
                    first * sine + second * cosine,
                )

            with W.vla(rotary_dimension, head_dimension) as coordinate:
                W.store(
                    head_destination + coordinate,
                    W.load(head_source + coordinate),
                )
