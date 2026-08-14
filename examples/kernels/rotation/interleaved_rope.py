import weft
import weft.language as W


@weft.kernel
def interleaved_rope_f32(
    source: W.ptr[W.f32, W.readonly, W.noalias],
    cosine: W.ptr[W.f32, W.readonly, W.noalias],
    destination: W.ptr[W.f32, W.writeonly, W.noalias],
    tokens: W.index,
    heads: W.index,
    pairs: W.index,
    token_stride: W.index,
    head_stride: W.index,
) -> None:
    for token in W.range(0, tokens):
        token_source = source + token * token_stride
        token_destination = destination + token * token_stride
        angle_base = token * pairs
        angle_values = cosine + angle_base * 2
        for head in W.range(0, heads):
            head_source = token_source + head * head_stride
            head_destination = token_destination + head * head_stride
            with W.vla(0, pairs) as pair:
                input_base = head_source + pair * 2
                real = W.load(input_base)
                imag = W.load(input_base + 1)
                angle_pair = angle_values + pair * 2
                cos_value = W.load(angle_pair)
                sin_value = W.load(angle_pair + 1)
                output_base = head_destination + pair * 2
                W.store(output_base, real * cos_value - imag * sin_value)
                W.store(output_base + 1, real * sin_value + imag * cos_value)
