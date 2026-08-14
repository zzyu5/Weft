import weft
import weft.language as W


@weft.kernel
def timestep_embedding_f32(
    timesteps: W.ptr[W.f32, W.readonly, W.noalias],
    output: W.ptr[W.f32, W.writeonly, W.noalias],
    row_begin: W.index,
    row_end: W.index,
    dimension: W.index,
    maximum_period: W.f32,
    output_stride: W.index,
) -> None:
    half = dimension / W.index(2)
    logarithmic_period = W.log(maximum_period)
    for row in W.range(row_begin, row_end):
        timestep = W.load(timesteps + row)
        output_row = output + row * output_stride
        with W.vla(0, half) as coordinate:
            frequency = W.exp(
                -logarithmic_period
                * W.cast(coordinate, W.f32)
                / W.cast(half, W.f32)
            )
            angle = timestep * frequency
            W.store(output_row + coordinate, W.cos(angle))
            W.store(output_row + half + coordinate, W.sin(angle))

        if dimension % W.index(2) != W.index(0):
            W.store(output_row + half * W.index(2), W.f32(0.0))
