import weft
import weft.language as W


@weft.kernel
def window_partition_f32(
    source: W.ptr[W.f32, W.readonly, W.noalias],
    output: W.ptr[W.f32, W.writeonly, W.noalias],
    height: W.index,
    width: W.index,
    channels: W.index,
    window: W.index,
    windows_y: W.index,
    windows_x: W.index,
) -> None:
    for window_y in W.range(0, windows_y):
        for window_x in W.range(0, windows_x):
            window_index = window_y * windows_x + window_x
            for local_y in W.range(0, window):
                source_y = window_y * window + local_y
                for local_x in W.range(0, window):
                    source_x = window_x * window + local_x
                    valid = (source_y < height) & (source_x < width)
                    with W.vla(0, channels) as channel:
                        value = W.load(
                            source
                            + (source_y * width + source_x) * channels
                            + channel,
                            where=valid,
                            other=W.f32(0.0),
                        )
                        W.store(
                            output
                            + (
                                (window_index * window + local_y) * window
                                + local_x
                            )
                            * channels
                            + channel,
                            value,
                        )
