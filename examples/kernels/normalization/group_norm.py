import weft
import weft.language as W


@weft.kernel
def group_norm_f32(
    source: W.ptr[W.f32, W.readonly, W.noalias],
    output: W.ptr[W.f32, W.noalias],
    batches: W.index,
    channels: W.index,
    height: W.index,
    width: W.index,
    groups: W.index,
    epsilon: W.f32,
) -> None:
    plane = height * width
    batch_stride = channels * plane
    channels_per_group = (channels + groups - 1) / groups

    for batch in W.range(0, batches):
        for group in W.range(0, groups):
            channel_begin = group * channels_per_group
            channel_end = channel_begin + channels_per_group
            if channel_end > channels:
                channel_end = channels

            total = W.f32(0.0)
            for channel in W.range(channel_begin, channel_end):
                for row in W.range(0, height):
                    with W.vla(0, width) as column:
                        value = W.load(
                            source
                            + batch * batch_stride
                            + channel * plane
                            + row * width
                            + column
                        )
                        row_total = W.reduce(
                            value,
                            op="add",
                            identity=W.f32(0.0),
                            order="relaxed",
                            acc_dtype=W.f32,
                        )
                    total = total + row_total

            count = W.cast((channel_end - channel_begin) * plane, W.f32)
            mean = total / count
            sum_squared = W.f32(0.0)

            for channel in W.range(channel_begin, channel_end):
                for row in W.range(0, height):
                    with W.vla(0, width) as column:
                        centered = W.load(
                            source
                            + batch * batch_stride
                            + channel * plane
                            + row * width
                            + column
                        ) - mean
                        W.store(
                            output
                            + batch * batch_stride
                            + channel * plane
                            + row * width
                            + column,
                            centered,
                        )
                        row_squared = W.reduce(
                            centered * centered,
                            op="add",
                            identity=W.f32(0.0),
                            order="relaxed",
                            acc_dtype=W.f32,
                        )
                    sum_squared = sum_squared + row_squared

            scale = W.rsqrt(sum_squared / count + epsilon, math="native")
            for channel in W.range(channel_begin, channel_end):
                for row in W.range(0, height):
                    with W.vla(0, width) as column:
                        address = (
                            output
                            + batch * batch_stride
                            + channel * plane
                            + row * width
                            + column
                        )
                        W.store(address, W.load(address) * scale)
