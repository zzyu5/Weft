import weft
import weft.language as W


@W.pure
def _bilinear(top_left, top_right, bottom_left, bottom_right, dx, dy):
    return (
        top_left * (W.f32(1.0) - dx) * (W.f32(1.0) - dy)
        + top_right * dx * (W.f32(1.0) - dy)
        + bottom_left * (W.f32(1.0) - dx) * dy
        + bottom_right * dx * dy
    )


@weft.kernel
def roi_align_f32(
    source: W.ptr[W.f32, W.readonly, W.noalias],
    boxes: W.ptr[W.f32, W.readonly, W.noalias],
    batch_indices: W.ptr[W.u32, W.readonly, W.noalias],
    output: W.ptr[W.f32, W.writeonly, W.noalias],
    box_begin: W.index,
    box_end: W.index,
    input_height: W.index,
    input_width: W.index,
    channels: W.index,
    pooled_height: W.index,
    pooled_width: W.index,
    samples_y: W.index,
    samples_x: W.index,
    spatial_scale: W.f32,
) -> None:
    image_stride = input_height * input_width * channels
    input_row_stride = input_width * channels
    output_box_stride = pooled_height * pooled_width * channels
    sample_extent = samples_y * samples_x
    sample_count = W.cast(sample_extent, W.f32)
    maximum_y = W.cast(input_height - 1, W.f32)
    maximum_x = W.cast(input_width - 1, W.f32)

    for box in W.range(box_begin, box_end):
        image = W.cast(
            W.load(batch_indices + box, other=W.u32(0)), W.index
        )
        x1 = W.load(boxes + box * 4 + 0, other=W.f32(0.0)) * spatial_scale
        y1 = W.load(boxes + box * 4 + 1, other=W.f32(0.0)) * spatial_scale
        x2 = W.load(boxes + box * 4 + 2, other=W.f32(0.0)) * spatial_scale
        y2 = W.load(boxes + box * 4 + 3, other=W.f32(0.0)) * spatial_scale
        roi_width = W.maximum(x2 - x1, W.f32(1.0))
        roi_height = W.maximum(y2 - y1, W.f32(1.0))
        bin_width = roi_width / W.cast(pooled_width, W.f32)
        bin_height = roi_height / W.cast(pooled_height, W.f32)
        image_base = source + image * image_stride

        for pooled_y in W.range(0, pooled_height):
            for pooled_x in W.range(0, pooled_width):
                first_y = y1 + (
                    W.cast(pooled_y, W.f32)
                    + W.f32(0.5) / W.cast(samples_y, W.f32)
                ) * bin_height
                first_x = x1 + (
                    W.cast(pooled_x, W.f32)
                    + W.f32(0.5) / W.cast(samples_x, W.f32)
                ) * bin_width
                first_floor_y = W.floor(first_y)
                first_floor_x = W.floor(first_x)
                first_y0 = W.cast(
                    W.minimum(
                        W.maximum(first_floor_y, W.f32(0.0)), maximum_y
                    ),
                    W.index,
                )
                first_x0 = W.cast(
                    W.minimum(
                        W.maximum(first_floor_x, W.f32(0.0)), maximum_x
                    ),
                    W.index,
                )
                first_y1 = W.cast(
                    W.minimum(
                        W.maximum(first_floor_y + W.f32(1.0), W.f32(0.0)),
                        maximum_y,
                    ),
                    W.index,
                )
                first_x1 = W.cast(
                    W.minimum(
                        W.maximum(first_floor_x + W.f32(1.0), W.f32(0.0)),
                        maximum_x,
                    ),
                    W.index,
                )
                first_dy = first_y - W.cast(first_y0, W.f32)
                first_dx = first_x - W.cast(first_x0, W.f32)

                with W.vla(0, channels) as channel:
                    accumulator = _bilinear(
                        W.load(
                            image_base
                            + first_y0 * input_row_stride
                            + first_x0 * channels
                            + channel
                        ),
                        W.load(
                            image_base
                            + first_y0 * input_row_stride
                            + first_x1 * channels
                            + channel
                        ),
                        W.load(
                            image_base
                            + first_y1 * input_row_stride
                            + first_x0 * channels
                            + channel
                        ),
                        W.load(
                            image_base
                            + first_y1 * input_row_stride
                            + first_x1 * channels
                            + channel
                        ),
                        first_dx,
                        first_dy,
                    )
                    for sample in W.range(1, sample_extent):
                        sample_y = sample // samples_x
                        sample_x = sample % samples_x
                        source_y = y1 + (
                            W.cast(pooled_y, W.f32)
                            + (
                                W.cast(sample_y, W.f32) + W.f32(0.5)
                            )
                            / W.cast(samples_y, W.f32)
                        ) * bin_height
                        source_x = x1 + (
                            W.cast(pooled_x, W.f32)
                            + (
                                W.cast(sample_x, W.f32) + W.f32(0.5)
                            )
                            / W.cast(samples_x, W.f32)
                        ) * bin_width
                        floor_y = W.floor(source_y)
                        floor_x = W.floor(source_x)
                        y0 = W.cast(
                            W.minimum(
                                W.maximum(floor_y, W.f32(0.0)), maximum_y
                            ),
                            W.index,
                        )
                        x0 = W.cast(
                            W.minimum(
                                W.maximum(floor_x, W.f32(0.0)), maximum_x
                            ),
                            W.index,
                        )
                        y1_index = W.cast(
                            W.minimum(
                                W.maximum(
                                    floor_y + W.f32(1.0), W.f32(0.0)
                                ),
                                maximum_y,
                            ),
                            W.index,
                        )
                        x1_index = W.cast(
                            W.minimum(
                                W.maximum(
                                    floor_x + W.f32(1.0), W.f32(0.0)
                                ),
                                maximum_x,
                            ),
                            W.index,
                        )
                        dy = source_y - W.cast(y0, W.f32)
                        dx = source_x - W.cast(x0, W.f32)
                        accumulator = accumulator + _bilinear(
                            W.load(
                                image_base
                                + y0 * input_row_stride
                                + x0 * channels
                                + channel
                            ),
                            W.load(
                                image_base
                                + y0 * input_row_stride
                                + x1_index * channels
                                + channel
                            ),
                            W.load(
                                image_base
                                + y1_index * input_row_stride
                                + x0 * channels
                                + channel
                            ),
                            W.load(
                                image_base
                                + y1_index * input_row_stride
                                + x1_index * channels
                                + channel
                            ),
                            dx,
                            dy,
                        )
                    W.store(
                        output
                        + box * output_box_stride
                        + pooled_y * pooled_width * channels
                        + pooled_x * channels
                        + channel,
                        accumulator / sample_count,
                    )


@weft.kernel
def roi_align_f32_equivalent(
    source: W.ptr[W.f32, W.readonly, W.noalias],
    boxes: W.ptr[W.f32, W.readonly, W.noalias],
    batch_indices: W.ptr[W.u32, W.readonly, W.noalias],
    output: W.ptr[W.f32, W.writeonly, W.noalias],
    box_begin: W.index,
    box_end: W.index,
    input_height: W.index,
    input_width: W.index,
    channels: W.index,
    pooled_height: W.index,
    pooled_width: W.index,
    samples_y: W.index,
    samples_x: W.index,
    spatial_scale: W.f32,
) -> None:
    image_stride = input_height * input_width * channels
    row_stride = input_width * channels
    box_stride = pooled_height * pooled_width * channels
    sample_extent = samples_x * samples_y
    maximum_y = W.cast(input_height - 1, W.f32)
    maximum_x = W.cast(input_width - 1, W.f32)
    for box in W.range(box_begin, box_end):
        image = W.cast(
            W.load(batch_indices + box, other=W.u32(0)), W.index
        )
        box_base = boxes + box * 4
        x1 = W.load(box_base + 0, other=W.f32(0.0)) * spatial_scale
        y1 = W.load(box_base + 1, other=W.f32(0.0)) * spatial_scale
        x2 = W.load(box_base + 2, other=W.f32(0.0)) * spatial_scale
        y2 = W.load(box_base + 3, other=W.f32(0.0)) * spatial_scale
        bin_width = W.maximum(x2 - x1, W.f32(1.0)) / W.cast(
            pooled_width, W.f32
        )
        bin_height = W.maximum(y2 - y1, W.f32(1.0)) / W.cast(
            pooled_height, W.f32
        )
        image_base = source + image * image_stride
        output_base = output + box * box_stride
        for pooled_y in W.range(0, pooled_height):
            for pooled_x in W.range(0, pooled_width):
                first_y = y1 + (
                    W.cast(pooled_y, W.f32)
                    + W.f32(0.5) / W.cast(samples_y, W.f32)
                ) * bin_height
                first_x = x1 + (
                    W.cast(pooled_x, W.f32)
                    + W.f32(0.5) / W.cast(samples_x, W.f32)
                ) * bin_width
                first_floor_y = W.floor(first_y)
                first_floor_x = W.floor(first_x)
                first_y0 = W.cast(
                    W.minimum(
                        maximum_y,
                        W.maximum(W.f32(0.0), first_floor_y),
                    ),
                    W.index,
                )
                first_x0 = W.cast(
                    W.minimum(
                        maximum_x,
                        W.maximum(W.f32(0.0), first_floor_x),
                    ),
                    W.index,
                )
                first_y1 = W.cast(
                    W.minimum(
                        maximum_y,
                        W.maximum(W.f32(0.0), first_floor_y + W.f32(1.0)),
                    ),
                    W.index,
                )
                first_x1 = W.cast(
                    W.minimum(
                        maximum_x,
                        W.maximum(W.f32(0.0), first_floor_x + W.f32(1.0)),
                    ),
                    W.index,
                )
                first_dy = first_y - W.cast(first_y0, W.f32)
                first_dx = first_x - W.cast(first_x0, W.f32)
                with W.vla(0, channels) as channel:
                    next_top_left = W.load(
                        image_base
                        + first_y0 * row_stride
                        + first_x0 * channels
                        + channel
                    )
                    next_top_right = W.load(
                        image_base
                        + first_y0 * row_stride
                        + first_x1 * channels
                        + channel
                    )
                    next_bottom_left = W.load(
                        image_base
                        + first_y1 * row_stride
                        + first_x0 * channels
                        + channel
                    )
                    next_bottom_right = W.load(
                        image_base
                        + first_y1 * row_stride
                        + first_x1 * channels
                        + channel
                    )
                    accumulator = (
                        next_top_left
                        * (W.f32(1.0) - first_dx)
                        * (W.f32(1.0) - first_dy)
                        + next_top_right
                        * first_dx
                        * (W.f32(1.0) - first_dy)
                        + next_bottom_left
                        * (W.f32(1.0) - first_dx)
                        * first_dy
                        + next_bottom_right * first_dx * first_dy
                    )
                    for sample in W.range(1, sample_extent):
                        grid_y = sample // samples_x
                        grid_x = sample % samples_x
                        source_y = y1 + (
                            W.cast(pooled_y, W.f32)
                            + (W.cast(grid_y, W.f32) + W.f32(0.5))
                            / W.cast(samples_y, W.f32)
                        ) * bin_height
                        source_x = x1 + (
                            W.cast(pooled_x, W.f32)
                            + (W.cast(grid_x, W.f32) + W.f32(0.5))
                            / W.cast(samples_x, W.f32)
                        ) * bin_width
                        floor_y = W.floor(source_y)
                        floor_x = W.floor(source_x)
                        y0 = W.cast(
                            W.minimum(
                                maximum_y,
                                W.maximum(W.f32(0.0), floor_y),
                            ),
                            W.index,
                        )
                        x0 = W.cast(
                            W.minimum(
                                maximum_x,
                                W.maximum(W.f32(0.0), floor_x),
                            ),
                            W.index,
                        )
                        y1_index = W.cast(
                            W.minimum(
                                maximum_y,
                                W.maximum(W.f32(0.0), floor_y + W.f32(1.0)),
                            ),
                            W.index,
                        )
                        x1_index = W.cast(
                            W.minimum(
                                maximum_x,
                                W.maximum(W.f32(0.0), floor_x + W.f32(1.0)),
                            ),
                            W.index,
                        )
                        dy = source_y - W.cast(y0, W.f32)
                        dx = source_x - W.cast(x0, W.f32)
                        sample_top_left = W.load(
                            image_base
                            + y0 * row_stride
                            + x0 * channels
                            + channel
                        )
                        sample_top_right = W.load(
                            image_base
                            + y0 * row_stride
                            + x1_index * channels
                            + channel
                        )
                        sample_bottom_left = W.load(
                            image_base
                            + y1_index * row_stride
                            + x0 * channels
                            + channel
                        )
                        sample_bottom_right = W.load(
                            image_base
                            + y1_index * row_stride
                            + x1_index * channels
                            + channel
                        )
                        accumulator = accumulator + (
                            sample_top_left
                            * (W.f32(1.0) - dx)
                            * (W.f32(1.0) - dy)
                            + sample_top_right * dx * (W.f32(1.0) - dy)
                            + sample_bottom_left * (W.f32(1.0) - dx) * dy
                            + sample_bottom_right * dx * dy
                        )
                    W.store(
                        output_base
                        + pooled_y * pooled_width * channels
                        + pooled_x * channels
                        + channel,
                        accumulator / W.cast(sample_extent, W.f32),
                    )
