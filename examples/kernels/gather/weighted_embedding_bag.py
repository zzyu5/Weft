import weft
import weft.language as W


@weft.kernel
def weighted_embedding_bag_f32(
    table: W.ptr[W.f32, W.readonly, W.noalias],
    indices: W.ptr[W.u32, W.readonly, W.noalias],
    weights: W.ptr[W.f32, W.readonly, W.noalias],
    offsets: W.ptr[W.u32, W.readonly, W.noalias],
    output: W.ptr[W.f32, W.writeonly, W.noalias],
    bag_begin: W.index,
    bag_end: W.index,
    embedding_dim: W.index,
    table_stride: W.index,
    output_stride: W.index,
) -> None:
    for bag in W.range(bag_begin, bag_end):
        begin = W.cast(W.load(offsets + bag, other=W.u32(0)), W.index)
        end = W.cast(W.load(offsets + bag + 1, other=W.u32(0)), W.index)
        if begin < end:
            first_row = W.cast(
                W.load(indices + begin, other=W.u32(0)), W.index
            )
            first_weight = W.load(weights + begin, other=W.f32(0.0))
            with W.vla(0, embedding_dim) as channel:
                accumulator = (
                    W.load(
                        table + first_row * table_stride + channel,
                        other=W.f32(0.0),
                    )
                    * first_weight
                )
                for entry in W.range(begin + 1, end):
                    row = W.cast(
                        W.load(indices + entry, other=W.u32(0)), W.index
                    )
                    weight = W.load(weights + entry, other=W.f32(0.0))
                    value = W.load(
                        table + row * table_stride + channel,
                        other=W.f32(0.0),
                    )
                    accumulator = accumulator + weight * value
                W.store(output + bag * output_stride + channel, accumulator)
        else:
            with W.vla(0, embedding_dim) as channel:
                W.store(
                    output + bag * output_stride + channel,
                    W.f32(0.0),
                )


@weft.kernel
def weighted_embedding_bag_f32_equivalent(
    table: W.ptr[W.f32, W.readonly, W.noalias],
    indices: W.ptr[W.u32, W.readonly, W.noalias],
    weights: W.ptr[W.f32, W.readonly, W.noalias],
    offsets: W.ptr[W.u32, W.readonly, W.noalias],
    output: W.ptr[W.f32, W.writeonly, W.noalias],
    bag_begin: W.index,
    bag_end: W.index,
    embedding_dim: W.index,
    table_stride: W.index,
    output_stride: W.index,
) -> None:
    for bag in W.range(bag_begin, bag_end):
        offset_address = offsets + bag
        begin = W.cast(W.load(offset_address, other=W.u32(0)), W.index)
        end = W.cast(W.load(offset_address + 1, other=W.u32(0)), W.index)
        if end > begin:
            first_row = W.cast(
                W.load(indices + begin, other=W.u32(0)), W.index
            )
            first_weight = W.load(weights + begin, other=W.f32(0.0))
            entry_count = end - begin
            with W.vla(0, embedding_dim) as channel:
                first_row_base = table + first_row * table_stride
                accumulator = (
                    W.load(first_row_base + channel, other=W.f32(0.0))
                    * first_weight
                )
                for relative_entry in W.range(1, entry_count):
                    entry = begin + relative_entry
                    row = W.cast(
                        W.load(indices + entry, other=W.u32(0)), W.index
                    )
                    row_base = table + row * table_stride
                    weight = W.load(weights + entry, other=W.f32(0.0))
                    accumulator = accumulator + weight * W.load(
                        row_base + channel, other=W.f32(0.0)
                    )
                output_base = output + bag * output_stride
                W.store(output_base + channel, accumulator)
        else:
            output_base = output + bag * output_stride
            with W.vla(0, embedding_dim) as channel:
                W.store(output_base + channel, W.f32(0.0))
