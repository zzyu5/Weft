import weft
import weft.language as W


@weft.kernel
def argsort_f32(
    values: W.ptr[W.f32, W.readonly, W.noalias],
    indices: W.ptr[W.u32, W.noalias],
    scratch_indices: W.ptr[W.u32, W.noalias],
    histogram: W.ptr[W.u32, W.noalias],
    row_begin: W.index,
    row_end: W.index,
    columns: W.index,
    value_row_stride: W.index,
    index_row_stride: W.index,
) -> None:
    for row in W.range(row_begin, row_end):
        value_base = row * value_row_stride
        index_base = row * index_row_stride
        for column in W.range(0, columns):
            W.store(indices + index_base + column, W.cast(column, W.u32))

        for radix_pass in W.range(0, 4):
            for bucket in W.range(0, 256):
                W.store(histogram + bucket, W.u32(0))

            shift = W.cast(radix_pass * W.index(8), W.u32)
            for position in W.range(0, columns):
                source_index = W.u32(0)
                if radix_pass % W.index(2) == W.index(0):
                    source_index = W.load(
                        indices + index_base + position,
                        other=W.u32(0),
                    )
                else:
                    source_index = W.load(
                        scratch_indices + position,
                        other=W.u32(0),
                    )
                source_position = W.cast(source_index, W.index)
                bits = W.bitcast(
                    W.load(
                        values + value_base + source_position,
                        other=W.f32(0.0),
                    ),
                    W.u32,
                )
                key = bits ^ W.u32(0x80000000)
                if (bits & W.u32(0x80000000)) != W.u32(0):
                    key = bits ^ W.u32(0xFFFFFFFF)
                digit = (key >> shift) & W.u32(0xFF)
                bucket = W.cast(digit, W.index)
                count = W.load(histogram + bucket, other=W.u32(0))
                W.store(histogram + bucket, count + W.u32(1))

            offset = W.u32(0)
            for bucket in W.range(0, 256):
                count = W.load(histogram + bucket, other=W.u32(0))
                W.store(histogram + bucket, offset)
                offset = offset + count

            for position in W.range(0, columns):
                source_index = W.u32(0)
                if radix_pass % W.index(2) == W.index(0):
                    source_index = W.load(
                        indices + index_base + position,
                        other=W.u32(0),
                    )
                else:
                    source_index = W.load(
                        scratch_indices + position,
                        other=W.u32(0),
                    )
                source_position = W.cast(source_index, W.index)
                bits = W.bitcast(
                    W.load(
                        values + value_base + source_position,
                        other=W.f32(0.0),
                    ),
                    W.u32,
                )
                key = bits ^ W.u32(0x80000000)
                if (bits & W.u32(0x80000000)) != W.u32(0):
                    key = bits ^ W.u32(0xFFFFFFFF)
                digit = (key >> shift) & W.u32(0xFF)
                bucket = W.cast(digit, W.index)
                destination = W.load(histogram + bucket, other=W.u32(0))
                destination_position = W.cast(destination, W.index)
                if radix_pass % W.index(2) == W.index(0):
                    W.store(scratch_indices + destination_position, source_index)
                else:
                    W.store(
                        indices + index_base + destination_position,
                        source_index,
                    )
                W.store(histogram + bucket, destination + W.u32(1))
