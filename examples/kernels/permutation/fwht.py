import weft
import weft.language as W


@weft.kernel
def fwht_f32(
    data: W.ptr[W.f32, W.noalias],
    batches: W.index,
    extent: W.index,
) -> None:
    for batch in W.range(0, batches):
        batch_base = data + batch * extent
        power_of_two = (extent != W.index(0)) & (
            (extent & (extent - W.index(1))) == W.index(0)
        )
        if power_of_two:
            half = W.index(1)
            while half < extent:
                block_extent = half * 2
                for block in W.range(0, extent, block_extent):
                    with W.vla(0, half) as lane:
                        left_pointer = batch_base + block + lane
                        right_pointer = batch_base + block + half + lane
                        left = W.load(left_pointer, other=W.f32(0.0))
                        right = W.load(right_pointer, other=W.f32(0.0))
                        W.store(left_pointer, left + right)
                        W.store(right_pointer, left - right)
                half = block_extent
