import weft
import weft.language as W


@weft.kernel
def sam_add_relative_position_f32(
    scores: W.ptr[W.f32, W.readonly, W.noalias],
    relative_width: W.ptr[W.f32, W.readonly, W.noalias],
    relative_height: W.ptr[W.f32, W.readonly, W.noalias],
    output: W.ptr[W.f32, W.noalias],
    patches: W.index,
    query_height: W.index,
    query_width: W.index,
    key_size: W.index,
) -> None:
    query_plane = query_height * query_width
    score_stride = query_plane * key_size * key_size

    for patch in W.range(0, patches):
        with W.vla(0, score_stride) as offset:
            W.store(
                output + patch * score_stride + offset,
                W.load(scores + patch * score_stride + offset),
            )

        for query_y in W.range(0, query_height):
            for query_x in W.range(0, query_width):
                query = query_y * query_width + query_x
                score_base = output + patch * score_stride + query * key_size * key_size
                relative_base = patch * query_plane + query

                with W.vla(0, key_size) as key:
                    horizontal = W.load(
                        relative_width + relative_base * key_size + key
                    )
                    vertical = W.load(
                        relative_height + relative_base * key_size + key
                    )
                    for key_y in W.range(0, key_size):
                        address = score_base + key_y * key_size + key
                        W.store(address, W.load(address) + horizontal)
                    for key_x in W.range(0, key_size):
                        address = score_base + key * key_size + key_x
                        W.store(address, W.load(address) + vertical)
