import weft
import weft.language as W


@weft.kernel
def argsort_f32(
    values: W.ptr[W.f32, W.readonly, W.noalias],
    indices: W.ptr[W.u32, W.noalias],
    row_begin: W.index,
    row_end: W.index,
    columns: W.index,
    value_row_stride: W.index,
    index_row_stride: W.index,
) -> None:
    for row in W.range(row_begin, row_end):
        index_base = row * index_row_stride
        value_base = row * value_row_stride
        for column in W.range(0, columns):
            W.store(indices + index_base + column, W.cast(column, W.u32))

        inserted = W.index(1)
        while inserted < columns:
            child = inserted
            active = child > W.index(0)
            while active:
                parent = (child - W.index(1)) // W.index(2)
                parent_index = W.cast(
                    W.load(indices + index_base + parent, other=W.u32(0)),
                    W.index,
                )
                child_index = W.cast(
                    W.load(indices + index_base + child, other=W.u32(0)),
                    W.index,
                )
                parent_value = W.load(
                    values + value_base + parent_index,
                    other=W.f32(0.0),
                )
                child_value = W.load(
                    values + value_base + child_index,
                    other=W.f32(0.0),
                )
                if parent_value < child_value:
                    W.store(
                        indices + index_base + parent,
                        W.cast(child_index, W.u32),
                    )
                    W.store(
                        indices + index_base + child,
                        W.cast(parent_index, W.u32),
                    )
                    child = parent
                    active = child > W.index(0)
                else:
                    active = False
            inserted = inserted + W.index(1)

        end = columns
        while end > W.index(1):
            end = end - W.index(1)
            root_index = W.load(indices + index_base, other=W.u32(0))
            end_index = W.load(indices + index_base + end, other=W.u32(0))
            W.store(indices + index_base, end_index)
            W.store(indices + index_base + end, root_index)

            root = W.index(0)
            active = True
            while active:
                child = root * W.index(2) + W.index(1)
                if child >= end:
                    active = False
                else:
                    best = child
                    right = child + W.index(1)
                    if right < end:
                        left_index = W.cast(
                            W.load(
                                indices + index_base + child,
                                other=W.u32(0),
                            ),
                            W.index,
                        )
                        right_index = W.cast(
                            W.load(
                                indices + index_base + right,
                                other=W.u32(0),
                            ),
                            W.index,
                        )
                        left_value = W.load(
                            values + value_base + left_index,
                            other=W.f32(0.0),
                        )
                        right_value = W.load(
                            values + value_base + right_index,
                            other=W.f32(0.0),
                        )
                        if left_value < right_value:
                            best = right

                    root_value_index = W.cast(
                        W.load(
                            indices + index_base + root,
                            other=W.u32(0),
                        ),
                        W.index,
                    )
                    best_value_index = W.cast(
                        W.load(
                            indices + index_base + best,
                            other=W.u32(0),
                        ),
                        W.index,
                    )
                    root_value = W.load(
                        values + value_base + root_value_index,
                        other=W.f32(0.0),
                    )
                    best_value = W.load(
                        values + value_base + best_value_index,
                        other=W.f32(0.0),
                    )
                    if root_value < best_value:
                        W.store(
                            indices + index_base + root,
                            W.cast(best_value_index, W.u32),
                        )
                        W.store(
                            indices + index_base + best,
                            W.cast(root_value_index, W.u32),
                        )
                        root = best
                    else:
                        active = False
