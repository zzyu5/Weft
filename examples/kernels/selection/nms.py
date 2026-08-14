import weft
import weft.language as W


@weft.kernel
def greedy_nms_f32(
    boxes: W.ptr[W.f32, W.readonly, W.noalias],
    scores: W.ptr[W.f32, W.readonly, W.noalias],
    suppressed: W.ptr[W.u8, W.workspace, W.noalias],
    selected: W.ptr[W.u32, W.writeonly, W.noalias],
    candidates: W.index,
    maximum_selected: W.index,
    iou_threshold: W.f32,
) -> None:
    W.storage(suppressed, (candidates,))

    for candidate in W.range(0, candidates):
        W.store(suppressed + candidate, W.u8(0))

    count = W.index(0)
    while count < maximum_selected:
        best_index = W.index(0)
        best_score = W.neg_inf(W.f32)
        found = W.i1(False)
        for candidate in W.range(0, candidates):
            active = W.load(suppressed + candidate, other=W.u8(1)) == W.u8(0)
            score = W.load(scores + candidate, other=W.neg_inf(W.f32))
            if active & (
                (score > best_score)
                | ((score == best_score) & (candidate < best_index))
            ):
                best_index = candidate
                best_score = score
                found = W.i1(True)

        if found:
            W.store(selected + count, W.cast(best_index, W.u32))
            W.store(suppressed + best_index, W.u8(1))
            winner_x1 = W.load(
                boxes + best_index * 4 + 0, other=W.f32(0.0)
            )
            winner_y1 = W.load(
                boxes + best_index * 4 + 1, other=W.f32(0.0)
            )
            winner_x2 = W.load(
                boxes + best_index * 4 + 2, other=W.f32(0.0)
            )
            winner_y2 = W.load(
                boxes + best_index * 4 + 3, other=W.f32(0.0)
            )
            winner_area = W.maximum(
                winner_x2 - winner_x1, W.f32(0.0)
            ) * W.maximum(winner_y2 - winner_y1, W.f32(0.0))
            with W.vla(0, candidates) as candidate:
                active = (
                    W.load(suppressed + candidate, other=W.u8(1))
                    == W.u8(0)
                )
                x1 = W.load(
                    boxes + candidate * 4 + 0, other=W.f32(0.0)
                )
                y1 = W.load(
                    boxes + candidate * 4 + 1, other=W.f32(0.0)
                )
                x2 = W.load(
                    boxes + candidate * 4 + 2, other=W.f32(0.0)
                )
                y2 = W.load(
                    boxes + candidate * 4 + 3, other=W.f32(0.0)
                )
                right = W.minimum(winner_x2, x2)
                left = W.maximum(winner_x1, x1)
                bottom = W.minimum(winner_y2, y2)
                top = W.maximum(winner_y1, y1)
                intersection_width = W.maximum(right - left, W.f32(0.0))
                intersection_height = W.maximum(bottom - top, W.f32(0.0))
                intersection_area = intersection_width * intersection_height
                candidate_area = W.maximum(
                    x2 - x1, W.f32(0.0)
                ) * W.maximum(y2 - y1, W.f32(0.0))
                union_area = winner_area + candidate_area - intersection_area
                overlap_ratio = intersection_area / union_area
                suppress = active & (overlap_ratio > iou_threshold)
                W.store(
                    suppressed + candidate,
                    W.u8(1),
                    where=suppress,
                )
            count = count + 1
        else:
            count = maximum_selected
