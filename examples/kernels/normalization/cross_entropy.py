import weft
import weft.language as W


@weft.kernel
def cross_entropy_loss_gradient_f32(
    logits: W.ptr[W.f32, W.readonly, W.noalias],
    labels: W.ptr[W.u32, W.readonly, W.noalias],
    gradient: W.ptr[W.f32, W.writeonly, W.noalias],
    loss: W.ptr[W.f32, W.writeonly, W.noalias],
    row_begin: W.index,
    row_end: W.index,
    classes: W.index,
) -> None:
    for row in W.range(row_begin, row_end):
        row_offset = row * classes
        with W.vla(0, classes) as class_index:
            maximum = W.reduce(
                W.load(logits + row_offset + class_index),
                op="max",
                identity=W.neg_inf(W.f32),
                axis=None,
                order="relaxed",
                acc_dtype=W.f32,
            )
        with W.vla(0, classes) as class_index:
            exponential = W.exp(
                W.load(logits + row_offset + class_index) - maximum,
                math="fast",
            )
            total = W.reduce(
                exponential,
                op="add",
                identity=W.f32(0.0),
                axis=None,
                order="relaxed",
                acc_dtype=W.f32,
            )

        label = W.cast(W.load(labels + row, other=W.u32(0)), W.index)
        label_logit = W.load(logits + row_offset + label, other=W.f32(0.0))
        W.store(loss + row, W.log(total) + maximum - label_logit)
        with W.vla(0, classes) as class_index:
            probability = W.exp(
                W.load(logits + row_offset + class_index) - maximum,
                math="fast",
            ) / total
            active_class = W.load(logits + row_offset + class_index)
            one_hot = W.select(
                class_index == label,
                W.f32(1.0),
                active_class * W.f32(0.0),
            )
            W.store(
                gradient + row_offset + class_index,
                probability - one_hot,
            )
