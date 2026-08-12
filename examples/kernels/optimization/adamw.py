import weft
import weft.language as W


@weft.kernel
def adamw_f32(
    parameters: W.ptr[W.f32, W.noalias],
    gradients: W.ptr[W.f32, W.readonly, W.noalias],
    first_moment: W.ptr[W.f32, W.noalias],
    second_moment: W.ptr[W.f32, W.noalias],
    begin: W.index,
    end: W.index,
    learning_rate: W.f32,
    beta1: W.f32,
    beta2: W.f32,
    inverse_bias1: W.f32,
    inverse_bias2: W.f32,
    epsilon: W.f32,
    weight_decay: W.f32,
) -> None:
    with W.vla(begin, end) as index:
        parameter = W.load(parameters + index, other=W.f32(0.0))
        gradient = W.load(gradients + index, other=W.f32(0.0))
        old_first = W.load(first_moment + index, other=W.f32(0.0))
        old_second = W.load(second_moment + index, other=W.f32(0.0))
        next_first = beta1 * old_first + (W.f32(1.0) - beta1) * gradient
        next_second = beta2 * old_second + (
            W.f32(1.0) - beta2
        ) * gradient * gradient
        corrected_first = next_first * inverse_bias1
        corrected_second = next_second * inverse_bias2
        update = corrected_first / (W.sqrt(corrected_second) + epsilon)
        next_parameter = parameter - learning_rate * (
            update + weight_decay * parameter
        )
        W.store(first_moment + index, next_first)
        W.store(second_moment + index, next_second)
        W.store(parameters + index, next_parameter)


@weft.kernel
def adamw_f32_equivalent(
    parameters: W.ptr[W.f32, W.noalias],
    gradients: W.ptr[W.f32, W.readonly, W.noalias],
    first_moment: W.ptr[W.f32, W.noalias],
    second_moment: W.ptr[W.f32, W.noalias],
    begin: W.index,
    end: W.index,
    learning_rate: W.f32,
    beta1: W.f32,
    beta2: W.f32,
    inverse_bias1: W.f32,
    inverse_bias2: W.f32,
    epsilon: W.f32,
    weight_decay: W.f32,
) -> None:
    one_minus_beta1 = W.f32(1.0) - beta1
    one_minus_beta2 = W.f32(1.0) - beta2
    with W.vla(begin, end) as index:
        parameter_pointer = parameters + index
        first_pointer = first_moment + index
        second_pointer = second_moment + index
        parameter = W.load(parameter_pointer, other=W.f32(0.0))
        gradient = W.load(gradients + index, other=W.f32(0.0))
        next_first = W.load(first_pointer, other=W.f32(0.0)) * beta1 + (
            gradient * one_minus_beta1
        )
        next_second = W.load(second_pointer, other=W.f32(0.0)) * beta2 + (
            gradient * gradient * one_minus_beta2
        )
        denominator = W.sqrt(next_second * inverse_bias2) + epsilon
        decayed = parameter * weight_decay
        update = next_first * inverse_bias1 / denominator + decayed
        W.store(first_pointer, next_first)
        W.store(second_pointer, next_second)
        W.store(parameter_pointer, parameter - learning_rate * update)
