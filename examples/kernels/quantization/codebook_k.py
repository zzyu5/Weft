import weft
import weft.language as W

@W.helper(effects=("read",))
def load_f32_le(pointer):
    b0 = W.cast(W.load(pointer, other=W.u8(0)), W.u32)
    b1 = W.cast(W.load(pointer + 1, other=W.u8(0)), W.u32)
    b2 = W.cast(W.load(pointer + 2, other=W.u8(0)), W.u32)
    b3 = W.cast(W.load(pointer + 3, other=W.u8(0)), W.u32)
    return W.bitcast(b0 | (b1 << W.u32(8)) | (b2 << W.u32(16)) | (b3 << W.u32(24)), W.f32)


@weft.kernel
def iq2_S_q8_K(
    weight: W.ptr[W.u8, W.readonly, W.noalias],
    activation: W.ptr[W.u8, W.readonly, W.noalias],
    blocks: W.index,
) -> W.f32:
    result = W.f32(0.0)
    for block in W.range(0, blocks):
        x = weight + block * W.index(82)
        y = activation + block * W.index(292)
        code_axis = W.block(32)
        high_axis = W.block(8)
        sign_axis = W.block(32)
        scale_axis = W.block(8)
        activation_axis = W.block(256)
        result = W.iq2_s_i8_dot(
            W.load(x + 2 + code_axis, other=W.u8(0)),
            W.load(x + 66 + high_axis, other=W.u8(0)),
            W.load(x + 34 + sign_axis, other=W.u8(0)),
            W.load(x + 74 + scale_axis, other=W.u8(0)),
            W.bitcast(W.load(y + 4 + activation_axis, other=W.u8(0)), W.i8),
            W.load_f16_le(x),
            load_f32_le(y),
            result,
        )
    return result


@weft.kernel
def iq3_S_q8_K(
    weight: W.ptr[W.u8, W.readonly, W.noalias],
    activation: W.ptr[W.u8, W.readonly, W.noalias],
    blocks: W.index,
) -> W.f32:
    result = W.f32(0.0)
    for block in W.range(0, blocks):
        x = weight + block * W.index(110)
        y = activation + block * W.index(292)
        code_axis = W.block(64)
        high_axis = W.block(8)
        sign_axis = W.block(32)
        scale_axis = W.block(4)
        activation_axis = W.block(256)
        result = W.iq3_s_i8_dot(
            W.load(x + 2 + code_axis, other=W.u8(0)),
            W.load(x + 66 + high_axis, other=W.u8(0)),
            W.load(x + 74 + sign_axis, other=W.u8(0)),
            W.load(x + 106 + scale_axis, other=W.u8(0)),
            W.bitcast(W.load(y + 4 + activation_axis, other=W.u8(0)), W.i8),
            W.load_f16_le(x),
            load_f32_le(y),
            result,
        )
    return result


@weft.kernel
def iq1_M_q8_K(
    weight: W.ptr[W.u8, W.readonly, W.noalias],
    activation: W.ptr[W.u8, W.readonly, W.noalias],
    blocks: W.index,
) -> W.f32:
    result = W.f32(0.0)
    for block in W.range(0, blocks):
        x = weight + block * W.index(56)
        y = activation + block * W.index(292)
        code_axis = W.block(32)
        high_axis = W.block(16)
        scale_axis = W.block(8)
        activation_axis = W.block(256)
        result = W.iq1_m_i8_dot(
            W.load(x + code_axis, other=W.u8(0)),
            W.load(x + 32 + high_axis, other=W.u8(0)),
            W.load(x + 48 + scale_axis, other=W.u8(0)),
            W.bitcast(W.load(y + 4 + activation_axis, other=W.u8(0)), W.i8),
            load_f32_le(y),
            result,
        )
    return result


@weft.kernel
def q6_K_q8_K(
    weight: W.ptr[W.u8, W.readonly, W.noalias],
    activation: W.ptr[W.u8, W.readonly, W.noalias],
    blocks: W.index,
) -> W.f32:
    result = W.f32(0.0)
    for block in W.range(0, blocks):
        x = weight + block * W.index(210)
        y = activation + block * W.index(292)
        low_axis = W.block(128)
        high_axis = W.block(64)
        scale_axis = W.block(16)
        activation_axis = W.block(256)
        result = W.q6_k_i8_dot(
            W.load(x + low_axis, other=W.u8(0)),
            W.load(x + 128 + high_axis, other=W.u8(0)),
            W.bitcast(W.load(x + 192 + scale_axis, other=W.u8(0)), W.i8),
            W.bitcast(W.load(y + 4 + activation_axis, other=W.u8(0)), W.i8),
            W.load_f16_le(x + 208),
            load_f32_le(y),
            result,
        )
    return result
