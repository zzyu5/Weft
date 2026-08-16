import weft
import weft.language as W

@W.helper(effects=("read",))
def load_f32_le(pointer):
    b0 = W.cast(W.load(pointer, other=W.u8(0)), W.u32)
    b1 = W.cast(W.load(pointer + 1, other=W.u8(0)), W.u32)
    b2 = W.cast(W.load(pointer + 2, other=W.u8(0)), W.u32)
    b3 = W.cast(W.load(pointer + 3, other=W.u8(0)), W.u32)
    return W.bitcast(b0 | (b1 << W.u32(8)) | (b2 << W.u32(16)) | (b3 << W.u32(24)), W.f32)


@W.helper(effects=("read",))
def load_i16_le(pointer):
    low = W.cast(W.load(pointer, other=W.u8(0)), W.u16)
    high = W.cast(W.load(pointer + W.index(1), other=W.u8(0)), W.u16)
    return W.cast(W.bitcast(low | (high << W.u16(8)), W.i16), W.i32)


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


@weft.kernel
def iq4_nl_q8_0(
    weight: W.ptr[W.u8, W.readonly, W.noalias],
    activation: W.ptr[W.u8, W.readonly, W.noalias],
    codebook: W.ptr[W.u8, W.readonly, W.noalias],
    output: W.ptr[W.f32, W.writeonly, W.noalias],
    row_begin: W.index,
    row_end: W.index,
    elements: W.index,
) -> None:
    blocks = elements / W.index(32)
    weight_row_stride = blocks * W.index(18)
    for row in W.range(row_begin, row_end):
        result = W.f32(0.0)
        for block in W.range(0, blocks):
            x = weight + row * weight_row_stride + block * W.index(18)
            y = activation + block * W.index(34)
            member = W.block(16)
            packed_codes = W.load(x + W.index(2) + member, other=W.u8(0))
            table = W.bitcast(
                W.load(codebook + member, other=W.u8(0)), W.i8
            )
            low = W.decode(packed_codes & W.u8(15), table, out_dtype=W.i8)
            high = W.decode(packed_codes >> W.u8(4), table, out_dtype=W.i8)
            activation_low = W.bitcast(
                W.load(y + W.index(2) + member, other=W.u8(0)), W.i8
            )
            activation_high = W.bitcast(
                W.load(y + W.index(18) + member, other=W.u8(0)), W.i8
            )
            low_sum = W.reduce(
                W.cast(low, W.i32) * W.cast(activation_low, W.i32),
                identity=W.i32(0),
                axis=0,
                acc_dtype=W.i32,
                order="relaxed",
            )
            high_sum = W.reduce(
                W.cast(high, W.i32) * W.cast(activation_high, W.i32),
                identity=W.i32(0),
                axis=0,
                acc_dtype=W.i32,
                order="relaxed",
            )
            scale = W.load_f16_le(x) * W.load_f16_le(y)
            result = result + scale * W.cast(low_sum + high_sum, W.f32)
        W.store(output + row, result)


@weft.kernel
def iq4_xs_q8_K(
    weight: W.ptr[W.u8, W.readonly, W.noalias],
    activation: W.ptr[W.u8, W.readonly, W.noalias],
    codebook: W.ptr[W.u8, W.readonly, W.noalias],
    output: W.ptr[W.f32, W.writeonly, W.noalias],
    row_begin: W.index,
    row_end: W.index,
    elements: W.index,
) -> None:
    blocks = elements / W.index(256)
    weight_row_stride = blocks * W.index(136)
    for row in W.range(row_begin, row_end):
        result = W.f32(0.0)
        for block in W.range(0, blocks):
            x = weight + row * weight_row_stride + block * W.index(136)
            y = activation + block * W.index(292)
            block_scale = W.load_f16_le(x)
            activation_scale = load_f32_le(y)
            scales_high = W.cast(W.load(x + W.index(2), other=W.u8(0)), W.u16) | (
                W.cast(W.load(x + W.index(3), other=W.u8(0)), W.u16)
                << W.u16(8)
            )
            for group in W.range(0, 8):
                scales_low = W.load(
                    x + W.index(4) + group // W.index(2), other=W.u8(0)
                )
                low_shift = W.cast(
                    (group % W.index(2)) * W.index(4), W.u8
                )
                low_scale = (scales_low >> low_shift) & W.u8(15)
                high_shift = W.cast(group * W.index(2), W.u16)
                high_scale = W.cast(
                    (scales_high >> high_shift) & W.u16(3), W.u8
                )
                local_scale = W.cast(
                    low_scale | (high_scale << W.u8(4)), W.f32
                ) - W.f32(32.0)
                member = W.block(16)
                packed_codes = W.load(
                    x + W.index(8) + group * W.index(16) + member,
                    other=W.u8(0),
                )
                table = W.bitcast(
                    W.load(codebook + member, other=W.u8(0)), W.i8
                )
                low = W.decode(
                    packed_codes & W.u8(15), table, out_dtype=W.i8
                )
                high = W.decode(
                    packed_codes >> W.u8(4), table, out_dtype=W.i8
                )
                activation_low = W.bitcast(
                    W.load(
                        y + W.index(4) + group * W.index(32) + member,
                        other=W.u8(0),
                    ),
                    W.i8,
                )
                activation_high = W.bitcast(
                    W.load(
                        y + W.index(20) + group * W.index(32) + member,
                        other=W.u8(0),
                    ),
                    W.i8,
                )
                low_sum = W.reduce(
                    W.cast(low, W.i32) * W.cast(activation_low, W.i32),
                    identity=W.i32(0),
                    axis=0,
                    acc_dtype=W.i32,
                    order="relaxed",
                )
                high_sum = W.reduce(
                    W.cast(high, W.i32) * W.cast(activation_high, W.i32),
                    identity=W.i32(0),
                    axis=0,
                    acc_dtype=W.i32,
                    order="relaxed",
                )
                result = result + (
                    block_scale
                    * activation_scale
                    * local_scale
                    * W.cast(low_sum + high_sum, W.f32)
                )
        W.store(output + row, result)


@weft.kernel
def iq2_xxs_q8_K(
    weight: W.ptr[W.u8, W.readonly, W.noalias],
    activation: W.ptr[W.u8, W.readonly, W.noalias],
    grid_table: W.ptr[W.u8, W.readonly, W.noalias],
    sign_table: W.ptr[W.u8, W.readonly, W.noalias],
    blocks: W.index,
) -> W.f32:
    result = W.f32(0.0)
    for block in W.range(0, blocks):
        x = weight + block * W.index(66)
        y = activation + block * W.index(292)
        block_scale = W.load_f16_le(x) * load_f32_le(y) * W.f32(0.125)
        for group in W.range(0, 8):
            metadata = W.cast(W.load(x + W.index(6) + group * W.index(8), other=W.u8(0)), W.u32)
            metadata = metadata | (W.cast(W.load(x + W.index(7) + group * W.index(8), other=W.u8(0)), W.u32) << W.u32(8))
            metadata = metadata | (W.cast(W.load(x + W.index(8) + group * W.index(8), other=W.u8(0)), W.u32) << W.u32(16))
            metadata = metadata | (W.cast(W.load(x + W.index(9) + group * W.index(8), other=W.u8(0)), W.u32) << W.u32(24))
            member = W.block(32)
            field = member // W.index(8)
            lane = member % W.index(8)
            grid_code = W.load(
                x + W.index(2) + group * W.index(8) + field,
                other=W.u8(0),
            )
            sign0 = W.load(sign_table + W.cast(metadata & W.u32(127), W.index), other=W.u8(0))
            sign1 = W.load(sign_table + W.cast((metadata >> W.u32(7)) & W.u32(127), W.index), other=W.u8(0))
            sign2 = W.load(sign_table + W.cast((metadata >> W.u32(14)) & W.u32(127), W.index), other=W.u8(0))
            sign3 = W.load(sign_table + W.cast((metadata >> W.u32(21)) & W.u32(127), W.index), other=W.u8(0))
            zero = grid_code ^ grid_code
            signs = W.select(
                field == W.index(0),
                zero | sign0,
                W.select(
                    field == W.index(1),
                    zero | sign1,
                    W.select(field == W.index(2), zero | sign2, zero | sign3),
                ),
            )
            grid_offset = W.cast(grid_code, W.index) * W.index(8) + lane
            grid = W.load(grid_table + grid_offset, other=W.u8(0))
            sign_bit = (signs >> W.cast(lane, W.u8)) & W.u8(1)
            signed_grid = W.cast(grid, W.i32) * (
                W.cast(sign_bit, W.i32) * W.i32(-2) + W.i32(1)
            )
            activation_values = W.cast(
                W.bitcast(
                    W.load(
                        y + W.index(4) + group * W.index(32) + member,
                        other=W.u8(0),
                    ),
                    W.i8,
                ),
                W.i32,
            )
            integer_sum = W.reduce(
                signed_grid * activation_values,
                identity=W.i32(0),
                axis=0,
                acc_dtype=W.i32,
                order="relaxed",
            )
            local_scale = W.f32(1.0) + W.f32(2.0) * W.cast(
                metadata >> W.u32(28), W.f32
            )
            result = result + block_scale * local_scale * W.cast(integer_sum, W.f32)
    return result


@weft.kernel
def iq2_xs_q8_K(
    weight: W.ptr[W.u8, W.readonly, W.noalias],
    activation: W.ptr[W.u8, W.readonly, W.noalias],
    grid_table: W.ptr[W.u8, W.readonly, W.noalias],
    sign_table: W.ptr[W.u8, W.readonly, W.noalias],
    blocks: W.index,
) -> W.f32:
    result = W.f32(0.0)
    for block in W.range(0, blocks):
        x = weight + block * W.index(74)
        y = activation + block * W.index(292)
        block_scale = W.load_f16_le(x) * load_f32_le(y) * W.f32(0.125)
        for group in W.range(0, 8):
            scale_byte = W.load(x + W.index(66) + group, other=W.u8(0))
            member = W.block(32)
            field = member // W.index(8)
            lane = member % W.index(8)
            packed_offset = field * W.index(2)
            code_low = W.load(
                x + W.index(2) + group * W.index(8) + packed_offset,
                other=W.u8(0),
            )
            code_high = W.load(
                x + W.index(3) + group * W.index(8) + packed_offset,
                other=W.u8(0),
            )
            low_index = W.cast(code_low, W.index)
            high_index = W.cast(code_high, W.index)
            grid_index = low_index | ((high_index & W.index(1)) << W.index(8))
            sign_index = high_index >> W.index(1)
            grid_offset = grid_index * W.index(8) + lane
            grid = W.load(grid_table + grid_offset, other=W.u8(0))
            signs = W.load(sign_table + sign_index, other=W.u8(0))
            sign_bit = (signs >> W.cast(lane, W.u8)) & W.u8(1)
            scale_vector = (code_low ^ code_low) | scale_byte
            scale_shift = W.cast(
                (field // W.index(2)) * W.index(4), W.u8
            )
            local_scale = (
                W.cast((scale_vector >> scale_shift) & W.u8(15), W.i32)
                * W.i32(2)
                + W.i32(1)
            )
            signed_grid = W.cast(grid, W.i32) * (
                W.cast(sign_bit, W.i32) * W.i32(-2) + W.i32(1)
            )
            activation_values = W.cast(
                W.bitcast(
                    W.load(
                        y + W.index(4) + group * W.index(32) + member,
                        other=W.u8(0),
                    ),
                    W.i8,
                ),
                W.i32,
            )
            integer_sum = W.reduce(
                local_scale * signed_grid * activation_values,
                identity=W.i32(0),
                axis=0,
                acc_dtype=W.i32,
                order="relaxed",
            )
            result = result + block_scale * W.cast(integer_sum, W.f32)
    return result


@weft.kernel
def iq3_xxs_q8_K(
    weight: W.ptr[W.u8, W.readonly, W.noalias],
    activation: W.ptr[W.u8, W.readonly, W.noalias],
    grid_table: W.ptr[W.u8, W.readonly, W.noalias],
    sign_table: W.ptr[W.u8, W.readonly, W.noalias],
    blocks: W.index,
) -> W.f32:
    result = W.f32(0.0)
    for block in W.range(0, blocks):
        x = weight + block * W.index(98)
        y = activation + block * W.index(292)
        block_scale = W.load_f16_le(x) * load_f32_le(y) * W.f32(0.25)
        for group in W.range(0, 8):
            metadata = W.cast(W.load(x + W.index(66) + group * W.index(4), other=W.u8(0)), W.u32)
            metadata = metadata | (W.cast(W.load(x + W.index(67) + group * W.index(4), other=W.u8(0)), W.u32) << W.u32(8))
            metadata = metadata | (W.cast(W.load(x + W.index(68) + group * W.index(4), other=W.u8(0)), W.u32) << W.u32(16))
            metadata = metadata | (W.cast(W.load(x + W.index(69) + group * W.index(4), other=W.u8(0)), W.u32) << W.u32(24))
            member = W.block(32)
            field = member // W.index(8)
            subrow = (member % W.index(8)) // W.index(4)
            lane = member % W.index(4)
            code_offset = field * W.index(2) + subrow
            grid_code = W.load(
                x + W.index(2) + group * W.index(8) + code_offset,
                other=W.u8(0),
            )
            sign0 = W.load(sign_table + W.cast(metadata & W.u32(127), W.index), other=W.u8(0))
            sign1 = W.load(sign_table + W.cast((metadata >> W.u32(7)) & W.u32(127), W.index), other=W.u8(0))
            sign2 = W.load(sign_table + W.cast((metadata >> W.u32(14)) & W.u32(127), W.index), other=W.u8(0))
            sign3 = W.load(sign_table + W.cast((metadata >> W.u32(21)) & W.u32(127), W.index), other=W.u8(0))
            zero = grid_code ^ grid_code
            signs = W.select(
                field == W.index(0),
                zero | sign0,
                W.select(
                    field == W.index(1),
                    zero | sign1,
                    W.select(field == W.index(2), zero | sign2, zero | sign3),
                ),
            )
            grid_offset = W.cast(grid_code, W.index) * W.index(4) + lane
            grid = W.load(grid_table + grid_offset, other=W.u8(0))
            sign_bit = (
                signs >> W.cast(member % W.index(8), W.u8)
            ) & W.u8(1)
            signed_grid = W.cast(grid, W.i32) * (
                W.cast(sign_bit, W.i32) * W.i32(-2) + W.i32(1)
            )
            activation_values = W.cast(
                W.bitcast(
                    W.load(
                        y + W.index(4) + group * W.index(32) + member,
                        other=W.u8(0),
                    ),
                    W.i8,
                ),
                W.i32,
            )
            integer_sum = W.reduce(
                signed_grid * activation_values,
                identity=W.i32(0),
                axis=0,
                acc_dtype=W.i32,
                order="relaxed",
            )
            local_scale = W.f32(1.0) + W.f32(2.0) * W.cast(
                metadata >> W.u32(28), W.f32
            )
            result = result + block_scale * local_scale * W.cast(integer_sum, W.f32)
    return result


@weft.kernel
def iq1_s_q8_K(
    weight: W.ptr[W.u8, W.readonly, W.noalias],
    activation: W.ptr[W.u8, W.readonly, W.noalias],
    grid_table: W.ptr[W.u8, W.readonly, W.noalias],
    blocks: W.index,
) -> W.f32:
    result = W.f32(0.0)
    for block in W.range(0, blocks):
        x = weight + block * W.index(50)
        y = activation + block * W.index(292)
        block_scale = W.load_f16_le(x) * load_f32_le(y)
        for group in W.range(0, 8):
            metadata_low = W.cast(
                W.load(x + W.index(34) + group * W.index(2), other=W.u8(0)),
                W.u16,
            )
            metadata_high = W.cast(
                W.load(x + W.index(35) + group * W.index(2), other=W.u8(0)),
                W.u16,
            )
            metadata = metadata_low | (metadata_high << W.u16(8))
            member = W.block(32)
            field = member // W.index(8)
            lane = member % W.index(8)
            low_index = W.load(
                x + W.index(2) + group * W.index(4) + field,
                other=W.u8(0),
            )
            high0 = W.cast(metadata & W.u16(7), W.index) << W.index(8)
            high1 = W.cast((metadata >> W.u16(3)) & W.u16(7), W.index) << W.index(8)
            high2 = W.cast((metadata >> W.u16(6)) & W.u16(7), W.index) << W.index(8)
            high3 = W.cast((metadata >> W.u16(9)) & W.u16(7), W.index) << W.index(8)
            zero_index = field - field
            selected_high = W.select(
                field == W.index(0),
                zero_index | high0,
                W.select(
                    field == W.index(1),
                    zero_index | high1,
                    W.select(
                        field == W.index(2),
                        zero_index | high2,
                        zero_index | high3,
                    ),
                ),
            )
            grid_index = W.cast(low_index, W.index) | selected_high
            grid_offset = grid_index * W.index(8) + lane
            grid = W.cast(
                W.bitcast(
                    W.load(grid_table + grid_offset, other=W.u8(0)), W.i8
                ),
                W.i32,
            )
            activation_values = W.cast(
                W.bitcast(
                    W.load(
                        y + W.index(4) + group * W.index(32) + member,
                        other=W.u8(0),
                    ),
                    W.i8,
                ),
                W.i32,
            )
            grid_sum = W.reduce(
                grid * activation_values,
                identity=W.i32(0),
                axis=0,
                acc_dtype=W.i32,
                order="relaxed",
            )
            activation_sum = load_i16_le(
                y + W.index(260) + group * W.index(4)
            ) + load_i16_le(y + W.index(262) + group * W.index(4))
            local_scale = W.f32(1.0) + W.f32(2.0) * W.cast(
                (metadata >> W.u16(12)) & W.u16(7), W.f32
            )
            delta = W.select(
                (metadata & W.u16(0x8000)) != W.u16(0),
                W.f32(-0.125),
                W.f32(0.125),
            )
            result = result + block_scale * local_scale * (
                W.cast(grid_sum, W.f32)
                + delta * W.cast(activation_sum, W.f32)
            )
    return result
