import weft
import weft.language as W


@weft.kernel
def codebook_lookup_affine_f32(
    codes: W.ptr[W.u8, W.readonly, W.noalias],
    codebook: W.ptr[W.f32, W.readonly, W.noalias],
    scale: W.ptr[W.f32, W.readonly, W.noalias],
    bias: W.ptr[W.f32, W.readonly, W.noalias],
    output: W.ptr[W.f32, W.writeonly, W.noalias],
    elements: W.index,
) -> None:
    table_axis = W.axis(16)
    table = W.load(codebook + table_axis)
    with W.vla(0, elements) as index:
        code = W.load(codes + index)
        decoded = W.lookup(table, code)
        W.store(output + index, decoded * W.load(scale + index) + W.load(bias + index))


@weft.kernel
def codebook_lookup_affine_f32_equivalent(
    codes: W.ptr[W.u8, W.readonly, W.noalias],
    codebook: W.ptr[W.f32, W.readonly, W.noalias],
    scale: W.ptr[W.f32, W.readonly, W.noalias],
    bias: W.ptr[W.f32, W.readonly, W.noalias],
    output: W.ptr[W.f32, W.writeonly, W.noalias],
    elements: W.index,
) -> None:
    lane = W.axis(16)
    table = W.load(codebook + lane)
    with W.vla(0, elements) as index:
        code_address = codes + index
        code = W.load(code_address)
        value = W.lookup(table, code, where=W.i1(True))
        product = W.load(scale + index) * value
        result = W.load(bias + index) + product
        W.store(output + index, result)
