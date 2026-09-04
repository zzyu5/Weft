from __future__ import annotations

import formats.ggml as ggml
import functions.quantization as qf
import weft
import weft.language as wl


@weft.kernel
def row_dequantize_iq3_xxs(
    W: wl.View[ggml.IQ3_XXS, (K,)],
    grid: wl.View[wl.i8, (1024,)],
    signs: wl.View[wl.i8, (1024,)],
    Y: wl.View[wl.f32, (K,)],
):
    with wl.L.blocks(K, extent=256) as kb:
        w = wl.admit(W[kb])
        group = wl.index(0)
        with wl.L.subs(kb, extent=32) as group_point:
            metadata = w.metadata[group]
            subscale = wl.f32(0.5) + wl.f32(qf.extract_bits(metadata, 28, 4))
            scale = wl.f32(w.d) * subscale * wl.f32(0.5)
            code = wl.index(0)
            with wl.L.subs(group_point, extent=4) as code_point:
                payload = wl.iota(4, dtype=wl.u32, axis="k")
                entry = code // wl.index(2)
                code_in_entry = code % wl.index(2)
                storage_coordinate = group * wl.index(8) + code
                grid_index = wl.u32(w.q[storage_coordinate])
                sign_index = qf.extract_bits(metadata, wl.u32(entry) * wl.u32(7), 7)
                grid_value = qf.nonlinear_lookup(grid, grid_index * wl.u32(4) + payload)
                sign = qf.nonlinear_lookup(
                    signs,
                    sign_index * wl.u32(8)
                    + wl.u32(code_in_entry) * wl.u32(4)
                    + payload,
                )
                wl.commit(
                    qf.iq_codebook(wl.i32(grid_value) * wl.i32(sign), scale),
                    Y[kb][group_point][code_point],
                )
                code += wl.index(1)
            group += wl.index(1)
