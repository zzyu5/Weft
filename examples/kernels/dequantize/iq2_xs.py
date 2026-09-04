from __future__ import annotations

import formats.ggml as ggml
import functions.quantization as qf
import weft
import weft.language as wl


@weft.kernel
def row_dequantize_iq2_xs(
    W: wl.View[ggml.IQ2_XS, (K,)],
    grid: wl.View[wl.i8, (4096,)],
    signs: wl.View[wl.i8, (1024,)],
    Y: wl.View[wl.f32, (K,)],
):
    with wl.L.blocks(K, extent=256) as kb:
        w = wl.materialize(wl.admit(W[kb]))
        d = wl.materialize(wl.f32(w.d))
        group = wl.index(0)
        with wl.L.subs(kb, extent=32) as group_point:
            scale_byte = wl.materialize(w.scales[group_point])
            scale_low = wl.materialize(
                d
                * (wl.f32(0.5) + wl.f32(qf.extract_bits(scale_byte, 0, 4)))
                * wl.f32(0.25)
            )
            scale_high = wl.materialize(
                d
                * (wl.f32(0.5) + wl.f32(qf.extract_bits(scale_byte, 4, 4)))
                * wl.f32(0.25)
            )
            entry = wl.index(0)
            with wl.L.subs(group_point, extent=8) as entry_point:
                lane = wl.iota(8, dtype=wl.u8, axis="k")
                word = w.q[group * wl.index(4) + entry]
                grid_index = wl.narrow(
                    qf.extract_bits(word, 0, 9),
                    wl.u16,
                    rounding="rtz",
                    saturation=False,
                )
                sign_index = wl.narrow(
                    qf.extract_bits(word, 9, 7),
                    wl.u16,
                    rounding="rtz",
                    saturation=False,
                )
                scale = scale_low
                if entry >= wl.index(2):
                    scale = scale_high
                grid_value = wl.lookup(
                    grid, grid_index * wl.u16(8) + wl.u16(lane), bounds="in_bounds"
                )
                sign_value = wl.lookup(
                    signs, sign_index * wl.u16(8) + wl.u16(lane), bounds="in_bounds"
                )
                wl.commit(
                    qf.iq_codebook(wl.i32(grid_value) * wl.i32(sign_value), scale),
                    Y[kb][group_point][entry_point],
                )
                entry += wl.index(1)
            group += wl.index(1)
