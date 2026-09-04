from __future__ import annotations

import formats.ggml as ggml
import functions.quantization as qf
import weft
import weft.language as wl


def compute(
    W: wl.View[ggml.IQ1_M, (K,)],
    X: wl.View[ggml.Q8_K, (K,)],
    grid: wl.View[ggml.I8X8, (2048, 8)],
    f16_bits: wl.View[wl.f32, (65536,)],
):
    result = wl.f32(0.0)
    grid_values = grid.values
    with wl.L.blocks(K, extent=256) as kb:
        w = wl.admit(W[kb])
        x = wl.admit(X[kb])
        sc0 = wl.u32(w.scales[0]) | wl.u32(w.scales[1]) << wl.u32(8)
        sc1 = wl.u32(w.scales[2]) | wl.u32(w.scales[3]) << wl.u32(8)
        sc2 = wl.u32(w.scales[4]) | wl.u32(w.scales[5]) << wl.u32(8)
        sc3 = wl.u32(w.scales[6]) | wl.u32(w.scales[7]) << wl.u32(8)
        scale_bits = sc0 >> wl.u32(12) | sc1 >> wl.u32(8) & wl.u32(240)
        scale_bits = scale_bits | sc2 >> wl.u32(4) & wl.u32(3840)
        scale_bits = scale_bits | sc3 & wl.u32(61440)
        block_scale = qf.nonlinear_lookup(f16_bits, scale_bits)
        group = wl.iota(8, dtype=wl.u32, axis="group")
        scale_part = wl.iota(2, dtype=wl.u32, axis="scale_part")
        entry = wl.iota(2, dtype=wl.u32, axis="entry")
        payload = wl.iota(8, dtype=wl.u32, axis="payload")
        qh = wl.widen(w.qh[group * wl.u32(2) + scale_part], wl.u32)
        high_shift = entry * wl.u32(4)
        grid_index = wl.widen(
            w.q[group * wl.u32(4) + scale_part * wl.u32(2) + entry], wl.u32
        ) | (qh >> high_shift & wl.u32(7)) << wl.u32(8)
        weight = wl.lookup(
            grid_values, grid_index * wl.u32(8) + payload, bounds="in_bounds"
        )
        activation = x.q[
            group * wl.u32(32) + scale_part * wl.u32(16) + entry * wl.u32(8) + payload
        ]
        dot = wl.contract(activation, weight, over=("entry", "payload"), acc=wl.i32)
        scale_word = wl.widen(
            w.scales[group // wl.u32(2) * wl.u32(2)], wl.u32
        ) | wl.widen(
            w.scales[group // wl.u32(2) * wl.u32(2) + wl.u32(1)], wl.u32
        ) << wl.u32(
            8
        )
        scale_shift = group % wl.u32(2) * wl.u32(6) + scale_part * wl.u32(3)
        scale = wl.i32((scale_word >> scale_shift & wl.u32(7)) * wl.u32(2) + wl.u32(1))
        main = wl.reduce(wl.reduce(dot * scale, axis="scale_part"), axis="group")
        correction = wl.i32(0)
        for correction_group in range(8):
            correction_part0 = wl.i32(0)
            correction_part1 = wl.i32(0)
            for correction_entry in range(4):
                correction_qh = wl.u32(
                    w.qh[correction_group * 2 + correction_entry // 2]
                )
                correction_delta = wl.i32(1)
                correction_delta_shift = 3 + correction_entry % 2 * 4
                if correction_qh >> wl.u32(correction_delta_shift) & wl.u32(
                    1
                ) != wl.u32(0):
                    correction_delta = wl.i32(-1)
                correction_sum = wl.i32(0)
                for correction_lane in range(8):
                    correction_sum += wl.i32(
                        x.q[
                            correction_group * 32
                            + correction_entry * 8
                            + correction_lane
                        ]
                    )
                if correction_entry < 2:
                    correction_part0 += correction_sum * correction_delta
                if correction_entry >= 2:
                    correction_part1 += correction_sum * correction_delta
            correction_word = wl.u32(w.scales[2 * (correction_group // 2)]) | wl.u32(
                w.scales[2 * (correction_group // 2) + 1]
            ) << wl.u32(8)
            correction_shift = correction_group % 2 * 6
            correction_scale0 = wl.i32(
                (correction_word >> wl.u32(correction_shift) & wl.u32(7)) * wl.u32(2)
                + wl.u32(1)
            )
            correction_scale1 = wl.i32(
                (correction_word >> wl.u32(correction_shift + 3) & wl.u32(7))
                * wl.u32(2)
                + wl.u32(1)
            )
            correction += (
                correction_part0 * correction_scale0
                + correction_part1 * correction_scale1
            )
        combined = wl.f32(main) + wl.f32(0.125) * wl.f32(correction)
        result += wl.f32(block_scale) * wl.f32(x.ds) * combined
    return result


@weft.kernel
def quantized_vec_dot_iq1_m_q8_k(
    W: wl.View[ggml.IQ1_M, (K,)],
    X: wl.View[ggml.Q8_K, (K,)],
    grid: wl.View[ggml.I8X8, (2048, 8)],
    f16_bits: wl.View[wl.f32, (65536,)],
    Y: wl.View[wl.f32, (1,)],
):
    wl.commit(compute(W, X, grid, f16_bits), Y[0])
