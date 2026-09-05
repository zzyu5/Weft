from __future__ import annotations

import formats.ggml as ggml
import kernels.quantize.q8_k as quant_q8_k
import weft
import weft.language as wl


def _iq2_xxs_group_products(w, x, grid, signs, entry_lane, codebook_lane, group):
    group_byte = group * 8
    grid_index = wl.widen(w.q[:, group_byte + entry_lane], wl.u32)
    word1 = wl.widen(w.q[:, group_byte + 4], wl.u32) | wl.widen(
        w.q[:, group_byte + 5], wl.u32
    ) << wl.u32(8)
    word1 = word1 | wl.widen(w.q[:, group_byte + 6], wl.u32) << wl.u32(16)
    word1 = word1 | wl.widen(w.q[:, group_byte + 7], wl.u32) << wl.u32(24)
    sign_index = word1 >> wl.u32(entry_lane * 7) & wl.u32(127)
    weight = wl.lookup(grid, grid_index * wl.u32(8) + codebook_lane, bounds="in_bounds")
    sign = wl.lookup(signs, sign_index * wl.u32(8) + codebook_lane, bounds="in_bounds")
    signed_weight = weight * sign
    activation = x.q[:, group * 32 + entry_lane * 8 + codebook_lane]
    integer = wl.reduce_dot(activation, signed_weight, over=("entry", "payload"), acc_dtype=wl.i32)
    scale = wl.i32((word1 >> wl.u32(28)) * wl.u32(2) + wl.u32(1))
    return integer * scale


@weft.kernel
def production_mul_mat_iq2_xxs_staged(
    W: wl.View[ggml.IQ2_XXS, (N, K)],
    X: wl.View[wl.f32, (M, K)],
    Xq: wl.View[ggml.Q8_K, (M, K)],
    grid: wl.View[wl.i8, (2048,)],
    signs: wl.View[wl.i8, (1024,)],
    Y: wl.View[wl.f32, (M, N)],
):
    quant_q8_k.quantize_matrix(X, Xq)
    for row in range(M):
        for column in range(N):
            wl.store(Y[row, column], wl.f32(0.0))
    with wl.level.tiles(N, extent=wl.auto("NC")) as nc:
        with wl.level.tiles(K, extent=wl.auto("KC")) as kc:
            wp = wl.stage(wl.load(W[nc, kc]))
            with wl.level.tiles(M, extent=wl.auto("MC")) as mc:
                with wl.level.rows(mc, group=wl.auto("MR")) as mb:
                    with wl.level.cols(nc, group=wl.auto("NR")) as nb:
                        f32_acc = wl.state(wl.f32, [MR, NR], init=wl.load(Y[mb, nb]))
                        with wl.level.blocks(kc, extent=256) as kb:
                            w = wp[nb, kb]
                            x = wl.load(Xq[mb, kb])
                            entry_lane = wl.arange(0, 4, axis="entry")
                            codebook_lane = wl.arange(0, 8, axis="payload")
                            block_sum = wl.state(wl.i32, [MR, NR], init=wl.i32(0))
                            group_index = wl.index(0)
                            with wl.level.subtiles(kb, extent=32) as group:
                                group_partial = _iq2_xxs_group_products(
                                    w,
                                    x,
                                    grid,
                                    signs,
                                    entry_lane,
                                    codebook_lane,
                                    group_index,
                                )
                                block_sum += group_partial
                                group_index += wl.index(1)
                            f32_acc += (
                                wl.f32(w.d) * wl.f32(x.ds) * wl.widen(block_sum, wl.f32)
                            )
                        wl.store(Y[mb, nb], f32_acc)
    with wl.level.tiles(N, extent=wl.auto("NC")) as nc:
        with wl.level.tiles(M, extent=wl.auto("MC")) as mc:
            with wl.level.rows(mc, group=wl.auto("MR")) as mb:
                with wl.level.cols(nc, group=wl.auto("NR")) as nb:
                    value = wl.load(Y[mb, nb])
                    wl.store(Y[mb, nb], wl.f32(0.125) * value)
