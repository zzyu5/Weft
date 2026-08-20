from __future__ import annotations

import weft
from weft.language import View, f16
from weft.std import flash_attention


@weft.kernel
def online_flash_attention(
    Q: View[f16, (Tq, D)],
    K: View[f16, (Tk, D)],
    V: View[f16, (Tk, D)],
    O: View[f16, (Tq, D)],
):
    flash_attention(Q, K, V, O)
