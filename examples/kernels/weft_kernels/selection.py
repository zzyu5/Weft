from __future__ import annotations

from math import inf

from weft.language import View, commit, f32, i32, new


def topk(X: View[f32, (N,)], out: View[i32, (K_,)]):
    heap = new(f32, [K_], init=-inf)
    idx = new(i32, [K_], init=-1)
    for i in range(N):
        score = X[i]
        if score > heap[K_ - 1]:
            pos = K_ - 1
            j = K_ - 1
            while j > 0:
                if score > heap[j - 1]:
                    pos = j - 1
                j -= 1
            for j in range(K_ - 1, pos, -1):
                heap[j] = heap[j - 1]
                idx[j] = idx[j - 1]
            heap[pos] = score
            idx[pos] = i32(i)
    commit(idx, out)
