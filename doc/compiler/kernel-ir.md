# Canonical Weft Kernel IR

## 19. Canonical Weft Kernel IR

### 19.1 唯一算法真理

Canonical Kernel IR 保存：

- kernel ABI；
- scalar / pointer / constexpr types；
- scalar control flow；
- VLA region；
- logical block axis与 region value；
- pointer/index；
- logical predicate / masked value；
- load/store/atomic/fence；
- pointwise；
- reduce/scan/summary fold；
- contract 与 extension primitive；
- source meta-parameter；
- source location；
- numerical attributes。

它不得保存：

- exact `vl`；
- LMUL；
- register number；
- register microtile；
- provider ID；
- IME fragment；
- instruction spelling；
- build measurement；
- thread count或 launch policy。

### 19.2 核心 op 族

概念 op：

```text
weft_kernel.kernel
weft_kernel.range
weft_kernel.vla
weft_kernel.block_axis
weft_kernel.load / store / prefetch
weft_kernel.atomic / fence
weft_kernel.mask / fill
weft_kernel.reduce
weft_kernel.scan
weft_kernel.summary_fold
weft_kernel.contract
weft_kernel.reshape / transpose / broadcast
weft_kernel.cast / bitcast
weft_kernel.meta_value
```

扩展可以注册 sibling dialect 的局部 semantic op，但 `weft-compile` 的正式输入必须能独立 parse、verify 并链接所需扩展 dialect。

### 19.3 Python frontend

Python frontend 必须直接生成 canonical Kernel IR。它不得维护另一套长期 typed Python IR、另一套 verifier 或另一份算法 authority。

---
