# Lowering 与 Artifact

## Target lowering

Target lowering 的完整事实来源只有：

```text
Canonical Kernel IR + static target facts + explicit backend config
```

它在一次调用内完成 capability、legality、resource calculation、物理参数选择与 emission。
这些机制可以拆成内部 C++ 模块，但不能各自成为长期 stage、IR dialect 或可从外部注入的
authority。

Target lowering 可以决定：

- VLA 的动态 `vl` mechanics、SEW、LMUL 与 unroll；
- unit-stride、strided、indexed、segment 或 scalar memory realization；
- reduction tree、widening accumulator 与跨 strip state；
- contract 的 register microtile、K unroll、fragment 与短生命周期 packing；
- decode、lookup、repack 与 contract 的局部 fusion；
- RVV intrinsic、IME 或 vendor-extension inline asm leaf；
- 由 backend config 明确绑定的物理参数。

它不得：

- 根据 kernel 名、算子名、q-format 名或整体 shape 选择完整模板；
- 从 tensor shape 猜 source loop、state boundary 或 contraction identity；
- 创建 source 中不存在的 cache blocking、staging 或 persistent layout；
- 从 `materials/`、GGML、旧 route registry 或旧 emitter 读取运行时代码；
- 对 unsupported primitive 使用 catch-all、默认 scalar 或 legacy fallback。

## Intrinsic C 与 inline asm

正常输出是可读的 C translation unit：

- 普通 ABI、scalar control 与 pointer arithmetic 使用 C11；
- RVV 使用 `<riscv_vector.h>` intrinsic；
- 只有 system intrinsic 无法表达的扩展 leaf 使用局部 inline asm；
- 生成 kernel 内不得出现 `std::vector`、动态 tensor wrapper、逐元素临时容器或 C++ runtime；
- local stack/scratch 只有在 target lowering 明确选择且资源合法时存在。

Inline asm 只能实现一个 typed primitive/fragment leaf，不能接管完整 kernel outer loops。

## Artifact 类型

唯一主链必须能够形成：

```text
canonical Kernel MLIR
readable intrinsic C / necessary inline asm
relocatable object
static library
C-compatible public header
```

Object 与 library 由系统 C compiler 和 archiver 从同一份 generated source 产生，不建立新的
compiler IR stage。

## Runtime ABI

生成 entry 具有普通 C ABI。实际参数由 kernel source 决定，例如：

```c
void rms_norm_worker(
    const float *restrict x,
    float *restrict y,
    size_t row_begin,
    size_t row_end,
    size_t cols,
    size_t stride,
    float eps);
```

Weft artifact 不创建线程。外部 runtime 负责 worker ranges、thread pool、affinity 和多核调度。

## Extension 与 multiversion

一个 target 可以构建多个 object variant；轻量 AOT dispatcher 可以根据 shape predicate、
alignment、stride、fixed VLEN 或 extension fact 选择已经生成的 entry。Dispatcher 不管理
线程，不调用 compiler，也不是 unsupported lowering 的 fallback。
