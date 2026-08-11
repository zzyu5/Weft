# Lowering、工具与 Artifact

## `weft-compile` 的当前边界

`weft-compile` 接收一份可独立parse/verify的canonical Weft MLIR module。当前public emit只有：

```text
--emit=kernel-ir
--emit=intrinsic-c
```

`kernel-ir` 打印canonical module；`intrinsic-c` 直接调用RISC-V target lowering。Target参数是
`--march`、`--abi`、`--vlen-bits`、`--matrix-extension`，source meta通过
`--meta=NAME=INTEGER`绑定。不存在selected IR、physical-plan输入、provider front door或
legacy emitter。

Target lowering 的完整事实来源只有：

```text
canonical Kernel IR
+ linked local extension dialect
+ static RISC-V target facts
+ explicit meta/backend bindings
```

它在一次调用内完成capability、legality、resource calculation、物理参数选择与emission。
内部C++可以按primitive family拆分，但这些机制不能成为长期stage、可单独输入的dialect或
第二份authority。

## Lowering 拥有的物理选择

Target lowering可以决定：

- VLA的dynamic `vl` mechanics、SEW、LMUL与unroll；
- unit-stride、strided、indexed、segment或scalar memory realization；
- reduction tree、widening accumulator与跨strip physical state；
- contract的register microtile、K-unroll、fragment与短生命周期packing；
- decode、lookup、repack与contract的local fusion；
- RVV intrinsic、IME或vendor-extension inline asm leaf；
- backend config显式绑定或target内部唯一合法的物理参数。

它不得：

- 根据kernel名、operator名、q-format名或整体shape选择完整模板；
- 从tensor shape猜source loop、state boundary或contraction identity；
- 创建source中不存在的cache blocking、staging、persistent layout或state algebra；
- 从`materials/`、GGML、旧route registry或旧emitter读取production代码；
- 对unsupported primitive使用catch-all、legacy或GGML fallback。

普通scalar control/memory的C lowering是canonical primitive的正式realization，不是fallback。

## Intrinsic C 与 inline asm

正常输出是可读的C translation unit：

- 普通ABI、scalar control与pointer arithmetic使用C11；
- RVV使用 `<riscv_vector.h>` intrinsic；
- system intrinsic无法表达或会破坏必要register organization的extension leaf可以使用局部
  inline asm；
- generated kernel不得出现`std::vector`、dynamic tensor wrapper、逐元素临时容器或C++
  runtime；
- local stack/scratch只在target明确选择且资源合法时存在。

一个typed leaf可以展开为多条目标指令，包括setup、decode、fragment operation和accumulator
update；inline asm边界由semantic primitive决定，不以“一条指令”计数。它不能接管完整
kernel outer loops或引入source中不存在的算法语义。

## Build-owned artifact

System C compiler与archiver从同一份generated source继续形成普通artifact：

```text
canonical Weft Kernel IR
→ readable intrinsic C / necessary inline asm
→ relocatable object
→ optional static library
→ application-owned C declaration/header
```

Object、archive与header不是新的compiler IR stage。当前 `weft-compile` CLI止于Kernel IR或
intrinsic C；`examples/run/weft.sh` 展示的是build层继续调用target system compiler并形成
executable的真实路径。Application可以从source-defined ABI维护普通declaration；不得让
header反向成为第二份kernel semantics。

## Runtime ABI

Generated entry具有普通C ABI。实际参数由kernel source决定，例如：

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

Weft artifact不创建线程。外部runtime负责worker ranges、thread pool、affinity与多核调度。

## Multiversion

同一source可以针对shape/alignment/stride/fixed VLEN/extension fact构建多个AOT object。
Lightweight dispatcher只在这些已生成entry中选择；shape predicate只能选择已声明算法的
specialized artifact，不能推断primitive、loop或state identity。Dispatcher不管理线程、
不调用compiler，也不是unsupported lowering的fallback。
