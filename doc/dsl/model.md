# Weft 编程模型

## 定义

Weft 是一门面向非 SIMT 处理器的 **single-controller, multi-engine operator DSL**：

> 一份 kernel 由一个 worker/hart 从入口持续执行到返回。作者用普通有序控制流组织完整算子，
> 并在需要局部并行、规约、矩阵乘、量化或不规则访存时，向当前 core 可用的执行能力发出具有
> 完整语义的局部命令。编译器把这些命令映射为 scalar、RVV lane、寄存器重复、局部流水或
> extension fragment。

这里的 controller 是语言中的单个 worker 程序，不是运行时 command queue；engine 是
编译器可选择的 core-local 实现资源，不是用户可见的设备对象。

## 为什么需要独立模型

CPU/RISC-V 高性能 kernel 的关键作者决策通常是：

- 一个 worker 按什么顺序遍历输出、reduction 和不规则成员；
- cache blocking、workspace、staging、长期 packed layout 和跨循环 state 如何组织；
- 哪个局部关系是 VLA map、dot、matmul、scan、lookup 或量化计算；
- 哪些值跨循环或多个局部命令存活并复用。

这些都不是“把一个 tile 分配给一组协作线程”的问题。若把它们隐藏在完整算子模板里，语言只
剩库调用；若让后端从普通 SSA 图猜回它们，Weft 又会退化成脆弱的图编译器。因此 Weft 要求
作者显式拥有算法结构，同时让编译器接管 core 内的机器组织。

## 不是 Triton-CPU

Triton 的根对象是一个 program instance 逻辑拥有的 tile；program-id 空间、GPU SIMT
执行和 tile 到 warp/thread 的布局共同构成模型。Weft 没有 grid、program id、隐式 hart id
或协作线程 tile：

- 一个 Weft worker 可以顺序处理许多 block；
- accumulator、workspace view 和 state 可以跨多个作者循环存活；
- 作者显式拥有外层 M/N/K 遍历和生命周期；
- `W.gemm` 只授权当前局部 block product，不创建完整 GEMM traversal。

把 Triton 的 tile 换成 CPU vector 并不能表达这些所有权，也会诱使编译器暗中创建作者没有
写出的 blocking 和 state 算法。

## 不是 TileLang-CPU

TileLang 将 shared memory、fragment、parallel copy、thread binding 和 pipeline stage 暴露给
开发者，是因为 GPU 的协作层级与软件管理存储本来就是算法实现的一部分。CPU 不应照搬
`alloc_l1`、vector register 或 IME fragment：cache 通常由硬件管理，物理寄存器由系统编译器
最终分配，LMUL/VLEN/fragment 属于 target realization。

Weft 暴露的是 **为什么一个值存在以及活多久**：workspace、persistent packed input、逻辑
block、state、reuse 和局部命令；编译器决定它最后保持在 scalar/vector register、局部临时
内存还是 extension fragment。TileLang 值得复用的是“显式语义授权与内部布局解耦”，不是
GPU 的存储和线程表面。

## 一门 DSL，两个可混用的抽象层次

Weft 不拆成“普通 kernel DSL + realization DSL”。两门语言会产生两份语义、两套组合规则和
难以界定的跨层 ABI。Weft 只有一个 Python DSL，内部提供两个可混用层次：

### 局部计算命令

作者直接写出不可从普通数据流无歧义恢复的局部关系：

```text
vdot / gemm
reduce / scan / argmax / online summary
sort / lookup / decode
typed quant compute
```

命令具有完整数值语义和普通 SSA 结果，但不拥有外围 traversal、storage 或 kernel ABI。

### 可组合构造

当标准命令不足以表达新局部算法时，作者在同一语言中使用：

```text
scalar for / while / if
W.vla
W.axis
load / store / transfer
pointwise / cast / select
ordinary SSA carry and state
```

这不是 escape 到 intrinsic。应用作者永远不写 RVV intrinsic、LMUL、寄存器编号或 IME asm。

## 三种概念值

### Scalar value

控制、地址、loop counter 和 scalar state。它可以参与有序控制，也可广播到局部数据域。

### Memory view

带 element type、access、alignment、alias 与 storage class 的 typed pointer，加上由地址表达式
形成的逻辑域。Memory view 不承诺物理 cache 层级。`W.load`、`W.store` 和 `W.transfer` 使数据
移动显式可见。

### Engine value

具有逻辑轴和 dtype 的普通 SSA 值。它可由 load、pointwise、reduce、`W.vdot`、`W.gemm` 或
量化命令产生，可多 use、跨控制流 carry，并进入另一个合法命令。名称表示它允许编译器选择
core-local 机器组织，并不意味着它当前位于物理向量寄存器或 fragment。

Canonical IR 内仍用 block/region type 表示 engine value；这是编译器类型，不是第二种用户
语言。

## 作者拥有的结构

作者必须显式写出：

- worker 的 `for / while / if` 与外层 traversal；
- algorithmic/cache blocking 和循环顺序；
- accumulator 与 state 的创建、更新和生命周期；
- staging、workspace、persistent packed layout 和算法 variant；
- pointer、index、predicate、effect 和 alias 事实；
- 一遍还是多遍、保存还是重算等可观察算法选择；
- 哪个局部关系被授权为 VLA、dot/matmul、reduce/scan、quant 或 extension command。

普通 scalar loop 不会被自动变成 VLA；普通 multiply/add/reduce 图不会被猜成 matmul 或
online softmax。

## 编译器拥有的结构

在显式授权边界内，编译器决定：

- 逻辑轴如何映射到 sequential、lane、register repetition、unroll 和 fragment；
- vector shape、SEW/LMUL、multiple accumulator 和 register microtile；
- unit/strided/indexed/segment memory form；
- 同一值在 load、cast、state、compute 和 store 间的 handoff；
- reload、rematerialize、短生命周期 local pack、prefetch 和 loop-local pipeline；
- RVV、IME 或其他已注册局部硬件实现；
- intrinsic C 与 typed local asm 的具体拼写。

系统 C 编译器继续负责最终寄存器分配、机器调度、peephole 和机器码生成。

## GEMM 判据

自然的 Weft GEMM 由作者写出 M/N/K block traversal、A/B 地址、accumulator 跨 K-loop 的
生命周期、可见 staging 和最终 epilogue：

```python
for m0 in W.blocks(m_begin, m_end, BM):
    for n0 in W.blocks(0, n, BN):
        mi = W.axis(BM)
        ni = W.axis(BN)
        acc = W.accumulator((mi, ni), W.f32, init=0.0)
        for k0 in W.pipeline(W.blocks(0, k, BK)):
            ki = W.axis(BK)
            a_block = W.load(...)
            b_block = W.load(...)
            acc = W.gemm(a_block, b_block, init=acc, acc_dtype=W.f32)
        W.store(..., acc, where=...)
```

`W.pipeline` 只授权当前作者循环做依赖合法的局部流水，不指定 stage 数，也不允许创建新的
算法阶段。`W.gemm` 只拥有 `[M,K] × [K,N] + init` 的局部语义。RVV outer-product、多个
vector accumulator 或 IME fragment 都是同一命令的 target realization。

## 量化与扩展

量化格式的 packed bits、scale、minimum、codebook 与 correction 是可观察数值关系，不能都
伪装成普通 GEMM，也不能由后端按格式名猜测。因此当前使用 `W.quant.<typed-command>` 表达
真实局部关系。它们共享编译器里的 packed-axis mapping、decode、widen、gather、reuse、资源
和流水能力，但不会被错误压成一个语义不完整的 `qgemm`。

新 RISC-V 扩展若只提供已有局部语义的新机器实现，只增加 target capability 和 leaf；只有当
硬件暴露了应用可观察、普通命令无法表达的新数值关系时，才增加一个局部 DSL command。

## 明确不进入语言的内容

当前不提供：

- `engines=[rvv, ime]` 一类 target 名称；
- LMUL、VLEN、register tuple 或 fragment 类型；
- `W.tune`、候选集合或 benchmark winner；
- 隐式 local allocation、cache level 或物理 register allocation；
- 任意轴 contraction、完整算子黑盒或按 kernel 名选择实现；
- 第二种 realization DSL。

`constexpr` 只表示作者允许构建期实例化的算法参数；候选空间和实测选择属于构建系统与
target lowering。

## 最终合同

```text
one worker-local DSL kernel
→ one canonical Kernel IR
→ op-specific semantic rules
→ shared axis/value/memory/lifetime reasoning
→ target-local physical decisions
→ intrinsic C / local asm
→ system compiler
```

这个合同同时保留作者对 CPU 算法组织的控制，也让同一个 `W.gemm`、`W.vdot`、state 或
quant command 在不同外围程序与不同 RISC-V target 上获得不同但语义等价的机器实现。
