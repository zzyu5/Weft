# RISC-V Local Leaf

## 1. Leaf 的边界

local leaf 是 final RISC-V IR 中已经选定、合同闭合的 target operation。它可以是：

- 一条 RVV intrinsic；
- 少量固定 RVV instructions 实现的 closed primitive；
- 一个 IME fragment operation；
- 一个 ISA 规定的固定 opaque instruction sequence。

leaf 可以隐藏 ISA spelling 与 primitive-private temporaries，不能隐藏 canonical Level、outer loop、blocking、staging、state、local-pack loop、pipeline、spill、workspace、persistent Encoding、kernel ABI 或完整 operator traversal。

固定 fragment 内的有限指令序列可以 opaque；其 trip count、operand window、effects、resources 和 result 必须由合同完全封闭。trip count 来自 source Level、dynamic shape 或 physical pipeline 时，循环必须在 [RISC-V IR](riscv-ir.md) 中显式存在。

## 2. Typed Leaf Contract

每个 leaf op 必须由类型和 verifier 声明：

```text
local numerical/transfer semantics
typed operands/results、logical axes与physical layouts
dtype、shape、mask/tail、rounding与overflow
memory descriptor、alias、effect与order
fragment、register/local temporaries与resources
allowed conversions与result handoff
intrinsic/asm spelling key、headers与toolchain requirement
asm operands、constraints、clobbers与memory semantics
```

opaque IME leaf 还必须声明完整 local ABI、fragment handoff、volatile 与 synchronization contract。缺少这些字段的 helper，即使能生成 asm，也不是合法 leaf。

## 3. Leaf Selection

RVV、IME或其它extension之间的选择会改变layout、fragment、conversion与resources，必须在physical passes中完成：

1. `SelectRISCVOperations`按canonical op、typed operands、axes、Encoding、profile、requirement和固定优先级选择target-op family；
2. layout、memory、schedule、resource passes将它具体化并验证；
3. `LowerRISCVComposites`产生最终primitive leaf ops；
4. `VerifyFinalRISCV`确认每个leaf只有一个完整合同和可用spelling。

leaf definitions和target profile capability构成可用集合；不按kernel、operator family或量化格式注册。没有合法leaf时当前module明确unsupported，不能回到emitter重选或调用旧helper。

若两个实现的operand representation、fragment、resources、effects或instruction sequence不同，它们是不同physical leaves。若只有Clang intrinsic API或asm语法不同，而机器语义和合同相同，则属于toolchain spelling adapter，不是physical选择。

## 4. RVV Leaf

RVV leaf必须把SEW、LMUL、`vl`、mask/tail、operand/result layout和intrinsic semantics写入op contract。动态strip、source Level loop、pack loop与pipeline必须已经在IR中展开；leaf不能通过C helper重新生成这些结构。

一对一intrinsic可直接保留到terminal。固定closed sequence只有在所有temporary、资源、effects和局部顺序都可验证时才可作为opaque leaf；否则必须继续拆成primitive RVV ops。

## 5. IME / Opaque ASM Leaf

IME leaf必须描述fragment family/shape、operand packing requirement、accumulator/result representation、RVV↔fragment handoff、register/fragment resources、asm constraints与clobbers。

完整GEMM/GEMV函数、按M/N/K dispatch的helper、persistent repack和带outer traversal的长asm都不是leaf。IME需要另一棵Level/materialization/persistent Encoding时，作者在lowering前选择另一份std tree；不能由leaf改树。

## 6. Materials 与失败边界

`materials/`中的intrinsic/asm name、constraint、clobber和局部ABI可用于spelling；其中的microtile、operand reuse、fragment handoff和pipeline必须先进入target op、layout或schedule。完整旧kernel、runtime和format route不能包装成leaf。

无合法leaf、合同字段不全、toolchain不支持spelling、resource/ABI/effect不闭合，都必须在final verifier前失败。不得fallback、静默scalar化或让terminal emitter补决定。
