# Intrinsic C 与 Local ASM Emission

## 1. 输入

Terminal translator 只接收通过 `VerifyFinalRISCV` 的 [RISC-V IR](riscv-ir.md)。此时：

- control、address、pack、pipeline与spill结构已经物化；
- every physical value已有layout；
- every target operation已成为合同闭合的[local leaf](leaves.md)；
- ABI、headers、toolchain requirement和spelling key已经确定。

## 2. 允许的工作

translator 可以：

- 把 `func/scf/cf/arith` 写成普通 C control与scalar expression；
- 把RVV leaf写成确定的Clang intrinsic与C vector type；
- 把IME/opaque leaf写成确定的typed asm或intrinsic；
- 打印已物化的pointer arithmetic、ABI、headers和declarations；
- 根据explicit toolchain profile选择等价的API spelling。

toolchain adapter只处理相同机器合同的API/语法差异。若两种写法改变operand representation、instruction sequence、resources、clobbers或effects，它们必须是上游不同physical leaves。

## 3. 禁止的工作

translator不能：

- 补layout、part、lane、stride或resource默认值；
- 从canonical op、shape、VLEN、kernel或格式名选择RVV/IME leaf；
- 生成Level loop、contract K loop、pack loop或pipeline prologue/steady/epilogue；
- 推断memory form、broadcast、operand swap、fragment、spill或local ABI；
- 同时读取canonical module与assignment表合成physical program；
- 调用GGML、materials、legacy helper或fallback。

信息缺失必须返回final-RISC-V verifier错误。

## 4. System Compiler 边界

Weft已经决定logical-to-physical mapping、intrinsic/asm operation和局部schedule。系统C compiler继续负责最终register allocation、machine scheduling、peephole、constant folding与machine code。

生成C中的局部pragma可以阻止system compiler破坏已经选定的physical形态，但不能要求它替Weft重新发现SIMD、microkernel或pipeline。
