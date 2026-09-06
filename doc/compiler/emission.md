# Intrinsic C 与 Local ASM Emission

## 1. 输入边界

terminal translator只接收通过`VerifyFinalRISCV`的[RISC-V IR](riscv-ir.md)。此时control、
address relation、layout conversion、window、fragment packing、pipeline、spill与ABI都已经是
typed program entities；每个target operation也已有闭合的[local leaf](leaves.md)。

translator内部可以用临时binding map把SSA value对应到C expression、RVV variable、local array
或fragment pointer。该表只保存已发射值的文本，不是跨pass decision authority。

纯表达式可以内联，但 memory-read SSA result 必须在对应的读取点求值并形成稳定绑定。
同一结果的重复 use 不能变成重复解引用，也不能将读取延迟到可能别名的 store 之后。
这项要求同时覆盖 scalar、scalar tuple、encoded field 与 local load；source record 到
field 的投影也必须保留原有读取/effect 边界，不能把描述符字符串当成已经加载的值。
带 `snapshot_storage` 的 encoded load 只按已选 byte-copy leaf 写入对应 local allocation，
随后绑定该私有存储；translator 不自行决定是否复制、复制范围或生命周期。

## 2. 允许的确定 lowering

translator可以：

- 把`func/scf/arith`写成普通C control和scalar expression；
- 根据`ValueType/LayoutAttr`枚举静态time/register parts并选择已确定RVV C type；
- 根据`MemDescType/AccessAttr`打印已确定的pointer arithmetic、stride、gather和bit extraction；
- 根据typed `convert_layout` source/result映射打印固定slide/splat/split/merge序列；
- 根据WindowType与exact leaf展开有限group/term指令序列；
- 根据FragmentPackingAttr和IME leaf打印固定local packing与typed asm；
- 打印explicit local alloc/guard/spill/reload、ABI、headers和declarations；
- 在机器合同不变时，由toolchain adapter选择等价API spelling。

这些工作是TTGIR→LLVM conversion pattern意义上的terminal lowering：输入type/op已经唯一决定
结果。生成多条C语句不代表translator拥有结构选择权。

## 3. 禁止的决定

translator不能：

- 补layout、lane、part、stride、access、resource或capacity默认值；
- 从canonical op、kernel/格式名、VLEN、target型号或source closure选择RVV/IME；
- 改LMUL、lane/register mapping、memory form、operand swap、fragment或packing schema；
- 生成source Level、contract reduction loop、动态local-pack loop或pipeline结构；
- 决定spill/reload/rematerialize、workspace或persistent ABI；
- 同时读取Canonical Kernel IR与side assignment合成physical program；
- 调用GGML、仓库外reference code、legacy helper或fallback。

translator若发现binding/type/leaf之间不一致必须失败。这表示前序pass或verifier合同遗漏，不能在
这里加一个default或第二selector。

## 4. System compiler边界

Weft已经决定logical-to-physical mapping、target-local instruction sequence和loop-local schedule。
system C compiler继续负责最终register allocation、machine scheduling、peephole、constant
folding与machine code。生成C中的局部pragma可以阻止system compiler破坏已选physical形态，
但不能要求它重新发现SIMD、microkernel或pipeline。
