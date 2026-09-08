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

encoded projection 以新表示供应 numeric value 时，`field_read` 显式给出读取点、
source projection、access 和 result layout；terminal 不在 pure `time_to_lane` 中重新读取。
每个 part 的 active length 来自该 value 的 layout 与对应 `physical_point`，不扫描同 block
其它 result 推导 scope VL。跨 operation 的提取共享由 SSA/CSE 决定；单个 conversion 内
允许复用固定展开中相同的 source part/lane，不把这种临时量缓存扩到相邻 operation。
field/extract 的存储投影绑定独立保留；数值物化不能覆盖它，再让另一个显式 Read op
根据偶然的 binding kind 决定是否读取。projection map 只保存对应 IR 的地址/坐标描述，
不共享已发射的 load/decode，也不决定 layout 或资源。

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

## 5. 可调用 artifact

同一次 terminal translation 同时生成 C prototype 与 `RISCVKernelABI`。ABI 的参数顺序、
Encoding identity、record span、logical extents、alignment、access 和 alias groups 均来自
已验证的 kernel/MemDesc，而不由 runtime 解析 C 文本或按格式名补齐。
`RISCVCompilationResult` 保存代码和这些 ABI；`weft-compile --emit=artifact` 机械序列化它们。
动态 shape 参数顺序与实际 prototype 一致，source binding 与发现的 target 随产物保持明确。
加载器负责系统编译、句柄与调用生命周期，不参与 leaf 或资源选择。

非交错自然布局的 scalar field read 将已验证的访问对齐在读取点传给 system compiler；
注解只保留访问对齐、record byte stride 与读取宽度共同保证的二次幂因子。
它不增强 kernel 参数的 ABI 对齐、不改变空 View 的调用条件，不把 byte storage
转成更强别名类型，也不根据字段或 kernel 名选择读取算法。

## 6. 实现模块边界

`lib/Target/RISCVIntrinsicC.cpp` 保留唯一公开 translation 入口；`Emission/Control.cpp`
与 `Emitter.h` 管理共同的 ABI、C control 和 SSA binding。`Memory.cpp`、`Encoding.cpp`、
`Layout.cpp` 和各 `RVV*.cpp` 分别实现存储、编码、表示与已选 RVV 局部操作，均为独立编译单元。

`Emission/IME.cpp` 处理 typed fragment 的 pack/unpack 与普通 binding 交接；
`IME/FragmentEmission.cpp` 仅拼写已选 ISA atom，经有限的 fresh-name/line 接口输出局部 C/asm，
不接管公共遍历或 ABI。packing/MMA 私有存储与 clobber 的合同来自同一目标描述，不能在
发射时增加。固定数组往返是已实现 atom 的局部策略，不是所有未来 fragment 的公共 ABI。
