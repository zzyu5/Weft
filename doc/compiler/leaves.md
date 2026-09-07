# RISC-V Local Leaf

## 1. 边界

local leaf 是 final RISC-V IR 中已经选定、合同闭合的 target-local operation。它可以是一条
RVV intrinsic、一个固定短指令序列、一个 IME fragment operation，或 ISA规定的固定 opaque
asm sequence。

leaf的完整合同不是一个 instruction字符串，而是：

```text
physical op kind
+ operand/result physical types
+ layout/access/conversion/window/fragment attributes
+ LeafAttr exact instruction、spelling、mask/tail与resources
```

translator可以根据这些 typed facts展开一个有限、唯一的指令序列。它不能在多个layout、
memory form、microtile、fragment、engine或pipeline之间再选一次。

leaf不能隐藏 canonical Level、outer traversal、blocking、state、source materialization、
caller workspace、persistent Encoding、dynamic pack loop、pipeline或完整operator ABI。固定序列的
slots、terms、trip count、effects和resources若不能由op合同静态闭合，就必须继续拆为真实IR。
primitive-private C array或fragment scratch也必须写入leaf的`local_bytes`，并由resource pass计入
kernel local-storage上界；不能只在terminal translator中出现。

## 2. RVV leaf

一对一RVV operation由leaf固定 intrinsic family，value type固定SEW/LMUL/`vl`/register parts，
access或conversion固定operand form。terminal lowering只把这些事实投影为Clang RVV spelling。

grouped MAC、encoded dot和contract step允许作为closed sequence：reduction loop已经在IR中，
window type固定slots/terms/result parts，access固定storage geometry，step leaf固定widen/MAC
sequence。改变group、unroll、lane operand或memory form会产生另一份physical op，而不是
emitter分支选择。

`convert_layout`同样是terminal physical op。source/result type和`ConversionAttr`唯一决定
split/merge/slide/splat/extract；translator实现该映射不等于重新推导layout。

spill/reload必须具有exact transfer leaf、typed local slot和完整value layout；translator不能仅凭
binding kind自行选择一条隐藏路径。

连续 lane slice 的载体容量按 `VLEN × LMUL / SEW` 计算，不按源 value 的 active lanes
比例推算 LMUL；未占满载体不是新的逻辑维度，也不能使合法窗口被误判成非整数 LMUL。
缩小载体还必须保证每个完整窗口的起点与终点位于同一寄存器组内；长度放得下不能替代
offset 合法性，跨组窗口保留能够完整容纳它的载体。

`rvv.partial-repack.split` 仅表示可直接寻址的完整寄存器组切片；fractional 结果或未占满
寄存器的窗口使用显式选定的 `rvv.partial-repack.slice`。后者的 leaf 固定 addressable carrier，
terminal translation 只据此展开 subgroup extract、intra-group slide 与 LMUL truncation。
不能输出不存在的 fractional `vget`，也不能在 emitter 临时换载体。
每个独立 partial slot 按至少一个寄存器组计数；窗口 materialization 的结果与临时载体
必须计入 leaf 资源，不能将 fractional slots 假定为共享同一寄存器的免费视图。
逐槽 reduction 的资源计数遵守消费时序，允许已消费输入与对应结果复用寄存器，不将
所有旧输入和所有新输出重复计为同时存活。

一个已经选定的grouped/layered byte-window load可以使用固定的
`scalar-prime + vector-load` leaf。其op必须只含一个raw window，logical base、byte offset、active
lane数和vector load form均已闭合；translator只把“读取最后一个active byte，再发原vector load”
拼成两条有序指令。该leaf没有跨iteration对象、distance、buffer或额外result，不能冒充通用
prefetch，也不能由translator临时加到普通load上。

## 3. IME / opaque asm leaf

IME capability和operations共同固定：

- fragment family、logical shape、dtype与axis relation；
- lhs/rhs/accumulator fragment role及其tiled packing schema、tile rows/columns、tile与
  element order、storage width与alignment；
- MMA group/chunk数和fragment resources；
- fragment→RVV handoff；
- exact asm spelling key、constraints、clobbers与memory effect。

当前SpaceMIT IME1 leaf是一个固定的 signed-i8 `M4×K8` 与 signed-i8 `K8×N4`
fragment product，产生 i32 `M4×N4` accumulator。lhs按`[M,K]`排列，rhs按
`[N,K]`的物理次序装入同一个`[K,N]`逻辑值，result按`[M,N]`排列。packing tile geometry、
axis order和各role的resource groups都是typed attribute，不能只靠`role=lhs`在emitter中复原；
固定的pack、`vmadot`和unpack序列可以opaque，因为shape、trip count、signedness、storage width
和clobber集合已由唯一capability/spelling合同封闭。i4 operand、其它fragment shape以及非
VLEN256的IME1 target当前均是明确unsupported，不会由terminal translator改走另一条实现。

完整GEMM/GEMV、persistent repack、M/N/K dispatch和outer loop不属于IME leaf。若extension需要
不同Level/materialization或persistent Encoding，作者在lowering前选择另一份std tree。

## 4. 选择与失败

1. `SelectRISCVOperations`只选择局部structural family；
2. layout、memory与composite passes把representation和真实control闭合；
3. `FinalizeRISCVLeaves`为尚未终结的普通op写唯一exact instruction；
4. 已实例化的local memory-leaf参数可以在完整storage/use relation上改写exact leaf；
5. resource pass补全leaf resource counts，final verifier核对整个合同。

leaf/capability集合不按kernel、operator或量化格式注册。无合法leaf、packing/access不完整、
toolchain不支持spelling、resource/effect不闭合时当前module失败；不能回到emitter重选、静默
scalar化或调用仓库外reference/GGML helper。
