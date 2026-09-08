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

`rvv_multiply_high_scalar` 表示无符号 u8/u16 vector 与同宽 scalar 乘积的高半部分，
结果完整 type 必须等于 vector 输入，leaf 固定为 `rvv.vmulhu.vx`。它没有 rounding、
saturation、outer loop 或私有 scratch；active lanes 与全部 time/replica parts 仍由输入
layout 确定。数值语义为 `floor(unsigned_product / 2^SEW)`，遵守
[RVV 单宽整数乘法合同](https://docs.riscv.org/reference/isa/unpriv/v-st-ext)。
它是独立的 selected local op，不扩大普通 BinaryOp，也不授权 scalar fallback。

`rvv_byte_gather` 从连续 byte/word descriptor 或一个完整、byte-aligned natural word field
读取指定的原始字节，返回 unsigned u8 RVV value。byte offsets 是显式的 u16/u32 value，
所有 index/result axes 和 active lanes 必须一致，EEW/EMUL 与 indexed-memory 能力由
verifier 核验；leaf 固定为 `rvv.byte-gather`。它只读取所选字节，不在 emitter 中恢复
word、shift 或 mask。encoded value 的原 load/快照身份仍经 field operand 保留。

`rvv_byte_windows_load` 读取同一 storage 的一至四个连续字节窗口，按显式 `byte_bases`
operand 顺序拼成一个 full-valid signed/unsigned RVV byte value。每个 base 是相对 storage
的同宽 unsigned u16/u32 字节偏移；base 的计算与 wrap 已在 scalar SSA 中完成，窗口内读取
连续 `window_bytes` 字节，不再隐式做窄整数 wrap。窗口宽度与 result 的完整 lane 数必须闭合。
由 indexed read 选择此 leaf 时，pass 必须证明每个窗口内部不跨原 index 的 wrap 边界；不同窗口
可以来自不同动态 base，不要求 storage 地址有序或互不相交。leaf 固定为 `rvv.byte-windows-pack`：逐窗口 `vle8`，后续窗口以 tail-undisturbed
`vslideup` 拼接，不能增加读取字节、对齐要求或外围遍历。多窗口只保留一个额外载体，
其完整寄存器组数必须计入 leaf temporary；scalar bases 与原 storage/快照身份仍显式存在。

grouped MAC、encoded dot和contract step允许作为closed sequence：reduction loop已经在IR中，
window type固定slots/terms/result parts，access固定storage geometry，step leaf固定widen/MAC
sequence。改变group、unroll、lane operand或memory form会产生另一份physical op，而不是
emitter分支选择。

`convert_layout`同样是terminal physical op。source/result type和`ConversionAttr`唯一决定
split/merge/slide/splat/extract；translator实现该映射不等于重新推导layout。

`rvv_masked_negate` 在 mask bit 为 1 时对 signed integer data 取负，否则保留 data；
result 的完整 type 不变，逐 lane 保留该整数宽度的运算边界。mask 来自同一读取点的
`rvv.bitmask-window-mask`，shape、axis、time/lane/replica mapping、active lanes 和
SEW/LMUL mask ratio 必须与 data 一致。leaf 固定为 `rvv.masked-negate`，每个 physical
part 只拼写同宽 `vneg` 与 `vmerge`；一个额外 data carrier 计入 temporary groups。
它不包含数据 widening、浮点 scale、reduction 或额外读取，也不在 translator 匹配表达式。

`field_read` 是 Read-effect 的 terminal transfer：input 保留 encoded field/projection 的
原 load/快照身份，access 固定读取与 bit mapping，result 是具有完整布局的 RVV/scalar numeric SSA。
它可以读取同一表示，也可以直接供应已选的另一表示；不能把一个已经计算的 numeric value
追溯回 storage 后重读。pure conversion 不承担这项存储供应职责。

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

ISA capability、packing、fragment groups、私有 scratch 与 clobbers 的唯一描述在
`lib/Target/IME/FragmentContracts.cpp`。不依赖 MLIR 的 `WeftRISCVContracts` 同时由 target
profile 与 IR dialect 消费；`FragmentOps.cpp` 负责属性序列化及针对该合同的 verifier，
`IME/FragmentMaterialization.cpp` 物化 selected product，`IME/FragmentEmission.cpp` 实现拼写。
属性不能用一个已知 instruction key 冒充不同的 dtype、shape、VLEN 或寄存器约束。
MMA 的固定 vector clobbers 必须在 issue 点全部预留；已经 live 的 lhs/rhs fragment groups
只抵扣一次，其余寄存器写入 leaf 的 additional fragment groups。不能假定固定 asm 的
accumulator 寄存器与输入重叠，或仅用指令前后的最大 live set 代替执行期间的需求。

使用已有 atom 的新算子在 primitive/数值/存储合同已覆盖时，只增加 source/std 表达与显式
绑定。同类 fragment 合同的新 atom 修改上述目标局部描述、合法性、packing/spelling 与静态
登记，不修改公共 pipeline、ABI 或 live-set 算法。新的 effect、residency 或数值机制仍需
显式扩充对应通用合同；这不承诺任意新硬件无需公共改动。

RVV 与 IME 共享 frontend/Canonical IR、外围 control、memory/effect、compiler API、ABI 和
region-aware live-set 分析；两者的寄存器需求进入同一预算。RVV 的 LMUL/partial/spill 策略
不因此自动适用于 fragment。标准 ISA 的 native JIT 发现不等于厂商 matrix 扩展发现，后者
没有实现时仍明确 unsupported。

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
