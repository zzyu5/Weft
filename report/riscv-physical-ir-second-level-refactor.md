# Weft 第二层 RISC-V Physical IR 完整重构报告

## 1. 本轮结论

本轮把原先围绕 `problem / assignment / planning side record` 组织的 RISC-V lowering 全部删除，重建为唯一的第二层程序 IR：

```text
Canonical Weft Kernel IR
→ typed, target-aware Weft RISC-V IR
→ intrinsic C / local asm
→ system C compiler
```

第二层不再是一张附着在 canonical program 旁边、供 emitter 查询的决定表。layout、memory descriptor、conversion、local storage、window、fragment、schedule、spill/reload 和 exact leaf 都是可验证、可改写、可 dump 的 MLIR type、attribute 或 operation。每个 pass 的产物通过 IR 本身交给下一 pass；terminal emitter 不再读取另一套 assignment，也不再从 canonical closure 重建物理程序。

本轮同时纠正了原 IME probe 对硬件能力的错误描述。当前正式接入的 SpaceMIT IME1 局部能力是 signed-i8 `M4×K8 × K8×N4 → i32 M4×N4`，而不是未经硬件合同证明的 q4/i4 路径。新的 typed fragment 链已经在 K1/X60 上完成真实数值执行。

## 2. 参考机制与 Weft 取舍

重构前先核对了本地 Triton 与 TileLang 实现，而不是从旧 Weft selector 继续抽象：

- Triton 在 TTIR→TTGIR conversion 时把 target layout 写进 tensor type encoding；`ttg.convert_layout` 是真实 operation，matmul acceleration、coalescing、layout propagation、conversion elimination 和 pipeline 都改写同一份 TTGIR。
- TileLang 在同一 TIR module 上逐步物化 target-aware loop、buffer、copy、pipeline 与 target call。target namespace 不等于新增一层持久 IR，terminal codegen消费的是已经物化的 target program。

Weft采用相同的结构原则，但不采用SIMT thread/warp/CTA ownership作为物理根坐标。第二层 value 保留 canonical logical axes，再将它们分解到 time、RVV lane、register replica、IME fragment 与 local storage。由此只有两层MLIR：canonical Kernel IR和target-aware RISC-V IR；target profile、tuner和pass不是第三层IR。

## 3. 新的 typed RISC-V IR

### 3.1 物理类型和属性

新 dialect 明确承载：

- `ValueType`：logical element、shape、axis identity和完整`LayoutAttr`；
- `LayoutAttr`：每条axis的time/lane/replica/fragment/local factor，以及SEW、LMUL、VL、register groups和validity；
- `MemDescType`：canonical Encoding identity、shape/axis/stride/origin、alignment、address class、alias、storage bits和interleave facts；
- `LocalType`：worker-local object的element、shape、axis、capacity、alignment、purpose和birth identity；
- `FragmentType` / `FragmentPackingAttr`：extension family、role、logical axes、tile geometry、axis order、storage width和resource groups；
- `WindowType`：grouped MAC、encoded dot或contract的operand/result domain、reduction axis、slot/term/result-part几何；
- `TargetAttr`：ISA/ABI/VLEN、SEW/LMUL、vector resources、memory capability、local-storage limit和typed fragment capabilities；
- `AccessAttr`、`ConversionAttr`、`ScheduleAttr`、`ImplementationAttr`、`LeafAttr`：分别保存memory edge、真实representation conversion、Level-local schedule、结构实现锚点和最终硬件leaf合同。

所有已赋值representation都检查factor正值、rank/axis一致、静态extent覆盖和乘法溢出。fragment、window、local object与leaf各自验证其typed resource和几何关系，不能依靠emitter临时补默认值。

### 3.2 真实物理 operations

第二层包含普通physical value operations、typed memory load/store/field、`convert_layout`、local alloc/bind/load/store、spill/reload、physical point与Level loop、operand window load/step、IME pack/MMA/unpack等真实operations。

尤其是representation冲突不再写成`assignment[edge] = conversion`：`PropagateRISCVLayouts`插入真实`weft_riscv.convert_layout`，`CanonicalizeRISCVLayouts`对这些SSA operations做identity删除、链折叠、CSE和合法rematerialization。若conversion仍存在，emitter必须按其typed source/result layout机械发射；若缺失，final verifier直接失败。

## 4. 唯一 pass 主链

当前编译器固定运行以下十个pass：

1. `ConvertWeftToRISCV`：把canonical kernel、Encoding declaration、artifact pack、Level/control和值/内存operation转换为未决但typed的RISC-V program；转换不了的type立即报错，不构造null type。
2. `SelectRISCVOperations`：仅从typed primitive、operand axes、target capability和显式config选择结构实现family；不按kernel名、格式名或VLEN名字选择whole-kernel route。
3. `PropagateRISCVLayouts`：沿use-def传播axis representation、SEW/LMUL/VL和carrier，修改SSA result/block argument/loop result types，并为冲突插入真实conversion op。
4. `PlanRISCVMemory`：根据descriptor、Encoding mapping、pointer/index relation与target capability写入unit/strided/indexed/segment等`AccessAttr`和exact transfer/lookup leaf。
5. `CanonicalizeRISCVLayouts`：删除identity conversion、折叠可组合conversion、CSE共享conversion，并只在有typed producer可重建时rematerialize。
6. `LowerRISCVComposites`：把Level、local materialization、grouped/encoded reduction、regular contract和IME fragment结构投影成真实loop/local/window/fragment operations。
7. `PipelineRISCVLevels`：depth 1保持顺序；当前depth 2只对已闭合的two-op window cluster生成prologue/steady/epilogue和两份buffer version。
8. `FinalizeRISCVLeaves`：把尚存的typed implementation锚点终结成唯一exact leaf，然后删除`implementation`；不保留备用leaf。
9. `MaterializeRISCVResources`：按真实SSA live interval、layout、temporary、fragment和local object计算资源；只对合法的同block、非fragment、无memory effect value插入typed spill/reload与capacity guard。
10. `VerifyFinalRISCV`：只读验证最终program。任何canonical op、unassigned layout/access、残留implementation/schedule、未闭合leaf、错误engine/carrier、错误Level identity或资源低报都会使当前module失败。

第十个pass是final verifier，不声称改写IR。其他pass在当前输入无需变换时也可能合法no-op；“pass存在”本身不是能力证据，真实证据是下文的typed IR链和目标执行。

## 5. Decision authority 与 emitter 边界

本轮将物理决定分配到唯一producer：

- target profile产生ISA、VLEN、resource和fragment capability事实；
- Select产生局部结构family；
- Propagate产生value representation与conversion；
- PlanMemory产生memory edge和lookup/load/store leaf；
- Lower/Pipeline产生局部loop、window、fragment与buffer版本；
- MaterializeResources产生resource peak与spill/reload；
- Finalize产生其余普通operation的exact instruction。

terminal emitter只接受最终physical operations及其types/attributes。它仍然负责复杂但确定的工作，例如根据`FragmentPackingAttr`展开固定次数的pack C语句、根据exact leaf拼RVV intrinsic API或typed inline asm constraints；这不等于重新选择fragment shape、layout、memory form或pipeline。

最终审计专门构造并关闭了几类“verifier能过、emitter才失败”的反例：

- lookup现在要求index carrier、`AccessAttr`和exact `scalar.lookup`/`rvv.vluxei`三者一致，删除了没有producer的`rvv.unit-lookup-window`死分支；
- RVV splat统一为exact `rvv.splat`，不能用任意`rvv.splat.*`字符串绕过；
- iota只接受unsigned integer或index，index在RVV中使用unsigned vector spelling；
- unary/binary/compare/cast/narrow/widen/reduce/fold2的Finalize和final verifier共享同一typed instruction mapping；
- local/storage/spill/reload/update、IME pack/MMA/unpack、window load/step均检查exact leaf合同；
- emitter所有physical-factor乘积改用checked product，dialect在type创建时拒绝跨axis乘积溢出。

## 6. IME局部能力的纠正与闭合

旧probe把一个未经target事实证明的q4/i4结构当成IME能力，并用M1×N16×K32的高层shape暗示完整quant path。该入口被删除，没有兼容route。

当前K1 profile只在RV64、完整RVV、显式VLEN256和`spacemit-ime1`同时成立时注册一项typed capability：

```text
lhs: signed i8 [M=4,K=8], packing [M,K], 1 group
rhs: signed i8 [K=8,N=4], physical order [N,K], 1 group
acc/result: i32 [M=4,N=4], 2 groups
instruction: spacemit-ime1-i8-mma / vmadot
```

`SelectRISCVOperations`根据普通contract的dtype、free/reduction axes和shape匹配该capability；`LowerRISCVComposites`生成typed pack→fragment MMA→unpack；emitter只消费已经闭合的packing、constraints、clobbers和effect。i4 operand、其它shape及非VLEN256 IME1均明确unsupported，不会退回隐藏RVV路径。

本轮还修复了一个真实的Level lowering错误：不同logical axis的嵌套Level曾错误继承父Level的base/active，导致M→N→K独立循环使用错误边界。现在只有父子Level属于同一axis时才继承strip point；不同axis从自身origin和extent建立physical point。IME repro的数值错误由此消失。

## 7. 删除的旧状态

仓库已删除旧planning dialect以及`ConstructProblems / ConstrainRepresentations / ConstrainInstructions / ConstrainResources / Solve / PropagateStorageMappings / ResolveLayoutConversions / ScheduleLevels`整条side-record主干，也删除对应support、旧API和CMake入口。

静态审计在`include/`、`lib/`、`tools/`、`python/`和`examples/`中未找到旧Problem/Assignment API、旧pass工厂、compatibility layer、legacy emitter、silent fallback或按kernel/量化格式接管whole-kernel的route。`materials/`仍只是历史知识供体，不进入当前build或runtime。

## 8. 验证证据

编译器使用本轮全部改动重新构建：

```bash
cmake --build build --target weft-compile -j2
```

构建成功，无旧pass或新dialect的链接缺失。

唯一手动repro命令：

```bash
./examples/run/weft-kernel.sh k1 ime_i8_contract 3
```

本轮最终输出：

```text
kernel=ime_i8_contract
target=K1/X60
M=4
N=4
K=8
numeric=exact
repetitions=3
median_us=0.542
gop_s=0.472325
```

这条命令真实完成Python DSL emit、canonical IR、十个physical passes、intrinsic C/local asm、K1交叉编译和目标执行。`numeric=exact`对本例整数leaf成立。三次重复的时延只作为端到端可执行证据，不是production benchmark，也没有对应同口径source baseline，因此本轮没有更新`weft-kernel-performance.csv`。

RVV vector compare、lookup exact leaf和index iota的闭合修复已经通过完整C++构建与静态contract审计，但本轮没有为它们新增独立测试脚手架或伪造runtime结论。

## 9. 当前明确边界

本轮成熟的是第二层IR骨架、authority和一条真实IME leaf链，不等于所有physical optimization已经完整：

- encoded-dot当前要求静态reduction extent；dynamic encoded-dot明确失败；
- pipeline当前只有顺序和depth-2、two-buffer、single-carry window cluster；普通register contract、multi-carry、其它cluster和depth>2不冒充已支持；
- spill只支持同block、非fragment、无memory effect的普通physical value；fragment/cross-block spill不支持；
- regular register contract的pipeline depth必须为1；
- 当前target compiler要求完整`V`扩展和显式正VLEN；`Zve`、纯标量target、未知VLEN、async/wait/barrier、独立prefetch域和通用latency model不在实现范围；
- SpaceMIT IME1当前只支持上述signed-i8 M4×N4×K8局部primitive；没有i4 IME、其它fragment family或whole-GEMM microkernel；
- 本轮没有做production全量性能重测，也没有以这个局部repro替代baseline实验。

这些边界都在physical pass或typed verifier处明确拒绝，不由emitter静默缩小LMUL、改engine、标量化或调用旧实现。
