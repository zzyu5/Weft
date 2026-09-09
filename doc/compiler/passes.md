# RISC-V 编译 Pass

所有 pass 都改写同一份 [RISC-V IR](riscv-ir.md)。`implementation`、`schedule`、
`lane_operand` 等中间属性附着在具体 operation 上，并在其消费者 pass 中删除；它们不是
module 外的 assignment 表，也不构成新的 IR 层。

## 1. 固定顺序

```text
ConvertWeftToRISCV
→ SelectRISCVOperations
→ PropagateRISCVLayouts
→ SelectRISCVOperations
→ PlanRISCVMemory
→ CanonicalizeRISCVLayouts
→ PlanRISCVMemory
→ SelectRISCVOperations
→ LowerRISCVComposites
→ MaterializeRISCVPrograms
→ VectorizeRISCVRecordLoops
→ CanonicalizeRISCVLayouts
→ PlanRISCVNestedMemory
→ FuseRISCVBitplanes
→ HoistRISCVLoopInvariants
→ ScheduleRISCVLevels
→ PipelineRISCVLevels
→ SCCP
→ ShareRISCVLayeredWindows
→ PlanRISCVPartialTopologies
→ MaterializeRISCVPartialAccumulators
→ UnrollRISCVLevels
→ ShareRISCVLayeredWindows
→ CoalesceRISCVPartialExtractions
→ EliminateDeadRISCVLayouts
→ CanonicalizeRISCVLayouts
→ PlanRISCVMemory
→ SelectRISCVMemoryByteProjections
→ MaterializeRISCVReplicaStorageLoads
→ HoistRISCVLoopInvariants
→ SelectRISCVOperations
→ FinalizeRISCVLeaves
→ SelectRISCVScalarLoadPrimes
→ MaterializeRISCVReadSnapshots
→ CloseRISCVLeafResources
→ CSE
→ FuseRISCVPhysicalIssueLoops
→ EliminateDeadRISCVLayouts
→ MaterializeRISCVResources
→ EliminateDeadRISCVLayouts
→ VerifyFinalRISCV
```

该顺序是依赖关系，不是阶段标签。`LowerRISCVComposites` 必须先产生带 typed
operation/effect 的显式 physical loop，`schedule` 才能从真实 use-def 分配 stage/order，
`pipeline` 再只负责版本化 SSA 值并生成 prologue/steady-state/epilogue。partial planner 必须在
generic unroll 之前读取未复制的 contraction/reduction use-def，冻结 carrier、combine topology
与 issue unroll；materializer 生成 issue loop后，`UnrollRISCVLevels`才机械复制迭代。展开后才完整
出现的 grouped/layered issue relation 再交给同一个幂等 sharing owner，物化为共享
raw-window load 与显式 layer decode。exact leaf
完成后，resource pass 才能把 leaf temporary 与 SSA live interval 一起计入峰值。

最终 layout 改写前先清理失去消费者的纯 index/conversion 链，避免死路径阻止基于真实
use 数量的 rematerialization；资源闭合后的清理不能替代这个位置的清理。
派生节点继承已有 `source_origin`；只有没有来源标签时才从 location 建立初始来源。
解析 Physical IR 的文件位置不能覆盖作者来源，否则同一程序重放会改变 CSE 的匹配集合。

`weft-compile --emit=riscv-layout-input` 在这次清理与 `CanonicalizeRISCVLayouts`
之前输出同一层 Physical IR。module 的 `weft.riscv.layout_input` 保存尚未消费的
`scalar_load_prime` 单值绑定；其余 target/binding 已体现在 typed program 中。
`--resume-layout-input --emit=intrinsic-c` 从解析后的该边界运行与普通编译完全共用的
layout、memory、leaf、read-snapshot、resource 和 final verification 后缀，再机械生成 C。
缺少标记、已经资源闭合或试图用 CLI 更换 target/binding 均拒绝；不清除 final resource
marker 来强行重放。该边界不增加 IR 层，也不是备用编译主链。

## 2. 各 pass 合同

### `ConvertWeftToRISCV`

输入是一个已经实例化的 Canonical Kernel IR candidate、一份 target profile 和一组单值
physical parameter binding。它建立 target-aware `ValueType`、`MemDescType`、Level/control、
完整 runtime `memory_view`、artifact builder 与 numerical operations，然后删除 canonical
operations。初始 layout、access 和 leaf 可以是 `unassigned`，但 source axis、Value、Level、
birth、handoff、effect、Encoding 与 ABI identity 必须一一保留。

一个 physical module 只含一组 meta/unroll/pipeline binding。候选枚举和实测发生在 driver
之外；pass pipeline 不接收列表，也不在 module 内保存备用值。

### `SelectRISCVOperations`

读取 operation semantics、typed operands、axes 和 target capability，按 target 固定优先级
写入一个 `ImplementationAttr`：scalar、RVV 或 IME 的局部结构 family。它不选择 memory
form。没有合法局部结构时当前 module 失败。

对已经物化、尚未资源冻结的局部 target op，也可选择合同等价的基本操作：full-valid、
逐轴同形的 byte 乘积、多项 i16 累加链，其编译期整数零初值可由一个 widening multiply
起链，后续 accumulate 和最终 reduction 不变。signedness 来自 typed operands，继承完整
carrier/坐标，不重选 partial plan；新 leaf 及资源由原有 verifier/materializer 重新检查。
该规则不处理独立乘积、tail carrier 或更宽的 partial，不将代数上少一项运算当作其成本结论。

单轴 full-valid i16 loop carry 若仅沿同 block 的 byte widening product + add 链流向其
对应 yield，可将整条链选择为已有 widening accumulate。每个 product 必须单 use、
逐坐标同形；保留原 i16 加法边界和迭代顺序，不合并不同 carry，也不只改写链的前缀。
这条关系不增加作者 reduction，不改变最后的归约位置；资源仍由后续 pass 重新闭合。

两个单 use 的整数 `widen` 若直接供给同一个 add，且窄 operands 的 signedness/type 与完整
物理坐标相同、宽 result 保持 signedness 并具有双 SEW/LMUL，可以选择局部
`rvv_widen_add`。op 只保留原 active VL 上的逐元素加法，分别拼写 `vwadd` 或 `vwaddu`；
不接受 mixed signedness，不穿过其它计算或改变 reduction。选择器与 verifier 共用类型及
target 合法性判据，memory reads 和外层 traversal 不变；其 window projection、资源重闭合
和 terminal part 映射仍必须显式成立。

### `PropagateRISCVLayouts`

从 selected operation anchors、logical axes、producer/consumer、control carry 和 target VLEN
推导每个 shaped SSA value 的 time/lane/register-replica/local factors、SEW、LMUL、`vl`、
validity 与 register groups。普通无特殊 anchor 的 shaped value使用 target 的固定规则：
最内逻辑轴进入 lane；一元素表示保持 scalar。LMUL 是实现该 lane extent 的最小合法值，
不是 emitter 默认值。
primary lane axis 即使也由其它 use 传播到 coalesced 集合中，其 lane span 只计一次。
同位宽 integer Cast 的输入与结果共享完整物理 mapping；所需表示变化在 Cast 外显式
插入 `convert_layout`，op verifier 与 final verifier 均检查这个关系。
bounded register reduction 可以保留一致的 free-axis memory supply，以 register replica
表示被消除轴；结果必须逐轴保留其余 mapping，不强制将归约轴搬入 SIMD lane。
已实例化的 LMUL binding 是基础表示宽度，不是所有 SSA 值的统一上限。
保持同一 logical lane span 的 cast/widen/narrow 链会按 SEW 比例唯一派生每个值的
LMUL；widening contraction 的两个 operand 再共享该 lane span。一维同轴的宽度转换还须
按 operand SEW 与目标最大合法 LMUL 限制 result lane capacity，其余元素保留为 issue time；
不能让窄 result 要求无法表示的宽 operand。没有合法派生 LMUL 或最终联合 live set
超资源时，当前 candidate 失败，不回头改作者树。
整除的 proper domain Extract 是投影，不是父子 lane-span 等价关系：子域可以使用更窄的
carrier，但不能反向压缩父值。父值保持独立合法表示，后续 typed extract 或 partitioned
reduction 消费它；逻辑分组、数值宽度与 reduction 边界不改变。
storage layer 宽度约束 raw field 和保持表示的 cast/conversion 链；lookup 产生新的
数值结果，不继承其 indices 来源的 storage lane 上限。读取与查询的不同 lane span
由显式表示转换连接，不能把超过 grouped/layered 宽度的逻辑 lane 伪装成一次读取。
完整 register table 与查询使用相同 SEW/LMUL，但有效 table 元素数可以少于查询 lanes。
table 完整元素能放入所需 register capacity 时，shared materialize 或同 SEW 的单消费者
table load 可匹配查询的 LMUL；原读取点、逻辑 extent、VL 和 lane/time mapping 均不改变。
不同 SEW 的 memory lookup 不属于该寄存器匹配候选，仍由 typed indexed-memory 关系选择。

同一个 widening contraction 的两个 operand 必须共享一套 reduction carrier。两侧
storage proposal 一致时保留该 mapping；不一致但拥有相同 reduction axes 与 extents 时，
按作者声明的 reduction-axis 顺序建立共同 carrier：最后一轴作为 primary lane，其余轴在其外
coalesce。storage order 只决定 operand 怎样供应这个 carrier，不能分别替两个 operand 决定
互不相容的 contraction layout；所需差异必须由后续显式 memory form 或 conversion 承载。

pointwise、state 与 control handoff 的要求不一致时，pass 插入真实 `convert_layout`；
`for/if/while` 的 carried/result types 同时被改写。无法保持 logical axes 或无合法 LMUL 时
失败，不通过静默 scalarization 继续。

若一个单轴 encoded metadata value 经保持既有 axis 的 pointwise 链扩展成 lookup index，
原 metadata axis 的 storage-contiguous carrier 仍沿 use-def 保留；新加的 entry/payload axis
由 lookup relation 决定。该规则只接受保持原 axis extent 的 typed 链，不能把任意 affine
storage proposal 当成 contraction carrier。

### `PlanRISCVMemory`

读取 `MemDescType`、Encoding field mapping 和 value layout，为每条 memory edge写入
`AccessAttr` 与 transfer leaf。dense access 得到 unit/strided/indexed；encoded field 得到
natural/grouped-layered/joined 的完整 storage geometry。lookup table被保留为 descriptor edge，
不是先整表 load。contract 的 lane operand与source storage relation在这里确定，并在
composite lowering中消费。partial materialization后新出现的replica memory edge会再次经过
该 pass写入typed storage relation；它最终采用unit、strided还是保留indexed，唯一由后置的
`MaterializeRISCVReplicaStorageLoads`根据最终consumer layout决定。

从 admitted table 改成 consumer-local memory lookup，必须证明原读取点到 consumer
没有可能别名写入，包括嵌套循环的较早迭代。无法保持原读取且没有合法的已物化 table
表示时，明确拒绝该候选。memory-descriptor lookup 声明 Read effect；只有寄存器 table
lookup 是 pure，CSE 不得跨写入合并前者。
同轴的一维full寄存器table若以`iota + 常量`读取完整、对齐且不越界的连续窗口，
可直接选已有`rvv_issue_slice`，不再生成向量gather索引。常量求值最多访问32个节点，
只处理无溢出的index加乘与值域可精确保留的cast；iota及加法也必须在索引位宽内。
这只投影已经读取的numeric SSA，不新增、延迟或扩大memory read。
不能合法延迟的静态连续 dense table 在原 load 点形成 `local_alloc → dense_snapshot`，
后续 lookup 读取该私有 descriptor。显式 stage 的 owner/birth/lifetime 保存在 allocation；
没有 stage 时采用原读取的 Level owner。超出目标 local 容量或非连续、非静态表明确拒绝。

indexed-entry 的地址索引若仅由同形、同轴的 unsigned 8/16-bit value 扩展至 32-bit，
且完整 byte offset 范围可由 16-bit 表示，可以直接选择 16-bit extension/identity 与
对应 EEW。必须验证 byte stride、范围、两侧 layout projection 和目标能力；不在一般
32-bit 算术链末尾追加窄化，也不改变 table/data 的读取宽度或位置。

物理 word read 后只观察一个完整字节时，可以选择 `rvv_byte_gather`。当前匹配 unsigned
u16/u32 read 的 byte-aligned shift 加 `&255`，或非饱和 `rtz` 的低字节窄化链；shift 的
byte selector 必须经有界 use-def 与范围证明位于原 word 内。encoded field 必须是
little-endian、完整的一维 natural storage；dense descriptor 必须连续。word 到最终投影
的中间链只能有唯一消费者；读取与投影之间最多检查 256 个同 block 操作，可跨普通
只读 effect，但拒绝写入、分配/释放、未知 effect 和 region。不能把已物化 register table
改为重新读内存。返回字节以显式 widening 恢复原消费者的类型与 layout，不改变数值图。
该 widening 也适用上述 indexed-entry 16-bit 地址规则，包括较早 nested-memory pass
已选中的 entry edge；不为其它 32-bit 解码链追加窄化。
字节地址中的 unsigned quotient/remainder 可按
`(base + (x >> k)) * 2^k + (x & (2^k - 1)) = base * 2^k + x`
合成，当前只处理 2/4-byte word、至多四层 pure axis/layout projection，以及同一 SSA
或完整 type/start/end 一致的 iota。该等式保持原整数宽度的 wrap 语义，不跨读取边界，
不使用无界结构比较。`SelectRISCVMemoryByteProjections` 只在最终 layout canonicalization
后的 memory 收尾选择该 RVV leaf，不在较早的 memory planning 提前冻结通用 lookup；scalar consumer 的反向
rematerialization 必须先完成。

已选 byte gather 的地址若可证明为同宽 unsigned 仿射表达式，连续的 lane 后缀可形成
多字节窗口。分析最多访问 64 个节点、深度 24；只穿过 pure layout/broadcast 与同宽
add/sub/常数乘法，不穿过 shaped narrowing 或数据查表。匹配只形成一至四个窗口，
每个窗口至少两个字节，result 必须是一个完整 RVV carrier；不枚举其它 layout。
相对 storage 的 `byte_base` 数值与窗口偏移必须是 power-of-two 窗口宽度的倍数，从而
证明窗口内部不会跨过原 unsigned index 的 wrap 边界；这不提高 storage 地址的对齐要求。
各窗口起点仍按原位宽计算并物化为显式 scalar SSA bases，原 gather 的字节集合、顺序、
读取点和 snapshot 关系不变。选择结果及窗口数/宽度/bases 均可直接从 Physical IR 观察；
它只替换字节读取和地址供应，不改变后续索引值、contraction 或 reduction。

同一 leaf 也可承载连续 byte descriptor 上的双窗口 pointwise lookup：
`(a*(1-h)+b*h)*w+p`，其中 `h=lane/w`、`p=lane%w`，lane 是同一一维逻辑域内
`0..2*w` 的 unsigned iota，w 是至少 2 的 power-of-two。关系最多比较两个乘法顺序，
每条坐标链最多穿过 16 个 unsigned value-preserving cast/widen 或 pure layout conversion。
穿过的每个 value 与 iota 必须具有一致的 full-valid time/lane/replica 等 pointwise mapping；
允许不同 SSA 的等价 iota，不把 SSA 身份当成逻辑坐标相等的必要条件。
`h` 的取值只有 0/1，两个 scalar `a*w`、`b*w` 在原 unsigned 位宽内计算，其低位为零，
所以窗口内部不会 wrap。两个 unit load 和一个 slide pack 替代向量插值地址与 indexed load；
原 source、读取点、byte signedness、输出 lane 顺序和后续数值运算保持不变。

同一block内，若一个byte-aligned natural encoded scalar沿纯一元链形成一个多消费者
supply，pass插入`register_materialize(realization=physical-share)`。该op没有作者
birth；它只冻结一次已选load/decode结果，使后续不同layout consumer共享同一SSA值。

若typed regular-repeat relation证明少量encoded scalar各自覆盖完整lane window，pass先物化
typed regular-repeat gather，再沿pure pointwise use-def追到第一个vector consumer。只有source
count、repeat、lane/time/replica mapping与每个physical part的source base全部闭合时，才把该链
改写成`rvv_regular_repeat_scalar_load`与scalar-carrier pointwise链；vector boundary保留并消费
scalar-vector leaf。该改写不依赖format名字，也不把任意storage proposal当作contraction
carrier。无法闭合或不会到达vector boundary时保留原typed gather。

joined field 的最后一个 role header 与 low-bit payload 物理相邻时，完整零基 source
window 可选择 `rvv.regular-repeat-joined-unit`：一次连续 byte load 后用已物化的 repeat
indices 做 register gather。group 为不超过128的二次幂，完整两组byte均落在field storage
内，所有结果同type、full、SEW8且carrier容纳整个窗口。shift后不再需要的mask由planner
证明并冻结为leaf参数，verifier重算；scratch固定4倍result register groups。该关系不把
不同逻辑field当成同一field，也不移动读取越过write或unknown effect。

对一个完整 encoded field 同时有外层 RVV consumer 与内层 `group_index` scalar
consumer 的情形，pass 只在 field owner/name、logical shape/axes、parent/sub-Level point
几何、分组步长和重复 decode 全部一致时，在两类 use 的共同支配 Level 物化
`local_alloc → local_bind → rvv_local_materialize`，子 Level 以真实 `local_load` 读取。
这种 `physical-share` local lifetime 完全属于物理 IR，不增加作者 staged birth。
其他跨block/跨Level placement 仍需更一般的 dominance/LCA、effect 与 cost-independent
reuse 证明，不由该窄合同猜测。

### `CanonicalizeRISCVLayouts`

该 pass 只做已经实现的真实 rewrite：相邻 pure 逆 conversion 消除、单 use 纯 pointwise producer
的 backward rematerialization，以及同一 block 内相同 conversion 的 SSA CSE。它不声称已经
实现跨 block hoist/sink、全局 conversion algebra 或任意 producer rematerialization。
conversion CSE 与 backward rematerialization 处于同一个固定点：合并conversion后新出现的
single-use producer须在本次pass内继续处理，不留给下一次解析/重放才改变其表示。

partial materialization 后，`physical-share` 的 input/result 都变成单 use 时，若到唯一
conversion 的同 block 区间没有写入或未知 effect，允许移除这条已无共享用途的边界，
继续按原有 typed index/layout 合同 backward rematerialization。作者 materialization、
多 use supply、跨 block、跨写入和 final resource closure 后的情形不适用。
同位宽 integer Cast 可随 conversion 重物化，但其 input/result 必须采用完全相同的
目标 layout，不能在 reinterpret intrinsic 内隐式改变 LMUL 或 logical mapping。
indexed Extract 的 index 投影可以沿单 use 纯 pointwise 链和 iota 重物化；预检查至多访问
32 个索引节点，并逐输入核对可执行的 layout projection。不能直接拼接旧 lane parts，
不等于该纯 index 程序不能在目标 layout 中重新形成；memory producer 不由这条规则复制。
若 Extract 仍投影 grouped/layered storage，不能把超过对应 storage layer 的 SIMD lane
span 向读取侧重物化；保留合法窗口读取后的显式 numeric pack，而不是制造未闭合的跨层读取。
若最小合法 LMUL 使 source part 留有空余 capacity，part-to-lane pack 仍按 pieces 倍率
使用闭合的中间 carrier，再显式缩到 consumer LMUL；pure conversion 合成不得消除这个
必需中间表示。两条边保持完整逻辑坐标、VL 与读取点，资源按实际 carrier 重新闭合。

单 use memory lookup 接 scalar conversion 时，在同 block 无跨写入的条件下，可以将
indices 投影为 consumer 的 scalar time/replica mapping 并直接 scalar lookup。新的 memory
form 和 leaf 由后续 `PlanRISCVMemory` 重选；该规则不把寄存器 table 退回内存，也不 hoist
潜在空循环中的读取。
完整的 regular-repeat gather 仅被 scalar conversion 使用时，可在原读取位置直接形成
scalar supply，复用重复的 source byte。该规则保留归约轴及单元素 free-axis mapping，
要求同 block、无跨写入、完整窗口与精确 part-base；不需要重物化整段坐标/算术程序。

一个单 use、无 snapshot 的一维 dense unit load，经 pure conversion 只供给一个 pointwise
consumer 时，可以在原读取点直接采用该 consumer 要求的分片。仅接受已物化的 domain-point
slice、整除的 full lane/time 分片、同 element/axis/validity、无 replica/local/fragment 扩张，
并检查目标 layout 与同 block 无跨写入。读取字节集合和读取位置均不变；memory form、leaf
与资源须重新闭合，不把宽 load 本身当作收益，也不复制多 use 或 indexed 读取。

### `LowerRISCVComposites`

把 source-level physical composites 改写为终端前的真实程序结构：

- Level loop变成带 Level identity 的 `scf.for`；
- state/local materialization变成 local alloc/bind/load/store 或 register/staged SSA；
- grouped/encoded reduction变成显式 reduction loop与 typed window load/step；
- ordinary contract变成显式 reduction loop与 RVV step；
- IME contract变成 typed fragment pack/MMA/unpack；
- layout conversion获得唯一 terminal conversion leaf。

未知 conversion kind、缺失 reduction extent、缺失 access 或不闭合 fragment capability均明确
失败。该 pass 不创建 source outer traversal、Level、workspace 或 persistent Encoding。

`MaterializeRISCVPrograms` 紧随其后，把仍以 transient macro op 承载的已选
local-pack 与 grouped-MAC program 展开成真实 `scf` control、point、load/step 和 SSA
use-def，然后删除 macro op。该 pass 只展开已冻结的 plan，不重选 layout、memory
form、tail 或 leaf；final verifier 拒绝任何尚待 terminal emitter 解释的 macro program。

`VectorizeRISCVRecordLoops` 消费 target profile 的 record-axis 固定结构优先级。选择
`across-records` 时，它只接受 ascending、zero-based、unit-step、无 carry 的 exact Level，且
source 与 destination 都必须证明为同一一维 unit-stride record axis；point 必须有静态且相等的
active/partition，store value不能依赖被替换 loop 的 induction variable。loop body只允许一个
domain-point store、同一 encoded record 的 field load、闭合 pure pointwise use-def，以及不被
可观察闭包使用的 dead pure op；额外 read/write 或未知 effect 都拒绝。pass 将
动态 trip count显式拆成整 cohort 的主 `scf.for` 与原语义的 scalar tail，并物化
`record_cohort`、record-strided storage load/decode/store。cohort width、每条 field 的 storage
unit、静态 byte stride、logical ownership 与 output offset全部进入 typed op；terminal emitter
不能重新推导 ramp、stride 或 tail。任何 proof 不闭合时保留原 within-record physical program，
而不是猜测跨 record 映射。

contraction 的 unroll 只能属于它真实的多 issue reduction loop。单 issue 已经覆盖
完整 reduction carrier 时，不存在可展开的 local issue loop；该 unroll 不得改挂到
最近的外层 Level，否则会复制整个 storage-block program。

### `FuseRISCVBitplanes`

连续 packed-u1 的 bitmask decode 根据 field geometry、record origin、子 Level point 与
目标 time/lane 范围选择。pure `time_to_lane` conversion 与已经直接承载 RVV layout 的
`extract` 共用同一关系；直接 extract 限于每 byte 八个连续 logical bit 的 layer=1 表示。
选择前核对 byte alignment、lo-first 位序、完整 group、point partition 与唯一 streamed axis，
不以 conversion 节点是否恰好仍存在作为能力边界。

点式 signed integer `data * (1 - 2 * widen(bit))` 可以在同一 block 的单 use 链上选择
`rvv_masked_negate`：bit 必须来自已物化的 logical-u1 window，sign 与 data 的完整 type
一致，mask 与 data 的全部 physical coordinates 和 mask ratio 一致。匹配最多剥离 16 个
同 type、单 use 的 pure copy；不穿过 read conversion，不复制或移动 mask 读取。
data 上已有的 widening/cast 原位保留，因而不能用窄整数取负替代宽整数取负；也不改变
scale/reduction 的位置。`1-2*bit` 与 reduction 使用的 `2*bit-1` 分别匹配，不能混用符号方向。

尚未被product/reduction关系消费的两种符号表达，也可直接选择 `splat(±1)` 加既有
`rvv_masked_negate`，复用 window-load 或 point-anchored bitmask-decode 的 mask-only leaf。
mask读取点、time/lane/replica、tail、下游原宽乘法均保持；取负对象只有常量±1，不将
任意i8数据取负后再widen。一个选择展开成多个register replica时，所在物理loop保留
已选unroll绑定，禁止系统编译器根据缩短的intrinsic表面再次展开该register cohort。

### `ScheduleRISCVLevels` 与 `PipelineRISCVLevels`

`ScheduleRISCVLevels` 是 scheduler。它读取已 lowering 的 physical `scf.for`、SSA use-def、
loop carry 和 operation effect，把与下一迭代无关的 pure/read producer 放在 stage 0，
把依赖 carry 并生成 yield 的 consumer 放在 stage 1，并在具体 operation 上写入
stage/order。它不通过 op 数量、相邻关系或 window family 命名 cluster。嵌套 region、write/
unknown effect 目前没有 predication/ordering 合同，因此明确拒绝，不得重排。

`PipelineRISCVLevels` 是 expander。depth 1 只消费 schedule；depth 2 消费 scheduler 写入的
stage/order，为所有跨 stage SSA value 增加 loop-carried 版本，支持多个 source carries，
并生成空迭代 guard、prologue、steady-state loop 和 epilogue。当前只实现距离一迭代、
depth=2/buffer=2；嵌套 region predication、write effect、depth>2 和显式 local-storage ping-pong
尚无合法 physical program 时直接 unsupported。仅设置 `pipeline_depth=2` 而没有
dependency-derived producer/consumer cluster 仍会失败，不算 pipeline。

### `ShareRISCVLayeredWindows`

该 pass 读取 typed storage geometry、field owner、logical point、consumer layout 与 affine
record relation，物化 layered stream、record load 和 field-to-register replica load；在 pure
conversion、无中间 write/unknown effect 时，它还能证明多个 layer 共享一个 RVV storage
window。成功时都以真实 physical operation替换 extract。当前仅对 full-validity layer做共享；
tail cohort没有证明安全时保持原程序。

### `MaterializeRISCVReplicaStorageLoads`

该 pass 位于最终一次 memory planning 之后。它读取已经闭合的 typed field/storage relation、
affine index、time/lane/replica分解、consumer layout与target facts，按target固定优先级选择
保留indexed、unit或strided field-to-register form。成功时把extract物化为
`rvv.replica_storage_load`；选中的op明确携带window plan、每个physical part的source window
以及exact load leaf，layered relation不会伪装成strided load。

这个 pass只闭合现有 field-to-register edge，不创建logical axis、Level或source traversal。
partial materializer若需把一个已选unit load投影到issue window，只能机械投影原有plan与leaf，
不能重新选择memory form。
投影后的正 lane extent 可以为 1；它仍是已有 RVV carrier 的一个显式窗口，须满足
`vl == lane factors 的乘积` 及完整 part/window 映射。若随后转为 scalar，terminal 按已选
part 坐标提取 lane 0；这不改变 layout propagation 对普通 singleton 值默认选 scalar 的规则。
同一 block 内的单寄存器 natural/unit 读取可以复用一个已经存在的完整连续窗口。该规则
向前检查至多 32 个候选，要求同一 field SSA、record 坐标、常量逻辑范围与无写入/未知
effect 的区间；相同范围共享 SSA，连续子范围用已有 `rvv_issue_slice`，不增加更宽读取。
子范围只缩小最外侧的有效 lane axis，并保持其余坐标与 validity。跨动态 issue、其它 memory
节点与无法闭合的范围不由这条规则处理；完整资源分析仍核算共享后的存活区间。

同一 block 内两个完整一维 natural strided window，若 field owner、Encoding、type 与坐标
相同，步长都是 2、静态 base 相差 1，且每个 time part 覆盖相同的相邻交错区间，可以选择
`rvv_segment_pair_load`。匹配至多向后检查 32 个 load，并拒绝中间 write/unknown effect；
必须证明两路读取的并集没有越界、element alignment、NF×LMUL 与完整 tuple 临时资源合法。
两个结果保持原 layout，后续 widening、加法和 reduction 不变；emitter 只拼写已选 segment
load 与 tuple component binding，不重新选择 memory form。

### `PlanRISCVPartialTopologies` 与 `MaterializeRISCVPartialAccumulators`

两者分别实现在 `lib/Target/Partial/PlanTopologies.cpp` 与 `Materialize.cpp`，共同的
typed topology/几何工具在 `Support.h/.cpp`，add-tree 的规划与物化共用 `AddTrees.cpp`；
planner 与 materializer 不通过 include 实现片段形成第二份算法。

planner 读取 contraction/reduction 的 typed axes、time/lane/replica 分解、storage window、
consumer use 与 target resource budget，唯一写入 product carrier、issue slices、partial-set、
combine/final-reduction topology、selected local instructions 和联合资源合同。顺序 fused contraction
若跨多个 issue，必须得到显式 `sequential_partial_plan`；该 plan 给出每个 issue 的 source-part
投影、每个 operand 已选的 `slice` 或 `storage-rematerialize` supply、唯一 widened accumulator 与
最终 reduction，而不是让 composite dot 按 issue 数预留一组临时 partial。storage rematerialize
只对 typed grouped/layered field window 及其保持 axis 的 pure producer closure 合法；它投影已经选定的
storage plan，不能重选 memory form。当前该 fused program 只在 lhs/rhs 的 issue slice 与 reduction-only accumulator
具有同一完整 physical-part 映射时合法；带互异 free-axis replicas 的 outer contraction 由 planner
明确选择 `sequential_per_stream`，materializer 不得在失败后自行降级。

materializer 只核验仍存在的 SSA graph 与 plan 一致，并实例化真实
`rvv_issue_slice → rvv_widen_accumulate → rvv_finalize_widen_dot` 或对应 partial-set program。
它不能重新选择 carrier、combine tree、instruction 或 resource groups。没有闭合 plan 的已选
topology 直接失败；terminal emitter 只按 `source_parts` 投影已有 vector binding，并拼写已选
accumulate/reduction leaf。nested、layered 与 scaled/reduced-scaled plan 同样必须在 planner
中冻结 issue types、scale supply、narrow-scale type、storage-window/decode instruction、partial
reduction、finalization leaf 与同时存活的 resource groups；materializer 只能按这些字段创建
operation，不能从 shape 或当前 SSA spelling 再次推断。

nested planner 的最大 issue window 若不能闭合资源，可以沿同一物理轴依次减半，
只考虑能精确划分原逻辑 extent 的窗口，至多检查 64 个候选，优先选择 issue 数最少的合法项。
这不改变作者的 reduction axes、scale 位置或整数宽度。选定的 window、issue 数与资源估计
写入原有 nested plan；全部不合法时报告尝试数并失败，materializer 和 emitter 不重试。
窗口缩到一个元素时，带轴坐标仍保留其轴；memory pass 必须沿已支持的仿射表达式
物化无轴标量地址基值，不能把单元素 shaped value 直接冒充 scalar。

需要 partial-repack 的 layout、nested 或 add-tree plan 还必须保存已选 repack leaf，包含
addressable carrier 与临时资源；无需 repack 的位置使用显式空项。materializer 核验并消费
该 leaf，不重新选择 `split`/`slice` 或载体宽度。

带 Level-local issue unroll 的 scaled contraction 在展开前只有一个 loop-carried
contribution。planner 必须从该 seed 冻结完整 slots、product carrier、scale supply、
combine topology 与 resource groups；materializer 再按该 slots 机械展开。scale 可以是
typed scalar carrier 或已终结的 signed-i32 scalar，两者都必须由 verifier 证明为单一
replica；不能因展开时的 SSA 拼写不同而重选 topology。

nested scale 与 indexed operand 共用具体 field 供应时，planner 保留表示转换路线，
不能仅因 replica 数较多就选择会重新读取字段的 scalar rematerialization。deferred scale
的 issue 克隆复用已有 field extract、Read-only supply 及共有 pointwise decode；
width/layout conversion 仍可在各 use 点重新物化，避免为共享而延长宽载体的 lifetime。
复用以同一个 SSA source 为键，不以字段名称、格式名称或相似地址猜测等价。

若完整scale已经有其它consumer、单轴full值可驻留一个vector register，且issue窗口
至少有四项，nested plan可选择`shared-before-issue`：保持原scale SSA，在各issue中以
显式numeric lookup取得连续窗口，不再克隆其算术链。该选择冻结在plan中，并额外预算
共享源、窗口与索引的存活资源；最终resource pass仍检查完整live set。它不改变原scale
的计算宽度或乘法/reduction结合位置，也不把没有共享收益的小窗口强制搬到vector上。

当每个 logical slot 本身跨多个 issue 时，planner还必须冻结 issue operand supply 与完整
product carrier。materializer先用显式`rvv_widen_multiply/rvv_widen_accumulate`形成每个slot，
再由`rvv_partial_collect`按SSA operand顺序建立typed slot集合，之后才执行partial reduction与
scale combine。`rvv_partial_collect`不生成新的数学运算；它使同时存活的独立product、slot顺序、
birth/lifetime和资源总量成为可验证的IR合同，terminal translator不能从原dot重新构造这棵程序。

independent i16 partial tree 若没有同宽跨 stream 合并的合法证明，首层 pairwise combine
必须 widening 到 i32，后续层保持 i32，再做最终 reduction；这同样适用于四路及更多
power-of-two streams，不能仅为两路特殊处理。选择前同时计入 operand/source-set live set、
第一层 source/widened-result live set 和最终收敛资源，并遵守 target 的单载体 widening 限制。
已冻结计划的 verifier 拒绝未经证明仍在 i16 中合并的首层；materializer 只消费这些 typed stages。

上述 plan attributes 是第二层内部、一次 lowering 中的瞬态冻结结果。完成物化后它们必须删除；
final verifier拒绝任何残留plan，terminal translator也不读取它们。只有单stream的closed widening-dot
leaf可以直接保留typed topology、lane/source-part relation与exact leaf，而不保留待解释program plan。

### `CoalesceRISCVPartialExtractions`

issue 展开后，重用上述 add-tree planner/materializer 合并已完成 reduction 与 scale 的
signed-i32 singleton partial，再做一次 scalar extract。只接受同一 block 内、单一 consumer、
同一 reduction axis 与完整 RVV layout 的 `rvv.partial-finalize.extract`；不接纳 i16 partial、
尚未 reduction 的 lane 集合、原始 dot 或 scale 前移，不改变 source Level/carry 或浮点结合。
局部搜索最多访问32个add节点和32个partial leaf；plan只保留互不重叠的最大可接受树，防止
子树改写使父计划的leaf身份失效。物化只生成既有 `rvv_partial_merge` 与finalize，完整资源
分析仍重新计算延长的partial lifetime；局部预算检查不是全局live-set合法性的替代。

对已归约的signed-i32/VL1 partial，另比较scalar scale/combine与显式
`rvv_partial_packed_scale`：后者按完整slot permutation把2至8个结果装入m1的有效lane，
与同轴、同序、完整的i32 scale向量相乘，再归约为原来的单slot。输入允许有界的partial-set
与scale-window列表：分别按operand顺序连接slot与scale lane，数量必须相等并适合target的
m1容量；slot permutation索引连接后的partial序列，不合并或丢弃各slot的独立权重。
输入/输出partial类型、
原scale位置、整数宽度与terms归属不变；不接纳tail、多个输出slot或跨轴scale。
scale的LMUL调整保留显式conversion，terminal仅拼写已选pack/multiply/reduce。
同block、single-use、顺序明确的相邻scale/combine可成对选择，最多检查32对；每对在第二个
原consumer处完成，避免把全部partial延长到最终merge。满unroll且两个窗口可容纳于m1时，
完整的一维scale可在至多两个register groups内供各issue共享，仍保留原读取和数值结合位置。
已选scale slice若来自同一数值SSA的连续同轴窗口，operation selector可合为一个完整窗口。
完整且整组对齐的issue slice由同一selector选择`rvv.issue-slice.split`，verifier按target
VLMAX、full active lanes、整数LMUL比和offset对齐闭合，terminal只拼写`vget`。
slot pack统一使用最终VL：每次slide保留已构造前缀，未构造lane在唯一的multiply/reduce之前
全部被后续slide覆盖；输入的未定义尾lane不会进入最终数值运算。
局部选择先检查target、坐标与资源合同，再比较包含pack、seed、reduction和VL设置的成本；
可scalar-rematerialize的供应以无vector extract的成本下界比较。供应搜索最多32个节点，
包括静态一维`iota`：同逻辑range/axis的scalar replica使用既有`register.iota`，
不能因初始vector坐标表示漏掉这条合法供应；查询与物化接受相同的producer集合。
估计以`weft.riscv.packed_scale_cost = [packed, scalar]`保存在selected op中，不作为合法性证明。
单scale窗口声明三个temporary vector groups，多窗口拼接声明四个；最终live-set仍由资源pass重算。

该pass位于普通编译与layout-input恢复共用的收尾入口，并由`weft-opt`公开供重放。已带
final resource marker的图不重新选择；独立副本须先清marker和统计后才能重放此改写。

### `FinalizeRISCVLeaves`

此收尾还将 conversion 输入侧的 encoded 读取显式物化为 `field_read`：
`local_load`/`time_to_lane` 的字段供应直接产生已选结果；其它转换先读取原表示，再转换 numeric SSA。
复用既有 access 与 result layout，不追穿已经计算的 numeric SSA，不改原 load/快照身份。
`field_read` 的 Read effect、读取点和结果存活区间进入后续 CSE、snapshot 与资源核算；
final pure `time_to_lane` 只允许已闭合的 numeric RVV pack，不保留 emitter 重读字段的例外。
已选的 register Extract 也是 numeric consumer：若它的 input 仍是 encoded storage
projection，先形成同表示的 `field_read`；register gather 的 result 不能重新解释成地址。

读取 `ImplementationAttr` 与已经确定的 value carrier/SEW/operand form，为 pointwise、cast、
reduce、state transfer等仍未闭合的 operation写入 exact instruction/spelling leaf，并删除
`ImplementationAttr`。结构 family只在 `SelectRISCVOperations` 产生。由
`LowerRISCVComposites`新建的window step、RVV contract step、fragment和layout-conversion op在
创建时已有自己的exact leaf；本pass只终结仍保留generic operation的那一组。两者作用于互斥的
physical op集合，不为同一个operation重复选择instruction。

在资源闭合前，整数 RVV pointwise 的 `x+0`、`0+x`、`x-0` 可直接复用输入；
输入与结果的完整 type（含 axes、validity、layout）必须一致。零值证明最多访问 32 个
producer，仅穿过整数常量、整数转换、纯布局转换、splat/broadcast 和整数乘零，
不把浮点恒等式或轴消除混入该规则。失去用途的广播链由后续同一条 CSE/dead-layout 主链清理。

无符号 RVV 整数值除以或模一个直接标量、非零二次幂常量时，分别选择同 type 的
`shr(log2(divisor))` 或 `and(divisor-1)` 与对应 exact leaf；不改 Canonical IR。
lhs/result 的完整 type 必须相同，rhs 必须是同元素类型的直接整数常量，因而不存在
广播/validity 变化或有符号负数舍入歧义。零除数、非二次幂、动态除数及有符号除法不匹配。
常量计算使用定宽 APInt，不靠宿主有符号移位；`weft-constant-div` 输出实际选中的商/余数数量。
emitter 沿用已选 shift/and 的拼写，资源由同一后续主链闭合。

已选 unsigned widening vector-scalar multiply 的结果若只被同宽右移消费，右移量是
输入 SEW 的直接整数常量，且右移也只被非饱和、rtz narrow 消费，narrow 的完整结果 type
恰好恢复原 vector type，则三步选择为单个 `rvv_multiply_high_scalar`。匹配只检查这条
固定 use-def 链；u8/u16 的 doubled-width unsigned product 不溢出，前置 wrap、后续
累加/归约与全部 axes 不变。emitter 只拼写该已选 leaf，资源由原有收尾重新核算。

同 block 的 unsigned `shr` 若只被非饱和、rtz 的二倍窄化消费，可选择
`rvv_narrow_shift_right`，精确拼写 `vnsrl.wv/wx`。输入与结果须有相同完整坐标和
validity，输入 SEW/LMUL 为结果的两倍；旧 shift 的 exact leaf 也须匹配该 operand form。
向量 shift amount 以显式 narrow 保留指令所读的低 `log2(input SEW)` 位，标量 amount
保持原输入元素类型。单个已物化 scalar 输入由既有 extract/splat 表达，amount 只允许
补入完整 singleton physical axes；不隐式改变其它 broadcast 或 register mapping。
融合留在原 shift 位置，不移动读取。相同 result type 的 scalar-splat/narrow-shift 对，
在输入与窄结果的 register groups 相等时，可将后一个纯 splat 提至其 numeric producer 后：
窗口最多 32 个操作，须跨同型 narrow-shift，且不能跨 region 或任何 effect。实际载体寿命
仍由后续资源主链核算。`weft-narrow-shift` 输出融合数和重排的 splat 数；emitter 不识别
producer 模式。每次融合消除一个已有 shift，不复制 shift producer，工作表有输入 DAG 上界。

### `SelectRISCVScalarLoadPrimes`

该pass消费已经实例化的boolean physical parameter，并且只在`FinalizeRISCVLeaves`之后运行。
启用时，它在每个physical loop中沿单result、pure、同block use-def查找第一个送入
`rvv_partial_set`的grouped/layered storage edge。合法edge必须已经具有完整window plan、exact
vector-load leaf；`rvv_replica_storage_load`还必须只有一个source window。pass把该op的leaf改为
对应的`scalar-prime` form，不改变type、SSA、control或storage geometry。启用但整个module没有
合法edge时当前candidate失败，而不是把参数静默当成零。

这个leaf固定为同一raw window末端的一次有序scalar byte read加原vector load。它是有限的局部
memory spelling，不携带future-iteration、distance、buffer或pipeline schedule；若需要跨iteration
prefetch，必须先在RISC-V IR中建立真实producer/carry/control，不能扩展本pass暗做。

### `MaterializeRISCVReadSnapshots`

encoded `load` 的 field/projection 可以延迟实现，但不能跨越可能修改原 bytes 的写入。
该 pass 沿读取结果的 use-def 检查中间 effects、参数 alias set 和嵌套循环；存在干扰时，
在原读取位置插入显式 `local_alloc`，将它作为 `load.snapshot_storage` operand，选择
完整 record byte-copy leaf。之后的 field 投影读取这份私有存储，原 Value 的 Encoding、
shape、axes 和 Level 归属保持不变。私有 local writes 不与 pinned View 混为一类。

快照要求静态、一维、unit-stride、完整且非 interleaved 的 record 区间；未满足的干扰读取
必须拒绝。容量和 lifetime 由 `LocalType` 承载，资源 pass 核算分配；没有干扰的读取不增加
本地复制。final verifier 拒绝仍跨干扰写入但没有快照的 encoded load。
copy leaf 在 target 合法的 m1/m2/m4 中按精确 transfer 次数与 scratch 占用选择，并记录
全部所需参数。它完成显式 exact-window transfer，不把热路径交给带未知向量 clobber
的库函数调用。
已有 snapshot storage operand 的 load 保留这个已验证的 copy leaf；重放 memory planning
不能把它改回普通 load，清 resource marker 也不改变 snapshot 的归属和读取合同。

### `CloseRISCVLeafResources`

从已选 op 的 typed operands/results 统一 leaf 的 operand/result resource fields，不冻结
kernel 的 final resource marker，不改变 leaf instruction、临时资源或表示。随后 CSE 与
dead-layout 清理可合并仅因旧资源字段不同而未合并的相同纯供应；真实 SSA live set、spill
和峰值仍在 `MaterializeRISCVResources` 中重新核算，不能在清理后复用旧统计。

### `FuseRISCVPhysicalIssueLoops`

仅消费 nested partial materializer 写在生成的 `scf.for` 上的 `weft.riscv.issue_window`
（axis、extent），不从作者循环或名称猜测 issue。候选必须具有相同边界、步长与属性，
两个平坦 body 只含纯操作或 Read effect；第二个循环的初值和外部 operand 必须支配第一个
循环，循环之间不能跨越副作用。每次只合并两个遍历，每个 kernel 最多检查 32 对候选；
合并结果不再参与扩大合并，候选标记在 pass 结束时消费掉。

候选副本先 CSE，再按 SSA 读取前沿安排独立消费者，避免机械展开后的共享供应跨越后续
issue 才被消费；每条原有累加依赖保持不变。仅当实际静态 Read 数减少、且原有
region-aware 资源分析的峰值不超过 target 寄存器预算时接受。候选数、共享 Read 数、
资源拒绝数与最大已选峰值写入候选诊断，并接入 pass statistics。这个估计不代替 effect/数值/type
合法性，也不保证 intrinsic C 后的 LLVM 寄存器分配没有 spill；有限物理绑定仍由 kernel
旁的配置提供并通过真实运行选择。最终资源 pass 重新核算实际选择的 Physical graph。

### `MaterializeRISCVResources`

用 region-aware SSA live interval和 leaf temporary计算 vector/fragment peak，并把每个 leaf 的
operand/result resource groups写实。首次核算前，可将同 block、single-use 的纯整数计算链
移到唯一 consumer 前：候选仅含 full RVV `binary/cast/widen/narrow`，最多32个节点，
输入边界仅接受常量或已物化的 numeric read；不移动读取点，也不穿过 deferred field/address。
只考虑跨 partial set 的最多256个操作窗口，窗口内不得有 region、write或未知 effect。
移动必须降低 `Σ register_groups × live_interval_length` 且不增加该 block 的实际 peak；
否则恢复原位置。估计不替代合法性，候选数、选中数和驻留成本减量由
`weft-resource-sink` 诊断输出。该改写不复制计算、不改整数结合树、Level或storage归属。
资源超限时，只对同一 block 内可合法保存的普通
`ValueType` 插入显式 local slot、`spill` 和各 use-site `reload`；fragment spill、跨 block spill
和任意 pure-producer rematerialization尚未实现时判当前 module非法。
pure 或仅有 Read effect 的 producer 均可提供被 spill 的数值 SSA；原 producer 保持在
原位置，只保存其已经产生的值并在 use-site reload，不复制读取，也不延迟读取。
encoded storage projection 本身不是 spill 对象。捕获索引 spill 后，同一 consumer 的相同
索引复用一次 reload；`field_read` 的纯地址投影在该读取点按它自己的索引 operands 重绑，
严格保留 selector、type 与来源关系，重新核算资源，不克隆数值 producer 或扩大读取字节。

若 CSE 共享已有 reload 后扩大驻留区间并再次超出预算，可以把该 reload 分到同一 block 的
各 consumer 前；同一 consumer 的重复 operands 仍共享一次 reload。此改写只接受来自
私有 allocation、由同 block 中唯一且先于 reload 的 spill 写入、其余用户均为 reload 的
slot，且该值在超限点仅驻留而非必须作为输入存活。不新增 slot 或重新执行原数值 producer；
使用点本身所需资源超限不能借此通过。改写仍受初始 SSA 候选数量的尝试上限约束并重新核算峰值。

kernel 创建时 `resources_materialized=false`；只有资源分析、必要 spill 与峰值汇总全部闭合后，
本 pass 才把它设为 `true`。初始化为零的 peak 不是闭合证据。layout canonicalization 仅在该
marker 为 `false` 时允许改变 producer lifetime 的 backward rematerialization；marker 为
`true` 后只能做不改变资源合同的等价 conversion CSE。本 pass 对已经闭合的 kernel 幂等跳过。

动态 local object必须紧邻一个 typed `local_capacity_guard`。guard给出 target-bounded byte
上界，kernel resource summary按上界计算；缺少闭合上界时拒绝生成 C VLA。

### `VerifyFinalRISCV`

terminal eligibility 与 required-leaf 由 `RISCVOps.td` 的 `WeftRISCVTerminal` /
`WeftRISCVLeaf` traits 唯一定义；final verifier 不另维护 op 名单。terminal dispatcher
只登记实际实现，不能用自身分支补齐未选择的 leaf。

最终 verifier 拒绝：残留 canonical op、非 terminal RISC-V op、`unassigned` layout/access、
`implementation`、未展开 schedule、未选 leaf、丢失 Level birth/handoff、绕过 runtime
`memory_view` 的 ABI edge、`resources_materialized=false`，以及超出 target 的
register/fragment/local-storage 使用。

operation/type verifier继续检查 logical domain、axis projection、Encoding field、conversion、
window、fragment role/packing和 exact leaf之间的局部合同；final verifier还把descriptor
storage facts与Encoding declaration、leaf widening/memory要求与target capability逐项核对。
成功只说明这份 RISC-V module满足
当前已实现的 terminal op 集合；没有实现的 spill、pipeline或extension结构必须在前序 pass
明确失败。

## 3. 可观察性

每个 pass 的输入和输出都是 program。用 MLIR pass instrumentation dump 同一 module时，应
直接看到 type/layout变化、`convert_layout` 插入或删除、memory attributes、loop/window/
fragment rewrite、pipeline expansion与spill/reload。只打印 analysis table或 assignment
dictionary不能证明程序已经物理化。

独立 parse/verify 和指定 pass/CSE 的二次 diff=0 只说明该样本在该流程上稳定，不证明
全部 pass 正确或幂等。尤其 final resource marker 会跳过 layout canonicalization 的主要
rematerialization 分支。机械重放的完整解释与真机验收边界见
[物理优化方法](optimization-principles.md#12-机械验收的证明范围)。

## 4. 与 Triton/TileLang 的机制关系

下列路径相对于仓库根目录位于同级reference checkout `../ref/`。Triton 的 layout encoding、`ttg.convert_layout`、Coalesce、AccelerateMatmul、
RemoveLayoutConversions 与 Pipeline同样在 TTGIR 上改写真实程序：

- `../ref/triton/include/triton/Dialect/TritonGPU/IR/TritonGPUOps.td:32`
- `../ref/triton/lib/Dialect/TritonGPU/Transforms/Coalesce.cpp:71`
- `../ref/triton/lib/Dialect/TritonGPU/Transforms/AccelerateMatmul.cpp:441`
- `../ref/triton/lib/Dialect/TritonGPU/Transforms/RemoveLayoutConversions.cpp:42`
- `../ref/triton/lib/Dialect/TritonGPU/Transforms/Pipeliner/PipelineExpander.cpp:51`

TileLang也把layout、buffer与pipeline结果落到真实block/loop，再由LowerTileOp和
InjectSoftwarePipeline重写body：
`../ref/tilelang/src/transform/layout_inference/layout_inference.cc:1196`、
`lower_tile_op.cc:1080`、`inject_pipeline.cc:3608`。

Weft复用的是 typed representation、explicit conversion 与 real rewrite；逻辑值不因此获得
thread/warp/CTA ownership。
