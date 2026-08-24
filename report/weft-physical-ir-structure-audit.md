# Weft 物理 IR 与 RISC-V 编译结构事实审计

## 审计范围

本轮只读取当前源码、唯一规范和参考仓库，没有修改编译器、没有运行 kernel、没有产生性能数字。

核对范围：

- Weft 唯一规范：`report/weft-spec.md`；
- Weft canonical dialect、RISC-V planning dialect、八个 RISC-V pass、intrinsic-C emitter；
- Triton 的 TritonGPU layout、`convert_layout`、layout conversion removal/rematerialization；
- TileLang 的 layout、pipeline planning 与 pipeline injection；
- 当前 `doc/compiler/*.md` 与源码是否一致。

报告中的“物理 IR”采用一个可外部检验的定义：

> 机器相关的 value type/layout、conversion、memory operation、local operation、loop/schedule 是程序中的 typed SSA 实体；pass 的结果是改写后的程序，后续 pass 和 verifier 直接消费这些实体。

“瞬态”与“是不是 IR”没有关系。TritonGPU IR 也是编译期瞬态；关键区别是它是不是一份可验证、可改写、可逐 pass 打印的程序。

## 总事实

当前 Weft 有一个名为 `weft_riscv` 的 MLIR dialect，但没有上述意义上的 RISC-V 物理程序 IR。

`weft_riscv` 只有两个 module-level operation：

```text
weft_riscv.problem
weft_riscv.assignment
```

它们都没有 SSA operand/result、没有 region、没有 physical value type。全部内容位于：

```text
DictionaryAttr target/candidate/resources
ArrayAttr values/operations
StringAttr stage/status
```

定义见 `include/Weft/Dialect/RISCV/IR/RISCVOps.td:8-51`。verifier 只检查 stage 名称和少量顶层 dictionary key，见 `lib/Dialect/RISCV/IR/RISCVPlanningDialect.cpp:16-64`；它不验证：

- 每个 physical value 的 layout/type 是否完整；
- operand/result layout 是否兼容；
- 每条不兼容 use edge 是否存在 conversion；
- conversion 是否有 SSA producer/result；
- schedule 是否与实际 loop/body 对应；
- memory mapping 是否覆盖实际 access；
- selected local operation 是否与 physical operand type 一致。

八个 RISC-V pass 从不改写 canonical `weft_kernel` 程序。它们在 canonical module 的 clone 中追加并更新 `problem` 记录，最后复制 winner 为 `assignment`，删除 `problem`。intrinsic-C emitter 随后同时遍历：

```text
原封不动的 canonical Kernel IR
+
最终 assignment side record
```

并在 C++ 中合成实际物理程序。这个事实由 `lib/Target/RISCVCompiler.cpp:15-27,117-142` 直接给出。

因此核心怀疑成立，但需要精确措辞：

> 当前 Weft 有“用 MLIR op 包装的物理 assignment side record”，没有“typed SSA 物理程序 IR”。

---

# 一、Triton、TileLang 与 Weft 的具体机制差异

## 1.1 Layout 存放位置

### Triton

TritonGPU 的 layout 是 tensor type 的 encoding attribute。`BlockedEncodingAttr`、`SliceEncodingAttr`、`DotOperandEncodingAttr`、MMA/MFMA/WMMA encoding 等都实现 layout encoding；线性 layout 明确把：

```text
register / lane / warp / block
→
logical tensor dimensions
```

作为映射。定义集中在：

- `ref/triton/include/triton/Dialect/TritonGPU/IR/TritonGPUAttrDefs.td:597-701`；
- 同文件 `738-1494` 的 blocked、MMA、slice、dot-operand encodings。

一个 SSA value 的 layout 因而是它的 type 的一部分。改变 layout 会改变 value type。

### Weft

canonical `!weft_kernel.value<element, shape, axis_ids>` 只有逻辑 element、shape 和 axis identity；规范明确禁止 hardware layout、vector width、register tile、fragment，见 `include/Weft/Dialect/Kernel/IR/KernelOps.td:9-18,52-60`。

RISC-V physical layout 存在 `problem.values` 中每个 dictionary 的字段：

```text
physical_kind
physical_sew
lane_axis / physical_lanes
stream_parts / register_parts / vector_parts
register_axes / register_extents
lmul / vl
materialization
```

这些字段不是 value type，也不属于 canonical SSA value；只是用字符串 id 与 canonical value 对应。

### 直接后果

Triton 中 layout 不完整或 operand/result type 不兼容，会出现在 typed IR 和 verifier/canonicalization 边界。Weft 中缺少字段不会自然造成类型错误；emitter 目前会对多项缺失字段使用默认值：

- `stream_parts` 缺失取 1：`RISCVIntrinsicC.cpp:787-790`；
- `register_parts` 缺失取 1：`:807-810`；
- `register_extents` 缺失时填 1：`:825-836`；
- `physical_lanes` 缺失取 1：`:1083-1086`；
- `lane_axis` 缺失取 0：`:1089-1090`；
- `vector_parts` 缺失时重新计算：`:1113-1117`。

所以“conversion 漏了但没有报错”和“某条路径突然退化为 scalar/单 part”在当前结构中是允许发生的，不需要通过 verifier。

## 1.2 Conversion 是什么

### Triton

`ttg.convert_layout` 是真实 op：

```mlir
%b = ttg.convert_layout %a : tensor<..., #layout_a> -> tensor<..., #layout_b>
```

TableGen 定义见 `ref/triton/include/triton/Dialect/TritonGPU/IR/TritonGPUOps.td:27-45`。它有：

- typed source/result；
- layout verifier；
- canonicalizer；
- 独立 GPU-to-LLVM lowering。

`RemoveLayoutConversions` 不是修改一张 conversion 表。它可以：

1. 用 `value.setType(tensorType.cloneWithEncoding(...))` 原地改变 value type；
2. 无法统一时真正创建 `ConvertLayoutOp`；
3. backward-rematerialize producer，使 producer 直接产生 consumer layout；
4. 删除冗余 conversion op；
5. 反复 canonicalize，直到 rematerialization 不再变化。

关键代码见：

- 创建 conversion：`RemoveLayoutConversions.cpp:491-501`；
- 原地改 value type：`:515-520`；
- 改写 operand/result：`:522-550`；
- 完整 propagation/rematerialization/hoist/cleanup 顺序：`:1610-1695`。

conversion 是否被“消除”可以用两件事直接观察：IR 中 `ttg.convert_layout` 数量减少，或者 producer/result type encoding 改变。

### Weft

`ResolveRISCVLayoutConversionsPass` 不创建 op。它在 `problem.operations[i]` dictionary 上写：

```text
use_conversions = [
  { operand, source, target, relation,
    source_parts, lane_offsets,
    source/target lane axis, SEW, LMUL }
]
```

关系只有 `identity/scalar-broadcast/project/lane-to-register`。实现见 `lib/Target/RISCVResolveLayoutConversions.cpp:106-215,244-311`。

当前只为 pointwise、cast、widen/narrow 和 lookup 的 mapped operand 生成这些记录，见同文件 `87-103`。Level carry、for/while/if carry、handoff、admit/commit、reduce、contract、pack 等 use edge 没有等价 conversion 实体。

所谓 lane-to-register conversion 的“消除/共享”目前发生在 emitter 的字符串 cache：

```text
source-id : source-part : lane-offset
→ C temporary variable name
```

见 `RISCVIntrinsicC.cpp:965-1019`。它避免重复输出相同 `vslidedown + vmv.x.s`，但没有删除或改写任何 IR operation。

### 直接后果

当前 Weft 可以修改 conversion record，也可以让 emitter 少打印一次转换，但不能完成 Triton 意义的 conversion rewrite：

- 没有 conversion SSA result 可被多个 consumer 共享；
- 没有 conversion op 可做 CSE、DCE、hoist、sink；
- 没有 dominance/use-def 自动保证 conversion 覆盖所有 consumer；
- 没有 op verifier 强制 source/result physical type 对应；
- conversion pass 无法重物化一段 producer IR，只能修改表或依赖 emitter cache；
- “删除 conversion”在程序中没有可观察对象。

因此当前报告中“typed conversion 已经成为 value-use edge 的瞬态决定”是字典 schema 的事实；它不等于“已经建立 typed conversion IR”。

## 1.3 Pass 的输入输出

### Triton

Triton layout pass 的输入输出都是 program IR：

```text
带 tensor encoding 的 SSA program
→
value type 被改写、operand 被替换、convert op 被插入/删除后的 SSA program
```

layout propagation 的 anchor/conflict 分析可以是临时 C++ 数据结构，但结果必须投影回程序 type/op。后续 Coalesce、AccelerateMatmul、LLVM lowering 读取改写后的程序。

### TileLang

TileLang 说明“attribute”本身不是问题。它的 pipeline stage/order 先作为真实 `For` annotations 存在；`PipelinePlanning` 返回一个 body/annotations 已重建的新 `For`，见：

- `ref/tilelang/src/transform/pipeline_planning.cc:1172-1213`；
- 写 stage/order 并返回新 For：`:1281-1337`。

`InjectSoftwarePipeline` 再消费这些 For annotations，改写 buffer version、access index、async commit/wait，并生成实际 prologue/steady-state/epilogue，见 `inject_pipeline.cc:2824-3044`。

layout 也附在真实 SBlock/Buffer 上，copy/gemm 是真实 tile-op call。annotations 是程序 IR 上的跨 pass 状态，不是与程序平行的一份日志。

### Weft

Weft 八个 pass 的共同输入输出是：

```text
未改变的 canonical Kernel IR
+
被更新的 problem side record
→
未改变的 canonical Kernel IR
+
下一 stage 的 problem side record
```

`ConstructRISCVProblems` 首先把 canonical SSA 序列化为 value/op dictionaries：

- value：type、shape、axes、logical SEW、encoding、domain facts；
- op：name、operand/result string ids、source attributes、level/control path；
- Level/for/if/while carry：压成 union-find `handoff_class`。

见 `lib/Target/RISCVConstructProblems.cpp:54-186,189-267`。

后续 pass 不再重写 canonical op/type/region，而是继续给这些 dictionaries 加 key。

### 直接后果

1. MLIR 的 SSA use list、dominance、region branch、type conversion、pattern rewrite、canonicalization 不能直接作用于 physical decisions；pass 自己维护 string id 和 flat arrays。
2. canonical Level 的三个 region、births/handoff terminator 和 typed block arguments，在 planning side record 中被压成 `level_path/control_path/handoff_class`。原结构仍存在于旁边的 canonical program，但 planning schema 本身不携带它。
3. 后续 pass 需要更精确结构时，只能扩充 dictionary、重新查询 canonical program，或把工作留给 emitter。
4. pass 的 stage verifier 只能验证 side record 的外壳，无法证明 side record 与 canonical program 逐 use 一致。

## 1.4 Pass 之间如何传递，怎样观察

### Triton

pass 之间通过改写后的 typed IR 传递。Triton Python compiler 可以保存每个 stage module；`RemoveLayoutConversions` 自身在 propagation、rematerialization、hoist、cleanup 后有 IR dump，见：

- `ref/triton/python/triton/compiler/compiler.py:326-344`；
- `RemoveLayoutConversions.cpp:1617-1695`。

观察对象是 tensor type encoding、真实 conversion op、重写后的 producer/control IR。

### Weft

Weft 用 `problem.stage` 字符串约束 pass 顺序：

```text
facts
→ representations
→ conversions
→ storage-mappings
→ operations
→ schedule
→ resources
→ assignment
```

公共工具只输出：

```text
--emit=kernel-ir
--emit=physical-assignment
--emit=intrinsic-c
```

见 `tools/weft-compile/weft-compile.cpp:24-49,105-180`。`physical-assignment` 是手写的 dictionary pretty-printer，见 `lib/Target/RISCVCompiler.cpp:30-112`；当前没有每个 pass 的 physical program stage 输出。

即便借助通用 MLIR instrumentation 打印 module，能看到的也只是 `problem` 的 ArrayAttr/DictionaryAttr 变化；canonical program 的 op、type、region 和 use-def 不变。

### 直接后果

当前判断一个 pass “做了什么”只能比较 side record key 或最终 C，不能直接比较：

- 哪个 physical op 被替换；
- 哪条 conversion 被插入/删除；
- 哪个 producer 被 rematerialize；
- 哪个 loop body 已经变成 pipeline；
- 哪个 spill/reload 已进入程序。

这正是此前必须依赖 assignment dump、生成 C 和性能反推 pass 是否真实工作的结构原因。

---

# 二、八个 RISC-V pass 逐项事实

唯一 pipeline 顺序在 `lib/Target/RISCVCompiler.cpp:15-27`。

## 2.1 ConstructRISCVProblems

文件：`lib/Target/RISCVConstructProblems.cpp:454-546`。

读取 canonical kernel、typed SSA、Level/control、target 和 auto bindings；在 module 尾部新建一个或多个 `riscv.problem`，stage=`facts`。

程序改写：**否。** canonical op/type/region 不变。

写入记录：value/op snapshot、string use-def、level/control path、handoff class、target、candidate。

## 2.2 AssignRISCVRepresentations

文件：`lib/Target/RISCVConstrainRepresentations.cpp:124-145,1555-1621`。

读取 facts dictionaries 和 canonical kernel；给 value records 写 physical kind、SEW、LMUL、vl、lane/register axes、part counts、materialization 等。

程序改写：**否。** 没有改变 value type，没有插 conversion。

## 2.3 ResolveRISCVLayoutConversions

文件：`lib/Target/RISCVResolveLayoutConversions.cpp:218-320`。

读取 representations records；为一部分 operation operand 写 `use_conversions`。

程序改写：**否。** 没有 conversion op/result/use。

## 2.4 PropagateRISCVStorageMappings

文件：`lib/Target/RISCVPropagateStorageMappings.cpp:147-173,217-454`。

读取 canonical encoding/derive/pack/materialize，加上 handoff/use-def records；给 value/op records 写 base family、interleave rows、layout identity、pack axis 等。

程序改写：**否。** 没有 physical memory type、buffer、address op 或 pack op rewrite。

## 2.5 SelectRISCVLocalOperations

文件：`lib/Target/RISCVConstrainInstructions.cpp:313-339,1648-1685`。

读取 storage-mapping records、canonical primitive/encoding 和 target facts；给 operation records 写 realization、memory_edge、local_operation、co-reduce partner。

程序改写：**否。** generic canonical op 仍存在，没有被替换成 RVV/IME/local physical op。

## 2.6 ScheduleRISCVLevels

文件：`lib/Target/RISCVScheduleLevels.cpp:194-217,556-739`。

读取 Level/control path、flat use-def、memory roots、candidate unroll/pipeline；给 Level/loop/member operation records 写 level mapping、local cluster、producer/consumer/frontier、schedule。

程序改写：**否。** loop body 没有改变，没有 versioned buffer，没有 prologue/steady/epilogue op。

实际 pipeline transform 位于 emitter 的 `compilePipelinedFor`，见 `RISCVIntrinsicC.cpp:1898-2074`。

## 2.7 CheckRISCVResources

文件：`lib/Target/RISCVConstrainResources.cpp:21-43,89-177,261-425`。

读取 flat operation order、value records、temporary/fragment budget；写 live range、peak、resource class、reload/rematerialize 或 invalid。

程序改写：**否。** `reload-per-use` 和 `rematerialize-per-register-part` 只是 value record 字符串，没有 load/recompute op 插入程序。

## 2.8 SelectRISCVWinner

文件：`lib/Target/RISCVSolve.cpp:19-118`。

读取所有 resource-stage problem，按 cost 选择；复制 winner dictionaries 为 `riscv.assignment`，删除所有 `problem`。

程序改写：**否。** canonical kernel 仍未改变；assignment 仍是 side record。

## 2.9 逐 pass 事实的总结果

八个 pass 中：

```text
改写 canonical/physical SSA 程序的 pass：0
只建立或更新 side record 的 pass：8
```

`PassManager::enableVerifier(true)` 会在每个 pass 后验证 module，但 `ProblemOp::verify()` 只检查 stage、candidate 与 target 的少量 key。它不能把“每 pass 后 verifier 成功”解释为 physical program 合法。

---

# 三、Emitter 当前补出的物理信息

## 3.1 必须先区分两类工作

最终 emitter 合理保留的工作：

- 把已经确定的 physical scalar/vector/fragment type 拼成 C type 名；
- 把已经确定的 local op 拼成 intrinsic 或 inline asm；
- 生成函数声明、header、普通 C 语法；
- 实现 canonical encoding 已经唯一规定的 bit extraction 算术。

这些工作即使有真正 physical IR 也仍然存在。

当前超出机械拼写、由 emitter 补出的工作如下。

## 3.2 Physical value 缺字段的默认与重建

`RISCVIntrinsicC.cpp:787-836,1083-1117`：

- part 数缺失取 1；
- register axes 缺失回退旧 `register_axis`；
- register extents 缺失按 part count 合成；
- physical lanes/lane axis 缺失取 1/0；
- vector part count 缺失重新计算。

这是 coverage 漏洞的静默补值，不是 spelling。

## 3.3 未被 conversion pass 覆盖的 value handoff

`projectPart/projectRegisterPart/projectBinding/assignBinding` 位于 `RISCVIntrinsicC.cpp:859-1080`。它们从 source/result logical axes、register extents、lane axis 和 stream counts 重新计算物理 part 对应。

这些 helper 被 Level carry、for/if/while carry、pipeline frontier、new initializer、handoff/result 等没有 `use_conversions` 的路径调用。此时 emitter 是 conversion producer；失败时才报告需要 explicit conversion。

## 3.4 Kernel memory ABI 与地址模型

`initializeKernelArguments()` 在 `RISCVIntrinsicC.cpp:570-600` 自行：

- 从 canonical View shape/axis 建 extents；
- 假定 dense memory row-major；
- 计算 strides；
- 设 origins=0；
- 从 commit 扫描结果推断 const/writable argument。

`denseAddress()` 在 `:1222-1274` 再解释 selector、origin、stride 与 local offset，生成线性地址。

这些不是 pass-selected memory object/type 的机械打印，因为 assignment 没有完整 physical memory type/ABI object。

## 3.5 Level 与普通控制的物理程序生成

`compileLevel()` 在 `RISCVIntrinsicC.cpp:1638-1803` 从 canonical Domain 和 `level_mapping` 生成：

- 实际 C loop；
- physical step；
- active/tail 表达式；
- axis origin/base；
- body argument/carry binding。

`compilePipelinedFor()` 在 `:1898-2074` 真正创建 current/next banks、prologue、steady loop、epilogue 与 rotation。

Schedule pass 只选择 cluster/depth；物理 program rewrite 实际发生在 emitter。

## 3.6 Encoding 与 storage 的第二解释

`recordForSlice()` 在 `RISCVIntrinsicC.cpp:2218-2357` 使用一条 fallback chain 决定 source encoding family，并自行计算 record bytes、record stride 和 derived-interleave pointer。

`compileAdmit()` 在 `:2386-2398` 根据 ephemeral/base-packed/runtime facts 决定保留 slice 还是 materialize record。

`singleStorageFragment()` 在 `:89-120` 自行解释 natural/grouped/layered；`emitInterleavedField()` 在 `:4221-4424` 自行解释 joined、bit ranges、mask、sign extension 和 unpack。

canonical encoding 的 bit mapping 属于源语义，最终 bit 算术可以留在 lowering；但“这次 use 采用什么 physical memory edge、何时 materialize、何种 source family”没有完全由 physical op/type钉死。

## 3.7 Local pack

`compileMaterialize()` 在 `RISCVIntrinsicC.cpp:5467-5649` 从 canonical result axes、active scope 和部分 local-operation records重新构造 pack loop、临时 buffer 与地址。

源码 `:5620-5628` 仍有一个默认 destination 分支，但前面的 `:5500-5509` 已强制 rank=2 且 `packAxis` 必须属于两个 axes，因此该分支在当前合法输入上不可达。它是冗余代码，不是本报告据以认定第二决策的运行路径；真正的结构事实是 pack loop/buffer 本身尚未在 physical program 中出现，而由 emitter首次构造。

## 3.8 Reduce

`compileReduce()` 在 `RISCVIntrinsicC.cpp:3169-3523` 自行选择：

- register-axis reduction 坐标；
- seed（0、正负无穷）；
- integer/floating add/max/min intrinsic；
- widening reduction instruction；
- result part projection。

assignment 有 eliminated axis 和 representation，但没有一个已经选定的 typed physical reduce op。

## 3.9 Pointwise 与 broadcast

`compileBinary()` 在 `RISCVIntrinsicC.cpp:3653-4045` 根据 Binding kind、commutativity、scalar-on-left 和 conversion relation 选择：

```text
scalar C op
RVV vv
RVV vx/vf
splat
operand swap/projection
```

其中 token 到 intrinsic family 的映射也在 emitter。intrinsic 拼写可保留，但 `vv/vx/vf` 与 broadcast/swap 形态本应是 selected physical op 的一部分。

## 3.10 Contract

`compileContract()` 在 `RISCVIntrinsicC.cpp:5652-5893`：

- 把 `over.front()` 当作唯一 reduction axis；
- 根据 `lane_operand` 区分 lane/repeated operand；
- 固定当前 f32 result/memory 支持；
- 自行生成 K loop、load、broadcast、FMA 与 load cache。

这不是一个 selected contract op 的机械打印；完整 local microkernel 结构仍由 emitter构造。

## 3.11 IME

matrix branch 在 `RISCVIntrinsicC.cpp:5692-5761` 只接受固定：

```text
spacemit-ime1-i4i8-mma
M1 × N16 × K32
```

module emission 再扫描 realization prefix，决定是否输出固定 helper 与 inline asm，见 `:6117-6177`。

fragment shape、helper ABI 和部分 legality 因而仍在 emitter 重复出现。

## 3.12 Emitter 第二决策的完整类别

按事实归并，当前第二决策类别是：

1. physical mapping 缺字段时的默认值；
2. 未覆盖 use edge 的 part/layout projection；
3. kernel memory ABI、row-major stride 与地址线性化；
4. Level/control 的实际 physical loop/tail；
5. pipeline 的实际 program transformation；
6. encoding source/storage materialization 路径；
7. local pack loop与临时 buffer 的实际构造；
8. reduce algorithm/intrinsic family；
9. pointwise vv/vx/vf、broadcast 与 operand swap；
10. contract reduction axis、K loop、load cache与FMA组织；
11. IME fragment/helper shape；
12. reload/rematerialize record 对应的实际 C 重建。

这份清单不包含正常的 C 标识符生成、RVV C API 名字拼写、函数声明和已完全选定 asm 文本。

---

# 四、Level、iota、conversion、storage mapping、cluster 在哪里

## 4.1 Level

Level 是真实 canonical op：

```text
weft_kernel.level
  domain
  carried operands/results
  state_names / staged_names
  state_births region
  staged_births region
  body region
```

`births_yield` 与 `handoff` 也是真实 canonical terminator。定义见 `include/Weft/Dialect/Kernel/IR/KernelOps.td:164-190`，region/block-argument/type verifier 见 `lib/Dialect/Kernel/IR/KernelDialect.cpp:495-543`。

但 physical Level mapping、unroll、pipeline、cluster 不在 Level op 上；它们在 assignment operation dictionary。

## 4.2 iota

`weft_kernel.iota` 是真实 canonical op，定义见 `KernelOps.td:227-231`，verifier 见 `KernelDialect.cpp:628-637`。

它只建立逻辑 shaped axis；`register.iota/rvv.iota` realization 位于 assignment record，真实 tuple/`vid` C 代码由 emitter产生。

## 4.3 Conversion

canonical dialect 没有 conversion op，RISC-V planning dialect也没有 conversion op。

conversion 只存在于：

```text
riscv.problem.operations[i].use_conversions
```

winner 后复制到 `riscv.assignment.operations[i]`。

## 4.4 Storage mapping

两层需要区分：

- logical storage semantics 是 canonical：`EncodingType`、`encoding_decl`、`pack`、`interleave`；
- target physical storage mapping 是 side record：base family、interleave rows、record bits、physical layout identity、pack/record axes。

没有 physical memory type、physical buffer SSA value或 target load/store op。

## 4.5 Local cluster

local cluster 不是 canonical op/type，也不是 planning dialect op。它只是某个 loop operation record 的：

```text
local_cluster = {
  producer_ops,
  consumer_ops,
  frontier_values,
  buffer_groups,
  pipeline_depth
}
```

定义与写入位于 `lib/Target/RISCVScheduleLevels.cpp:149-170,556-638`。实际 cluster loop rewrite 位于 emitter。

## 4.6 Canonical IR 承载与未承载的事实

canonical Kernel IR真实承载：

- encoding family、logical storage layout、derived family/instance identity；
- View/Slice/Value 的逻辑 shape 与 axis ids；
- Domain 的 relation、extent、partition、multiplicity、tail；
- Level 的三个 region、births、carried state、handoff；
- 普通 for/if/while 的 ordered control 与 carry；
- new/materialize/admit/commit 的逻辑生命周期边界；
- explicit engine role；
- local numerical op 的 typed SSA use-def。

canonical Kernel IR按设计不承载：

- physical layout；
- SEW/LMUL/vl；
- lane/register/fragment mapping；
- physical memory form；
- selected local instruction；
- conversion；
- unroll/pipeline/prefetch；
- live interval/resource/spill。

这些“不在 canonical IR”本身不是错误；问题是它们没有进入另一份 typed physical program。

## 4.7 canonical → planning 时的结构压缩

`KernelFactCollector` 保留 logical type、shape、axes、encoding、domain facts和 source op attributes，但把：

- SSA value/op 转成字符串 id；
- nested region 转成 flat operation array + `level_path/control_path`；
- Level/for/if/while carry 与 births/handoff 转成 union-find `handoff_class`；
- lifetime 转成后续按 flat operation ordinal 计算的整数区间。

原 canonical program 并没有被删除，所以不是“算法语义从 module 中丢失”。准确后果是：

> physical planning record 丢失了 typed region/SSA structural identity；原 canonical program仍在旁边，导致后续 pass或emitter需要在两份表示之间来回查找。

这形成了当前的双来源：canonical program 提供结构，assignment提供部分物理决定。

---

# 五、是否需要真正 physical IR

## 5.1 判断

如果目标仍是：

```text
每项 physical decision 有唯一 producer
conversion 覆盖全部 use edge并可插入/消除
schedule pass 真实生成 pipeline program
resource pass真实插入reload/rematerialize/spill
emitter只做intrinsic C / local asm拼写
每个pass后可以验证和dump程序变化
```

那么当前 side-record 结构不能直接满足。需要一份真正的 typed physical program IR。

这不是因为 Triton 是 GPU、Weft 是 CPU；它来自两者共同使用 SSA compiler 的机制要求。TileLang 也证明，不一定所有决定都必须成为 type——For/SBlock annotation 可以是合法跨 pass state——但它们必须附着在真实程序实体上，并被后续 pass改写为真实 program structure。

## 5.2 最小正确结构

保留 canonical Kernel IR，不把 LMUL、lane、fragment 写回语言 authority。每个完整 std/auto candidate 克隆并转换为一份瞬态 target physical program：

```text
canonical Kernel IR
→ Kernel-to-physical conversion
→ typed physical value/layout propagation
→ insert physical convert ops
→ lower encoded storage accesses / physical memory ops
→ rewrite generic local ops to selected RVV/IME ops
→ rewrite Level/control to physical loops and schedules
→ insert pipeline buffers / prologue / steady / epilogue
→ insert reload/rematerialize/spill ops or reject candidate
→ verified/dumpable physical program
→ intrinsic C / local asm
```

“physical dialect”可以是新增 dialect，也可以把当前 `weft_riscv` dialect彻底改造成上述程序 dialect。机制上必须至少有：

### Physical types/attributes

- scalar/vector/register-tuple/fragment value representation；
- logical-axis → sequential/lane/register/fragment layout；
- physical SEW/LMUL/vl；
- physical memory view、stride、alignment、packed-record layout；
- target/ABI identity。

### Physical ops

- function/kernel boundary；
- `convert_layout`；
- physical load/store/gather/segment；
- pack/unpack/bit extraction；
- selected RVV arithmetic/reduce/contract；
- selected IME fragment operation；
- physical loop/control或合法使用 `scf`；
- pipeline/versioned buffer；
- reload/rematerialize/spill；
- ABI-visible builder/packed object boundary。

### Verifier

- operand/result layout compatibility；
- conversion source/result类型；
- memory object与access mapping一致；
- engine/target legality；
- loop/schedule nesting；
- fragment shape与资源；
- pin boundary不允许隐式conversion。

这里不要求把每一条机器指令变成独立 op，也不要求替代 LLVM 的寄存器分配。目标是把 Weft 自己负责的 C/D 组决定和结构性 program rewrite变成 IR；最终机器寄存器编号、指令调度和peephole仍交给系统编译器。

## 5.3 为什么“不新增 physical dialect，只把当前表补完整”不能解决核心问题

理论上可以继续给 `problem.values/operations` 增加字段、编写更强 verifier，并要求每条 use edge都有 conversion record。这能减少漏项，但仍然没有：

- conversion SSA result；
-改写后的loop/body；
-实际pipeline buffer；
-reload/spill operation；
-对 physical program 使用 MLIR pattern/canonicalization/dominance；
-pass 后可直接交给下一 lowering 的 program。

最终仍需要一个组件把 canonical program + 完整 side record 合成为 physical program。当前这个组件就是 emitter。若改写成自定义 C++ physical AST，它事实上就是另一份 physical IR，只是放弃了 MLIR 的 type、verifier、rewrite和dump能力。

因此“继续补表”与“新建自定义 AST”都不会消除 physical program层；只会推迟或重造它。

## 5.4 Canonical Kernel IR 是否保留

保留。

理由不是兼容历史，而是职责不同：

- canonical IR 是 spec 2.2 中作者树的 authority；
- physical IR 是一个 candidate 在一个 target 上的表示与执行程序；
- 同一 canonical tree 可产生多个 candidate physical modules；
- candidate失败或被淘汰不应修改作者树；
- physical IR可丢弃，不成为第二份持久算法 authority。

直接给 canonical `ValueType` 加 LMUL/lane encoding并持续改写，也可以在技术上工作，但会让 target-neutral canonical dialect同时承担 target-specific candidate状态，违反当前唯一持久 authority的边界。把 clone 转换到独立 physical dialect更清楚，也更接近 Triton TTIR → TTGIR 的机制。

## 5.5 工程量事实

当前规模：

| 部分 | 当前规模 | 与 side record 耦合 |
|---|---:|---|
| canonical `KernelOps.td` | 约 386 行 | 无，应该保留 |
| canonical verifier | `KernelDialect.cpp` 约 974 行 | 无，应该保留 |
| Python frontend | `frontend/compiler.py` 约 2430 行 | 主要生成 canonical IR，应该保留 |
| std 数值树 | `python/weft/std` 约 2900 行 | 属于作者树，应该保留 |
| 八个 planning pass | 约 7938 行 | 大量 dictionary schema读写 |
| intrinsic-C emitter | 约 6196 行 | 大量 assignment index/string dispatch与结构重建 |
| target profile | header 约 80 行 + cpp 约 233 行 | 低，可直接复用 |

新 physical dialect 的初始 type/op/verifier/TableGen 只是数百到约一千行量级；真正工作量不是建 dialect 文件，而是迁移后两项：

- 八个 pass 的分析算法可以参考，但 dictionary input/output接口大部分要重写为 SSA/type/op rewrite；
- emitter中真正的RVV intrinsic/IME asm拼写、encoding bit arithmetic可以保留，assignment索引、part projection、control/pipeline/reduce/contract结构重建需要移出或重写。

因此工作量是一次完整 backend结构重构，不是增加一个 conversion op 的局部补丁。当前代码给不出可靠人日估计；可确定的是需要触及数千行，且主要风险集中在 7938 行 planning与6196行emitter的边界重划。

## 5.6 可以直接搬与必须重写

### 可以保留或直接搬

- canonical Kernel dialect、Python frontend、std trees、examples；
- `RISCVTargetProfile` 的 ISA/ABI/VLEN/SEW/LMUL/IME capability；
- axis/free/reduction/use-def分析的算法思想；
- SEW/LMUL、resource与fragment legality公式；
- encoding grouped/layered/joined 的storage mapping数学；
- RVV C intrinsic命名/类型拼写；
- IME typed asm leaf；
- runner、tuner与外部数值/性能协议。

### 必须重写或大幅改造

- `ProblemOp/AssignmentOp` blob schema；
- canonical snapshot与string id fact collector；
- 六个主要 representation/conversion/storage/operation/schedule/resource pass的字段读写；
- winner从“复制assignment表”改为选择完整physical module/function；
- emitter的assignment索引、Binding投影与silent defaults；
- Level/control/pipeline的实际program generation；
- reduce/contract/pack的结构生成；
- reload/rematerialize/spill从字符串决定到真实op；
- physical type/op/parser/printer/verifier与pass dump入口。

## 5.7 仍无法从本轮事实唯一决定的设计点

以下问题当前源码和参考实现不能替项目作出唯一选择：

1. physical layout 是采用一个通用线性映射属性，还是 RVV lane/register/fragment 的小型结构化属性集合；
2. canonical Level 转换后保留一个 `physical.level` 直到 scheduling，还是在 physical IR入口立即变成 `scf.for` + typed state；
3. 多个 candidate 是多个 module、多个 function，还是同 module 的候选 region；
4. selected RVV op保留到多高层，何时降低成接近 intrinsic的一对一 op；
5. encoding bit mapping在 physical IR中是显式逐步op，还是一个 typed packed-access op到末端再展开。

这些是下一次设计必须明确的选择；本轮事实只能确定：无论具体选择哪一种，都必须让 conversion、memory、loop、pipeline、spill与local op成为程序中的typed实体，而不是继续只存在于旁路assignment。

---

# 六、当前文档与源码的事实冲突

`doc/compiler/riscv-planning.md` 已落后于源码，不能作为当前实现事实：

- `:31-34,62-66` 写“value-use convert尚未实现”，但当前已有 `ResolveRISCVLayoutConversionsPass`，只是record而非op；
- `:46-49,68-78` 写“没有spill realization”，但当前resource pass已有 admitted reload和pure computed rematerialization record；
- `:55-57,80-92` 写emitter只机械生成，但源码存在本报告第三部分列出的结构重建与默认；
- `:51-53` 写真机winner未接入，当前runner tuner可在一次调用中实测并重跑winner，但compiler内部`SelectRISCVWinner`仍只按静态resource cost复制assignment。

这不改变本轮判断，因为唯一规范仍是 `report/weft-spec.md`，实现事实来自源码。它说明当前 doc 对side-record结构的描述本身也不足以观察真实边界。

## 最终事实判定

当前主链准确表示为：

```text
canonical Kernel SSA program
        +
八个pass逐步填充的physical assignment side record
        ↓
一个同时读取两者、并重建physical program的intrinsic-C emitter
```

而不是：

```text
canonical Kernel IR
→ typed physical IR
→ intrinsic-C lowering
```

前者能够生成并运行代码，但它结构性地允许 conversion漏项、side record与program不一致、pass只改记录而不改程序，以及emitter成为第二个physical decision producer。若目标是spec 5.2与5.6写下的真正pass/compiler边界，建立一份瞬态、typed、可验证、可逐pass dump的physical program IR是必要重构，不是可选的代码整理。
