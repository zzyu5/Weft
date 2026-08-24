# 两层 MLIR 编译主干

## 1. 只有两层程序 IR

Weft 的编译主干只有两层 MLIR：

```text
Weft Kernel IR
    作者程序：logical Value、axis、Level、Encoding、operation、effect
        ↓  ConvertWeftToRISCV
Weft RISC-V IR
    target-aware physical program：layout、conversion、memory、schedule、RVV/IME
        ↓  TranslateToIntrinsicC
intrinsic C / local asm
        ↓
system C compiler
```

第一层对应 Triton 的 TTIR，第二层对应 TTGIR。Weft 不再在两者之间增加一层通用 Physical IR，也不在 RISC-V IR 之后增加 LLVM dialect；terminal translation 直接输出 intrinsic C。

一份 RISC-V IR module 可以同时使用 `func`、`scf`、`arith` 和 `weft_riscv` operation。多个 dialect namespace 不等于多个 IR 层。layout propagation、memory planning、pipeline、resource handling 和 target-op lowering 是对同一份 RISC-V IR 的 pass，不是新的程序表示层。

非 SIMT 物理抽象机器定义第二层 IR 能表达什么，不是第三层 IR。target profile、build config、候选枚举和 tuner 是编译输入或 driver 行为，也不是 IR 层。

## 2. 两层各自承载什么

### 2.1 Weft Kernel IR

Kernel IR 是唯一持久的作者程序。它承载：

- typed logical SSA；
- logical shape 与 axis identity；
- Level、birth、staged lifetime 与 handoff；
- ordinary `for/while/if`；
- pinned View、Encoding、effects 与 ABI identity；
- numerical operations 与其精度、顺序和有限位宽语义。

它不承载 target layout、LMUL、local pack、RVV/IME 选择、pipeline、spill 或 intrinsic spelling。

### 2.2 Weft RISC-V IR

RISC-V IR 是一份瞬态、target-aware、可执行结构已经逐步显式化的 physical program。它承载：

- logical coordinate 到 time、lane、register replica、fragment 和 local storage 的 mapping；
- 每个 physical value 的 element width、LMUL、`vl`、tail 与 carrier；
- typed layout conversion；
- physical memory descriptor、访问形式与 encoded access；
- local object、local pack、spill、reload 与 rematerialized producer；
- selected scalar、RVV、IME 和 transfer operations；
- source Level 在 target 内的 strip、unroll、cluster、pipeline 与 buffer version；
- 完整 ABI 地址关系与 resource legality。

RISC-V IR 不是第二棵算法树。每个 operation 和 value 必须保留 source origin；target pass 不能增加 canonical Value、Level、effect、persistent artifact 或算法阶段。

## 3. RISC-V IR 的核心表示

### 3.1 Physical value

shaped physical value 使用带 target layout 的类型：

```text
tensor<logical-shape x element,
       #weft_riscv.layout<
           axes = logical-axis identities,
           map = logical coordinates -> time/lane/replica/fragment,
           carrier = scalar|rvv|ime,
           validity = mask/tail relation>>
```

layout 必须保留所有未被 numerical operation 消去的 logical axes。reduce `[M,K] -> [M]` 后，M 轴仍在 result type；register replica 是 M 的物理映射，不是新的 canonical Value。

### 3.2 Memory descriptor

physical memory 使用 typed descriptor：

```text
!weft_riscv.memdesc<
    logical shape,
    Encoding/storage mapping,
    extents/strides/origin/alignment,
    address class,
    alias/effect/lifetime>
```

kernel 参数 descriptor 完整保存 pinned Encoding 与 ABI。动态 extent、stride 和 origin 是显式 SSA operand或 function argument。local descriptor 只能表示 invocation-local object，不能越过 pin boundary 或替代 caller-visible workspace。

### 3.3 Explicit conversion

representation 冲突必须成为真实 operation：

```mlir
%b = weft_riscv.convert_layout %a
    : tensor<..., #layout_a> -> tensor<..., #layout_b>
```

它保持 logical shape、axes、Level identity 和 numerical value，只改变 target representation。conversion 有 SSA result、verifier 与 rewrite pattern；多个 consumers 可以共享同一个 conversion result。消除 conversion 意味着重写或删除这个 operation，不是修改旁路表。

final RISC-V IR 中，通用 conversion 必须已经降低为确定的 RVV slide/gather/splat、tuple split/merge、register/local transfer 或 RVV/IME handoff。

### 3.4 Memory、local object 与 target op

RISC-V IR 至少需要真实 operation 表示：

```text
load / store
encoded_load / encoded_extract
local_alloc / local_load / local_store / local_pack
spill / reload
convert_layout
RVV operations
IME operations
```

每个 selected target operation 必须声明 typed operands/results、layout constraints、dtype/shape/mask/tail legality、effects、resources 和最终 intrinsic/asm contract。不能只保存一个 `realization = "..."` 字符串，再由 emitter 重建 microkernel。

允许 pass 暂时保留已选 microtile/unroll/layout 的 composite RISC-V op；在 terminal translation 前，它必须展开为明确的 `scf`、memory 与 primitive target ops。只有一对一 intrinsic 或拥有完整局部 ABI 的 opaque asm leaf 可以保留到最后。

### 3.5 Level、ordinary control 与 schedule

canonical Level 转入 RISC-V IR 后保留一一对应的 origin、domain、partition、multiplicity、births、carried values 与 handoff。ordinary control 保持有序标量语义，普通 scalar loop不会因为进入 RISC-V IR 而获得 shaped axis 或 lane mapping。

pipeline pass 必须改写真实 region/loop，生成：

- strip 与 tail control；
- prologue、steady state 与 epilogue；
- versioned local buffers；
- load/compute ordering 与 wait/barrier；
- 明确的 loop-carried physical values。

`pipeline_depth = 2` 之类的 attribute 只有在后续 pass 物化出上述结构时才有意义。最终 emitter 不生成 pipeline。

## 4. 一份 candidate 对应一份 RISC-V IR module

driver 先绑定一份 std/source candidate 与 source `auto`，再为有限 physical parameter bindings 分别建立 RISC-V IR module：

```text
一个 canonical candidate
× 一个 target profile
× 一组 physical parameters
× 一个按固定优先级选定的 structural realization
```

参数候选之间不共享一张全局 `assignment`。每个 module 独立走完整 pass、verification 和编译；tuner 比较的是可执行 artifacts。

结构性选择不是 tuner 的搜索维度。target 按固定规则尝试最高优先级的合法结构；若完整 legality/resource check 证明其非法，driver 从原 canonical candidate 重新建立下一优先级的 RISC-V IR。一个 module 内不保存备用结构。

## 5. 同一份 RISC-V IR 上的 Pass

### 5.1 `ConvertWeftToRISCV`

把 canonical function boundary、View/Encoding、Value、Level 与 ordinary control 转成 target-aware RISC-V IR：

- 建立完整 typed ABI descriptor；
- 给每个 physical entity附着 source origin；
- 保留 canonical numerical op 的语义边界；
- 不创建 source 没有的 logical Value、Level、effect 或 artifact。

### 5.2 `SelectRISCVOperations`

根据 operation semantics、typed operands、axes、Encoding、target profile、build requirements 和固定结构优先级，选择 scalar/RVV/IME target operation。selected target op 是 layout constraint anchor，不是 whole-kernel implementation selector。

### 5.3 `PropagateRISCVLayouts`

从 pinned memory 与 target-op anchors 沿 use-def 传播完整 layout。producer/consumer要求不一致时，在具体 use edge 插入 `convert_layout`。pass 结束后不存在“缺字段就用 scalar、part=1 或第一种能装下的 LMUL”的默认表示。

### 5.4 `PlanRISCVMemory`

根据 typed descriptor、Encoding mapping、layout、all consumers、reuse 和 target memory capability，将访问改写为 unit/strided/indexed/segment、encoded access 或 local pack，并把地址关系显式写入 RISC-V IR。

### 5.5 `CanonicalizeRISCVLayouts`

执行 conversion propagation、backward rematerialization、conversion hoist/sink、CSE/DCE 和冗余 conversion 消除。pass 的效果通过 value type 和 `convert_layout` operation 的变化直接观察。

### 5.6 `PipelineRISCVLevels`

在作者已有 Level 或 ordinary loop 内，根据真实 SSA use-def、alias、effect、state carry 与 validity 建立局部 cluster，并把固定的 unroll/pipeline/buffer/prefetch 参数物化成真实 loop 与 buffer structure。它不能创建 source 未写的 outer traversal、staging、workspace 或 persistent packing。

### 5.7 `MaterializeRISCVResources`

基于最终 physical SSA live intervals计算 register、fragment、temporary 和 local-storage 使用。若 target contract允许，可以插入真实 spill/reload 或克隆 pure producer作 rematerialization；否则拒绝当前 module。Emitter 不参与资源补救。

### 5.8 `LowerRISCVComposites`

把 encoded access、local pack、composite reduce/contract、layout conversion 与 schedule helper 降为明确的 control、memory 和 primitive RVV/IME ops。此后不能再存在需要 emitter生成的局部循环、microkernel、pack loop 或 buffer rotation。

### 5.9 `VerifyFinalRISCV`

final verifier 至少要求：

- 所有 shaped values 都有完整 layout；
- 每条不兼容 use edge 都有 typed conversion；
- memory descriptor/access 与 pinned Encoding 一致；
- source axes、Level、control、effects 与 handoff identity 保持；
- target op、fragment、mask/tail 和 resources 合法；
- 不存在未展开的 composite/schedule op；
- build requirements在完整 RISC-V program 上成立。

## 6. Terminal translation

terminal translator 只接收通过 `VerifyFinalRISCV` 的 module。它负责：

- 把 `func/scf/cf/arith` 写成普通 C；
- 把已选 RVV op 写成确定 intrinsic；
- 把已选 IME/local leaf 写成确定 intrinsic 或 typed asm；
- 打印已经物化的 ABI、pointer arithmetic、header 和声明。

它不能推导 layout、选择 memory form/engine/fragment、生成 Level loop/pack loop/pipeline，不能同时读取 canonical module与一张 assignment table合成机器程序。缺失信息是 final verifier错误。

## 7. 与 Triton 的准确对应

Triton 的结构不是“每个 pass 一层 IR”：

```text
TTIR
    ↓ TritonToTritonGPU conversion
TTGIR
    ├─ Coalesce
    ├─ AccelerateMatmul
    ├─ RemoveLayoutConversions
    └─ Pipeline
    ↓ TritonGPU/target ops to LLVM
LLVM IR
```

TTGIR module中可以同时出现 Triton、TritonGPU、NvidiaGPU和标准MLIR operation，但它们共同构成一个target-aware程序层。layout位于tensor type encoding，`ttg.convert_layout`是真实op，coalescing、matmul acceleration、conversion elimination和pipeline都重写同一份程序。

对应机制可以直接在参考实现中定位：

- TTIR→TTGIR conversion：`ref/triton/lib/Conversion/TritonToTritonGPU/TritonToTritonGPUPass.cpp:678`；
- tensor type encoding建立：`ref/triton/lib/Conversion/TritonToTritonGPU/TritonGPUConversion.cpp:19`；
- typed `ttg.convert_layout`：`ref/triton/include/triton/Dialect/TritonGPU/IR/TritonGPUOps.td:27`；
- memory coalescing改写：`ref/triton/lib/Dialect/TritonGPU/Transforms/Coalesce.cpp:71`；
- dot→MMA layout/op改写：`ref/triton/lib/Dialect/TritonGPU/Transforms/AccelerateMatmul.cpp:441`；
- conversion propagation/rematerialization：`ref/triton/lib/Dialect/TritonGPU/Transforms/RemoveLayoutConversions.cpp:42`；
- loop-carried SSA与prologue/epilogue生成：`ref/triton/lib/Dialect/TritonGPU/Transforms/Pipeliner/PipelineExpander.cpp:51`。

Weft采用相同的层次原则：

```text
Kernel IR
    ↓
RISC-V IR
    ├─ layout/memory/operation passes
    ├─ conversion canonicalization
    ├─ pipeline/resource passes
    └─ RVV/IME lowering
    ↓
intrinsic C
```

差异只在第二层机器语义和最终出口：TritonGPU表示thread/warp/CTA/value ownership并降到LLVM；Weft RISC-V IR表示time/lane/register replica/fragment/local storage并直接翻译到intrinsic C。这个差异不构成额外一层。

TileLang也没有把每个规划步骤变成一层IR。layout inference把结果附着到真实block/loop，`LowerTileOp`再重写buffer与index（`ref/tilelang/src/transform/layout_inference/layout_inference.cc:1196`、`ref/tilelang/src/transform/lower_tile_op.cc:1080`）；software-pipeline injection会真实生成buffer version、barrier和重写后的body（`ref/tilelang/src/transform/inject_pipeline.cc:3608`）。Weft采用的是同一条机械原则：跨pass结果必须落在真实程序实体上，而不是要求复制TileLang的SIMT storage/thread模型。

## 8. 明确排除的结构

Weft 不采用：

- `weft_phys -> weft_riscv` 两级physical lowering；
- `weft_riscv.problem` / `weft_riscv.assignment` decision dictionaries；
- canonical program加side records，再由emitter首次构造physical program；
- 把target profile、build config、tuner或pass stage当成IR层；
- 同一语义并存的新旧physical路径；
- final emitter中的默认表示、selector或source-closure reconstruction。

未来增加另一类target时，它实现同一非SIMT物理机器合同并拥有自己的target-aware physical IR；它替换第二层，不插入第三层，也不要求另一份source DSL。
