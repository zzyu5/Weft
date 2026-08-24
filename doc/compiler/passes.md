# RISC-V 编译 Pass

所有 pass 改写同一份 [RISC-V IR](riscv-ir.md)，不是各自产生新的 IR 层。

## 1. Pass Pipeline

### `ConvertWeftToRISCV`

建立 target-aware types、完整 ABI descriptor、source origin 及一一对应的 Level/control。保留 canonical numerical op 边界，不创建 source 没有的 logical Value、Level、effect 或 artifact。

### `SelectRISCVOperations`

根据 operation semantics、typed operands、axes、Encoding、target profile、build requirements 和固定优先级，选择 scalar/RVV/IME target-op family。selected op 是 layout constraint anchor，不是 whole-kernel implementation selector。

### `PropagateRISCVLayouts`

从 pinned memory 和 target-op anchors 沿 use-def 传播完整 layout，在不兼容 use edge 插入 `convert_layout`。缺失 layout 使当前 module 失败，不能默认 scalar、`part=1` 或第一种能装下的 LMUL。

### `PlanRISCVMemory`

根据 descriptor、Encoding mapping、layout、all consumers、reuse 和 target capability，物化 unit/strided/indexed/segment、encoded access 与 invocation-local pack。

### `CanonicalizeRISCVLayouts`

执行 conversion propagation、backward rematerialization、hoist/sink、CSE/DCE 和冗余 conversion 消除。效果通过 value type 和真实 operations 的变化观察。

### `PipelineRISCVLevels`

在作者已有 Level/loop 内，根据 SSA use-def、alias、effect、state carry 与 validity 形成 local cluster，并把固定 unroll/pipeline/buffer/prefetch 参数物化成真实 loop 与 buffer structure。

### `MaterializeRISCVResources`

基于 physical SSA live intervals 计算 register、fragment、temporary 与 local-storage 使用；插入合同允许的 spill/reload 或 pure-producer rematerialization，否则拒绝 module。

### `LowerRISCVComposites`

把 encoded access、local pack、composite reduce/contract、conversion 与 schedule helper 降为明确 control、memory 和 primitive [RVV/IME leaf ops](leaves.md)。

### `VerifyFinalRISCV`

验证 [final RISC-V IR 不变量](riscv-ir.md#6-final-ir-不变量)和 build requirements；成功后才允许 [terminal emission](emission.md)。

## 2. Pass Contract

每个 pass 必须声明：

```text
允许出现哪些 op/type
读取哪些 program facts
产生、替换或消除哪些 entities
pass 后必须满足哪些不变量
失败返回 invalid 还是 unsupported
```

pass-local analysis map 可以存在；跨 pass 结果必须写回 type、operation、region 或真实 entity 的 typed attribute。stage 字符串和 value/op dictionaries 不能成为跨 pass authority。

## 3. 可观察性

每个 pass 后必须能够 dump 同一份 RISC-V module。dump 中应直接看见该 pass 实际造成的 type/layout 变化、`convert_layout` 插入或删除、memory/target-op 替换、physical loop 与 pipeline 展开，以及 spill/reload/rematerialization。只打印 analysis table、assignment dictionary 或 pass 名称不能证明程序已经被改写。

若某项跨 pass 决定只能从 side record 观察，说明它尚未进入 physical IR；若 final emitter 需要回查 Canonical Kernel IR 或补默认字段才能输出，说明前序 pass contract 未闭合。

## 4. 与 Triton/TileLang 的机制关系

Triton 的 Coalesce、AccelerateMatmul、RemoveLayoutConversions 和 Pipeline 都在 TTGIR 上重写同一份程序，不是四层 IR：

- `ref/triton/lib/Conversion/TritonToTritonGPU/TritonGPUConversion.cpp:19`：layout进入tensor type encoding；
- `ref/triton/include/triton/Dialect/TritonGPU/IR/TritonGPUOps.td:27`：typed `ttg.convert_layout`；
- `ref/triton/lib/Dialect/TritonGPU/Transforms/Coalesce.cpp:71`：memory op与conversion rewrite；
- `ref/triton/lib/Dialect/TritonGPU/Transforms/AccelerateMatmul.cpp:441`：MMA layout/op rewrite；
- `ref/triton/lib/Dialect/TritonGPU/Transforms/RemoveLayoutConversions.cpp:42`：propagation/rematerialization；
- `ref/triton/lib/Dialect/TritonGPU/Transforms/Pipeliner/PipelineExpander.cpp:51`：loop-carried SSA与pipeline expansion。

TileLang也把layout结果附着到真实block/loop，再由LowerTileOp和InjectSoftwarePipeline重写buffer、index、barrier与body；见`ref/tilelang/src/transform/layout_inference/layout_inference.cc:1196`、`lower_tile_op.cc:1080`、`inject_pipeline.cc:3608`。

Weft复用的是typed representation、explicit conversion和real rewrite，不复用SIMT thread/storage ownership。
