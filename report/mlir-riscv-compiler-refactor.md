# Weft MLIR RISC-V 编译主干重构报告

## 结论

本轮把新 Core-local command DSL 之后的 RISC-V 编译入口收缩为一条真正由 MLIR 管理的主干：

~~~text
Python DSL kernel
→ canonical weft_kernel / weft_ext MLIR
→ ModuleOp verification
→ one MLIR RISC-V artifact pass
→ per-KernelOp MLIR facts analysis
→ transient physical decisions
→ intrinsic C / primitive-local asm + public header
→ system C compiler
~~~

旧的 RISCVLowering public API、body/header 两个 public 编译入口和 tool 直接调用 lowering function
的路径已经删除。现在 public target API 只有
compileRISCVModule(ModuleOp, RISCVCompilerOptions)，一次成功调用同时返回 intrinsic C 与 header。
不支持的 target 明确失败，不存在标量、legacy、GGML 或 materials fallback。

这次完成的是唯一 MLIR 编译主干和 authority 替换，不是宣称 Weft 已经具备 Triton 的全部优化
成熟度。通用 loop pipeline、自动 compile-and-measure tuning 和更细的 emitter 内部拆分仍是
真实未完成项；它们不构成第二主干，也没有用兼容路径伪装已完成。

## 一、重构前后的实质变化

重构前的公开路径是：

~~~text
weft-compile
├─ lowerToRISCVIntrinsicC(ModuleOp, options, body output)
└─ emitRISCVHeader(ModuleOp, options, header output)
~~~

RISCVLowering.cpp 只是普通 C++ facade；RISCVKernelFacts 由 compiler 手动调用；一个大型
KernelCompiler 同时负责 ABI、facts、planning、selected implementation 收集与 C 发射。它虽然
操作真实 MLIR op，但 compiler invocation 没有进入 MLIR pass/analysis 生命周期。

重构后：

1. compileRISCVModule 构造 mlir::PassManager；
2. PassManager 只运行一个 OperationPass<ModuleOp>；
3. pass 通过 MLIR AnalysisManager 为每个 canonical KernelOp 获取
   KernelPhysicalFactsAnalysis；
4. lowering 先 prepare 全部 physical decisions，再 emit；
5. intrinsic prelude、kernel body 与 public header 只有全部成功后才组成一个 RISCVArtifact；
6. 任意 kernel lowering、leaf spelling 或 header projection 失败都会使整个 pass 失败，不会
   产生半成品或改走另一条路。

关键实现位置：

- include/Weft/Target/RISCVCompiler.h：唯一 public options/artifact/compiler API；
- lib/Target/RISCVCompiler.cpp：MLIR artifact pass 和 PassManager invocation；
- lib/Target/RISCVKernelFacts.{h,cpp}：MLIR analysis 所有的 typed facts；
- lib/Target/RISCVKernelCompiler.cpp：RISC-V decision preparation 与 artifact body projection；
- lib/Target/RISCVHeader.cpp：只在 lib/Target 内部可见的 header sink；
- tools/weft-compile/weft-compile.cpp：parse/verify 后只调用 compileRISCVModule。

## 二、对 Triton 的实际借鉴

本轮参考的是 /home/kingdom/phdworks/ref/triton 当前源码，而不是抽象的 “Triton 印象”。

Triton 的 BaseBackend 把 target options、dialect loading 和顺序 compilation stages 归属给一个
compatible backend；compiler.py 为一次 compile 创建 context/module，再顺序运行 backend
stages：

- python/triton/backends/compiler.py:23-72；
- python/triton/compiler/compiler.py:226-363。

Weft 借鉴的是一次 compile invocation 内唯一 backend、明确 stage 所有权和统一失败。因此
body/header 两个 public 入口被收缩到一个 MLIR artifact pass。

Triton 的 AxisInfoAnalysis 是 MLIR data-flow analysis，结果由当前 compiler invocation 消费，
不是第二份源程序 IR：

- include/triton/Analysis/AxisInfo.h:216-293。

Weft 对应地把 axis、use-def、memory relation 和 lifetime 事实提升为
KernelPhysicalFactsAnalysis，由 AnalysisManager 构造和缓存，不再由 reuse/planner/emitter
各自递归 SSA。

TritonGPU-to-LLVM 通过 TargetInfoBase 隔离 target capability 与 lowering pattern：

- include/triton/Conversion/TritonGPUToLLVM/TargetInfoBase.h:11-144。

Weft 保留同一责任方向：ISA/ABI/VLEN/LMUL legality/IME availability 来自单一
RISCVTargetProfile；op planning 不以 SG2044、K1 或 VLEN128/256 作为 whole-kernel route 身份。

没有复制 Triton 的 TTGIR layout、CTA/warp ownership、shared-memory allocation 或 GPU
tensor-core conversion。Weft 不增加 GPU physical dialect，也不把 lane/LMUL/fragment 写回
canonical Kernel IR。当前输出目标是 intrinsic C/asm，因此正确边界是：

~~~text
canonical typed MLIR ops / SSA / regions
+ PassManager and AnalysisManager
+ one-invocation transient target decisions
+ direct artifact translation
~~~

为形式相似而新造可持久化 RISC-V Physical dialect，反而会违反项目只有 canonical Kernel IR 与
最终 artifact 两个持久 authority 的设计。

## 三、对 TileLang 的实际借鉴

/home/kingdom/phdworks/ref/tilelang 当前 checkout 是 TVM TIR 编译器，不是 MLIR 编译器。本轮
没有伪称复用了 “TileLang MLIR pass”。

值得借鉴的是 op 与 compiler stage 的职责：

- src/op/operator.h:217-237 为每个 tile op 注册明确 builder；
- tilelang/cpu/pipeline.py:15-88 先保留高层 op/access information，再做 layout/reducer
  planning，然后 LowerTileOp，最后才进入 storage/vector/codegen；
- src/transform/layout_inference/layout_inference.cc:92-272 让 op 贡献 layout requirement，再沿
  use-def/alias 传播，对不一致布局明确报错。

Weft 对应采用：

1. 每个 explicit canonical op 保存自己的数值、axis 与 effect 语义；
2. op 只向共享 physical planning 贡献 typed facts/requirements，不返回 whole-kernel template；
3. mapping/resource/selected decision 在 emission 之前完成；
4. emitter 可以复杂地处理 intrinsic/ABI/asm spelling，但不重做物理选择。

没有复制 TileLang 的 GPU kernel launch、thread binding、shared/fragment allocation、TMA 或
persistent thread-block scheduler，因为它们与 Weft 的 single-worker/hart 根模型冲突。

## 四、新 MLIR compiler invocation 的权责

### Parse 与 canonical verification

weft-compile 只注册 weft_kernel 和 weft_ext dialect，把 textual MLIR 解析为 ModuleOp 并运行
MLIR verifier。--emit=kernel-ir 只是打印已验证 canonical IR，不是第二 target backend。

### 唯一 artifact pass

CompileRISCVArtifactPass 是 OperationPass<ModuleOp>。它拥有当前 target/options，并且只在
body、prelude 与 header 都成功后才提交 RISCVArtifact。

该 pass 是 programmatic pass，没有注册成 weft-opt 的第二 target CLI。Artifact compile 必须
拥有具体 target profile、backend config 和输出 sink；在 weft-opt 再暴露一个不完整 artifact
contract 的 pipeline 会形成第二入口。

### Per-kernel MLIR analysis

KernelPhysicalFactsAnalysis 的 key 是 canonical KernelOp。它是以下 target-independent facts
的唯一 producer：

- block/VLA logical axis identity、extent 与 ordered parents；
- SSA value 的 logical axes、consumer、last use、multi-use、cross-region 与 control carry；
- memory pointer/root/predicate/element type；
- pointer 对 logical axis 的 unit/strided/indexed/non-affine relation；
- indexed 与 interleaved memory groups；
- operation ordinal/lifetime 基础。

RISCVReuseAnalysis 不再保留另一份 dependsOn 递归；compiler 内的 constant/SSA dependency helper
也收缩到 facts module。因此 reuse 和 target planning 消费同一份 canonical facts。

### Op-specific planning，而非 whole-kernel classification

preparePhysicalDecisions 现在用一个 typed TypeSwitch<Operation *> 遍历主要 explicit
primitive。VLA、dot/matmul、sort、F16 load 和 typed quant/codebook command 都从各自 op
semantics 进入 decision producer。它不看 kernel/example 名，也不以 VLEN128/256 选择一份
whole-kernel compiler。

仍有少量附加 walk 用于 materialized block value、pointwise/store 和 block store/reduce group
的跨 op 聚合。它们构造 multi-op physical entity，不是另一 primitive route。准确表述是
“一个主 op planning walk + 少量 entity aggregation walks”，而不是虚假声称整个 compiler 只
遍历 IR 一次。

Typed quant op 仍有不同 decision function，因为 affine/grouped/codebook/bitplane 具有不同的
局部可观察数值语义。它们使用共享 mapping/resource/quant candidate infrastructure，不应为了
表面统一而折叠成按格式字符串分支的 opaque op。

### Prepare 与 emit

KernelLowering::prepare 先建立 ABI projection、facts-backed physical plan 与 selected local
implementations；KernelLowering::emit 随后用 selected plan 生成 C body。

Emitter 仍须读 canonical op 的 scalar control、address 和数值语义，否则无法生成普通 C。
“emitter 只消费 decision”的准确含义是：它不重新选择 LMUL、microtile、memory form、
fragment、decode family 或 pipeline。当前 active emitter 中没有调用 select*Physical 重做这些
选择。

## 五、删除的旧 authority

已经删除：

- include/Weft/Target/RISCVLowering.h；
- lib/Target/RISCVLowering.cpp；
- public RISCVLoweringOptions；
- public lowerToRISCVIntrinsicC；
- public emitRISCVHeader；
- tool 层 body/header 分别调用 lowering 的路径；
- reuse/compiler 内重复的 SSA dependency 与 constant facts producer；
- 主 primitive planning 中每类 op 单独 kernel.walk 的结构。

Header/body 仍是两种 artifact sink，但它们只在 lib/Target 内部可见，并由同一个 artifact pass
调用。Header 投影 ABI/storage metadata，不构造 target physical decisions，因此这是两个输出
sink，不是两条 compiler authority。

## 六、现行规范同步

本轮修改了：

- doc/compiler/architecture.md：写明 MLIR artifact pass、AnalysisManager 与当前 artifact 边界；
- doc/compiler/lowering.md：区分 intrinsic-C/header 与 system compiler 之后的
  object/library/executable；
- doc/compiler/target-lowering.md：明确通用 W.pipeline loop scheduler 当前尚未实现。

一个重要纠正是：weft_kernel.for pipeline=true 只保存作者授权。当前 target 对此使用合法的
顺序 realization，尚未生成通用 prologue/steady/epilogue。Dot/matmul 内已有的 K-unroll 和
register load buffering 是 explicit primitive 内部机器实现，不是 W.pipeline 已完成的证据。

当前 compiler API 的 artifact 只包含 intrinsic C 与 public header。Object、library 和
executable 是 system C compiler/runtime 的后续产物，不是 compileRISCVModule 返回字段。

## 七、实际验证

本轮只执行了一条允许的真实 repro，没有新建 test/framework/case matrix。

本地 compiler build：

~~~bash
cmake --build build --target weft-compile -j2
~~~

结果：成功。

K1 / RVV256 blocked GEMM：

~~~bash
./examples/run/weft.sh k1-rvv256 blocked_gemm_f32 decode 1
~~~

真机结果：

~~~text
kernel=f32_dense_projection
model_shape=Llama-8B.hidden_projection
phase=decode
M=1
N=4096
K=4096
repetitions=1
max_absolute_error=0
max_relative_error=0
weft_ms=21.066269
weft_gop_s=1.592804
~~~

这证明：

~~~text
新 DSL source
→ canonical MLIR
→ 新 MLIR artifact pass
→ intrinsic C/header
→ K1 system compiler
→ 真机执行
~~~

数值误差为零。由于只有 repetitions=1，时间数字只是主链 repro 附带输出，不是按正式协议重测
的性能结果。因此本轮没有修改 report/weft-kernel-performance.csv，避免把一次冷/热状态不明
的样本写成当前性能事实。

## 八、完成状态与真实缺口

本轮已经完成：

- MLIR PassManager target compile invocation；
- per-KernelOp AnalysisManager facts lifetime；
- 唯一 public RISC-V compiler API 与一次 artifact transaction；
- 主 primitive 的单一 typed planning dispatch；
- planning 之前唯一 axis/use/memory facts producer；
- prepare-before-emit 的 target decision authority；
- 旧 public lowering/header API 和 tool 双入口删除；
- fixed-RVV target 不满足时明确失败，无 fallback；
- 新主干真机零误差 repro。

仍然存在且本报告不伪装已完成：

1. 通用 loop-local pipeline scheduler

   W.pipeline 授权已在 canonical IR，但 target 尚未从普通 loop use/effect/alias 依赖生成
   prologue/steady/epilogue。当前只有 primitive-local K-unroll/load buffering。

2. 自动 compile-and-measure selector

   Candidate enumeration、resource filtering 和多个 backend config 维度已经存在，但当前是单次
   compile 中的确定性选择。仓库没有自动编译、真机测量并固化 winner 的 tuning pipeline。

3. 更小的 compiler/emitter 模块

   KernelLowering 仍是较大的实现文件。本轮已用 MLIR analysis、prepare/emit 边界与 typed plan
   固定 authority，但还没有把所有 ordinary-C、block、quant 和 IME projection 物理拆到更小
   文件。后续拆分必须保持同一 selected plan，不得重新造 selector。

4. 跨 op entity 仍需附加 aggregation walks

   Materialized block group/store/reduce 需要聚合多个 canonical op，目前在主 TypeSwitch 之外
   使用附加 walk。它们不是第二 lowering path，但 entity planning 仍可进一步收敛。

5. 没有改为 LLVM/EmitC target dialect

   这不是缺一层“更真”的 MLIR，而是当前产品边界：Weft 发射 RVV intrinsic C 和 typed IME
   asm，再把寄存器分配、最终机器调度与 peephole 交给 system compiler。只要 direct
   translation 消费 verified MLIR 与 selected decisions，而不是重建算法，就无需为形式对称
   新造 target IR。

## 九、本轮后的冻结边界

~~~text
registered canonical MLIR ops
→ MLIR-managed facts
→ op-specific requirements
→ shared transient physical planning
→ selected local decisions
→ direct intrinsic-C/asm artifact translation
~~~

后续优化可以扩大 mapping、resource、reuse、pipeline 或 local leaf，也可以物理拆分 emitter；
但不得恢复：

- 另一个 public lowering API；
- target 端 body/header 各自选择 implementation；
- 另一份 physical/selected authority；
- 按 kernel/example/q-format/VLEN/target 名选 whole-kernel path；
- emitter 内重新计算 LMUL/microtile/pipeline/fragment；
- scalar、legacy、GGML 或 materials fallback。

这次重构没有让 Weft 变成 Triton-CPU 或 TileLang-CPU。它只借鉴真实编译器的 typed IR、analysis
lifetime、backend authority 和 planning-before-emission 原则，并保留 Weft 的
single-controller、worker/hart-local、non-SIMT 边界。
