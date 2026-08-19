# RISC-V Target Lowering

## 输入

Target lowering 只接收：

1. canonical Kernel IR 的 typed semantics；
2. 显式 backend config/constexpr instance；
3. 一份 target profile：ISA、ABI、VLEN、SEW/LMUL、vector register budget、标准扩展、IME/vendor
   extension、fragment 和 memory capability。

VLEN128/VLEN256、SG2044/K1 或量化格式不能成为 implementation identity。

## 决策分类

### 唯一合法推导

包括 axis role、pointer relation、validity、widen/narrow relation、effect/alias legality、fragment
shape legality 和 live resource accounting。这些事实只有一个正确答案，在共享逻辑中产生一次。

### 结构性选择

包括：

- scalar/vector state；
- M-lane/N-lane 或 register-repetition 组织；
- RVV register microkernel/IME fragment；
- shared/reload/rematerialize/local-pack handoff；
- unit/strided/indexed/segment schedule；
- sequential/single-buffer/double-buffer local pipeline；
- decode materialize 或 decode-compute fusion。

结构选择只能作用在 source 显式 command 和作者循环内部。

### 参数性选择

包括 LMUL、register factor、MR/NR、accumulator count、K-unroll、buffer count、prefetch distance
和已由 source 标记为 constexpr 的 block extent。参数实例化结构，不定义新的算法。

## 统一物理坐标

逻辑轴可被分解到：

```text
sequential inner iteration
lane
register repetition / accumulator tuple
unroll
optional extension fragment coordinate
```

Pointwise 保持映射；cast 传播元素身份并改变 SEW/resource；memory relation 约束 lane；reduce/scan
约束被消去轴和顺序；gemm 将 M/N/K free/reduction axis 贡献给同一 mapping；quant command 将
packed/group/codebook axis 贡献给同一资源与 schedule。

## Value-chain planning

Lowering 联合观察合法 use-def 链，而不是逐 op 选孤立 LMUL：

```text
load → cast/decode → gemm/vdot → pointwise/state → multiple consumers → store
```

同一 value 的 lane、SEW、mask/index 与 handoff 必须一致。Consumer 要求冲突时，在资源和代价
允许的共享、reload、rematerialize、shuffle 或 primitive-private local pack 中选择；不能因为
新增普通 consumer 就失去 command lowering。

## Reuse-driven microkernel

Gemm/vdot compile rule 提供 free/reduction axes 和 numerics；共享 reuse analysis 读取 operand
invariance、pointer stride、output lifetime 与 target resources，生成 register microtile、multiple
accumulator、operand load window、K-unroll 和 load/compute order。GEMM、Conv、Attention、MoE
与 OutProd 不拥有各自的 dense selector。

## Loop-local pipeline

只有 `weft_kernel.for pipeline=true` 才授权 target 调度当前 loop。Scheduler 基于真实 producer、
effect、alias、state dependence 与 temporary lifetime 构造 prologue/steady/epilogue。顺序实现始终
合法；target 不能创建跨 loop workspace、persistent buffer 或作者未写的算法 stage。

## Quant 与 packed compute

Typed quant op 保留不同数学关系；共享 lowering 处理 packed-axis mapping、decode chunk、widen、
gather、scale/correction accumulator、cross-output reuse、resource 与 pipeline。格式名不能选择
完整 microkernel；最低层 leaf 只实现一个局部硬件序列。

## Resource

资源从最终 mapping 与 schedule 统一计算：operand、accumulator、mask/index、state、cast/decode
temporary、pipeline buffer、handoff 与 extension fragment 同时计入。超预算 candidate 在 emission
前整体非法，emitter 不得偷偷缩小 LMUL、microtile 或 buffer。

## Target capability

RVV、IME 或未来 extension 只描述：支持的 explicit command、dtype/axis mapping、memory/mask/tail
限制、input/output physical shape、resource 与 intrinsic/asm spelling。新增硬件应扩张已有 command
的合法 realization；只有新可观察语义才修改 DSL/Kernel IR。
