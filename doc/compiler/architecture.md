# 编译器架构

## 唯一主链

```text
Python DSL kernel
→ canonical Weft Kernel IR
→ one target-specific analysis / decision / lowering
→ intrinsic C or typed local asm
→ system C compiler
→ object, header and executable/library
```

Persistent representation 只有 canonical Kernel IR 与最终 artifact。Kernel IR 可以保存作者的
`pipeline` 授权；具体 candidate、layout、LMUL、microtile、packing、pipeline schedule/buffer
和 fragment 都是一次 target lowering 内的短生命周期 C++ 数据，不形成 Physical IR、
Selected IR 或第二份 authority。

## 前端职责

Python frontend：

- 解析一门 Core-local command DSL；
- 保留作者 control、blocking、storage、effect、axis、value lifetime 与 command semantics；
- 把高层 source 名映射为稳定 canonical op；
- 在最早语义边界拒绝不完整或含混程序。

例如 source `W.axis/W.buffer/W.vdot/W.gemm` 分别进入 canonical
`block_index/storage/dot/matmul`。Canonical 名称描述稳定语义，不是兼容 source surface。

## Target lowering 的四个职责

### 1. 提取程序事实

只读取 Kernel IR 已写明的 axis、loop、use-def、pointer relation、predicate、effect、storage、
lifetime、dtype 与 explicit command semantics。

### 2. 进行唯一推导

推导 free/reduction/broadcast axis、memory relation、validity、cast/widen relation、target legality
和真实 live resource。这些结论错了会导致错误代码，因此只能有一个 producer。

### 3. 构造和选择物理实现

在 source 授权范围内构造 time/lane/register/unroll/fragment mapping、memory schedule、value
handoff、microkernel、local packing 与 loop-local pipeline。结构选择与参数实例都读取同一
facts/target facts；不按 kernel、格式、VLEN 或 source closure 进入完整路径。

### 4. 发射

Emitter 只把 selected decision 拼写为普通 C control/address、RVV intrinsic 与 primitive-local
IME asm。它不重新识别 command，不重新选择 LMUL、memory form、microtile、decode chunk、
pipeline 或 fragment。

## Op-specific rule 与共享 mapping

统一的不是所有 op 的语义。每个 explicit command 有自己的 compile rule：pointwise 保持元素
对应，memory 贡献 pointer relation，reduce/scan 贡献 order/state 约束，gemm 贡献 M/N/K 关系，
quant command 贡献 packed 数值关系。

这些 rule 向同一个物理 mapping、resource 与 local scheduler 提供约束。它们不能返回完整
whole-kernel template。

## Realization 的位置

Realization 是 target 内部对一个局部 command 的透明实现候选，例如 RVV outer-product 或 IME
fragment。它具有明确输入/输出物理形态、资源、mask/tail、dtype 与 instruction requirements，
但不拥有 outer loop、workspace、persistent layout 或 ABI。用户不编写 realization 程序。

## Product 边界

普通 DSL kernel 必须在不依赖 examples catalog 的情况下编译为可调用 artifact。`source/` 是
baseline source，`materials/` 是只读知识供体；二者不进入 include/import/link/runtime，也不
构成 fallback。

## 明确禁止

- kernel/example/q-format/target-name route；
- exact op count、one-use、direct-store、source adjacency 或 whole-region matcher；
- 从普通 scalar/SSA graph 猜 VLA、matmul、online softmax 或算法 variant；
- legacy/scalar/GGML/materials fallback；
- analysis 和 emitter 两次决定同一物理事实；
- system compiler 职责范围内的最终寄存器分配与机器调度。
