# 当前实现状态与最终规范差距

> 审计基准：2026-08-10 的当前工作树源码与 `WEFT_FINAL_SPEC.md`。本轮只做源码静态审计
> 和文档重构，没有重新构建或在 RISC-V 目标机运行；因此本文不把历史运行结果写成当前
> 已验证事实。

## 1. 总结

当前仓库已经有独立 Python package、Kernel/Execution/RVVExecution/IMEExecution dialect、
`weft-compile`、RVV/IME source emission 和部分 object packaging。这些证明“独立 Weft
compiler 外壳”有可抽取资产。

但其语言与 selection 主干建立在 `grid_rank/task_id/fixed arange/whole-kernel owner` 上，
而最终规范明确禁止这些概念。它不是最终设计的早期子集，而是一个语义方向不同的原型。
因此当前树应整体视为 donor；不能把“已有 dialect/CLI”误写成“最终架构已基本成立”。

## 2. 能力对照

| 最终能力 | 当前源码事实 | 判断 |
|---|---|---|
| worker-local kernel、无 grid | Kernel/Python frontend 保存 `grid_rank` 并暴露 `task_id` | 必须重写 |
| 一等 `W.vla(begin,end)` | 只有 fixed `arange` 与结构化 scalar `for` | 缺失 |
| VLA strip 对 source 不可观察 | RVV emitter 从 rank/arange 生成物理 loop | 入口模型错误 |
| logical block 仅为局部 value | block/rank/layout group 主导 whole scope | 必须重新分层 |
| logical predicate + masked value | load/store 固定 mask/other，无 validity-carrying value | 缺失 |
| reduce/scan/summary/sequential carry | 仅 reduce 与特判 block-carried loop | 大部分缺失 |
| primitive-local provider | selector 按整个 kernel/target route 选择 RVV 或 IME | 必须重写 |
| Scalar/RVV/IME 可组合 | CLI/selector 主要是互斥 target owner | 缺失 |
| canonical 是唯一算法真理 | 有 generic builder，但生成旧 canonical schema | 基础设施可抽取，语义重写 |
| selected 只存物理选择 | 有独立 Execution dialect，但含 task/layout whole-scope 假设 | 思路可抽取，schema 重写 |
| source/object/header artifact | RVV 有 source→object 机械路径；IME 主要停在 source | 部分机械资产可抽取 |
| Intent 外置 bridge | core 已可接受 canonical MLIR，但旧文档曾把 bridge 绑定错误根模型 | 文档已纠正，代码未接 |

## 3. 直接冲突的实现入口

以下符号是源码审计的关键证据；行号会随重建变化，文件和符号比行号更稳定：

### 3.1 Frontend 与 Kernel dialect

- `python/weft/api/definitions.py::KernelDefinition` 保存 `grid_rank`；`kernel()` 暴露该参数。
- `python/weft/frontend/compiler.py::FrontendCompiler` 写入 `grid_rank`，并 lowering
  `weft_kernel.task_id`。
- `python/weft/language/builtins.py` 暴露 `task_id/arange`，没有最终规范要求的 `vla`、
  predicate helpers、scan 或 summary fold。
- `include/Weft/Dialect/Kernel/IR/KernelOps.td::WEFTKernel_KernelOp` 带 `grid_rank`；
  `WEFTKernel_TaskIdOp` 直接物化 launch identity。
- 当前 load/store schema 强制 pointer/mask/other operands，没有一等 masked-value validity。

结论：不能给 `task_id/arange` 增加兼容别名来“过渡”到 `vla`；KernelOp、types、ops、
verifier 和 Python lowering 应在新骨架一起重建。

### 3.2 Selection 与 Selected IR

- `lib/Transforms/SelectRISCvExecution.cpp` 以整个 `kernel::KernelOp` 为分析和选择单位，
  统一构造 RVV layout/group/reduction/contract config，并包含 rank/shape 特判。
- `lib/Transforms/SelectIMEExecution.cpp` 要求固定 contraction 形态，并检查 entry 中其他节点
  是否被该 site 覆盖，体现 whole-owner 假设。
- `lib/Dialect/Execution/IR/ExecutionDialect.cpp` 的 verifier 依赖 task-axis bindings 与
  whole-scope layout groups。
- `tools/weft-compile/weft-compile.cpp` 在 IME target 与 RVV target 之间选择整条路线，而非
  对 primitive 组合 providers。

结论：独立 selected dialect 与 canonical anchor 的思想可保留，但当前 selector、plan
schema 和 whole-target dispatch 不能成为新 provider framework 的兼容层。

### 3.3 Emission

- `lib/Target/RVV/SelectedExecutionRVVSource.cpp` 给 ABI 追加 task 参数，依赖共同 `arange`
  extent 与 rank-specific layout group，再生成物理 strip loop。
- `lib/Target/IME/SelectedExecutionIMESource.cpp` 同样依赖 task binding，并假定固定
  fragment/shape/flat-body 结构。
- `lib/Plugin/IME/IMEBackendEmissionDriver.cpp` 的后半部分包含 q4/q8/格式特定 whole-kernel
  emission；这属于旧产品语义，不是局部 extension provider。

结论：两个 selected source emitter 的 orchestration 必须舍弃，只抽取 intrinsic spelling、
mask/load/store、reduction 或单条 IME asm 等叶子知识。

## 4. Donor 资产清单

### 4.1 高价值机械资产

- `python/weft/frontend/ir.py`：通用、单向的 MLIR `Value/Region/Operation` builder/renderer；
- `python/weft/frontend/source.py`：source capture、location、closure/global static binding；
- `include/Weft/Dialect/*` 与相应 CMake：TableGen/generated include/registration 骨架；
- `InitWeftDialects`：dialect registry 机械入口；
- `tools/weft-compile`：parse module、选择输出 artifact 种类的 CLI 外壳；
- `RVVTargetSupportBundle`：RISC-V clang target、source→relocatable object 的 toolchain 机械层；
- `TargetArtifactExport`/`ConstructionTemplateArtifactAdapter`：header/object/bundle 写出与
  fail-closed handoff 的通用逻辑。

这些资产可在新根中按职责复制；旧 target/library 不应继续被链接。

### 4.2 可抽取的硬件与算法无关知识

- RVV capability profile 对 VLEN、SEW、LMUL、register count 的 typed 解析；
- Kernel/Layout dialect 中 broadcast、shape、extent provenance、layout algebra helper；
- RVV intrinsic type/LMUL spelling、masked memory 与 ordered/tree reduction instruction mapping；
- IME 单条 instruction/helper wrapper、fragment/resource arithmetic；
- selected record 通过 canonical anchor 查找 source node 的机制。

这些代码当前混在旧 plugin/whole-owner 对象图中，需要小块抽取并改写依赖。

### 4.3 不作为资产的旧路径

- Plugin `CanonicalProblem/Variant/FamilyConstruction/route` 主干；
- RVV/IME whole-kernel source front door；
- q-format/operator 名称分派；
- grid/task ABI、layout-group root orchestration；
- 把旧 emitter 包成 fallback 的任何 adapter。

## 5. 当前 artifact 边界

从源码可确认，CLI 已区分 selected MLIR、generated source 和 relocatable object 输出，RVV
路径包含调用外部 RISC-V clang 的 object packaging；IME 新路径仍主要是 source-only。
本轮没有重新配置工具链、生成 object 或在目标机链接运行，所以这里只陈述源码能力，
不声称当前 HEAD 的产物已重新验证。

## 6. 这份状态对开发意味着什么

- 不应继续向当前 KernelOps 添加最终 DSL op；那会让 `grid/task` 与 `vla/worker-local`
  schema 共存。
- 不应把现有 selector 泛化成更多 if/else；应先建立 primitive provider contract。
- 不应优先补 IME object path；先补会固化错误 owner/ABI。
- 不应因为 artifact packaging 可用就保留其上游 family-construction 调用链；只抽取机械层。
- 下一次源码改动应先执行仓库隔离与新根 skeleton，再实现最小
  `VLA + predicate + Scalar/RVV` 纵向切片。

具体动作与目录边界见
[`REPOSITORY_RECONSTRUCTION.md`](REPOSITORY_RECONSTRUCTION.md)。
