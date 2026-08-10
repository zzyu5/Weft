# Weft 仓库协作规则

## 1. 唯一设计权威

- 根目录 [`WEFT_FINAL_SPEC.md`](WEFT_FINAL_SPEC.md) 是 Weft 语言、IR、provider、
  lowering 与 artifact 的唯一规范性设计。
- `doc/` 是工程解释、现状审计和重建决策，不得另立语义，不得覆盖规范。
- 当前源码只说明“旧实现现在做了什么”，不说明“最终设计应该是什么”。源码、注释、
  旧提交或辅助文档与最终规范冲突时，以最终规范为准。
- 不保留旧设计的兼容语义、弃用期或双轨入口；历史由 Git 保存。

## 2. 项目定位

- Weft 是面向 RISC-V 的 worker-local AOT kernel DSL/compiler，作用类似 Triton
  面向 GPU kernel，但它不是 Triton 的 SIMT/grid 模型移植。
- 一个 Weft kernel 描述一个普通 RISC-V worker/hart 内执行的算法。线程创建、worker
  编号、工作切分、NUMA 与线程池属于外部 runtime。
- Weft core 不得出现 `program_id`、`task_id`、`grid_rank`、隐式 hart identity 或
  launch grid。跨 worker 的任务分配必须通过普通 ABI 参数或 descriptor 显式传入。
- `W.vla(begin, end)` 是一等的一维 VLA iteration region；strip、`vl`、LMUL 与寄存器
  分组是物理实现，不得暴露给 source 或 canonical IR。
- logical block 是局部 region value，不是根程序模型、任务、寄存器 tile、cache tile
  或 ISA fragment。上述层次必须分别建模。
- logical predicate、masked value 和 physical tail 是三个不同概念；不得用 RVV tail
  mask 代替算法有效性。
- `reduce`、`scan`、`summary_fold` 与普通 sequential carry 是不同的 state algebra，
  不得统一塞入一个模糊的 loop/reduction 路径。

## 3. 编译边界

Weft 只有三类持久化承重表示：

1. canonical Weft Kernel IR：唯一算法真理；
2. Selected Execution IR：只记录多个合法实现中选中的物理决定；
3. source/object/static-library/header/dispatcher artifact。

分析索引、候选集合、owner-local IR、EmitC/LLVM IR 和 target helper 都可以是瞬态实现，
但不得成为第二份算法真理或第二份物理计划。发射只可读取 canonical IR、selected IR
和静态 target facts；能从这三者推导的字段不得再持久化一份 schema。

Python eDSL 是参考前端和 AOT 构建入口，不是运行时 JIT。它应直接构造 canonical
Kernel IR；允许使用通用的单向 MLIR assembly builder，但不得维护另一套长期 typed IR
或 verifier。

IntentDSL 和其他上游只可在 Weft 之外 lowering 到 canonical Weft Kernel IR：

```text
IntentDSL / other frontend  ->  canonical Weft Kernel IR  ->  Weft
```

Weft core 不解析 `intent.*`，不链接 IntentDSL，也不反向调用上游。除非用户明确授权，
`/home/kingdom/phdworks/intentdsl` 始终只读。

## 4. Provider 纪律

- Provider 绑定 primitive/interface、typed operands、target facts 和局部上下文；不得
  按 kernel 名、算子名、量化格式名或 whole-kernel route 接管整条 lowering。
- Scalar、RVV、IME 和未来扩展是可组合的 primitive realization providers，而不是
  互斥的整 kernel 后端。
- 语义相同的扩展只能提供 realization；语义不同的扩展必须先成为明确的 canonical
  extension primitive，不能藏在 provider 内替换算法。
- packing、局部 fusion、microtile、指令选择和寄存器约束属于 provider/selection；
  算法分块、状态代数、逻辑 predicate 和数值语义属于 source/canonical IR。
- 未实现的 primitive 或组合必须明确失败；禁止静默走旧 emitter、Scalar fallback、
  默认 route 或假成功路径。

## 5. 仓库重建边界

当前源码树是 donor，不是最终骨架。工程决策见
[`doc/REPOSITORY_RECONSTRUCTION.md`](doc/REPOSITORY_RECONSTRUCTION.md)：在同一 Git
仓库内建立全新的根骨架，不在现有 grid/task 主干上渐进修补，也不立即拆成新仓库。

执行重建时必须遵守：

- 先把旧实现整体隔离到 `materials/legacy-source/`，再在根目录建立新的独立依赖图；
- `materials/` 永远不进入 CMake、include path、Python package path、链接、安装或运行时；
- 复用的含义是阅读并把最小代码/硬件知识抽取到新接口下，不是调用、包装或链接旧路径；
- 不建 compatibility layer，不让新旧主干并行可达，不用 feature flag 切回旧实现；
- 每个抽取文件都必须能脱离 donor 独立解释，名称与依赖服从新架构；
- 不确定某段代码是算法语义还是物理实现时，先回到最终规范，不按现有目录猜归属。

## 6. 文档纪律

- 语义定义只写入 `WEFT_FINAL_SPEC.md`；需要修改规范时必须由用户明确决定。
- 工程文档必须标明它描述的是规范解释、当前事实还是未来实施决定。
- 不新增“V2”“旧版”“迁移指南”“总纲副本”或按阶段复制的设计文档。
- 旧文档被替代后直接删除；不在仓库里另建 archive。代码材料与历史文档不是一回事。
- 重要断言尽量指向实际符号或文件；不要用计划表把未实现能力写成已完成能力。

## 7. 工作方式

- 搜索和阅读本地文件优先使用 FastCtx；已知位置的少量文件和即将修改的代码由主代理
  亲自读取。
- 跨目录审计、宽检索和独立核验可并行交给只读子代理；设计取舍、实际修改与最终验证
  由主代理完成。
- 工作区可能含用户或其他代理的改动。不得覆盖、回滚或顺手整理无关内容。
- 未经用户明确要求，不执行 commit、push、rebase、reset 或清理未跟踪文件。
- 不添加测试目录、覆盖率、边界测试账本或验证脚手架。验证采用与改动对应的最小真实
  AOT repro：DSL/Kernel IR → selected IR → source/object → 目标机数值对照。目标机不可用
  时，明确停在哪个 artifact 边界，不伪造运行结果。
- 不用 hash、checksum、版本号、CHANGELOG、deprecation 或兼容层制造额外协议。

## 8. 完成判据

一项实现工作只有在以下条件同时成立时才算进入新主干：

- 数据流符合最终规范且没有读取 donor 路径；
- canonical 与 selected 的事实没有重复；
- provider 只拥有局部 primitive realization；
- unsupported 组合 fail closed；
- 产物边界和实际验证状态被如实说明。
