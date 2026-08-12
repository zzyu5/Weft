# 协作规则

## 唯一目标

推进 Weft 编译器骨架。判据：本轮改动是否让某个语言构造或 IR 节点更接近
“能 lowering 出可执行的 RISC-V 后端代码”？不是，就不要做。

## 设计边界

- [`doc/index.md`](doc/index.md) 及其链接的模块是唯一规范；源码是当前实现事实。
- Weft kernel 是 worker/hart-local 程序。core 中禁止 grid、task identity、隐式 hart identity
  和 GPU SIMT 根模型。
- 持久表示只有 canonical Kernel IR 和最终 artifacts。LMUL、microtile、packing、fragment、
  capability、legality 与候选选择都是单次 target lowering 内的瞬态实现。
- Target lowering 只根据 primitive、typed operands、局部 use relation、显式 backend config
  与 target facts 生成代码，禁止按 kernel、算子或量化格式接管 whole-kernel lowering。
- IntentDSL/其他前端在仓库外生成 canonical Weft IR；Weft 不解析或链接 IntentDSL。
  `/home/kingdom/phdworks/intentdsl` 默认只读。
- `materials/` 只作代码与硬件知识供体，不进入 CMake、include、import、link 或 runtime；
  新代码不得调用、包装或 fallback 到旧实现。

## 目录结构纪律

目录结构就是架构，内部层级和顶层目录同样重要。动手前先从整体结构判断文件归属，
不能只找一个能放的位置。

- 同一层保持一致抽象，以稳定职责和 lowering 边界组织文件。
- 模块拆到职责清楚，但不为单次任务随手造层级；强耦合、共同演进的文件应相邻。
- `source/` 按语言、上游来源、算子职责和变体形成可读层级，source 与 runtime 相邻。
- `examples/` 只放手工 repro 入口，不保存第二份算法 source。
- `report/` 是扁平的工作记录区。Markdown报告是一次性交付快照，产出后不持续更新、同步
  或维护；新一轮需要报告时新建主题文件，除非用户明确指定，不回写旧Markdown报告。
  `weft-kernel-performance.csv` 是当前Weft性能数字表，真实重测后直接更新；它不附带阈值、
  同步或校验逻辑。实现问题、真实结果和阻塞写入当轮主题报告，不进入 `doc/`。
- 缓存、环境、临时产物和 generated artifacts 不得进入仓库。

## 卡住时

遇到设计歧义、信息缺失或方案分叉，停下来问用户。禁止用“造一个可交付物”填补
不确定性；宁可只提出一个关键问题，也不要生成一组自洽但无用的文件。

## 验证

唯一允许的验证是一条可手动执行的真实 repro 命令：把 DSL source emit 成后端代码，
实际运行并对照数值。目标机或后端尚不可用时，明确报告停在哪个 artifact 边界。
不建 test 目录，不用 pytest/lit，不留 fixture、case matrix 或验证脚手架。

## 提交

- 一个连贯节点完成且 repro 通过后，立即自行 commit，保持工作区干净，不等待用户提醒。
- 未完成或卡住的工作不提交；先报告唯一阻塞点。
- 未经用户明确要求，不 push、rebase、reset、改写历史或提交其他人的无关改动。

## 禁止

- 任何 hash / SHA / checksum 校验来源或产物。
- 上述 repro 之外的一切测试：单测、边界测试、版本兼容测试、脚手架。
- 版本号、CHANGELOG、迁移指南、deprecation 标记——有 Git 就够了。
- 未经要求的重构、目录整理、注释批量补写或 README/doc 更新。
- 在 `doc/` 新建计划、进度、状态快照或设计总纲副本；可变事实只写入 `report/`。
- 兜底代码：吞异常、默认值兜底、“防御性”分支、静默 fallback 或假成功路径。
  未实现就直接 `raise NotImplementedError` 或返回明确的 unsupported error，不要假装能跑。
- compatibility layer、旧入口、双主干、feature flag 回退或按阶段保留旧路径。

## 交付形式

- 只改必要文件。
- 回复结构：改了什么（一句）→ 关键设计取舍 → 卡住的地方。
- 不复述执行流水账，不罗列“下一步建议”。
