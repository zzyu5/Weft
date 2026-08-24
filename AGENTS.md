# 协作规则

## 唯一目标

推进 Weft 的语言、Canonical Kernel IR 与 RISC-V 编译器主干。判据：本轮改动是否让一个
明确的语言构造或 IR 节点更接近经唯一主链生成可执行的 RISC-V artifact；实验只用于检验
这条主链。不是，就不要做。

## 设计边界

- [`doc/index.md`](doc/index.md) 及其链接是唯一设计规范；源码是当前实现事实；`report/`
  中的 Markdown 与 CSV 只保存一次性结果或测量数据，没有设计权威。
- Weft kernel 最终在 CPU 调用上下文中执行，但 worker/hart 不是语言根对象。core 中禁止 grid、
  task identity、隐式 hart identity 和 GPU SIMT 根模型。
- 程序 IR 只有两层：Canonical Kernel IR 与 target-aware RISC-V Physical IR。前者保存作者程序，
  后者保存 layout、conversion、memory、schedule、resource 与 selected local target operation；之后
  直接 terminal translation 到 intrinsic C / typed local asm，不增加第三层 planning IR。
- 改变 canonical operation/value/effect graph 或 Value 的 Level/artifact 归属，属于作者程序；
  只改变同一程序的物理表示，属于目标编译器。source `auto` 的有限程序域由作者/std 声明；
  physical parameter 的有限域由 target 声明，tuner 只测量合法绑定，不发明程序树或物理结构。
- Target lowering 只根据 primitive、typed operands、局部 use relation、显式 backend config
  与 target facts 生成代码，禁止按 kernel、算子或量化格式接管 whole-kernel lowering。
- RVV/IME leaf 是 final RISC-V IR 中合同闭合的局部 target operation，不能拥有 outer traversal、
  Level、workspace、persistent layout、local pack/pipeline 或 kernel ABI。leaf 由 physical passes 选择；
  terminal emitter 只能拼写已选 op，缺少决定必须报 verifier 错误，不能重新选择或 fallback。
- IntentDSL/其他前端在仓库外生成 canonical Weft IR；Weft 不解析或链接 IntentDSL。
  `/home/kingdom/phdworks/intentdsl` 默认只读。
- `materials/` 只作代码与硬件知识供体，不进入 CMake、include、import、link 或 runtime；
  新代码不得调用、包装或 fallback 到旧实现。
- `source/` baseline 决定比较时作者应自然表达的算法 variant、blocking、staging、persistent
  layout 与计时边界；作者树冻结后，它不参与 layout、leaf 或参数选择，也不进入生成代码。

## 目录结构纪律

目录结构就是架构，内部层级和顶层目录同样重要。动手前先从整体结构判断文件归属，
不能只找一个能放的位置。

- 同一层保持一致抽象，以稳定职责和 lowering 边界组织文件。
- 模块拆到职责清楚，但不为单次任务随手造层级；强耦合、共同演进的文件应相邻。
- `source/` 按语言、上游来源、算子职责和变体形成可读层级，source 与 runtime 相邻。
- `examples/` 只放手工 repro 入口，不保存第二份算法 source。
- `report/` 保存一次性工作快照；`report/baseline/` 保存 baseline 来源与固定测量，
  `report/weft-kernel-performance.csv` 保存当前 Weft 性能数字。Markdown 报告产出后不持续维护；
  新一轮需要报告时新建主题文件，除非用户明确指定，不回写旧报告。实现进度、真实结果和
  阻塞不进入 `doc/`。
- 缓存、环境、临时产物和 generated artifacts 不得进入仓库。

## 卡住时

遇到设计歧义、信息缺失或方案分叉，停下来问用户。禁止用“造一个可交付物”填补
不确定性；宁可只提出一个关键问题，也不要生成一组自洽但无用的文件。

## 验证

实现交付必须给出可手动执行的真实 repro：把 DSL source emit 成后端代码，实际运行并按
[`doc/experiments/`](doc/experiments/index.md) 的合同对照数值；生成 IR/C 或编译成功不能代替
运行。目标机或后端尚不可用时，明确报告停在哪个 artifact 边界。不建 test 目录，不用
pytest/lit，不留 fixture、兼容矩阵或验证脚手架。

## 提交

- 一个连贯节点完成且相应验证通过后，立即自行 commit，保持工作区干净，不等待用户提醒。
- 未完成或卡住的工作不提交；先报告唯一阻塞点。
- 未经用户明确要求，不 push、rebase、reset、改写历史或提交其他人的无关改动。

## 禁止

- 任何 hash / SHA / checksum 校验来源或产物。
- 上述 repro 之外的一切测试：单测、边界测试、版本兼容测试、脚手架。
- 版本号、CHANGELOG、迁移指南、deprecation 标记——有 Git 就够了。
- 未经要求的重构、目录整理、注释批量补写或 README/doc 更新；设计合同改变时必须同步修改
  `doc/index.md` 链接到的对应规范，不能另写一份总纲。
- 在 `doc/` 新建计划、进度、状态快照或设计总纲副本；可变事实只写入 `report/`。
- 兜底代码：吞异常、默认值兜底、“防御性”分支、静默 fallback 或假成功路径。
  未实现就直接 `raise NotImplementedError` 或返回明确的 unsupported error，不要假装能跑。
- compatibility layer、旧入口、双主干、feature flag 回退或按阶段保留旧路径。

## 交付形式

- 只改必要文件。
- 回复结构：改了什么（一句）→ 关键设计取舍 → 卡住的地方。
- 不复述执行流水账，不罗列“下一步建议”。
