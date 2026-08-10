# 协作规则

## 唯一目标

推进独立的 Weft RISC-V kernel DSL/compiler 新主干。

判据：本轮改动是否让某个真实的 Weft 语言构造或 Kernel IR 节点，更接近
“能够被 Weft 编译器发射成可运行的 RISC-V kernel”？
不是，就不要做。

## 项目定位

- Weft 是一门独立的 RISC-V blocked/VLA kernel DSL 与编译器，地位类似
  Triton 对 GPU；不是图编译器、算子目录、kernel 名称分派器或旧量化格式路由器。
- canonical Weft Kernel IR 是 Weft 编译边界内的算法唯一真理。它可以由 Weft
  Python eDSL 直接生成，也可以由 Intent 或其他前端 lowering 生成。
- Weft core 不解析 intent.*，不依赖 intentdsl。Intent→Weft conversion 属于
  Intent 侧 target plugin 或独立 bridge，依赖方向只能是 Intent → Weft。
- bridge 负责把逻辑算法单向 lowering 成参数化 blocked program；Weft 负责在该
  程序语义允许的空间内选择 meta-parameter、VLA layout 和 Scalar/RVV/IME 实现。
  两边都不得靠 kernel 名替换整段算法。
- RISC-V domain 可以包含 Scalar、RVV、IME 和未来扩展；owner 按结构化节点或物理
  region 接入，不要求整个 kernel 只能属于一个 owner。
- /home/kingdom/phdworks/intentdsl 是活跃上游。除非用户明确要求，始终只读，
  不修改、不整理、不替它提交。

## 表示层纪律

Weft 内部正式、可持久化的承重层只有三层：

1. canonical Weft Kernel MLIR；
2. 一份 selected RISC-V execution/layout MLIR；
3. 生成的 RISC-V source/object/header artifact。

分析结果、索引、typed body、EmitC 和 target helper 可以作为单向生成的瞬态实现，
但不得成为第二份算法真理或第二份物理决策：

- 能从 Weft Kernel IR、selected execution 和静态 target 表推导的字段，不得另立 schema；
- 瞬态 IR 不得被缓存后绕过上游重算，不得反向驱动 selection；
- emitter 只从 Weft Kernel IR、selected execution 和 target 静态表取事实；
- owner-local 信息不得塞进跨 owner 的 giant optional struct。

## 目录结构纪律

目录结构就是架构，内部层级和顶层目录同样重要。动手前先从完整数据流判断文件归属，
不能只找一个能放的位置。

- 同一层保持一致抽象，用稳定职责和 lowering 边界组织文件。
- Python frontend、Weft Kernel dialect、RISC-V analysis/execution layout、
  owner realization、emission、artifact 必须分层放置。
- Scalar、RVV、IME 的 owner-local 代码各自收拢；共享层不得包含 owner 名分支。
- 强耦合、共同演进的文件相邻；共享内容放在职责明确的共同边界。
- 缓存、环境、临时产物和一次性实验输出不得进入项目。
- 落点或边界不明确时停下来问用户。

## 新旧路径

- 新主干不得经过旧 canonical-problem → whole-kernel variant → format body 路径。
- 新主干也不得以“Intent adapter”为核心入口；weft-compile 只接受 canonical
  Weft Kernel IR，Intent bridge 不能进入 Weft core。
- 旧代码只是供体：只按新接口逐件抽取硬件知识、typed primitive、ABI 或 artifact
  机制，不为复用而保留旧入口。
- 新路径替代某段旧路径时，同一轮删除被替代分支、派生表示、兼容旁路和死代码。
- 不建 compatibility layer，不保留 deprecation path；历史由 git 保存。
- 未实现的结构直接报 unsupported 或 NotImplementedError，不得静默回落旧 emitter。

## 卡住时

遇到设计歧义、上游接口仍在变化、信息缺失或方案分叉，停下来问用户。

禁止用“先造一个可交付物”、兼容壳、默认值或假实现填补不确定性。宁可一轮只提出
一个真正阻塞的问题，也不要生成一堆自洽但不会进入 lowering 主干的材料。

## 验证

唯一允许的验证是一条可手动执行的真实 repro：

Weft DSL kernel → canonical Weft Kernel MLIR → selected RISC-V execution →
RISC-V source/object → 真实目标运行 → 数值对照。

- 独立 DSL 主干先用手写 Weft kernel 跑通；Intent 接入时再用 intentdsl 现有
  kernel 生成同一 Weft IR，不复制出测试 corpus。
- 不建 test/ 或 tests/，不用 pytest/lit/CTest，不留 fixture，不累计测试数字。
- 一个改动只需要覆盖它推进的真实语言构造；不补边界测试、兼容测试或脚手架。
- 如果当前没有可用 RISC-V 机器，必须明确停在 source/object，不用模拟结果冒充。

## 禁止

- 任何 hash、SHA、checksum 来源或产物校验；
- 上述 repro 之外的单测、边界测试、版本兼容测试和验证脚手架；
- 版本号、CHANGELOG、迁移指南、deprecation 标记；
- 未经要求的目录整理、注释批量补写、README 扩写或历史总结；
- 按 kernel 名称、算子名字、量化格式名选择整条 lowering 路径；
- try/except 吞异常、默认值兜底、防御性 fallback 和空壳成功；
- emitter 从 kernel 名、operand 顺序或 route 字符串重新推导算法或 physical layout。

## 交付形式

- 只改推进当前真实 lowering 所必需的文件。
- 用户明确要求设计文档时可以创建；否则不新建计划、进度、迁移或总结文档。
- 回复结构：改了什么（一句）→ 关键设计取舍 → 真正卡住的地方。
- 不用测试数量、文件数量或历史覆盖率包装完成度。
