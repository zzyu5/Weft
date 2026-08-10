# Weft 工程文档

本目录解释如何把 [`WEFT_FINAL_SPEC.md`](WEFT_FINAL_SPEC.md) 落成代码，并记录
当前实现与规范的差距。它不定义另一套 Weft 语言或编译器语义。

## 权威顺序

1. [`WEFT_FINAL_SPEC.md`](WEFT_FINAL_SPEC.md)：唯一规范，回答“系统必须是什么”。
2. `doc/`：非规范性的工程解释与已作出的仓库决策，回答“组件如何分界”和“当前如何重建”。
3. 新根源码：未来实现事实；`materials/` 中的旧源码只作历史供体，不能反过来修改设计含义。

发生冲突时严格按上述顺序处理。特别地，`materials/legacy-source/` 中的 `grid_rank`、
`task_id`、fixed `arange` 根模型、whole-kernel RVV/IME selector 和格式路由都不是兼容要求。

## 文档地图

- [`ARCHITECTURE.md`](ARCHITECTURE.md)：系统定位、worker-local 执行模型、核心语义
  边界、三层表示和端到端数据流。
- [`LANGUAGE_AND_IR.md`](LANGUAGE_AND_IR.md)：Python DSL 到 canonical Kernel IR 的
  合同、VLA/block/predicate/state algebra，以及 canonical/selected verifier 边界。
- [`PYTHON_DSL.md`](PYTHON_DSL.md)：第一里程碑的完整 Python frontend、canonical dialect、
  verifier、实现顺序和验收边界。
- [`PROVIDERS_AND_SELECTION.md`](PROVIDERS_AND_SELECTION.md)：primitive-local provider、
  target facts、compiler/tuner 权限与 Scalar/RVV/IME 组合方式。
- [`INTEGRATION_AND_ARTIFACTS.md`](INTEGRATION_AND_ARTIFACTS.md)：IntentDSL/其他前端接入、
  runtime ABI、AOT artifact 与 multiversion dispatcher 的责任边界。
- [`REPOSITORY_RECONSTRUCTION.md`](REPOSITORY_RECONSTRUCTION.md)：原地修改、另建仓库和
  同仓库净室重建的比较；最终选择、目标目录、`materials/` 隔离与代码抽取规则。
- [`CURRENT_STATE.md`](CURRENT_STATE.md)：当前干净根边界、尚未实现能力和 donor 状态。
- [`../materials/README.md`](../materials/README.md)：可抽取机械资产、硬件知识和禁止复用路径。

## 已作出的工程裁决

本项目没有在旧语义主干上逐步打补丁，也没有创建一个失去历史与材料邻接关系的新 Git
仓库。已经完成的方案是：

> 在同一个 Git 仓库中，把旧实现隔离为不可构建的材料，从根目录建立全新、独立依赖图
> 的 Weft 骨架；旧代码只允许按内容抽取，不允许被新主干调用。

旧树隔离已经完成；当前根 CMake 不包含任何旧 target。具体状态见
[`REPOSITORY_RECONSTRUCTION.md`](REPOSITORY_RECONSTRUCTION.md)。

## 文档维护规则

- 不再维护“总纲 + 重构提案 + A–H 模块”这种多份平行设计。
- 旧文档已直接删除，不在仓库内复制成历史版本；需要追溯时使用 Git。
- 语义变化只能先修改最终规范；工程文档随后同步，不得先在这里暗改语言。
- `CURRENT_STATE.md` 必须把“已实现”“源码推断”“尚未验证”和“目标设计”分开。
