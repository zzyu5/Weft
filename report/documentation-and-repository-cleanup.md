# 文档与仓库清理报告

## 结论

原先 `doc/` 只有一份长规范和一个空 `modules/` 目录，无法从目录看出语言、编译器阶段与
kernel 模板的职责。现已按 `dsl/`、`compiler/`、`kernels/` 拆分；`doc/index.md` 是唯一
入口，原规范章节只移动和分组，没有改写语义。

顶层 `report/` 与设计文档分离。这里记录实现中发现的问题、真实结果和阻塞，不定义语言
或编译器合同，也不需要索引或分类层级。

## 本轮发现

- 活动根唯一空目录是旧 `doc/modules/`，本轮已经删除；
- 活动 Python package 和历史材料中残留 generated `__pycache__`，本轮已经清除；
- 其余一行 CMake 文件都承担真实 `add_subdirectory` 路由，不是空壳；
- `materials/` 中零字节 stderr、占位源码和 `.gitkeep` 没有可读取内容，本轮已经删除；历史
  仍由 Git 保存。

## 边界

本轮没有修改 Python frontend、MLIR dialect、CMake target、operator source 或 reference
runtime。没有产生新的 backend artifact，也没有把文档整理描述成端到端 lowering 已完成。
