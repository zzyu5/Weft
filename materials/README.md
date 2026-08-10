# Weft 历史材料与代码供体

`materials/` 保存重建前的实现、实验与产物，用于查找机械代码、硬件事实和历史证据；
它不属于现行 Weft 编译器，也不是兼容层。

## 硬边界

- 根 CMake、include path、link graph、Python import、安装和 runtime 都不得进入本目录。
- 新代码不得调用、包装或 fallback 到这里的 target、library、script 或 emitter。
- 复用只允许阅读后把最小代码内容或硬件知识抽取到新文件，并按最终规范重新建模。
- 历史实验与 artifacts 不能充当新实现的 benchmark、测试 corpus 或正确性证明。

## 目录

| 路径 | 内容 | 地位 |
|---|---|---|
| `legacy-source/` | 重建前的 CMake、C++/MLIR、Python frontend、tools 和 examples | 代码与硬件知识供体 |
| `experiments/` | 旧量化、selector、RVV/IME 与板端材料 | 历史事实 |
| `artifacts/` | 旧 source/object/header/bundle 与测量输出 | 非现行产物 |
| `archive-local/` | ignored 的本机历史材料 | 仅本机追溯 |

可以抽取独立的 parser/CLI 外壳、MLIR/CMake 脚手架、target facts、intrinsic spelling 和资源
公式；不得整体搬入 task/grid 模型、whole-kernel RVV/IME selector、format/family route、
monolithic block-dot catalog、旧 plugin root 或 compatibility/fallback 路径。无法脱离旧
whole-kernel 上下文独立解释的代码只能阅读，不能复用。

最终语义以 [`../doc/WEFT_FINAL_SPEC.md`](../doc/WEFT_FINAL_SPEC.md) 为准。
