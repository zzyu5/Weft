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

最终语义以 [`../doc/index.md`](../doc/index.md) 及其链接的模块为准。

## 知识所有权

历史实现中经常把算法结构、物理选择和指令拼写写在同一个 emitter 中。抽取前必须按它改变的
对象重新归属，不能沿用旧文件或旧 pass 的所有权。

### DSL 算法知识

改变 outer loop、blocking、staging、persistent organization、跨阶段 reuse 或 accumulator
生命周期的内容属于 source/Kernel IR：

- `legacy-source/examples/repro_gemm.py` 中的 `BM/BN/BK` loop、K staging 和 accumulator 组织；
- `legacy-source/lib/Conversion/RVV/RVVToEmitCBlockQuantLinear.cpp` 的 block/column traversal、
  weight decode hoist 与跨 column reuse。

这些文件中的 grid、task identity、whole-kernel traversal 和 persistent layout只能帮助作者形成
新的 worker-local DSL source，target lowering不得从普通SSA或kernel名重新发明它们。

### Compiler 物理知识

不改变source-visible算法，只改变当前显式VLA或local primitive内部physical realization的内容
属于一次target lowering：

- `legacy-source/lib/Transforms/SelectRISCvExecution.cpp` 的SEW/LMUL合法性和register-footprint公式；
- `legacy-source/lib/Conversion/RVV/RVVToEmitCBlockQuantLinear.cpp` 的narrow/wide LMUL关系、
  per-column accumulator、local decode packing和widen/reduce instruction organization。

这些关系可以重建为typed facts、候选、handoff、resource budget和local schedule decision；旧
selector、problem/variant、format route和whole-region owner不能一并复用。

### Local leaf知识

只决定intrinsic或inline-asm名字、operand constraint、clobber和局部ABI的内容属于emitter leaf：

- `legacy-source/lib/Target/RVV/SelectedExecutionRVVSource.cpp` 的RVV suffix、masked/unit/strided
  intrinsic spelling；
- `legacy-source/lib/Target/IME/SelectedExecutionIMESource.cpp` 的`vsetvli`、`vmadot`、fragment
  load/store、constraint与clobber spelling；
- `experiments/cold-start/` 下generated kernel只可核对局部拼写，不提供当前算法或性能authority。

Leaf必须实现一个明确local semantic primitive，不能拥有outer traversal、persistent storage、
kernel ABI或调用旧runtime。
