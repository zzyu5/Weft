# Weft 历史材料与代码供体

`materials/` 保存重建前的实现、实验与产物。它们用于查找硬件事实、机械代码和历史
证据，不属于现行 Weft 编译器，也不是兼容层。

## 硬边界

- 根 CMake、include path、link graph、Python import、安装和 runtime 都不得进入本目录。
- 新代码不得调用、包装或 fallback 到这里的 target、library、script 或 emitter。
- “复用”只允许把最小代码内容或硬件知识复制到新文件，并按最终规范重新命名和建模。
- 旧文件不再接收 feature；发现 bug 也应在新实现中修复，除非只是补充材料说明。
- 历史实验与 artifacts 不能充当新实现的测试 corpus 或当前正确性证明。

## 目录

| 路径 | 内容 | 地位 |
|---|---|---|
| `legacy-source/` | 重建前的 CMake、C++/MLIR、Python frontend、tools 和旧 examples | 代码与硬件知识供体 |
| `experiments/` | 旧量化格式、selector、RVV/IME 和板端运行材料 | 历史事实，不是现行 benchmark |
| `artifacts/` | 旧 source/object/header/bundle 与测量输出 | 非现行产物，不可链接 |
| `archive-local/` | 原本地 `_attic`，保持 ignored，fresh clone 不保证存在 | 仅本机历史追溯 |

最终语言与编译器语义仍以 [`doc/WEFT_FINAL_SPEC.md`](../doc/WEFT_FINAL_SPEC.md) 为准；
工程入口在 [`doc/README.md`](../doc/README.md)。

## 第一类：可直接抽取的机械基础

下列内容基本不拥有旧算法模型。抽取时仍应复制到新路径，不能链接 donor target。

| 供体路径 | 有用内容 | 抽取要求 |
|---|---|---|
| `legacy-source/python/weft/frontend/source.py` | `SourceUnit`、AST/source capture、location、closure/global binding | 保持瞬态，只输出 canonical IR |
| `legacy-source/python/weft/frontend/ir.py` | `Value`、`Region`、`Operation`、`IRBuilder` 和 generic MLIR renderer | 只做单向 assembly，不拥有平行 schema/verifier |
| `legacy-source/python/weft/api/definitions.py` | source/signature capture 与 AOT-only call guard | 删除 `grid_rank` 与旧 KernelDefinition 合同 |
| `legacy-source/python/weft/language/annotations.py` | `PtrSpec`、`ConstexprSpec` annotation 机制 | 对齐最终 pointer qualifier 与 constexpr 规则 |
| `legacy-source/lib/InitWeftDialects.cpp` | MLIR dialect registry 的机械入口 | 只注册新 canonical/selected/provider dialects |
| `legacy-source/include/Weft/Dialect/*/CMakeLists.txt` | TableGen/generated-include/CMake 组织方式 | 复制脚手架，不复制旧 op schema |
| `legacy-source/tools/weft-compile/weft-compile.cpp` | parse module、诊断和 artifact 输出分支的 CLI 外壳 | 删除 RVV/IME whole-target dispatch |
| `legacy-source/include/Weft/Target/TargetArtifactExport.h` | exporter registry、bundle/header/object 出口形态 | 去除 route/family/candidate 旧身份 |
| `legacy-source/lib/Target/TargetArtifactExport.cpp` | symbol、bundle 写出与 fail-closed handoff 的机械逻辑 | 入口只能是 canonical + selected + target facts |
| `legacy-source/lib/Target/RVV/RVVTargetSupportBundle.cpp` | clang RISC-V source→relocatable object packaging | 与旧 exact-body route 解耦 |

## 第二类：只抽取硬件知识，接口必须重写

| 供体路径 | 可用知识 | 新归属 |
|---|---|---|
| `legacy-source/include/Weft/Target/RISCVTargetProfile.h`、`legacy-source/lib/Target/RISCVTargetProfile.cpp` | typed RISC-V target fact materialization | `Target/` architectural profile |
| `legacy-source/include/Weft/Plugin/RVV/RVVCapabilityProfile.h`、`legacy-source/lib/Plugin/RVV/RVVCapabilityProfile.cpp` | VLEN、SEW、LMUL、vector register/resource 推导 | RVV provider capability/legality |
| `legacy-source/lib/Dialect/Kernel/IR/KernelDialect.cpp` | broadcast、shape、extent provenance helper | 新 canonical verifier，逐函数抽取 |
| `legacy-source/lib/Dialect/Layout/IR/LayoutDialect.cpp` | layout projection/insertion algebra | selected/layout analysis，不进入 canonical 语义 |
| `legacy-source/include/Weft/Dialect/RVV/IR/RVVOps.td` | `vl` token、tail/mask policy、LMUL vector type 等硬件事实 | RVV provider-local transient/selected IR |
| `legacy-source/lib/Target/RVV/SelectedExecutionRVVSource.cpp` | LMUL/type spelling、`vsetvl`、masked memory、reduction intrinsic 叶子 | 分拆到 VLA/memory/reduce providers |
| `legacy-source/include/Weft/Dialect/IMEExecution/IR/IMEExecutionOps.td` | IME fragment、4×4×8 config 与 accumulator facts | IME contract selected record |
| `legacy-source/lib/Plugin/IME/IMEBackendEmissionDriver.cpp` | `vmadot` signedness mnemonic、单条 asm helper、register-resident MAC leaf | IME primitive-local lowering |
| `legacy-source/lib/Plugin/IME/IMEFormulaConstruction.cpp` | fragment/register/resource arithmetic | IME capability 与 legality equations |
| `legacy-source/include/Weft/Dialect/Execution/IR/ExecutionOps.td` | canonical anchor 与 selected record 分层思想 | 新 Selected Execution IR，字段重建 |

这些文件大多把有用叶子包在错误的 whole-kernel orchestration 中。不得整体搬文件；应先
识别一个 primitive 的输入/输出和 target facts，再抽取最小函数、常量或公式。

## 第三类：绝不能进入新主干的旧语义

| 禁区 | 代表路径 | 原因 |
|---|---|---|
| task grid / launch identity | `legacy-source/include/Weft/Dialect/Kernel/IR/KernelOps.td` 中的 `grid_rank`、`TaskIdOp` | 最终模型是 worker-local、无 program grid |
| 旧 Python 根合同 | `legacy-source/python/weft/frontend/compiler.py`、`legacy-source/python/weft/language/builtins.py` | 发射 `task_id/arange`，没有一等 `W.vla`/masked value/state algebra |
| whole-kernel RVV selector | `legacy-source/lib/Transforms/SelectRISCvExecution.cpp` | 按 rank/layout scope 接管整个 kernel |
| whole-kernel IME selector | `legacy-source/lib/Transforms/SelectIMEExecution.cpp` | 要求整个 entry 被固定 contraction site 覆盖 |
| rank/group emitter | `legacy-source/lib/Target/RVV/SelectedExecutionRVVSource.cpp`、`legacy-source/lib/Target/IME/SelectedExecutionIMESource.cpp` 的 orchestration | 从 task bindings/fixed arange 重建 source skeleton |
| format/family front door | `legacy-source/lib/Plugin/RVV/FrontDoor/`、`legacy-source/lib/Plugin/RVV/Construction/` | 按 q-format/problem/route 选择整条 lowering |
| monolithic block-dot catalog | `legacy-source/include/Weft/Plugin/RVV/RVVMonolithicBlockDotFamily.h` | 把模型格式和 whole-kernel variant 当核心 schema |
| old plugin root | `legacy-source/lib/Plugin/RVV/RVVExtensionPlugin.cpp`、`legacy-source/lib/Plugin/IME/IMEExtensionPlugin.cpp` | 聚合 whole-family source front doors 与格式路由 |
| compatibility/fallback | 任何从新代码调用 `materials/legacy-source` 的 adapter | 会恢复双主干与第二份 authority |

## 抽取流程

每次复用只做一个可审阅的小单元：

1. 先从 `doc/WEFT_FINAL_SPEC.md` 确定新组件的语义所有权；
2. 定位 donor 中一个机械 helper、硬件表或资源公式；
3. 在新根创建独立文件，使用新 namespace、types 和 canonical anchors；
4. 删除对旧 route、problem、format、task/grid、hash/provenance object graph 的依赖；
5. 确认新 target 的 include/link/import 图不经过 `materials/`；
6. 只用对应的新 DSL→canonical→selected→artifact 真实链路验证。

如果无法把一段代码从旧 whole-kernel 上下文中独立解释，它就不是可复用单元，只能作为
阅读材料。
