# 仓库重建边界

> 本文记录已经完成的仓库裁决与下一实施顺序，不定义 Weft 语言语义。

## 1. 已完成的裁决

本项目采用“同仓库净室重建”：

> 保留当前 Git 仓库和历史，把旧实现整体隔离为不可构建的材料；根目录从零建立新的
> Weft 依赖图。旧代码只可按内容抽取，不可调用、链接或 fallback。

没有选择原地渐进修补，因为旧 `grid/task + whole-kernel selector` 是相反的根语义；也
没有立即另建 Git 仓库，因为 target/toolchain、硬件叶子和实验材料仍需要邻接审计。

## 2. 已执行的边界切换

| 原路径 | 当前路径 | 处置 |
|---|---|---|
| `CMakeLists.txt`、`cmake/` | `materials/legacy-source/` | 旧构建根，只读 |
| `include/`、`lib/` | `materials/legacy-source/` | 旧 C++/MLIR 主干，只读 |
| `python/`、`tools/`、`examples/` | `materials/legacy-source/` | 旧 frontend/CLI/repro，只读 |
| `experiments/` | `materials/experiments/` | 历史实验材料 |
| `artifacts/` | `materials/artifacts/` | 非现行产物 |
| `_attic/` | `materials/archive-local/` | 本机 ignored 历史材料 |
| `build/`、旧 `artifacts/tmp/` | 系统回收站 | 纯派生物，可重建 |

根目录随后创建了全新的 `CMakeLists.txt`。它不查找 LLVM/MLIR、不添加 subdirectory、
不读取 materials，并明确输出当前没有 compiler targets。

## 3. 为什么保留最终规范

`doc/WEFT_FINAL_SPEC.md` 没有删除。其他工程文档已经吸收总体架构，但以下唯一规范信息仍
只在该文件中完整存在：

- Python DSL 的完整类型、API、默认值和约束；
- contraction 与四类 state algebra 的精确语义；
- numerical order、math mode、跨 VLEN 可复现性和 quantization 规则；
- target/tuning/build specification 与 specialization 细节；
- canonical/selected record 全集；
- verifier/作者义务和完整示例。

删除它会把可执行语言合同降成几份架构摘要，因此不满足“思想已全部吸收”的前提。

## 4. 当前根结构

```text
TianchenRV/
├── AGENTS.md
├── README.md
├── CMakeLists.txt                 # 新根；当前零 target
├── doc/                           # 最终规范与现行工程文档
└── materials/
    ├── README.md                  # 抽取索引与禁区
    ├── legacy-source/             # 旧实现，保持原相对路径
    ├── experiments/               # 历史运行材料
    ├── artifacts/                 # 非现行产物
    └── archive-local/             # 本机 ignored，可不存在
```

未来的 `include/`、`lib/`、`python/`、`tools/` 和 `examples/` 只在新实现真正需要时创建。
不使用 `weft-next/`、版本化目录、软链接或双 build root。

## 5. Materials 硬隔离

必须始终满足：

- 根 CMake 不 `add_subdirectory(materials)`，也不 glob 其中源码；
- 新 target 的 include/link/generated-header 路径不指向 materials；
- Python package/import path 不包含 materials；
- 安装、打包和 runtime 不复制或调用旧 binary/source；
- 旧实现不接收 feature，不作为 Scalar fallback；
- 抽取后的新文件可以脱离 materials 独立解释和构建。

精确资产与禁区见 [`../materials/README.md`](../materials/README.md)。

## 6. 新主干的依赖顺序

第一里程碑整体定义见 [`PYTHON_DSL.md`](PYTHON_DSL.md)。它完成全部 Python source surface
及对应 canonical types/ops/verifier，不包含物理 backend。

### 6.1 Canonical contract

先建立 worker-local KernelOp、types、VLA region、logical block、predicate/masked value、
memory/effect 与四类 state semantics。不存在 `task_id/grid_rank` alias。

### 6.2 Reference frontend

抽取 source/AST/generic builder 的机械代码，直接发射 canonical MLIR；不建立第二份长期
typed Python IR 或 verifier。

### 6.3 第二里程碑：Provider 与 selected contract

先建立 typed target profile、primitive provider interface 和 canonical-anchor selected
records。Scalar baseline 先证明接口，随后加入一个 RVV VLA pointwise/memory provider。

### 6.4 Artifact

从 canonical ABI + selected records 单向生成 readable source、relocatable object 和 C
header。只抽取旧 toolchain/packaging 机械代码，不接旧 emitter。

### 6.5 后续能力

第二里程碑的 elementwise backend 纵向链闭环后，再为第一里程碑已经定义的
reduce/scan/summary、logical block/contract 语义补充 providers，并随后加入 IME、AOT
tuning 和外部 Intent bridge。Compiler 不自动发明 source algorithm variants。

## 7. 第二里程碑的首条真实链路

```text
@weft.kernel worker-local elementwise
  -> W.vla(begin, end)
  -> logical predicate + masked load/store
  -> canonical Kernel IR
  -> Scalar provider + RVV provider
  -> Selected Execution IR
  -> source + relocatable object + C header
  -> real RISC-V numerical comparison
```

目标机不可用时如实停在 source/object 边界。不建立 test corpus、coverage 工程或旧行为
兼容矩阵。

## 8. 重新讨论分仓的条件

只有新根能独立构建、materials 零依赖、外部 frontend 只交换 canonical MLIR、provider
API 容纳 Scalar/RVV/extension、普通 ABI 稳定且真实目标链路闭环后，才重新评估是否拆成
独立发布仓库。届时分仓是部署选择，不承担逃离旧架构的职责。
