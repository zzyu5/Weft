# 当前仓库状态

> 状态基准：2026-08-10 完成旧树隔离后的工作区。本文陈述当前事实，不定义语言语义。

## 1. 当前结论

仓库根已经是干净的重建边界：

- `doc/WEFT_FINAL_SPEC.md` 保留完整规范；
- `AGENTS.md`、`README.md` 和 `doc/` 是现行工程入口；
- 根 `CMakeLists.txt` 不读取 materials，也暂不定义 compiler target；
- 根目录尚无 `include/`、`lib/`、`python/`、`tools/` 或 `examples/` 新主干；
- 旧源码、实验和 artifacts 均位于 `materials/`；
- 没有 compatibility layer、fallback 或双主干 build。

因此，当前状态不是“旧 compiler 已被重构完成”，而是“错误骨架已经退出活动依赖图，
可以从 canonical contract 开始重建”。

## 2. 当前根目录的承重内容

```text
AGENTS.md                 开发与边界纪律
README.md                 项目入口
CMakeLists.txt            零 target 的新构建根
doc/                      最终规范、工程解释与重建决策
materials/                只读 donor、历史实验与旧产物
```

新实现目录会在对应 contract 落地时创建，不提前放空壳或从 materials 软链接回来。

## 3. Materials 状态

| 路径 | 来源 | 是否活动代码 |
|---|---|---|
| `materials/legacy-source/` | 旧 CMake、include/lib/python/tools/examples | 否，只读 donor |
| `materials/experiments/` | 历史量化格式、selector、RVV/IME、板端运行记录 | 否，历史事实 |
| `materials/artifacts/` | 旧 source/object/header/bundle | 否，不可链接 |
| `materials/archive-local/` | 原本地 `_attic` | 否，本机 ignored 材料 |

详细抽取索引见 [`../materials/README.md`](../materials/README.md)。

## 4. 旧实现为何不能回到根目录

旧实现的根合同与最终规范直接冲突：

| 最终能力 | Donor 实现 | 处置 |
|---|---|---|
| worker-local、无 grid | `grid_rank` + `task_id` | schema 重写，不设 alias |
| 一等 `W.vla(begin,end)` | fixed `arange`/rank block root | 新 structured region |
| logical validity/masked value | 固定 mask + eager `other` | 新类型与传播语义 |
| reduce/scan/summary/sequential carry 分离 | reduce 与 block-carried loop 特判 | 新 state algebra |
| primitive-local providers | whole-kernel RVV/IME selector | 新 provider contract |
| Scalar/RVV/IME 可组合 | 互斥 target route | primitive 级组合 |
| selected 只存物理选择 | task/layout whole-scope plan | selected schema 重建 |

这些不是缺少 feature，而是相反的根模型。把 donor target 重新加入 CMake 会立即破坏这次
边界切换。

## 5. 仍有价值的 donor

可抽取但不可链接的主要资产包括：

- Python source capture、AST location、generic MLIR assembly builder；
- MLIR TableGen/CMake/dialect registry 机械骨架；
- RISC-V/RVV target facts、VLEN/SEW/LMUL/resource 解析；
- RVV intrinsic spelling、masked memory/reduction 叶子；
- IME fragment/resource arithmetic 与单条 asm helper；
- source→RISC-V object、header/object/bundle packaging。

每项精确路径和禁止路径都记录在 materials 索引，不在本文重复。

## 6. 尚未实现

当前新主干还没有：

- worker-local KernelOp 与 canonical types；
- `W.vla`、logical block、masked value、state algebra；
- Python reference frontend；
- target profile/provider/Selected Execution IR；
- Scalar/RVV/IME 新 providers；
- source/object/header artifact pipeline；
- 新语义下的真实 RISC-V repro。

这些能力按“完整 source→canonical 语言闭环”和“最小 executable backend 纵向链”两个
里程碑建立，不能用 materials 中的旧命令输出冒充。

## 7. 第一实施边界

第一里程碑是完整 Python reference frontend + canonical Kernel IR：

1. 完整 core type/annotation 与 worker-local KernelOp；
2. scalar control、`W.vla`、memory/effect、predicate/masked value；
3. logical block、四类 state semantics、contract 与 numerical attributes；
4. Python source 直接生成 canonical MLIR；
5. canonical dialect 独立 parse/print/verify；
6. 最终规范的六个完整示例全部通过 canonical verifier。

详细定义见 [`PYTHON_DSL.md`](PYTHON_DSL.md)。在语言闭环前，不接 selected/provider、
RISC-V emitter、IME、tuning、Intent bridge 或旧量化格式。
