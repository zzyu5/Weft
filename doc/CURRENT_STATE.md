# 当前仓库状态

> 状态基准：2026-08-10 第一里程碑实现后的工作区。本文只陈述实现与验证事实，不定义
> Weft 语言语义。

## 1. 结论

新的活动根已经完成 Python source 到 canonical Kernel MLIR 的语言闭环。它不是旧编译器
复活，也不是 RISC-V backend 已完成：

- Python frontend 直接读取 `@weft.kernel` source AST 并生成 generic canonical MLIR；
- `weft_kernel` dialect 拥有 worker-local ABI、scalar control、VLA、logical block、masked
  value、memory/effect、state algebra、contract 和 numerical attributes；
- `weft_ext` sibling dialect 拥有规范示例中的 `block_scaled_contract` 局部语义 primitive；
- `weft-opt` 独立注册两个 dialect，并负责 parse/print/verify；
- 六个规范 source 与新增完整 online-softmax baseline 均已完成
  source→canonical→独立 parse/verify；
- 十个固定 baseline 均已有可独立编译运行的 C reference/runtime，其中六个量化项保留
  GGML packed block ABI 与通用标量语义；
- 活动构建与 import 路径不读取 `materials/` 或 IntentDSL。

目前尚无 target/provider/selected/artifact 路径，因而没有新的 RISC-V source、object、header
或目标机数值结果。

## 2. 活动根结构

```text
AGENTS.md
CMakeLists.txt
README.md
doc/
include/Weft/Dialect/
  Kernel/IR/                 canonical core dialect + types/ops
  Extension/IR/              typed sibling semantic extension
lib/Dialect/
python/weft/
  api/                       AOT kernel definition
  language/                  public types, annotations and intrinsics
  frontend/                  source lowering + generic MLIR assembly
tools/weft-opt/              canonical parse/print/verify driver
source/weft/weft/            normative source + active operator corpus
source/c/                    adjacent reference/runtime source
examples/repro/              manual repro entrypoints
materials/                   inactive donor/history/artifacts
```

`build/` 是 ignored 的可重建目录，不是持久表示或 artifact authority。

## 3. 已实现的 source/canonical 能力

| 领域 | 当前实现 |
|---|---|
| Entry/ABI | `@weft.kernel`、scalar/index、qualified pointer、`constexpr` |
| Scalar control | Python scalar `if`、受限 `while`、`W.range`、SSA carry、helper inlining |
| VLA | `W.vla(begin,end)`、单一 lexical VLA axis、escape/mutation checks |
| Memory/validity | load/store/prefetch/atomic/fence、masked value、valid/fill/select |
| Logical block | block_axis/full/zeros、expand/broadcast/reshape/transpose/slicing sugar |
| State algebra | reduce、scan、summary_fold；与 sequential carry 分离 |
| Structured compute | contract、dot、permute、lookup、decode、widen、narrow |
| Numerics | dtype、acc/out dtype、order、math、rounding、saturation、exceptional policy |
| Extension | `weft_ext.block_scaled_contract` typed sibling op |

Python 中的 AST、environment、value/type spelling 和 generic Operation/Region node 都只在一次
lowering 内存在；它们不持久化、不拥有独立 op schema、verifier 或优化 pipeline。Canonical
TableGen dialect 是唯一算法 schema 与最终 legality authority。

## 4. 已执行的验证

本机使用 LLVM/MLIR 20.1.8 完成：

1. CMake configure；
2. TableGen 生成 Kernel/Extension dialect；
3. C++ dialect 与 `weft-opt` 编译链接；
4. Python package `compileall`；
5. `source/weft/weft/` 下六个规范 source 与完整 online-softmax baseline 分别由
   `python -m weft` 生成 MLIR；
6. 每份结果由新构建的 `weft-opt` parse/print/verify；
7. `examples/repro/source/run_c_baselines.sh` 用系统 C 编译器分别编译并运行十份
   `reference.c + runtime.c`，十项均输出 `PASS`。

验证没有调用 materials 下的旧 binary，也没有建立 pytest/lit/coverage 或兼容矩阵。当前
结果证明 Weft source/canonical 边界和独立 C baseline 可运行；C reference 的通过不证明
Weft 已经生成它们，也不证明 RISC-V lowering、性能或目标机数值正确性。

## 5. Materials 状态

| 路径 | 作用 | 活动依赖 |
|---|---|---|
| `materials/legacy-source/` | 旧 CMake、源码和工具的只读 donor | 无 |
| `materials/experiments/` | 历史 RVV/IME/量化实验 | 无 |
| `materials/artifacts/` | 旧 source/object/header/bundle | 无 |
| `materials/archive-local/` | 本机 ignored 历史材料 | 无 |

精确抽取索引和禁区见 [`../materials/README.md`](../materials/README.md)。

## 6. 尚未实现

- typed RISC-V target profile；
- primitive provider interface 与 candidate discovery；
- Selected Execution IR；
- Scalar、RVV、IME realization providers；
- canonical+selected 到 source/object/header/static library 的 artifact pipeline；
- build specification、specialization/tuning 与 dispatcher；
- IntentDSL/其他 frontend 到 canonical Weft IR 的外部 bridge；
- 新语义主干上的真实 RISC-V 数值与性能结果。

十项 baseline 及其 source readiness 以 [`../source/README.md`](../source/README.md) 为准。
六项量化格式目前只有 GGML-derived C reference；在 canonical packed decode/quant contract
语义完成前，不创建虚假的 Weft source。

下一步仍应是最终规范定义的第二里程碑最小 executable vertical slice，而不是回填旧 emitter
或按 kernel/operator 名增加 whole-kernel route。
