# 仓库重建决策

> 本文是工程裁决，不是语言规范。它回答“如何从当前仓库到达最终规范”，并明确区分
> 本轮已做的文档工作与后续尚未执行的源码搬迁。

## 1. 结论

采用：

> **保留当前 Git 仓库，但不在当前语义主干上渐进修改；先把旧实现整体隔离进
> `materials/legacy-source/`，再在仓库根目录建立一套全新的 Weft 源码骨架与独立依赖图。
> 旧实现只作为代码内容和硬件知识供体，不作为可调用的 legacy library。**

这可以简写为“同仓库净室重建”。它在代码层面是新建，不是给旧骨架修修补补；在版本
管理层面仍沿用同一仓库，保留提交历史、实验材料邻接关系和逐项抽取的可审计性。

本轮只重写文档和 `AGENTS.md`。**没有创建 `materials/`，没有移动源码，没有改变 CMake，
也没有开始新实现。** 源码边界切换应作为下一次独立、可审阅的操作完成。

## 2. 为什么不能原地渐进修改

现有主干的问题不是目录命名或少几个 op，而是根语义相反：

| 最终规范 | 当前主干 |
|---|---|
| worker-local、无 grid identity | `grid_rank` + `task_id` |
| 一等 `W.vla(begin,end)` region | fixed `arange`/rank block root |
| logical validity/masked value | load/store 显式 mask + eager `other` |
| reduce/scan/summary/sequential carry 分开 | reduction 与 block-carried loop 的特判 |
| primitive-local providers 可组合 | whole-kernel RVV/IME selector 与 emitter |
| extension 只实现局部 semantic interface | format/route/whole-owner construction |
| selected 只存物理选择 | task-axis/layout-group 主导的整体 plan |

如果在原目录中逐步改，会长期同时存在两个 KernelOp 合同、两个 selector 假设和两套
emitter 数据源。为了“先让它编过”势必出现 adapter、fallback、feature flag 或旧 schema
兼容，最终仍然由旧依赖图决定新架构。这正是用户要停止的混乱来源。

## 3. 为什么现在不立即创建另一个 Git 仓库

另起仓库表面最干净，但当前阶段成本更高：

- canonical schema、provider interface 与第一条 Scalar/RVV slice 仍需一起收敛；现在拆仓
  容易把未经验证的接口当成稳定跨仓协议；
- 许多可复用资产是文件内局部 helper，而不是可发布 library；同仓库更适合逐段审计抽取；
- target/toolchain、IME instruction leaf、artifact packaging 和历史 repro 仍是重要事实来源；
- 分仓会迫使我们现在决定版本、发布和依赖管理，而这些不是语言设计问题。

等新根骨架能独立构建、没有任何 `materials/` 依赖、API/ABI 边界稳定后，是否拆成独立
发布仓库才是有事实基础的部署选择。现在做它只会增加迁移协议，不会改善核心架构。

## 4. 为什么不是在旧树旁边放一个 `weft-next/`

临时并排的第二源码树会让根 CMake、include path、Python import 与工具名继续含糊，且
“next”很快成为永久版本分支。最终目录本来就应叫 `include/`、`lib/`、`python/`、`tools/`；
因此下一次源码操作应先切断旧根，再建立最终根，而不是长期维护两个可构建产品。

## 5. 目标仓库形态

源码切换后的目标结构如下。具体子目录可随实现细化，但依赖方向不得变化：

```text
TianchenRV/
├── WEFT_FINAL_SPEC.md
├── AGENTS.md
├── README.md
├── CMakeLists.txt                    # 只构建新主干
├── cmake/
├── doc/
├── include/Weft/
│   ├── Dialect/Kernel/               # canonical language IR
│   ├── Dialect/Execution/            # selected records + interfaces
│   ├── Analysis/                     # 可重算事实
│   ├── Target/                       # typed target profiles
│   ├── Provider/                     # provider contracts/registry
│   ├── Compiler/                     # selection + pipeline
│   └── Artifact/                     # ABI/export contracts
├── lib/
│   ├── Dialect/Kernel/
│   ├── Dialect/Execution/
│   ├── Analysis/
│   ├── Target/
│   ├── Provider/
│   │   ├── Scalar/
│   │   ├── RVV/
│   │   └── IME/
│   ├── Compiler/
│   └── Artifact/
├── python/weft/
│   ├── api/
│   ├── language/
│   └── frontend/
├── tools/
│   ├── weft-compile/
│   └── weft-opt/
├── examples/                         # 少量真实 AOT repro source
└── materials/
    ├── README.md                     # 隔离规则与来源说明
    ├── legacy-source/                # 当前旧骨架，保持原相对路径供查阅
    ├── experiments/                  # 需要保留的历史运行材料
    └── artifacts/                    # 需要保留的非现行产物
```

不预建 `test/`/`tests/`。真实 repro 只在 `examples/` 保存最小 source 和运行说明，不累计
coverage、边界测试矩阵或旧行为兼容测试。

## 6. `materials/` 是硬隔离区

`materials/` 不是 monorepo 中的 legacy component，而是只读研究材料。必须同时满足：

- 根 CMake 不 `add_subdirectory(materials)`，也不 glob 其中源码；
- 新 target 的 include path、link libraries、generated headers 不指向 materials；
- Python package/import path 不包含 materials；
- 安装、打包和 artifact export 不复制 materials；
- runtime、compiler 和 tool 不存在调用旧 binary/source 的 fallback；
- 新实现不能用 wrapper 把旧 API 伪装成 provider；
- 需要一段旧逻辑时，把最小知识重新落入新文件，并按新 contract 命名、验证、维护。

`materials/legacy-source/` 保存的是当前实现供定位和对照，不继续接收 feature。历史设计
文档不需要再复制进去；Git 已足够追溯，被最终规范取代的文档应从工作树删除。

## 7. 代码复用分级

### 7.1 可直接抽取的机械基础设施

这些资产基本不携带错误根模型，但仍需在新依赖图中逐文件确认：

- `python/weft/frontend/ir.py` 的单向 generic MLIR assembly builder；
- `python/weft/frontend/source.py` 的 source capture、AST location 和 static binding；
- MLIR dialect 的 CMake/TableGen/generated-include 注册骨架；
- dialect registry 与基础 CLI 的 parse/output 外壳；
- RISC-V target triple/toolchain 驱动、temporary source 到 relocatable object 的 packaging；
- 通用 header/object/bundle 写出、symbol 与 no-overwrite 机械逻辑。

“可直接抽取”仍然是复制到新路径并切断旧 include，不是让新 target 链接 donor target。

### 7.2 可抽取知识、必须重写接口的内容

- pointer/block/constexpr type helper、broadcast/shape verifier 与 layout algebra；
- RVV VLEN/SEW/LMUL/register facts 的解析与 typed materialization；
- RVV intrinsic type spelling、`vsetvl`、masked load/store、reduction instruction 的叶子知识；
- IME 单条 asm leaf、fragment/resource arithmetic、typed schedule field 的验证知识；
- canonical-anchor 到 selected record 的查找思想；
- artifact handoff 的 fail-closed 验证思想。

这些逻辑当前通常嵌在 whole-family、route、layout-group 或 old plugin object graph 中，必须
拆到 primitive provider、target facts 或 artifact component 下，不能整体搬文件。

### 7.3 绝不能继承的语义骨架

- `grid_rank`、`task_id`、task-axis ABI binding、fixed `arange` 根模型；
- 当前 `FrontendCompiler` 的 grid/task DSL contract；
- whole-kernel RVV/IME selector 与 rank/group source emitter；
- canonical-problem → variant/route → family construction → format body 路径；
- q4/q8/模型格式名驱动的整 kernel emission；
- 要求 entry 所有节点被一个 IME contraction site 覆盖的 ownership；
- selected IR 中复制算法或由 emitter 重猜 source facts 的字段；
- 任何兼容 adapter、旧 emitter fallback 或双主干 dispatch。

这些代码可以留在 materials 供查硬件细节，但不能成为新架构的父类、library 或入口。

## 8. 下一次源码切换的原子边界

以下是依赖顺序，不是本轮已经执行的结果：

1. **盘点工作区。** 确认 tracked/untracked/其他代理改动，列出将移动的旧源码与需保留
   材料；不得覆盖用户改动。
2. **一次性隔离旧根。** 把现有 CMake/include/lib/python/tools/test 等旧实现按原相对
   结构移入 `materials/legacy-source/`，写明它不再构建。
3. **建立空的新依赖图。** 根 CMake 只认识新目录；先让一个空 dialect/tool skeleton
   独立配置，机械检查不存在 materials include/link/import。
4. **先立 canonical contract。** 实现 worker-local KernelOp、VLA region、logical block、
   masked value 与四类 state semantics，不保留 grid/task alias。
5. **接参考 frontend。** 抽取 source/AST/generic builder，直接发射 canonical MLIR。
6. **立 provider/selected contract。** 先用 Scalar baseline 跑通接口，再添加 RVV VLA
   pointwise/memory/reduce，之后才接 contract/IME。
7. **抽取 artifact 机械层。** 从新 canonical ABI + selected records 生成 source/object/header；
   不接旧 whole-kernel emitter。
8. **跑一条真实链路。** 从最小 DSL kernel 到真实 RISC-V artifact/运行；不能运行时如实
   停在 artifact 边界。

第一步和第二步应与建立新根骨架在同一可审阅改动中完成，避免出现“旧根已经搬走但新根
不可配置”或“新旧根同时可构建”的中间常态。

## 9. 新骨架的首个纵向切片

首个实现不应从 GEMM/IME 开始，因为那会先固化最复杂的 provider。更可靠的承重切片是：

```text
@weft.kernel worker-local elementwise
  -> W.vla(begin, end)
  -> logical predicate + masked load/store
  -> canonical Kernel IR
  -> Scalar provider and one RVV provider
  -> selected execution
  -> source + relocatable object + C header
```

这个切片会同时验证根程序模型、VLA 不可观察性、masked-value contract、provider 粒度、
selected/canonical 边界和普通 C ABI。它成立后，再加入 reduce/scan/summary、logical block
contract 和 IME，能显著降低再次长出 whole-kernel 模板的风险。

## 10. 何时再讨论独立仓库

只有满足以下事实后才重新评估拆仓：

- 新根从 clean checkout 独立配置与构建；
- 构建/安装图对 `materials/` 为零引用；
- Python frontend 与外部 frontend 都只交换 canonical MLIR；
- provider API 能容纳至少 Scalar、RVV 和一个 extension provider；
- object/header ABI 稳定到可以作为发布边界；
- 一条真实目标 repro 已经闭环。

届时分仓是发布与维护选择，不再承担“逃离旧架构”的职责。
