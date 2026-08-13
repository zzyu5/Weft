# 编译器总架构

## 唯一 production 主链

```text
Python Weft DSL / other frontend
                │
                ▼
Canonical worker-local Weft Kernel IR
  ├─ scalar control / pointer / effect
  ├─ VLA iteration regions
  ├─ logical predicates / masked values
  ├─ logical block values
  ├─ reduce / scan / typed summary
  ├─ dot / matmul / lookup / decode / permute
  ├─ source meta-parameters
  └─ linked typed local extension primitives
                │
                ▼
RISC-V target lowering
  Kernel IR + target facts + explicit backend config
  ├─ local capability and legality
  ├─ LMUL / unroll / register microtile
  ├─ packing / fragment / local fusion
  └─ RVV / IME / vendor-extension spelling
                │
                ▼
intrinsic C / necessary inline asm
                │
                ▼
system C compiler
                │
                ▼
relocatable object / static library / C header
                │
                ▼
llama.cpp / ggml / framework / application runtime
  owns threads, work partition and multi-core scheduling
```

Canonical Kernel IR及其已链接 sibling extension dialect是唯一长期编译器表示。
Capability query、legality、resource equation、候选枚举、
物理配置选择和 target-local owner data 可以存在于一次 lowering 调用中，但不形成独立
pipeline stage、持久 IR、可单独输入的 front door 或第二份 authority。

## 单 kernel 编译边界

Weft 的一次编译输入是一份完整的 worker-local kernel，不是计算图、算子节点集合或等待
识别的通用 SSA graph。作者或外部 frontend 已经决定 outer control、blocking、staging、
state、memory effect 和 structured primitive；Weft 只为这些结构生成 RISC-V realization。

因此 Weft core 不执行：

- graph import、partition 或跨算子 fusion；
- 从普通 multiply/add graph 发现 GEMM、attention 或量化算子；
- 根据 kernel 名、算子名、模型格式或整体 shape 选择整段实现；
- 创建线程、分配 worker 或把外部 runtime work partition 变成语言语义。

IntentDSL 或其他上游若使用 Weft，必须在仓库外生成一份完整 canonical Weft Kernel IR。
Weft 不回读上游 graph，也不从 target lowering 反向补算法骨架。

## Target lowering 的决策单位

Target lowering 面对 kernel 内可组合的 canonical anchor：scalar control、VLA、memory、
reduce、scan、summary、dot、matmul、decode、lookup 和 extension primitive。一个物理实现的输入
只能是：

```text
canonical primitive
+ typed operands and effects
+ logical axes / local use relation
+ target facts
+ explicit backend config
```

授权边界固定为：普通scalar loop保持作者写下的有序traversal，不能自动变成VLA；只有显式
`weft_kernel.vla`授权SIMD logical axis；只有显式`weft_kernel.dot`或`weft_kernel.matmul`
授权局部乘加域重组。Reduce、scan、typed summary与sequential carry是不同语义，不能按use graph互换。
Blocking、staging、persistent packing、outer traversal和算法variant只由source/Kernel IR
定义，target-local fusion不得重建或替换这些关系。

局部 lowering 可以吸收相邻的纯 decode、cast、scale、packing producer 或 consumer，也可
在不改变 source loop/state boundary 的前提下联合安排相邻 primitive。选择 authority必须
锚定 canonical primitive/interface及其block axis、typed operand、access/use projection，
不能把整个loop nest归类成一个`KernelKind`，也不得用完整 kernel shape或symbol选择模板。
无法合法生成时直接报unsupported；不存在 legacy、GGML 或旧 emitter fallback。普通 scalar
control的 C lowering是正式 realization，不是 fallback。

每个physical decision只有一个producer。Analysis/selection可以读取typed operands、axis、
effect、局部use relation和target facts来产生该decision；intrinsic-C/asm emitter只读取它，
不得再按use-count、相邻op数量或完整kernel source重新选择物理形态。

## 定位一句话

> **Weft 是一门面向单个 RISC-V worker/hart 的 AOT kernel DSL：作者用普通控制流、一等 VLA iteration region、logical block、显式 predicate/state algebra 与 structured compute primitive 编写完整 worker-local 算法；一次 target lowering 直接从唯一 Kernel IR 与 target/config facts 生成动态 `vl`、LMUL、register microtile、memory、RVV 与扩展实现，再交给系统 C compiler 形成普通 object。**
