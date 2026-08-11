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
  ├─ reduce / scan / summary fold
  ├─ contract / lookup / decode / extension primitives
  └─ source meta-parameters
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

Kernel IR 是唯一长期编译器表示。Capability query、legality、resource equation、候选枚举、
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
reduce、scan、summary、contract、decode、lookup 和 extension primitive。一个物理实现的输入
只能是：

```text
canonical primitive
+ typed operands and effects
+ logical axes / local use relation
+ target facts
+ explicit backend config
```

局部 lowering 可以吸收相邻的纯 decode、cast、scale、packing producer 或 consumer，但不得
用完整 kernel shape 或 symbol 选择模板。无法合法生成时直接报 unsupported；不存在 legacy、
scalar、GGML 或旧 emitter fallback。

## 定位一句话

> **Weft 是一门面向单个 RISC-V worker/hart 的 AOT kernel DSL：作者用普通控制流、一等 VLA iteration region、logical block、显式 predicate/state algebra 与 structured compute primitive 编写完整 worker-local 算法；一次 target lowering 直接从唯一 Kernel IR 与 target/config facts 生成动态 `vl`、LMUL、register microtile、memory、RVV 与扩展实现，再交给系统 C compiler 形成普通 object。**
