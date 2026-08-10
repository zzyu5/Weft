# 编译器总架构

## 27. 最终架构总图

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
RISC-V target profile
  architectural facts + optional microarchitecture hints
                │
                ▼
Primitive-local realization providers
  capability + parameter family + legality + resource + lowering
                │
                ▼
Compiler-derived legal execution space
                │
        build-time AOT tuner
                │
                ▼
Selected Execution IR
  vl mechanics / LMUL / memory plan / microtile / fragment / strategy
                │
                ▼
Mechanical scalar + RVV + extension lowering
                │
                ▼
object / static library / C header / optional AOT dispatcher
                │
                ▼
llama.cpp / ggml / framework / application runtime
  owns threads, work partition and multi-core scheduling
```

### 27.1 单 kernel 编译边界

Weft 的一次编译输入是一份完整的 worker-local kernel，不是计算图、算子节点集合或等待
识别的通用 SSA graph。作者或外部 frontend 已经决定 outer control、blocking、staging、
state、memory effect 和 structured primitive；Weft 只为这些 canonical 结构选择并生成
RISC-V realization。

因此 Weft core 不执行：

- graph import、partition 或跨算子 fusion；
- 从普通 multiply/add graph 发现 GEMM、attention 或量化算子；
- 根据 kernel 名、算子名、模型格式或整体 shape 选择整段实现；
- 创建线程、分配 worker 或把外部 runtime work partition 变成语言语义。

IntentDSL 或其他上游若使用 Weft，必须在仓库外生成一份完整 canonical Weft Kernel IR。
Weft 不回读上游 graph，也不从 target lowering 反向补算法骨架。

### 27.2 逐局部结构构造 realization

Compiler 面对的是 kernel 内多个可组合的 canonical anchor：scalar control、VLA、memory、
reduce、scan、summary、contract、decode、lookup 和 extension primitive。合法执行空间从这些
局部结构及其 logical axes 分别构造，再联合检查资源与依赖；不存在先把整个 kernel 分类，
再进入互斥 lowering 分支的阶段。具体决策单位与表示边界见
[Selected Execution IR](selected-ir.md)。

---

## 28. 定位一句话

> **Weft 是一门面向单个 RISC-V worker/hart 的 AOT kernel DSL：作者用普通控制流、一等 VLA iteration region、logical block、显式 predicate/state algebra 与 structured compute primitive 编写完整 worker-local 算法；compiler 从 target facts 与 primitive provider 中构造合法的动态 `vl`、LMUL、register microtile、memory 与扩展 realization，构建期 tuner 选择性能点，多核调度由外部 runtime 负责。**
