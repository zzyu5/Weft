# Weft 整体架构

> 本文是 `WEFT_FINAL_SPEC.md` 的工程解释，不是独立规范。语义冲突时以同目录最终规范为准。

## 1. 我们到底在构建什么

Weft 是一个单 worker/hart 的 RISC-V kernel DSL/compiler。它让作者表达局部数值算法，
由编译器把同一份算法语义实现为 Scalar、RVV、IME 或未来扩展的组合。Weft 的目标不是
识别整算子、调度计算图或托管线程池，而是在给定 worker-local 工作范围内忠实且高效地
发射 RISC-V kernel。

“像 Triton”指的是它拥有独立的 kernel 语言、编译边界、target specialization 与 AOT
artifact，而不是照搬 GPU 的 program grid。Weft 不需要一层 SIMT 抽象：外部 runtime
负责把全局工作分给 worker；kernel 内的可伸缩数据并行由 `W.vla` 表达，RVV 的 `vl`、
LMUL、strip-mining 和寄存器分组由 provider 实现。

## 2. 根执行模型

一个 kernel 是普通函数：它接收指针、标量、descriptor 和 compile-time meta-parameter，
执行普通标量控制流，并可进入一维 VLA region。它不知道自己是第几个线程，也没有隐式
launch identity。

```text
external runtime
  ├─ creates workers and decides work slicing
  └─ calls the same AOT kernel with source-defined slice arguments/descriptors
       └─ one worker-local Weft kernel
            ├─ ordinary scalar control
            ├─ W.vla(begin, end)
            ├─ local logical block values
            └─ reduce / scan / summary_fold / sequential carry
```

因此，以下概念不得进入 source 或 canonical IR：

- program grid、program id、task id、grid rank；
- 隐式 hart id 或 runtime thread id；
- 由 kernel 自己创建线程、切分跨 worker 任务；
- source 可观察的 `vl`、strip 次数或某轮 tail 长度。

## 3. 两类一等数据域

### 3.1 VLA iteration region

`W.vla(begin, end)` 表示逻辑区间 `[begin, end)` 恰好覆盖一次。它是一等 structured
region，不是返回向量值的 `arange`。同一 lexical scope 最多有一个 VLA axis；物理实现
可以自由 strip-mine，但 strip 必须对 source 不可观察。

跨整个 VLA domain 的状态只能通过 `reduce`、`scan`、`summary_fold`、effectful memory
或 atomic operation 产生。非 atomic memory effect 必须满足 lane independence；有冲突或
顺序依赖时应使用 atomic、scalar loop 或显式 ordered primitive。不能把任意
loop-carried vector state 偷渡成普通 Python 变量。

### 3.2 Logical block value

Logical block 是某个局部 region 内的结构化 operand/result value。它有显式静态 block
axes，可与当前 VLA axis 组合，但它不是根程序调度单位。Contraction 只能缩并 logical
block 的静态 axes；VLA axis 不能被缩并，但可以作为 contract 的 batch/free axis。

### 3.3 必须分开的分块层次

| 层次 | 谁拥有 | 在哪里表达 | 不能混同为 |
|---|---|---|---|
| algorithmic/cache block | 作者或 source meta-parameter | DSL / canonical IR | program grid |
| logical operand block | canonical 算法语义 | Kernel IR value/axes | register layout |
| register microtile | provider 物理选择 | Selected Execution IR | 算法 block |
| ISA fragment | extension/ISA realization | provider-local selected record / transient lowering | 通用 block type |

这四层可建立映射，但不能共用一个 `tile` 字段，否则算法、寄存器与指令约束会再次耦合。

## 4. Predicate、masked value 与 tail

系统同时存在三个不同对象：

1. logical predicate：算法上哪些元素有效；
2. masked value：值与 validity 一起传播；
3. physical tail：某次 RVV strip 的硬件活跃 lane。

`W.load(..., where=p)` 在没有显式 `other` 时产生 validity-carrying value。后续算术保留
validity，consumer 决定无效元素的含义：store 跳过，reduce 使用 identity，contract
依据 operand predicate 处理。只有在证明 predicate 等价于连续边界时，provider 才可把
它吸收到 AVL；该优化不能改变 canonical 语义。

## 5. 状态代数

状态不是一个统一的“带 carry 循环”类别：

- `reduce`：交换/结合或 ordered reduction，使用 identity 和明确数值顺序；
- `scan`：为每个逻辑位置产生前缀状态；
- `summary_fold`：每个 strip 先形成可合并 summary，再以合法 algebra 合并；
- sequential carry：下一位置真实依赖前一位置，不能任意重分组。

Provider 只能选择某个已声明 algebra 的合法 realization，不能为了向量化把 sequential
carry 重新解释为 reduction，也不能在 emitter 内从循环形状猜 algebra。

## 6. 端到端编译数据流

```text
Weft Python reference frontend       IntentDSL / another frontend
              │                                  │
              └──────────── canonical Kernel IR ┘
                                 │
                    semantic verification + analysis
                                 │
          target facts ──> primitive provider candidate discovery
                                 │
          build spec / tuner ──> legal selection
                                 │
                        Selected Execution IR
                                 │
                 transient provider-local lowering
                                 │
         source / object / static library / header / dispatcher
                                 │
               external runtime passes explicit worker slice
```

持久化承重层只有 canonical、selected 和 artifact。候选、分析索引、EmitC、LLVM IR、
intrinsic helper 与 instruction fragments 都是可重建的瞬态产物。

## 7. 组件责任

| 组件 | 负责 | 明确不负责 |
|---|---|---|
| Python frontend | source capture、DSL type checking、直接构造 canonical IR | JIT、runtime、第二份 typed IR |
| Kernel dialect | 算法语义、types、regions、predicate、state algebra、numerical attributes | layout、LMUL、指令名 |
| Analysis | 从 canonical IR 派生事实与局部上下文 | 持久化第二份算法 |
| Target profile | ISA/ABI/architecture facts，可选 uarch facts | cost model、算法识别 |
| Provider | primitive-local capability、合法配置、成本特征、lowering | whole-kernel route、算法替换 |
| Compiler/selector | 组合候选、解决跨 primitive 约束、写 selected IR | 发明 source 语义 |
| Tuner | 在声明的 source/provider knob 空间做 AOT search | 改变 correctness contract |
| Artifact pipeline | 生成并打包 source/object/header/dispatcher | 重新选择物理计划 |
| Runtime | workers、work slicing、调用与 dispatch | 解释 Weft DSL、内核内 strip |

## 8. 不可退化的架构判据

出现以下任一现象，说明实现又回到了错误骨架：

- 入口先判断“这个 kernel 属于 RVV、IME、GEMM、q4_0 还是某个 route”；
- emitter 从 kernel 名、operand 顺序或 shape 猜算法；
- selected IR 复制 canonical expression、predicate 或 loop body；
- canonical IR 含 `vl`、LMUL、register group、task axis binding；
- 一个 provider 必须拥有整个 kernel 才能工作；
- IntentDSL 类型或 op 进入 Weft core；
- materials 下的旧源码仍在构建、链接或 fallback 路径中。

这些不是暂时的实现欠账，而是必须阻止的架构回退。
