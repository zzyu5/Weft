# Provider 与物理选择

> 本文描述 provider/selector 的工程协议；所有语义与权限边界以最终规范第 15–20 节为准。

## 1. Provider 的粒度

Provider 是某类 semantic primitive 的参数化物理实现族。它匹配：

```text
primitive interface + typed operands/results + canonical local context + target facts
```

它绝不匹配：

```text
kernel/operator/model/format name + route string
```

因此，一个 kernel 可以同时使用 scalar control lowering、RVV pointwise/memory provider、
RVV reduce provider、IME contract provider 和 vendor lookup provider。系统里不存在“把整个
kernel 交给 IME backend”这类所有权。

## 2. Provider 必须提供的能力面

具体 C++/MLIR API 可在实现时收敛，但职责面必须完整：

| 职责 | 含义 |
|---|---|
| semantic interface | provider 实现哪一类 canonical primitive |
| capability predicate | target/element type/shape relation 是否受支持 |
| parameter space | LMUL、microtile、repeat、fragment、packing 等可选物理参数 |
| legality constraints | register、dtype、mask、memory、fragment 与 ABI 约束 |
| resource equations | 寄存器、scratch、alignment 等可计算需求 |
| selected schema | 只保存本次选中的 provider-local 决策 |
| lowering/emission | 从 canonical anchor + selected record 单向生成实现 |
| local fusion envelope | 可选；声明哪些邻接操作可在不改语义时局部吸收 |

能力查询和 legality 必须是 typed、fail-closed 的。字符串可以用于诊断或稳定 provider
identity，但不能作为 route 重新推导算法。

## 3. Core primitive interfaces

Core 至少要能查询以下 provider interface families：

- VLA pointwise；
- scalar/vector memory；
- reduce、scan、summary fold；
- contract；
- permute、gather、table lookup；
- widen、narrow、convert；
- decode/dequantize；
- math primitive；
- atomic/fence。

一个扩展只注册真正加速的局部接口。其余 primitive 继续由 Scalar/RVV baseline 实现，
而不是因为 kernel 中出现扩展就把所有节点切到扩展专用 emitter。

## 4. Baseline 与扩展

### 4.1 Scalar baseline

Scalar lowering 是语义基线，也是 target 没有 vector extension 时的合法 realization。
它必须直接理解 canonical VLA/predicate/state algebra，而不是调用旧 whole-kernel C emitter。
是否选 Scalar 由 selection 明确记录；unsupported extension semantic op 不能被偷偷标量化。

### 4.2 RVV providers

RVV provider 负责把 VLA 和局部 primitive 映射到动态 `vl`、SEW/LMUL、mask/tail policy、
intrinsic/instruction family、register grouping 和局部 unroll。它可以证明连续 predicate 被
AVL 吸收，但必须保持 logical validity。RVV emitter 不得要求 canonical root 先存在 fixed
`arange` 或 rank-specific layout group。

### 4.3 IME 与其他 matrix extensions

如果 IME 只是以相同 observable semantics 实现 `W.contract`，它是 contract provider，
可选择 fragment、packing、microtile 与指令 leaf。若它需要 block-scaled accumulation、
特殊 saturation 或不同 output relation，则先定义局部 canonical extension primitive，再为
该 primitive 注册 provider。

IME provider 不能生成 source 中不存在的 GEMM outer loops，不能以 q4/q8 格式名接管
kernel，也不能要求 entry 内所有操作都属于同一个 contraction site。

## 5. Target facts

每次编译绑定一个 typed RISC-V target profile，至少能够表达 XLEN/ABI/endianness、ISA
extensions、scalar/vector element widths、LMUL 集合、vector register count、VLEN 的
fixed/range/unknown 状态、mask/tail 能力、extension fragments、rounding/saturation 与
memory instruction classes。

可选 microarchitecture profile 只提供 cache、throughput、latency、bandwidth、preferred
unroll 等排序提示。它不能使 architectural-illegal candidate 合法。Target profile 不是
cost model；无法可靠建模的 residual performance 由 AOT measurement 处理。

## 6. 权限分配

| 决策 | 作者/source | Provider | Compiler | Tuner |
|---|:---:|:---:|:---:|:---:|
| worker ABI、outer control、VLA logical range | 是 | 否 | 验证/重算 | 否 |
| algorithm/cache blocking、staging skeleton | 是 | 否 | 验证 | 在声明域中选值 |
| logical predicate、state algebra、numerical policy | 是 | 否 | 保持 | 否 |
| primitive 是否存在、explicit algorithm variant | 是 | 否 | 不发明 | 可在已提供 variants 中选 |
| primitive-local LMUL/microtile/fragment/packing | 否 | 声明空间与约束 | 组合并选择 | 在合法候选中测量 |
| dynamic `vl`、tail、`vsetvl` placement | 否 | realization | 推导 | 不是 knob |
| selected records 与 artifact | 否 | 提供 schema/lowering | 生成 | 选择固化方案 |

Tuner 只能测量 compiler 已证明合法的候选，不能创建 loop/primitive、改变 numerical
policy 或绕过 verifier。

## 7. Selection 数据流

```text
canonical primitive anchors
        + target profile
        + source meta domains / specialization predicates
        │
        ├─ derive local typed facts
        ├─ query all matching providers
        ├─ enumerate physical parameter candidates
        ├─ solve legality and cross-primitive resource constraints
        ├─ optionally measure legal candidates at AOT build time
        └─ write one coherent Selected Execution IR
```

候选集合和成本估计是瞬态分析，不进入 selected artifact。Selected IR 只保存最后选中的
决定和 canonical references。跨 primitive 约束由 compiler 组合解决，不能通过创建一个
giant whole-kernel provider 来回避。

## 8. Local fusion 与 packing 边界

Provider 可以声明局部 fusion envelope，例如把紧邻的 cast、fill、activation 或 packing
吸收到某个 primitive realization。允许的前提是：

- canonical data/effect relation不变；
- 被吸收节点仍可由 canonical anchor 追溯；
- selection 显式记录该局部决定；
- fusion 不跨越未声明的 state/effect boundary；
- emitter 不因 fusion 重新识别整算子。

Algorithm/cache packing 的存在与持久存储格式由 source/variant 决定；register/fragment
packing 由 provider 决定。二者不得共用同一“packed format route”。

## 9. Fail-closed 规则

当没有 provider 能合法实现一个 primitive 或多 provider 组合无法满足资源约束时，编译
必须报告具体 canonical anchor、候选和违反的 typed constraint。禁止：

- 默认切换到旧 backend；
- 因 target string 未识别而假定通用 RVV；
- 丢掉 predicate/numerical attributes 后继续生成代码；
- 用 kernel 名选择一个手写 source template；
- 生成不完整 selected IR 再让 emitter 补决定。
