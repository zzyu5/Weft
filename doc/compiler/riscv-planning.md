# RISC-V 编译空间

本模块落实 [`weft-spec.md`](../../report/weft-spec.md) 第五部分。它只求作者未写的 C 组 9 项与 D 组 7 项；Level 树、数值支路、逻辑类型、birth、handoff、cohort、staging 和持久 encoding 都从 canonical Kernel IR 原样读取。

## 表示边界

编译调用中临时注册 `weft_riscv` dialect：

```text
weft_riscv.problem
    有限程序候选、canonical facts、target facts、变量域和约束

weft_riscv.assignment
    一个完整候选的 value/op 表示、C/D 决定和资源结果
```

两者都不是输入语言，也不在编译调用之间保存。`problem` 不能作为输入混入 canonical IR；求解完成后它被唯一的 `assignment` 替换。assignment 是随后 intrinsic-C / local-asm 发射的瞬态输入，不是新的算法 authority。

## Pass 主干

Pass 按“建立哪一部分求解表示”划分，而不是按 kernel family 或代码改写划分：

```text
ConstructRISCVProblems
    冻结 canonical tree；收集 axis、Level、typed value、encoding、
    use-def、handoff class、effect 与 target facts；实例化
    当前 canonical specialization × auto 的有限候选。

ConstrainRISCVRepresentations
    建立 value representation、SEW、LMUL、lane axis、cohort、
    widen/pointwise/reduce/carry 的传播约束。

ConstrainRISCVInstructions
    建立 encoding field packing、派生 interleave、memory form、
    scale broadcast、partial layout、reduce placement、RVV/IME
    instruction domain 之间的双向约束。engine role 在这里是硬限制。

ConstrainRISCVResources
    从 SSA live interval 与 Level/for/if/while handoff class 建立
    accumulator、operand、pipeline buffer、spill 和寄存器压力回边；
    pipeline/unroll/prefetch 绑定到承载显式局部 primitive 的 Level。

SolveRISCVProblems
    对每个完整程序候选联合求解所有 C/D 变量，删除不可行候选，
    选择静态代价最低的完整 assignment。
```

Dense、quant、state 或某个 VLEN 不拥有另一套 pass。普通 `for`、`if`、`while` 的 compile rule 只有 `ordered-scalar-control`；只有显式 Level/value/engine 事实能够贡献宽向量或矩阵表示约束。

## 为什么采用“枚举 + 传播 + 回溯”

当前解法由三部分组成，但它们不是串行决定阶段：

1. **只枚举作者给出的完整程序候选。** typed std overload 在前端 inline 时已经选出一棵明确的 canonical specialization；本主干只枚举该 specialization 与 `auto` 有限取值的笛卡尔积。q4_K 当前因此只有一棵固定树。编译器不枚举新 Level、不改 handoff、不搜索另一棵数值树。
2. **每个候选内部做约束传播。** SEW/LMUL、nibble unpack、interleave、load form、partial layout、reduce placement、pipeline 与 prefetch 等有限域反复收窄到不动点。
3. **仍有多个合法值时做 DFS 回溯。** 每个分支再次传播；只有全部变量单值化后才计算按 live handoff class 得到的寄存器峰值、spill 与成本。资源失败会回到同一候选的 LMUL/layout/instruction/schedule 选择点，而不会改作者树。

不用纯前向 pass，是因为 LMUL、instruction/layout 与寄存器压力存在回边。只做不动点也不够，因为不动点通常保留多个互不支配的离散选择。把整个问题交给无边界结构搜索同样不对：那会允许编译器发明作者没有写的树。有限候选枚举确定结构边界，约束传播解决唯一合法推导，回溯只处理 C/D 结构与参数选择，正好对应规范的权责分界。

候选选择是联合的。例如 `MR`/`NR` 的 `auto` 绑定会先约束 `lane_axis ↔ cohort`，然后与 LMUL、accumulator grouping、pipeline buffer 和寄存器预算一起求解；不存在“先永久选择 auto，再单独补物理表示”的三阶段流程。

## Assignment 的完整性

一个 complete assignment 必须同时含有：

- 每个 value 的逻辑类型、物理种类、physical SEW、LMUL、VL、寄存器组、storage 与固定 layout identity；
- 每个 operation 的局部 realization 与 validity；Level 明确区分 cohort-lane 与 ordered-sequential，并给出当前层的 active extent、父层 validity 交集、tail policy 及 pipeline 所属层；
- C 组 `sew/lmul/vl/tail/accumulator_grouping/register_budget/partial_layout/horizontal_reduce/vlen_specialization`；
- D 组 `nibble_unpack/mac_instruction/scale_broadcast/byte_interleave/load_stride_alignment/prefetch_distance/pipeline_unroll`；
- peak vector groups、spill、stack bytes 与候选成本。

Encoding field 的 realization 读取 canonical `field_packing`：`nibble:lo_first`、`packed:12` 与 `natural` 分别进入不同的抽取形式，禁止由 `SEW < 8` 猜测。派生 encoding 的 assignment 把抽象 family 实例化为 `derived_instance`，并同时给出 builder、base-record bits、cohort instance size、alignment、bit/byte order、layout identity 和 target compatibility；这是后续 builder/artifact 的唯一输入，不是 emitter 后补的布局选择。

当前 D23 assignment 只承载一个派生 interleave family。一个 kernel 同时消费多个派生 interleave family 时会明确报 unsupported，不会把其中一个 family 的 layout identity 套到其他 value，也没有隐藏的 canonical-layout fallback。

Instruction 也按 operation 单独选择。D21 是一个 operation-indexed 数组，每项记录 engine、typed lhs/rhs、operand binding 和指令；同一 kernel 中的 `@wide` 与 `@matrix` 不共享一个全局 instruction 开关。寄存器结果带峰值时 live 的 handoff class 与 pipeline/unroll temporary 明细，prefetch 决定带所属 Level 和实际 producer 对象，而不是只有一个无法兑现的整数。

## Verifier 与发射边界

Kernel dialect verifier 只检查程序是否合法：domain 父子关系、Level region、admit/commit 域、类型/shape、encoding 字段和 engine role。RISC-V assignment verifier只检查 complete assignment 是否包含全部 C/D 决定和资源结果。两者都不证明数学等价、不证明不溢出，也不评价性能。

本轮终点是可打印的完整 physical assignment，尚未生成 intrinsic C。后续发射只能读取 assignment 中已经确定的 value shape、LMUL、layout、memory form、instruction、pipeline 和 fragment；不得重新扫描 Kernel IR 选择一次，也不得把向量化交给系统 C 编译器。

## q4_K 手工复现

```bash
PYTHONPATH=python python -m weft examples/kernels/quantization/q4_k_gemv.py \
  > /tmp/q4_k.mlir

build/tools/weft-compile/weft-compile /tmp/q4_k.mlir \
  --emit=physical-assignment \
  --march=rv64gcv_zfh_zfhmin_zvfh_zvfhmin \
  --abi=lp64d --vlen-bits=128 --matrix-extension=none \
  -o /tmp/q4_k.assignment
```

该命令从同一 canonical q4_K 树打印完整 assignment。当前 VLEN128 解选择 16-lane cohort、`rvv.vwmaccsu`、`and-shift` 的 `nibble:lo_first`、pair-major 的 `Q4K_I16` 实例、unit-stride load、sub Level 上 depth-2 / unroll-4 / prefetch-1 的局部流水，并在 32 个向量寄存器组预算内无 spill。把 `--vlen-bits` 改为 256 会重新求每个 value 的 LMUL 与资源峰值，而不会进入另一条编译路径。
