# RISC-V 前向 physicalization

本模块落实 [`weft-spec.md`](../../report/weft-spec.md) 第五部分。Level 树、数值支路、逻辑类型、birth、handoff、cohort、staging 和持久 encoding 从 canonical Kernel IR 原样读取。Target lowering 只决定具体 value、op、memory edge 与 Level 的物理实现。

## 瞬态表示边界

编译调用可以使用临时 `weft_riscv` operation 保存当前候选及逐实体 attributes；它们不构成输入语言，也不在编译调用之间保存。不存在 kernel-global 的 `c_decisions` / `d_decisions` 字典。

决定的归属固定为：

```text
value        physical kind / SEW / LMUL / vl / materialization / spill
operation    local realization / instruction / result handoff
memory edge  canonical layout mapping / load form / unpack / stride / alignment
Level        tail / unroll / pipeline / prefetch
target       ISA / ABI / VLEN / extension set / register budget
```

全局寄存器预算可以同时检查全部实体，但不能把不同实体的决定合并成一个全局开关。

## 候选与 Pass 主干

std 函数在前端 inline 后形成已经选定的 source tree；编译主干当前只展开该树中的 `auto` bindings。每个完整 auto candidate 独立运行同一串前向 pass：

```text
ConstructRISCVCandidates
    冻结 canonical tree；收集 axis、Level、typed value、structured
    encoding mapping、use-def、handoff、effect 与 target facts；为每个
    auto candidate 建立一个独立瞬态候选。

AssignRISCVRepresentations
    根据 typed value、显式 wide use 与 target facts，为每个 value 写入
    physical kind、SEW、LMUL、vl、lane axis 与 materialization。当前尚未
    实现 value-use convert；没有单一合法表示时该候选明确 invalid。

SelectRISCVLocalOperations
    根据 typed operands、encoding mapping、engine role 与 target facts，
    为每个 op 和 memory edge 选定 instruction、load form、unpack、
    fragment 与 operand/result handoff。

ScheduleRISCVLevels
    为每个 Level 写入 iteration/tail 归属。当前可执行候选固定为
    unroll=1、pipeline_depth=1、prefetch_distance=0；这些字段不代表已经
    存在多种流水实现。

CheckRISCVResources
    根据 SSA 首末使用、value register groups、matrix fragment 与保留组
    计算峰值。当前没有 spill realization：超出预算的候选直接 invalid，
    该 pass 不回头修改前面的决定。

SelectRISCVWinner
    按当前静态 resource cost 保留一个 winner。真机 compile-and-measure
    尚未进入这条实现。

EmitRISCVIntrinsicC
    只读 winner 上已经存在的实体 attributes，机械生成普通 C、RVV
    intrinsic 与 typed local asm。
```

Pass 只能读取 canonical facts 和前序 attributes，并且只写自己负责的实体属性。同一决定只有一个 producer。资源失败不会触发 arc propagation 或 DFS 回溯。当前 LMUL、instruction 与 Level schedule 在一个 auto candidate 内是确定的；要探索它们的其他合法取值，必须先增加一个真实可生成的外层 physical config，而不能在 resource pass 或 emitter 中回退。

## Layout 与 convert

Encoding declaration 保存结构化 `natural/grouped/layered/joined` mapping。Local-operation pass 将每次字段访问展开为 memory-edge mapping，不按 encoding 名称进入分支。

当前主干只接受一条前向传播后自洽的 value chain。显式 value-use convert 尚未实现，因此冲突会使整个候选 invalid；emitter 不会临时插 coercion。跨调用 pin 也不允许偷偷插 coercion。

## 资源检查

资源检查读取已经决定的：

- value LMUL、register bundle 与 lifetime；
- canonical value 的 register groups；
- matrix local operation 的固定 fragment groups；
- emitter 保留的固定寄存器组；
- target 的 vector register、fragment 和 stack budget。

它只产生 `valid` 或明确的 invalid reason。当前不生成 spill。它不能缩小 LMUL、减少 accumulator、替换 instruction、改变 Level schedule 或修改作者树。

## Emitter 契约

Emitter 不扫描 source closure，不根据 dtype、shape、VLEN、encoding family 或 target 名称决定结构。它只能读取：

```text
final value representation
final op realization
final memory-edge mapping/form
final Level schedule
target intrinsic/asm spelling table
```

缺少任何一项都表示前序 pass 契约不完整，必须在 emission 前明确失败。RVV intrinsic API 变化只影响 spelling；不得穿透到 representation、operation selection 或 resource pass。

普通 `for`、`if`、`while` 始终是 `ordered-scalar-control`。只有显式 Level、shaped value、local op 与 engine role 可以贡献宽向量或矩阵实现。

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

打印结果必须能逐项指向具体 value、operation、memory edge 或 Level；不再输出全局 C/D 摘要。真实 correctness repro 使用随机 packed bytes，让 Weft intrinsic C 与 GGML reference 消费同一段 Q4_K/Q8_K storage，再比较数值。
