# RISC-V dot layout、分层存储投影与受影响性能报告

## 1. 本轮范围

本轮只处理上一轮 reduction-lane 改动之后暴露出的三类物理编译问题：

1. 已经 shaped 的 Q4_0、Q4_1、Q5_1、IQ4_NL 在 SG2044 上仍低于 source；
2. Q1_0、Q2_K、TQ2_0 的 sub-byte storage projection 尚未形成合适的物理操作；
3. Q4_K persistent 在 SG2044 上报告 34 个 vector groups，超过 32 组预算。

没有运行 206 条全量语料。真实重测仅覆盖被这些改动影响的 8 个 prefill 入口，并在 SG2044/VLEN128 与 K1/VLEN256 上分别执行 10 次。当前结果已经写回 `report/weft-kernel-performance.csv`；Q4_K persistent 不属于该 CSV 的固定 production manifest，因此只记录在本文。

## 2. 与 Triton、TileLang 的机制对照

### 2.1 指令要求是 layout anchor，但 anchor 不是唯一合法公式

Triton 的 `TritonDotPattern` 在 `ref/triton/lib/Conversion/TritonToTritonGPU/TritonToTritonGPUPass.cpp:192-259` 中先为 dot result 构造 blocked encoding，再为 A/B 显式构造 `DotOperandEncodingAttr`，并插入真实 `ConvertLayoutOp`。这说明 dot 的指令要求是物理 layout anchor；它不是从普通 pointwise SSA 猜出来的。

`RemoveLayoutConversions.cpp:42-59` 随后执行 anchor 发现、向 use 链传播、冲突处插 conversion、按 dominance 顺序重写程序。`isLayoutAnchor` 也明确把 `DotOpInterface`、部分 load/store、atomic 与 gather/reshape 作为 anchor（同文件 `:218-256`）。

Weft 上一轮把 reduction axis 放到 RVV lane、free axes 放到 register replicas，和这一机制是同一类工作：显式 contraction 给出 axis relation，target operation 给出物理 anchor。本轮发现这个 anchor 不能机械固定成一种映射。Q4_K correction 的 `m[16,8] × bsum[8] -> correction[16]` 若把 K=8 放 lane，会先物化 16 个 output replicas，再在 inner loop 逐列抽取；当 reduction 小于一条基础 RVV vector、free axis 可由一条 vector 承载时，正确结构是 free axis 进 lane、reduction 进 issue time。

因此本轮在 `PropagateRISCVLayouts.cpp` 中加入的不是 Q4_K 分支，而是一条 target-aware dot 规则：

```text
所有 reduction axes 都真实存在且 extent 已知
reduction elements < VLEN / SEW
存在一条 extent <= VLEN / SEW 的非 reduction axis
→ free axis = lane，reduction axes = sequential time
否则保持 reduction-lane 规则
```

这个规则读取 typed axes、extent、SEW 与 target VLEN。它不读取 kernel、格式或文件名。

### 2.2 Triton backward rematerialization 不能直接解决 34/32

Triton 的 backward slice 从一个真实 `ConvertLayoutOp` 开始，沿 SSA producer 反向传播目标 encoding；冲突时失败，遇到已有目标 conversion 可停止，并处理 `if` yield 与 `for` init/yield（`ref/triton/lib/Dialect/TritonGPU/Transforms/Utility.cpp:884-1008`）。`RemoveLayoutConversions` 的 rematerialization 再 clone 该 slice，使 producer 直接产生目标 encoding。

但它的收益判断比较 conversion 与 rematerialization 成本，不读取同时 live 的 register groups。它不能把同一 `rvv_contract_step` 必须同时读取的 lane operand 和 accumulator 变成“不同时 live”。Triton 中更接近寄存器压力处理的是独立的 `ReorderInstructions`：`ref/triton/lib/Dialect/TritonGPU/Transforms/ReorderInstructions.cpp:31-40,92-123` 把 `LocalLoadOp` 或产生 DotOperand layout 的 conversion 下沉到首次使用或 loop 内。

Q4_K 的真实 physical SSA 进一步说明，问题位于 dot anchor，而不是 conversion：

- 原表示把 `m[16,8]` 设为 K-lane × M-replica，单值占 16 groups；
- inner reduction 又把它 extract 成 M-lane vector；
- resource pass 为 accumulator 与 extract result 做 spill/reload 后，峰值仍为 34/32；
- 改成 M-lane × K-time 后，physical peak 降到 20 groups；
- 同一作者树在 SG2044 上生成并运行到 10.128 GOP/s。

所以本轮没有把 Triton backward rematerialization 照搬成一个不适用的修复。Weft 当前 `CanonicalizeRISCVLayouts` 仍只对 direct producer 做 rematerialization；对于未来真实存在的长 conversion backward slice，这仍是一个未闭合能力，但它不是本次 34/32 的原因。

### 2.3 storage window 必须是 typed physical operation

TileLang 把 pipeline planning 与 expansion 分开：`pipeline_planning.cc` 先收集 buffer read/write region 和 conflict，`inject_pipeline.cc:66-85` 再消费 stage/order 与 buffer lifetime，生成真正的 pipeline program。这个分工说明，仅在 emitter 中缓存一段地址字符串不能替代可被 pass、verifier 与 resource analysis 观察的物理实体。

本轮因此增加两种真实 RISC-V IR operation：

- `rvv_bitmask_decode`：表示 byte-aligned logical-u1 field 在一个明确 sub-Level point 上变成 RVV value；
- `rvv_layered_stream`：表示 grouped/layered storage 的多个 logical time parts 共享同一 byte window。

二者都携带 typed field/access/layout/leaf，经过 verifier，并成为 final RISC-V IR 的 terminal operation。Emitter 只按已确定的 group、layer、order、time/lane/register mapping 拼写 load、shift、mask 或 `vlm`/`vmerge`。

`rvv_layered_stream` 最初只覆盖 `group = 2 × layer`。本轮将其改为由以下几何条件生成：

```text
layers = group / layer > 1
element_width × layers <= 8
唯一一条 stream axis
time × lane = logical extent
lane 整除 layer
其余轴只能是 register replicas
```

因此同一个 operation 同时覆盖 Q4/Q5 的 2-layer nibble 和 TQ2 的 4-layer u2 storage，没有格式分支。

## 3. 实际代码改动

### 3.1 Dot 物理表示

`PropagateRISCVLayouts.cpp` 的 `Roles` 增加 `sequentialAxes`。它不是 DSL 构造，也不改变 logical axes；它阻止明确选择为 issue-time 的 reduction axis 又被 `addSmallReplicas` 自动放回 register replicas。

该改变使 Q4_K persistent 的 correction operand 从：

```text
time=[1,1], lane=[1,8], replica=[16,1], groups=16
```

变成：

```text
time=[1,8], lane=[16,1], replica=[1,1], groups=1
```

最终 kernel 的 `vector_register_peak` 为 20，而不是 34。

### 3.2 Logical-u1 bitmask decode

`FuseRISCVBitplanes` 现在在 composite lowering 之后读取真实 `time_to_lane convert_layout`，并由以下 typed facts建立 `rvv_bitmask_decode`：

- field element 是 unsigned 1-bit；
- access 是 byte-aligned `grouped_layered`；
- `group = 8 × layer` 且 `lo_first`；
- sub-Level point 明确从 origin point 派生；
- result 的 time/lane factors 完整覆盖该 logical axis。

此前 Q1 对每个输出生成 8 次 byte gather 和 index chain；现在生成 mask load 与 vector merge。普通 Python/scalar loop没有被识别成 SIMD axis。

### 3.3 多层 packed stream

`ShareRISCVLayeredWindows` 现在对 typed field geometry 产生 `rvv_layered_stream`。同一 storage byte window只加载一次，再按 physical layer 执行 shift/mask。Verifier 检查 source 确实是同一个 `FieldOp`、access 完全相等、bit offset byte-aligned、只有一条完整 stream axis。Emitter 对缺失的 register-axis byte stride直接失败，不再静默使用 base pointer。

该能力在 Q4_0、Q4_1、Q5_1、IQ4_NL 与 TQ2_0 上实际生效；Q2_K 的 q field 在 sub-Level extract 前仍是 local encoded value，所以没有进入这个 stream op。

### 3.4 Widen-dot resource closure

`RVVWidenDot` 的 temporary groups 改为按真正同时 live 的 partials计算：per-stream reduction保留一个 partial/stream，fused reduction只保留一个 accumulated partial，再加 seed 与 reduction result。Output replicas由 emitter 顺序生成，不再错误地全部计入一个 issue point。最终 resource verifier仍以 physical SSA liveness和 target budget为准。

### 3.5 参数实测

SG Q5_1 的同一棵树上，参数实例表现为：

| MR×NR | pipeline | GOP/s |
|---|---:|---:|
| 2×1 | 2 | 3.578 |
| 8×1 | 2 | 5.053 |
| 4×1 | 2 | 约 5.9 |
| 2×2 | 2 | 6.464（10 次正式中位数） |

因此 SG production runner选择 `MR=2, NR=2, depth=2`。这是参数性选择；没有改变 std 树或 lowering family。Q4_1 在 SG 使用 `MR=8, NR=1, depth=1`，K1 仍使用 `MR=4, NR=2, depth=1`。

## 4. 受影响项目的真实结果

协议：Clang 18、`-O3 -ffp-contract=fast`、相同 runtime/source wrapper口径、1 次 warmup、10 次计时中位数。数值标准为项目当前的 tolerance 检查；所有下列条目均为 `within-tolerance`。

### 4.1 SG2044 / VLEN128

| Kernel | 本轮前 Weft | 本轮 Weft | source | Weft/source | 本轮提升 |
|---|---:|---:|---:|---:|---:|
| Q4_0 prefill | 7.113 | 8.642 | 7.580 | 1.140× | +21.5% |
| Q4_1 prefill | 4.711 | 6.691 | 2.535 | 2.639× | +42.0% |
| Q5_1 prefill | 3.984 | 6.464 | 5.951 | 1.086× | +62.2% |
| IQ4_NL prefill | 6.049 | 8.629 | 6.954 | 1.241× | +42.7% |
| Q1_0 prefill | 3.376 | 6.912 | 4.250 | 1.627× | +104.8% |
| Q2_K prefill | 3.197 | 3.196 | 9.462 | 0.338× | 无变化 |
| TQ2_0 prefill | 1.133 | 1.564 | 6.983 | 0.224× | +38.1% |
| Q4_K persistent | 资源非法 | 10.128 | 9.699 | 1.044× | 从 34/32 变为 20/32 |

用户指定的四条 shaped SG kernel 已全部超过 source。这个结论只覆盖这四条与本轮所列入口，不外推到全仓。

### 4.2 K1 / VLEN256

| Kernel | 本轮前 Weft | 本轮 Weft | source | Weft/source | 说明 |
|---|---:|---:|---:|---:|---|
| Q4_0 prefill | 1.858 | 2.297 | 28.431 | 0.081× | source 是 IME1，Weft production 是 RVV，不是同引擎比较 |
| Q4_1 prefill | 1.501 | 1.948 | 24.581 | 0.079× | source 是 IME1，Weft production 是 RVV，不是同引擎比较 |
| Q5_1 prefill | 1.558 | 1.757 | 1.768 | 0.994× | 同为 RVV |
| IQ4_NL prefill | 1.976 | 2.466 | 2.869 | 0.859× | 同为 RVV |
| Q1_0 prefill | 1.828 | 3.479 | 3.793 | 0.917× | 同为 RVV |
| Q2_K prefill | 1.202 | 1.203 | 2.410 | 0.499× | 同为 RVV，基本无变化 |
| TQ2_0 prefill | 2.312 | 3.681 | 4.979 | 0.739× | 同为 RVV，+59.2% |
| Q4_K persistent | 先前未闭合本轮 SG 资源路径 | 3.227 | 24.577 | 0.131× | source 是 IME1，Weft是RVV |

VLEN256 不再只是把 lane 数翻倍：Q1 使用 typed mask realization，TQ2 使用四层 shared stream，Q4/Q5/IQ4 使用 grouped/layered stream。但 K1 上 IME仍未进入这些 production std 树，所以不能把 Q4_0/Q4_1 的巨大差距归因于 RVV layout细节。

## 5. Q1_0、Q2_K、TQ2_0 的瓶颈分解

### 5.1 Q1_0

Q1 的共同瓶颈确实是 logical-bit projection。把 `Field -> Extract -> time_to_lane conversion` 变成 typed bitmask decode后：

- SG：3.376 → 6.912 GOP/s，超过 source 62.7%；
- K1：1.828 → 3.479 GOP/s，达到 source 的 91.7%。

因此 Q1 的主缺口已经被定位并移除；K1 剩余约 8.3% 不是 MR/NR 扫描问题。K1 上 `MR=4,NR=2`、`MR=8,NR=1`、`MR=2,NR=2` 都慢于当前 `MR=2,NR=4`。

### 5.2 Q2_K

Q2 没有吃到 field-level stream，因为其 inner tree是：

```text
local encoded record
→ 每 16 elements 做 sub-Level extract
→ q2 vector
→ widening multiply/reduce
→ scale/min/bsum correction
```

SG 汇编的 inner loop每个乘积立即执行 `vwredsum + vmv.x.s`，同时还有 indexed scale loads、scale/min unpack、bsum correction 与 scalar-to-vector materialization。它与 TQ2 共享 grouped/layered storage geometry，但不共享数值 cluster。

正式结果 3.196/9.462 表明，本轮通用 field stream没有误称为覆盖 Q2。下一项需要的是能消费 `local encoded field + typed sub-Level point` 的 physical extract/window，以及让多个 product partial在一次 reduction前累积；这不是再给 `rvv_layered_stream` 增加一个格式分支。

### 5.3 TQ2_0

TQ2 的 q field现在真实进入四层 `rvv_layered_stream`。SG 从 1.133 提到 1.564，K1 从 2.312 提到 3.681，证明 storage projection是一个共性成本，但不是全部成本。

最新 SG 汇编的 decode hot loop仍包含约：

```text
120 × vwmacc.vv
128 × vl1r.v
80 × vs1r.v
72 × vle8.v
207 × csrr
152 × mul
```

大量 register spill/reload和重复 `vlenb` 地址计算主导剩余差距。GGML donor对同一 block做四平面提取、四次 widened MAC、一次最终 reduction；Weft仍把 output/reduction组织展开成大量独立 partial。TQ2 的剩余问题因此是 dot issue/partial accumulation与真实 cluster schedule，不是 radix-3；TQ2本身是 u2 layered storage。

## 6. 正反结论

### 已被外部结果支持

- 同一套 typed layered geometry在 Q4_0、Q4_1、Q5_1、IQ4_NL、TQ2_0 与两种 VLEN上产生实际提升；不是只服务一个格式。
- 同一作者 Q4_K tree只改变 physical axis mapping，就从资源非法变为 20/32，并在 SG超过 source；没有改变 logical value集合或 Level归属。
- Q1 的 typed bitmask op在两台机器上均产生接近 2 倍提升；逐 lane gather确实是错误物理表示。

### 仍未成立

- Q2_K 仍只有 source 的 33.8%（SG）与 49.9%（K1）。
- TQ2 虽跨 target提升，SG仍只有 source 的 22.4%；减少 storage load没有自动形成 donor式 accumulation cluster。
- K1 Q4_0/Q4_1 production仍未进入 IME，和 IME baseline不可比。
- Weft 的 layout canonicalization仍不是 Triton式任意 backward slice rematerializer。本轮证明 Q4_K 资源问题不需要它，但没有证明这项能力已经完整。
- 14 个 row×column quant MUL_MAT tree、12 个逐元素 row-dequant tree与9个逐元素 vec-dot tree没有在本轮铺开。优先处理的一至四项中，Q2/TQ的编译器瓶颈仍未闭合，因此没有用批量改树掩盖它们。

## 7. 可复现命令

单条正式 repro 例如：

```bash
./examples/run/weft-mul-mat.sh sg2044 q5_1 prefill 10
```

Q4_K 资源与运行闭合：

```bash
./examples/run/weft-mul-mat.sh sg2044 q4_k_persistent prefill 10
```

K1 的对应命令只将首参数改为 `k1`。所有命令均经过 DSL → canonical Kernel IR → RISC-V physical IR → intrinsic C → Clang → 真机执行；没有调用 source 或 materials 作为 Weft fallback。
