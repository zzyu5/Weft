# RISC-V typed partial accumulation、分层窗口与 Q2 树边界报告

## 1. 本轮范围与结果

本轮只处理上一轮留下的三项横向问题：

1. TQ2_0 的 widening product 过早归约，形成大量独立 partial、spill/reload 和重复地址计算；
2. Q2_K 的 local encoded field 在 sub-Level point 上没有 typed physical window；
3. K1 上 TQ2_0、Q2_K、Q1_0 的同引擎差距。

没有运行 206 条全量语料，也没有铺剩余 std 树。真实运行只覆盖 Q2_K、TQ2_0 和用户点名的回归入口。

最终结果是：

- TQ2_0 在 SG2044 从 1.564 提升到 19.977 GOP/s，在 K1 从 3.681 提升到 6.245 GOP/s；两台机器都超过同目标 source；
- Q2_K 获得 typed local encoded window，但只从 3.196/1.203 提升到 3.338/1.239 GOP/s；剩余差距不是同一个 pass 能合法解决的问题；
- 当前 Q2_K 作者树在每个 16-element sub-Level 内先结束 contraction，再执行 per-sub scale/min correction。GGML donor把 scale组织进 reduction 前的 vector product。把前者改成后者会改变 logical value graph 和中间累加关系，按 spec 2.2 必须由作者改树，编译器不能暗中分配律重写；
- SG 上已经过线的六条入口没有退化；K1 Q1_0 仍为约 3.48 GOP/s，剩余约 8% 不是本轮 partial/address问题。

正式 10 次结果已写回 `report/weft-kernel-performance.csv`。

## 2. 参考实现实际解决的是什么

### 2.1 Triton：dot result layout 锚定 operand layout，reduce 是独立的物理阶段

Triton 的 `DotOperandEncodingAttr` 明确规定 A/B 的 `opIdx`，并把 dot result 的 layout作为 `parent`（`ref/triton/include/triton/Dialect/TritonGPU/IR/TritonGPUAttrDefs.td:1431-1467`）。`AccelerateMatmul.cpp:442-485` 先从 result shape、warp数和目标 MMA version构造 result encoding，再为 accumulator插 `ConvertLayoutOp`，最后从 result encoding派生 A/B operand encoding。

这里的重要机制不是“见到 dot 就套一个模板”，而是：

```text
selected target dot operation
→ accumulator representation
→ operand representation
→ explicit conversion
```

Triton 的普通 reduction lowering又明确分成 thread 内、warp 内和必要的跨 warp/block 阶段，最后才转换到 output layout（`ref/triton/lib/Conversion/TritonGPUToLLVM/ReduceOpToLLVM.cpp:45-104`）。也就是说，partial 的物理存在期与 final reduction 在 IR/lowering中是可区分的；不是每产生一个 product 就立刻变回 scalar。

### 2.2 TileLang：partial storage/update 与 finalize 是一等物理计划

TileLang 的 reducer materialization 在 layout 已冻结后决定 partial 放在哪里、参与者怎样 finalize、update执行几次；其 narrow plan从 update site的 reduction axes投影出 partial layout（`ref/tilelang/src/transform/reducer_plan_materialize.cc:1-39`）。

物理 update写入 partial buffer；可选 packed plan把串行 combine chain拆成两条 accumulation lane（同文件 `:1166-1200`）。finalize阶段先合并 packed lanes，再只对 planner证明需要归约的物理 split执行 collective（`:1231-1291`）。

Weft 不能照搬 GPU thread/warp collective，但需要同一个结构事实：

> product、loop-carried partial 和 final reduction 必须是 physical program中不同的 typed实体，resource pass才能看到它们何时存活，emitter也不需要从一大串 SSA closure重新猜 accumulation结构。

## 3. 修改前的真实物理问题

上一轮 TQ2_0 已能把 grouped/layered u2 storage变成四层 RVV stream，但后续 `rvv_widen_dot` 仍一次接收整个 time-stream value。最终 C/汇编把所有 stream parts同时展开：

```text
120 × vwmacc.vv
80  × vs1r.v
128 × vl1r.v
207 × csrr
152 × scalar mul
```

这里的 `csrr vlenb` 不是 canonical/RISC-V IR 中一个可被 `HoistRISCVLoopInvariants` 移动的 operation。它由 Clang在处理大量 vector spill的动态栈地址时产生；152 个乘法也主要来自 emitter把大 time-stream拆成 C 地址表达式之后的重复展开。因此给 LICM 增加 pure whitelist不能解决它：问题在进入 C 之前没有形成正确的 partial program。

GGML 的 TQ2 VLEN256 donor则在每个 32-byte storage window上：

1. 只 load一次 packed byte vector；
2. 解出四个 2-bit plane；
3. load四个 Q8 window；
4. 对同一个 i16 accumulator执行四次 `vwmacc`；
5. 最后执行一次 `vwredsum`。

对应源码在 `source/c/ggml/llama.cpp/ggml/src/ggml-cpu/arch/riscv/quants.c:6399-6444`。

## 4. 新的 typed physical program

### 4.1 新 type 与 operation

RISC-V dialect增加一个短生命周期 storage-window type和五个真实 operation：

```text
!weft_riscv.layered_window

weft_riscv.rvv_storage_window
weft_riscv.rvv_layered_storage_load
weft_riscv.rvv_layered_storage_decode
weft_riscv.rvv_widen_accumulate
weft_riscv.rvv_finalize_widen_dot
```

它们分别表达：

- local encoded field在一个明确 sub-Level origin和logical offset上的 lane window；
- 一个 packed byte window及其复用的 register replicas；
- 从同一个 raw window选择一个 physical bit layer；
- 在一个 `scf.for` carry上更新 widened RVV partial；
- 消去 reduction lane并保留所有 free-axis register replicas。

这些节点定义在 `include/Weft/Dialect/RISCV/IR/RISCVOps.td`，verifier位于 `lib/Dialect/RISCV/IR/RISCVDialect.cpp:1013-1059,2810-3135`。它们不是 side record：pass之后打印 RISC-V IR可以直接看见 storage loop、partial iter_arg、四个 accumulate和最终 reduce。

### 4.2 `MaterializeRISCVPartialAccumulators` pass

新 pass位于 `lib/Target/MaterializeRISCVPartialAccumulators.cpp`，接在 layered-window sharing之后、leaf finalization和resource materialization之前（`lib/Target/RISCVCompiler.cpp:37-42`）。

它做两种转换：

1. 对 `local encoded field + typed exact sub-Level point`，把普通 extract变成 `rvv_storage_window`；
2. 对具有一个 layered storage root的 fused widening dot，生成显式 storage-window循环、loop-carried partial和final reduction。

第二种转换的前置事实全部来自 type/access/use-def：

```text
一个显式 widening dot 和一个 reduction axis
两侧 reduction time decomposition一致
恰有一侧来自一个 grouped/layered stream
另一侧有唯一 typed field root
packed element width × layer count = 8 bits
bit offset byte-aligned，bit order明确
sub-Level reduction domain是 exact
layer extent能被 lane window整除
至少存在两个不同 storage windows
result free axes已经表示为 register replicas
```

pass不读取 encoding family、kernel名或文件名。Q4/Q5/IQ 是否适用由上述几何与 use-def决定，不由格式身份决定。当前固定 production prefill入口的静态 incidence检查中，完整 partial loop只在 TQ2_0出现，Q2_K只产生 typed storage window；其余格式没有误生成这些节点。

### 4.3 verifier闭合的关系

独立审查暴露了几个 production整除 shape不会触发、但 IR 合同必须拒绝的情况。本轮在提交前补齐：

- `LayeredWindowType.resource_groups` 必须等于其 RVV result layout的真实 register groups；
- storage window只能消费 exact reduction domain，实际 load lane数来自 reduction-axis lane factor，而不是把 layout的最大 `vl` 当成逻辑窗口；
- free axes只能保留为 register replicas；符号逻辑 extent允许由正的固定 cohort factor表示，不要求 cohort等于完整运行时 extent；
- layered byte window要求 `element_width × layers = 8`、byte-aligned bit offset和明确 lo/hi order；
- widened accumulator的轴顺序是两侧 free axes的有序并集加 reduction axis；free axes映射到 replicas，reduction axis映射到 lane，SEW/LMUL按 widening关系闭合；
- accumulator必须位于一个 `scf.for` 中，从对应 iter_arg出发，经唯一 accumulate chain到同一 carry的 `scf.yield`；
- final reduction只接受该 loop result，必须消去唯一 reduction lane，并把每个 free-axis replica投影到 scalar tuple result；
- pass在读取 layered reduction-axis position前先验证该 axis确实存在，不再允许 malformed IR触发 optional解引用。

新 op创建时也直接填写精确 leaf operand/result/temporary groups；后续 `FinalizeRISCVLeaves` 的重算现在是幂等检查，不再负责掩盖 stale资源字段。

### 4.4 emitter只拼写最终结构

`RISCVIntrinsicC.cpp:7482-7746` 分别拼写 window load、layer shift/mask、`vwmacc/vwmaccsu` 和最终 `vwredsum`。Emitter读取：

- operation上已经确定的 access/group/layer/order；
- value type上已经确定的 lane/register mapping与SEW/LMUL；
- 显式 `scf.for` 和 accumulator SSA chain。

它不决定是否合并 partial、何时 reduce、窗口有几层或 free axis放在哪里。缺失任何 typed relation都会由 verifier或emitter明确失败。

## 5. TQ2_0：外部结果与汇编变化

协议：Clang 18、`-O3 -ffp-contract=fast`、相同 shape/runtime/source wrapper、1次 warmup、10次计时中位数。四条正式结果均为 `within-tolerance`。

| Target | 修改前 Weft | 当前 Weft | source | 当前/source | 本轮提升 |
|---|---:|---:|---:|---:|---:|
| SG2044 / VLEN128 | 1.564 | 19.977 | 6.983 | 2.861× | 12.77× |
| K1 / VLEN256 | 3.681 | 6.245 | 4.979 | 1.254× | 1.70× |

当前 SG 汇编统计为：

```text
32 × csrr
36 × scalar mul
5  × vs1r.v
5  × vl1r.v
32 × vwmacc.vv
12 × vwredsum.vs
```

`vwmacc=32` 对应当前 MR=4、NR=2 的八个 output accumulator，每个 accumulator消费四个 decoded layers。最终每个 output replica只做一次 widened reduction。当前仍有5对vector spill/reload和32次`csrr`，但它们不再位于旧的大规模展开结构中。

这组结果说明两件不同的事：

- typed partial program确实关闭了本轮的横向编译缺口；同一机制在 VLEN128与VLEN256上都生成合法且高于 donor的实现；
- SG 比 source快2.86倍并不表示所有 grouped/layered格式都会自动获得同样收益。TQ2的作者树允许整个 256-element dot在scale前保持一个整数 partial；Q2的树不允许。

## 6. Q2_K：storage侧已闭合，accumulation侧撞到作者树

| Target | 修改前 Weft | 当前 Weft | source | 当前/source | 本轮提升 |
|---|---:|---:|---:|---:|---:|
| SG2044 / VLEN128 | 3.196 | 3.338 | 9.462 | 0.353× | +4.4% |
| K1 / VLEN256 | 1.203 | 1.239 | 2.410 | 0.514× | +3.0% |

Q2_K现在真实产生 `rvv_storage_window`，因此“local encoded field + typed sub-Level point”不再到 emitter才解释。提升很小，是因为后续 logical tree仍是：

```text
for each 16-element sub:
    integer = outer_contract(x.q[sub], w.q[sub])
    correction = x.bsum[sub] * minimum[sub]
    acc += ds * (d * scale[sub] * integer - dmin * correction)
```

每个 `integer` 在 scale/min correction前已经是一个独立 logical value。编译器若把多个 sub的 product先合并成一个 RVV partial再reduce，必须把 `scale[sub]` 分配进 contraction内部，改变中间值、widening位置和整数overflow边界。

GGML VLEN256 donor确实采用另一种树：先把四个 q2 plane分别与per-sub scale相乘，形成 i16 `p0..p3`，再与四个 Q8 vectors做i32 widened product，合并后执行两次最终 reduction（`source/c/ggml/llama.cpp/ggml/src/ggml-cpu/arch/riscv/quants.c:896-930`）。这不是同一作者树的纯物理表示。

因此 Q2剩余差距是明确的 spec 2.2 信号：

> 要接近 donor，需要作者提供把 per-sub scale放在 reduction内部的 std 数值树；本轮编译器不能把现有 `scale * reduce(product)`自行改成`reduce(scale * product)`。

本轮没有越过这条边界，也没有用 Q2格式分支把 donor tree塞进 pass或emitter。

## 7. K1 Q1_0 与回归结果

K1 Q1_0正式先前结果为3.466 GOP/s，本轮一次受影响复核为3.480；source为3.793，约0.917×。当前热循环没有`csrr`、scalar address `mul`或vector spill，逻辑结构也和 GGML donor一致：mask load、Q8 load、negate/merge、32-lane widened reduce、四个sub-block scale accumulation（donor在`arch/riscv/quants.c:484-520`）。

因此剩余约8%不能归到本轮的 storage-window/partial pass。静态汇编只把可能位置缩小到每个32-element partial的reduction/extract与scale边界、四个sub-block之间的load-use调度，以及外层MR2×NR4 cohort开销；现有证据不足以在三者间定量归因，本轮没有顺手改leaf拼写。

SG2044一次回归结果：

| Kernel | 本轮前正式 GOP/s | 本轮复核 GOP/s | 数值 |
|---|---:|---:|---|
| Q4_0 | 8.642 | 8.658 | within-tolerance |
| Q4_1 | 6.691 | 6.706 | within-tolerance |
| Q5_1 | 6.464 | 6.457 | within-tolerance |
| IQ4_NL | 8.629 | 8.626 | within-tolerance |
| Q1_0 | 6.912 | 6.903 | within-tolerance |
| Q4_K persistent | 10.128 | 10.118 | within-tolerance |

这些波动均小于0.2%，没有观察到结构回退。K1 IQ4_NL一次复核为2.470 GOP/s，与此前2.466相同水平。

## 8. 可复现命令

正式受影响入口：

```bash
./examples/run/weft-mul-mat.sh sg2044 tq2_0 prefill 10
./examples/run/weft-mul-mat.sh k1 tq2_0 prefill 10
./examples/run/weft-mul-mat.sh sg2044 q2_k prefill 10
./examples/run/weft-mul-mat.sh k1 q2_k prefill 10
```

保留当轮 generated C与汇编用于逐条检查：

```bash
WEFT_KEEP_ARTIFACTS=1 ./examples/run/weft-mul-mat.sh sg2044 tq2_0 prefill 1
```

所有入口均经过：

```text
DSL source
→ canonical Kernel IR
→ RISC-V physical IR
→ intrinsic C
→ Clang 18
→ SG2044/K1真机执行
```

source只作为同目标数值与性能参照，没有进入Weft编译或运行路径。
