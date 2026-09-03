# 全量账后 failure 与性能簇收口

日期：2026-09-04

范围：本报告记录全量账之后按顺序处理的 correctness、compile-failure 与性能簇。只记录当轮
真实 IR、生成代码和双机运行结果；未重跑的历史行不在这里重新解释。

## 1. IQ3_XXS encoded-entry alignment

### 1.1 修改前工作账

- IQ3_XXS record 为 98 bytes，声明 alignment=2；metadata 是 offset=66 bytes 的 u32 field。
- 每个 indexed-entry consumer 只发生一次 metadata load；没有两个相同 logical offset 可由 CSE
  合并。因此该修复预计删除的动态 load 数为 0。
- 原 Physical IR 选择 element-width unit load，生成 `vle32`；连续 record 的奇数项只保证
  2-byte alignment。K1 standalone、MUL_MAT decode、prefill 均 SIGBUS。
- 改动不改变 logical value、Level、widening、consumer 数、reduction 或 pipeline；它只把同一
  memory edge 的合法 leaf 从 EEW32 改为 EEW8 load 后在同一 register group 内 reinterpret。
- 即使指令计数不下降也必须实现，因为原 leaf 在已证明的 alignment 上不合法；收益首先是从
  runtime fault 恢复成正确程序，而不是吞吐优化。

### 1.2 结果

从 Encoding base alignment、field offset 与 entry byte stride 推出 effective alignment，Physical
IR 显式选择 `rvv.indexed-entry-byte-load`；final verifier 拒绝夸大的 alignment 或未闭合的宽 load。
96 份 row-dequant/vec-dot Physical IR 中只有 SG/K1 的 IQ3_XXS vec-dot 选择该 leaf，其余 94 份
保持原 leaf。96/96 均可独立 parse/verify，安全运行 canonicalizer、CSE、Share 两次；同一流水
第二次运行文本 diff 为 0。

| target | entry | 修前 | 修后（10 repetitions） | source | 修后/source | numeric |
|---|---|---:|---:|---:|---:|---|
| SG2044 | standalone vec-dot | 0.283 | 2.433 GOP/s | 2.674 | 0.910× | error 0 |
| SG2044 | MUL_MAT decode | 0.283 | 2.448 GOP/s | 2.618 | 0.935× | error 0 |
| SG2044 | MUL_MAT prefill | 0.284 | 2.439 GOP/s | 2.920 | 0.836× | error 0 |
| K1 | standalone vec-dot | SIGBUS | 1.239 GOP/s | 1.448 | 0.856× | error 0 |
| K1 | MUL_MAT decode | SIGBUS | 1.246 GOP/s | 1.436 | 0.867× | error 0 |
| K1 | MUL_MAT prefill | SIGBUS | 1.245 GOP/s | 1.472 | 0.846× | error 0 |

## 2. IQ2 staged widening-dot closure：实现前预测

IQ2_S、IQ2_XS、IQ2_XXS 的 staged prefill 在 SG2044/K1 共六个入口均停在同一个 typed
widening-dot legality check。三棵 canonical tree 已显式包含 MR/NR free axes、entry/payload
reduction、K blocking、materialize 与 accumulator lifetime；当前没有作者树缺轴的证据。

七步工作账得到：

1. hot contraction 的 result 是保留 MR/NR free axes 的 scalar register tuple；两个 si8 operand
   都是 RVV carrier，target widening 与 doubled LMUL 合法。
2. 数学 MAC 数、logical entry/payload coordinate 与 source value graph不变；本修复预计不删
   canonical work。
3. supply identity不合并；storage proposal只是 producer形态，不能成为另一 operand 的
   contraction carrier。
4. 失败发生在 memory leaf选择之前；本节点不预测改变 unit/indexed/gather 数。
5. 两 operand 的 reduction lane/time decomposition不一致，导致 lane slices、stream count或
   part projection不能形成同一个 target operation。
6. pipeline尚未形成，本节点不碰 load-to-use距离。
7. Triton 先以 result MMA encoding为 parent构造两个 DotOperandEncoding并插真实
   `convert_layout`；Weft 应同样由 layout pass冻结共同 reduction carrier，Lower只验证和消费。

预测：在最后一次 use-def propagation 后重新冻结同一个 typed reduction carrier，六个入口会从
compile failure变成合法 Physical IR；Q4/Q5 等既有 widening-dot入口的最终 topology与真机数字
不变。若六个入口仍失败，说明问题不只是 carrier owner；若能编译但动态工作没有形成既有 partial
program，则该节点只修 legality，不能把性能收益归给它。第二输入不是另一个格式名，而是三种不同
IQ2 storage relation与现有 Q4/Q5 widening-dot关系共同经过同一规则。

### 2.1 实际闭合过程

预测只命中了 legality，没有命中“现有 partial program 会自然形成”这一隐含期待。实现与反例依次是：

1. layout propagation 只在两个 operand 各自带有独立 surviving free axis 的 blocked contraction 上
   冻结共同 reduction carrier。reduction axes 保持各自的 typed axis identity，最后一个 lane-bearing
   reduction axis 是 primary axis，其余 reduction axes 在同一 lane carrier 中线性化。
2. `PartialSetType.termsPerSlot` 改为完整多轴 lane carrier 的元素数；planner 冻结 primary axis、
   source set 与 combine set，materializer、verifier、emitter 共用同一 lane-count 与 slice-LMUL 合同。
3. 第一次生成的 C 数值错误且三档 LMUL 得到同一个错误结果。静态逐 lane 检查定位到 IQ2_S 的 sign
   window：逻辑布局是 `[scale_group=2, entry=2, payload=8]`，旧 memory plan 把 entry 放在 replica，
   却把每个 physical part 当作 16 个连续 bit 载入。entry 0 实际需要 bit `0..7,16..23`，`vlm`
   读成了 `0..15`。
4. bitmask window verifier 现在逐 lane 证明 logical bit offset 连续；memory planner 选择不超过 consumer
   lane width 的最宽连续 storage suffix。IQ2_S 因而形成 8-bit unit mask window，再由显式
   RVV `time_to_lane` conversion 合成消费方的 16-lane carrier。
5. 直接把 32 个 sign bit 全放进 lane 会使 live peak 达到 43/32 vector groups，因此未保留。8-bit
   window 的 peak 为 4 groups，既不伪造连续性，也不靠超预算 carrier。

这里的 typed conversion 采用完整 source/result axis→time/lane/replica 关系；terminal translation
只按已闭合映射拼接 source parts。Triton 的对应机制是 `minimalCvtLayout` 先求完整 relative layout，
`ConvertLayoutOpToLLVM.cpp:282-415` 再把 register/lane mixed mapping 分解并拼写；
`RemoveLayoutConversions.cpp:1629-1689` 则在此前独立完成传播、rematerialization 与 cleanup。Weft
不能直接采用 thread/warp basis，但“中间 conversion 允许被 pass 消除，final conversion 才必须有
terminal contract”的边界相同。

实现中主动撞出一个反例：若在通用 `ConvertLayoutOp` verifier 创建 op 时就要求 terminal
part-to-lane 拼写，IQ1_S、IQ3_XXS、TQ1_0 双机六个既有 vec-dot 会在后续 DCE 之前被拒绝。
严格合同因此只放在 final verifier；这六个入口恢复后均通过下面的机械验收，没有扩大 emitter。

### 2.2 双机结果

六个原 compile-failure 均在真实 production shape 上数值零误差并完成 10 repetitions：

| format | target | Weft | source | Weft/source | 结果 |
|---|---|---:|---:|---:|---|
| IQ2_S | SG2044 | 0.528363 GOP/s | 2.506224 | 0.211× | 正确，但慢 |
| IQ2_S | K1 | 0.771796 GOP/s | 1.469172 | 0.525× | 正确，但慢 |
| IQ2_XS | SG2044 | 0.355132 GOP/s | 4.809702 | 0.074× | 正确，但慢 |
| IQ2_XS | K1 | 0.427212 GOP/s | 1.796205 | 0.238× | 正确，但慢 |
| IQ2_XXS | SG2044 | 3.981193 GOP/s | 4.111831 | 0.968× | 正确 |
| IQ2_XXS | K1 | 1.481867 GOP/s | 1.705121 | 0.869× | 正确 |

因此本节点只把 6 个 FAIL 变成了 6 个可执行结果，不能记作六条性能闭合。final Physical IR 的
差异直接解释了分档：

| typed physical work / inner body | IQ2_S | IQ2_XS | IQ2_XXS |
|---|---:|---:|---:|
| `rvv_issue_slice` | 16 | 16 | 0 |
| `rvv_widen_accumulate` / finalize | 8 / 8 | 8 / 8 | 0 / 0 |
| partial set / combine / finalize | 0 / 0 / 0 | 0 / 0 / 0 | 2 / 4 / 2 |
| generated-C `vslideup` | 8 | 8 | 0 |
| generated-C `vslidedown` | 40 | 10 | 2 |
| generated-C widening reductions | 36 | 12 | 6 |

IQ2_XS 还在每个 inner issue 中执行 8 次 16-bit scalar scale load，随后再次载入 4 个 packed
scale byte；其 16 个 8-lane table loads 先 slide-pack，再由 indexed gather 消费。IQ2_XXS 则已把
work 物化成两个完整 partial set，combine 后才 finalize。低速来自这三棵 staged tree 进入不同
typed materialization contract 后的真实动态工作，不是 conversion legality 或一个漏掉的 CSE。

### 2.3 机械验收与回归

- 24 row-dequant + 24 vec-dot × 2 targets：96/96 生成 final RISC-V IR，96/96 独立 parse/verify，
  运行 canonicalizer、CSE、layout canonicalization、Share 两次与 final verifier；同一流水第二次
  运行 96/96 文本 diff 为零。
- 新增 IQ2 staged 三格式 × 两 target：6/6 通过同一流水，第二次运行 6/6 文本 diff 为零。
- bitmask legality 改动实际覆盖 Q1_0、Q5_0、Q5_1、IQ2_S、IQ3_S。五类 row-dequant、standalone
  vec-dot、MUL_MAT decode 在双机共 30 个真实入口全部数值零误差、10 repetitions。与 CSV 旧快照
  相比除 SG Q1_0 standalone `4.300→4.622` 外均为小幅测量波动；最大负向波动是 SG Q1_0 decode
  `4.522→4.315`（-4.6%），没有数量级退化。

当前性能边界没有被隐藏：IQ2_S/XS staged 已经有合法共同 carrier 和正确 memory edge，但仍没有
形成 IQ2_XXS 那种完整 partial set；继续优化它们属于 partial/materialization 的下一类工作，不能
归入本次 layout closure 的收益。

## 3. F32 dense contraction carrier

### 3.1 修改前工作账与参照

作者树已经在 `python/weft/std/dense.py:21-26` 表达 NC/MC blocking、MR/NR output cohort 与
`over="k"` contraction。旧 Physical IR 却把 MR=2 放进 lane、K 放进 issue time；在 decode
shape 上每个热 K 点动态执行一次 `vlse32`、两次 scalar `flw`、两次 `vfmacc.vf` 和三次数据
指针递增。K=4096、N=4096 时共执行 8,388,608 次 strided X load、16,777,216 次 scalar W
load/FMA 和 25,165,824 次热循环数据指针递增。没有 spill，也没有可由 CSE 删除的重复 supply。

GGML 的 `ggml_vec_dot_f32`（`source/c/ggml/llama.cpp/ggml/src/ggml-cpu/vec.cpp:87-102`）
让 K 骑 contiguous RVV lane，以两次 `vle32`、一次 `vfmacc.vv` 和末端一次 `vfredusum` 完成
单输出 dot。差异首先是 P1 carrier 与 P3 memory form，不是 pipeline，也不要求改变 canonical
Value、Level 或作者 blocking。

动手前对照的机制是：

- Triton `LinearLayoutConversions.cpp:877-915` 从 result blocked layout 派生 FMA
  DotOperand layout，K 在 register basis 中完整保留、lane/warp 在 K 上 broadcast；
  `FMADotUtility.cpp:88-164` 再消费冻结的 M/N ownership 与 K pairing；
  `DotOpToLLVM/FMA.cpp:9-45` 只拼写最终 FMA。
- TileLang `loop_vectorize.cc:895-958,1068-1171` 先证明 lane substitution 后 flat offset 是
  unit Ramp 且 chunk base 合法，再由 `loop_vectorize.cc:1003-1041` 机械 strip-mine。其 CPU
  `src/cpu/op/gemm.cc:25-49` 仍固定 `cpu.scalar`，因此不是可直接移植的 F32 GEMM leaf；可借鉴的
  是先冻结 ownership/access、后展开的责任顺序。

预期可见计数是 strided/scalar operand supply 消失，改为两侧 unit stream load、`vfmacc.vv`
和末端 reduction；若这些计数不变，这个改动没有实现价值。第二个输入不是另一个格式，而是
同一 F32 tree 的 decode 与 prefill 两个 shape regime；两者必须同时形成相同 typed stream
contract，参数绑定可以不同。

### 3.2 Physical IR 改动与静态结果

layout propagation 现在只在已选 operation 为 `rvv.vfmacc`、且两个 operand 各自贡献一个独立
surviving free axis 时，把它解释为 RVV outer-product physical relation：K 留在完整 lane
carrier，M/N 留在 register replicas。已有 `RVVStreamContractOp` 随后物化 K loop、两侧 unit
`rvv_stream_load`、`rvv_stream_contract_step` 和 `rvv_stream_finalize`；没有新增 physical op，也
没有改 terminal emitter。

默认 MR2×NR2,m2 的生成汇编在两台机器上都变成每个 K chunk 四次 unit `vle32`、四次
`vfmacc.vv`，tile 末端四次 reduction；SG 的 VL=8，K1 的 VL=16，均无 vector spill。保持旧
参数时，单独这一项改动已经产生：

| target | phase | 修改前 | carrier 修改后、旧参数 | source |
|---|---|---:|---:|---:|
| SG2044 | decode | 1.494676 | 2.523213 | 1.717123 |
| SG2044 | prefill | 1.528872 | 7.392094 | 7.432725 |
| K1 | decode | 0.564268 | 1.780391 | 2.011619 |
| K1 | prefill | 1.108171 | 2.397623 | 2.574251 |

剩余差距是参数性绑定。静态枚举先排除了不合法组合：m8 只有 MR1×NR1 合法；m8 多输出的
同时 live peak 是 40--64 groups。合法 Pareto 点中，单输出 m8 虽接近 donor，却使 prefill 降到
SG 4.035 / K1 2.143 GOP/s；它减少 output reuse。MR4×NR4,m1 的 peak 为 24 groups，在两台机器
都保留 16 个 output accumulators 且不 spill，成为 prefill 实测 winner。这个过程只绑定作者声明的
auto 参数和 target physical LMUL，没有产生新的程序树或结构候选 pass。

### 3.3 十次真机结果

| target | phase | 最终 binding | Weft | source | Weft/source | numeric |
|---|---|---|---:|---:|---:|---|
| SG2044 | decode | NC16/MC64/MR2/NR2, m2 | 2.501721 | 1.717123 | 1.457× | error 0 |
| SG2044 | prefill | NC16/MC64/MR4/NR4, m1 | 8.888305 | 7.432725 | 1.196× | error 0 |
| K1 | decode | NC64/MC64/MR2/NR1, m4 | 2.088328 | 2.011619 | 1.038× | error 0 |
| K1 | prefill | NC16/MC64/MR4/NR4, m1 | 3.723850 | 2.574251 | 1.447× | error 0 |

四项均使用 `dense-fixed-values`，Clang 18、`-O3 -ffp-contract=fast`，10 repetitions。runner
现在按 target/shape 绑定这些已实测 auto 参数；canonical F32 tree 保持一份。

四个最终 binding 生成的 RISC-V IR 均可由 `weft-opt` 独立 parse/verify，安全运行
canonicalizer、CSE、layout canonicalization、Share 两次与 final verifier；同一流水第二次文本
diff 为零。四项的 vector peak 分别为 16、24、20、24 groups，均低于 32-group 合同。

## 4. Q1_0 single-output packed contraction

### 4.1 七步工作账与参照

旧 SG2044 Physical IR 让 32-element reduction carrier 分成两个 16-lane parts。每个 128-element
record 因而执行 8 次 mask load、8 次 Q8 byte load、8 次 widened product，随后才做 4 次
reduction。GGML VLEN128 donor（`source/c/ggml/llama.cpp/ggml/src/ggml-cpu/arch/riscv/quants.c:523-559`）
以 i8m2 保持完整 32-lane carrier：每个 sub-block 各一次 `vlm`、`vle8`、`vneg`、`vmerge` 与
`vwredsum`，不建立显式 product vector。这里首先是 P1 carrier 与 P3 selected leaf，不是 P2/P4。

第一次把整个 `FuseRISCVBitplanes` 提前到 nested-memory planning 之前，Q1 能形成 signed leaf，但
96-entry 机械扫描新增 7 个 compile failure：IQ2/IQ3 的 decode/window 关系被过早物化。该排程已
删除。最终实现保留原 pass 顺序，让 nested-memory owner 先建立 `rvv_bitmask_window_load`，再由
fusion 把其 typed mask result 与 signed-i8 data 合成 `rvv_signed_bitmask_reduce`。signed op 现在
消费真实 mask SSA value；mask 地址、window 与 memory effect 属于前一个 op，terminal emitter不再
从 field/origin/point 重建地址。

这和 Triton 把 masked load 的 vector width约束在 mask alignment 上、随后将 mask作为普通SSA值
交给消费者的分层一致（`LoadStoreOpToLLVM.cpp:161-173`）；TileLang reducer同样只在 layout冻结后
物化 update/finalize（`reducer_plan_materialize.cc:5-11`）。两者都没有 Q1 的专用 signed-mask
reduction leaf；Weft保留该局部 target operation，但不让它拥有 memory edge。

SG 汇编在上述变化后已与 donor具有相同的每-record向量工作计数，但第一次真机仍只有约
3.4--4.6 GOP/s。硬件计数显示两者 retired instructions 与 cache misses基本相同，而 Weft cycles
约为 donor的1.78倍。逐段排列的唯一实质差异是 Q1 scalar scale 在四次 reduction之后才 load；
临时 C 变体只把这一次 load移到 loop前，吞吐即达到 11.553 GOP/s。这个实验不删除指令，收益来自
增加 load-to-use距离，因此属于 P5；若位置不变，该机制没有实现价值。

`HoistRISCVLoopInvariants` 现在对至少两次迭代、全体 effect均为 read/pure、field owner定义在 loop外
的物理 loop，在 preheader建立显式 `register_materialize<physical-share>`。`physical_point` 是无
memory effect的坐标构造，但为保留 iteration identity刻意不带通用 `Pure` trait；pass按这个精确
合同处理它。Triton 的 LICM 对 readonly load要求 loop只读（`LoopInvariantCodeMotion.cpp:20-77`），
其 pipeliner utility则以 dominance和producer DAG决定可移动集合（`PipeliningUtility.cpp:42-105`）；
TileLang最接近的合同是 scalar use-def加“不读 loop内写入 buffer”
（`loop_unswitching.cc:350-387`、`bind_utils.h:22-49`）。当前 Weft采用与 Triton LICM同样保守的
whole-loop-read-only边界，没有假设尚不存在的精确 alias证明。

第二输入静态检查中，Q2_K新增3个 preheader scalar materialization、Q3_K新增2个；Q2_K的既有
local bytes/peak仍为32/18，三次短跑为 SG 6.335/6.218、K1 3.443/3.414 GOP/s（standalone/decode），
与全量账处在同一档，没有用 Q1收益掩盖横向退化。Q3_K的实际收益留给下一簇判定，不在这里宣称。

### 4.2 参数与十次真机结果

SG 的完整 32-lane carrier需要 m2；K1 的 VLEN256以 m1即可承载。K1 对 unroll=1/2/4/8的短扫分别
约为3.71/4.22/4.45/3.71 GOP/s，最终绑定 unroll=4。它只机械减少 issue-loop控制并扩大独立
signed leaves，不改变作者树或 reduction结构。

| target | entry | 修改前 | 修改后 | source | Weft/source | numeric |
|---|---|---:|---:|---:|---:|---|
| SG2044 | standalone vec-dot | 4.622418 | 11.651234 | 11.448178 | 1.018× | error 0 |
| SG2044 | MUL_MAT decode | 4.315042 | 11.709463 | 10.371659 | 1.129× | error 0 |
| K1 | standalone vec-dot | 2.798971 | 4.538486 | 3.772297 | 1.203× | error 0 |
| K1 | MUL_MAT decode | 2.807047 | 4.536049 | 3.691932 | 1.229× | error 0 |

四项均为 10 repetitions、相同合法 record输入和 Clang 18 flags。standalone/decode同步过线，说明
收益位于共享 contraction底座，不依赖 GEMM wrapper。SG final IR peak为7 groups/local 0，K1为
4 groups/local 0。

24 row-dequant + 24 vec-dot × 2 targets再次机械验收：91份用默认 profile直接通过；其余5份使用
production runner已记录的 physical auto binding。合计96/96可生成、独立parse/verify，并安全运行
canonicalizer、CSE、layout canonicalization、Share两次与final verifier；同一流水第二次文本
diff为0。
