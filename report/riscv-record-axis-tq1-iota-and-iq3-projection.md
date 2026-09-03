# Record-axis 物理化、TQ1_0 iota 与 IQ3_S projection

本报告记录同一 checkout 上的定向结果。正式性能 CSV 未更新；按实验协议，定向回归只写入
当轮报告。所有真机数值检查使用相同固定 seed 输入和 tolerance 判据。

## 1. 动手前的工作账与 reference 对照

### 1.1 跨 record loop vectorization

按 `doc/compiler/optimization-principles.md` 第九节逐项检查：

1. **carrier**：row-dequant 的外层 `blocks` Level 原本骑 issue time，record 内 32 个输出骑
   RVV lane；待验证结构把多个完整 `blocks` instance 放进 lane，每个 record 内的位置改为
   静态 issue。
2. **工作量**：以 SG2044、LMUL=m1 的 Q4_0 为例，四个 record 的原路径动态执行约 4 次
   packed unit load、24 次 slide 和 32 次 unit store；record cohort 路径执行 17 次 strided
   load、0 次主循环 slide 和 32 次 strided store。计数会变化，因此不是“只给工作命名”。
3. **供应 identity**：一个 cohort 内每个 field storage byte 只加载一次；Q4_0 的同一 byte
   同时覆盖 low/high nibble，scale 每个 record 只取一次。
4. **memory form**：变化是 unit record load/store 到跨 record 的 `vlse`/`vsse`，不是地址
   字符串的等价改写。
5. **reduction**：row-dequant 没有 contraction/reduction，本机制不依赖 partial 系列 pass。
6. **pipeline**：本轮没有把该循环流水化；main/tail 拆分只处理完整 cohort 与余数语义。
7. **donor 与 owner**：同 flags 编译的 SG2044 Q4_0/Q8_0 donor 汇编会把 record induction
   变成 strided vector memory；Q5_0/Q5_1 donor 则保留 scalar record loop，在 record 内处理
   q/qh。source 数值循环见 `source/c/ggml/llama.cpp/ggml/src/ggml-quants.c:401-509`。

TileLang 在 `ref/tilelang/src/transform/loop_vectorize.cc:1003-1042` 只对静态、zero-based、
extent 可整除的 loop 做 strip-mine，并在 `:1115-1170` 证明 substituted index 是 scalar 或
unit-stride ramp；证明失败就保持 scalar。Triton 的 AxisInfo 在
`ref/triton/lib/Analysis/AxisInfo.cpp:493-589` 沿 add/addptr 传播 contiguity、constancy 和
divisibility，load/store width 再消费这些 facts。Weft 不能直接复用它们的 SIMT layout 或 TIR
Ramp，因为 Weft Level point 是完整 logical value 的坐标，不预先属于 thread；可复用的是
zero-based boundary、无 carry/effect、静态 partition、unit record stride和显式 main/tail proof。

### 1.2 TQ1_0 prefill iota

失败的 `[5]` iota 已经有完整 logical axis。layout canonicalization 将它物化成 scalar carrier，
但保留 `time_factors=[5]`；旧 terminal selection 只检查 register-replica product，因此错误选择
`scalar.iota`。数学工作量、作者 axis 和 Level 均不需要改变；缺的是 leaf selection 对已有
physical layout 的完整消费。

### 1.3 IQ3_S

IQ3_S 的两个相邻 4-byte codebook entry 与 sign nibble 是两条不同 storage relation。现有
`rvv_bitmask_window_load` 能证明完整 8-bit grouped/layered window，却不能把一个 4-bit half-byte
投影单独当成闭合 byte window。24 个 Encoding 的静态扫描没有第二个同时具有“相邻多 entry
payload + 独立 half-byte sign projection”的输入；IQ3_XXS 的 sign 位来自同一个 u32 metadata，
不是这个关系。因此本轮没有新增 two-entry 或 half-byte composite op。

## 2. 实现结果

### 2.1 Record cohort 是可观察的 Physical IR

RISC-V Physical IR 新增：

- `RecordCohortType` / `record_cohort`：绑定 Level domain、静态 partition 与 lane cohort width；
- `rvv_record_storage_load`：保存 field、storage unit、logical ownership、selected byte stride和
  exact strided leaf；
- `rvv_record_storage_decode`：显式保存 sub-byte shift/mask；
- `rvv_record_store`：保存 output element offset、selected byte stride和exact strided leaf。

定义与 verifier 位于 `include/Weft/Dialect/RISCV/IR/RISCVOps.td:758`、`:1522` 和
`lib/Dialect/RISCV/IR/RISCVDialect.cpp:6924`。`VectorizeRISCVRecordLoops` 在
`lib/Target/VectorizeRISCVRecordLoops.cpp` 证明 closure，并生成整 cohort 的主
`scf.for` 和保留原语义的 scalar tail。Emitter 只消费已经选定的 storage index、shift/mask、
byte stride与leaf，见 `lib/Target/RISCVIntrinsicC.cpp:10554-10678`。

提交前的独立语义审计又补出了四项原 matcher 没有闭合的 proof：完整 cohort 现在要求
`active == partition`；source slice必须和 destination 共享同一个 physical point/domain；store 的
pointwise closure不能引用将被替换的旧 induction variable；未知 effect 和闭包外 read 都拒绝，
只有闭包外 dead pure op 可以随原 loop 一起删除。RVV layout verifier 同时新增 VLMAX 和目标
vector-register 数量上界，storage bit offset 的乘加改为 checked arithmetic。上述约束都在
Physical IR verifier中复算，不依赖 emitter。

该结构选择进入 target profile：`within-record` 与 `across-records` 是固定优先级，不是 source
构造、format 分支或 tuner 参数。两个当前实测 profile 默认 `within-record`；定向 repro 通过
`--record-axis-policy=across-records` 验证另一种合法物理程序。Q4_0、Q4_1、Q8_0 能形成
record cohort；Q5_0/Q5_1 的 qh 已先成为 `rvv_bitmask_window_load`，不是这条只消费 encoded
Field + pointwise closure 的 relation，且 donor 本身也没有采用 cross-record 组织。单独放宽
“shaped Field 数量”不会改变这一点，静态实验仍形成 0 个 cohort，代码没有保留。Q8_0 是
独立于 Q4 的第二个输入。
强制 `across-records` 时，Q5_0/Q5_1 在两台 target 上生成的 terminal C 与默认路径逐字节相同
（4/4）；这证明它们没有被该机制暗中接管。

### 2.2 TQ1_0

`RISCVPhysicalSupport.cpp:929-957` 现在同时计算 time 与 replica product；任一 scalar-carrier
value 的 `time × replica > 1` 时选择 `register.iota`。TQ1_0 prefill 的最终 IR 含 2 个
`register.iota`、0 个 `scalar.iota`。该规则只读取 layout，不识别 TQ、kernel 或 target 名。

## 3. 真机结果

### 3.1 Record-axis 正负结果

下表都是 10 repetitions；source 来自固定 baseline。`across/default` 用同一 checkout、同一轮
运行直接比较。

| format | target | default within-record | forced across-records | source | across/default | forced/source |
|---|---|---:|---:|---:|---:|---:|
| Q4_0 | SG2044 | 744.409 MElements/s | 215.626 MElements/s | 426.623 | 0.290× | 0.505× |
| Q4_0 | K1 | 432.101 MElements/s | 224.214 MElements/s | 171.620 | 0.519× | 1.306× |
| Q4_1 | SG2044 | 742.691 MElements/s | 211.298 MElements/s | 372.007 | 0.285× | 0.568× |
| Q4_1 | K1 | 277.431 MElements/s | 223.085 MElements/s | 150.617 | 0.804× | 1.481× |
| Q8_0 | SG2044 | 730.544 MElements/s | 206.019 MElements/s | 333.989 | 0.282× | 0.617× |
| Q8_0 | K1 | 502.223 MElements/s | 213.219 MElements/s | 183.911 | 0.425× | 1.159× |

所有十二次运行均为 `within-tolerance`，观测到的最大绝对/相对误差都是 0。负结果不是 LMUL=m1
偶然选坏：Q4_0/SG 的 forced 路径扫到 m2=343.040、m4=389.209 MElements/s，m8 因 48 个
vector groups 超预算而合法失败；Q4_0/K1 的 m2/m4 为 231.665/244.278。Q8_0 的 3-repetition
筛查在 SG 为 m2=299.077、m4=329.554，在 K1 为 208.109/216.843，仍明显低于同轮 default。

结论是：跨 record carrier 是真实可执行能力，并在 Q4/Q8 两种 Encoding 上由同一 pass 形成；
但 `vlse/vsse` 的代价高于它消掉的 unit load/slide 工作，不能成为当前两台机器的默认结构。
因此默认 IR 和已过线性能没有被这个负实验覆盖。

### 3.2 TQ1_0 prefill

| target | 修前 | 修后（10 repetitions） | source | 修后/source | numeric |
|---|---|---:|---:|---:|---|
| SG2044 | compile failure | 6.098 GOP/s | 6.181 | 0.987× | within-tolerance，误差 0 |
| K1 | compile failure | 4.332 GOP/s | 3.456 | 1.253× | within-tolerance，误差 0 |

这项改动的验收是恢复一棵已存在 blocked prefill 程序的合法 leaf，不是本轮性能优化；SG 的
2% 差距没有被隐藏成通过。

## 4. Physical IR 机械验收

默认 target profile 下，24 个 row-dequant 与 24 个 vec-dot 在两台机器形成 96 份 Physical IR：

```text
generated                         96/96
parse + verify + canonicalizer+CSE 96/96
Share 连跑两次                   96/96
同一整套 pipeline 第二次文本 diff=0 96/96
```

每份依次运行 `--canonicalize --cse --weft-riscv-canonicalize-layouts`
`--weft-riscv-share-layered-windows` 两次、`--weft-riscv-verify-final --verify-each`
`--verify-roundtrip`，随后整套再运行一次。另对 Q4_0/Q4_1/Q8_0 × SG/K1 的六份
`across-records` IR 做相同检查，结果为 6/6 parse/verify、6/6 CSE/Share 安全、6/6 第二轮
diff 为零；每份都含一个真实 `record_cohort`。这同时验证了 Share 不会再把已经物化的
record storage edge 二次改写成 layered stream。

## 5. 未伪装成已解决的边界

- IQ3_S 仍是 SG2044 258.543/350.612 MElements/s（73.7%）的已知 P3 差距；K1 已是
  156.072/152.032（102.7%）。half-byte relation 单独不足以决定 donor 的 two-entry unit-load
  program，而组合合同没有第二个输入；本轮代码保持不变。
- record cohort 当前只证明一维 unit-stride source/destination、zero-based unit-step exact Level、
  单 store 与 pure pointwise closure。多轴 record view、carry、write/unknown effect、动态 stride
  或多 storage root 不会被猜测物化。
- `across-records` 是合法但在当前两台机器上较慢的 target structural priority。它不是 fallback，
  也没有写入正式性能 CSV；默认 `within-record` 的 Q4/Q8 数字由当轮真机复测确认未退。
