# 物理决定体系第一轮

本轮固定四十个 kernel、四十二条 phase，不修改 DSL 算法。工作只落在 RISC-V target
lowering：物理决定的唯一生产、候选空间及其资源合法性，以及这些决定的机械发射。

## 权责收口

此前 block decode、相邻 block store 分组和 block reduce 分组仍在 emitter traversal
中首次识别和选择。本轮把三者全部前移到统一 physical preparation：

- `DecodeOp` 按 typed codes/table/result 与 target facts 产生唯一 decode decision；
- block store 先决定 owner、共享 axis、producer closure、相邻 sink group 和 strip
  realization；
- block reduce 同样先决定 group、closure 和 reduction strip realization；
- emitter 只查询已准备的 decision，不再重新收集 use closure、决定分组或选择
  byte shape。

VLA、contract、quantized extension primitive 和 IME 原有的 decision producer 保持不变。
逐项审查没有发现除上述三类之外的高置信“analysis 与 emitter 对同一物理事实各选一次”。

## 候选空间

候选全部是一次 target lowering 内的瞬态对象，不进入 Kernel IR 或 artifact。

### VLA

- 普通 VLA LMUL 继续由 access、state、predicate、nested traversal 和寄存器组数选择；
- 存在外围 scalar indexed load 的 VLA，按这一局部 memory/context fact尝试
  `LMUL={1,2,4,8}`，ADD_ID 因而选择 LMUL1；
- f32→i8 narrow 从固定默认扩成 `{8,4}`，合法性计入 f32 live vectors、两级
  narrowing intermediate 和 allocator headroom；
- relaxed add/max reduction增加 `ScalarCarry` 与 `VectorCarry` placement。只有动态、
  非 tiled、非 conditional region 选择 vector carry；静态 block、tiled quantization、
  conditional dot、ordered scan 和 summary algebra保留各自原 realization。

### Contract

- F32 contract 的 LMUL 与 `K-unroll={1,2,4}` 形成笛卡尔候选，并由 accumulator、
  streamed operands 与 target vector-register 数过滤；
- F16 contract 的 row microtile divisors 与 input LMUL `{1,2,4}` 形成候选，资源式为
  `(3 * rows + 1) * inputLMUL + allocator headroom`；
- 真机实测表明 F32 K-unroll 2/4 在当前 load projection 中显著退化，因此该维度
  保留为可显式选择的合法空间，默认排序仍以 unroll1 优先；F16 默认优先 row reuse，
  不以更宽 LMUL替代作者 row tile。

### Block 与 extension

- block store的 `e8mf4 micro / e8m1 fixed / e8m1 dynamic` 与 block reduce的
  `e8m1 fixed / dynamic` 已改为枚举候选；typed closure、lane-index需求和寄存器资源
  共同决定合法性；
- IQ4 decode、Q1 sign-bit dot、MXFP4 E2M1 table-dot和 IME leaf继续由各自 typed
  primitive拥有。Donor审查确认 MXFP4与当前GGML VLEN128指令序列已同构；IME1当前
  只有经过真机验证的 VLEN256 4×4×8 fragment，不虚构第二个候选；
- 没有引入 kernel name、q-format route、GGML/materials runtime调用、legacy fallback
  或 whole-region selector。

## 没有做的算法改写

Q8_0的两个显式 VLA traversal理论上可以做跨 region fusion，但这会把作者明确分开的
reduce pass与narrow pass合并，违反本轮冻结的授权边界，因此没有实现。Prefetch、
indexed RVV gather和跨 primitive pipeline也没有从普通 scalar/VLA source中凭空发明。

F16 dot、online softmax、F16 GEMM和 IME 的窄 local closure均被审查。固定四十项中没有
一个自然表达被这些条件错误拒绝，所以没有为了“看起来更通用”而放宽未经当前语料覆盖的
legality。

## 真机结果

四十二条 phase均沿唯一 production path完成 DSL→Kernel IR→RISC-V physical
decisions→intrinsic C/local asm→system compiler→真机执行。SG2044使用 RV64GCV VLEN128，
K1/X60使用 RV64GCV VLEN256 + IME1。当前数字已写入
`report/weft-kernel-performance.csv`。

相对本轮开始时同一 CSV 的四十二条 phase，median time几何平均比为 **0.979**，即约
**2.1%** 改善。主要变化为：

| 压力点 | 本轮前 median ms | 本轮后 median ms | 变化 |
| --- | ---: | ---: | ---: |
| RMSNorm | 1.365566 | 1.012684 | 快25.8% |
| LayerNorm | 1.442806 | 0.982864 | 快31.9% |
| ADD_ID | 7.701514 | 6.328268 | 快17.8% |
| F16 GEMM prefill | 379.894115 | 352.083699 | 快7.3% |
| Q4_K×Q8_K decode | 13.462768 | 12.842926 | 快4.6% |
| Dense Conv | 804.604125 | 788.391226 | 快2.0% |

F32 GEMM prefill仍为754.323717 ms，说明当前缺口不在合法 LMUL/K-unroll枚举，而在
contract-local load schedule、reuse与真正 pipeline。Q8 narrow采用更宽合法候选后为
3.516246 ms，变化很小；Q1与MXFP4 leaf仍约为56.218825 ms和19.581418 ms，donor对照未
发现可在不扩大 primitive envelope的情况下直接替换的更快 local sequence。

本轮首次全量过程中，Q4_0 IME曾因 tiled max reduction错误选择 vector carry而出现确定性
数值错误。决定条件改为读取 VLA end的local producer事实后，tiled reduction恢复
ScalarCarry；最终重跑 activation code mismatch为0，最大绝对误差0.0312509537，
median 3.159286 ms。这个问题没有以 kernel或format名称修补。

## 当前边界

主要 primitive现已遵守 analysis/decision→emission的单向权责，且 VLA、state、contract、
block strip和narrow均拥有受资源约束的候选空间。仍未成熟的是 contract-local
load/prefetch/pipeline、真正 indexed RVV memory、以及超出当前唯一已验证 envelope的IME
fragment；它们在本轮保持明确未实现，没有静默 fallback。
