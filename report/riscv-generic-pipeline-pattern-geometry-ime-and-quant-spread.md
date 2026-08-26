# RISC-V 通用流水、typed geometry、IME production 缺口与量化树铺开

## 1. 本轮问的问题

本轮没有用“不含格式名”或“决定在 pass 中”作为成功证据。检查方式是：

1. 阅读 Triton 与 TileLang 的同类实现，确认 scheduler 和 expander 分别负责什么；
2. 用没有为新 pipeline 改写过的 Q4_0 blocked tree 与 F16 blocked tree 验证该机制是否真正形成流水；
3. 在 `convert_layout` 与 index cast 插入后再跑 layered-window sharing，检查 pass 读的是 typed storage geometry 还是原始 SSA 拼写；
4. 把 Q4_1 和 IQ4_NL 改成 blocked MUL_MAT，看哪些只需改 std，哪些真的暴露 target 表示传播缺口；
5. 不实现 IME production path，只对照 K1 donor 把 author tree、derived Encoding 与 compiler fragment mapping 的缺口分开。

实验中保留 generated C/assembly 的 `WEFT_KEEP_ARTIFACTS` 仅用于当轮检查；这些临时产物不进入仓库。

## 2. Pipeline：从专用两 op matcher 到 scheduler + expander

### 2.1 参考实现实际怎样分工

Triton 的 `PipelineExpander` 接收 `(Operation *, stage)` schedule，不自己选 stage：

- `../ref/triton/include/triton/Dialect/TritonGPU/Transforms/PipelineExpander.h:26-32,79-96`；
- latency 与 stage/cluster 安排位于
  `../ref/triton/lib/Dialect/TritonGPU/Transforms/Pipeliner/AssignLatencies.cpp:302-330` 与
  `../ref/triton/lib/Dialect/TritonGPU/Transforms/Pipeliner/ScheduleLoops.cpp:152-246,302-386`；
- expander 负责 schedule legality、跨 stage SSA live range、prologue、steady-state 与
  epilogue，见
  `../ref/triton/lib/Dialect/TritonGPU/Transforms/Pipeliner/PipelineExpander.cpp:130-275,296-453,480-577,657-851`；
- 实体 buffer allocation/versioning 在 expander 之外的
  `../ref/triton/lib/Dialect/TritonGPU/Transforms/Pipeliner/LowerLoops.cpp:76-130,434-624`。

TileLang 也是两段责任：`PipelinePlanning` 从 buffer read/write region 和 scalar use-def
生成 stage/order，`InjectSoftwarePipeline` 校验依赖、计算 buffer version，再展开三段结构：

- `../ref/tilelang/src/transform/pipeline_planning.cc:108-257,454-475,634-760,1172-1337`；
- `../ref/tilelang/src/transform/inject_pipeline.cc:1195-1250,1401-1529,3095-3177`。

这两份实现共同的机制不是“识别 GEMM 然后变成 depth 2”，而是先从真实程序得到
per-operation schedule，再由独立 expander 保持 SSA/effect 合同地改写 loop。

### 2.2 本轮的程序结构

之前 `PipelineRISCVLevels` 要求 loop body 恰好是一个 grouped/encoded window load 和一个
step，要求单 carry，然后手写这一种两阶段结构。这个限制已被替换为两个 pass：

- `ScheduleRISCVLevels`：消费 `LowerRISCVComposites` 产生的、带 `weft.riscv.schedule` 的
  physical `scf.for`、loop carry、SSA use-def 和 effect，
  在具体 operation 上写 stage/order。它不要求两个 op、相邻 op 或特定 window family。
- `PipelineRISCVLevels`：只消费已安排的 stage/order，为任意数量的跨 stage SSA value
  增加 steady-loop arguments，支持多个 source carries，生成 empty guard、prologue、steady-state
  与 epilogue。

当前能力边界仍然是距离一迭代、depth=2/buffer=2。scheduler 只允许顶层 pure/read
operation；嵌套 region、write 或 unknown effect 因没有 predication/ordering 合同而明确失败。
这是目前已证明安全的子集，不是“支持任意 loop body”的宣称。

pipeline 展开后，source Level identity 附着在外层 guarded physical region，final verifier 检查
handoff arity 与其 logical point/domain；steady loop 保留有序方向。

### 2.3 外部结果

下列是本轮临时运行记录中 production shape、真机数值检查通过后的调查性
3-repetition 流水 A/B；generated artifacts 已按约定删除。确定使用的当前路径另行按
10-repetition 协议重跑并更新 `weft-kernel-performance.csv`。A/B 只改
`--auto-pipeline-depth`，其余 tree/meta/target 不变。

| Kernel | Target | depth 1 | depth 2 | 变化 | 观察 |
|---|---:|---:|---:|---:|---|
| Q4_0 blocked prefill | SG2044/VLEN128 | 7.777527 | 11.773182 GOP/s | +51.4% | 没有为新 pipeline 改写 tree，生成真实跨 K-block overlap |
| Q4_0 blocked prefill | K1/VLEN256 | 2.261673 | 2.769195 GOP/s | +22.4% | 同一 pass 在第二 target 上同样生效 |
| Q4_1 blocked prefill | SG2044/VLEN128 | 6.620159 | 9.627385 GOP/s | +45.4% | 第二棵同类 tree 上生效 |
| Q4_1 blocked prefill | K1/VLEN256 | 1.960523 | 2.665530 GOP/s | +36.0% | 第二 tree/第二 target 交叉验证 |
| F16 blocked prefill | SG2044/VLEN128 | 15.856819 | 14.450291 GOP/s | -8.9% | KC=K=4096 使 source K Level 只有一次迭代，只增加 guard/prologue/epilogue，无重叠可获得 |
| Q4_0 decode | SG2044/VLEN128 | 7.615277 | 7.515899 GOP/s | -1.3% | GEMV 特化不从两阶段获益 |
| IQ4_NL blocked prefill | SG2044/VLEN128 | 8.638911 | 8.577386 GOP/s | -0.7% | lookup 路径在当前 cluster 上没有流水收益 |

最终选定的 Q4_0 depth-2 prefill 再按 10 repetitions 重跑：SG2044 为
11.766204 GOP/s，K1 为 2.784785 GOP/s，两者数值检查均通过。

因此 runner 中 Q4_0/Q4_1 prefill 使用 depth 2，其 decode 与 IQ4_NL 保留 depth 1。
这不是按格式选 leaf：这些是 source/std 已实例化的 physical parameter bindings，正负结果一起
说明 depth 参数不能由 pass 猜成固定值。

可手动复现的命令形式为：

```bash
WEFT_AUTO_PIPELINE_DEPTH=1 ./examples/run/weft-mul-mat.sh sg2044 q4_0 prefill 3
WEFT_AUTO_PIPELINE_DEPTH=2 ./examples/run/weft-mul-mat.sh sg2044 q4_0 prefill 3
WEFT_AUTO_PIPELINE_DEPTH=1 ./examples/run/weft-mul-mat.sh k1 q4_0 prefill 3
WEFT_AUTO_PIPELINE_DEPTH=2 ./examples/run/weft-mul-mat.sh k1 q4_0 prefill 3
```

## 3. Share/Fuse：从语法拼写到 typed geometry

### 3.1 `ShareRISCVLayeredWindows`

之前该 pass 要求两个直接 `Field -> Extract`、一维 result、直接 `PhysicalPoint`、
相邻的 parent/index 形状。现在判定使用：

- typed `AccessAttr(mapping=grouped_layered, group, layer, order, storage_bits)`；
- field owner/name、point parent identity，以及 point domain axis 在 result 中的 layer extent；
- 带 integer index cast 的线性 byte-coordinate 关系；
- 穿过保持 shape/axis 的 pure conversion；对 `local_load` read conversion，还要求
  两次 read 之间没有 write/unknown effect，共享 op 本身保留一次 typed memory read；
- 保留其他 logical axes，只要 point 对应的 axis extent 等于 layer extent。

成功时 pass 实际用一个 `rvv_layered_window` op 替换两个 extract。Q4_K row-dequant
在 final RISC-V IR 中从 0 个该 op 变为 1 个，并在两台机器上数值一致：

| Kernel | Target | 结果 |
|---|---|---:|
| Q4_K row dequantize | SG2044/VLEN128 | 469.994263 MElements/s |
| Q4_K row dequantize | K1/VLEN256 | 314.179334 MElements/s |

Q4_K persistent 的 2-D tail cohort 仍没有生成 `rvv_layered_window`；原因是 result validity 为
tail，当前 pass 没有证明共享窗口不会越界。该路径不在本轮被写成“已泛化”。

### 3.2 横向 pattern 审查的实际结论

`FuseRISCVBitplanes` 已经有强 typed geometry guard，但它仍要求
`or -> shl -> and -> shr -> extract -> div/mod -> iota` 的一种直接 use-def 链。
`LowerRISCVComposites` 中的 partitioned reduction、stream dot、encoded dot、grouped MAC 也仍有
类似的 direct-defining-op 门槛。这些是当前剩下的源码拼写依赖，本轮没有把它们写成
“全部已解决”。

Triton 的对应机制是以 layout anchor、axis/contiguity/order facts、forward propagation、
backward slice rematerialization 和 dominance-order rewrite 为基础：

- `../ref/triton/lib/Dialect/TritonGPU/Transforms/RemoveLayoutConversions.cpp:42-59,218-348,895-960,1220-1464`；
- `../ref/triton/lib/Dialect/TritonGPU/Transforms/Coalesce.cpp:77-119`；
- `../ref/triton/lib/Dialect/TritonGPU/Transforms/CoalesceUtils.cpp:31-55`。

本轮另外收紧了 `CanonicalizeRISCVLayouts`：只有 effect=`pure` 的 conversion 可以被当作
逆 conversion 消除、rematerialize 或 CSE。`local_load`/`local_store` 等 read/write conversion 不能
仅因 source/result type 相同就被删除。

## 4. Target facts 跨 lookup/storage 链的传播

IQ4_NL blocked tree 在 SG2044 上可运行，但第一次在 K1/VLEN256 上发射失败：

```text
encoded field has no selected numeric load realization
(field=q, mapping=grouped_layered, form=indexed, selected-carrier=3)
```

不是 emitter 少了一个 IQ4_NL 分支。RISC-V IR 显示，VLEN256 把 lookup result 提到 32 lanes，
但其 index 来自只有 16 元素 storage layer 的 `grouped_layered(layer=16)` field；之前
storage width 没有穿过 `convert_layout -> lookup` 传给下游。

`PropagateRISCVLayouts` 现在沿 shape/axis-preserving pure/read conversion 与 lookup index 链传播
encoded lane limit。修正后 K1 选择 16-lane × 2 issue-time strips，真机数值检查通过，
吞吐为 2.746756 GOP/s。SG2044 同一 tree 为 8.637518 GOP/s。这条结果说明
VLEN 变化不再只改一个孤立 lane 数；storage geometry 已经参与了实现合法性。

## 5. IME 从 blocked tree 到 production fragment 缺什么

### 5.1 Donor 和当前 Weft leaf 不是同一条数值链

K1 donor 的 Q4_K 路径不是把 canonical nibble 直接塞入一条 MMA：

- canonical Q4_K/Q8_K 字段和 min correction 见
  `source/c/ggml/llama.cpp/ggml/src/ggml-cpu/arch/riscv/repack.cpp:287-385`；
- persistent x8/x16 布局的 row interleave、nibble 和 scale high/low 重排见
  `source/c/ggml/llama.cpp/ggml/src/ggml-cpu/repack.h:43-58` 与
  `source/c/ggml/llama.cpp/ggml/src/ggml-cpu/repack.cpp:2836-2957`；
- activation 以 4-row group 量化，见
  `source/c/ggml/llama.cpp/ggml/src/ggml-cpu/spacemit/ime1_kernels.cpp:97-155`；
- IME2 donor 的 `MB_ROWS x NB_COLS` 组织及 zero-point correction 见
  `source/c/ggml/llama.cpp/ggml/src/ggml-cpu/spacemit/ime2_kernels.cpp:35-115`。

当前 Weft target profile 只声明一种 signed-i8 `4x4x8` IME1 capability，packing 也固定为
lhs 4x8、rhs 4x8-transposed、accumulator 4x4：

- `include/Weft/Target/RISCVTargetProfile.h:21-44`；
- `lib/Target/RISCVTargetProfile.cpp:211-227`；
- `lib/Target/RISCVPhysicalSupport.cpp:365-384`。

`LowerRISCVComposites.cpp:1847-1949` 只能将 homogeneous i8 contract 变成
`ime_pack -> mma -> ime_unpack`；`RISCVIntrinsicC.cpp:5620-5838` 也只会将已经形成的 i8 operands 写成
4x4x8 helper。它不包含 Q4_K 的 q/sc/m/bsum/d/dmin/ds 数值链。

### 5.2 责任分类

作者程序必须写：

- `d*sc`、`-dmin*m`、Q8_K `bsum` correction 及 fp16 舍入位置；它们改变 logical
  intermediate values，是另一棵 std 数值树；
- 跨调用保留的 row-major nibble/scale tile；它改变 artifact bytes 与 ABI，是 derived
  Encoding；
- source `group=16` cohort 和它的 Level/birth/handoff。

编译器必须求：

- 把已存在的 group=16 logical cohort strip-mine 成多个 4-row/4x4x8 physical fragments；
- fragment operand layout、nibble/byte 的 primitive-local pack 顺序、多 accumulator 与 K schedule；
- RVV value 与 fragment 之间的 typed conversion/handoff 及资源。

cohort width 和 fragment 形状不需要在 core DSL 中绑定。现有 IR 已有 `FragmentType`、
`FragmentPackingAttr`、`ime_pack/unpack/mma`；group=16 可以作为一个 source cohort 被物理分成
多个 4-row fragments，不能被 compiler 改写成四个 source `group=4` Levels。当前缺的是
Q4_K typed tree/derived Encoding 到这些 fragment operations 的 lowering 与资源闭合，不是新 DSL
permission 或 engine annotation。

RVV 和 IME 应是两个 std 函数由调用方选择：它们的 logical intermediates、materialization
和 persistent ABI 不同，因此是两个作者程序。build config 可以要求 IME 且在无合法 leaf
时失败，但不替调用方选程序。

## 6. 量化 MUL_MAT 与 shaped 程序铺开

本轮没有一次改 24 个格式。实际铺开如下：

| 项目 | std/source 改动 | compiler 改动 | 外部结果 |
|---|---|---|---|
| Q4_1 MUL_MAT | blocked N/M tiles、materialized W/Xq、MRxNR acc、K32 outer contract；另保留 GEMV decode specialization | 未增加 Q4_1 规则；使用本轮通用 pipeline | SG prefill 9.641521、decode 6.319240 GOP/s；K1 prefill 2.612493、decode 2.261088 GOP/s，数值通过 |
| IQ4_NL MUL_MAT | blocked N/M tiles、materialized codebook/W/Xq、shaped lookup、MRxNR acc；另保留 decode specialization | 需要一个通用的 conversion/lookup storage-width 传播修正 | SG prefill 8.637518、decode 5.846692 GOP/s；K1 prefill 2.746756、decode 2.179507 GOP/s，数值通过 |
| Q1_0 row dequantize | 把 `for j in range(128)` 改成显式 `iota(128, axis="k")` shaped bit decode/commit | 无 | SG 117.518477、K1 339.325135 MElements/s，数值一致 |
| Q1_0 vec-dot | 尝试将 scalar loop 改为 shaped sub-axis，但没有保留未闭合改动 | 无 | frontend 拒绝：`w.q[xb]` 是 rank-0 storage field，再用 lane index 会“too many indices for logical value” |

Q1_0 vec-dot 的阻塞不是 target 应该从 scalar loop 猜回一条 axis。当前 Q1_0 Encoding
将 `q` 声明为 `u8[16]`，但 128 个 logical bits 与 storage bytes 的 sub-Level projection 仍留在
numerical tree 的 index/shift 中。row-dequant 可以用完整 128-axis 一次表达；vec-dot 对某个
sub-Level 取字段时没有一个 typed logical bit-axis projection。本轮没有为了继续铺树而让
compiler 从循环次数猜 axis，也没有临时增加新语言构造。

## 7. 当前仍然没有闭合的事实

1. 量化铺开完成了 Q4_1、IQ4_NL 和 Q1 row-dequant，不是全部格式。Q1 vec-dot 暴露了
   logical bit axis 与 storage field 之间的源表示问题。
2. `production_mul_mat_q4_k_staged` 在本轮 head 上仍在 selected RVV widening-dot lowering 失败，
   报 operand/result shape 不闭合；
   depth=1 也失败，因此不是新 pipeline 引入的回归。persistent Q4_K
   路径重跑数值通过，SG2044 为 10.123244 GOP/s。
3. `ShareRISCVLayeredWindows` 已经穿过 representation conversion/index cast，但 tail cohort
   还没有 safety proof。`LowerRISCVComposites` 中仍有多个 exact SSA closure。
4. pipeline 现在是通用 top-level pure/read dependency cluster 的两阶段 expander，不是 Triton/TileLang
   的完整异步、多 stage、effect-predicated pipeline。F16 的负结果证明这一边界会真实影响性能。
5. K1 production runner 仍未接通 Q4 的 IME tree/derived Encoding/fragment path；当前只有独立
   signed-i8 4x4x8 local contract leaf 的证据。

## 8. 这轮实际说明了什么

- 在本轮 Q4_0/Q4_1 blocked tree lowering 出的 physical loop 上，新 pipeline 不再依赖
  grouped-MAC 的两 op 拼写，并在 SG2044/K1 上都产生了可测的正收益。
- 这不等于“pipeline 问题已经解决”：F16、decode 和 IQ4_NL 的负结果直接划出了当前
  cluster/Level 结构不能获益的部分。
- IQ4_NL 并非“只改 std 就自动高性能”；它在 K1 上真实暴露了一条横向
  storage-geometry propagation 缺口。修正后同一规则依赖 typed field/conversion/lookup 关系，
  不依赖 IQ4_NL 名称。
- IME 的问题不是再给 `contract` 加一个 engine role。Donor 需要一棵不同的作者数值树
  与 derived Encoding，而 target 需要将该树的现有 logical cohort 分解为 fragments；这两个缺口
  不能互相代替。
