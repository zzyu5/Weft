# 十个事后选择 Kernel 与 Weft 外推结果

这份报告记录一组在当前 lowering 结构基本确定后才选出的真实 kernel。选择时没有先检查
Weft 是否已有 fast path；十项用于检验自然 DSL 程序能否依靠逐实体语义事实进入同一条
RISC-V 主链，而不是继续扩大已经成熟的 kernel family。

固定的 [`baseline/`](baseline/) 没有修改。Weft 真机数字追加在
[`weft-kernel-performance.csv`](weft-kernel-performance.csv)；八个 GGML public op 的同形
对照由 `examples/repro/ggml/unfamiliar_runtime.cpp` 运行，Q1_0 与 MXFP4 的数字直接读取固定
baseline 中已有的同形手写 RVV vec-dot 记录。

## 结论

十项都已经通过唯一 production 主链生成并在 SG2044 单 hart 上执行：

```text
natural Weft Python DSL
→ canonical worker-local Kernel IR
→ RISC-V local physical decisions
→ intrinsic C
→ GCC 15 system compiler
→ executable
```

本轮没有增加 kernel/operator 名分派、完整 source closure 分类、GGML/materials 调用、legacy
fallback 或程序规范化 pass。作者写下的排序算法、row scatter、窗口顺序、卷积坐标、
output-owned backward traversal、packed storage 和 outer reduction 都保留在 DSL/Kernel IR。

初次引入后的三个明显缺口已经由显式source variant修正：Argsort使用作者声明的radix
scratch/histogram，Dense Conv2D显式staging patch和weight，ConvTranspose2D显式选择
input-owned traversal与contiguous input-channel packing。修正后八个public-op对照中七项更快，
Dense Conv2D仍慢44.8%；Q1_0和MXFP4距固定手写RVV baseline分别为8.3%和11.1%。

## 十个 workload

| Kernel | 真实 workload / shape | 给编译器的结构压力 | source 明确保留的算法结构 |
|---|---|---|---|
| F32 Argsort | batched logits，`8 × 32000` | integer key transform、indexed histogram/scatter | four-pass radix traversal、scratch/histogram ABI |
| F32 SetRows | KV cache，`8 × 4096 × 128`，128 updates | indexed destination、VLA row copy | update 顺序、共享 row index、group/row strides |
| SAM Window Partition | `C=768,H=W=64,w=14` | scalar window control、padding predicate、VLA memory | window order、zero padding 与 output layout |
| Dense Conv2D | SAM，`64²,IC=OC=256,K=3,pad=1` | source staging、contiguous local contract | im2col/weight packing、position microtile和layout |
| ConvTranspose2D | decoder upsample，`16²×256 → 32²×128,K=2,S=2` | input-owned traversal、contiguous local contract | source/weight packing、non-overlap ownership和kernel traversal |
| RMSNorm backward | `512 × 4096` | two reductions followed by a VLA consumer | exact backward equation、epsilon和row traversal |
| F32 OutProd | TinyLlama linear gradient，`2048 × 5632 × 32` | contract free axis换位、短 reduction | samples/rows/columns relation与output layout |
| Im2Col backward | SAM，`64²,IC=256,K=3,pad=1` | ordered window accumulation、strided VLA loads | output-owned dInput traversal与窗口边界 |
| Q1_0 × Q8_0 | Llama FFN，`N=14336,K=4096` | packed sign bits、128→4×32 relation | 18/34-byte strides、sub-block loop、scales和outer rows |
| MXFP4 × Q8_0 | Llama FFN，`N=14336,K=4096` | E2M1 codebook、E8M0 exponent、packed nibble lookup | 17/34-byte strides、exponent load和outer reduction |

ConvTranspose2D 的 shape 是符合 GGML p0 输出公式的 decoder 级 workload 映射，不宣称来自
仓库中的某个固定模型实例；其余 vision shape 和 quantized shape 均沿用 GGML 注释或已有
模型级 baseline。

## 十项逼出的编译器能力

### Data-dependent ordered control

最初heapsort先证明了真正scalar loop-carried `while` 和data-dependent permutation可以忠实
lower。性能修正没有让target识别“sort kernel”，而是作者显式改用four-pass F32 radix：
sortable-key bit relation、256-bin histogram、prefix sum、ping-pong index scatter及scratch ABI
全部进入Kernel IR。编译器只补齐通用scalar F32→U32 bitcast spelling。

### Indexed write 与 window memory 复用普通实体决策

SetRows 和 Window Partition 没有要求新 compiler branch。前者由 scalar indexed base、
unit-stride VLA load/store 组合；后者由普通 scalar window loop、scalar predicate 与 channel
VLA memory 组合。两项证明同一套 access decision 能在 gather-like write 和 padded vision
layout 中复用，而不是按 operator 增加 route。

### Free-axis local contract 进入新坐标关系

Dense Conv2D、ConvTranspose2D 与 OutProd 共同扩展了 local F32 contract 的适用上下文：

- reduction extent 可以由 source scalar expression给出；
- LHS predicate可以随 reduction coordinate变化；
- RHS free axis和output access分别选择unit-stride或strided memory mode；
- row microtile、LMUL、reduction axis、load/store projection成为一次local decision的事实。

Target 消费显式 `ContractOp`、block/VLA axis、pointer relation和predicate；它不根据
`conv2d`、`out_product` 名字判断实现。Dense Conv的patch/weight staging以及ConvTranspose的
input-owned traversal/packing均由source显式提供，现有local contract family无需增加分支即可
消费。当前仍只有一个主要row-microtile配置，因此Dense Conv尚未获得成熟物理候选空间。

### Output-owned backward traversal

Im2Col backward 保留每个 input element拥有一次最终store的source算法。内层九个window
位置按顺序累加，每次load的address与predicate独立投影；target只把input-channel域映射到
RVV。这一项没有增加 `im2col_back` primitive，说明ordered scalar window与strided VLA
memory可以在新的producer-consumer环境中组合。

### 两个新的局部量化语义 primitive

Q1_0 和 MXFP4 不能无损伪装成普通 F32 contract，因为sign-bit mapping、E2M1 codebook和
E8M0 exponent都是可观察数值语义。本轮因此增加两个local extension：

```text
weft_ext.sign_bit_i8_dot
  block<4,u8> signs × block<32,i8> activation
  × explicit activation/sign scales + init → f32

weft_ext.e2m1_e8m0_i8_dot
  block<16,u8> packed codes + scalar u8 exponent
  × block<32,i8> activation × explicit activation scale + init → f32
```

它们只拥有一次32-lane numerical relation。Q1 的128-element block如何拆成四个sub-block、
MXFP4/Q8_0 persistent bytes、row stride、outer block traversal和public ABI仍全部出现在
Kernel IR。

RISC-V lowering依据typed operands、little-endian target fact、VLEN128与显式block axes选择：

- Q1：`vlm` sign mask、signed-i8 widen、sign merge、i32 widening reduction；
- MXFP4：low/high nibble拼接、E2M1 table gather、i8 widening multiply与i32 reduction。

Emitter只拼写这两个decision所选的intrinsic family；没有调用GGML或materials leaf，也没有
以q-format route string重新决定outer loop。

## SG2044 同形性能

全部运行固定为单线程/单hart、RVV VLEN128、64 MiB cache eviction后取中位数。Weft计时
generated kernel entry；八个GGML public op计时 `graph_compute + synchronize`。量化两项的
GGML数字来自固定baseline中的同形单thread vec-dot。比值为 `Weft ms / GGML ms`，小于1表示
Weft更快。

| Kernel | Repetitions | Weft ms | GGML ms | Weft / GGML | Weft throughput | GGML throughput |
|---|---:|---:|---:|---:|---:|---:|
| Argsort | 5 | 8.958199 | 21.902416 | 0.409 | — | — |
| SetRows | 10 | 0.355782 | 0.432772 | 0.822 | 2.947 GB/s | 2.423 GB/s |
| Window Partition | 10 | 5.165583 | 5.575184 | 0.927 | 5.350 GB/s | 4.957 GB/s |
| Dense Conv2D | 3 | 911.200486 | 629.354844 | 1.448 | 5.303 GOP/s | 7.677 GOP/s |
| ConvTranspose2D | 5 | 16.333010 | 17.179675 | 0.951 | 4.109 GOP/s | 3.906 GOP/s |
| RMSNorm backward | 10 | 6.358188 | 10.622016 | 0.599 | 329.835 MElements/s | 197.434 MElements/s |
| OutProd | 3 | 128.860142 | 231.133128 | 0.558 | 5.729 GOP/s | 3.194 GOP/s |
| Im2Col backward | 5 | 35.573495 | 374.418412 | 0.095 | 29.476 MElements/s | 2.801 MElements/s |
| Q1_0 × Q8_0 | 3 | 56.174805 | 51.859000 | 1.083 | 2.091 GOP/s | 2.265 GOP/s |
| MXFP4 × Q8_0 | 3 | 19.462305 | 17.522000 | 1.111 | 6.034 GOP/s | 6.702 GOP/s |

Weft correctness口径：Argsort、SetRows、Window Partition、RMSNorm backward和Im2Col backward
做全输出检查；Dense Conv2D、ConvTranspose2D、OutProd、Q1_0与MXFP4做固定代表点检查。
记录的最大误差除RMSNorm backward的 `1.10268593e-05` 外均为0。

### 性能含义

- Argsort的显式radix variant比GGML `std::sort` 快59.1%；scratch与四次histogram/scatter均在
  timed kernel内，没有把预处理藏到runtime。
- SetRows、Window Partition、RMSNorm backward、OutProd和Im2Col backward分别比同形GGML
  public op快17.8%、7.3%、40.1%、44.2%和90.5%。这些结果说明普通indexed memory、VLA
  reduction和local contract在陌生上下文中已经能形成有效RVV realization。
- Q1_0与MXFP4分别慢8.3%和11.1%。这是本组最强的对照：GGML两项都是真实手写RVV
  intrinsic实现。Weft已进入同一性能区间，但outer-loop overhead、load scheduling和寄存器组织仍有差距。
- Dense Conv2D经显式patch/weight staging从2.025提升到5.303 GOP/s，但仍比GGML慢31.0%
  （按时间为44.8%）。剩余差距是local contract只有固定LMUL4/row6，没有microtile、unroll、
  packing和software-pipeline候选。
- ConvTranspose2D经显式input-owned traversal与packing从0.693提升到4.109 GOP/s，比GGML快
  4.9%。当前`stride=kernel=2`保证各input/kernel pair拥有不同output位置；重叠variant仍必须
  由source显式采用atomic或另一种ownership。

## 架构审计

本轮新增路径没有读取kernel symbol做选择，没有exact op-count分支，没有GGML/materials
链接，也没有“新路径失败后走旧实现”。Q1/MXFP4 extension、VLA contract和memory decision
都以typed local entity为anchor。

较早的F16 conversion/fill/dot/update/normalize、online-softmax envelope、F16 GEMM和affine
IME1 N/K路径仍要求较精确的loop/use closure，但选择与发射已经拆开：analysis产生瞬态typed
decision，emitter只消费bounds、pointers、strides、tiles、layout和realization。它们不再形成
emission-time第二authority；尚未解决的是对等价source closure的接受范围仍窄。

最终判断：**Weft已经能让十个事后选择、结构差异明显的自然worker-local程序通过同一条
算子编译主链，并由局部语义事实生成可执行RVV实现；其中七项超过同形public op、两项量化
dot进入手写RVV的12%以内。Dense Conv的物理候选空间与旧specialized decision的source
敏感性仍未达到Triton级通用性，但三个最明显性能缺口已由正确的source/compiler所有权修复。**
