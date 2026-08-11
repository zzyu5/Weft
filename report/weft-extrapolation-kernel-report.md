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

外推结论是“编译与组合成立，性能只部分成立”：五项同形 public op 对照更快；Q1_0 和
MXFP4 距固定手写 RVV baseline 分别为 8.3% 和 11.1%；Argsort、Dense Conv2D 与
ConvTranspose2D 仍明显落后。失败没有被 fallback 隐藏，也没有被改写成三个 whole-kernel
emitter。

## 十个 workload

| Kernel | 真实 workload / shape | 给编译器的结构压力 | source 明确保留的算法结构 |
|---|---|---|---|
| F32 Argsort | batched logits，`8 × 32000` | data-dependent ordered control、indexed permutation | heap construction、sift-down 与 row-local index storage |
| F32 SetRows | KV cache，`8 × 4096 × 128`，128 updates | indexed destination、VLA row copy | update 顺序、共享 row index、group/row strides |
| SAM Window Partition | `C=768,H=W=64,w=14` | scalar window control、padding predicate、VLA memory | window order、zero padding 与 output layout |
| Dense Conv2D | SAM，`64²,IC=OC=256,K=3,pad=1` | predicate-dependent contraction、strided weight free axis | spatial/block traversal、flattened reduction relation和layout |
| ConvTranspose2D | decoder upsample，`16²×256 → 32²×128,K=2,S=2` | inverse coordinate predicate、contract in a different index relation | output-owned traversal、stride divisibility和kernel layout |
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

Argsort 让 `while` 从静态控制补充为真正的 scalar loop-carried control。条件和 carried value
都来自 canonical use-def；target 直接保留作者的执行顺序并生成普通 C control，不把它识别
为“sort kernel”。同时修正了多次使用的 scalar load 必须物化一次的问题，避免生成 C 表达式
重复读取可变 index storage。

这项能力解决的是语义与可组合性，不自动把作者的 heapsort 换成 `std::sort`、radix sort 或
RVV sorting network。算法替换仍是 source variant，不是 target 猜测。

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
`conv2d`、`out_product` 名字判断实现。当前只有一个主要 row-microtile 配置，也没有自动创建
im2col scratch或改变source traversal，因此“能进入同一 contract family”不等于卷积已经有
成熟物理候选空间。

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
| Argsort | 5 | 59.368179 | 21.902416 | 2.711 | — | — |
| SetRows | 10 | 0.355782 | 0.432772 | 0.822 | 2.947 GB/s | 2.423 GB/s |
| Window Partition | 10 | 5.165583 | 5.575184 | 0.927 | 5.350 GB/s | 4.957 GB/s |
| Dense Conv2D | 3 | 2386.263326 | 629.354844 | 3.792 | 2.025 GOP/s | 7.677 GOP/s |
| ConvTranspose2D | 5 | 96.806903 | 17.179675 | 5.635 | 0.693 GOP/s | 3.906 GOP/s |
| RMSNorm backward | 10 | 6.358188 | 10.622016 | 0.599 | 329.835 MElements/s | 197.434 MElements/s |
| OutProd | 3 | 128.860142 | 231.133128 | 0.558 | 5.729 GOP/s | 3.194 GOP/s |
| Im2Col backward | 5 | 35.573495 | 374.418412 | 0.095 | 29.476 MElements/s | 2.801 MElements/s |
| Q1_0 × Q8_0 | 3 | 56.174805 | 51.859000 | 1.083 | 2.091 GOP/s | 2.265 GOP/s |
| MXFP4 × Q8_0 | 3 | 19.462305 | 17.522000 | 1.111 | 6.034 GOP/s | 6.702 GOP/s |

Weft correctness口径：Argsort、SetRows、Window Partition、RMSNorm backward和Im2Col backward
做全输出检查；Dense Conv2D、ConvTranspose2D、OutProd、Q1_0与MXFP4做固定代表点检查。
记录的最大误差除RMSNorm backward的 `1.10268593e-05` 外均为0。

### 性能含义

- SetRows、Window Partition、RMSNorm backward、OutProd和Im2Col backward分别比同形GGML
  public op快17.8%、7.3%、40.1%、44.2%和90.5%。这些结果说明普通indexed memory、VLA
  reduction和local contract在陌生上下文中已经能形成有效RVV realization。
- Q1_0与MXFP4分别慢8.3%和11.1%。这是本组最强的对照：GGML两项都是真实手写RVV
  intrinsic实现。Weft已进入同一性能区间，但outer-loop overhead、load scheduling和寄存器组织仍有差距。
- Argsort慢2.71倍。Weft faithfully执行作者写下的heapsort；GGML调用C++ `std::sort`。这里暴露的
  首先是缺少更高性能source算法variant，而不是target应该偷偷替换算法。
- Dense Conv2D慢3.79倍。GGML direct path会构造contiguous patch并进入成熟dot/GEMM组织；
  Weft当前直接消费strided free-axis contract，没有spatial tile、patch reuse、packing候选或
  多种microtile选择。
- ConvTranspose2D慢5.63倍。Weft source是output-owned inverse mapping，许多reduction lane由
  stride predicate屏蔽；GGML采用input-position与kernel-position遍历并执行contiguous
  input-channel dot。把二者互换属于算法traversal选择，不能由target无条件猜测。

## 架构审计

本轮新增路径没有读取kernel symbol做选择，没有exact op-count分支，没有GGML/materials
链接，也没有“新路径失败后走旧实现”。Q1/MXFP4 extension、VLA contract和memory decision
都以typed local entity为anchor。

整个 `RISCVLowering` 仍不能描述为已经完全摆脱source closure。较早的F16 conversion/fill/
dot/update/normalize、online-softmax envelope、F16 GEMM和部分IME1 N/K路径仍在emission期间
检查较精确的loop/body结构。本轮没有利用这些路径，也没有为了十项成功继续扩大它们；这仍是
后端重构尚未完成的明确边界。

最终判断：**Weft已经能让十个事后选择、结构差异明显的自然worker-local程序通过同一条
算子编译主链，并由局部语义事实生成可执行RVV实现；其中五项超过同形public op、两项量化
dot进入手写RVV的12%以内。但排序source variant与两种卷积的物理/算法自由度仍不足，因此
本轮证明了真实外推能力，也同时否定了“已经达到Triton级性能通用性”的过强结论。**
