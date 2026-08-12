# 第三轮：事后陌生 Kernel 的反过拟合与性能泛化

## 结论

本轮在 lowering 结构确定后锁定十个此前未纳入 Weft 的真实 kernel，十项均通过同一条
`Python DSL → canonical Kernel IR → RISC-V physical decision → intrinsic C → GCC → SG2044`
主链执行。没有新增 kernel 名、算子名、量化格式、完整 region 形状或 op-count 入口；新增
target 能力只有逐 access 的 indexed memory、typed vector predicate、segment-aware scan state
以及 f32 vector `sqrt` realization。

十项不是十条 fast path。它们分别复用 scalar ordered traversal、显式 VLA、reduce、scan、
summary state、pointer relation 和 source-owned effects；不具备共享 primitive 的地方仍然保留
为可见性能缺口，没有调用 GGML、`materials/` 或旧 emitter。

## 十项压力语料

| Kernel | 作者显式拥有的结构 | target 消费的局部事实 | SG2044 结果 |
| --- | --- | --- | --- |
| Weighted EmbeddingBag | bag offsets、动态 entry traversal、weighted sum | VLA 内动态 scalar loop 与 f32 vector carry | 64.861903 ms，1.554945 GB/s |
| Segmented inclusive scan | 显式 `segment_start` 与 ordered prefix | u8 load、typed predicate、segment scan state | 9.956464 ms，421.264417 MElements/s |
| CSR SpMV | CSR row traversal、indexed vector lookup、row reduction | u32 index vector、`Indexed` load、f32 reduction | 42.275206 ms，0.146729 GOP/s |
| CSR sparse attention | CSR edge traversal、online max/sum、scratch accumulator | unit VLA load/reduce/store | 72.423638 ms，0.694608 GOP/s |
| ROIAlign | box traversal、sampling grid、bilinear interpolation | scalar coordinates、VLA channel arithmetic | 51.974708 ms，4.201389 GB/s |
| Greedy NMS | ordered selection、mutable suppression、IoU effects | scalar state/control/memory primitives | 9.271941 ms，226.182630 MPairChecks/s |
| Top-p nucleus sampling | descending selection、ordered prefix、dynamic cutoff、uniform draw | argmax summary 与 scan state | 123.840724 ms，33.868536 MCandidates/s |
| FWHT | stage、block、butterfly 与 in-place effect | unit-stride VLA load/store/arithmetic | 28.867527 ms，1.743539 GOP/s |
| Cross-entropy loss+gradient | stable max/sum、label update、gradient store | reduce、vector exp、predicate/select | 22.636580 ms，185.288767 MElements/s |
| AdamW | optimizer equations、moment/parameter effects | pointwise memory/arithmetic 与 `vfsqrt.v` | 68.661300 ms，6.841730 GB/s |

所有性能数字均为 SG2044 单核 RV64GCV、VLEN=128、64 MiB eviction 后的 median；完整 shape、
repetition 和误差数字已写入 `weft-kernel-performance.csv`。NMS 与 Top-p 比较离散结果，分别为
selected index，以及排序、cutoff 与 sampled token 全量零 mismatch；其余项比较完整数值输出。

## 等价 DSL 对 closure 的压力

本轮不是把 source 规范化为同一个模板，而是保留四组自然等价写法：

* EmbeddingBag：inline pointer arithmetic 与显式 row/base factoring；
* segmented scan：absolute coordinate 与 base+relative coordinate，并交换 predicate operand；
* ROIAlign：pure bilinear helper 与 inline SSA arithmetic；
* AdamW：直接方程与分解后的 address/SSA organization。

结果分别为：

| Kernel | primary | equivalent |
| --- | ---: | ---: |
| Weighted EmbeddingBag | 64.861903 ms | 65.475786 ms |
| Segmented scan | 9.956464 ms | 9.658303 ms |
| ROIAlign | 51.974708 ms | 52.015907 ms |
| AdamW | 68.661300 ms | 68.419220 ms |

四组均通过全量 correctness。等价写法没有经过 normalization pass；它们获得同类别 realization，
因为 decision 读取相同的 pointer、dtype、predicate、state 和 use facts。

## Candidate space 的实际变化

同一 canonical program 使用现有 explicit backend config 重复进入唯一 lowering，并在真机执行：

| Kernel | 候选 | 结果 | 物理变化 |
| --- | ---: | ---: | --- |
| Segmented scan | LMUL=1 | illegal | u8/f32 同 lane footprint 无合法整数 LMUL，明确拒绝 |
| Segmented scan | LMUL=4 | 9.881044 ms | f32m4 + u8m1 segment state |
| CSR SpMV | LMUL=1 | 23.999645 ms | u32m1 offsets + f32m1 indexed gather |
| CSR SpMV | LMUL=4 | 20.758271 ms | u32m4 offsets + f32m4 indexed gather |
| FWHT | LMUL=1 | 26.429536 ms | 更小 strip/butterfly vectors |
| FWHT | LMUL=4 | 28.755026 ms | 更宽 strip/butterfly vectors |
| AdamW | LMUL=1 | 83.227385 ms | f32m1 pointwise/sqrt |
| AdamW | LMUL=4 | 68.594461 ms | f32m4 pointwise/sqrt |
| AdamW | LMUL=8 | 61.426150 ms | f32m8 pointwise/sqrt |

这证明 candidate 并非 kernel-specific 常量：相同 `vlaLMUL` 维度在 typed resource legality、memory
relation 与局部 live vectors 下产生不同合法集合和不同实测最优。默认结果仍由 target facts 与资源
模型选出；上述数据只是对 candidate space 的真实展开，不形成新 compiler stage。

## 本轮逼出的共享能力

### Indexed VLA memory

CSR SpMV 的 `vector + column` 使非 affine Region index 首次成为明确的 `Indexed` lane relation。
analysis 记录 base、typed offset、index SEW/LMUL 和 data shape；emitter 只按 decision 生成
`vmul` byte offsets 与 `vluxei32`。连续的 `values`、`column_indices` 仍分别生成 unit-stride load。

### Typed predicate 与 segmented state

`segment_start` 没有被 scan emitter 从周围 op 猜测。u8 vector load、vector-scalar compare 与
segment-aware scan 各自先得到 typed decision；scan decision 持有唯一 segment predicate、data
LMUL、segment LMUL 与 mask ratio。发射用显式 head mask传播、strip carry continuation 和 reset。

### Local vector square root

AdamW 暴露了 DSL 缺少自然 `sqrt` 的真实语言缺口。新增 `W.sqrt` 对应 canonical unary `sqrt`，
RISC-V realization 为局部 `vfsqrt.v`。算法方程、bias correction、weight decay 和 memory effects
仍完全由 DSL source 拥有。

## 仍然暴露的性能缺口

Top-p 的 123.840724 ms 来自作者显式 repeated selection；目前没有局部 sort/permutation primitive
及其 physical candidate family。这个差距没有被 Top-k matcher、C library sort 或 whole-kernel
emitter遮盖。NMS 当前同样是 scalar ordered selection/effect，尚未拥有局部 box-block overlap 的
VLA realization。CSR SpMV 的 indexed gather 已正确但只有 0.146729 GOP/s，说明 indexed memory
scheduling、prefetch 与 row-length-sensitive LMUL 仍需成为更成熟的共享候选维度。

因此本轮证明的是：六十个累计 kernel 后，陌生组合仍可沿显式授权与逐实体事实进入唯一主链，
并且 physical candidate 确实随资源改变；它没有证明每一类陌生算法都已达到成熟库的性能。
