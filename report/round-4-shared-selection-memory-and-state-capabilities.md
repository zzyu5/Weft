# 第四轮：Selection、Indexed Memory 与 State 语义收敛

本轮没有增加 kernel，继续使用已有真实程序暴露出的缺口完善共享编译能力。改动集中在四个
局部实体：masked VLA effect、local index ordering、indexed memory resource decision 与
typed summary state。所有程序仍沿唯一主链生成 RISC-V intrinsic C；没有新增 kernel 名、
算子名、量化格式或 whole-region lowering 入口。

## Masked VLA effect

Greedy NMS 保留作者显式写下的 winner scan、ordered outer traversal 和 suppression state，只将
对当前 winner 的候选 IoU 检查写成显式 `W.vla`。Target 新增的是可复用的 u8 vector store、
masked store 和组合 predicate projection，不拥有 NMS traversal。

SG2044（RV64GCV，VLEN=128）完整 selected-index 对照为零 mismatch，median 从
9.271941 ms 降至 6.103287 ms，吞吐从 226.182630 提升至 343.610255 MPairChecks/s。

## Local index ordering

Top-p 原来的 repeated argmax 不是 target 应当识别并替换的 source closure；算法真正缺少的是
局部排序语义。因此 DSL 与 canonical Kernel IR 新增 `sort_indices`：

```text
input slice: f32
output: u32 indices
order: ascending | descending
NaN: last
tie: index ascending
```

Target 为这个 primitive 唯一产生 stable radix decision，radix-8 与 radix-11 是同一 realization
family 的构建期候选。Emitter 只消费 radix width、pass count 和 order，不读取 surrounding
kernel。Primitive-local scratch 不再泄漏进作者 kernel ABI。

Top-p 使用 radix-8，在 `rows=64, vocabulary=256` 上完成排序、cutoff 和 sampled token 全量
零 mismatch，median 从 123.840724 ms 降至 0.810183 ms。已有 Argsort 使用同一个 primitive
的升序语义，并在 `rows=8, columns=32000` 上选择 radix-11；其 DSL ABI 删除了显式 scratch
indices 与 histogram 参数，最终 median 为 9.846183 ms、25.999923 MValues/s。

## Indexed memory resource decision

Indexed VLA load 继续由 pointer relation 得到 `Indexed` memory mode，并生成 u32 index vector、
byte-offset vector 与 `vluxei32`。本轮修正的是候选合法性：decision 现在显式计算 index、
byte-offset temporary、gather result，以及该局部 consumer 同时需要的 f32 companion vectors。
Emitter 中的 index width 与 LMUL只来自该 access decision，不再硬写一份独立选择。

CSR SpMV 的默认候选因此选择 LMUL4；显式 LMUL8 因真实 live-set 超过寄存器资源而明确
unsupported。完整数值对照误差为零，`rows=65536, nnz=3101508` 的 median 从
21.997717 ms 降至 20.554190 ms，吞吐从 0.281985 提升至 0.301788 GOP/s。

同一规则也通过 Weighted EmbeddingBag 的两份自然等价 DSL 与 CSR sparse attention。前者
primary/equivalent 分别为 64.541444/64.831086 ms；后者为 72.693900 ms。两者没有经过 source
normalization，也没有进入 CSR 或 embedding 专用路径。

## Typed summary state

此前 RISC-V lowering 会遍历 generic `summary_fold` 的 helper SSA closure，精确识别 argmax 与
online-softmax algebra。这使算法语义由 target 猜测，并让无关的 operand 顺序或临时 SSA
可能决定高性能路径。

本轮新增两个明确的局部 DSL/Kernel IR primitive：

```text
argmax(input, coordinate, tie=lowest_coordinate, order=relaxed)
online_softmax_summary(input, math=native, order=preserve)
```

`argmax` 显式拥有 f32 maximum、logical coordinate 与最低 coordinate tie；
`online_softmax_summary` 显式拥有稳定的 `(maximum, scaled_sum)` algebra。Target 仍独立决定
coordinate memory mode、LMUL、state placement 和 RVV intrinsic realization。原有两套 helper
closure matcher已删除；generic `summary_fold` 对未知 algebra明确 unsupported，不猜测、也不
fallback。

真实执行结果保持原性能区间：Argmax 为 16.380862 ms、1002.191918 MElements/s；Top-K 的
primary/equivalent 为 2.906673/2.879913 ms；Softmax 全量数值对照的最大绝对/相对误差为
5.20255123e-08/5.16609979e-07，median 为 2.643301 ms、198.345894 MElements/s。

## 本轮结果

四项能力都由显式 DSL 构造或逐 access typed facts授权；作者的 scalar control、outer traversal、
blocking、persistent state 和算法 variant没有被 target重写。最终仍只有：

```text
Weft DSL
→ canonical Kernel IR
→ RISC-V physical decisions
→ intrinsic C / local asm
→ system compiler
```

当前性能数字记录在 `weft-kernel-performance.csv`；该表只保存本次真实重测结果，不附带同步、
阈值或验证逻辑。
