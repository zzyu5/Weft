# Weft 四轮编译器化推进

## 结论

四轮工作的对象不是四组独立 kernel case，而是同一条 Weft RISC-V kernel compiler 主链：

```text
Weft Python DSL
→ canonical Kernel IR
→ RISC-V physical decisions
→ intrinsic C / local asm
→ system compiler
→ object / executable
```

第一轮以四十个 kernel、四十二条 phase 收紧物理决定权责并建立资源约束候选空间；第二、
第三轮各事后引入十个陌生 kernel，检查 contract、state、quant、irregular memory 和新组合能否
在没有预设 whole-kernel lowering 的情况下外推；第四轮不增加 kernel，而是把前三轮暴露的
NMS、Top-p、CSR indexed memory 和 summary closure 问题收敛成共享编译能力。

截至本轮结束，六十个真实 kernel 仍走一条 production path。新增能力均由显式 DSL 构造、
typed operands、axis、predicate、memory relation、state primitive、target facts 与 backend config
驱动；没有新增 kernel-name、q-format、exact-op-count、whole-region emitter、GGML/materials
runtime 调用或 legacy/scalar fallback。

## 冻结的授权边界

作者显式拥有：

- scalar loop 与 ordered outer traversal；
- blocking、staging、persistent layout 与算法 variant；
- `W.vla` 授权的 SIMD logical axis；
- `W.contract` 授权的局部 contraction domain；
- reduce、scan、summary、sequential carry 各自不同的可观察语义；
- memory effects、coordinate、predicate、packed format 与 extension primitive。

Target lowering 只在上述授权内部决定：

- VLA strip、`vl`、LMUL、mask 与 memory form；
- state placement 与跨 strip realization；
- contract microtile、K-unroll、operand schedule 与局部 packing；
- packed decode、narrow/widen、RVV/IME fragment；
- intrinsic family 与 typed inline-asm spelling。

Target 不创建作者没有写下的 outer loop，不替换 traversal、blocking、persistent packing 或算法
variant，也不把 scalar loop 自动改成 VLA。寄存器分配、最终机器调度、peephole 与机器码生成
继续交给 system compiler。

持久表示只有 canonical Kernel IR 与最终 artifact。LMUL、microtile、packing、fragment、
capability、legality 与候选均是一次 target lowering 内的瞬态事实。每个物理决定只有一个
producer；emitter 只能消费决定，不能再次识别算法或重选物理策略。

## 四轮如何推进

| 轮次 | 压力语料 | 主要问题 | 形成的共享能力 |
| --- | --- | --- | --- |
| 第一轮 | 固定40个kernel、42条phase | decision仍贴近emitter，候选过窄 | 统一physical preparation；VLA、contract、block、narrow与state资源候选 |
| 第二轮 | 10个事后陌生kernel | contract方向/init过窄，packed quant缺局部语义 | Region-init contract、vector-dot方向、四个typed packed-dot primitive |
| 第三轮 | 再10个陌生kernel及等价写法 | indexed access、segmented state、局部数学能力不足 | Indexed VLA memory、typed predicate、segmented scan、vector sqrt |
| 第四轮 | 固定现有60个kernel | NMS/Top-p/CSR与summary仍暴露低质量或closure依赖 | masked VLA effect、sort primitive、indexed live-set模型、typed summary primitive |

## 第一轮：物理决定体系

### Decision authority

此前 block decode、相邻 block store 分组与 block reduce 分组仍在 emitter traversal 中首次
识别。本轮把它们全部移入统一 physical preparation：

- `DecodeOp` 根据 typed codes、table、result 和 target facts 产生唯一 decode decision；
- block store 先决定 owner、共享 axis、producer closure、sink group 与 strip realization；
- block reduce 先决定 group、closure 与 reduction strip realization；
- emitter 不再重新收集 closure、决定分组或选择 byte shape。

VLA、contract、quantized extension primitive 与 IME leaf 也遵守同一
`analysis/decision → emission` 权责。

### Physical candidate space

VLA LMUL 根据 access、state、predicate、nested traversal 和寄存器组数选择；f32→i8 narrow
由固定值扩成 `{8,4}`，合法性计入 f32 live vectors 与两级 narrowing intermediate；relaxed
add/max reduction拥有 ScalarCarry 与 VectorCarry placement。

F32 contract 将 LMUL 与 `K-unroll={1,2,4}` 组合，并按 accumulator、streamed operands 与
target vector-register 数过滤。F16 contract 将 row-microtile divisors 与 input LMUL
`{1,2,4}` 组合。Block store拥有 `e8mf4 micro / e8m1 fixed / e8m1 dynamic`，block reduce
拥有 `e8m1 fixed / dynamic` 候选。

IME1 只保留真实执行过的 VLEN256 4×4×8 fragment，没有为形成“候选数量”虚构第二个实现。

### 结果

四十二条 phase 均完成真实编译与执行。相对本轮开始，median time几何平均约改善2.1%。

| 压力点 | 本轮前 median ms | 本轮后 median ms | 变化 |
| --- | ---: | ---: | ---: |
| RMSNorm | 1.365566 | 1.012684 | 快25.8% |
| LayerNorm | 1.442806 | 0.982864 | 快31.9% |
| ADD_ID | 7.701514 | 6.328268 | 快17.8% |
| F16 GEMM prefill | 379.894115 | 352.083699 | 快7.3% |
| Q4_K×Q8_K decode | 13.462768 | 12.842926 | 快4.6% |
| Dense Conv | 804.604125 | 788.391226 | 快2.0% |

Q4_0 IME 的 tiled max reduction曾错误选择 VectorCarry。修复读取的是 VLA end 的局部 producer
事实，而不是 kernel/format 名；最终 activation code mismatch为0，最大绝对误差
0.0312509537，median为3.159286 ms。

## 第二轮：十个事后陌生 Kernel

名单在接入 compiler 前锁定，没有根据现有 fast path 换题：

| Kernel | 用来施压的组合 |
| --- | --- |
| Gated Linear Attention | ordered state + VLA projection |
| RWKV-WKV7 | 多项 sequential state + VLA |
| Gated Delta Net / KDA | gate + state + projection |
| Batched lower-triangular solve | ordered traversal + unfamiliar contract context |
| GroupNorm | 多层 reduction + VLA normalization |
| SAM relative-position add | indexed relation + irregular scatter effect |
| IQ2_S × Q8_K | packed codebook + sign + scale + compute |
| IQ3_S × Q8_K | packed lookup + sign + compute |
| IQ1_M × Q8_K | codebook + delta + packed scale |
| Q6_K × Q8_K | narrow decode + grouped scale + local asm leaf |

### Contract 外推

`W.contract` 从 `[BM,K] × [VLA,K]`、零 init 的窄 envelope 扩展到自然的
`[VLA,K] × [K] → [VLA]`、Region init 与有序 outer context。Decision一次性物化 free/blocked
operand、shared reduction axis、Region init、memory mode、LMUL、K-unroll、output memory form
和 absorbed local closure。Emitter 不再根据 surrounding source 重新选择方向或 microtile。

Triangular solve 对同一 canonical program执行多个合法候选：

| Contract config | median ms | GOP/s |
| --- | ---: | ---: |
| LMUL4, K-unroll 1 | 5.854985 | 5.730917 |
| LMUL4, K-unroll 2 | 6.306948 | 5.320233 |
| LMUL4, K-unroll 4 | 6.863750 | 4.888644 |
| LMUL1, K-unroll 4 | 20.022708 | 1.675819 |
| LMUL2, K-unroll 4 | 10.624787 | 3.158127 |

最终记录为5.818646 ms、5.766708 GOP/s。候选维度属于 shared contract lowering，没有固化为
triangular-solve规则。

### Packed quant primitive

IQ2_S、IQ3_S、IQ1_M 与 Q6_K 分别成为有明确 typed operand 合同的256-element local semantic
primitive。它们拥有一块 packed weight与一块Q8_K activation的局部数值关系，但不拥有 outer
block traversal、persistent layout或kernel ABI。IQ2_S/IQ1_M使用RVV gather/widen/dot，IQ3_S
使用局部 decoded buffer加RVV dot，Q6_K使用只覆盖单个block的inline-asm leaf。

| Kernel | Weft median ms | Weft GOP/s | GGML GOP/s | Weft / GGML |
| --- | ---: | ---: | ---: | ---: |
| IQ2_S × Q8_K | 65.380747 | 1.796255 | 2.393391 | 0.751× |
| IQ3_S × Q8_K | 68.639682 | 1.710971 | 0.822512 | 2.080× |
| IQ1_M × Q8_K | 37.359844 | 3.143496 | 4.591404 | 0.685× |
| Q6_K × Q8_K | 19.161684 | 6.128924 | 4.895281 | 1.252× |

IQ2_S 与 IQ1_M 的剩余差距指向共享 codebook gather、scale accumulation 与 register
organization，未使用 whole-loop code 掩盖。

## 第三轮：陌生组合与反过拟合

第二批十个事后陌生 kernel重点检查相同 primitive 在新组合、等价 DSL 与不同候选下是否仍
获得同类 realization。

| Kernel | 暴露的结构 | 本轮结束时结果 |
| --- | --- | ---: |
| Weighted EmbeddingBag | ragged offsets + indexed row + weighted reduction | 64.861903 ms |
| Segmented inclusive scan | typed segment start + ordered prefix | 9.956464 ms |
| CSR SpMV | indexed gather + row reduction | 21.997717 ms |
| CSR sparse attention | CSR traversal + online state | 72.423638 ms |
| ROIAlign | irregular coordinates + bilinear samples | 51.974708 ms |
| Greedy NMS | ordered selection + mutable suppression | 9.271941 ms |
| Top-p sampling | ordering + prefix + dynamic cutoff | 123.840724 ms |
| FWHT | explicit stages + in-place butterfly | 28.708866 ms |
| Cross-entropy loss+gradient | stable reduction + indexed label update | 22.636580 ms |
| AdamW | pointwise state update + sqrt | 68.661300 ms |

这一轮新增了逐 access 的 Indexed memory relation、u8 typed vector predicate、segment-aware scan
state 与 `W.sqrt → vfsqrt.v`。EmbeddingBag、segmented scan、ROIAlign 与 AdamW 的自然等价
写法没有经过 normalization pass；它们读取相同的 pointer、dtype、predicate、state 与 use
facts，因此获得同类别 realization。

候选实测也证明 physical config不是 kernel-specific 常量：CSR SpMV的LMUL1/4分别为
23.999645/20.758271 ms；AdamW的LMUL1/4/8分别为83.227385/68.594461/61.426150 ms；
segmented scan的LMUL1因u8/f32 lane footprint不合法而明确失败。

更重要的是，这一轮没有掩盖失败：Top-p仍因 repeated selection达到123.840724 ms，NMS仍是
9.271941 ms，CSR的indexed resource模型仍不完整。这三个缺口成为第四轮的直接输入。

## 第四轮：闭合前三轮暴露的问题

### Masked VLA effect

Greedy NMS 保留 scalar winner scan、ordered outer traversal 与 suppression state，只把当前
winner对应的候选IoU检查写成显式 `W.vla`。Target新增通用u8 vector store、masked store与
组合predicate projection，不拥有NMS traversal。

完整 selected-index 对照为零 mismatch；median由9.271941降至6.103287 ms，吞吐由
226.182630提升至343.610255 MPairChecks/s。

### Local index ordering

Top-p 暴露的不是“需要一个 Top-p fast path”，而是 DSL 缺少局部排序语义。新增 canonical
`sort_indices` primitive，显式记录：

```text
input: f32 pointer slice
output: u32 indices
order: ascending | descending
NaN: last
tie: index ascending
```

Target唯一产生stable radix decision；radix-8与radix-11是同一family的构建期候选。Emitter
只消费radix width、pass count和order。Primitive-local scratch不再泄漏到作者kernel ABI。

Top-p在`rows=64, vocabulary=256`上选择radix-8，排序、cutoff与sampled token均为零
mismatch；median由123.840724降至0.810183 ms。已有Argsort复用升序语义，在
`rows=8, columns=32000`上选择radix-11；其DSL ABI删除显式scratch indices和histogram，
最终为9.846183 ms、25.999923 MValues/s。

### Indexed memory live-set

Indexed load的candidate legality现在显式计算index vector、byte-offset temporary、gather
result，以及局部consumer同时存活的f32 companion vectors。Index width与LMUL只来自该access
decision；emitter不再硬写第二份选择。

CSR SpMV默认选择LMUL4；LMUL8因真实live-set超过寄存器资源而明确unsupported。完整数值
对照误差为零，median由21.997717降至20.554190 ms，吞吐由0.281985提升至0.301788 GOP/s。
同一规则覆盖Weighted EmbeddingBag两份等价DSL与CSR sparse attention，没有CSR/embedding
专用入口。

### Typed summary state

此前 target 会遍历 generic `summary_fold` 的 helper SSA closure，精确识别 argmax 与 online
softmax。这会让算法语义由target猜测，也会对无关SSA组织敏感。本轮新增两个明确的局部
DSL/Kernel IR primitive：

```text
argmax(input, coordinate, tie=lowest_coordinate, order=relaxed)
online_softmax_summary(input, math=native, order=preserve)
```

`argmax`显式拥有f32 maximum、logical coordinate与最低coordinate tie；
`online_softmax_summary`显式拥有稳定的`(maximum, scaled_sum)` algebra。原有两套helper
closure matcher已删除。Generic `summary_fold` 对RISC-V未知algebra明确unsupported，不猜测
也不fallback。

Argmax为16.380862 ms、1002.191918 MElements/s；Top-K primary/equivalent为
2.906673/2.879913 ms；Softmax最大绝对/相对误差为5.20255123e-08/5.16609979e-07，
median为2.643301 ms、198.345894 MElements/s。

## 当前共享编译能力

| 能力 | Canonical authority | Target physical freedom |
| --- | --- | --- |
| VLA / predicate / memory | 显式VLA axis、pointer、predicate、dtype | strip、VL、LMUL、unit/strided/indexed、mask |
| State | reduce、scan、segment、argmax、online summary、sequential carry | scalar/vector carry、cross-strip、RVV reduction/scan |
| Contract | operands、axes、init、order、math、作者blocking | microtile、LMUL、K-unroll、load schedule、local packing |
| Quant / decode | typed packed fields与local numerical primitive | decode family、widen/narrow、RVV gather/dot、local asm |
| Selection / ordering | explicit coordinate、tie、order、sort primitive | radix width、local scratch、RVV state realization |
| Extension | 明确的local semantic primitive | RVV intrinsic或IME fragment |

这些能力是逐实体组合的结果，不回答“这个kernel属于哪一类”。一个kernel可以同时包含scalar
control、多个VLA、indexed memory、state、contract与extension primitive；每个实体独立产生
typed facts与physical decision。

## 当前性能与边界

所有当前数字以 `weft-kernel-performance.csv` 为准；固定 GGML 参考保存在 `baseline/`，未进入
production path。

仍然明确存在的共享缺口：

- F32 prefill GEMM与Dense Conv仍受contract-local load schedule、reuse和pipeline限制；
- IQ2_S与IQ1_M仍需更好的codebook gather、scale accumulation和register organization；
- indexed memory已有typed resource model，但prefetch与更宽的memory scheduling尚未成熟；
- IME1仍只有真实验证过的VLEN256 4×4×8 local fragment；
- generic `summary_fold` 的任意用户algebra尚无通用RISC-V realization，会明确unsupported。

这些缺口没有备用路径。当前仓库保持的目标仍是：作者写显式worker-local VLA kernel，compiler
根据逐实体语义与资源组合出RISC-V physical realization，而不是积累六十条kernel发射模板。
