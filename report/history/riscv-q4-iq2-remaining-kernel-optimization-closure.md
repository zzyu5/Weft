# Q4_K、IQ2 与剩余 Kernel 优化收口报告

日期：2026-08-31

## 1. 范围与结论

本报告覆盖从“继续按照优化思想，修复剩下的 kernel，包括 Q4_K staged / persistent、通用 P2 owner、staged local pack 与 P5”开始的这一段工作。

这一段没有全量重跑 206 条 production 目标。实际完成并写入当前性能 CSV 的是 12 个受影响行：Q4_K、IQ2_XXS、TQ1_0、IQ2_S、IQ2_XS 的 prefill 双机结果，以及 IQ2_XS 的 decode 双机结果。其余入口只做了必要回归或静态盘点，不能据此更新全仓分布。

结果不是“所有问题都解决”，但有四个可由外部现象检验的推进：

1. Q4_K staged 的 local pack 已经是 typed local allocation 加 RVV `vlse8/vse8` transfer，不再是 scalar copy。SG production prefill 从本段开始时的 `3.508368` 提到正式 10 次结果 `9.909454` GOP/s，超过同 shape RVV source `9.699175`。
2. shaped entry/payload contraction 与 replica-reduced partial topology 在 IQ2_XXS、TQ1_0、IQ2_S 上产生了实际减少工作。三个 prefill family 在至少一台机器达到或超过 source；TQ1_0 与 IQ2_S 在两台可比 RVV source 上都超过。
3. IQ2_XS 新增了按 8-byte entry 布局描述的 typed indexed-entry path。K1 decode 从 `0.298879` 提到 `0.521853` GOP/s；同一实现放到 SG 反而从 `0.690402` 降到 `0.670427`，因此调用方保留 SG scalar decode 与 K1 entry-shaped decode 两棵程序，而不是在 compiler pass 中按 target 分支。
4. 通用 multi-consumer P2 owner 没有为了“补齐架构”而硬造。静态 identity/work ledger 表明当前候选无法删除足够动态工作；按 P0 不实现假 `SupplyOp`。当前闭合的仍是有真实 consumer 和可见收益的 explicit materialize residency、grouped supply、partial lifetime 与 typed entry load。

## 2. 提交与实际改动

### 2.1 `054606f03`：Q4_K staged packed supply

层次：physical IR、layout/operation selection、composite lowering、intrinsic-C emitter，以及 staged 作者树的最小配合。

- `materialize` 形成真实 `LocalType` allocation 与 `EncodedLocalPackOp`；
- pack leaf 是 `rvv.local-pack.interleave`；
- intrinsic C 对 canonical record 发 `vlse8`，向 interleaved local storage 发 `vse8`；
- 后续 grouped MAC 从 local view unit-load，不再在每个 consumer 重新读 canonical strided bytes。

当前实现位置：`LowerRISCVComposites.cpp:993-1001` 与 `RISCVIntrinsicC.cpp:6940-7024`。因此旧报告中“staged local pack 仍为 scalar copy”的边界已经失效。

### 2.2 `3dfab1e4a`、`4992078d5`：production 入口与参数绑定

层次：runner/caller 与参数性选择，不改 logical tree，不在 emitter 中补结构。

- production prefill 改为显式调用 staged Q4_K entry；
- SG/K1 分别绑定已经能生成的 NC/KC/MC/MR/NR；
- `unroll=2` 的 10 次结果把 SG `9.552765→9.868481`，K1 `4.413429→4.482609` GOP/s；随后回归重测为当前 CSV 的 `9.909454 / 4.478264`。

### 2.3 `dd50ccd8d`、`875789736`：IQ2_XXS shaped contraction

层次：作者树、canonical axis 表达、physical layout/partial materialization。

- Python 展开的 entry/payload 工作改为显式 shaped Level/value；
- contraction 的 entry/payload reduction 在 physical IR 中成为一组宽 partial，而不是每个 payload 独立 scalar carry；
- verifier、layout propagation 与 partial materialization按 axis/type 关系工作，不读取 IQ2_XXS 格式名。

正式 prefill 结果：SG `0.682041→4.193430`，K1 `0.303588→1.372448` GOP/s。

### 2.4 `001ada1c2`：shaped IQ2 window 与 grouped supply

层次：encoding facts、作者树、physical memory ops、memory planning、emitter。

该提交建立了 typed entry-window/indexed-entry 物理链，并为 IQ2_S、IQ2_XS、IQ2_XXS 的 staged 树提供真实 consumer。它没有把“indexed load 出现”当成性能结论；之后的实测继续区分 P3 access form、P2 supply 次数和 P4 partial topology。

### 2.5 `248488faf`：统一 replica-reduced partial topology

层次：physical IR、partial topology pass、resource contract、emitter。

- 多个 register replicas 上的 contraction partial 先保持为 typed partial set；
- combine/final reduction 由同一个 topology 决定，不再由外围 stream gate 与 materializer各算一次；
- emitter只消费 materialized topology。

该机制使 TQ1_0 prefill：

- SG `2.230880→6.287893` GOP/s；
- K1 `1.468909→4.119001` GOP/s。

同时 TQ2_0 回归保持在 `20.466060 / 6.505267`，没有重现此前错误门控造成的数量级退化。

### 2.6 `3d3472eed`：pipeline descriptor replay

层次：physical schedule/expander 边界。

expander 复制 body/版本化值时同步重放 memory descriptor 与 origin facts，避免 schedule 展开后丢失 typed storage relation。它服务的是已有 schedule，不通过“这是某种 quant kernel”猜流水。

同一节点将 IQ2_S staged prefill 接入 production：SG `0.196590→5.299330`，K1 `0.094295→1.508821` GOP/s。

### 2.7 `37b9766cb`：aligned IQ2 entry table 的 indexed gather

层次：Encoding、两棵 decode 作者程序、physical layout propagation、memory planning、verifier、intrinsic-C。

- 新增 `I8X8`，只表达“一个 record 是 8 个连续 i8，8-byte aligned”的 storage layout；不包含 codebook 数值语义；
- IQ2_XS decode 明确分成 `mul_mat_iq2_xs_scalar` 和 `mul_mat_iq2_xs_entry`；prefill 仍是独立 `mul_mat_iq2_xs_staged`，没有试图让一棵 DSL 同时解决 decode/prefill；
- `PlanRISCVMemory` 能从一维或二维 entry-major descriptor 产生 `RVVIndexedEntryLoadOp`；
- layout propagation只在 typed relation证明 64-bit payload、8-byte alignment 和合法 bit offset 后让 entry axes 加入 contraction lane domain；
- register/time-to-lane conversion 与 RVV tuple repartition按每个 axis 的 time×lane×replica 守恒处理，不再假设只有一个 lane axis。

关键源码位置：

- `python/weft/std/encodings.py:185`；
- `python/weft/std/mul_mat.py:1016-1070`；
- `lib/Target/PropagateRISCVLayouts.cpp:384-477`；
- `lib/Target/PlanRISCVMemory.cpp:1920-1998`；
- `lib/Dialect/RISCV/IR/RISCVDialect.cpp:2866-2944`；
- `lib/Target/RISCVIntrinsicC.cpp:5390-5540`；
- `examples/run/weft-mul-mat.sh:137-166`。

## 3. 优化思想如何实际使用

本段按 `doc/compiler/optimization-principles.md` 的 P0–P5 与 R 诊断，而不是从“还缺哪个 pass 名”出发。

### P0：先数工作

以下方向没有保留：

- generic P2 supply op：TQ1 power table/当前 IQ lookup 的合法 SSA identity 并不能合并不同 digit、offset 或 payload；静态可删除工作不足；
- IQ2_XS `scale_groups=4/8` 作者变体：SG 一次结果 `0.230`，K1 `0.515`，没有优于保留树；
- 为动态 zero shift 再命名一个 op：不会减少 shift 或地址工作；
- 独立的更晚 residency pass、静态 exact-tail 与 Q3 m8 load：此前已经分别被 invariant failure、合法 tail 和 spill 证伪，本段没有恢复。

### P1：载体归属

IQ2 entry、payload、scale-group 和 output 不再套一个“reduction axis 永远进 lane”的公式。entry/payload 是否进入 lane 由下游 contraction、storage payload 宽度、target VLEN 和完整 layout conservation共同决定；surviving output 仍保留在 replica/time 中。

K1 entry path 的 LMUL m2/m4 组合分别达到约 `78/82` vector groups，超过 32-group 资源预算；合法 m1 约为 `24–31` groups。更宽载体因此被判非法，而不是交给 emitter偷偷缩小。

### P2：供应唯一

已经闭合的 P2 范围有明确 SSA owner：

- Q4 staged 的一次 local allocation/pack 服务后续 grouped consumers；
- shaped IQ2 的 grouped supply 与 partial lifetime；
- indexed entry table的一次 typed load/gather result。

一般 multi-consumer producer 的 share/rematerialize/hoist 仍没有统一 owner。没有第二个能静态减少动态工作的 input，所以本段没有增加一个只“表达共享意图”却删除 0 条工作的实体。

### P3：形态跟随

`I8X8` 的 entry-major contiguous geometry允许 memory planner选择 indexed-entry gather；普通 canonical byte table在同一规则下不满足 alignment/payload contract，不会误入该 leaf。这个决定来自 descriptor、axis、alignment 与 consumer relation，不来自 `IQ2_XS` 名字。

### P4：延迟收敛

`rvv_partial_set` 与 replica-reduced topology让 entry/payload/radix products在 vector/register 形态内完成 combine，再做 final reduction。IQ2_XXS、TQ1_0、IQ2_S 的 prefill 提升是这一机制减少 reduction/scalar carry 的外部证据。

### P5：issue-time 重叠

本段修了 descriptor 在 schedule/expander间的传播，但没有把 Q4_K 强行设为 depth 2：

- SG staged 的 `KC=256` 与内部 K block同为 256，局部 kb loop只有一次动态 iteration；
- K1 `KC=512` 只有两次，增加 buffer 的收益尚未超过资源与 prologue/epilogue成本；
- 因此 Q4_K 当前收益来自 P2/P3/P4 和参数 unroll，不是 P5。

这不是“P5 已解决”，也不是漏做；它是当前 Q4_K 实例不满足已证实的收益条件。

## 4. 与 Triton / TileLang 的具体对照

结构决策前检查了参考实现，而不是完成后才补名字：

- Triton `TritonGPUTypes.td` 的 `MemDescType` 与 `local_alloc/local_load/local_store` 把 allocation、view 和 transfer做成真实 IR；Weft 对应采用 `LocalType + EncodedLocalPackOp`，但不复制 CTA/shared-memory owner。
- Triton `RemoveLayoutConversions.cpp:895-959` 以 `(SSA value, target encoding)` 和 dominance/backward-slice legality决定 rematerialization共享；这支持“共享必须有真实 SSA identity”，也否定了按相似地址合并 IQ payload。
- Triton `LinearLayout` / conversion 处理多维 basis conservation；Weft 的多轴 time/lane/replica conversion采用同一类表示原则，但载体是完整 CPU logical value，不是 thread/warp ownership。
- Triton `OptimizeDotOperands` 将 local allocation/load提升到共同 operand base；Weft只在 canonical `materialize` 或 typed entry relation已经提供 owner时做对应 rewrite，不从重复 consumer猜回逻辑轴。
- TileLang `reducer_plan_materialize.cc` 只让 structurally equal update plans共享 partial storage；这对应 Weft 的 typed partial owner/topology，而不是让 CSE 合并数值相同但 lifetime 独立的 partial。
- Triton/TileLang 都将 pipeline scheduling 与 expansion分开；Weft的 `ScheduleRISCVLevels`/`PipelineRISCVLevels`也保持该分工，descriptor replay只是保证展开后的程序仍携带选定 memory relation。

## 5. 正式性能结果

统一为 `M=128,N=4096,K=4096` 的 prefill 或 `M=1,N=4096,K=4096` 的 decode，Clang 18.1.8，`-O3 -ffp-contract=fast`，10 repetitions，数值均为当前 CSV 的 `within-tolerance`。

| 入口 | Target | 本段前 Weft | 当前 Weft | source | 当前/source |
|---|---|---:|---:|---:|---:|
| Q4_K staged prefill | SG | 3.508368 | 9.909454 | 9.699175 RVV | 102.17% |
| Q4_K staged prefill | K1 | 2.002277 | 4.478264 | 24.576693 IME1 | 不同引擎，不比较 |
| IQ2_XXS prefill | SG | 0.682041 | 4.193430 | 4.198543 | 99.88% |
| IQ2_XXS prefill | K1 | 0.303588 | 1.372448 | 1.707026 | 80.40% |
| TQ1_0 prefill | SG | 2.230880 | 6.287893 | 6.181190 | 101.73% |
| TQ1_0 prefill | K1 | 1.468909 | 4.119001 | 3.456469 | 119.17% |
| IQ2_S prefill | SG | 0.196590 | 5.299330 | 2.804593 | 188.95% |
| IQ2_S prefill | K1 | 0.094295 | 1.508821 | 1.475642 | 102.25% |
| IQ2_XS decode | SG | 0.690402 | 0.670427 | 3.489556 | 19.21% |
| IQ2_XS decode | K1 | 0.298879 | 0.521853 | 1.759671 | 29.66% |
| IQ2_XS prefill | SG | 0.689895 | 2.760653 | 4.088130 | 67.53% |
| IQ2_XS prefill | K1 | 0.298958 | 0.745638 | 1.808918 | 41.22% |

IQ2_XS 的结果同时给出正反证据：entry-shaped indexed path在K1 decode为 `1.746×`，在SG decode为 `0.971×`；blocked staged prefill在SG/K1分别为旧值的 `4.002× / 2.494×`，但仍没有到 source。

Q4_K staged / persistent 另外做了当前 HEAD 的 1-repetition双机回归：

| 入口 | SG | K1 | numeric |
|---|---:|---:|---|
| staged | 9.869509 | 4.449318 | zero error in this run |
| persistent `Q4K_I[16]` | 10.161120 | 3.226367 | zero error in this run |

这证明两条路当前都能编、能跑。它也证明不能把 persistent 当成普遍更快的物理实现：SG persistent略快，K1 staged明显更快。persistent改变跨调用 artifact/ABI，staged消费 canonical Q4_K 并在 kernel 内 pack；它们是两个作者程序，不应被 compiler合并成一个隐藏选择。

## 6. 负实验与没有保留的代码

| 实验 | 观察 | 处理 |
|---|---|---|
| IQ2_XS full-block typed entry path直接用于两机 | K1一次约 `0.523`，优于旧 `0.299`；SG约 `0.449`，劣于旧 `0.690` | 保留通用 physical能力；调用方按 target选择两棵作者程序 |
| SG `scale_groups=4` / K1 `scale_groups=8` | SG约 `0.230`；K1约 `0.515`，没有改善 | 撤销作者变体 |
| entry path LMUL m2/m4 | resource peak约 `78/82 > 32` groups | 判非法，不让 emitter降级 |
| 未加约束的 downstream-axis传播 | staged prefill达到约 `43` groups并失败，说明同一规则污染非entry payload | 加入64-bit payload、alignment、bit-offset typed条件 |
| generic multi-consumer Supply | 静态 identity检查不能合并不同offset/digit/payload；可删除动态工作不足 | 按P0不实现 |
| Q4 depth-2 pipeline | SG loop只有一次，K1仅两次；缺少可证明收益且增加live set | 不纳入当前参数绑定 |

## 7. 还剩多少

### 7.1 能准确回答的 production MUL_MAT 范围

当前 CSV 有 206 行，但本段没有全量重跑，因此不能给出新的全仓“低于 50%”计数。旧的 `128/206` 已被本段 12 行更新改变，不能继续当作当前事实。

按当前源码、当前 CSV 与本段实际覆盖，可明确剩下：

1. **4 个 prefill 作者树尚未迁移为 shaped blocked tree**：IQ1_S、IQ1_M、IQ3_S、IQ3_XXS；双机共 8 行。
2. **8 个 IQ/TQ decode family仍未过 source**：IQ1_S、IQ1_M、IQ2_S、IQ2_XS、IQ2_XXS、IQ3_S、IQ3_XXS、TQ1_0；双机共 16 行。它们并非都缺同一种树，必须分别区分作者 axis、P3 lookup/bit-plane 与 P4 partial。
3. **2 个已 shaped prefill 性能缺口**：IQ2_XS 双机仍为 source 的 67.53%/41.22%；IQ2_XXS K1 为 80.40%，共 3 个 target 行。
4. **Q4_K canonical decode**：SG为 `5.162729 / 9.797607 = 52.70%`；K1 source是IME1，不与当前RVV路径造比值，共2个target行需要分别处理/界定。

因此，在量化 production MUL_MAT 内，当前至少还有 **14 个 logical family×phase问题，加1个仅K1存在的额外性能缺口**；按 target 行计是 **29 行**。这个数量不包含 row-dequant、vec-dot 中仍存在的 Python逐元素展开，也不包含没有进入当前 production CSV 的 GEMM/GEMV/IME/attention/Top-K手工入口。

### 7.2 编译器还没有闭合的能力

- **通用 P2 owner**：仍只有 explicit materialize、typed layered/grouped supply、partial owner等局部 owner；没有任意 multi-consumer producer 的统一 placement/lifetime pass。只有出现第二个可静态减少真实动态工作的 input 才值得实现。
- **IQ2 indexed-entry 后续 partial/reuse**：typed gather已经解决一部分P3，但 IQ2_XS 仍远低于 donor，说明 entry result到多output partial的P2/P4组织尚未闭合。
- **P5覆盖**：scheduler/expander结构存在，Q4_0/Q4_1历史上有正结果；当前Q4_K实例没有证明跨K steady state，不能写成P5已完成。
- **K1 IME production**：Q4_K当前Weft数字是RVV，source是IME1；这仍是不同引擎/不同作者树/derived artifact如何进入production的问题，不是把24.58/4.48直接写成compiler差距。

## 8. 最终判断

这段推进不是“把一个格式写死”：Q4 local residency、replica-reduced topology、多轴 layout conversion、indexed-entry planning分别读取 typed allocation、axis、storage relation、use-def与resource facts，并在Q4、TQ1、IQ2_S/XS/XXS等不同输入上得到正反结果。

但它也没有证明一般 P2、全部 IQ decode 或 P5 已成熟。最重要的未完成事实是：IQ2_XS indexed entry已经把 memory form写对一部分，性能仍只有source的19%–68%；剩余工作不在“再增加一个entry op”，而在同一entry supply如何服务多个output partial、何时combine/reduce，以及对应live set能否留在32个vector groups内。

本段唯一可复现的验证入口仍是：

```bash
ninja -C build weft-compile -j2
./examples/run/weft-mul-mat.sh sg2044 iq2_xs decode 10
./examples/run/weft-mul-mat.sh k1 iq2_xs decode 10
./examples/run/weft-mul-mat.sh sg2044 iq2_xs prefill 10
./examples/run/weft-mul-mat.sh k1 iq2_xs prefill 10
./examples/run/weft-mul-mat.sh sg2044 q4_k_staged prefill 1
./examples/run/weft-mul-mat.sh sg2044 q4_k_persistent prefill 1
./examples/run/weft-mul-mat.sh k1 q4_k_staged prefill 1
./examples/run/weft-mul-mat.sh k1 q4_k_persistent prefill 1
```
