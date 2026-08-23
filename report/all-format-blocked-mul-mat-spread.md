# Q4_K blocked 结构向全格式铺开的反例报告

## 结论

这一轮没有把 Q4_K 路径机械复制到 24 个格式后宣称完成。第一个陌生格式
IQ2_XXS 和第二个 direct-packed 格式 Q4_0 已经足以否定这个前提：

> 当前“新增格式 = encoding 声明 + std 数值函数”只对已有 scalar production
> tree 成立，对 blocked high-performance tree 不成立。

Q4_K 已经拥有的核心形态是“output cohort 进入 RVV lane，packed integer MAC 直接在
这些 lane 上执行”。IQ2_XXS 的手写高性能形态却需要“codebook 的 8 个 reduction
element 进入 RVV lane，多个 output 进入寄存器重复”。Q4_0 又要求 reduction 消去 K
以后继续保留 M/N free axes。当前表示系统只能把一个仍存在于 result value 上的
`lane_axis` 记录下来，不能表示这两种关系。因此每铺一个格式都会重新撞后端，泛化
尚未成立。

本轮保留了一条新的、真实可执行但性能不好的 IQ2_XXS blocked local-pack 路径，更新了
真实性能 CSV；Q4_0 试验因为数值错误已经从代码中删除。由于同一重构还让 F32 dense
的原固定候选暴露为资源非法，本轮代码是未收敛状态，没有提交。

## 用什么外部判据检查

本轮没有用“pass 有字段”“emitter 没按格式名分支”作为泛化证据，而是做了三种交叉
输入：

1. Q4_K：已有 grouped/layered direct packed MAC；
2. IQ2_XXS：grid/sign lookup、32-element group、8-element codebook reduction；
3. Q4_0：同样是 grouped/layered nibble，但需要 `-8 * sum(q8)` correction。

同时核对了参考编译器的现有做法：

- Triton 的 reduce result 使用从 parent layout 切片得到的 encoding；被消去的 reduction
  轴不再是 result shape 的普通轴，但它与物理分布的关系没有丢失。
- Triton 的 layout propagation 沿 SSA use-def 传播，冲突显式插
  `ConvertLayoutOp`，之后再消除可去掉的 conversion。
- TileLang 的 operation 先贡献 layout 关系，layout inference 再沿 buffer alias、dtype
  和 storage bits 传播；fragment/storage mapping 不是由 terminal emitter 临时重建。

当前 Weft 与之相比缺的不是另一个 selector 名称，而是 operation-local mapping：

```text
被 contract/reduce 消掉的 reduction axis -> SIMD lane
保留下来的一个或多个 free axes       -> register repetitions
result value                            -> reduction 后的 scalar/vector tuple
```

把 `lane_axis` 只挂在 result value 上，并要求 result 仍携带该轴，无法表达这件事。

## 实际试了什么

### 1. 先用 IQ2_XXS 反证 Q4_K 路径

新增的作者程序是 `mul_mat_iq2_xxs_local_pack`。它显式写了：

- `NC/KC/MC/MR` blocking；
- canonical IQ2_XXS weight panel 的 kernel-local pack；
- 16-output cohort；
- Q8_K activation quantize 与 workspace；
- IQ2_XXS 的 grid index、sign selector、subscale 和 8-element codebook 数值树；
- accumulator 跨 K block 生存；
- 最终 `0.125` epilogue。

这不是调用 GGML 或 materials，也没有按 IQ2_XXS 名称进入后端。grid/sign ABI 在该 blocked
特化中改成 `i8` table，原因是 GGML 的真实局部数值就是 8-byte grid 和 `±1` sign
plane；原 canonical tree 的 `f32` table 是为标量表达方便准备的，不是高性能物理组织。

第一次生成和执行依次暴露了以下共享问题：

| 暴露点 | 原假设 | 反例 | 本轮处理 |
|---|---|---|---|
| local pack 轴 | downstream 只能有一个 lane axis，pack 就沿它 | IQ2 同时有 output N、row M 和 codebook use | pack 只读取作者显式 `along`，不再从 consumer 猜 |
| dtype promotion | 任意不同 dtype 都生成 `widen` | `u32 -> i32` 是同宽转换 | 前端区分 widen 与 cast |
| table representation | f32 lookup 后再转 i32 | 手写路径的 grid/sign 是 i8 | blocked std/runtime 使用 i8 table；integer widen 走已有 RVV |
| lane 轴平局 | rows/cols 同优先时取遍历中第一个 | `[MR,16]` 误选 M，N-lane lookup 无法进入 accumulator | 同优先级比较实际 cohort；N16 胜过 MR2 |
| register repetition | 只给 outer_contract result 建 register axis | 普通 pointwise accumulator 也有另一条 free axis | rank-2 wide value 的另一条 rows/cols 轴进入 register repetition |
| scalar tuple field | dynamic field index 被替成常数 0 | activation 每次都读 `x.q[0]` | scalar-tuple load 消费真实 per-use index |
| float mul-add | scalar C 在 `-ffp-contract=fast` 下收缩，intrinsic C 固定拆成 mul+add | 输出相差数十 ULP | pass 为合法 vector one-use mul→add 产生 fused cluster，emitter机械写 `vfmacc` |

这些修改后，同一份 IQ2_XXS blocked tree 在 SG2044 和 K1 都得到 bit-exact 结果。

### 2. IQ2_XXS 的真实数字

所有下面的新路径数字都来自完整
`DSL -> Kernel IR -> RISC-V passes -> intrinsic C -> target compiler -> target execution`
repro。新行是 cold、`repetitions=3`、无 warmup；source 列沿用当前 CSV 中同 shape 的既有
source 数字，所以它们只用于判断数量级，不伪装成一次完全重测。

| target | path | phase | Weft GOP/s | source GOP/s | Weft/source |
|---|---|---:|---:|---:|---:|
| SG2044 / VLEN128 | canonical IQ2_XXS | decode | 1.911395 | 4.067610 | 46.99% |
| SG2044 / VLEN128 | canonical IQ2_XXS | prefill | 1.863710 | 4.239853 | 43.96% |
| SG2044 / VLEN128 | blocked local pack | decode | 0.553423 | 4.067610 | 13.61% |
| SG2044 / VLEN128 | blocked local pack | prefill | 1.309321 | 4.239853 | 30.88% |
| K1 / VLEN256 | blocked local pack | decode | 0.509911 | 1.664643 | 30.63% |
| K1 / VLEN256 | blocked local pack | prefill | 1.178748 | 1.705884 | 69.10% |

这个结果不是“blocked 参数没调好”。生成的 IQ2 内层对每个 8-element grid entry 发出
8 次 byte gather，因为 SIMD lane 被分给 output N；手写 IQ2 路径把这 8 个 codebook
element 放进 SIMD reduction lane，一次取出/展开 grid，再让 output 成为寄存器重复。
`unroll/pipeline/LMUL` 不会把前一种程序变成后一种物理映射。

### 3. 用 Q4_0 检查是否只是 IQ 特殊

Q4_0 没有 codebook，表面上最接近 Q4_K。试验 tree 使用同一套 16-output local pack 和
encoded widening MAC，额外计算：

```text
raw = dot(unsigned_q4, signed_q8)
centered = raw - 8 * sum(q8)
```

encoded MAC 能生成，但 `sum(q8)` 的 reduction 错误地把所有 physical parts 归成一个
标量，再把同一个 correction 广播给所有 M/N result。真实要求是只消去 K，保留 M
free axis，再广播到 N。该结果数值不正确，因此没有记录性能，试验入口已删除。

这说明缺口并不限于 codebook：只要 reduction 的结果还带 free axes，当前 value-only
`lane_axis` 模型就不完整。

### 4. dense 是第三个外部交叉检验

修正 register repetition 后，原 SG2044 F32 candidate
`NC=32, MR=4, VLEN128` 不再被错误地计成少量寄存器。资源 pass 看到 contract result
和 carried accumulator 各 32 个 vector groups，峰值为 66，超过 32。

临时缩小到 `NC=16, MR=2` 后可以 bit-exact 运行，但只有 `0.105278 GOP/s`，低于 CSV
中的旧 Weft `0.234716 GOP/s`，所以该临时候选没有保留在 runner。这个交叉检验说明：

- 旧资源合法性确实漏算 rank-2 register repetitions；
- `acc += outer_contract(...)` 仍未形成真正的 in-place contraction/accumulator handoff；
- 当前 candidate enumeration 没有在 target 资源变严后自动找新的高质量合法点。

因此，本轮的 representation 修正还没有收敛成可以提交的完整主干。

## 24 个格式的铺开状态

“只改 std”表示：在现有后端不变的前提下，新的 blocked tree 已经生成、真机 bit-exact
并有数字。当前没有任何新增格式满足这一栏；Q4_K 是此前已经纵向建设过的路径，不算
本轮新增格式。

| family | format | 本轮状态 | 只改 std 已证实 | 被迫改后端或暴露的缺口 |
|---|---|---|---|---|
| symmetric | Q1_0 | 未铺 | 否 | 未实测，不作泛化声明 |
| symmetric | Q4_0 | 已写 tree 后删除 | 否 | reduce 消去 K 后不能保留 M free axis |
| symmetric | Q5_0 | 未铺 | 否 | 未实测，不作泛化声明 |
| symmetric | Q8_0 | 未铺 | 否 | 未实测，不作泛化声明 |
| affine/min | Q4_1 | 未铺 | 否 | 未实测；有额外 min/sum branch |
| affine/min | Q5_1 | 未铺 | 否 | 未实测；有 high-bit 与 min/sum branch |
| K | Q2_K | 未铺 | 否 | 未实测，不把 Q4_K 结果外推给它 |
| K | Q3_K | 未铺 | 否 | 未实测，不把 Q4_K 结果外推给它 |
| K | Q4_K | 已有 persistent 与 local 两条 | 不适用 | 此前后端已为 interleave、grouped MAC 和跨 group pipeline 建能力 |
| K | Q5_K | 未铺 | 否 | 未实测，不把 Q4_K 结果外推给它 |
| K | Q6_K | 未铺 | 否 | 未实测，不把 Q4_K 结果外推给它 |
| IQ/codebook | IQ1_S | 未铺 | 否 | 未实测；codebook reduction mapping 未闭合 |
| IQ/codebook | IQ1_M | 未铺 | 否 | 未实测；codebook/derived scale mapping 未闭合 |
| IQ/codebook | IQ2_S | 未铺 | 否 | 未实测；codebook reduction mapping 未闭合 |
| IQ/codebook | IQ2_XS | 未铺 | 否 | 未实测；grid/sign reduction mapping 未闭合 |
| IQ/codebook | IQ2_XXS | blocked local path 双机 bit-exact | 否 | pack axis、cast、free-axis layout、dynamic tuple index、FMA；高性能 reduction-lane mapping仍缺 |
| IQ/codebook | IQ3_S | 未铺 | 否 | 未实测；high-bit/codebook reduction mapping 未闭合 |
| IQ/codebook | IQ3_XXS | 未铺 | 否 | 未实测；grid/sign reduction mapping 未闭合 |
| IQ/codebook | IQ4_NL | 未铺 | 否 | 未实测；nonlinear lookup mapping 未闭合 |
| IQ/codebook | IQ4_XS | 未铺 | 否 | 未实测；lookup + per-sub scale mapping 未闭合 |
| ternary | TQ1_0 | 未铺 | 否 | 未实测；radix digit 的 reduction mapping 未闭合 |
| ternary | TQ2_0 | 未铺 | 否 | 未实测；ternary unpack 的 reduction mapping 未闭合 |
| FP4 | MXFP4 | 未铺 | 否 | 未实测；codebook + exponent-scale mapping 未闭合 |
| FP4 | NVFP4 | 未铺 | 否 | 未实测；codebook + per-block scale mapping 未闭合 |

准确计数是：

- 已有高性能终点：1 个格式（Q4_K，前序工作）；
- 本轮新增、双机 bit-exact：1 个格式（IQ2_XXS），但没有进入高性能区间；
- 写过自然 blocked tree、因共享后端语义缺口撤回：1 个格式（Q4_0）；
- 尚未铺开：21 个格式；
- 本轮证实“只改 std 即可”的新增格式：0 个。

## Q4_K 数字为什么仍然要保留

Q4_K 是有效的局部能力证据，但不是全格式泛化证据。当前 CSV 追加了两种作者程序：

| path | phase | Weft GOP/s | source GOP/s | Weft/source | pack 口径 |
|---|---:|---:|---:|---:|---|
| Q4K_I16 persistent | decode | 9.388334 | 9.555431 | 98.25% | persistent pack 不计入 kernel |
| Q4K_I16 persistent | prefill | 11.108464 | 9.720730 | 114.28% | persistent pack 不计入 kernel |
| canonical local pack | decode | 1.260617 | 9.555431 | 13.19% | pack 计入 kernel |
| canonical local pack | prefill | 10.505987 | 9.720730 | 108.08% | pack 计入 kernel |

这四个数字说明 Q4_K 的 grouped/layered + output-cohort realization 是真的；IQ2_XXS
反例说明它只泛化到具有同一局部物理关系的程序，不能代表 codebook、ternary 或所有
direct-packed 格式。

## spec 2.2 是否被证伪

没有。

IQ2_XXS 的高性能 tree 必须显式出现 8-element reduction domain、output cohort 和
accumulator 生命周期；这些是作者树。编译器不能把普通 scalar `for lane in range(8)`
私自变成 SIMD Level。这个边界是正确的。

当作者写出该 reduction domain 后，编译器应在不改变逻辑值集合和 Level 归属的前提下
选择：

```text
K8 reduction axis -> RVV lanes
N/M free axes     -> register repetitions
lookup/decode     -> lane-local memory/instruction realization
```

当前做不到，属于 physical mapping/contract/reduce lowering 不完整，不是需要编译器改树。
Q4_0 的 correction 也是同样情况：树已经明确写出 `reduce(q8)`，缺的是“reduce 只消去 K，
保留其他 free axes”的物理实现。

## 当前唯一阻塞点

当前不能继续机械复制 21 个格式，因为那只会得到 21 份已知会退化或算错的 tree。需要先
让 RISC-V physical mapping 能表示并生成：

```text
operation-local eliminated-axis mapping
multi-free-axis register tuple
reduce/contract 后的 value handoff
layout conflict 的显式 conversion
与上述映射一致的资源计数
```

这不是新增持久 Physical IR，也不是让编译器搜索作者 tree；它是当前一次 target lowering
内缺失的表示能力。在它闭合以前，“全格式只改 std”是已被真实反例否定的说法。

## CSV 说明

`report/weft-kernel-performance.csv` 已直接写入：

- Q4K_I16 persistent decode/prefill；
- Q4_K canonical local-pack decode/prefill；
- IQ2_XXS canonical SG2044 的本轮真实 decode/prefill；
- IQ2_XXS local-pack 在 SG2044、K1 的 decode/prefill。

新 local/persistent 行用不同 kernel 名和 `scope`，没有覆盖 canonical Q4_K。新测 Weft
数字是 cold、无 warmup；沿用的 source 数字来自 CSV 里同硬件同 shape 的已有 baseline，
空白或口径差异不被解释成零。
