# Encoding 布局、前向 physicalization 与 q4_K/q8 性能

## 结论

本轮关闭了两个错误抽象：Encoding 不再把 packed field 默认为线性
`offset + i * width`，RISC-V lowering 也不再把 C/D 组决定收集进 kernel-global
字典后做带回边搜索。当前主干是：

```text
canonical Kernel IR
→ 每个 auto candidate 独立进入前向 pass 序列
→ value / op / memory edge / Level 各自获得瞬态属性
→ resource pass 接受或拒绝完整 candidate
→ winner
→ 只读属性的 intrinsic-C / local-asm emitter
```

Q4_K 使用与 GGML 一致的真实 byte layout 后，在 SG2044 上用固定种子随机 packed
bytes 得到 bit-exact 结果。spec 中的 `mac_pairs` 树为 **5.765439 GOP/s**；只把作者
树改为已经存在的 `mac_groups(n=4)` 特化，其他结构不动，为
**6.757720 GOP/s**。当前同 shape GGML donor 是 **9.027928 GOP/s**。

这个结果同时暴露出一个源程序层判断：`mac_pairs` 没有算错，但它是偏保守的 q4_K
分解；`mac_groups(n=4)` 在相同 i16 范围内快 17.21%。编译器不能替作者暗中换树，
所以本轮没有修改 spec 中的 q4_K 数值分解。

## 一、GGML packed layout 调查

调查依据是 llama.cpp 当前 `ggml-common.h` 的实际 block storage，以及
`ggml-quants.c` 的 pack/decode 索引关系，不依据 C struct 的表面字段顺序猜逻辑映射。

### 1.1 反复出现的布局关系

全格式并没有逼出“每个格式一个词”。若把默认 `natural` 也算在内，重复出现的是下面
五类关系；其中四类已经有本轮真实 consumer：

| 关系 | 含义 | 典型格式 | 当前状态 |
|---|---|---|---|
| `natural` | 一个逻辑元素依次占自然 storage unit | Q8_0/Q8_1/Q8_K 的 q、普通 scale | 已实现 |
| `grouped(N)` | 每 N 个逻辑元素重新开始一个 storage group | classic 32-element block、K 系列 super-block | 已实现 |
| `layered(P, order)` | group 内每 P 项一层；相同位置共享 byte 的不同 bit range | Q4_0、Q4_K.q、IQ4_NL | 已实现 |
| `joined(group, fields, low_bits, order)` | 多个同形逻辑字段共享前半完整值和尾部低-bit层，高 bits 回填前半空位 | Q4_K/Q5_K scale+min、IQ4_XS scale | 已实现 |
| `bit_planes(...)` | 一个逻辑值的低位、高位、符号或 mask 位来自独立 storage planes | Q5、Q3_K、Q6_K、IQ1/2/3 | 已确认，未公开半成品 surface |

`lookup/codebook`、scale、zero-point、delta 和符号数值解释不属于 Encoding。它们仍是
局部 typed operation 的数值语义；Encoding 只回答 bits 在哪里。

### 1.2 classic formats

| 格式 | packed q 的 storage 关系 | 其他字段 |
|---|---|---|
| Q4_0/Q4_1 | `grouped(32) @ layered(16, lo_first)` | d/m 为 natural |
| Q5_0/Q5_1 | Q4 low-nibble layer + 独立 high-bit plane | d/m 为 natural |
| Q8_0/Q8_1 | natural i8[32] | d/s 为 natural |

Q4 的“前 16 个元素走低 nibble、后 16 个走高 nibble”已经说明线性
`i * 4` 是错误解释；两种解释 storage bytes 数量相同，但逻辑索引不同。

### 1.3 K formats

| 格式 | quants | scale / min metadata |
|---|---|---|
| Q2_K | grouped 2-bit layers | 4-bit scale/min 分层 |
| Q3_K | low-2-bit layers + high-bit mask plane | 6-bit joined scale |
| Q4_K | `grouped(64) @ layered(32, lo_first)` | 两个 u6[8] 共享 `joined(4,2,4,lo_first)` 的 12 bytes |
| Q5_K | Q4_K low-nibble layer + high-bit plane | 与 Q4_K 相同的 joined scale/min |
| Q6_K | low-4-bit layer + high-2-bit plane | natural i8 scales |
| Q8_K | natural i8[256] | natural f32 delta + i16[16] block sums |

Q4_K.q 的确切映射是：

```text
group64 = i / 64
within  = i % 64
byte    = 32 * group64 + within % 32
shift   = 4 * (within / 32)
```

这与 GGML 写入式 `q[l] = L[j+l] | (L[j+l+32] << 4)` 一致。Q4_K 的 sc/m
则与 `get_scale_min_k4` 一致：前四项分别占 scales[0:4] 和 scales[4:8] 的低六位，
后四项的低四位位于 scales[8:12] 的低/高 nibble，高两位回填前八个 bytes 的高位。

### 1.4 IQ formats

IQ 系列没有引入新的“每格式布局原语”，但会组合更多 plane：

| family | storage 关系 | 不属于布局的部分 |
|---|---|---|
| IQ2_XXS/IQ2_XS | grouped compound words；XS 另有 subgroup scales | grid/codebook 数值 |
| IQ2_S | low-index plane + high-bit plane + scale plane | codebook 数值 |
| IQ3_XXS | grouped 3-byte index records | codebook 数值 |
| IQ3_S | low-index、high-bit、sign、scale planes | sign/codebook 数值 |
| IQ1_S/IQ1_M | low-index + high-index/shift + scale planes | grid shift 与 delta 数值 |
| IQ4_NL | grouped/layered nibble indices | nonlinear lookup table |
| IQ4_XS | layered nibble indices + joined high/low scales | nonlinear lookup table |

因此抽象没有失败：格式差异可以分成少数纯布局关系和独立的局部数值 primitive。
本轮只公开了已有 lowering consumer 的 `natural/grouped/layered/joined`；
`bit_planes` 留在 spec 的未闭合项中，没有用一个不能生成代码的接口冒充支持。

## 二、前端与 canonical IR 的改变

Q4_K 现在直接写成：

```python
class Q4_K:
    d:    f16
    dmin: f16
    sc:   u6[8]   @ joined(4, 2, 4, lo_first)
    m:    u6[8]   @ joined(4, 2, 4, lo_first)
    q:    u4[256] @ grouped(64) @ layered(32, lo_first)
```

canonical `encoding_decl` 不再有 `field_packing` 字符串，而是为每个字段保存
`field_layouts`。sc/m 的两个逻辑字段具有不同 role，但共享同一个 96-bit storage span；
verifier 明确验证 group、layer、role、字段宽度和共享跨度，普通字段仍不得任意重叠。

删除了旧 `packed(...)` / `nibble(...)` 入口。Q4_0 也使用同一组合词加入 std encoding，
没有格式名驱动的 parser 或 emitter 路径。

## 三、RISC-V pass 主干

### 3.1 瞬态表示

`weft_riscv.problem` 现在代表一个已经实例化的完整 candidate，保存 canonical facts 和
当前 pass 已经写出的实体属性；`weft_riscv.assignment` 只保存 winner。二者都只存在于
一次编译调用中。

已经删除：

```text
c_decisions
d_decisions
decision_domains
constraint arcs
DFS/backtracking solver
family fallback selector
```

当前前向 pass 的真实契约是：

| Pass | 读取 | 写入 |
|---|---|---|
| Construct | canonical axes/values/ops/Level/use-def + target + auto binding | 一个 candidate problem |
| Assign representations | typed values、显式 wide uses、target VLEN/SEW/LMUL | 每个 value 的 kind/SEW/LMUL/vl/lane axis/materialization |
| Select local operations | value attrs、typed op、engine、encoding declaration、target capabilities | 每个 op 的 realization/local operation；每个 field use 的 memory edge |
| Schedule Levels | domain、candidate partition、value lane mapping | 每个 Level 的 iteration/tail/schedule |
| Check resources | value lifetime/register groups、fragment groups、target budget | candidate resources 或 invalid reason |
| Select winner | 所有完整合法 candidate | 一个 assignment |
| Emit intrinsic C | assignment 上的 value/op/memory-edge/Level attrs | C/asm 文本；不写属性 |

当前实现边界也应如实说明：compile-and-measure、显式 value-use convert、spill 和多种真实
pipeline candidate 尚未实现。Level schedule 当前是保守的 unroll=1、pipeline=1、
prefetch=0；文档不再把存在字段写成已经具备多实现空间。

### 3.2 emitter 权责收回

本轮逐项移除了用户指出的决定泄漏：

- `compileDot` 只根据 dot op 已选 `realization` 分派 spelling，不再从 Field/Fold2 邻接闭包
  识别 quant dot；
- grouped MAC 读取 mac op 的 instruction、lhs LMUL、partial SEW/LMUL/layout，以及 q field
  use 的结构化 memory edge；不再读取全局 `nibble_unpack`，也不再从 result LMUL反推 partial；
- min dot 的 accumulator SEW/LMUL、group count、pair relation由 dot op 的
  `local_operation` 唯一给出；
- IME emitter 读取 fragment instruction/factors/result representation，不再读取 target VLEN
  并自行判定 `VLEN256`；
- emitter 构造时不再从第一个 RVV value 重建 kernel-global lane axis。load、extract、
  contract 和 store 分别读取所消费 value 的 `lane_axis`；materialize 读取自己的 op 属性。

Emitter 中保留的分支现在用于根据已经选定的 realization 拼 intrinsic/asm，或检查 spelling
前置条件；它不写 physical attribute，也不按 kernel、格式、target 名或 VLEN 选择结构。

## 四、q4_K 随机 byte repro

运行入口：

```bash
./examples/run/weft-kernel.sh sg2044 q4_k_gemv 10
```

runtime 使用固定种子生成 q4_K 的 scales/q bytes 和 q8_K 的 q/bsum bytes；只把 d、dmin、
ds 覆写成有限随机数值，避免任意浮点 bit pattern 产生 NaN。Weft 先按派生 encoding 做 rows=16
持久 interleave，generated C 再消费该布局；独立 reference 直接按 GGML
`get_scale_min_k4` 和 q4_K byte indexing 消费原始同一段 bytes。没有调用 GGML quantize，
也没有 GGUF。

最终五条定向运行均为 bit-exact：q4_K pair、q4_K groups4、q8_0、q8_1、q8_K。

## 五、性能

硬件均为 SG2044 / RV64GCV / VLEN128，单线程，64 MiB eviction，表中是最终代码状态的
10 次 cold median。baseline 文件没有改动。

| kernel | Weft | GGML baseline | Weft / baseline throughput | 时间差 |
|---|---:|---:|---:|---:|
| q4_K×q8_K，spec `mac_pairs` | 20.369742 ms / 5.765439 GOP/s | 13.009 ms / 9.027928 GOP/s | 63.86% | 56.58% slower |
| q4_K×q8_K，`mac_groups(4)` | 17.378719 ms / 6.757720 GOP/s | 13.009 ms / 9.027928 GOP/s | 74.85% | 33.59% slower |
| quantize q8_0 | 4.965072 ms / 369.583324 MEl/s | 3.292 ms / 557.450745 MEl/s | 66.30% | 50.82% slower |
| quantize q8_1 | 5.385064 ms / 340.758810 MEl/s | 4.503 ms / 407.500644 MEl/s | 83.62% | 19.59% slower |
| quantize q8_K | 5.614015 ms / 326.861970 MEl/s | 3.975 ms / 461.631658 MEl/s | 70.81% | 41.23% slower |

用户给出的 9.663 GOP/s 属于此前上下文；当前固定 baseline 表中同硬件、同 shape 的
q4_K×q8_K 数字是 9.027928 GOP/s，本报告只用后者作比较。此前 6.427287 GOP/s 的 Weft
记录使用错误的线性 nibble/sc/m 解释，虽然自洽，但不是真实 GGML layout，不能当作有效的
correctness+performance 基点。

## 六、落到具体实体的归因

### 6.1 q4_K

最终 assignment 的关键实体是：

```text
op23 q field memory edge
    grouped(64) + layered(32), raw u8m1, and-shift

op27 mac_pairs
    rvv.vwmaccsu, lhs u8m1, partial i16m2, group-major

op20 subs Level
    ordered sequential, unroll=1, pipeline=1, prefetch=0

op30/op38 sc/m memory edges
    joined(4,2,4), raw u8m1

op39 min dot
    rvv.vmacc, accumulator i32m4, 8 groups, 2 bsum values/group
```

这里没有“某个全局 LMUL 太保守”这一归因：在 16-row cohort、VLEN128 下，q raw u8m1、
i16 partial m2、i32 result m4 是一致的最小链。主要差距在两个更具体的位置：

1. **作者树的 MAC 粒度。** `mac_pairs` 每两个元素清零一次 partial、做两次 unpack/load/MAC、
   widen 后并入 i32。改成 `mac_groups(4)` 不改其他字节，吞吐提高 17.21%。4×15×128
   仍在 i16 内；因此当前 spec 树正确但性能保守。这一项必须由作者修改 std/spec 树，
   不能由 compiler 改写。
2. **op27 + op20 的局部生成空间仍窄。** 即使 group=4，当前仍是 unroll=1、pipeline=1，
   q unpack/load 与 MAC 没有跨 group 交错，也没有另一个真实 local schedule candidate。
   剩余 25.15% throughput gap 首先落在 D20（该 q value-use edge 的 unpack/load 组织）和
   D26（subs Level/local-op cluster 的 unroll/pipeline），而不是 Encoding 或整 kernel route。

### 6.2 q8_K

初版前向 pass 把 max/min 两个 reduce 分别物化，同一 block 被读取两遍，定向运行曾为
8.885181 ms。`SelectLocalOperations` 现在在两个 reduce op 上分别记录
leader/follower 与同一 input 的 `shared-per-stream-part` realization；emitter 只消费该选择。
最终 5.614015 ms，比该中间状态减少 36.82%。这说明原来“全局 reduction decision”确实
遮住了实体级共享关系。

剩余差距位于 q8_K 的具体 value/Level：256-element q result 是 8 个 i8m2 stream parts，
随后跨入 16-element subs Level 计算 bsum；当前 extract 需要在 8 个 part 中选择，subs
Level 仍是 unroll=1/pipeline=1。最大/最小已经共享，余下主要是该 q value 的跨 Level
handoff/materialization 与 subs Level schedule，而不是 q8_K 格式名。

q8_0 没有这组 max/min sharing，当前 f32m8 单 strip 的 reduction→scale→narrow 只有一个
确定实现；它的 50.82% 时间差说明 value LMUL/reduction/narrow 尚没有真实的外层 physical
candidate 空间。q8_1 多一条 sum 支路，但差距较小。三条结果没有支持“删掉全局字典就会
自动变快”的说法；架构改对只使瓶颈可定位，性能仍要由真实 candidate 和局部生成补齐。

## 七、当前明确阻塞

K1 IME 不在本轮性能范围，但 emitter 权责迁移后做了生成边界检查：当前 candidate 在
resource pass 被明确拒绝，因为 target capability 声明 IME fragment 固定占 28 groups，
再加 live RVV values 和保留组超过 32。旧 emitter 的 `VLEN256` 私判已经删除，但 IME
尚不能生成最终 C；这里不能用 emitter 缩小 LMUL或绕过 resource check。需要先统一“当前
实际 asm leaf 的 clobber/resource”与 target fragment capability，或在前序 pass 形成明确
的 memory/fragment handoff candidate。

