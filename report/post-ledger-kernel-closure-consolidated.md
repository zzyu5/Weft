# 全局对照表建立后的 kernel 收口合并报告

日期：2026-09-05

## 1. 范围与证据口径

本报告合并从提交 `3d16f3700`（首次建立人可读性能对照表）到当前提交
`d4e8525b0` 的工作，并记录报告暂停前尚未提交的 Q4_K / IQ1_S 静态调查。
这一范围内原有的十份阶段性 Markdown 已由本报告替代；完整逐行性能数据仍以
`report/kernel-performance-comparison.csv` 为准，固定 source 数据与当前 Weft 全量数据的
职责不变。

事实来源按以下优先级使用：

1. 当前源码和当前对照 CSV；
2. Git 中已提交的代码 diff 与当时的十次 targeted rerun；
3. 已撤销实验只记录可复核的静态计数和真机数字，不把其结论写成现状；
4. 尚未完成真机验证的调查只写成“暂停点”，不更新 CSV。

起点报告有 206 个结果；当前表有 202 个结果。减少的四项是 IQ2_S 与 IQ2_XS
staged prefill 的双机入口。它们没有可执行结果，因而没有用旧数字或 `FAIL` 占位。
为了避免分母变化制造进展，下表同时给出原始口径和共同 202-key 口径：

| 口径 | 结果数 | ≥100% | 90–100% | 70–90% | 50–70% | <50% | ≥90% | <70% |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 起点原始账 | 206 | 132 | 28 | 27 | 6 | 13 | 160 | 19 |
| 起点、限制到当前 202 keys | 202 | 128 | 28 | 27 | 6 | 13 | 156 | 19 |
| 当前 HEAD | 202 | 139 | 31 | 27 | 4 | 1 | 170 | 5 |

当前没有数值 FAIL。当前 202 个结果中 108 行来自后续十次 targeted rerun，其余仍来自
最近一次全量锚点；`measurement` 列明确区分二者。

## 2. 这一段实际形成的主干能力

### 2.1 IQ1 多级 reduction topology

IQ1_M 的 canonical program 显式保留 `group=8`、`scale_part=2`、`entry=2`、
`payload=8`。contraction 只消去 `entry,payload`，随后分别消去 `scale_part,group`。
Physical planner 现在把所有存活轴、后续 reduction 轴、partial set 类型、combine arity、
scale supply stage 和资源峰值冻结在同一个 topology 中；materializer 只实例化：

```text
partial_set
  → partial_repack
  → partial_reduce
  → partial_scale_combine
  → partial_combine
  → partial_finalize
```

SG2044 静态结果从 `32 vwmul + 32 vwredsum + 39 vslidedown` 变成
`4 vwmul + 16 vwredsum + 0 vslidedown`，并形成真实 typed partial program。
这验证的是多级 topology 可以承载作者已经写出的 reduction 关系；它没有证明当前
IQ1_M 数值程序已经达到 donor 的实现形态。

同一提交还关闭了 IQ1_S 的 record-local affine base：`group*4` 不再因表达式复杂而被
默认为零，四个 SG issue 保留 `0,8,16,24` 的 typed record offsets。不能证明的表达式会明确
失败，而不是落入错误默认值。

### 2.2 Cartesian output carrier 与 sub-Level issue topology

Physical layout propagation 现在能把两个 operand 各自贡献的 free axes 组织成共同的
Cartesian output carrier；reduction sub-Level 的 issue decomposition 也在
`LowerRISCVComposites` 与 partial planner/materializer 之间闭合。由此得到的直接结果包括：

- Q2_K prefill：SG `5.734393→9.701467`，source `9.658093`；K1
  `1.570078→3.033368`，source `2.452543`；
- Q6_K prefill：SG `6.881827→7.423354`，source `7.443768`；K1
  `2.035951→2.066771`，source `2.506236`。

Q2_K 双机过线；Q6_K 的合法性和 topology 已闭合，但 K1 prefill 的性能仍未过 source。

### 2.3 IQ1 decode / prefill 的作者程序分离

IQ1_S 与 IQ1_M 的 decode 继续使用单输出 contraction；prefill 采用显式
`NC→MC→NR→MR→K-block`、W/Xq panel、`[MR,NR]` accumulator 与 output cohort。
这是两棵作者程序，不是 compiler 根据 M 或 VLEN 暗改 canonical tree。

IQ1_S blocked prefill 的结果具有明显正反两面：

| target | row×column | blocked / 当前 | source | 当前/source |
|---|---:|---:|---:|---:|
| SG2044 | 0.636727 | 3.505333 | 4.854846 | 72.20% |
| K1/X60 | 0.892714 | 2.836568 | 2.740584 | 103.50% |

K1 过线；SG 从中间版本的约 91.7% 回落到 72.2%。原因不是 blocking 消失，而是 selected
issue schedule 与联合 supply/resource 合同只在 K1 的 binding 上闭合得好。

IQ1_M blocked prefill 的当前结果：

| target | 旧 shaped prefill | 当前 blocked | source | 当前/source |
|---|---:|---:|---:|---:|
| SG2044 | 0.525161 | 2.254551 | 2.634283 | 85.59% |
| K1/X60 | 0.421602 | 1.562290 | 1.446879 | 107.98% |

### 2.4 exact two-stream widening combine

当 typed facts 证明存在恰好两个 signed-i16 partial、目标为 signed-i32 且联合资源合法时，
planner 冻结 widened combine，materializer 产生真实 `rvv_partial_combine`，emitter 只拼写
选定的 `vwadd`。动态形态从：

```text
2 × vwmul + 2 × vwredsum + 2 × scalar extract + scalar add
```

变为：

```text
2 × vwmul + vwadd + vredsum + scalar extract
```

K1 上该关系在 MXFP4、IQ4_NL、IQ4_XS、NVFP4 等独立输入中生效。当前代表性结果为：

| entry | standalone/decode/prefill 对 source |
|---|---|
| MXFP4 K1 | 91.94% / 91.24% / 109.73% |
| IQ4_NL K1 | 84.34% / 84.24% / 96.23% |
| IQ4_XS K1 | 153.14% / 154.48% / 139.34% |
| NVFP4 K1 | 596.88% / 584.75% / 657.56% |

这一机制在 VLEN128 上不是普遍收益，相关负结果见第 6 节。

### 2.5 row-dequant 的显式 entry/payload 与 physical parameter

IQ2_XXS 原程序把一个 32-element group 写成 flat axis，再以 `element/8`、`element%8`
恢复 entry/payload。作者树改为显式 `entry=4 × payload=8` 后，现有 indexed-entry mechanism
直接选择 unit payload load；数学输出、lookup bytes、浮点顺序、ABI 均不变。

final Physical IR 每个 entry 有两个 typed unit entry load，generic `vluxei` 与
`vslidedown` 均为零；SG/K1 register peak 分别为 4/2 groups。十次结果：

| target | 修改前 | 当前 | source | 当前/source |
|---|---:|---:|---:|---:|
| SG2044 | 345.542296 MEl/s | 692.394905 MEl/s | 443.356980 MEl/s | 156.17% |
| K1/X60 | 200.152757 MEl/s | 372.473967 MEl/s | 148.438660 MEl/s | 250.93% |

IQ2_S 的 shaped program 和 typed memory edge原本已经存在，低值来自 runner 的 LMUL=m1
把八个 f32 输出拆成两个四-lane carrier。SG 绑定到实测 LMUL=m8 后，两次 conversion、
两次 slide 和分裂 store 消失，peak 仍只有 4 groups：

| target | 当前 | source | 当前/source |
|---|---:|---:|---:|
| SG2044 | 673.905403 MEl/s | 426.506007 MEl/s | 158.01% |
| K1/X60 | 225.564634 MEl/s | 146.691980 MEl/s | 153.77% |

前者是作者逻辑轴补全；后者只是实测 physical parameter binding。两者不能混写成同一种
compiler pass 收益。

## 3. 测量账修正

这一段没有再相信旧 CSV 行来判断回归：

- Q6_K 旧账中 standalone/decode 相差 16.2%；当前同 revision 重跑后 SG 差 6.3%、K1
  差 0.2%，原异常属于旧快照污染；
- IQ2_XS SG standalone/decode 重跑为 100.17% / 104.68%；
- IQ4_NL K1 重跑为 standalone 84.34%、decode 84.24%、prefill 96.23%；
- row-dequant 改动只更新人可读 comparison CSV 的对应 targeted 行，没有把局部结果冒充
  新一轮 full-run；
- 所有写入 comparison CSV 的本段新数字均为当前源码、相同输入合同、十 repetitions 且数值
  在容差内。

当前低于 70% 的五项是：

| entry | target | Weft | source | ratio | 已确定边界 |
|---|---|---:|---:|---:|---|
| IQ1_M vec-dot | SG2044 | 1.091291 | 2.650472 | 41.17% | donor 需要不同 scale/reduction/widening 程序 |
| TQ1_0 vec-dot | SG2044 | 4.816287 | 8.123996 | 59.28% | donor 需要不同 radix contraction/reduction 程序 |
| IQ1_M MUL_MAT decode | K1/X60 | 0.846799 | 1.424159 | 59.46% | 与 vec-dot 同一 numerical boundary |
| TQ1_0 MUL_MAT decode | SG2044 | 4.784427 | 8.015470 | 59.69% | 与 standalone 同步，差距在 contraction 内部 |
| IQ1_M vec-dot | K1/X60 | 0.860998 | 1.435304 | 59.99% | 与 decode 同步 |

## 4. 提交与层次归属

| commit | 层次 | 已提交事实 |
|---|---|---|
| `3d16f3700` | ledger、作者 axes、memory legality | 建立 comparison CSV；Q6_K/IQ1_M 无 FAIL；暴露 IQ1 bottom cluster |
| `db2035a65` | Canonical + Physical topology/pass | IQ1 多级 reduction 与 record-local window 闭合 |
| `6efeeccb2` | Physical memory + runner | radix unit window 与 row-decode bindings 闭合 |
| `fafc7688d` | Physical layout pass | Cartesian contraction output carrier |
| `cb823cb75` | Physical composite/partial passes | reduction sub-Level issue topology；Q2_K prefill 双机过线 |
| `c0f6cbb62` | measurement | Q6_K decode 旧快照污染纠正 |
| `99d46d001` | Physical schedule/final verifier + runner | selected stream schedule进入 final contract |
| `0eb5dca65` | IQ1_S 作者树 + Physical partial plan | blocked prefill；K1 过线、SG 负结果保留 |
| `7e062b416` | measurement | IQ2_XS 刷新；IQ4_NL range boundary 隔离 |
| `a555a21de` | negative report | indexed-offset issue-window 不闭合，代码不保留 |
| `4f238206b` | IQ1_M 作者树 + runner | blocked prefill；K1 过线、SG 85.59% |
| `a9437b00e` | target profile + Physical plan/op/emitter | exact two-stream widening combine |
| `02c951877` | negative report | lookup carrier owner 过宽的反证，代码不保留 |
| `35a9711cf` | IQ2_XXS 作者树 + runner | entry/payload shaped row-dequant 双机过线 |
| `d4e8525b0` | runner physical binding | IQ2_S row-dequant SG 绑定 LMUL=m8，双机过线 |

## 5. 与 Triton / TileLang 的具体对应

本段没有把参考实现简化成“layout propagation”一个动作：

- Triton `RemoveLayoutConversions.cpp:218-413,884-1218`：先设 layout anchors，处理 consumer
  conflict，再沿 backward slice rematerialize；多 use、dominance、slice 外 use 与 remat 成本都
  是显式条件。Weft 当前只完成单 producer/部分 memory relation，IQ1_S 的跨轴 index DAG 尚未达到
  这一合同。
- Triton `ReduceOpToLLVM.cpp:228-351`：register bases、lane bases 与被消去轴共同决定
  local/lane reduction。Weft 的 IQ1 多级 partial topology借用的是“所有 surviving/reduction
  axes 保持显式，再 materialize combine”，不是 GPU thread collective。
- Triton `CoalesceUtils.cpp:17-94` 与 `Coalesce.cpp:80-119`：从 pointer contiguity、
  divisibility 与 consumer shape 选择 memory layout，并把选择写回真实 IR。IQ2_XXS 的
  entry/payload 修复说明 Weft 只有在作者坐标可见时才能做对应的 unit-payload 选择。
- TileLang `reducer_plan_materialize.cc:559-946`：从 update-site reduction axes 投影 partial
  storage，要求多个 update site 的 plan 结构一致。Weft 的多级 reduction topology采用了同样的
  “planner 冻结、materializer 验证并实例化”责任分离。
- TileLang `layout_inference.cc:342-425,1027-1170`：layout inference围绕同一 buffer/use
  component传播，并以 memory/register cost仲裁。Weft 不复制其 thread ownership 或全局 cost
  model，只借用 storage anchor、consumer demand、dominance 与显式 conflict 的组织。

## 6. 已执行且未保留的负实验

### 6.1 IQ1_S storage anchor 过宽

把多轴 natural storage root 直接当作 expanded lookup 的 carrier 后，IQ1_S/SG register peak
由 25 降到 16，但 indexed load 仍 4、unit-entry window 仍 8、gather 仍 8、reduction 仍 9、
slide-down 仍 8；只删了 3 次初始 `vslide1up`。P0 工作账没有闭合。

更重要的反例是 IQ2_S/XS staged 双机四个入口从合法的 20–22 groups 变为：

```text
nested scaled contraction has no legal full-product carrier under the target resource contract
```

原因是 storage proposal 越权接管了 contraction carrier。该规则完整撤销，只保留反证报告。

### 6.2 IQ1_S indexed-offset issue window

期望关系是共享完整 `[group=8,entry=4]` offset producer，再向 4/2 个 issue 投影窗口。
实际 producer layout 为：

```text
lane=[8,1], replica=[1,4]
```

consumer 需要：

```text
time=[4,1], lane=[2,4], replica=[1,1]
```

这同时把 group 从 lane 移到 time、entry 从 replica 移到 lane；现有 `rvv.issue_slice` 与
part-to-lane conversion 只能处理较窄的单轴关系。final verifier 正确拒绝：

```text
final part-to-lane conversion has no closed RVV pack or encoded rematerialization
```

新 window op 只给工作命名，没有形成 producer-direct consumer layout。所有新增 op、plan、
materializer/emitter 路径均撤销，CSV 没有写数字。用 source `materialize` 伪造 cache lifetime
也被 frontend 正确拒绝：Level birth 不能捕获同一 Level body 内才产生的 `w`。

### 6.3 IQ1_S schedule / supply 小实验

- scale 全标量化：吞吐下降；
- qh carrier 直接放宽：没有减少 issue-local supply；
- pipeline=2：没有形成可证明收益；
- 更宽 unroll：增加 live set或无收益；
- operation-major 宽泛化：K1 约 `2.048→1.789`，peak `15→23`；只保留
  `window_extent=1` 的窄合同；
- local vector scratch：减少部分 scalar op，但 stack roundtrip 后 K1
  `2.048→2.039`；
- bsum pair-fold：leaf 仍是两条 strided load，K1 `2.048→2.024`。

### 6.4 VLEN128 two-stream wide carrier

在 VLEN128 直接采用 wide carrier：

| entry | 修改前 decode/prefill | 实验 decode/prefill |
|---|---|---|
| MXFP4 SG | 6.522843 / 7.658325 | 6.003909 / 7.095727 |
| IQ4_NL SG | 6.529264 / 9.126334 | 5.861625 / 7.470585 |

live/resource 代价超过减少的 final reduction，实验撤销。K1 的正结果不能外推为
VLEN128 的固定规则。

### 6.5 IQ2_XXS flat-tree 参数扫描

flat tree 上 LMUL=m2 更慢、LMUL=m8 超资源预算、unroll=4 下降到
`214.568932 MEl/s`。这些负结果排除了继续扫物理参数；真正的缺口是作者没有表达
entry/payload axis。改树后的正结果见 2.5。

### 6.6 physical u16 lookup offset 单项实验

IQ codebook index 的范围可证明小于 65536，`vluxei32→vluxei16` 在表示上合法。但此前
IQ2_XXS 单项实验只改变 indexed intrinsic，SG `3.736→3.735`、K1 `1.689→1.688`，动态
producer/consumer 数不变，故撤销。只有当窄 offset 能使完整 index carrier 合法并进一步删除
重复 supply 时，这个选择才有新的 P0 理由。

## 7. 暂停前未提交调查

### 7.1 IQ1_M：topology 已完成，剩余是作者数值程序边界

当前 Weft 先完成 i32 `contract(entry,payload)`，再乘 scale并依次 reduce
`scale_part,group`；donor 保留两个 i32m4 accumulator，把 scale 融入 lane partial，最后只做
两次 reduction。把前者改成后者会改变 scale/reduction 结合位置和 widening/overflow boundary。
这不能由 Physical pass 改写，也不能用 target/VLEN 选择另一棵 hidden tree。

因此 IQ1_M 不是“partial topology 还没实现”；实现已经把 32 个 product leaf 收成 4 个，当前
剩余性能要继续推进，需要一棵明确的作者数值 variant。暂停时没有擅自增加该 variant。

### 7.2 IQ1_S：缺 consumer-driven index-DAG rematerialization

当前 SG 每个 record 的工作账：

- qh：一次完整 8-element load，另有 4 个 2-element issue loads；
- q：4 个 8-byte issue loads；
- grid：4 个 `vluxei32`；
- qh/index broadcast：8 个 `vrgather`；
- activation：4 个 64-byte `vle8`；
- bsum：2 个 `vlse16`；
- product/reduction：4 个 full-product issue、8 次 reduction。

donor 使用一次完整 qh load、一次完整 q load，再向四个 issue提供 index/activation工作。
现有可见的缺口是：由 selected indexed-entry consumer layout 提出需求，沿纯
`iota/mul/add/shift/mask` DAG backward rematerialize，并在 storage anchor 前停止；同时闭合
dominance、multi-use 和联合 resource contract。它不是 read-CSE，也不是再加一个 window 宏。

IQ1_S 的最大 codebook byte offset为 `2047*8+7=16383`，因此 physical-u16 offset 在值域上
合法并可把完整 index carrier资源减半；IQ2_XXS/IQ3 等提供第二份范围事实。但单独窄化已经
证明零收益，完整 carrier + backward rematerialization 的联合关系尚未实现，因而没有更新 CSV。

### 7.3 Q4_K canonical shaped 实验

暂停前曾临时把 canonical Q4_K 写成 `sub=8 × k=32` shaped contraction，并补足多 group
grouped/layered affine window与 issue partial。以下数字来自本次暂停前的命令输出与工作记录，
没有对应 Git 对象，因而不作为可由当前仓库独立重放的已提交结果：

- 初始资源为 `49/32` groups；定位到 32-group runtime coordinate后，typed window将 peak降到
  25；
- scale combine 从 8 个 fanout-1 chain 收成 2 个 fanout-4 chain，删除 6 个 scalar extract与
  6 个 scalar add；
- SG 十次结果从该实验中间态 `7.143287` 到 `7.824273`，仍低于 source `10.143147`，也低于
  当前已提交 canonical `8.093763`；
- K1 停在 `rvv_regular_repeat_scalar_load` legality；`min_pair=2` 又增加 8 个 two-element load
  和 slide packing。

所有源码/编译器实验文件均恢复到 `d4e8525b0`，CSV 未更新；当前工作区只剩本报告对十份
阶段报告的合并替换。donor 会先在 i16 合并两个
`bsum` 再 widening multiply，而当前作者程序先分别 widen 到 i32 再相加；继续追 donor 会改变
widening 与整数溢出边界，需要作者数值 variant，而不是再放宽 pass。

### 7.4 IQ3_S row-dequant

SG 当前为 `253.651418 / 358.175832 = 70.82%`。数学工作量已与 donor 一致；当前把两个相邻
4-byte entry 组织成一个 8-lane indexed lookup，同时 qh/sign 的 half-byte relation不能闭合为
独立 typed window。24 个格式中没有第二个同时具备“相邻多-entry payload + 独立 half-byte
sign projection”的输入；IQ3_XXS 只能证明 entry 一半。因此没有为单个组合关系冻结新 op。

## 8. 当前确定的未闭合边界

- IQ1_M：作者数值 variant，涉及 scale/reduction 结合和 widening；
- TQ1_0 SG vec-dot/decode：standalone 与 decode 同步，donor 的单 i16 carrier与最终 reduction
  需要另一棵 radix contraction program；
- IQ1_S SG standalone/decode/prefill：consumer-driven index-DAG rematerialization、完整 offset
  carrier与联合 resource contract；
- IQ4_NL K1：任意 i8 ABI 下不能无 range proof 使用 donor 的窄 accumulator；
- IQ3_S SG row-dequant：two-entry payload 与 half-byte sign projection 的组合关系缺第二输入；
- Q4_K canonical SG：当前 Physical program仍未达到 donor，继续改变 widening顺序属于作者边界；
- IQ2_S/XS staged prefill：实验入口仍有 full-product carrier legality 缺口，未用旧数字掩盖。

## 9. 机械验收状态

在涉及 Physical attr/schema 的最后一次完整结构收口点，24 个 vec-dot 与 24 个 row-dequant
在双机共 96 份 Physical IR 均能：

- 独立 parse / verify；
- 安全运行 canonicalizer 与 CSE；
- 连续运行 Share 两次；
- 通过 final verifier 与 round-trip verifier；
- 同一整套流水第二次运行文本 diff 为零。

随后 IQ2_XXS 只修改一个作者 row-dequant entry；其双机 Physical IR 单独通过同一机械流程，
IQ2_S 只修改 runner binding。暂停前 Q4_K 的实验未通过最终双机验收，已全部撤销；本报告
合并所描述的代码基线是已提交的 `d4e8525b0`。
