# IQ1 indexed-offset issue window

日期：2026-09-05

## 实现前预测

### 作者前置条件

作者程序已经显式给出完整 `group × entry` index value、codebook lookup 与 contraction
轴。改动不增加 logical axis，不改变 reduction、scale、widening、Level 或 ABI。

### 当前工作账

IQ1_S SG2044 standalone 每个 256-element record 当前包含：

- 一次完整 8-element qh load；
- 四个 issue，每个 issue 为 grid index 再载一次 2-element qh window；
- 四个 issue，每个 issue为 scale 再载一次同一 2-element qh window；
- 合计 8 次额外窄 qh load、8 次 index-side `vrgather`，以及 scale extraction 的
  8 次 scalar slide/extract。

不同 issue 的 logical offset 不同，因而这不是 read-CSE。donor 先从一次完整 qh/q
供应形成完整 32-entry index carrier，再向四个 issue 提供连续窗口。

### 物理决定

planner 在以下 typed facts 同时成立时选择 indexed-offset issue window：

- contraction operand 由 `rvv_indexed_entry_load` 产生；
- entry-offset value 的完整 RVV carrier 已闭合；
- issue window 沿一个显式 logical axis 构成连续、等宽分区；
- 完整 offset、projected offset、lookup result、product 与 accumulator 的联合寄存器
  合同不超过 target budget。

选择结果进入 nested partial plan。materializer 只创建真实 issue-window op 与已有
indexed-entry load；emitter 不重新选择是否 hoist 或怎样切 window。

### 可见预测

- IQ1_S SG 的 8 次窄 qh load 与对应重复 index construction 应减少；
- standalone vec-dot 与 MUL_MAT decode 应同步变化；
- IQ1_M 是第二个独立输入：它同样从完整 32-entry offset carrier 向多个 issue 投影，
  但 qh 和 scale encoding 不同；
- 若最终 IR/C 的供应计数不变，或联合资源峰值变成非法，则该机制不保留；
- 若只在 IQ1_S 生效，不能宣称跨格式泛化。

### 参考机制

Triton 的 memdesc/local-allocation 路径保留一个已选完整 carrier，再以显式 subslice
提供 consumer window；TileLang 的 pipeline buffer region 同样把 producer allocation
与 stage/consumer window分开。Weft 不复制 thread/warp ownership，而用 logical axis、
RVV lane window、use-def 和 register budget闭合相同关系。

## 实现后结果

该预测没有闭合，代码未保留。

第一次实现只接受完整 lane carrier；IQ1_S 的 entry-offset producer 实际是
`group lane=8 × entry replica=4`，因此候选数为 0，最终 IR 与基线逐行相同。
把准入条件扩到完整 time carrier 后，planner 确实选择了 issue window，但程序在
final verifier 停在：

```text
final part-to-lane conversion has no closed RVV pack or encoded rematerialization
```

失败的具体关系是同一个 `[group=8, entry=4]` `ui32` 值从
`lane=[8,1], replica=[1,4]` 变为
`time=[4,1], lane=[2,4], replica=[1,1]`。这不是单轴的 lane slice：它同时把
group 的一部分从 lane 移到 issue time，并把 entry 从 register replica 移到 lane。
现有 `rvv.issue_slice` 与 part-to-lane conversion 都不能表达这类跨两个 logical axes
的 repartition；让 emitter 临时拼装会把物理决定移回终端。

静态预测的 8 次窄 qh load 虽然真实存在，但新增 issue-window op 只给这个工作命名，
没有让 producer 直接产生 consumer 所需的 carrier。Triton 的对应做法不是在终端增加
一个 slice，而是由 `RemoveLayoutConversions` 对 backward slice 中的 producer/operand
逐个推导目标 encoding、克隆 producer，并在昂贵 memory anchor 前停止。Weft 当前的
`CanonicalizeRISCVLayouts` 只覆盖单用的纯 Unary/Binary/Cast/Narrow/Widen DAG；
storage/indexed-entry 根和这种跨轴 repartition 尚不在其合同内。

因此当前确定的缺口是：**以 selected indexed-entry consumer layout 为目标、跨纯
index DAG 回溯到 storage anchor 的 typed backward rematerialization**。它必须同时
冻结 producer result layout、各 operand projection、dominance/reuse 和联合资源合同，
而不是再增加一个由 materializer 或 emitter 解释的 window 宏。IQ1_S 是一个输入；
IQ1_M 的 field geometry 不同，而且当前 main/correction 已共享 qh/index，合法 read-CSE
可删除的供应为 0，不能把它当作第二个正例。

因为本次实现不能生成合法 Physical IR，也没有第二个独立输入，所有新增 op、plan 字段、
materializer/emitter 路径均撤销；CSV 没有写入这次实验的性能数字。

## 同轮 IQ1_M 复测与边界

当前 HEAD 的正式 targeted rerun 为：

| target | entry | Weft | source | ratio |
|---|---|---:|---:|---:|
| SG2044 | standalone vec-dot | 1.091291 | 2.650472 | 41.17% |
| SG2044 | MUL_MAT decode | 2.033534 | 2.639333 | 77.05% |
| K1/X60 | standalone vec-dot | 0.860998 | 1.435304 | 59.99% |
| K1/X60 | MUL_MAT decode | 0.846799 | 1.424159 | 59.46% |

K1 的 standalone/decode 同步；SG 的 standalone 在更长的 standalone 工作集下明显更慢，
不是一条旧 CSV 行可以解释的微小噪声。静态程序中 qh/index/delta 已在 main 与 correction
consumer 之间共享，按 identity 合法合并能删除的 raw load、index conversion、grid gather
均为 0。

当前 topology 把 `entry,payload` contraction 结束后留下的 `scale_part,group` 依次归约，
SG/K1 分别产生 16/8 次 vector reduction；donor 把 scale 组合进 i32 partial 后只做两次
最终 reduction。把前者变成后者会改变 scale 与 reduction 的结合位置以及中间 widening
边界，属于用户要求停下确认的数值程序边界，不能作为 physical pass 的等价改写继续做。
