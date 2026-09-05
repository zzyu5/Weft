# Lookup-index carrier arbitration

日期：2026-09-05

## 实现前预测

### 作者前置条件

IQ1_S 的 canonical program 已显式保存 `group × entry × payload`、
`q[group * 4 + entry]`、`qh[group]` 与 indexed lookup。改动不增加 logical axis，
不改变 reduction、widening、Level、artifact 或 ABI。

### 七步工作账

每个 256-element record 的数学 product 固定为 256 个 main product 加 256 个 correction
product。SG2044/K1 分别有 4/2 个 issue，数学上需要 4/2 个 indexed table load、
4/2 个 activation window 与 8 次 partial reduction；这些不应减少。

当前 SG2044 每个 issue 重新加载 qh 子窗口、加载 q 子窗口并重建 index，最终为一个完整
qh carrier 加 4 个 qh 子载荷、4 套 index construction；K1 对应为 2 套。donor 从一次
完整 qh load 与一次完整 q load 构造一个 32-entry index carrier，再向 4/2 个 issue
提供连续窗口。

### 物理决定

对 indexed lookup 的 entry-index DAG，若多个 encoded storage roots 中恰有一个 natural
affine root 覆盖全部 entry axes，并且其连续跨度严格大于其它 root，则该 root 决定完整
index carrier；较小 root 通过 typed broadcast/conversion 供应它，不能反向把完整 carrier
降成 replica tuple。完整 carrier 的 LMUL 由 entry-axis 总跨度、SEW 与 target VLEN 唯一推导，
资源不足则该 physical binding 非法。

partial planner 若把这一 lookup 按 issue-time 分解，必须保留完整 index SSA producer，
以真实 typed issue-window projection供给各 `rvv_indexed_entry_load`；materializer 不得重新
克隆 storage/index DAG，emitter 不得重新选择 carrier。

### 静态上限与资源

- IQ1_S/SG：4 套 qh/q/index 供应降为 1 套加 4 个 issue projections；预计删除至少
  3 次 qh 子载荷、3 次 q 子载荷及 3 套 shift/mask/or index arithmetic。
- IQ1_S/K1：2 套降为 1 套加 2 个 projections；对应删除 1 套重复供应。
- SG 的完整 ui32[32] index 使用 m8，K1 使用 m4；必须连同 issue operand、partial 与
  accumulator 计算真实 register peak，不能只检查 index 本身。
- 收益应在 Physical IR 中表现为一个支配所有 issue 的完整 index producer和显式
  issue-window op，在 C/assembly 中表现为 qh/q 主载荷各一次及重复 index arithmetic 消失。

若这些计数不变，本机制没有实现价值，不进入真机。

### 第二输入与反例

IQ2_S 与 IQ2_XS staged 同样有 record-local entry/payload window 和 indexed lookup，作为
广义第二输入；其 scale-group geometry 与 IQ1_S 不同，所以只能证明同一个 carrier owner，
不能证明 IQ1_S 的精确 repartition 拼写。Q6_K 没有 indexed lookup，必须零行 diff；
此前“任意 storage proposal 接管 contraction carrier”的过宽规则已造成 Q6_K/IQ2_S
退化，本轮规则不得传播到 contraction result。

### 参考机制

Triton 的 AxisInfo/Coalesce 以 contiguity、divisibility 与 consumer layout 决定 memory-facing
layout；显式 conversion 承载冲突。TileLang 的 layout inference 合并同一 value 的 storage
与 consumer constraints，冲突无法闭合时失败。Weft 没有 thread/warp owner，因此不复制其
具体 layout；这里只复用“完整 memory relation 的 owner 先于机械 issue expansion”这一组织。

## 结果

MISS，不保留源码改动。

放宽 `feedsExpandedIndexedLookup`，让含多个非 singleton axes 的 natural storage root
直接成为 expanded lookup 的 storage anchor 后，IQ1_S/SG 的寄存器峰值从 25 降到 16，
但真正的动态工作几乎没有变化：indexed load 仍为 4 次，unit-entry window load 仍为
8 次，gather 仍为 8 次，reduction 仍为 9 次，slide-down 仍为 8 次。唯一消失的是
初始 layout pack 的 3 次 `vslide1up`；完整 qh/q/index producer 没有形成，issue-local
供应也没有塌成一个支配所有 consumers 的 SSA producer。因此它没有实现本节预测的 P2/P3
机制。

第二输入给出更强的反证。IQ2_S 与 IQ2_XS staged 在两台机器上共四个入口均从合法的
20--22 vector groups 变为编译失败：

```text
nested scaled contraction has no legal full-product carrier under the target resource contract
```

多轴 storage root 被当成 widening contraction 的 carrier，正是此前已经观察过的 owner
越界。Q6_K 因没有 expanded indexed lookup 而保持合法，但这一零影响不能抵消 IQ2 的
四个硬回归。

结论是：`storageRolesFor` 的多轴自然连续性不足以决定完整 lookup-index carrier；它只提供
memory-facing proposal。需要的机制还必须显式区分 storage root、lookup index result 与
contraction operand，并在 issue expansion 之前冻结一个真实的共享 index SSA producer。
仅删除轴数门槛既没有减少预测的供应工作，也越过了 carrier owner 边界，故撤销。

## 显式 lifetime 实验

`grid_index[group, entry]` 已经是作者程序中的完整 shaped value；它在一个 record block
内产生，并被该 block 的全部 issue 消费。先用现有 `materialize` 显式声明这个 birth/lifetime，
不新增 logical value、axis、reduction、widening 或 ABI。预期是 4/2 套 index construction
分别降为一套共享 producer；若 final IR 计数不变，就说明现有 residency owner 不能承载
这个 shaped register value，该改动不保留。

该实验在 frontend 被拒绝。`materialize` 是 Level birth，不是任意 SSA cache；当前 record
的 `w = admit(W[kb])` 只在该 Level body 内可用，birth region 不能反向捕获 `w`。这项拒绝是
正确的：作者 SSA 已经把 `grid_index` 放在所有逻辑 consumers 之前，重复它是 physical issue
materialization 的选择，不能让作者伪造一个新 Level 来补编译器的 placement。代码不保留。
