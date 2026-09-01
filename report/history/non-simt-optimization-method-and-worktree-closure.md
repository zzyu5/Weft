# 非 SIMT 优化方法取证与未提交工作收口

日期：2026-08-30

本报告记录两件事：从 Git/CSV/历史报告和 Triton/TileLang 源码得到的优化方法结论；以及本轮对未提交 physical IR、TQ1、IQ2 工作的拆分结果。本轮没有调优或更新正式性能 CSV。所有新运行都是为了验证一个提交节点是否能生成、交叉编译、真机执行并通过数值检查，重复次数为 1。

## 1. 优化方法结论

正式方法已写入 [`doc/compiler/optimization-principles.md`](../doc/compiler/optimization-principles.md)。草稿经过证据检查后形成：

```text
P0 先数工作：typed实体只有减少动态工作才是优化
P1 载体归属：axis在lane/register replica/issue time/fragment间的分配
P2 供应唯一：每个distinct producer在合法复用域内只产生一次
P3 形态跟随：memory form服从storage geometry与consumer layout
P4 延迟收敛：partial在宽形态完成工作后才reduce/extract
P5 issue-time重叠：在真实依赖允许时用pipeline隐藏latency
R  资源仲裁：完整live set是五条原则共同的合法性约束
```

两个草稿判断被修正：

1. P0不是优化类别，而是所有新机制的进入门槛。projected window有真实typed op仍然变慢，read-CSE静态删除0个op，证明“有名字”不等于“少工作”。
2. software pipeline必须独立于P2/P4。它可能增加guard、buffer和指令但缩短critical path；Q4_0/Q4_1正收益与F16/decode负结果同时存在。

关于Triton的原猜测也过强。SIMT只在固定shape/thread/warp配置下提供部分覆盖守恒关系；`OptimizeThreadLocality`、`AccelerateMatmul`和warp-specialization仍会改变ownership维度、复制、shuffle、warp数、MMA microtile和register分配。Weft的差异不是“Triton总量固定”，而是Weft没有预先存在的thread/warp/CTA owner，必须直接为完整logical value决定time/lane/replica/fragment分解。

## 2. 证据来源

每条正式原则至少有两个独立历史结果：

| 原则 | 证据一 | 证据二 |
|---|---|---|
| P0 | `fec0c9943` Q2 projected window：SG `2.136→1.763`，K1 `1.823→1.660` | `ff89f1ec1` read-CSE删除0个op；独立Q5_K在`b72cefad0`中bytes不变而指令1.99×、吞吐0.729× |
| P1 | `00afcbbce` reduction-lane在十格式双机提升；SG Q4_0 `0.501→7.113`、Q5_K `0.345→6.968` | `8ca6a994c` Q4_K correction改为free-axis lane后峰值 `34→20`，资源非法变为SG `10.128` |
| P2 | `d2e5758bc` IQ4_XS共享activation window：SG `3.472→7.667`、K1 `1.277→2.326` | `8ca6a994c` Q1 raw byte一次供应八个consumers：SG `3.376→6.912`、K1 `1.828→3.479` |
| P3 | `8ca6a994c` Q4_1 layered geometry：SG `4.711→6.691` | 同一typed规则覆盖Q5_1 `3.984→6.464`和K1 TQ2 `2.312→3.681` |
| P4 | `c29d835ac` TQ2 typed partial：SG `1.564→19.977`、K1 `3.681→6.245` | Q2 independent partial：SG `4.845→9.970`、K1 `1.660→2.508` |
| P5 | `63de9c28f` Q4_0 pipeline：SG +51.4%、K1 +22.4% | Q4_1：SG +45.4%、K1 +36.0%；反例F16 SG -8.9% |

Q3_K的十一项实验支持资源仲裁结论，而不是另一条优化原则：扩大Q8 load会把load和partial consumer分成更大的独立SSA live set；m8实验降到约`2.68` GOP/s并产生spill，NR2/LMULm4达到`36/32`直接非法。缺口是能同时描述联合leaf输入、temporary、output与lifetime的typed local operation，不是再扫一次load width。

## 3. 与Triton/TileLang的机制关系

- Triton `CoalesceUtils.cpp`由pointer contiguity/divisibility、shape、thread count与load/store语义决定coalesced layout；对应P1/P3。
- Triton `RemoveLayoutConversions.cpp`比较conversion与rematerialization成本，并按dominance重写；对应P0/P2。
- Triton `ReduceOpToLLVM.cpp`从register/lane bases与reduction axis生成partial tree；对应P4。
- Triton pipeliner先生成schedule，再由expander生成prologue/epilogue；对应P5。
- TileLang layout inference既消费作者annotation，也在free mode用memory/register cost选root；这反驳了“layout完全唯一推导”。
- TileLang reducer planner从update site投影partial storage；pipeline planning与injection同样分开。

Weft采用这些机制的typed representation、explicit conversion、真实IR rewrite与scheduler/expander分工，不采用其thread/warp/shared-memory ownership作为source语义。

## 4. 工作区拆分结果

### 4.1 `3d7939a5f`：历史报告归档

用户已移动的10份报告内容与HEAD逐字一致，仅从`report/`迁入`report/history/`，单独提交。

### 4.2 `0480e2001` + `9193007da`：typed indexed-entry load

新增完整physical链：

```text
RVVIndexedEntryLoadOp / RVVUnitEntryWindowLoadOp
→ verifier
→ indexed-entry relation与memory planning
→ reduction/indexed layout projection
→ final verifier
→ intrinsic C
```

IQ2_XXS临时shaped tree在SG/K1 final IR均形成6个`rvv_indexed_entry_load`。双机1-repetition数值误差均为0：

| target | Weft | source | ratio |
|---|---:|---:|---:|
| SG2044 | 1.975669 | 4.051562 | 48.8% |
| K1 | 0.680800 | 1.684037 | 40.4% |

该结果证明indexed-entry physical operation可执行，但没有证明IQ2树高性能。

原本同时存在的`rvv_coalesced_widen_reduce`在intended IQ2 final IR中出现次数为0，因此没有独立consumer，也没有减少工作；本轮完整撤回。

### 4.3 `04264344e`：grouped MAC只消费storage-window plan

`LowerRISCVComposites`不再从group/layer/order重新计算physical layer、shift和mask；它只消费`storageWindowPlan`。Q4_K `mac_pairs`与`mac_groups(4)`在SG/K1均形成2个相同种类的`rvv_grouped_mac_reduce`，差异只保留作者真实group与term order；四份IR均通过`weft-opt --verify-each`。

### 4.4 `f7c5ba7a7`：TQ1 shaped partial与默认repro

TQ1 radix-3计算不再逐元素展开：

- 32-lane与16-lane段先形成i16 shaped partial，再widen/reduce；
- qh第三段形成16-lane regular-repeat gather；
- MUL_MAT形成blocked output cohort；
- runner为TQ1提供`NC32/MC16/MR2/NR2`默认绑定；
- dead SCF iter-arg cleanup删除不再使用的physical carries。

去掉dead-iter-arg cleanup时，SG/K1系统Clang都因`for_carry_* set but not used`在`-Werror`下失败；恢复后下列六个真实入口全部零误差：

| entry | SG GOP/s | SG/source | K1 GOP/s | K1/source |
|---|---:|---:|---:|---:|
| vec-dot | 3.469639 | 56.0% | 1.720083 | 50.5% |
| MUL_MAT prefill | 6.002272 | 97.1% | 2.827928 | 81.8% |
| MUL_MAT decode | 3.232528 | 52.6% | 1.506420 | 45.1% |

这些是提交闭合所需的1-repetition观察，不写入正式CSV。TQ1仍未达到source；文档预测其首要compiler缺口是每个32/16-lane段内五个radix-3 products的P4 partial topology，其次是P3 regular-repeat/storage organization。

## 5. IQ2负实验如何处理

IQ2_XXS的临时shaped std tree、staged wrapper、runtime ABI和runner selector均已撤回。原因不是数值错误，而是：

- indexed-entry op带来真实改进，但最终仍只有source的48.8%/40.4%；
- MUL_MAT仍是row×column，没有output cohort；
- group 1–7仍由Python loop展开；
- indexed load没有解决跨output supply、partial topology和pipeline。

负结果保存在本报告；源码不保留第二套IQ2入口或失败staged路径。

## 6. 预测边界

正式文档已经固定IQ1/IQ2/IQ3/TQ1/Q4_K canonical/Q5_K剩余入口的首要类别与预期IR/汇编现象。预测不是实现授权：作者轴不存在时先改std tree；physical规则仍只能读取axis、storage geometry、use-def、effects、target facts与resources。后续结果若有一半以上不符合主要分类，这套方法应被撤销重写，而不是修改解释迁就结果。
