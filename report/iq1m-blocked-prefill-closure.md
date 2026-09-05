# IQ1_M blocked prefill closure

日期：2026-09-05

## 实现前七步账

### 1. 作者前置条件

当前 `production_mul_mat_iq1_m` 在量化 activation 后逐 row、逐 column 调用同一个
IQ1_M vec-dot。vec-dot 已显式写出 `group=8, scale_part=2, entry=2, payload=8`，
并明确给出 main/correction 两条 `entry,payload` contraction、随后
`scale_part,group` reductions，以及每个 256-element K block 的 f32 累加。

### 2. 当前工作账

当前程序没有 output cohort、W/X panel materialization 或 MR×NR accumulator。同一个
activation block 为每个 column 重新进入完整 vec-dot，同一个 weight block为每个 row
重新进入完整 vec-dot。外层 GEMV/GEMM traversal 因而没有把作者已知的跨输出 reuse
写进程序。

### 3. 本轮作者决定

将 traversal 写成 `NC → MC → NR → MR → K-block`，为每个 `[MR,NR]` output cohort
建立独立 f32 accumulator，并在共同 NC/MC lifetime materialize W/Xq panel。单个输出内
仍严格保持：

```text
contract(entry,payload,acc=i32)
  -> reduce(scale_part)
  -> reduce(group)
  -> main + 0.125 * correction
  -> block_scale * x.ds
  -> 按 K-block 顺序累加 f32
```

输入 bytes、grid table、widening、scale/reduction 结合位置、Level 中的 K-block 顺序和 ABI
均不改变。

### 4. 编译器责任

Canonical IR 承载真实 Level、materialize、output axes 和 accumulator；Physical compiler
仍负责为这些 shaped values 选择 lane/time/replica carrier、memory form、partial topology
与资源合法 binding。compiler 不得把 row×column 程序自行改成 blocked 程序。

Triton 的作者同样显式写 program tile 与 `tl.dot` 的数值结构，
`AccelerateMatmul` 再把 result/operand encoding 写入 target IR；TileLang 的作者显式写
tile loop/materialized buffer，pipeline/materializer 只消费已有 region。这里复用的是这条
职责边界，不复制 thread/warp 或 GPU shared-memory 对象。

### 5. 资源合同

新增 live state 是 `[MR,NR]` f32 accumulator、同 cohort 的 main/correction i32 partial，
以及 W/Xq panel。MR/NR 由已有 auto/meta 参数绑定；不合法的 carrier 或超过 RVV register
预算必须明确失败，不能退回 row×column 或让 emitter 重建另一条路径。

### 6. 可见预测

- Canonical/Physical IR 中出现 NC/MC/NR/MR Level、panel materialize 和 output replicas；
- NR>1 时一个 activation window服务多个 weight/output consumer；MR>1 时一个 weight window
  服务多个 activation/output consumer；
- prefill 数字应变化，standalone vec-dot 与 MUL_MAT decode 不应变化；
- 若 MR/NR 全部只能取 1，或最终 IR 仍逐输出重建同一供应，则 blocked 作者树没有被 Physical
  compiler 接住，不能用“树已经改了”当作收益。

### 7. 保留条件

必须先通过数值运行，再用合法 meta 取得可见的共享或资源形态。若动态供应计数不变且性能
不升，改动没有保留理由；若只在一台 target 合法，报告具体 carrier/resource closure，不能
恢复 target-specific scalar 作者树。

## 结果

作者树已改为真实 `NC → MC → NR → MR → K-block` traversal，W/Xq panel、
`[MR,NR]` f32 accumulator、IQ1_M main/correction 与两级 reduction 都出现在
Canonical IR 中。双机结果数值误差均为 0。

### 参数筛选与十次正式结果

| target | binding | Weft before | Weft after | source | after/source |
|---|---|---:|---:|---:|---:|
| SG2044 | NC32, MC8, MR2, NR1, LMUL=m4, unroll1 | 2.035345 | 2.254551 | 2.634283 | 85.59% |
| K1/X60 | NC32, MC16, MR4, NR4, LMUL=m2, unroll1 | 0.846676 | 1.562290 | 1.446879 | 107.98% |

K1 的一遍筛选从 MR1×NR1 的 0.833、MR1×NR2 的 0.970、MR2×NR1 的
1.118、MR2×NR2 的 1.291、MR4×NR2 的 1.424，最终到 MR4×NR4 的
1.566 GOP/s；十次正式中位数为 1.562。output cohort 与跨输出 reuse 在这台机器上
形成可见、单调的收益。

SG 的方向不同：MR1×NR2 为 1.644，而 MR2×NR1 为 2.242，说明复制 codebook/weight
一侧的代价大于复用 activation 的收益。NC/MC 九点扫描在 2.234–2.255 GOP/s 之间，
不能解释剩余 14.4%。LMUL=m2 的多输出 binding 超预算；LMUL=m4 下 MR2×NR2 在
`rvv_partial_reduce` 前需要 34/32 vector groups，MR4×NR1 需要 38/32。

MR2×NR2 的峰值由 8-group activation window、8-group indexed weight、8-group
partial repack、4-group partial-reduce result及同 block live state组成。IQ1_S 在同 target、
LMUL=m4、MR2×NR2 下也以 36/32 停在同一类 operand/product overlap，构成第二个输入；
但 IQ1_S 可用 LMUL=m2 将峰值降到 28，而 IQ1_M 的多级 scale/group topology 在该
carrier 下仍不闭合。因此这是共同的联合资源关系，不是已经证明可共用同一 carrier 规则。

更重要的是，IQ1_M SG standalone 本身只有 source 的 41.17%。blocked MR2 已接近把这条
底座翻倍，因而 85.59% 与该上限一致。donor 把 scale 组合进 i32 lane partial 后只做两次
最终 reduction；当前作者图严格先完成 `contract(entry,payload)`，再乘 scale、再 reduce
`scale_part,group`，SG/K1 分别形成 16/8 次 vector reduction。继续得到 donor 形态会改变
scale/reduction 结合位置和 widening 边界，已到必须由作者确认的数值程序边界，不能由
Physical pass 暗改。

### 保留判断

该作者树与 K1 默认 binding 保留：它把 K1 prefill 从 58.52% 推到 107.98%，并使 SG
提高 10.8%。SG 没有被写成 target-specific scalar tree；同一 canonical blocked program
在两台 target 上仅选择不同的 physical/meta binding。decode 另有明确的
`production_mul_mat_iq1_m_decode` 作者入口，继续逐输出调用共享 vec-dot；双机一遍回归为
SG 2.036、K1 0.846 GOP/s，与改动前同档且数值误差为 0。

两个 prefill Physical IR 均可由 `weft-opt` 独立 parse/verify，安全运行 canonicalizer、
CSE、layout canonicalization、Share 两次和 final verifier；整套第二次运行的文本 diff
双机均为 0 行。
