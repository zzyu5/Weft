# Q2_K vec-dot 与 row-dequant 横向收口

本报告记录从 `e8568df21` 之后到本轮提交的事实。设计权威仍是 `doc/`；source 数字来自
`report/baseline/ggml-riscv-kernel-performance.csv`，当前 Weft 数字写入
`report/weft-kernel-performance.csv`。所有列出的真机结果均使用 Clang 18、`-O3
-ffp-contract=fast`、相同 ISA/ABI 与 10 repetitions；数值结果均为容差内，表中本轮重跑项的
最大绝对误差和最大相对误差均为 0。

## 1. 第九节工作账得到的结论

### 1.1 Q2_K 单输出 contraction

Q2_K vec-dot 的作者程序没有 GEMM 的 output cohort、local panel 或多 accumulator，这些不能由
编译器替作者加入。因此本轮只比较单输出程序内部的物理工作：encoded q2 供应、per-sub scale
供应、Q8 window、widened partial 与最终 reduction。

动手前的静态账显示，原程序把同一 layered storage relation 拆成普通 extract，并把 scale
供应留在子 Level 内；结果是 storage window、partial lifetime 与 scale lifetime 分属三段 SSA，
不能组成 donor 中的 layered partial program。对应改动分两个提交完成：

- `319b9bf8c`：`ShareRISCVLayeredWindows` 形成 typed layered root；partial planner 从 layout、
  storage geometry、reduction relation 与资源预算冻结 topology，materializer 只实例化。
- `0b17ea624`：将可证明相同的 cross-Level scale supply 放到一个显式 SSA lifetime 中；该
  lifetime 是 use-def 事实，不由 emitter 缓存字符串。

这两项没有改变 canonical value、reduction 结合位置或 Level 归属。最终结果：

| target | 入口 | Weft | source | Weft/source |
|---|---|---:|---:|---:|
| SG2044 | standalone vec-dot | 9.510021 GOP/s | 9.307765 | 102.17% |
| SG2044 | MUL_MAT decode | 8.953468 GOP/s | 8.815042 | 101.57% |
| K1/X60 | standalone vec-dot | 3.494847 GOP/s | 2.445995 | 142.88% |
| K1/X60 | MUL_MAT decode | 3.464766 GOP/s | 2.403987 | 144.13% |

standalone 与 decode 同步，说明收益位于共享 vec-dot 底座，不在 GEMV 外壳。两台机器使用同一
canonical 作者树；VLEN 只改变 physical layout、LMUL/vl 与 issue/lane 分解。

### 1.2 pure decode 的工作账

row-dequant 没有 contraction，因而 partial topology、product carrier 与 accumulator combine
都不能解释其性能。本轮从 memory form 和 carrier 开始数，得到三类实际关系：

1. codebook 的一个 entry 是标量选择，而 entry 内 payload 是连续 shaped axis。旧路径按每个
   payload consumer 做 indexed gather；新 `rvv_indexed_entry_load` 对标量 entry 发一次 unit
   payload load，对真正的 vector entry 才发 gather。
2. logical-u1 的 grouped/layered field 在嵌套 sub-Level 中需要 byte-aligned typed mask window。
   planner 先按 part bit offset 选择合法 load carrier，再显式转换到 consumer layout。
3. storage load 的 reduction axis 与 issue-time projection axis 可以正交。若所有窗口只是一个
   affine translation，materializer移动 scalar base 并保留同一个 typed load plan，不重新猜
   memory form。

第一类在两个独立格式上有真机正例：

| 格式 | target | 改前 | 当前 | 倍数 |
|---|---|---:|---:|---:|
| IQ1_S row | SG2044 | 149.014665 | 726.941598 MElements/s | 4.88× |
| IQ1_S row | K1/X60 | 48.395717 | 451.153992 MElements/s | 9.32× |
| IQ2_XS row | SG2044 | 500.879640 | 718.090231 MElements/s | 1.43× |
| IQ2_XS row | K1/X60 | 175.627063 | 252.037940 MElements/s | 1.44× |

第二类使 IQ3_S 从 SG2044 `125.277209` 提升到 `258.542511 MElements/s`，K1 从
`54.198465` 提升到 `156.071954 MElements/s`。它仍未超过 source，因此这里只证明 typed
bitmask window 减少了工作，不把它写成 pure-decode 已收口。

第三类在 SG2044 Q3_K、SG2044 Q6_K、K1 Q3_K 三个独立 relation 上闭合了同一种正交
projection。它没有减少动态 load 数，因此不计为性能优化；价值是让三份原本因 carrier
不闭合而失败的 physical program 进入同一主链。

## 2. 24 个 row-dequant 的双机账

单元格为 `Weft/source（比值）`，吞吐单位是 MElements/s。

| format | SG2044 | K1/X60 |
|---|---:|---:|
| Q1_0 | 239.192 / 450.553（0.531×） | 586.105 / 148.276（3.953×） |
| Q4_0 | 737.294 / 911.391（0.809×） | 433.078 / 170.684（2.537×） |
| Q4_1 | 737.177 / 889.906（0.828×） | 277.270 / 149.394（1.856×） |
| Q5_0 | 736.697 / 885.910（0.832×） | 273.503 / 160.035（1.709×） |
| Q5_1 | 735.636 / 872.782（0.843×） | 291.082 / 147.062（1.979×） |
| Q8_0 | 727.133 / 724.673（1.003×） | 371.844 / 183.901（2.022×） |
| Q2_K | 586.813 / 210.802（2.784×） | 157.533 / 134.414（1.172×） |
| Q3_K | 517.296 / 262.184（1.973×） | 280.449 / 113.985（2.460×） |
| Q4_K | 466.534 / 679.322（0.687×） | 313.627 / 166.886（1.879×） |
| Q5_K | 327.978 / 441.095（0.744×） | 238.742 / 112.231（2.127×） |
| Q6_K | 703.300 / 296.390（2.373×） | 145.301 / 101.620（1.430×） |
| IQ1_S | 726.942 / 686.765（1.059×） | 451.154 / 130.282（3.463×） |
| IQ1_M | 71.665 / 700.561（0.102×） | 29.490 / 122.728（0.240×） |
| IQ2_S | 108.565 / 418.088（0.260×） | 40.870 / 179.633（0.228×） |
| IQ2_XS | 718.090 / 519.730（1.382×） | 252.038 / 164.411（1.533×） |
| IQ2_XXS | 430.027 / 385.071（1.117×） | 208.266 / 186.441（1.117×） |
| IQ3_S | 258.543 / 407.880（0.634×） | 156.072 / 179.614（0.869×） |
| IQ3_XXS | 109.572 / 312.528（0.351×） | 35.469 / 174.039（0.204×） |
| IQ4_NL | 747.120 / 161.431（4.628×） | 442.013 / 155.008（2.852×） |
| IQ4_XS | 722.517 / 160.073（4.514×） | 345.187 / 158.062（2.184×） |
| TQ1_0 | 297.310 / 449.957（0.661×） | 67.500 / 110.512（0.611×） |
| TQ2_0 | 734.363 / 732.826（1.002×） | 603.400 / 173.940（3.469×） |
| MXFP4 | 730.806 / 169.239（4.318×） | 261.926 / 167.090（1.568×） |
| NVFP4 | 696.188 / 193.004（3.607×） | 246.296 / 100.879（2.441×） |

分布为：`>= source` 31 条，`90%–100%` 0 条，`70%–90%` 6 条，`50%–70%`
5 条，`<50%` 6 条。SG2044 为 `12/0/5/4/3`，K1 为 `19/0/1/1/3`。

这张账给出的聚类不是一个统一 pure-decode 缺口：

- IQ1_M、IQ2_S、IQ3_XXS 的作者程序仍以 Python 循环逐元素展开，没有可供 physical pass
  映射的完整 shaped axis；三者构成 6 条 `<50%` 中的 6 条。这里首先是作者树缺信息。
- IQ3_S 已是 shaped program，剩余差距在 P3：当前一个 indexed lookup 跨两个连续的 4-byte
  codebook entry，donor 是两个 unit entry load。它是明确的 compiler memory-form 缺口。
- Q1_0 在 K1 超过 source 3.95×、SG2044 只有 0.53×；同一作者树的反向结果把问题限定为
  VLEN128 下 byte-aligned bitmask carrier/memory form，而不是再写一棵作者树。
- Q4_K/SG2044、Q5_K/SG2044 与 TQ1_0 双机仍低；当前数据不足以把它们合并为同一个机制，
  本轮没有据此增加 pass。

## 3. IR 与 pass 的机械验收

当前 24 个 vec-dot 和 24 个 row-dequant 在两个 target 上形成 96 份 RISC-V Physical IR。
每份均执行：

```text
weft-compile --emit=riscv-ir
weft-opt --canonicalize --cse
         --weft-riscv-canonicalize-layouts
         --weft-riscv-share-layered-windows
         --weft-riscv-share-layered-windows
         --weft-riscv-verify-final --verify-each --verify-roundtrip
```

结果是 96/96 可 parse、可 verify、可运行标准 canonicalizer/CSE；同一管线再次运行后
96/96 文本 diff 为 0。该检查还覆盖了 Share pass 的第二次运行。

为避免一个宽 memory pass 在程序已部分物化后重新接管普通 extract，pipeline 将新增的
nested-domain decode 限定为独立的 nested-only planning 入口。此前直接重跑完整
`PlanRISCVMemory` 的静态结果只有 43/48 vec-dot；收窄后恢复 48/48，并与 48/48
row-dequant 合并为上述 96/96。

## 4. 与 Triton / TileLang 的机制对照

- Triton `Coalesce.cpp:77-119` 与 `CoalesceUtils.cpp:17-94` 先从 AxisInfo 的
  contiguity/divisibility 得到访问约束，再把选定 encoding 写回 IR；
  `LoadStoreOpToLLVM.cpp:81-224` 只消费该布局选择形成 load/store 宽度。本轮同样把
  storage geometry 分析、typed load/window 选择与 terminal intrinsic 拼写分开，但不能直接
  采用 Triton 的 thread/warp encoding，因为 Weft 的完整 logical value 没有预先存在的
  thread owner。
- TileLang `loop_vectorize.cc:780-819,1205-1228` 先建立 vectorization plan，再机械改写
  load/store；无法证明 invariant/continuous 的访问才保持标量。本轮 scalar-entry + contiguous
  payload 的边界与此相同：entry identity 决定基址，payload axis 决定 unit vector load。
- Triton 的 layout canonicalization 可以重物化纯 producer，但它不能替代 storage legality。
  本轮允许 multi-use iota 在 layout pass 中重物化，只用于消除等价写法造成的 terminal
  conversion；storage plan 仍必须由 field geometry 独立证明。

## 5. 负结果

下列代码均未保留：

| 实验 | 结果 | 结论 |
|---|---|---|
| IQ2_XS half16 作者树 | SG2044 `309`、K1 `155 MElements/s`，均低于 group-entry tree | 更早切碎 payload 增加 issue 工作，恢复完整 shaped tree |
| IQ1_S 在 nested Level 内 materialize `w.qh[group]` | 第 32 个元素开始数值错误 | materialize 的 birth 不等于运行时 group cache，撤销 |
| IQ3_S 4-element sign window | half-byte 起点被 verifier 拒绝 | 不是合法 byte window；改用 8-element pair geometry |
| 过宽 storage anchor | IQ1_S 一度到 SG2044 93.5%，同时 Q6_K/IQ2_S 退化 | storage proposal 不能接管 contraction carrier，撤销 |
| materialization 后重跑完整 memory planner | vec-dot 静态覆盖降至 43/48 | 改为只处理尚未物化的 nested-domain relation |
| 将 rematerialization `visited` 改为 recursion stack | IQ1_S/SG2044 从约 `725` 降至 `265 MElements/s` | 形式上允许共享 DAG，却改变结构选择且无工作量收益；按第九节准入规则撤销 |

## 6. 当前未闭合事实

- 三棵逐元素作者树使 6 条 row-dequant 低于 source 一半；在作者显式表达轴以前，编译器不应
  从展开后的 SSA 猜回 shaped value。
- IQ3_S、Q1_0/SG2044、Q4_K/SG2044、Q5_K/SG2044 和 TQ1_0 尚有具体或待进一步计数的
  pure-decode 缺口；因此不能把 row-dequant 宣称为已整体收口。
- 本轮 typed nested bitmask window、scalar-entry payload 与 orthogonal issue projection 都有
  第二个独立输入或机械覆盖证据；但它们只覆盖各自的几何关系，不构成一个通用 decode
  macro。
- 本轮没有验证 software pipeline；row-dequant 的结果不应被解释为 P5 证据。
