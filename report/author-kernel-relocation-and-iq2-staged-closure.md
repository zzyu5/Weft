# 作者 kernel 搬迁、Q4/Q5 correction 与 IQ2 staged 收口

## 1. 本轮范围

本轮只处理三件事：

1. 把用 Weft DSL 写成的 kernel 库从 `python/weft/std/` 搬到
   `examples/kernels/weft_kernels/`，让编译器包与作者程序在目录和 import 关系上分离；
2. 将 Q4_K/Q5_K 已存在的 8 个 min-correction logical coordinates 写成 shaped value，
   并让对应 joined regular-repeat storage relation 成为 typed memory form；
3. 让 IQ2_S/IQ2_XS staged prefill 与 IQ2_XXS 一样形成完整 product carrier 和
   partial set，而不是在 materializer 中重新切成 issue slices。

作者侧可以重写同一数学程序的表达方式。这里的 shaped correction 没有改变 scale 与
reduction 的结合位置、整数 widening、溢出边界或 ABI；它只把作者本来知道的 8 元素轴
从 Python 展开改成 canonical shaped value。编译器侧只决定该值的 physical carrier、
memory form 和 partial topology。

## 2. 动手前的工作账与参考机制

### 2.1 Q4_K/Q5_K correction

源程序有 8 个不同的 `m[g] * (bsum[2g] + bsum[2g+1])`，因此合法合并上限不是
read-CSE，而是将这 8 个 coordinates 保留成一个 shaped reduction。改动后的作者程序见
`examples/kernels/weft_kernels/vec_dot.py:322-373`。

对应的 storage edge 是两个 joined source parts 上的 regular-repeat access。Triton 也把
gather 的 source、indices、axis、result shape/encoding 放在 typed `GatherOp` 中，并在
verifier 检查几何，再由 lowering 消费既定 layout：

- `ref/triton/include/triton/Dialect/Triton/IR/TritonOps.td:949-979`
- `ref/triton/lib/Dialect/Triton/IR/Ops.cpp:1550-1571`
- `ref/triton/lib/Conversion/TritonGPUToLLVM/GatherOpToLLVM.cpp:109-139,193-245`

Weft 不能直接采用 Triton 的 warp-local shuffle/shared-memory 两种 realization，因为 RVV
没有 warp owner；这里复用的是“typed geometry 先闭合，terminal lowering 只消费”的机制。
`PlanRISCVMemory` 依据 joined field、repeat、group size 与 target legality 选择
`rvv.regular-repeat-joined-gather.pow2/div`，dialect verifier 检查 byte geometry，
Intrinsic-C 终端只拼写选定 leaf。

### 2.2 IQ2 staged carrier

改动前的静态事实是：

| 格式 | partial set | issue_slice | vslidedown |
|---|---:|---:|---:|
| IQ2_S | 0 | 16 | 40 |
| IQ2_XS | 0 | 16 | 10 |
| IQ2_XXS | 2 | 0 | 2 |

S/XS 的作者树已经给出 group×entry×payload axes，问题发生在 physical planning：
planner 没有冻结 surviving output axes、output replicas、source part、lane offset 与 scale
replica；materializer 因而重新投影 operand，把完整 product 切碎。

TileLang reducer 的做法是 planner 先从 reduction axes 冻结 partial storage 与 combine
steps，materializer 只实例化既定 plan：

- `ref/tilelang/src/transform/reducer_plan_materialize.cc:841-917,1081-1117`
- `ref/tilelang/src/transform/reducer_plan_materialize.cc:1187-1200,1267-1306`
- `ref/tilelang/src/backend/common/op/finalize_reducer.h:42-60,92-125`

Weft 不能照搬 participant/thread collective，但采用同一责任分界：
`NestedPartialPlanAttr` 现在冻结 output axes/replicas、source parts、lane offsets、
scale replicas、carrier types 与 resource groups；materializer 只验证并实例化
`RVVPartialSetOp → Repack → Reduce → ScaleCombine → Finalize`。

## 3. 作者 kernel 搬迁

提交 `14d2bd9dd` 将原 `python/weft/std/` 的 12 个文件原样搬到
`examples/kernels/weft_kernels/`。14 个 DSL examples 改从 `weft_kernels` 导入，
四个 runner 的 `PYTHONPATH` 显式包含 `python` 与 `examples/kernels`。

核验结果：

- `weft_kernels` 及 11 个子模块均能导入；
- 14 个 DSL example 均能加载；
- compiler/runtime 路径中不存在 `weft.std`、`python/weft/std` 或旧 import；
- 旧路径字符串只存在于历史报告，不参与编译或运行。

搬迁后 kernel 仍经唯一主链
`DSL source → Canonical Kernel IR → RISC-V Physical IR → intrinsic C` 运行；
编译器没有内置作者 kernel 库。

## 4. Physical IR 变化

### 4.1 IQ2

当前六个 staged final RISC-V IR 的静态计数：

| target | 格式 | partial set | unit entry window | issue_slice | vslidedown | vector groups peak |
|---|---|---:|---:|---:|---:|---:|
| SG2044 | IQ2_S | 8 | 1 | 0 | 0 | 20 |
| SG2044 | IQ2_XS | 4 | 1 | 0 | 0 | 21 |
| SG2044 | IQ2_XXS | 2 | 0 | 0 | 0 | 16 |
| K1 | IQ2_S | 24 | 1 | 0 | 0 | 22 |
| K1 | IQ2_XS | 8 | 1 | 0 | 0 | 20 |
| K1 | IQ2_XXS | 2 | 0 | 0 | 0 | 16 |

S/XS 都从同一个 typed `2×2 entry → payload 8` relation 形成一次
`RVVUnitEntryWindowLoadOp`；XXS 的四个 q fields 已是显式独立 source parts，
因此直接进入 indexed-entry loads，不需要 unit entry window。这给出了第二个格式、
两个 target 的重复输入，不是 IQ2_S 单点规则。

### 4.2 Q4_K

当前 standalone final IR 在两台机器上均形成 1 个 joined regular-repeat gather、1 组
partial collect/reduce/scale/finalize，并且没有 slide。SG 仍有 8 个 layered loads、
16 个 layered decode 和末端 scalar correction chain；K1 为 4 个 loads、8 个 decode。
这解释了为什么 shaped 作者树带来明显收益，但没有复现历史 10.478。

## 5. 双机结果

全部为 10 repetitions；baseline 与 Weft 使用相同 Clang/flags、相同合法量化记录输入，
数值均在容差内。

### 5.1 Q4_K/Q5_K standalone 与 MUL_MAT decode

| target | entry | 改前 GOP/s | 改后 GOP/s | source GOP/s | 改后/source |
|---|---|---:|---:|---:|---:|
| SG2044 | Q4_K standalone | 5.694 | 8.094 | 10.143 | 0.798× |
| SG2044 | Q4_K decode | 5.910 | 8.251 | 9.619 | 0.858× |
| K1 | Q4_K standalone | 2.357 | 3.587 | 2.505 | 1.432× |
| K1 | Q4_K decode | 2.354 | 3.561 | 2.462 | 1.446× |
| SG2044 | Q5_K standalone | 4.072 | 4.342 | 1.157 | 3.753× |
| SG2044 | Q5_K decode | 4.070 | 4.366 | 1.155 | 3.781× |
| K1 | Q5_K standalone | 1.586 | 1.701 | 1.255 | 1.356× |
| K1 | Q5_K decode | 1.588 | 1.703 | 1.246 | 1.367× |

standalone 与 decode 同步变化，说明收益位于共享 vec-dot contraction，不在 GEMV wrapper。
Q5_K 双机都超过 source；Q4_K 在 K1 超过，SG 仍差 14.2%–20.2%。

### 5.2 IQ2 staged prefill

| target | 格式 | 改前 GOP/s | 改后 GOP/s | source GOP/s | 改后/source |
|---|---|---:|---:|---:|---:|
| SG2044 | IQ2_S | 0.528 | 5.740 | 2.506 | 2.290× |
| SG2044 | IQ2_XS | 0.355 | 5.389 | 4.810 | 1.120× |
| SG2044 | IQ2_XXS | 3.981 | 3.979 | 4.112 | 0.968× |
| K1 | IQ2_S | 0.772 | 2.170 | 1.469 | 1.477× |
| K1 | IQ2_XS | 0.427 | 2.149 | 1.796 | 1.196× |
| K1 | IQ2_XXS | 1.482 | 1.483 | 1.705 | 0.870× |

S/XS 的收益与静态消除 `issue_slice/vslidedown` 同时出现；XXS 的 IR 与吞吐基本不变，
是回归对照。S/XS 已双机超过 source，XXS 未因本轮机制退化，但仍低于 source。

### 5.3 同步回归

Q2_K standalone/decode 当前为 SG `8.122/8.173`、K1 `3.447/3.408` GOP/s；
IQ3_XXS standalone 为 SG `2.396`、K1 `1.222` GOP/s。数值均正确。
这些行按当前源码重测并覆盖 CSV，没有沿用旧快照。

固定的已过线 standalone 回归组也全部运行 10 次并保持数值正确：
SG Q1_0/Q3_K/TQ2_0/IQ4_XS 为
`11.648/7.560/9.959/6.655` GOP/s，K1 为
`4.537/2.868/6.234/2.095` GOP/s。没有入口因新的 IQ2
carrier/materializer 合同变成编译失败或跌回 source 以下。

## 6. 被证伪或撤销的方向

### 6.1 历史 Q4_K 10.478 不是 shaped DSL 单项收益

历史 `10.478084 GOP/s` 来自对临时 generated C 的整段宏改写，而不是只把 std 中
8 次 Python scalar correction 改成 shaped value。该临时补丁同时把 scalar loop 改成：

- 8 个显式 product producers；
- 8 次 widening reduction；
- 两组各 4 次的 vector scaled MAC；
- 两次 vector-to-scalar extraction 后汇合。

因此当前合法主链的 `8.093763` 不能冒充已“恢复”到 `10.478084`。旧报告
`report/riscv-full-account-cluster-closure.md:407-411` 对这项实验的概括过窄；
本轮实测修正了该数据解释。

### 6.2 其它实验

- 把 IQ2 group axis 直接并入完整 lane carrier会使 resource peak 达到
  SG/K1 `84/132` vector groups；资源合同不合法，未保留。
- 一个 exact lane-permutation representation 曾被实现用于诊断，但在当前
  96+6 个 final IR 中消费者计数为 0；按 P0 删除，没有把零覆盖机制留在主干。
- IQ2_XS 初始 `MR=1, NR=2` 只得到 SG/K1 `3.715/1.036` GOP/s；
  在同一 canonical tree 和同一 physical relation 下实测合法 auto 参数后，
  SG 使用 `MR=2, NR=2`，K1 使用 `MR=4, NR=2`，得到表中结果。
- 除 Q4_K 这项被错误归因的 generated-C 实验外，Git 与历史报告中没有找到另一项
  “已经超过 source、只因属于作者树而删除”的可复现收益。最接近的是 IQ2_XXS
  临时 shaped tree，但当时只有 source 的 SG 48.8% / K1 40.4%。

## 7. 机械验收

由当前 DSL 源重新生成，不复用旧 canonical IR：

- 24 formats × row-dequant/vec-dot × 2 targets = 96/96：
  compile、独立 parse/verify、canonicalizer、CSE、Share 连跑两次均通过；
- 第二轮 pass 输出与第一轮逐文件文本 diff 为 0；
- IQ2_S/XS/XXS staged × 2 targets = 6/6 通过同一验收；
- `cmake --build build` 自然重链接 `weft-compile` 与 `weft-opt`。

## 8. 未闭合事实

- Q4_K shaped correction 是有效改动，但 SG standalone/decode 仍只有 source 的
  79.8%/85.8%；当前差距不能再用历史 10.478 证明已经解决。
- IQ2_XXS staged 保持正确且无退化，但仍只有 source 的 SG 96.8% / K1 87.0%。
- `reachesReductionThroughPureOps` 当前保守地沿 memory-effect-free 单结果链判断
  一个 cohort 后续是否仍被 reduction 消费；混合 consumer 图会保守禁用 cohort，
  但不会改写作者程序或造成错误 materialization。
