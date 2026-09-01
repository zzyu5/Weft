# Vec-dot shaped tree 与 reduction carrier 收口

## 1. 本轮范围与判据

本轮只收口上一轮留下的三类工作：IQ1_S / IQ2_XXS 的 shaped 作者树、multi-axis
widening contraction 的共同 reduction carrier，以及默认 CMake 构建图。没有新增格式、没有开启
新的性能聚类，也没有把静态生成当作真机完成。

动手前按 `doc/compiler/optimization-principles.md` 第九节核对：

1. 作者值图中 IQ1_S 与 IQ2_XXS 的 `group` 原为 `L.subs` 下的逐组标量 carry；目标值图应显式保留
   `group × entry × payload`，逻辑值集合变化归作者树。
2. 两个 widening operands 在同一 contraction 上得到不相容的 storage proposals；最终 contraction
   需要一个共同 issue/reduction carrier，归 `PropagateRISCVLayouts`。
3. 预期静态变化是 shaped contraction 能进入现有 partial planner，而不是 emitter 新增闭包 matcher。
4. Q6_K 与 IQ2_S 是负向输入：前者的 storage geometry 不能被 scale/index side value 的布局覆盖，
   后者 planner 冻结的 carrier 不能由 materializer 重新投影成另一种类型。
5. 若 48 个 target-format 入口的静态成功数不恢复，作者树和 owner 修复均不成立。
6. 真机验收同时看 standalone vec-dot 与 MUL_MAT decode；两者不同步时才归为 GEMV 外层问题。
7. 若生成/运行计数不变，本轮唯一仍有价值的改动是 CMake build graph，因为它消除“旧 executable
   测新库”的实验有效性缺口；其余编译器改动必须改变可达 physical program 才保留。

参考机制不是事后类比。Triton 的 `AccelerateMatmul` 先从 result 建 MMA encoding，再派生两个
dot operands 的 encoding 与 conversion；`ReduceOpToLLVM` 只消费已确定的 register/lane bases。
TileLang 的 reducer planner 从 update-site reduction axes 投影 partial layout，冻结 plan 后要求其他
update site structural equality，materializer 只执行该 plan。Weft 不能复制 warp/shared-memory 对象，
但采用同一 owner 分工：layout pass 决定共同 carrier，partial planner 冻结 issue 类型，materializer
只实例化。

## 2. 实际改动

### 2.1 作者树

- IQ1_S：把 8 个 group 从标量 `L.subs` carry 改为显式 `group × entry × payload`，main 与
  correction 都在 canonical IR 中显式按 `group` reduce。
- IQ2_XXS：同样把 8 个 group 显式化，grid/sign lookup、activation 与 scale 保持 shaped，最终
  reduce group。
- IQ1_M：把 grid ABI 统一为 `I8X8[2048,8]`；主分支显式表达
  `group × scale_part × entry × payload`，并保留原 donor 的 scale 结合位置。correction 分支仍是
  作者写出的有序标量循环；本轮没有伪装成已经完成的 shaped 路径。

### 2.2 Physical IR 与 pass

- `PropagateRISCVLayouts` 现在要求同一个 widening contraction 的 operands 共用 reduction carrier。
  两侧 storage proposal 一致时原样保留；不一致但 reduction axes/extents 相同时，按作者声明轴序，
  以最后一条 reduction axis 为 primary lane，其余 reduction axes coalesce。storage order 只负责
  供应该 carrier，不再分别拥有 contraction layout。
- storage owner 被收窄到“精确 shaped value 经 shape-preserving use 供应 widening contraction”；
  scale/index side values 不能仅因有仿射 storage order 就覆盖消费关系。
- `MaterializeRISCVPartialAccumulators` 可以为 natural `RVVReplicaStorageLoadOp` 实例化一个 frozen
  issue window。适用条件是 unit access、相同 reduction axis、record rank 0、单 physical part；
  不满足直接失败，没有 scalar fallback。
- 单 physical part 按 dialect verifier 的真实合同计算为
  `product(time_factors) × product(replica_factors)`，不再误用 LMUL/register groups。
- 设计合同同步到 `doc/compiler/passes.md`；terminal emitter 未改。

### 2.3 构建图

`add_mlir_tool()` 在当前 LLVM 配置下默认产生 `EXCLUDE_FROM_ALL` target。`tools/CMakeLists.txt`
现在显式把 `weft-opt` 与 `weft-compile` 加回默认 `all`。实际触碰 target 源文件后执行普通
`cmake --build build -j2`，日志自然出现静态库和 `tools/weft-compile/weft-compile` 的重新链接，
不需要手动指定 target 或额外 build 脚本。

## 3. 静态主链

最终重新生成目录为 `/tmp/weft-static-vec-dot.DycyG4`。24 个格式在 SG2044/VLEN128 与
K1/VLEN256 上共 48 个入口：

- frontend MLIR：48/48；
- intrinsic C：48/48，且 48 个输出均非空；
- frontend/compile stderr 非空：0；
- 输出中的 `TODO`、`unsupported`、`fallback`、`NotImplemented`：0。

本轮开始时真实结果为 46/48，仅 IQ1_M 双机因作者树/ABI 不完整失败；Q6_K 与 IQ2_S 的 owner
问题已经在收窄 storage owner 后恢复。最终 48/48 是生成覆盖，不代表 48 个真机运行覆盖。

## 4. 受影响入口的双机结果

下表均为 10 repetitions，Clang 18.1.8，`-O3 -ffp-contract=fast`，数值均在容差内。
“source 比”只比较 standalone vec-dot 的同 shape source microkernel。

| 格式 | SG standalone | SG decode | SG/source | K1 standalone | K1 decode | K1/source |
|---|---:|---:|---:|---:|---:|---:|
| Q6_K | 5.527 | 4.630 | 112.9% | 3.135 | 3.115 | 131.5% |
| IQ1_S | 2.295 | 2.310 | 81.8% | 1.272 | 1.275 | 46.8% |
| IQ1_M | 0.384 | 0.387 | 8.4% | 0.228 | 0.240 | 15.9% |
| IQ2_S | 2.264 | 2.285 | 94.6% | 1.428 | 1.432 | 97.7% |
| IQ2_XXS | 2.835 | 2.831 | 70.0% | 1.456 | 1.468 | 86.4% |

单位为 GOP/s。除 SG Q6_K 外，standalone/decode 差异均在 5.2% 内；这四个格式继续支持
“vec-dot 是 decode 的性能底座”。SG Q6_K 的 decode 比 standalone 低 16.2%，是仍待单独定位的
配对异常，不能并入本轮 carrier 结论。

IQ2_S 的当前 SG/K1 standalone 为 2.264/1.428；旧正式 CSV 中的 3.232/1.724 不能在 clean
`a1df62757` 上复现。用同一当前 runner 编译该 clean revision 得到 2.256/1.424，与本轮相差不足
0.5%。因此这不是共同 carrier 引入的退化，而是旧 CSV 混入了另一编译快照。当前 CSV 已改为
本轮可复现值。

## 5. 回归批次

以下既有入口双机 standalone/decode 均运行成功且数值在容差内。表中为 standalone GOP/s 与
同 shape source 比；“回归通过”只表示本轮改动没有破坏主链或数值，不等于每条都超过 source。

| 格式 | SG GOP/s | SG/source | K1 GOP/s | K1/source |
|---|---:|---:|---:|---:|
| Q1_0 | 4.536 | 200.3% | 3.778 | 100.1% |
| Q2_K | 5.540 | 59.5% | 1.354 | 55.4% |
| Q4_0 | 7.367 | 298.0% | 2.556 | 114.9% |
| Q4_1 | 5.486 | 216.6% | 2.244 | 101.0% |
| Q5_1 | 4.782 | 81.9% | 2.061 | 119.4% |
| Q5_K | 4.046 | 310.7% | 1.588 | 127.9% |
| IQ4_NL | 6.758 | 97.4% | 2.417 | 85.4% |
| IQ4_XS | 6.321 | 234.4% | 2.195 | 128.1% |
| TQ2_0 | 9.895 | 148.9% | 6.240 | 126.7% |

## 6. 被排除的方向与未闭合边界

- 过宽 storage anchor 曾提高 IQ1_S 的中间结果，但同时使 Q6_K 与 IQ2_S 退化；它把 storage
  proposal 错当成 contraction carrier，已删除，没有作为性能捷径保留。
- IQ1_M correction 的 shaped 化尝试暴露了另一组 conversion/contract 合同缺口。本轮没有为它
  新增 matcher，也没有让 emitter 重建；当前 scalar correction 是作者程序本身，因此 IQ1_M
  的 8.4%/15.9% 是真实边界。
- IQ1_S 与 IQ2_XXS 虽已形成 shaped group axis，但仍低于 source；已知剩余现象是 qh/index
  的 issue-local 重构、子窗口重复供应与 bsum/indexed gather 形态，属于后续 P2/P3，而不是本轮
  继续放宽 carrier 的理由。
- SG Q6_K standalone/decode 不同步；本轮没有用重跑均值掩盖。
- 本轮没有验证 IQ1_S 可选的 P2/P3 优化，也没有铺新格式。完成标志是静态 48/48、指定入口
  双机运行、既有批次不发生主链/数值回归，以及可提交的单一工作树；不是 vec-dot 全列过 source。
