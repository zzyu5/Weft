# 模块 G — Artifact / 发射

## 职责

把 canonical Weft Kernel IR + selected execution 生成的瞬态 owner IR，机械
降低为 C/C++/intrinsics/object/header 等最终 artifact。

## 与旧架构的关系

旧架构的 fully-legalized gate、object/header packaging 和 compiler/sysroot
接线可以抽取复用。依赖 old Exec/Variant/route string 的 artifact graph 不保留；
新 emitter 只接受 canonical Weft Kernel IR、selected execution 和 owner-local
静态表。

## 当前落地

新主干已有两条结构驱动、成熟度不同的发射面。RVV emitter 只读取 canonical
`weft_kernel`、selected `weft_execution`/`weft_rvv_execution` 与静态 intrinsic
spelling；相同入口可以继续交给 Clang packaging 生成 RISC-V relocatable object。
`weft-compile` 的正式边界是 `--emit=selected-mlir|source|object`：`source`
按 selected owner 路由，`object` 当前只支持 RVV。
selected RVV object packager 的 `-march` 直接来自 plan/CLI target，不再以固定
`rv64gcv_zvfh` 静默覆盖本次选择的 target identity。
本轮由该入口生成 RVV 源码，再用临时 target harness
完成了 RMSNorm、rank-2 affine、rank-2 rowwise/columnwise sum、rank-1 vector dot 和
单 scope rank-2 f32 contract、blocked GEMM 的真实目标运行。vector dot 的 `count=1027`
跨多个动态 `vl` strip，三个 task 对照标量参考得到 `max_abs=0`；
rank-2 contract 以 `tasks=2, M=5, N=37, K=29` 运行，得到
`max_abs=0, max_rel=0`；blocked GEMM 以
`BM=BN=BK=32, tasks=2×3, M=37, N=70, K=45` 运行，同样得到
`max_abs=0, max_rel=0`。独立 meta binding 又以
`BM=8, BN=32, BK=16, tasks=3×3, M=19, N=70, K=45` 覆盖三个方向的尾部，结果仍为
`max_abs=0, max_rel=0`。普通 splat-per-use layout conversion 以
`tasks=2, rows=19, cols=70, value=1.25` 分别运行 `rows×cols-VLA` 与
`cols×rows-VLA` 两个 group，得到 `max_abs=0`。这些数值结果来自本轮外部临时
harness，尚未作为仓库内 runtime driver 保留；仓库中的 Python example 只负责输出
canonical MLIR。

columnwise sum 选择 axis 0 为 RVV lane domain，并以 runtime element stride 发出 masked
`vlse32`；`tasks=2, rows=19, cols=70, stride=77` 的真实运行得到 `max_abs=0`。
`examples/repro_mixed_axis_sums.py` 又把 axis-1 与 axis-0 两个独立 reduction group 放进
同一个 kernel/function；selected MLIR round-trip 与 RVV object 已完成。共享目标机当时
已有硬件扫描进程，本轮没有为这个组合追加真机数值结论。
`examples/repro_dual_axis_load_sum.py` 进一步让同一个 masked load 的第二个 reduction use
在 axis-0 target group 中重新 materialize；生成 source 对 native axis-1 use 发出 `vle32`，
对 converted axis-0 use 从同一 canonical pointer graph 发出 `vlse32`。selected
round-trip 与 object 已完成；它不是 register shuffle，且同样没有新增真机数值结论。

新路径还完成了三个结构增量的 artifact 闭合：`repro_rowwise_max.py` 发出真实
`vfredmax`；`repro_softmax.py` 保留三段 canonical loop 并发出两段 selected
`exp_poly_v1`、max/sum reduction 与 normalize store；`repro_rank3_affine.py` 从
canonical unit-stride use 选择 `order=[1,2,0], vector_axes=[1]`，机械生成两层 serial
loop 加一层 VLA strip。三者均已
完成 canonical→selected round-trip→source→RISC-V object。exp 序列复用了旧低层中
已有数值证据的多项式/exceptional-range 算法，但重新由 canonical unary 与 owner-local
strategy 驱动，不读取旧 elementwise/softmax brick 或 route。共享 RVV 主机预检仍有
两个长期高 CPU 的 sysfs 扫描进程，因此这里不追加新主干 runtime/performance 结论。

第二条是 source-only 的 IME sibling。它只读取 canonical `weft_kernel`、shared
`weft_execution`、owner-local `weft_ime_execution.contract_config` 与静态 IME spelling
表；不调用旧 `IMEBackendEmissionDriver`，也不读取旧 Exec/Variant/route。当前唯一
verified physical envelope 是 signed `si8×si8→si32`、VLEN=256、`4×4×8` vmadot
fragment；canonical selected logical M/N/K 可以是任意正整数，且一个 kernel 可含多个
互不重叠的 contract site。“任意”只描述这个 flat、direct、zero-fill/zero-init
envelope 内的逻辑 extent，不包含嵌套控制流或 IME/RVV 混合 owner。
emitter 不重建固定 GEMM 地址模板，而是逐值标量化 canonical
`task_id/arange/expand_dims/binary/compare/ptr_add`，据此执行逐 tile/tail-safe scratch
pack、register-resident K-fragment `vmadot` loop 和 masked scatter。逻辑 tile 数来自
canonical extent，物理步长/fragment storage 来自 selected config，不再由 emitter
另写一份固定 4/4/8 kernel 模板。生成的 selected MLIR 可 round-trip，C++ 已通过
RISC-V clang syntax check；本机没有 SpacemiT assembler/IME target，所以 `object` 对
IME 明确失败，当前没有新主干真机数值或性能结论。

旧 EmitC lowering 及其 route 仍是仓库中可注册的资产，但不是两条
`weft-rvv-selected-execution-*` route 的中间真理或必经路径。后续若抽取其中通用
能力，只能作为机械 emission 组件复用，不能重新引入旧 Exec/Variant 权威。

## 尚未落地

- public header 与完整 runtime/launch packaging；
- 把目标部署、launch 与数值 harness 串成正式 runtime driver；
- Scalar owner 发射面；
- IME object/toolchain packaging、真实目标运行、更多 IME fragment/instruction family 与
  RVV/Scalar cleanup；
- 多 vector axis、strided store/indexed gather layout、rank>2/serial-axis/multi-axis
  reduction 与 contract、
  多状态/多阶段 GEMM、load/register/owner 的通用 layout conversion realization 和
  更一般的矩阵扩展 artifact；
- 对 owner-local 指令选择之外更多可复用机械 lowering 的裁定。
