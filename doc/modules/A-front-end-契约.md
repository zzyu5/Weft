# 模块 A — Weft Kernel DSL 与 Front-end 契约

## 职责

定义一门独立、可手写、可由其他编译器生成的 RISC-V blocked kernel language，
并让 Python eDSL 直接构造 canonical Weft Kernel MLIR。Python 不维护第二套
typed Kernel IR。

Weft Kernel IR 是 weft-compile 的唯一输入算法表示。Weft core 不解析 Intent
IR；Intent→Weft conversion 位于上游 target plugin 或独立 bridge。

## Weft source 的目标语义面

- 一个 kernel entry、指针/标量 ABI 和 compile-time meta-parameters；
- task/grid identity 与 blocked work ownership；
- block tensor、shape、broadcast、reshape 和 dtype；
- pointer/index arithmetic、logical mask、load/store 和 atomic effect；
- pointwise SSA、reduce、scan、dot/contract；
- structured for/while/if 与 loop-carried state；
- block tile、algorithmic loop decomposition 和可调参数空间。

这是 source language 的目标语义面，不是当前 frontend 的已实现清单。
当前已落地的确切子集以下文“当前已落地的第一片”和 source/IR
映射表为准；`reshape`、atomic、scan、`while/if` 仍属于待建模能力。

## Weft source 不表达

- framework graph、多个 runtime kernel 的自动融合/拆分；
- 固定 hardware lane、VLEN、LMUL 或 vector register number；
- Scalar/RVV/IME 指令 spelling；
- register allocation、spill 和低层 instruction schedule；
- 由 kernel 名、算子名或量化格式名选择的 opaque implementation。

Weft block extent 是逻辑 blocked-program extent，不等于 RVV lane count。一个
block 可以被编译器分成多次 VLEN-agnostic strip-mined RVV execution。

## 与 Intent 的关系

Intent backend 可以把完整 Intent algorithm 单向 lowering 成 canonical Weft
Kernel IR，正如其他系统生成 Triton program。转换完成后，Weft program 是较低层
kernel algorithm 的独立编译输入；Weft compiler 不回读 Intent，不维护双向同步，
也不允许从 Weft IR 反推 Intent。

## 与旧架构的关系

旧 bounded canonical problem op、source front-door family 和 whole-kernel variant
全部退出新主干。它们不能作为 Weft Kernel DSL 的 op，也不能成为 bridge 的中间层。

## 当前已落地的第一片

当前 Python frontend 采用与 intentdsl 相同的基本机制：`@weft.kernel` 只保存
Python 定义；`weft.lower_to_mlir()` 在编译时读取源码、解析 Python AST，并用一份
瞬态 generic-MLIR assembly builder 直接生成 `weft_kernel.*`。这份 builder 只有
MLIR operation/value/region 的通用结构，没有另一套 Weft op schema 或 verifier；
canonical 语义只由注册后的 C++ Kernel dialect parse/verify。

第一片 source/IR 映射为：

| Python source | canonical Kernel MLIR |
|---|---|
| `@weft.kernel(grid_rank=N)` | `weft_kernel.kernel` |
| `W.ptr[T]` | `!weft_kernel.ptr<T, "global">` |
| `W.constexpr` | `!weft_kernel.constexpr<index>` + `weft_kernel.meta_value` |
| `W.task_id(axis)` | `weft_kernel.task_id` |
| `W.arange(start, extent)` | `weft_kernel.arange` → `!weft_kernel.block<[-1], index>` |
| `W.expand_dims(value, axis)` | `weft_kernel.expand_dims`，在指定位置插入大小为 1 的逻辑轴 |
| `W.full_like(block, scalar)` / `W.zeros_like(block, dtype)` | `weft_kernel.splat`；仅从 shape-like canonical value 取逻辑形状 |
| pointer `+` scalar/block index | `weft_kernel.ptr_add` |
| Python pointwise arithmetic/comparison | `weft_kernel.binary` / `compare` / `unary` / `cast` |
| `W.load` / `W.store` | explicit masked `weft_kernel.load` / `store` |
| `W.reduce(value, init, axis=..., kind="sum|max|min")` | `weft_kernel.reduce`；显式 init 保留算法 identity/seed |
| `W.sum(...)` / `W.max(value, init, ...)` / `W.min(value, init, ...)` | 同一个 `weft_kernel.reduce` 的 source convenience |
| `W.contract(lhs, rhs, init, lhs_axes=..., rhs_axes=...)` | `weft_kernel.contract`；结果轴为 lhs 未收缩轴再接 rhs 未收缩轴 |
| `W.dot(lhs, rhs[, init])` | 同一个 `weft_kernel.contract`；rank-1/rank-2 下默认收缩 lhs 最后一轴与 rhs 第一轴 |
| `for ... in W.range(...)` | `weft_kernel.for` + explicit carried SSA + `yield` |

Block shape 中的 `-1` 是正式的 dynamic logical extent marker；实际 extent 仍是
`weft_kernel.arange` 的 SSA operand。它不表示 VLEN、`vl` 或未知的物理 lane 数。
对两个都是 `-1` 的轴，verifier 不会仅因 `-1 == -1` 就当作相等；
它会沿 `arange/expand_dims/broadcast/load/reduce/contract/for` 的 canonical SSA
图重算 extent identity。点算 broadcast、memory footprint、contract 配对轴与
loop-carried block 无法证明具有同一运行时 extent 时，canonical verifier 直接拒绝。
当前 canonical type/verifier 已支持 rank-N block、逐轴 singleton broadcast、
`expand_dims`、沿一个轴的 reduction shape projection，以及显式成对 contraction axes。
`weft_kernel.contract` 要求 lhs/rhs block 元素类型一致，paired extent 一致，显式
accumulator 为标量或结果形状 block；C++ verifier 会独立重算输出 shape。这里仍只建立
算法语义，不等价于 contract 的 RVV/IME physical mapping 已实现。当前普通 pointwise/
memory 后端已经把同一规则推广到 rank-N：恰好一个逻辑轴进入 RVV，并由
`getVectorFastestBlockedLayout` 放到 traversal 最快端；其余轴按 `order` 从慢到快形成
嵌套 serial loops。selector 逐轴检查 canonical memory/reduction constraints；
`examples/repro_rank3_affine.py` 的 axis 1 是唯一 unit-stride store axis，因而选择
`order=[1,2,0], vector_axes=[1]`，而不是默认最后一轴或进入一个 rank-3 kernel特判。blocked load
的 vector-axis 地址必须能由 canonical pointer graph 证明为单 lane-coordinate 的
affine stride，blocked store 当前仍要求 unit stride。

`grid_rank` 属于 kernel task-program contract；具体 launch-grid extents 仍由未来
standalone launch/runtime contract 提供，正如同一个 task program 可以用不同合法
grid 实例启动。当前第一片不能据此声称 runtime launch 已完成。

仓库保留少量算法型手写 source，而不是边界测试集合：RMSNorm、softmax、rank-2/
rank-3 affine、rowwise/columnwise sum/max、dual-axis reduction、rank-1 vector dot、
单 scope rank-2 contract 与 blocked GEMM 已走通 canonical→selected→source/object。
`examples/repro_gemm.py` 的
`splat + block-carried for + contract` 不再只停在语法层；selected plan 把
`init → body argument → yield → loop result` 建模为两个 canonical use-edge handoff，
RVV emitter 保留 source 中的 `range(0, K, BK)`，每个迭代只实现 body 的 BK
contraction。
真实机器临时 harness 已到达：

```text
Python Weft RMSNorm
  → canonical weft_kernel MLIR
  → selected layout/execution MLIR
  → 结构驱动的 RVV C++ source / RISC-V object
  → 真实 RVV 机器运行并与标量参考数值对照
```

`weft-compile` 和两条 `weft-rvv-selected-execution-*` route 直接读取 canonical Kernel
IR 与 selected execution，没有接回旧 Exec/Variant/route 主干；旧 route 目前仍在
旧工具中并行注册，尚未删除。当前可执行范围包含 rank-N pointwise/memory 的一个
VLA vector axis + 任意数量 ordered serial axes。rank-1 同时支持 elementwise、f32
sum/max/min 与 full f32 dot；rank-2 支持纯 elementwise，以及沿 selected vector axis
的 f32 sum/max/min；归约结果作为每个 serial row
的派生标量继续进入 rank-1 pointer/mask/store 尾部。单 scope rank-2 f32
contract 会形成三个 value-local group：lhs 是 serial `[M,K]` projection，rhs/result
在 local axis 1 上共享 VLA lane ratio；emitter 忠实产生 `M → N-strip → K`
的 sequential outer-product。真实 RVV 上以 `tasks=2, M=5, N=37, K=29`
对照标量参考得到 `max_abs=0, max_rel=0`。blocked GEMM 的严格第一片进一步以
`BM=BN=BK=32, tasks=2×3, M=37, N=70, K=45` 覆盖 M/N 尾 tile、K 尾块和
多次 VLA strip，得到 `max_abs=0, max_rel=0`。`examples/repro_dual_axis_sum.py`
进一步让同一个 canonical f32 splat 分别沿 axis 1 与 axis 0 归约；selected plan 为
第二个 use 建立 incoming-only group 与逐 operand `layout_conversion`，真实 RVV 上以
`tasks=2, rows=19, cols=70` 得到 `max_abs=0`。后续
`examples/repro_dual_axis_load_sum.py` 又让 direct masked load 的第二个 reduction use
在无 intervening store 且 target pointer affine 可证明时重新 load；这不是 register
shuffle。`examples/repro_softmax.py` 由三个显式 canonical loops 组成：max、
exp+sum、exp+normalize/store；selected plan 为 max/sum 分别保存 reduction strategy，
为两个 vector exp 保存 `weft_rvv_execution.unary_config(strategy="exp_poly_v1")`。
选择器按各阶段压力分别选择 lane ratio，发射器没有 softmax 名称分支。该路径与
rank-3 affine 已完成 selected round-trip/source/object，但共享目标机持续有硬件扫描
负载，尚无新主干真机数值结论。不能据此声称一般 load/register 数据重排、通用
owner conversion、沿物理 serial axis 或 rank>2 reduction、任意多状态/多阶段 GEMM、
IME 或完整 runtime launch 已完成。

`examples/repro_columnwise_sum.py` 则不经过 conversion：单个 group 直接选择
`order=[0,1], vector_axes=[0]`，从 canonical `row * stride + col` 重算 runtime element
stride，并发射 masked `vlse32`。真实 RVV 上以
`tasks=2, rows=19, cols=70, stride=77` 得到 `max_abs=0`；负 stride、strided vector
store 与 indexed gather/scatter 仍未开放。

当前编译入口可以直接复核为：

```bash
PYTHONPATH=python python3 examples/repro_rms_norm.py \
  | build/bin/weft-compile --emit=selected-mlir --block-elements=256 -o rms.selected.mlir
PYTHONPATH=python python3 examples/repro_rms_norm.py \
  | build/bin/weft-compile --emit=source --block-elements=256 -o rms.cpp
PYTHONPATH=python python3 examples/repro_rms_norm.py \
  | build/bin/weft-compile --emit=object --block-elements=256 -o rms.o
PYTHONPATH=python python3 examples/repro_rank2_affine.py \
  | build/bin/weft-compile --emit=selected-mlir --block-elements=256 -o rank2.selected.mlir
PYTHONPATH=python python3 examples/repro_rank2_affine.py \
  | build/bin/weft-compile --emit=object --block-elements=256 -o rank2.o
PYTHONPATH=python python3 examples/repro_rowwise_sum.py \
  | build/bin/weft-compile --emit=object --block-elements=256 -o rowwise-sum.o
PYTHONPATH=python python3 examples/repro_columnwise_sum.py \
  | build/bin/weft-compile --emit=object -o columnwise-sum.o
PYTHONPATH=python python3 examples/repro_dual_axis_sum.py \
  | build/bin/weft-compile --emit=object -o dual-axis-sum.o
PYTHONPATH=python python3 examples/repro_vector_dot.py \
  | build/bin/weft-compile --emit=object --block-elements=256 -o vector-dot.o
PYTHONPATH=python python3 examples/repro_contract.py \
  | build/bin/weft-compile --emit=object --block-elements=256 -o contract.o
PYTHONPATH=python python3 examples/repro_gemm.py \
  | build/bin/weft-compile --emit=object \
      --meta BM=8 --meta BN=32 --meta BK=16 -o gemm.o
PYTHONPATH=python python3 examples/repro_softmax.py \
  | build/bin/weft-compile --emit=object --meta BLOCK=64 -o softmax.o
PYTHONPATH=python python3 examples/repro_rank3_affine.py \
  | build/bin/weft-compile --emit=object -o rank3-affine.o
```

`--meta NAME=VALUE` 按 canonical `arg_names` 独立绑定每个 constexpr 参数；名称只用于
寻址，不触发任何算子或 kernel 分类。`--block-elements=N` 是显式要求所有 constexpr
统一取 N 的简写。两者互斥且没有隐藏默认值，所有 constexpr 都必须被完整绑定。
这些值与 `--register-budget` 只进入 selected execution；例如寄存器预算收紧时，
不同 lexical group 可以得到不同 lane ratio。

## 待裁问题

- launch-grid extent expression 与未来 multi-hart runtime 的最小 contract；
- 多 vector axis、strided store、indexed gather/scatter，以及
  load/register/owner 的一般 layout-conversion realization；
- rank>2 reduction/contract、沿 serial axis、多轴 reduction/scan、transpose/reshape
  的物理 lowering；
- reshape/transpose 及更完整的 static/symbolic extent 关系；
- `if`/`while`/helper、atomic/fence 的下一片 canonical 语义；
- 多个 carried state、多阶段 GEMM、自动/离线 BM/BN/BK tuning、更多 rank-2
  operand layout relation、IME owner 与 cleanup handoff；
- Scalar/IME owner，以及 RVV 之外的 execution candidate 与 tuning。
