# Weft 十算子全栈与 Selection 报告

日期：2026-08-11

## 结论

Weft 当前已经从完整 Python kernel 生成 canonical Kernel IR，再按 canonical anchor 生成
Selected Execution IR，最后投影为可编译 C++ source。十个现有手工 repro 均已由生成源码
在 `ssh rvv` 上重新编译并执行通过；它们不是十个 GGML RISC-V 性能 baseline。

这条路径不是从 GGML C/C++ graph 导入，也不识别“这是 RMSNorm、GEMM 或 q4_K”。
真正的 Weft source 是 `examples/kernels/` 中作者写出的完整 worker-local Python kernel。
数值 runtime 位于 `examples/repro/weft/`；emitter 不读取或链接 `source/c/`。真实 GGML
RISC-V source 与模型级执行边界另见 `riscv-llm-source-corpus.md`。

## 编译边界

```text
Python kernel
  → canonical weft_kernel IR
  → target-bound, anchor-local candidate selection
  → weft_execution plan
  → selected-provider source emission
  → C++ translation unit
  → ssh rvv compile and execute
```

Selection 的输入是 canonical op、typed operands、logical relation、effect、局部 use context 与
target facts。实现中没有 `KernelKind`、operator/q-format route、kernel-name dispatch 或整段
kernel shape template。Scalar 与 RVV 都是显式候选；没有合法 provider 时 selection 立即报错。

Emitter 只读取 canonical IR、对应 anchor 的 selected record 与 plan 中的 target facts。它不
重新推导算法骨架，也不从 `source/c/` 调用或链接 donor 实现。

## 十个现有 repro

统一命令为：

```bash
./examples/run/weft.sh <kernel>
```

该入口本地执行 frontend、selection 与 source emission，再把生成源码和相邻 runtime 放入
远端临时目录。生成的 C++ kernel 统一由 Clang 17 以 `-march=rv64gcv -mabi=lp64d`
编译；quant repro runtime 使用 C11。每次执行结束都删除临时目录。

| Repro | Python kernel | 当前 selected provider 摘要 | `ssh rvv` |
| --- | --- | --- | --- |
| `add_bias` | `examples/kernels/elementwise/add_bias.py` | VLA + 3 memory anchors：RVV | `PASS weft add_bias` |
| `rms_norm` | `examples/kernels/normalization/rms_norm.py` | 2 VLA + memory + reduce：RVV；`rsqrt`：Scalar | `PASS weft rms_norm_worker` |
| `online_softmax` | `examples/kernels/normalization/online_softmax.py` | summary、`exp`、memory、VLA：Scalar | `PASS weft online_softmax_f32` |
| `blocked_gemm` | `examples/kernels/contraction/blocked_gemm.py` | f16/f32 contract：RVV；block axis 与 memory：Scalar | `PASS weft gemm_worker` |
| `q4_0_q8_0` | `examples/kernels/quantization/block_dot.py` | block axis、indexed/unit u8 memory、decode、i32 reduce：RVV；scale：Scalar | `PASS q4_0_q8_0_block_dot` |
| `q4_1_q8_1` | 同上 | block axis、indexed/unit u8 memory、decode、i32 reduce：RVV；scale/correction：Scalar | `PASS q4_1_q8_1_block_dot` |
| `q5_0_q8_0` | 同上 | block axis、两路 indexed + unit u8 memory、decode、i32 reduce：RVV；scale：Scalar | `PASS q5_0_q8_0_block_dot` |
| `q5_1_q8_1` | 同上 | block axis、两路 indexed + unit u8 memory、decode、i32 reduce：RVV；scale/correction：Scalar | `PASS q5_1_q8_1_block_dot` |
| `q8_0_q8_0` | 同上 | block axis、unit u8 memory、widen、i32 reduce：RVV；scale：Scalar | `PASS q8_0_q8_0_block_dot` |
| `q4_K_q8_K` | 同上 | 两个独立 block axis、indexed/unit memory、mask/select/tuple、两次 i32 reduce：RVV；final scale：Scalar | `PASS q4_K_q8_K_block_dot` |

RMSNorm 在同一个 entry 内同时含 RVV reduction、RVV pointwise/memory 和 Scalar math；GEMM
在同一个 entry 内同时含 RVV contract 与 Scalar block/memory。`q4_K_q8_K` 则在同一个外层
block loop 中分别选择 256-lane 与 16-lane 两个 block owner，并各自闭合为 RVV reduction
slice。这些结果证明 provider 组合单位是局部 anchor 与 SSA relation，而不是互斥的
whole-kernel backend。

`rv64gc` target 不注册 RVV candidate；同一 add/GEMM canonical kernel 会选择纯 Scalar plan，
生成源码不包含 `riscv_vector.h`，并能由普通 C++17 compiler 编译。

## 当前 provider 覆盖

已实现：

- Scalar structured control、VLA、logical block、broadcast、masked memory、tuple/state、reduce、
  summary fold、rank-2 contract、cast/bitcast 与整数 bitwise；
- RVV e32m1 dynamic-strip VLA pointwise 与 unit-stride f32 load/store；
- RVV active-axis `relaxed` f32 add reduction，跨 strip 保留 canonical identity；
- RVV rank-2 `f16 × f16 → f32` relaxed/native contract；invalid operand 在 provider 内先按
  contraction additive identity 物化，再进入 vector multiply/reduction。
- RVV e32m1 block-owner strip：从 canonical SSA 重算 owner，支持 u64 index arithmetic、
  unit-stride 与 indexed u8 memory、computed mask + filled load、u8/u16/i8/i16/i32 cast 与
  bitcast、integer bitwise/shift、compare/select/tuple projection、i32 pointwise 与 relaxed
  add reduction；同一个 kernel 可含多个互不依赖的 owner slice。

现有 RVV legality 是 typed、局部且保守的。computed predicate、非 unit-stride VLA memory、
vector transcendental、不同 dtype/LMUL 或不同 contract relation不会偷偷进入不匹配的
provider。Block slice 还要求单一 owner、闭合 use relation、支持的 typed primitive 集合以及
i32 pointwise/reduction 的静态范围证明；失败时该 anchor 不产生 RVV candidate。

## 尚未完成

- Online softmax 的 `summary_fold(order="preserve")` 和 vector `exp` 尚无 RVV provider；当前
  选择 Scalar，未用近似多项式冒充 `math="native"`。
- `weft_ext.block_scaled_contract` 尚无满足其 rounding/saturation 语义的 provider；selection
  明确报告 `has no legal realization provider`，不会伪装成普通 contract。
- IME capability 尚未进入 target profile，也没有 IME fragment/provider/lowering。
- 当前 artifact 边界支持 canonical MLIR、selected MLIR 与 readable C++ source；object、static
  library、public header、dispatcher 与 tuner 尚未实现。
- Scan、atomic、prefetch、while 以及更一般的 shaped `if` result 尚未纳入当前 source provider；
  遇到这些已验证但未注册的 realization 时必须明确失败。

这些缺口是 provider/toolchain 覆盖范围，不是通过 kernel 分类或 fallback 绕开的理由。

## 审计判据

后续实现若出现以下任一形态，应直接判定为架构回退：

- 以 kernel 名、算子名、q-format 或整体 shape 决定整条 lowering；
- emitter 读取 GGML donor source、旧 route registry 或第三份调度结构；
- 一个新结构只能通过增加互斥 whole-kernel 分支获得支持；
- unsupported selected primitive 静默改走 Scalar 或旧 emitter；
- 为量化或扩展吞掉 rounding、saturation、validity 或 numerical mode。
