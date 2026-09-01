# RISC-V Physical IR 归一化与 terminal contract 实测

## 1. 本轮问题

本轮不改作者树，也不铺量化格式。检验对象是第二层 RISC-V Physical IR 是否真实承载会影响
use-def、控制流、资源、共享、CSE、调度和合法性的物理程序，而不是把一段待执行程序留在
attribute 或 terminal emitter 中解释。

诊断按 `doc/compiler/optimization-principles.md` 第九节执行：先固定入口和 physical binding，
数 final Physical IR 中的 loop、supply、memory edge、partial、MAC、reduction 和临时资源，再决定
改动。若动态工作计数不变，不以“新增 typed op”本身作为实现理由。

两个直接输入是：

- Q4_K `mac_pairs` / `mac_groups(4)`：原 final IR 含 opaque
  `rvv_grouped_mac_reduce`，动态 full/remainder/tail loop 由 emitter 构造；
- Q8_0 与 IQ1_S production MUL_MAT：multi-stream `rvv_widen_dot` 的 issue slicing、每 stream
  partial 和最终标量 combine 仍由 emitter 构造。

## 2. 参考实现给出的边界

动手前对照了以下机制：

- Triton `RemoveLayoutConversions.cpp:42-59` 把 layout propagation、冲突处理、
  rematerialization 和 dominance-order rewrite 放在 IR pass 中；terminal conversion 不再重建这些关系。
- Triton `DotOpToLLVM/FMA.cpp:18-43` 消费已经选定的 operand/register 组织，机械生成局部 FMA；
  它没有在 leaf conversion 中重新生成外层 dot traversal。
- Triton `AccelerateMatmul.cpp:42-83` 的结构优先关系来自 target-aware 实现策略，先按固定顺序
  尝试、再检查 legality，不以某一台机器的经验常量充当跨 target 定理。
- TileLang `reducer_plan_materialize.cc:559-733` 把 reducer plan 与 update-site axes 绑定，随后
  materialize 成实际 partial program；plan 不是 terminal printer 的脚本。

Weft 不能直接照搬 thread/warp ownership 或 GPU shared-memory object，但可以采用同一个责任边界：
plan 可作为一次 lowering 中的瞬态冻结结果，final Physical IR 必须是 verifier、CSE、scheduler 和
terminal translator 都能直接观察的程序。

## 3. 实际改动

### 3.1 动态 grouped-MAC 程序进入 IR

删除了 `RVVGroupedMacReduceOp` 以及它在 verifier 和 emitter 中的整条路径。所有 grouped-MAC
现在统一物化为：

```text
scf.for
  -> rvv_grouped_mac_load
  -> rvv_grouped_mac_step
  -> scf.yield
```

full-chunk/remainder/tail 的动态控制流由 `scf.for`/guard 承载；fixed-window loop 带明确的
`weft.riscv.system_unroll = "disable"`，final verifier 从每个 load 反查最近父 loop 并强制该合同。
`RVVGroupedMacLoadOp` 和 `RVVGroupedMacStepOp` 仍是 fixed slots/terms/parts 的闭合局部 op：它们的
memory form、window geometry、term order、leaf 和资源已经确定，emitter 只展开有限且唯一的
intrinsic 拼写。若以后需要在它们内部的单次 load/MAC 之间做跨 op 调度，这一事实会使它们再次
越过边界；当前没有这种隐藏的动态 loop。

同时把 grouped result 的 issue-time × register-replica part mapping 写进 typed window，tail VL
按每个 physical part 生成，不再假设所有输出只有 register replicas。

### 3.2 multi-stream partial program进入 IR

`SequentialPartialPlanAttr` 现在冻结：

- realization：`fused` 或 `per_stream`；
- reduction axes 与 issue count；
- lhs/rhs issue carrier type；
- accumulator type；
- 每个 output×issue 的 source part 与 lane offset；
- accumulate/finalize leaf；
- 联合 resource groups。

planner 先选择这些事实；materializer 只按 plan 创建：

```text
rvv_issue_slice
  -> rvv_splat
  -> rvv_widen_accumulate
  -> rvv_finalize_widen_dot
  -> explicit scalar add or rvv_assemble_replicas
```

`RVVIssueSliceOp` 的 type 和 attributes 明确描述 source part、lane offset、reduction axes 与目标
LMUL。emitter 只据此拼写 `vget`/`vslidedown`，不再从 composite dot 反推切片。

final IR 禁止 multi-stream `RVVWidenDotOp`。只有 `reduction_streams == 1` 的 closed local
widening-dot leaf 可以保留；五类 partial program plan attribute 在 materialization 后全部删除，
final verifier 对残留 plan 直接报错，terminal emitter 不读取这些 plan。

### 3.3 CSE 与 verifier 合同

标准 CSE 暴露了一个真实 verifier 错误：Q8_0 中 32 个相同零向量被合法合成 1 个、64 个相同
issue slice 合成 24 个，MAC 与 reduction 数保持 32/32；但旧 verifier 要求 splat seed 与每个
accumulate 位于同一 block。这个条件不是数值或资源语义，只是原 spelling 的偶然性质。

现在 pure seed 可以来自任一支配 block；partial 链仍要求 typed accumulate/finalize 关系闭合。
没有通过伪造 side effect 禁止 CSE。自定义 layout conversion CSE 的 identity 也从
`input + result type` 收紧为 `input + result type + conversion + source_access + leaf`，避免把不同
temporary/resource/leaf 合同的 conversion 静默合并。

### 3.4 target policy、alias 与 Share

- partial independent-vs-sequential 的固定优先关系进入 `TargetAttr`；planner 只读取 target fact
  和 typed geometry/resource legality，不含 target、kernel 或 format 名分支。independent resource
  gate 与 materializer 统一计入额外 2 个 temporary groups。
- kernel ABI 新增 `alias_groups`。未声明 View 全部属于 alias set 0（may-alias）；命名组从 1 开始，
  不同组表示调用方声明 disjoint。Canonical ABI、Physical `MemDesc` 与 C parameter qualifier 原样
  传播；final physical verifier 要求 ABI alias 数组和每个 `MemDesc` 完全一致。只有 singleton
  disjoint group 发出 `restrict`，默认入口不发。
- Share pass 的已物化 guard 覆盖 layered/window/replica storage operations，包括
  `RVVStorageWindowOp`。已处理 field 第二次运行不会再次插入 storage materialization。
- `weft-opt` 注册 RISC-V dialect、Share、layout canonicalization 与 final verifier；
  `weft-compile` 能把已经 physicalized 的 module直接翻译成 intrinsic C，不重新运行 lowering。

CSV 没有被本轮定向回归局部更新。实验协议现在要求正式 Weft CSV 只能由 clean worktree 上的
完整 manifest 运行整体、原子替换；仓库当前还没有实现这一完整 snapshot runner，因此本轮数据
只保存在本报告，不能声称正式 CSV 已代表当前 dirty checkout。

## 4. Physical IR 外部验收

### 4.1 独立 parse、标准 pass 与 terminal translation

Q8_0 与 IQ1_S 的 final IR 都实际执行了：

```text
weft-opt:
  canonicalize
  cse
  weft-riscv-canonicalize-layouts
  weft-riscv-share-layered-windows
  weft-riscv-share-layered-windows
  weft-riscv-verify-final
then:
  weft-compile --emit=intrinsic-c <physical-ir>
```

两份 IR 均通过 `--verify-each --verify-roundtrip`，随后从 replay 后的 Physical IR 直接生成 C。

| entry | final `rvv_widen_dot` | issue slice | widen accumulate | finalize | assemble |
|---|---:|---:|---:|---:|---:|
| Q8_0 | 0 | 64 | 32 | 32 | 2 |
| IQ1_S | 0 | 32 | 16 | 16 | 1 |

标准 CSE 后，Q8_0 issue slice 从 64 合并为 24、splat 从 32 合并为 1；32 个 accumulate 与
32 个 finalize 均未被错误合并。IQ1_S 仍为 32/16/16。final IR 中没有 partial program plan。

TQ2_0 的 independent partial 输入另行执行了标准 canonicalizer/CSE：前后均为
`1 rvv_partial_set + 3 rvv_partial_combine + 1 rvv_partial_finalize`，final verifier通过。
这直接检验了标准 CSE 不会破坏 independent partial identity。

### 4.2 grouped-MAC macro 消失

Q4_K `mac_pairs` 与 `mac_groups(4)` 在 SG2044/K1 四份 final IR 中：

- `rvv_grouped_mac_reduce`：0；
- 动态 traversal：真实 `scf.for`；
- local work：`rvv_grouped_mac_load + rvv_grouped_mac_step`；
- 两次 Share、标准 canonicalizer/CSE、round-trip parser 与 final verifier：全部通过。

### 4.3 Share 幂等性

当前 `row_dequantize_q4_k` final IR 含 2 个 `rvv_storage_window`。在该真实入口上分别运行一次
和第二次 Share，第二次输出 diff 为 0 行，且两次都通过 final verifier/round-trip。

当前仓库没有一个 DSL/example 会生成 `rvv_layered_window` pair op；因此该 guard 的代码路径已
闭合，但没有伪造 fixture 来宣称真实 runtime 覆盖。

### 4.4 target policy 与 alias

同一份 TQ2_0 canonical input、同一 K1 target 和 LMUL=m1：

- `independent-multilevel` 生成 1 partial set、3 combine、1 finalize；
- `sequential` 生成 8 widened accumulate 和 1 final reduction；
- 两者均通过标准 CSE 与 final verifier。

这证明 policy 是 Physical program 的 target-profile 输入，而不是 planner 中写死 SG2044 的
跨目标经验分支。

同一 F32 GEMV 作者程序的 alias 对照：

```text
default:  void gemv_f32(const float * W, const float * X, float * Y, ...)
disjoint: void gemv_f32(const float *restrict W,
                        const float *restrict X,
                        float *restrict Y, ...)
```

显式版本在 canonical/physical ABI 中均为 alias sets `[1, 2, 3]`，并通过 Physical IR replay；
默认版本为共同 may-alias 组，不发 `restrict`。

## 5. 双机运行结果

### 5.1 本轮直接受影响的 production path

全部为 10 repetitions，数值判据为 tolerance + 明显错误检查。

| kernel | target | phase | Weft GOP/s | source GOP/s | ratio | numeric |
|---|---|---|---:|---:|---:|---|
| Q8_0 | SG2044 | decode | 7.377 | 0.559 | 13.197× | within-tolerance |
| Q8_0 | SG2044 | prefill | 8.316 | 2.052 | 4.053× | within-tolerance |
| Q8_0 | K1 | decode | 2.655 | 2.097 | 1.266× | within-tolerance |
| Q8_0 | K1 | prefill | 2.209 | 2.230 | 0.991× | within-tolerance |
| IQ1_S | SG2044 | decode | 3.194 | 2.788 | 1.145× | within-tolerance |
| IQ1_S | SG2044 | prefill | 0.565 | 2.808 | 0.201× | within-tolerance |
| IQ1_S | K1 | decode | 2.014 | 2.678 | 0.752× | within-tolerance |
| IQ1_S | K1 | prefill | 0.814 | 2.748 | 0.296× | within-tolerance |

IQ1_S prefill 的低性能不是本轮造成的新结论，也没有被结构验收掩盖。本轮只证明原先由 emitter
解释的 multi-stream program 已成为真实 IR 并可执行；没有把 IR 归一化包装成 IQ1_S 性能优化。

### 5.2 Q4_K 等价作者写法与 persistent 回归

Q4_K grouped path 使用同一已验证 physical binding：LMUL=m4、unroll=2、pipeline depth=1。

| tree | target | GOP/s | numeric |
|---|---|---:|---|
| `mac_pairs` | SG2044 | 9.272 | within-tolerance |
| `mac_groups(4)` | SG2044 | 9.808 | within-tolerance |
| `mac_pairs` | K1 | 2.839 | within-tolerance |
| `mac_groups(4)` | K1 | 3.190 | within-tolerance |

Q4_K persistent production prefill 为 SG2044 9.815 GOP/s（source 9.699，1.012×），K1 RVV
3.192 GOP/s，均 10 repetitions、within-tolerance。K1 source 行是 IME 路径，不能用来评价本轮
RVV Physical IR 重构的同引擎 ratio。

一项负对照必须保留：若省略 binding、使用默认 LMUL=m1/unroll=1，`mac_pairs` 只有
SG 3.060/K1 1.432，`mac_groups(4)` 只有 9.813/1.588。补回相同 physical binding 后性能恢复。
因此这组低数字是参数口径不一致，不是 IR 归一化回归；也说明任何回归数字都必须同时记录
physical binding。

### 5.3 已过线批次回归

同一轮还运行了十个既有量化入口，全部 10 repetitions、within-tolerance：

| format | SG2044 GOP/s | K1 GOP/s |
|---|---:|---:|
| Q4_0 | 7.378 | 2.550 |
| Q4_1 | 5.491 | 2.245 |
| Q5_1 | 4.794 | 2.059 |
| Q1_0 | 4.261 | 3.780 |
| Q2_K | 5.537 | 1.359 |
| Q6_K | 4.679 | 3.136 |
| TQ2_0 | 9.834 | 6.214 |
| IQ4_XS | 6.457 | 2.198 |
| IQ4_NL | 6.764 | 2.421 |
| Q5_K | 4.063 | 1.586 |

## 6. 没有冒充完成的边界

1. 统一默认 binding 的 24 formats × 2 targets 静态扫描是 42/48，不是正式 production
   manifest。失败项为 Q3_K/Q6_K 的 byte-addressable indexed-entry 合同、K1 IQ2_S/IQ2_XS 的
   full-product resource binding、SG IQ3_S bitmask part mapping、SG TQ1_0 scalar iota。
   这些入口需要 runner 中各自的 production binding；没有为让一个假统一扫描变绿而放松 verifier。
2. `RVVGroupedMacLoad/Step`、single-stream `RVVWidenDot` 与 typed partial ops 仍会由 emitter展开
   固定数量的 intrinsic。它们当前是 operands/results/resources 完全闭合的局部 op，不含动态 outer
   traversal。该结论不是“emitter 行数少”，而是 final IR 已能独立计算动态 loop 数、op 数和 use-def。
3. alias facts目前只进入 ABI、MemDesc verifier 和 C `restrict`；hoisting pass仍采用“loop 内无写”
   的更保守判据，没有声称显式 disjoint 已经解锁跨写 hoist。
4. Share 在真实 `rvv_storage_window` 输入上已通过二次运行零 diff；当前语料没有
   `rvv_layered_window` 外部输入，因此该具体分支没有 runtime 证据。
5. 正式 CSV 的完整 snapshot runner尚不存在，本轮没有修改正式 CSV。协议已经把混合快照定义为
   非法，但自动化执行能力尚未存在。
