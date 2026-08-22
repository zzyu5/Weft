# use-def physical pass 与真实 auto 扫描

本轮没有增加 `std` 树，也没有让编译器在 `mac_pairs` 与 `mac_groups(n=4)` 之间选择。两者仍是作者写下的两份数值分解。本轮只补三件事：沿 use-def 推导表示与逐次访存形态、在 liveness 已知后决定 handoff/materialization、把原先固定为 `1/1/0` 的局部调度参数接入真实的编译—运行—排序流程。

## 1. 三类决定的边界

| 类别 | 当前实现中的内容 | 不允许做的事 |
|---|---|---|
| pass 推导 | lane identity、typed SEW、保持 lane 数所需的 LMUL、vl、每次 extract 的 memory form、live interval、reload/shared handoff | 不枚举、不按性能改写作者树 |
| auto tune | `unroll`、`pipeline_depth`、`prefetch_distance`，以及 DSL 中已有的 `NC/KC/MC/MR/NR/KB` 等 `auto` 数值 | 不创造新 Level，不改变逻辑值集合 |
| 作者写 | Level 树、cohort/block 数值口、`mac_pairs` 或 `mac_groups`、state 与 handoff 归属 | compiler 和 tuner 都不代选 |

一次 concrete candidate 的流水仍然是前向的：

```text
source auto bindings + physical auto bindings
→ representation pass
→ local operation / per-use memory pass
→ Level schedule pass
→ live-range resource pass
→ intrinsic C
```

内部的静态 winner 只用寄存器峰值过滤和排序；它不冒充真机性能。真机 winner 由构建期驱动对每个 concrete binding 独立编译、运行、排序。编译器一次接收多值范围时，q4_K 的 `3 × 2 × 3` 空间显示为 `evaluated_candidate_count = 18`、`legal_candidate_count = 18`。

## 2. physical pass 实际改变

### 2.1 SEW、LMUL 与 vl

旧规则是按一个全局 lane axis 套 `max(8, logical_sew)`，再为每个值取第一个能装下 lanes 的 LMUL。新规则先按 op-specific use-def transfer 与 handoff 建立 lane chain：

- `binary/unary/cast/widen/reduce/dot/contract/mac/extract` 各自声明是否保持 lane identity；
- `Level/for/if/while` 不再因为“出现在同一个 operation”而把不同 carried value 合并，carry 只通过自己的 handoff class 相连；
- typed value 决定 SEW；同一 chain 保持相同逻辑 lane 数，widen 后 LMUL 随 SEW 增长；
- 符号 extent 使用已实例化 Level partition，而不是错误退化成 1 lane；
- RVV 的最小合法 LMUL 同时受 SEW 约束，例如 e32 不会再生成不存在的 `mf4` 类型。

每个 value 都写入 `representation_chain`、`sew_derived_from`、`lanes_derived_from`、`lmul_derived_from`、`vl_derived_from` 和参与传播的 op 列表。

### 2.2 memory form 是 value-use edge 的决定

encoding declaration 现在只给 field 的布局事实；真正的访问形态挂到每一次 `extract`：

- q4_K `q[s]`：`unit-stride-layered-unpack`；
- q4_K `sc[s]`：`unit-stride-multi-load-join`；
- Q8_K `q[s]`：`scalar-indexed`；
- dense admit/commit：由 pass 明确给出 `runtime-strided`，emitter 不再检查 stride 字符串后自行选择 `vle/vlse` 或 `vse/vsse`。

raw LMUL 也根据这一次 extract 的 result mapping 计算，不再沿用 field 的全局访问形态。Emitter 只把已选 mapping 展开成 byte address、load 和位操作。

### 2.3 handoff 与资源

`admit` result 在表示 pass 中只标记为 `pending-liveness`。资源 pass 获得真实 definition/use ordinal 后才决定：

- 单 consumer：`reload-per-use`；
- 多个不同 consumer：先按 `shared-register` 计算；
- 超预算时，只对可从 admit source 重载的 value 改成 `reload-per-use` 并重新计算峰值；
- 仍超预算的 candidate 直接非法，不回头缩 LMUL、不换 instruction、不进入 fallback。

资源峰值还显式计入 unroll 产生的额外 partial 和 depth-2 buffering 产生的 decoded operand temporary。当前没有为 computed value 假造 stack spill；这类 candidate 若超预算会明确失败。

## 3. q4_K assignment dump 的关键链

完整 dump 可由下面一条命令生成；它不是提交到仓库的第二份 IR：

```bash
tmp=$(mktemp -d /tmp/weft-assignment.XXXXXX)
PYTHONPATH=python python -m weft examples/kernels/quantization/q4_k_gemv.py > "$tmp/kernel.mlir"
build/tools/weft-compile/weft-compile "$tmp/kernel.mlir" \
  --emit=physical-assignment \
  --march=rv64gcv_zfh_zvfh --abi=lp64d --vlen-bits=128 \
  --auto-unroll=1 --auto-pipeline-depth=2 --auto-prefetch-distance=0 \
  -o "$tmp/assignment.txt"
```

选中 assignment 中 q4 MAC 主链为：

| entity | typed provenance | lane chain | SEW | lanes | LMUL | memory / op realization |
|---|---|---:|---:|---:|---:|---|
| `v33`，q4 `q[s]` | `op24 weft_kernel.extract` | `r9` | 8 | 16 | m1 | `unit-stride-layered-unpack` |
| `v36`，pair partial | `op27 weft_kernel.mac_pairs` | `r9` | 16 | 16 | m2 | `rvv.vwmaccsu` |
| `v37`，widened partial | `op28 weft_kernel.widen` | `r9` | 32 | 16 | m4 | typed widen preserves lanes |
| q8 activation term | Q8_K natural field extract | 非 row-lane value | 8 | 1 | none | `scalar-indexed` |
| scale `sc[s]` | joined field extract | `r9` | 8 | 16 | m1 | `unit-stride-multi-load-join` |

`op27` 同时记录：

```text
instruction              = rvv.vwmaccsu
lhs_lmul_eighths         = 8
partial_lmul_eighths     = 16
rhs_access               = natural-scalar-field
schedule                 = unroll 1 / pipeline_depth 2 / prefetch_distance 0
temporary_vector_groups  = 2
```

因此这条链的 m1 → m2 → m4 不是“第一个能装下”的三次独立选择，而是同一 row-lane use-def chain 在 8/16/32-bit typed value 上保持 16 个逻辑 lanes 的结果。

## 4. auto 空间与实测扫描

构建期扫描入口是：

```bash
examples/run/weft-kernel-tune.sh sg2044 q4_k_gemv 5
```

它对每个组合重新生成 Kernel IR、运行完整 physical pass、生成 intrinsic C、交叉编译，在 SG2044 上先做 bit-exact 比对，再按实际 throughput 排序。

### 4.1 `mac_pairs` 树

单位均为 GOP/s；扫描使用 5 次重复。

| unroll | pipeline | prefetch 0 | prefetch 1 | prefetch 2 |
|---:|---:|---:|---:|---:|
| 1 | 1 | 5.765895 | 2.738437 | 2.693908 |
| 1 | 2 | **6.027521** | 2.777045 | 2.754082 |
| 2 | 1 | 5.505565 | 4.761255 | 4.911044 |
| 2 | 2 | 5.696572 | 4.484468 | 4.467032 |
| 4 | 1 | 2.577567 | 5.012094 | 2.274796 |
| 4 | 2 | 4.853429 | 2.751936 | 1.666105 |

Winner：`unroll=1, pipeline_depth=2, prefetch_distance=0`。

### 4.2 `mac_groups(n=4)` 树

它是独立作者树，单独走同一个 physical/tune 流程。

| unroll | pipeline | prefetch 0 | prefetch 1 | prefetch 2 |
|---:|---:|---:|---:|---:|
| 1 | 1 | **6.760668** | 3.048895 | 4.766244 |
| 1 | 2 | 6.224010 | 2.900626 | 4.095050 |
| 2 | 1 | 6.076516 | 3.780031 | 2.601127 |
| 2 | 2 | 5.731810 | 4.590127 | 1.958752 |
| 4 | 1 | 6.036102 | 5.627387 | 5.985294 |
| 4 | 2 | 4.860946 | 4.395399 | 4.840398 |

Winner：`unroll=1, pipeline_depth=1, prefetch_distance=0`。

### 4.3 DSL blocking auto

符号 `auto("MR")` 不再要求一次命令只绑定一个值。`--meta MR=2,4 --meta KB=32,64` 会在 compiler 外层形成 4 个完整 candidates；assignment 记录 4 个 evaluated、4 个 legal。

构建期驱动也实测了同一 GEMM 的四个 blocking 实例（1 次重复，仅证明扫描链真实工作，不写入性能 CSV）：

| MR | KB | GOP/s |
|---:|---:|---:|
| 2 | 32 | 3.475662 |
| 2 | 64 | 3.591714 |
| 4 | 32 | 5.757979 |
| 4 | 64 | **6.147325** |

这里 tuner 只填作者已经暴露的 `MR/KB` 数值口；它没有创造另一棵 GEMM 树。

## 5. 正式性能结果

正式记录使用 10 次重复并已写回 `weft-kernel-performance.csv`。

| 项目 | 上轮 Weft | 本轮 Weft | GGML 同 shape baseline | 本轮 / GGML |
|---|---:|---:|---:|---:|
| q4_K × q8_K `mac_pairs` | 5.765439 | **6.034362** | 9.027928 | 66.84% |
| q4_K × q8_K `mac_groups(4)` | 6.757720 | **6.753204** | 9.027928 | 74.80% |

`mac_pairs` 提升 4.664%，与 GGML 的差距从约 1.566× 缩到 1.496×。`mac_groups(4)` 的最佳配置仍是原来的 `1/1/0`，本轮数值相对上轮为 -0.067%，可视为持平；它仍比 pair 树快，但这是作者分解差异，不是 tuner 在两棵树之间做了选择。

受相同表示/访存 pass 影响的 q8_K activation quantize 也重新执行为 323.605118 MElements/s，上轮为 326.861970 MElements/s，变化 -0.996%；本轮没有宣称它得到性能改善。

## 6. 剩余差距的具体归因

这次结果排除了两个错误归因：

1. q4 主链的 LMUL 不是当前瓶颈。assignment 已给出与 widen use-def 一致的 m1/m2/m4 链，扩大 unroll 反而增加寄存器和控制开销。
2. “把 1/1/0 换成几个常数”也不是答案。18 个真实生成、真实运行的候选中，只有 pair 树的 depth-2 local buffering 有约 4.7% 收益；所有逐 term prefetch 都明显变慢。

剩余 1.496× / 1.337× 的差距主要落在两处：

- **推导规则仍缺 memory-edge 的 coordinate/address reuse。** 当前 pass 已推导访问 form，但 grouped/layered byte coordinate 仍在生成的局部循环里重复展开，尚未把同一 group 的地址公共部分、activation address 和 decode window 形成可复用 schedule fact。
- **当前 emit shape 只有 primitive-local 两级 buffering。** `pipeline_depth=2` 表示在一个 compiler-generated group 中先 load/decode、再执行 MAC；还不是跨 group iteration 的完整 prologue/steady-state/epilogue 流水。盲目增加 unroll 或逐 term prefetch 不能替代这种结构。

因此本轮性能没有卡在作者树错误，也没有证据要求 compiler 改树。`mac_groups(4)` 已证明更宽的作者分解可自然重解，但编译器内部仍缺更强的地址复用和跨局部迭代调度。这个缺口属于 shared memory/schedule derivation 与对应 emitter projection，不属于新增格式分支。

## 7. 当前明确边界

- pipeline 目前只有两个真实实现：depth 1 的即时 load/compute 与 depth 2 的局部 operand buffering；CLI 会拒绝大于 2 的伪候选。
- q4 主链没有产生不兼容 consumer layout，因此 assignment 中没有 layout conversion。一般性的 consumer layout 冲突与冗余 conversion 消除尚未由这条 repro 证明。
- resource pass 只会把可重新读取的 admitted value 变成 reload；computed value 超预算仍明确判 candidate 非法。
- 没有 kernel 名、q4 格式名、VLEN 型号或 fallback 进入 schedule/tuner 选择；q4 代码路径来自 typed op、encoding mapping、use-def 与 concrete auto binding。
