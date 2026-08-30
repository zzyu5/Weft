# Q2_K window 复用静态审计：重复劳动假设的反证

## 1. 结论

本轮先统计完整 physical identity，再决定是否修改 pass。结果不支持“Q2_K 的主要性能差距来自相同 window / decode / activation 被重复 materialize”这一判断。

最终 RISC-V IR 中确实有 16 个静态 `rvv_storage_window`，但完整 identity 必须包含：

```text
Field SSA
PhysicalPoint SSA
logical_offset SSA
reduction axis
projection base / stride / repeat / extent
AccessAttr
result type / layout / validity
leaf
```

加入 `logical_offset` 后，16 个 window 中没有两个具有相同完整 window identity。projection 的 `repeat` 仍可能使不同 logical offset 在运行时落到同一 storage element；scale 的 2× 地址别名就是这种情况。一个只按 `Field + Point + Access + projection` 合并、完全忽略 offset 与 repeat 的 pass 会读错数据。

真正存在的重复只有一处：SG2044/VLEN128 上，scale 的 `repeat=16` 使相邻两个 8-lane iteration 使用同一个 scale，因此 32 次动态 scale materialization 对应 16 个不同 scale。K1/VLEN256 使用 16 lanes 后，每个 scale 只 materialize 一次，这项重复为零。

该重复上限不能解释 SG 的 5.37× source 差距，也不能解释 K1 的 1.45× 差距。因此本轮没有增加 read-CSE、LICM、backward rematerialization 或 lifetime-expansion 代码；这些 pass 对当前 Q2_K 完整 identity 会删除 0 个 physical op。

## 2. Q2_K 的完整计数

静态生成命令：

```bash
PYTHONPATH=python python -m weft \
  examples/kernels/quantization/mul_mat.py \
  --kernel production_mul_mat_q2_k > /tmp/q2-k.mlir

build/tools/weft-compile/weft-compile /tmp/q2-k.mlir \
  --emit=riscv-ir \
  --march=rv64gcv_zfh_zfhmin_zvfh_zvfhmin_zfa_zba_zbb_zbc_zbs_zicbom_zicboz_zicbop_zicond_zawrs_zihintpause \
  --abi=lp64d --vlen-bits=128 \
  --meta NC=32 --meta MC=16 --meta MR=1 --meta NR=1 \
  --auto-unroll=1 --auto-pipeline-depth=1 \
  -o /tmp/q2-k.riscv.mlir
```

### 2.1 静态 physical op

| op | 数量 | 结构 |
| --- | ---: | --- |
| `rvv_layered_storage_load` | 2 | 两个 128-element half 各一个 |
| `rvv_layered_storage_decode` | 8 | 每个 half 的 4 个 bit layer |
| `rvv_storage_window` | 16 | 每个 half 4 个 activation window + 4 个 scale window |
| `rvv_widen_accumulate` | 8 | 每个 half 4 个 layer update |
| `rvv_finalize_widen_dot` | 2 | 两个 half 分别 finalize |

相关 pass 构造位置在 [`MaterializeRISCVPartialAccumulators.cpp`](../lib/Target/MaterializeRISCVPartialAccumulators.cpp#L575)。

16 个 storage window 按不含 offset 的共同属性分成四组：

| Field | projection | 静态数量 | 差异所在 |
| --- | --- | ---: | --- |
| first-half Q8 q | `(0,1,1,128)` | 4 | offset 分别含 layer base `0/32/64/96` |
| first-half scales | `(0,1,16,128)` | 4 | offset 分别含 layer base `0/32/64/96` |
| second-half Q8 q | `(128,1,1,128)` | 4 | offset 分别含 layer base `0/32/64/96` |
| second-half scales | `(8,1,16,128)` | 4 | offset 分别含 layer base `0/32/64/96` |

这些 op 位于同一个 loop body，但它们不是 CSE 等价表达式。`rvv_storage_window` 的 IR 合同本来就把 `logical_offset` 作为 operand：[`RISCVOps.td`](../include/Weft/Dialect/RISCV/IR/RISCVOps.td#L879)。

### 2.2 SG2044 / VLEN128 的动态工作

每个 128-element half 的 partial loop 执行 4 次；每次处理 4 个 storage layer，每个 layer 8 lanes：

| 工作 | 每 half 动态次数 | 两个 half | 可证明的不同数据 | 可消除上限 |
| --- | ---: | ---: | ---: | ---: |
| raw q byte-window load | 4 | 8 | 8 | 0 |
| q layer decode | 16 | 32 | 32 个 layer/window 组合 | 0 |
| Q8 activation load | 16 | 32 | 32 个不同 K chunk | 0 |
| scale load/broadcast | 16 | 32 | 16 个 scale | 16 次 materialize |
| widened MAC | 16 | 32 | 覆盖全部 256 个 logical products | 0 |
| final reduction | 1 | 2 | 两个作者定义的 half | 0 |

scale 的重复来自映射公式，而不是相同 offset：

```text
source_scale = floor(logical_offset / 16)
```

8-lane window 的 iteration 0/1 使用同一组 scale，iteration 2/3 使用下一组。因此这是“跨 iteration 的二版本 cache/schedule”机会，不是同 block SSA CSE，也不能把四个 layer 的不同 scale 合成一个相同 window。

即使把 scale 的重复 materialization 全部消除，activation、q decode、q×scale multiply、widened MAC 和两次 finalize 都不变。它不是当前 18.6% source 比值的主要解释。

### 2.3 K1 / VLEN256 的动态工作

K1 使用 16 lanes、MR2×NR1：

- 每个 half 的 partial loop只有 2 次；
- 8 次 scale materialization 对应 8 个不同 scale，没有相邻 iteration 重复；
- weight q/scale 没有 M free axis，只生成一个 physical part；
- activation window带 2 个 M register replicas，对两行加载不同 activation；
- `rvv_widen_accumulate` 的 typed operand/result layout将同一 weight q/scale part投影给两个 output accumulators；这是最终 IR 的静态关系，不是另一次独立硬件计数。

因此 K1 已经拥有本轮假设要求的跨 output weight/scale reuse。它仍只有 source 的 68.9%，说明剩余差距不在“同一个 scale 被两个 output重复 materialize”。

## 3. TQ2_0、Q5_K 与 IQ4 的对照

### 3.1 TQ2_0

TQ2_0 的最终 IR 是：

```text
1 layered storage load
4 activation storage windows
4 widened partial updates
1 final reduction
```

四个 activation window具有不同 logical offset；raw packed load 已被四个 layer共享。数值 scale 位于 contraction 之后，不在 partial loop 中重复 materialize。因此通用 read-CSE 对这条已经达到 source 2.86× 的路径同样没有可删除对象。

### 3.2 Q5_K

Q5_K 的 q、qh、scale、activation 都各自只有一个 SSA field producer：

- q 与 qh 虽来自同一 encoded record/point，但 storage width、group/layer geometry不同；
- scale/min 使用 joined storage relation；
- activation 来自另一个 record。

本轮静态生成的 C 中，一个 q/qh load结果已经服务四个 consumer。不存在两个完整 `Field + Point + Access + projection + offset` 相同的 producer。

### 3.3 IQ4_XS / IQ4_NL

- IQ4_XS 只有一个 typed codebook lookup producer；本轮静态生成的 C 中 codebook只加载一次。
- `scales_l`、`scales_h`、q payload和 activation的 storage relation不同，不能 CSE。
- IQ4_NL 的同一 raw byte producer已经在生成 C 中服务多个 shift/mask consumer；不同 layer decode 具有不同数值语义。

这些路径仍有“typed sharing 没完全写回 physical IR”的架构缺口，但当前生成代码已经共享单一 SSA producer。把相同 C emitter结果重新命名成 shared window不会减少执行工作。

## 4. 为什么 Triton/TileLang 机制不会改变上述计数

### 4.1 Triton backward rematerialization

Triton `RemoveLayoutConversions` 的复用 key 是 `(SSA Value, target encoding)`；只有已有 rematerialized value 支配当前 consumer 才复用：[`RemoveLayoutConversions.cpp`](../../ref/triton/lib/Dialect/TritonGPU/Transforms/RemoveLayoutConversions.cpp#L895)。

它是 per-use、dominance-aware 的 representation reuse，不会把 logical offset 不同的两个 load视为同一个值。其 backward slice按 `canBeRemat` 和 slice 条件拒绝 expensive load、dot 等不能安全重物化的 operation，并用 conversion cost 与 rematerialization cost判断是否值得：[`RemoveLayoutConversions.cpp`](../../ref/triton/lib/Dialect/TritonGPU/Transforms/RemoveLayoutConversions.cpp#L934)。

将同样规则应用于 Q2_K：

- raw q load已经是一个 SSA producer；
-四个 q decode layer是四个不同 value；
- activation/scale window的 offset不同；
- 没有可由 `(Value, layout)` map合并的重复 producer。

### 4.2 Triton dot operand allocation

`OptimizeDotOperands` 会把 tensor view 链重写成 `local_alloc + memdesc view + local_load`，但主要是单个 dot operand的局部 representation rewrite；它不把不同地址的 dot operands合并为一个 buffer。

跨 consumer共享时，Triton同样要求所有实际 consumers接受兼容 layout；不同 layout明确判 incompatible。Hopper channel buffer sharing按同一个 producer group共享 allocation，并要求选出的 consumer支配其使用；这仍然以“同一个 producer”为前提。

### 4.3 TileLang reducer lifetime

TileLang 的 reducer plan以同一个 reducer allocation的 data `VarNode*` 为 key：[`reducer_plan_materialize.cc`](../../ref/tilelang/src/transform/reducer_plan_materialize.cc#L130)。一个 epoch 的多个 update site只有在 induced layout和 step完全一致时共享同一个 partial buffer：[`reducer_plan_materialize.cc`](../../ref/tilelang/src/transform/reducer_plan_materialize.cc#L490)。

它不会根据相似 index把两个独立 reducer allocation或不同 update value合并。其共享来自作者已经写出的同一个 reducer epoch，不是一般 load CSE。

## 5. 真正暴露出的两个差距

### 5.1 SG2044：作者数值树不同

当前 Q2_K 作者树是：

```text
scaled_q = widen(q2) * per_sub_scale
partial  = contract(widen(q8), scaled_q, acc=i32)
```

SG2044 GGML VLEN128 donor不是这棵树。它先计算 q2×q8 的 i16 partial，对各 scale group分别 reduction，再把 8 个 scalar partial与 8 个 scale组合：[`quants.c`](../source/c/ggml/llama.cpp/ggml/src/ggml-cpu/arch/riscv/quants.c#L692)、[`quants.c`](../source/c/ggml/llama.cpp/ggml/src/ggml-cpu/arch/riscv/quants.c#L747)。

这改变中间值集合、scale 的结合位置、widening 和溢出边界。按 spec 2.2，它是另一棵作者 std 特化；reuse、LICM 或 scheduler不能把当前树改成 donor。

### 5.2 K1：物理 lane/partial 组织不同

K1 GGML VLEN256 donor与当前树的数值结合更接近，但物理组织不同：

- donor用 32-lane q/q8 vectors；
- scale/aux先以 16 个元素加载，再用 32-lane gather构造四个 scale vectors；
-四个 q plane分别形成 i16 product；
-四个 i32 product用两次最终 reduction组合。

对应代码见 [`quants.c`](../source/c/ggml/llama.cpp/ggml/src/ggml-cpu/arch/riscv/quants.c#L847) 与 [`quants.c`](../source/c/ggml/llama.cpp/ggml/src/ggml-cpu/arch/riscv/quants.c#L896)。

当前 K1 production使用 16-lane partial和MR2 output replicas。最终 IR 已将同一 weight part映射给两个 output replicas；剩余差距应归到 lane/partial/resource组织，而不是缺少跨 output SSA producer。

## 6. 本轮决定

本轮没有修改 compiler、std tree、emitter或性能 CSV，也没有运行新的性能数字。

原因不是“复用不值得做”，而是当前四个靶子的完整 physical identity没有支持本轮拟议 pass的输入：

- 一个合法的 read-CSE会删除 0 个 window/decode；
- 跨 loop 的 scale cache只在 SG 有 2× 的局部上限，K1 为 0；
- Q5/IQ 的单一 SSA producer已经在生成 C 中跨 consumer复用；
- SG 的主差距是作者树边界，K1 的主差距是物理 lane/partial组织。

因此若本轮强行加入 storage-read CSE、扩大所有 window lifetime，得到的只会是一个不改变生成代码的假 pass；若忽略 logical offset做合并，则会生成错误程序。
