# Projected layered window：Q2_K 闭合、跨格式覆盖与负向性能结果

## 1. 本轮范围与结论

本轮只补一项 RISC-V 物理能力：让被规则投影过的 encoded `Field` 在进入局部缩并时，仍以真实 typed IR 同时保留：

- 原始 encoded `Field`；
- typed `PhysicalPoint`；
- reduction axis；
- 投影的 `base / stride / repeat / extent`；
- `grouped_layered` storage geometry；
- 已确定的 RVV layout、resource 和 leaf。

这项能力已经让 Q2_K 从此前的普通 projected extract 变成完整 partial program。最终 RISC-V IR 中出现：

```text
2  rvv_layered_storage_load
16 rvv_storage_window
8  rvv_widen_accumulate
2  rvv_finalize_widen_dot
```

它在 SG2044/VLEN128 和 K1/VLEN256 上都能生成、交叉编译、真机运行并通过浮点容差。

但是性能结果是负向的：SG 从 2.136 降至 1.763 GOP/s，K1 从 1.823 降至 1.660 GOP/s。事实说明此前“只要形成 projected window，K/IQ 大多数格式就会自然得到 TQ2_0 的高性能 partial program”这一推断不成立。Q2_K 虽然已经得到完整 partial IR，但当前作者值图仍把两个 128-element half 分别 finalize，且每个 window 仍重复形成 scale、payload 和 activation 的局部链；它没有得到 donor 的跨 window、跨输出共享与一次最终 reduction。

因此，本轮闭合了缺失的物理表示，却没有闭合高性能 partial program 的全部条件。

## 2. 对照参考实现后采用的机制

### 2.1 Triton

Triton 并不把被投影后的值退化成一个无来源的普通 load：

- `SliceEncodingAttr` 把“消去一个逻辑维度以后，剩余值怎样继承 parent distribution”保存在 type encoding 中：[`TritonGPUAttrDefs.td`](../../ref/triton/include/triton/Dialect/TritonGPU/IR/TritonGPUAttrDefs.td#L1381)。
- `memdesc_index` 与 `memdesc_subslice` 是真实的物理 view operation，而不是 emitter 旁表：[`TritonGPUOps.td`](../../ref/triton/include/triton/Dialect/TritonGPU/IR/TritonGPUOps.td#L218)。
- dot operand 优化把 local allocation、memdesc view 和 local load 写回 IR，后续 pass 消费改写后的程序，而不是在最终 emission 时重新识别完整源码闭包。

本轮没有复制 Triton 的 GPU storage hierarchy；采用的是同一条机制原则：投影关系必须成为 typed physical operation 的 operand、attribute 和 result type，后续 partial pass 才能消费它。

### 2.2 TileLang

TileLang 的 reducer materialization 从 update site 的 reduction axes 计算 partial layout；`ComputeReducerLayout` 在消去 reduction dimension 时仍保留剩余物理分布关系：[`reducer_plan_materialize.cc`](../../ref/tilelang/src/transform/reducer_plan_materialize.cc#L565)、[`reduce.h`](../../ref/tilelang/src/backend/common/op/reduce.h#L57)。

这支持了本轮的边界：Q2_K 的 reduction axis 和数值树来自作者程序；编译器只把现有 Field/Point/axis relation 投影成 window，不从重复 SSA operation 发明新轴，也不做分配律改写。

## 3. 实现

### 3.1 真实 physical op

[`RISCVOps.td`](../include/Weft/Dialect/RISCV/IR/RISCVOps.td#L983) 新增 `weft_riscv.rvv_projected_layered_stream`：

```text
field              原始 encoded Field
origin             typed PhysicalPoint
reduction_axis     被局部缩并消费的逻辑轴
projection_base
projection_stride
projection_repeat
projection_extent
access             grouped/layered storage relation
leaf               已闭合的局部硬件操作
result             带完整 axis/layout 的 physical Value
```

`rvv_storage_window` 同样携带完整 projection；`rvv_layered_storage_load` 携带 projection base/extent。这样 materialization 后没有任何地址关系需要由 emitter 从格式或周围 operation 重新猜测。

verifier 检查：

- Field、Point 与 reduction axis 一致；
- projection 不越过 source extent；
- projection 和 group 边界对齐；
- typed point 的 partition 必须是 group 的整数倍，因而这个 window 不会从一个 storage group 的中间开始；
- `time × lane = projected extent`；
- layer 数、element bit width 和 byte window 闭合；
- free axes 保持为 register replicas；
- leaf、result layout 和 resource group 一致。

对应实现位于 [`RISCVDialect.cpp`](../lib/Dialect/RISCV/IR/RISCVDialect.cpp#L2857) 与 [`RISCVDialect.cpp`](../lib/Dialect/RISCV/IR/RISCVDialect.cpp#L3402)。

### 3.2 从 use-def 形成 projected stream

[`ShareRISCVLayeredWindows.cpp`](../lib/Target/ShareRISCVLayeredWindows.cpp#L234) 读取 typed extract 的：

```text
Field
PhysicalPoint
axis
index_pattern = [base, stride, repeat]
AccessAttr(group, layer, order, bit offset)
ValueType(shape, axes, time/lane/register layout)
```

只有这些关系能证明一个完整 grouped/layered projection 时才建立 projected stream。目前成立的核心条件是规则、连续、group-aligned 的投影；不匹配时保留原 IR，不静默伪造 window。若通向 Field 的 representation conversion 带有 `local_load` read effect，还必须证明从该 read 到 extract 之间没有不可跨越的 effect，不能通过剥 conversion 绕过真实内存读取。

[`RISCVPhysicalSupport.cpp`](../lib/Target/RISCVPhysicalSupport.cpp#L883) 沿 Field owner、typed extract index 和 representation conversion 找到同一逻辑轴的 `PhysicalPoint`。它不是 source location 字符串，也不是 emitter 私有 provenance。

### 3.3 partial materialization

[`MaterializeRISCVPartialAccumulators.cpp`](../lib/Target/MaterializeRISCVPartialAccumulators.cpp#L67) 将一个缩并 operand 的 roots 表示为 `ProjectedRoot`。每个 root 独立保存 Field、Point、Access 和 projection；同一 window iteration 中用 root→window replacement map 克隆局部 widen/multiply 链。

Q2_K 的两个 128-element half 分别得到：

```text
projected q payload
projected per-sub scale
projected Q8 activation
    -> per-window widen/multiply
    -> four layered widened MAC updates
    -> one i32 lane reduction
```

TQ2_0 原有 i8→i16 partial 路径保持不变。Q2_K 的 operand 已经是 i16，因此 partial type 和 finalize 被推广为 i16×i16→i32 partial，再用普通 i32 lane reduction；verifier 不允许 emitter 临时改变 accumulator width。

### 3.4 emission

emitter 只读取已经确定的 projection：

```text
source_index = projection_base
             + floor(logical_lane / projection_repeat) * projection_stride
```

- `repeat=1,stride=1` 生成连续 load；
- `repeat=1,stride>1` 生成 strided load；
- repeated projection 生成分段 broadcast/拼接；
- free-axis replicas 各自使用 type 中已有的 record coordinates；
- layered payload 的 projection base 在地址计算中显式加入。

实际 SG 汇编中，Clang 将 Q2 scale 的重复 load 写成了 stride-zero `vlse8.v`；这是对已确定 projection 的指令拼写，不是 emitter 对格式的选择。

## 4. 静态覆盖

统一 SG2044 profile 生成最终 RISC-V IR，统计结果如下：

| 格式 | layered load | storage window | partial update | finalize | 观察结果 |
| --- | ---: | ---: | ---: | ---: | --- |
| Q2_K | 2 | 16 | 8 | 2 | 新 projected relation 被完整消费 |
| TQ2_0 | 1 | 4 | 4 | 1 | 原有完整 partial program 保持 |
| Q5_K | 0 | 0 | 0 | 0 | q/qh 是两个 domain-projected layered roots，layer geometry 不同；不是单一规则 projection |
| IQ4_XS | 0 | 0 | 0 | 0 | q 是 domain projection，scale 包含 stride-2 relation，lookup index 来自 payload value |
| IQ4_NL | 0 | 0 | 0 | 0 | packed operand可形成普通 layered stream，但 codebook lookup 不是 storage-window root |

其余 K/IQ 中还有两类不会被本轮能力触发：

1. Q3_K、Q6_K、IQ1、IQ2、IQ3 的当前树没有保留 shaped bit-plane/grid/sign/codebook axis；pass 不能从 Python 展开的多个值重新发明一条逻辑轴。
2. Q5_K 与 IQ4 系列虽然有 shaped axes，但需要多 layered roots 的联合 window、joined high plane 或 lookup-aware window sharing；这些不是“一个 projected Field”的同义写法。

因此，本轮外部覆盖证据是：同一 typed projected mechanism 跨 VLEN 覆盖了 Q2_K，并保持 TQ2_0；它没有横向覆盖 Q5_K 或 IQ4。没有把这些负结果改写成格式分支或更宽松的 verifier。

静态 repro（下列命令展示 SG/VLEN128；K1/VLEN256 使用同一 kernel 与 pass 流水，只替换 target profile。跨 VLEN 的运行证据见第 5 节两台真机结果）：

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

## 5. 真机结果

全部结果均为 production shape `M=128,N=4096,K=4096`、Clang 18、10 次重复，且 Weft kernel、runtime wrapper 和 GGML reference 使用相同编译选项。GGML source 数字来自固定 baseline 记录：[`ggml-riscv-kernel-baseline.md`](baseline/ggml-riscv-kernel-baseline.md#L74)。

### 5.1 Q2_K

| target | 本轮前 | projected partial | GGML source | 本轮/source | 数值 |
| --- | ---: | ---: | ---: | ---: | --- |
| SG2044 / VLEN128 | 2.136 | 1.763 | 9.462 | 18.6% | abs `3.576e-7`, rel `1.658e-5` |
| K1 / VLEN256 | 1.823 | 1.660 | 2.410 | 68.9% | abs `3.576e-7`, rel `1.658e-5` |

负向结果把缺口进一步限定为：

- 旧程序的 248 个 widened MAC / 24 次 reduction 已被压成 typed partial chain，但当前仍是两个 half、两次 final reduction；
- 每个 partial window仍各自 materialize scale、decoded payload 和 activation；
- SG 热 partial loop中没有发现决定性的 spill，差距不是“资源 pass 错算导致落栈”；
- Clang 确实生成了 vector load、widen multiply、`vwmacc` 和最终 `vredsum`，不是形态被下游编译器标量化；
- donor 的四 plane/window 组织、跨 window scale reuse、跨 output reuse 和一次最终 reduction，并不会仅由 projected Field relation自动产生。

这不是继续调整 LMUL 或 emitter 指令拼写就能解释的差距。

Q2_K 新作者树按要求保留。对同一棵新作者树，本轮 projected partial 相比上轮的 2.136/1.823 GOP/s 在两机都退化。若跨作者树比较，更早的旧树在 SG/K1 分别是 3.338/1.239 GOP/s：新树只在 K1 占优。结合 SG/K1 donor 本来采用不同 partial 数值组织，这构成两台机器需要两个作者 std 特化的证据；历史树与数字来源见 [`riscv-typed-mechanism-coverage-q2-tree-and-quant-spread.md`](riscv-typed-mechanism-coverage-q2-tree-and-quant-spread.md#L114)。本轮没有替作者改树。

### 5.2 TQ2_0 回归

| target | 本轮前 | 本轮 | 数值 |
| --- | ---: | ---: | --- |
| SG2044 | 19.977 | 20.475 GOP/s | exact |
| K1 | 6.245 | 6.419 GOP/s | exact |

这说明多 root clone、i32 partial 与 projection attrs 没有破坏已有的 TQ2_0 partial program。

### 5.3 canonical Q4_K 恢复

错误地把 staged blocked tree 设为 canonical Q4_K 默认入口后，SG/K1 分别退化到 1.489/1.383 GOP/s。本轮恢复旧 canonical row/column tree，并删除默认 staged meta：

| target | 退化版本 | 恢复结果 | 数值 |
| --- | ---: | ---: | --- |
| SG2044 | 1.489 | 5.100 GOP/s | abs `0.3125`, rel `4.419e-4` |
| K1 | 1.383 | 1.957 GOP/s | abs `0.3125`, rel `4.419e-4` |

独立 `q4_k_staged` 与 `q4_k_persistent` 入口保留；这里只恢复 canonical `q4_k` 的程序与 runner 选择。

### 5.4 SG 已过线路径的一次回归观察

以下仅作一次运行的退化检查，不写入 10 次重复 CSV：

| case | GOP/s |
| --- | ---: |
| Q4_0 | 8.602 |
| Q4_1 | 6.703 |
| Q5_1 | 6.463 |
| IQ4_NL | 8.623 |
| Q1_0 | 6.903 |
| Q4_K persistent | 10.133 |

六条均通过数值容差，未观察到相对于各自上一轮结果的实质退化。

## 6. 最终边界

本轮得到的不是“多数 K/IQ 已获得高性能 partial”，而是更窄且可复现的结论：

1. encoded Field、typed Point、regular projection 和 layered geometry 现在能作为真实 physical IR 一起存活，并被 partial materialization消费。
2. 这足以把 Q2_K 从普通 extract 变成跨 VLEN 的合法 partial program。
3. 它不足以自动生成 donor 的跨 window/跨 output reuse，也不足以覆盖 joined high plane、payload-derived lookup 或未显式 shaped 的作者树。
4. Q2_K 的负向性能说明 TQ2_0 的高性能不只来自“存在 partial op”；还来自它的作者值图恰好允许四个 storage window进入一个共享 partial lifetime。
5. canonical Q4_K 已恢复；Q2_K 当前树保持不动。SG 与 K1 的不同 donor 数值组织应作为作者特化边界处理，而不是由 compiler根据 target暗改同一棵树。

当前性能数据已写入 [`weft-kernel-performance.csv`](weft-kernel-performance.csv)。
