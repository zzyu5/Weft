# RISC-V reduction-lane microkernel 与 blocked quant 性能修复

## 1. 本轮范围

本轮处理上一份 206 条结果中已经暴露、且可由同一个物理编译缺口解释的
blocked quant prefill 低性能问题。修改集中在 RISC-V 第二层 IR 的 operation
selection、layout propagation、typed conversion、composite lowering 和 resource
legality；没有修改 quant 数值树、Encoding、Level 归属或 kernel ABI。

本轮没有尝试把仍为 row-by-column 或 Python 标量展开的 std 函数伪装成 blocked
程序，也没有建设新的 whole-kernel leaf。正式性能更新只覆盖确实受到本轮修改的
10 个 blocked quant prefill、两个 target，共 20 行；其余 CSV 行保持上次各自的
真实测量值。

## 2. 实际病灶

Q5_K 的作者树已经具有：

- `NC/MC` output tile；
- `MR/NR` accumulator；
- 256-element K block 与 32-element sub-Level；
- activation 跨 output cohort 复用；
- scale 与 minimum 两条数值支路。

旧 physical IR 却把 output free axis 放进 RVV lane，把 reduction K 留成 issue-time
parts。结果是每个 K part 都重新生成 packed load、shift、merge 和小向量 MAC。
这不是作者树缺 blocking，而是编译器把同一棵树物理化错了。

这个现象也出现在 Q1_0、Q4_0/Q4_1、Q5_0/Q5_1、Q8_0、Q2_K、IQ4_NL 和
TQ2_0。格式的 decode 关系不同，但共同的 typed relation 是：

```text
[M,K] × [N,K] --reduce K--> [M,N]
```

对 RVV widening reduction，K 应进入 lane；M/N free axes应成为 register replicas。
旧实现只对 scalar-output dot 建立了这条关系，`outer_contract` 没有进入同一个
widen-dot lowering。

## 3. 编译器修改

### 3.1 从 operation relation 选择 reduction-lane widening-dot

`SelectRISCVOperations` 现在允许满足以下 typed 条件的 `outer_contract` 使用现有
`rvv.vwmul-vwredsum` realization：

- 两个整数 operand 的有效 SEW 对应；
- 至少一侧有符号；
- result 是 signed i32；
- 恰有一个 reduction axis；
- target 支持 widening integer；
- 当前局部 result microtile 的最低 register budget 可成立。

这里没有读取 kernel 名或格式名。选择发生在 layout propagation 之前，因为
reduction axis 必须先成为 operand layout 的 anchor。最终 LMUL、accumulator 和
temporary resource 仍在 physical layout 形成后再次校验。

资源预筛选也保留了必要的负例：Q4_K staged 的 `[MR,16]` 大 cohort 不会被误送到
small-register-microtile widening-dot，仍走其原有 local encoded realization。

### 3.2 free axes 形成 register tuple

`PropagateRISCVLayouts` 对 widening `outer_contract` 按以下关系赋形：

```text
operand 中的 reduction axis -> lane
operand/result 中剩余的小 free axes -> register replicas
result 不含 reduction axis -> scalar register tuple
```

这使 `[MR,NR]` integer result 不再先变成 N-lane 小向量再逐 K part 更新，而是
由多个完整 K-lane widening reduction 直接形成 MR×NR register tuple。

### 3.3 两种真实 typed conversion

结果进入 scale/minimum/pointwise 支路时，consumer 可能需要不同表示。本轮没有让
emitter 猜如何交接，而是在 `weft_riscv.convert_layout` 上闭合两种真实转换：

1. `register_to_lane`
   - source 的一个 register-replica axis 变成 target 的唯一 RVV lane axis；
   - verifier 检查 axis identity、完整因子守恒、其余轴不变、fragment/local factor
     均不参与；
   - emitter 只按已选 layout 用 `vmv/vfmv + slide1up` 拼成 RVV value。

2. `time_to_lane`
   - encoded field 在 source 中以 issue-time parts 表示同一逻辑轴，consumer 需要把
     这些 parts 直接按 lane 读取；
   - verifier 检查 `source_time == target_time × target_lane`，并要求其余逻辑轴的
     time/replica factor 原样保持；
   - layout canonicalization 必须把该 conversion 后向重物化到真实 encoded Field edge
     （含仍可追溯到 Field 的 Extract），final verifier 拒绝仍停在一般 computed
     scalar 或普通 record extract 上的 conversion；
   - Field/Slice 由既有 typed memory materialization 直接生成 target layout。

Q1_0 是第二种转换的外部反例。若只放宽 `splat`，它会在 terminal emission 报
“RVV splat requires one scalar input”；加入 `time_to_lane` 后，SG2044 与 K1 都恢复
数值正确。这说明缺的是一个物理 operation，不是 verifier 应该容忍错误 IR。

### 3.4 encoded layer width 对两种 VLEN 都生效

indexed memory 只表示 target 能执行 gather，不表示一个 encoded layer 可以被扩成
任意宽度。`encodedLaneLimit` 现在在 SG2044/VLEN128 与 K1/VLEN256 上都参与 lane
capacity 求交。IQ4_NL 在 K1 的同配置测量从约 1.676 提升到 1.749 GOP/s；最终配合
MR/NR 参数选择达到 1.976 GOP/s。

### 3.5 先拒绝错误 candidate

此前 Q5_K `NR=8/16` 会通过 candidate pipeline，但输出数值错误。原因是 computed
`outer_contract` result 同时具有 lane factor 与多个 time strips，而 compiler 没有
任何 typed handoff 证明这些 strips 如何对应。

本轮把该条件变成 lowering legality：没有 encoded access 或显式 conversion 证明时，
candidate 在 emission 前失败：

```text
computed outer-contract lane operand has no proven multi-strip handoff
```

`NR=4` 等有闭合表示的 candidate 继续工作；Q4_K staged 的直接 encoded Field access
也继续合法。错误 candidate 不再以“较慢候选”身份混入 tuner。

## 4. 参数实测

MR/NR 是 std 中已有的 `auto` 参数，由实测选择，不由 physical pass 猜测。

### 4.1 Q4_0

| target | MR×NR | GOP/s |
|---|---:|---:|
| SG2044 | 1×8 | 2.724 |
| SG2044 | 2×4 | 5.520 |
| SG2044 | 4×2 | 5.818 |
| SG2044 | 8×1 | **7.141** |
| K1 | 1×8 | 1.057 |
| K1 | 2×4 | 1.767 |
| K1 | 4×2 | **1.859** |
| K1 | 8×1 | 1.760 |

正式 runner 因此使用 SG `MR=8,NR=1`、K1 `MR=4,NR=2`。这同时证明 VLEN 改变后
最优参数并非只把 lane 数翻倍。

### 4.2 跨格式检查

| format | target | 2×4 | 4×2 | 8×1 | 采用 |
|---|---|---:|---:|---:|---:|
| Q5_K | SG | 6.967 | 6.988 | illegal | 2×4 |
| Q5_K | K1 | 2.359 | 2.260 | illegal | 2×4 |
| IQ4_NL | SG | 4.766 | **6.045** | 6.020 | 4×2 |
| IQ4_NL | K1 | 1.749 | **1.977** | 1.729 | 4×2 |

Q5_K 的 SG 4×2 只快约 0.3%，K1 反而更慢，因此 runner 保留 2×4。IQ4_NL 在两台
机器上都由 4×2 获益。Q1_0 的 4×2 首先暴露了缺失的 `time_to_lane`；闭合后它在
SG/K1 已可正确运行，但只有 1.964/0.907 GOP/s，明显慢于 2×4 的 3.376/1.828。
因此 runner 保留 2×4，而没有把一个参数 winner 硬套到所有格式。

## 5. 双机正式结果

共同口径：

- production shape：`M=128,N=4096,K=4096`；
- full projection，包含 invocation-local activation quantize；
- Clang 18，`-O3 -ffp-contract=fast`，单核；
- 64 MiB cache eviction；
- 10 次 cold median；
- 浮点结果使用现行绝对/相对容差。

完整当前值已写入 `report/weft-kernel-performance.csv`。

### 5.1 SG2044 / VLEN128

| format | 上次 GOP/s | 本轮 GOP/s | 提升 | max abs / rel |
|---|---:|---:|---:|---:|
| Q1_0 | 0.762 | 3.376 | 4.43× | 4.44e-2 / 1.34e-5 |
| Q4_0 | 0.501 | 7.113 | 14.20× | 0 / 0 |
| Q4_1 | 0.617 | 4.711 | 7.64× | 0 / 0 |
| Q5_0 | 0.465 | 6.379 | 13.72× | 0 / 0 |
| Q5_1 | 0.584 | 3.984 | 6.82× | 0 / 0 |
| Q8_0 | 1.490 | 10.767 | 7.23× | 0 / 0 |
| Q2_K | 1.536 | 3.197 | 2.08× | 1.19e-6 / 1.31e-5 |
| Q5_K | 0.345 | 6.968 | 20.18× | 1.91e-5 / 7.55e-6 |
| IQ4_NL | 0.659 | 6.049 | 9.17× | 0 / 0 |
| TQ2_0 | 0.178 | 1.133 | 6.36× | 0 / 0 |

当轮同命令的 SG source 诊断值显示：Q4_0 为 source 的 93.9%，Q4_1 74.9%，Q5_0
119.0%，Q5_1 87.6%，Q8_0 268.4%，IQ4_NL 91.9%。Q1_0、Q2_K、TQ2_0 仍只有
约 29.6%、33.0%、9.4%，说明 reduction-lane 修复没有消除它们各自的 storage/decode
开销。

### 5.2 K1 / VLEN256

| format | 上次 GOP/s | 本轮 GOP/s | 提升 | max abs / rel |
|---|---:|---:|---:|---:|
| Q1_0 | 0.508 | 1.828 | 3.60× | 4.44e-2 / 1.34e-5 |
| Q4_0 | 0.368 | 1.858 | 5.05× | 0 / 0 |
| Q4_1 | 0.441 | 1.501 | 3.41× | 0 / 0 |
| Q5_0 | 0.294 | 1.778 | 6.04× | 0 / 0 |
| Q5_1 | 0.374 | 1.558 | 4.16× | 0 / 0 |
| Q8_0 | 0.840 | 2.271 | 2.70× | 0 / 0 |
| Q2_K | 0.891 | 1.202 | 1.35× | 1.19e-6 / 1.31e-5 |
| Q5_K | 0.277 | 2.361 | 8.52× | 1.91e-5 / 7.55e-6 |
| IQ4_NL | 0.447 | 1.976 | 4.42× | 0 / 0 |
| TQ2_0 | 0.143 | 2.312 | 16.18× | 0 / 0 |

K1 的固定 baseline 中 Q4_0/Q4_1 使用 IME1 asm，而本轮 Weft 仍是 RVV，因此这两行
不能用 ratio 评价同一 realization。已有同引擎参照中，Q5_0 达到固定 RVV baseline
的约 96.1%，Q5_K 达到约 188.8%。

本轮额外重新调用当前 K1 source build 时，runtime 报告的 implementation 是
`rvv_quantized_vec_dot`，测得值与固定 baseline（特别是 IME 行）发生数量级漂移。
这些诊断值没有写回 benchmark authority，也没有用来重算 K1 ratio。

## 6. 交叉回归

以下不同结构在最终代码上继续真机运行：

- SG F16 decode/prefill：5.227 / 19.441 GOP/s，数值在容差内；
- K1 F16 decode/prefill：3.319 / 5.857 GOP/s，数值在容差内；
- SG/K1 F32 prefill：7.236 / 2.503 GOP/s，误差为 0；
- SG/K1 Q4_K staged prefill：1.485 / 1.383 GOP/s，误差为 0；
- K1 Q4_K persistent prefill：3.231 GOP/s，误差为 0。

Q5_K 最终 SG 手工 repro 为 6.937 GOP/s，Q1_0 最终 K1 手工 repro 为
1.829 GOP/s；它们分别覆盖 register-tuple reduction 与 encoded time-to-lane conversion。

## 7. 仍然存在的具体缺口

### 7.1 SG Q4_K persistent resource peak

SG 的 Q4_K persistent 派生路径当前在 resource pass 报：

```text
34 vector groups required, target budget 32
```

两个 i32[16] reload 在最终 consumer 前确实同时 live，每个占 4 groups；interval 的
结束点没有错误延长，继续 spill reload 也不能降低该 issue 点峰值。把
`encodedLaneLimit` 暂时恢复旧条件后错误不变，说明它不是本轮 storage-width 修改造成。
K1 因 VLEN256 下相同 16-lane i32 value 占用更少 group，路径仍可执行。

缺口是 SG 上 correction expression 的物理 schedule/issue decomposition：当前必须同时
持有两组 reload 与其余 live values。不能通过放宽 32-register budget、吞掉错误或让
emitter 临时拆 expression 解决。本轮未把该独立问题伪装成已闭合。

### 7.2 仍为标量/row-column 的 std 树

本轮改进只会消费作者已经写出的 shaped `outer_contract`。Q3_K、canonical Q4_K、
Q6_K、IQ1、IQ2、IQ3、IQ4_XS、TQ1_0、MXFP4/NVFP4 等仍有 row-by-column 或
scalar-expanded production tree；普通标量程序不会被 compiler 猜成 blocked program。
因此不能从这 20 行推导新的 206 条总体分布。

### 7.3 storage 与 local schedule

Q1_0、Q2_K、TQ2_0 的正式结果虽分别提升 4.43×、2.08×、6.36×，SG 上仍明显落后
source。剩余生成代码主要开销已不再是 K scalar MAC，而是 logical sub-byte/radix
projection、address/index construction 和未形成 steady-state 的 decode/load cluster。
这是下一类 compiler representation/schedule 缺口，不是再扫 MR/NR 可以解决。

## 8. 边界判断

本轮没有改变任何 logical value 集合或 Level 归属；相同的 blocked std 树获得了新的
K-lane/register-replica 物理表示。MR/NR 的变化来自既有 `auto` 参数实测，错误的
NR candidate 被 legality 拒绝。因此这些结果没有要求 compiler 改作者树，也没有
证伪 spec 2.2。

反过来，仍为标量或 row-by-column 的程序没有自动获得 blocked 实现。这是同一边界
的另一面：本轮证明的是 shaped tree 的 physical compiler 缺口可以横向修复，不是
普通 SSA 可以自动恢复算法 blocking。

## 9. 手工复现

```bash
ninja -C build weft-compile -j4
./examples/run/weft-mul-mat.sh sg2044 q5_k prefill 10
./examples/run/weft-mul-mat.sh k1 q1_0 prefill 10
./examples/run/weft-mul-mat.sh sg2044 iq4_nl prefill 10
```

Q5_K 非法多-strip candidate 的可见拒绝：

```bash
WEFT_META_BINDINGS='NC=32;MC=16;MR=2;NR=8' \
WEFT_AUTO_UNROLL=1 WEFT_AUTO_PIPELINE_DEPTH=1 \
WEFT_AUTO_LMUL_EIGHTHS=8 \
./examples/run/weft-mul-mat.sh sg2044 q5_k prefill 1
```

预期在 intrinsic C emission 前报告：

```text
computed outer-contract lane operand has no proven multi-strip handoff
```
