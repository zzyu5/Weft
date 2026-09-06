# 读取值语义、物理闭合与调优绑定复核

2026-09-06。本轮在 DSL 表面更新之后修改真实编译器、作者程序与 runner 配置，并完成受影响项和此前过线项的双机定向运行。数值验收通过，但性能验收没有全部通过：151 项中，30 项比原对照表下降超过 5%，12 项由高于 source 变为低于 source。本文保存正反结果，不把未闭合的性能问题写成完成。

## 1. 范围与证据

上一轮完成的是 DSL 表面重构，不是所有后端能力：111 份 source 全部生成并验证 canonical IR，105 份除位置外相同，其余 6 份只有明确对应的 7 个 op 拼写变化；227 个 runner 请求中，217 个实际数值通过，10 个保持原有编译拒绝。公开入口是 `dot`，不是 `dot2D`。原始范围见 [DSL 表面报告](dsl-surface-renaming-and-read-order.md)。本轮不改写该历史报告。

本轮最终证据：

- [逐项运行记录](read-value-physical-closure-runs.csv)：151 项，SG2044 70 项、K1/X60 81 项，保存 source/physical bindings、input policy、数值误差、吞吐、旧值、source 值及本地/远端 artifact 路径。
- [性能对照表](kernel-performance-comparison.csv)：只更新这 151 行，其余 51 行保持原值；`weft-kernel-performance.csv` 未修改。
- `/tmp/weft-closure-regression.cHyRjn/verified-final.json` 与 `final-summary.json`：最终选取的完整运行记录和汇总。
- `/tmp/weft-closure-regression.cHyRjn/terminal-code-audit.json`：用最终编译器重建 151 份 C，全部成功。最后两份变化的 IQ2_XS decode 已重新实际运行；最终生成 C 与这 151 份实际执行 C 全部逐字节相同，没有用摘要校验替代比较。
- `/tmp/weft-read-closure.HVGWnF/`：作者树比较、读取顺序 repro、重放、环境与中间失败记录。临时目录不属于持久归档，仓库 CSV 保存运行结果和定位信息。

最终 151 项中，147 项按现有 `within-tolerance` policy 通过、4 项 `bit-exact`；29 项有非零浮点误差，不能称为全部零误差。IQ1_M/TQ1_0 本轮最终 standalone、decode、prefill 的记录均为零误差。未放宽 runtime 容差或输入生成 policy。

## 2. 七步复核与实际修改

诊断顺序遵循 [optimization-principles 第九节](../doc/compiler/optimization-principles.md#9-可复用诊断动作)。参考 Triton 的 layout conversion、autotune 配置归属与 TileLang 的 layout 候选处理；具体量化数值树另与 GGML RISC-V donor 逐段比较。参考目录保持只读。

| 动作 | 本轮观察与处理 |
|---|---|
| axis/time/lane/replica 与资源 | 修复 primary lane 在 coalesced-axis 遍历中重复计数；同宽整数 reinterpret 保持完整 layout；IQ1_M prefill 的 34/36 register-group 拒绝没有通过放宽 32-group 限制绕过，而是允许原位只读结果存入 spill。 |
| source origin、坐标、动态工作 | 不把静态 leaf 数当作执行次数。移除满足单消费者、同块、无干扰写条件的 physical-share，IQ 高位供给的热循环动态 slide 从每块 24 次降为 0；个别常量初始化仍可在循环外有 slide。 |
| supply identity 与复用 | load 代表读取时的值，不能保存为之后才求值的解引用；单次完整 scalar-repeat 供给改为原读取点的 scalar supply，不改作者算术树。IQ2_XS 的 free-axis 单例约束同时覆盖 standalone 和 decode。 |
| memory edge 与地址 | MemDesc lookup 明确有 Read effect；读取延迟跨可能别名写入时，对受支持的有界连续 geometry 在原读取点保存快照，不能重读。寄存器 table lookup 仍为纯值操作。 |
| partial、reduce、extract、spill | 保留小范围 register reduction 的 free-axis 映射；单次 issue window 若投影类型与原值完全相同则复用原值。比较新旧 IQ1_M/TQ1_0 作者树时同时检查 scale 结合、widening、reduction 和实际吞吐。 |
| load-to-use 与 pipeline | 先处理错误延迟读取、重复供给和资源闭合。没有用更深 pipeline 掩盖这些问题；此前失败的 LMUL/pipeline 候选仍作为拒绝记录保存。 |
| donor 与归属 | “剩余全是另一棵数值树”的原判断不成立：仍有通用物理供给与资源问题。物理修复后 IQ1_M 原树胜过批准的新树；TQ1_0 新树只在 SG2044 获选。选择位于 kernel 旁配置，不在编译器按格式或机器名字分支。 |

这些修改仍走 Canonical Kernel IR -> RISC-V Physical IR -> terminal C/asm，未添加 whole-kernel 模板或第三层规划 IR。`doc/` 只同步相应语言/物理合同和验收边界，运行事实保存在本报告。

### 读取点与别名写入

`materializeScalarRead` 让 scalar、tuple/field scalar 和 local/reload scalar 在读取点成为有类型 C 变量；rank-0 dense 读取也进入同一语义。标量反例由错误的 `A=7, B=7` 修正为 `A=7, B=3`。

编码读取不能只修 scalar emitter。`MaterializeRISCVReadSnapshots` 跟踪值链与可能别名写，包括循环先前迭代的写；遇到已经保存值的 register materialization/local bind 边界停止继续推迟。连续、静态、有界且 byte-aligned 的完整记录可以复制到显式 Local storage；不支持的干扰几何明确拒绝。

`DenseSnapshotOp` 补齐 staged dense table 无法作为完整 register table 保存、但推迟读取又跨写的情况。它有明确 Read/Write effect、精确字节数、storage identity 和 target leaf，在原 Load 位置执行，后续 lookup 读私有副本。TQ1_0 prefill 的 powers table 与 NVFP4 table 因此可以在不改变读取语义的前提下编译。重复 table use 复用同一原始读的快照。

只读 producer 的 spill 保持 producer 在原读取点，紧接着保存结果，各 use reload；它不是把读取移动到 use，也没有放宽资源上限。四个手工 repro 的双机结果：

| repro | 实际结果 |
|---|---|
| `scalar_read_order` | `a=7 b=3` |
| `encoded_read_order` | `a=7 b=3` |
| `lookup_read_order` | `before=3 after=7` |
| `dense_table_read_order` | `table_after=7 snapshot_first=3 numeric=exact` |

最终 snapshot transfer 在合法 m1/m2/m4 中按精确 load/store 对数选择，平局取较小 LMUL，最多占 4 个 scratch groups；精确 tail 不越界。所选参数写入 leaf，emitter 只拼写，不重新决定策略。固定 m1 复制曾实测更慢；即使最终 pair-count 更少，也不保证实际吞吐更高，见第六节。非 RVV 的 scalar copy 是 target 显式选择的 leaf，不是运行时 fallback。

## 3. 两棵作者树与实际选择

新增的 IQ1_M lane16 树把 scale 合入 main/correction i32 lane partial，最后保留两次 i32 reduction。对任意 signed-i8 grid 值，单个 i8 product 的绝对值不超过 16384，scale 不超过 15；每个 i32 lane 不超过 `16 * 16384 * 15 = 3,932,160`，完整 main 不超过 62,914,560，correction 不超过 491,520。中间 i16 product 与 i32 累加均不溢出，最后 float conversion 的位置不变。

新 TQ1_0 lane16 树中，radix 解码值在 [-1, 1]，每 lane 共 16 个 signed-i8 乘积；绝对值上界 `16 * 128 = 2048`，单 i16 carrier 足够，完整总和绝对值上界 32768，最终用 i32 widening reduction。没有通过容差消除分组错误。

两棵新树均完成双机 standalone/decode 的零误差运行，但“数值成立”不等于“应该选用”：

- IQ1_M：新树的有界 standalone 扫描最好值为 SG 1.816136、K1 0.930469 GOP/s；较早的 paired 配置也有更慢的零误差结果。最终物理修复后的原树为 SG 2.187488、K1 0.946259。最终两个 target 的 standalone/decode 都保留原树，新树作为明确作者 variant 保存，不强行宣称优化成功。
- TQ1_0：新树在 SG 获选，standalone/decode 共用数值结构；K1 新树约 2.10/2.11 GOP/s，低于原树约 3.21/3.25，K1 保留原树。

最终结果如下，单位均为 GOP/s；source 是对照表中原有固定测量，本轮未重新测 source：

| kernel | target | 原 Weft | 最终 Weft | source | 最终比值 |
|---|---|---:|---:|---:|---:|
| IQ1_M standalone | SG2044 | 1.091291 | 2.187488 | 2.650472 | 0.825320 |
| IQ1_M decode | SG2044 | 2.033534 | 2.191551 | 2.639333 | 0.830343 |
| IQ1_M standalone | K1/X60 | 0.860998 | 0.946259 | 1.435304 | 0.659274 |
| IQ1_M decode | K1/X60 | 0.846799 | 0.929375 | 1.424159 | 0.652578 |
| TQ1_0 standalone | SG2044 | 4.816287 | 5.290342 | 8.123996 | 0.651199 |
| TQ1_0 decode | SG2044 | 4.784427 | 5.357824 | 8.015470 | 0.668435 |
| TQ1_0 standalone | K1/X60 | 3.210671 | 3.211882 | 3.408741 | 0.942249 |
| TQ1_0 decode | K1/X60 | 3.235087 | 3.249594 | 3.338873 | 0.973261 |

这八项仍全部低于各自 source。K1 TQ1_0 的不足 0.5% 变化不能称为稳定收益。prefill 没有切换到 decode 作者树；IQ1_M 的 SG/K1 最终为 2.515547/1.621238，TQ1_0 为 4.579534/4.505762，其中 SG TQ1_0 prefill 明显退步。

## 4. 资源闭合之前的重放

新增 `--emit=riscv-layout-input` 保存 unroll/share 之后、最终 layout 改写和资源闭合之前的 Physical IR。输入带专用 marker 和 scalar-prime 配置，`--resume-layout-input` 验证边界后进入与普通编译相同的唯一收尾后缀，重新处理 layout、memory、leaf、读取快照与完整资源检查。它不清除 final resource marker 假装重放，也不接受替换 target/meta/physical bindings。

`/tmp/weft-read-closure.HVGWnF/terminal-replay.json` 包含旧 IQ1_M/TQ1_0、新 IQ1_M/TQ1_0 各两个 target，共八个样本：

- 只重放 layout canonicalization：八项二次稳定，生成 C 与普通编译相同。
- 重放 `canonicalize-layouts -> cse -> eliminate-dead-layouts`：六项一次/二次相同；旧 IQ1_M 的两项需要二次/三次才相同，不能报告成全部“二次 diff=0”。
- 上述组合重放有四项生成 C 与普通编译不同，分别为旧 IQ1_M 两项、新 TQ1_0 两项。这四份 C 均实际在相应机器运行，最大绝对/相对误差均为 0；最终重建的八份 clean C 也与先前核验 artifact 逐字节相同。

这比 final-resource IR 上被跳过的重放更强，但仍是八个样本、指定流程和实际数值运行的证据，不证明所有 pass 或全部程序等价。IQ1_M SG 的直接编译实测 2.185895，clean 重放 2.170023，不能把物理修复收益归因于额外 CSE。

## 5. 放宽规则与调优绑定

实现中移除了确有 typed/effect 合同依据的限制：单消费者 physical-share、同宽 reinterpret 的完整 layout、bounded register reduction 的 free-axis supply，以及完整 scalar-repeat 到 scalar consumer 的供给关系。IQ3_S 的 sign 在当前 IR 已是完整 byte mask window，不存在先前怀疑的待补 half-byte relation；没有为凑“第二输入”虚构节点。

snapshot copy 使用最多三个合法候选及明确 pair-count 估计，估计不能覆盖 alias、数值、几何或资源合法性；外部 tuner 则对数值验收通过的候选按真实吞吐排序，没有静态成本时明确记录 `estimate=null`。此次没有实现任意结构搜索，也不声称一个正例已经证明跨格式泛化。

迁移后的绑定归属：

| 位置 | 数量与职责 |
|---|---|
| `examples/kernels/dense/tuning.json` | 9 个 runner 请求绑定 |
| `examples/kernels/gemv/tuning.json` | 4 个 |
| `examples/kernels/quantize/tuning.json` | 6 个 |
| `examples/kernels/mul_mat/tuning.json` | 120 个 |
| `examples/kernels/vec_dot/tuning.json` | 48 个 |
| `examples/kernels/dequantize/tuning.json` | 48 个 |
| C++ | target defaults、合法性、已声明物理候选；不是某机器某格式的实测 winner 表 |
| runner | host/compiler/core/ISA 与执行合同；kernel source/meta/physical 绑定从相邻 catalog 解析 |

合计 235 个请求绑定，对应先前 227 个审计请求加 8 个显式 staged-prefill 别名。source kwargs 与 physical bindings 分开，entry/symbol 显式记录。绝大多数标为 `historical-selection`，不冒充一次完整扫描的 winner；本轮有实测作者入口选择证据的条目单独标注。

物理默认搜索域为 LMUL 4 档、unroll 4 档、pipeline/scalar-prime/partial-policy/record-axis 各 2 档，共 256 个候选。默认预算 256，显式最多 1024，source 笛卡尔积也在展开前受限。完整 `search.json`、每项 stdout/stderr、拒绝原因、数值/性能与 `selected.json` 留在仓库外；选中结果不会自动覆盖历史 catalog。

真实正反例：

- `/tmp/weft-tune.ro932w_y`：SG `q8_0_quantize` 的两个候选均 bit-exact，LMUL-eighths 16 的本次最优值 920.219337 MEl/s，显式应用结果后复跑 919.762707，仍 bit-exact。这只是两项域中的本次 winner。
- `/tmp/weft-tune.dtzrbp5l`：读取 spill 修复前，IQ1_M SG prefill 的三个 LMUL 候选全部被资源检查拒绝；没有把拒绝折算成成功计时。
- `/tmp/weft-tune.bvs5qlsg`：dense-table snapshot 修复前，TQ1_0 SG prefill 的 192 个候选全部拒绝。该次调用误用了单数环境变量名，因此实际域比原打算的三项大；真实枚举域已记录，仍在默认预算内，不能把它写成三项扫描。

换机器、shape、source 或计时环境应在相应 runner 合同下重新扫描。`WEFT_TUNE_SELECTION` 校验 request、entry/symbol、target facts、搜索域和候选记录，不是任意源码/硬件变化的自动失效系统。没有把旧 winner 无条件迁移到新机器。

## 6. 尚未闭合的性能问题

原选中集合中有 139 项不低于 source，最终有 127 项；30 项比旧表 Weft 值下降超过 5%。主要负例包括：

| kernel / target | 原 Weft | 最终 Weft | 单位 | 变化 |
|---|---:|---:|---|---:|
| dequant NVFP4 / SG | 733.566819 | 169.973266 | MEl/s | -76.829% |
| dequant IQ4_XS / SG | 714.912733 | 322.785848 | MEl/s | -54.850% |
| dequant IQ2_S / SG | 673.905403 | 345.156714 | MEl/s | -48.783% |
| dequant IQ2_XXS / SG | 692.394905 | 357.295574 | MEl/s | -48.397% |
| dequant TQ1_0 / SG | 744.449030 | 404.040303 | MEl/s | -45.726% |
| Q5_0 prefill / SG | 10.459428 | 7.510873 | GOP/s | -28.190% |
| Q5_0 prefill / K1 | 3.073122 | 2.281910 | GOP/s | -25.746% |
| TQ1_0 prefill / SG | 6.063811 | 4.579534 | GOP/s | -24.478% |

十二项从过线变成低于 source，全部在 SG2044：IQ2_XS standalone/decode，Q2_K prefill，Q3_K standalone，以及 Q5_K、Q4_K、Q2_K、IQ3_XXS、TQ1_0、Q3_K、IQ2_XXS、IQ2_S 的 dequant。接近 1 的比值也如实保留，没有未经重复测量就断言微小变化的因果。

不能把这 30 项全部归因于本轮。与紧邻本轮之前的 [DSL 实际运行记录](dsl-surface-entry-replay.csv) 对照：

- SG Q4_K dequant 此前已经是 443.208491，本轮 443.732709；旧对照表 710.758518 没有反映这个已存在差距。
- K1 Q6_K decode 此前为 2.455462，本轮 2.446343，旧表为 3.122822，同样不是本轮才出现的全部差距。
- SG Q5_0 prefill 此前为 10.490698、K1 为 2.887160；本轮分别 7.510873、2.281910，确有新增下降。
- SG TQ1_0 dequant 此前为 746.230532，本轮 404.040303，明显新增下降。
- SG IQ2_XS standalone 此前为 3.712754，本轮 3.524658，旧表 4.171873；同时包含原已存在差距和新增下降。

读取语义闭合确实增加了一些 snapshot 工作，但这不是全部负例的已证实根因。source dequant donor 的函数参数含 `GGML_RESTRICT`，Weft 当前参数仍可能 alias；本轮没有未经作者授权静默增加 noalias 或缩窄输入域。移除 memcpy 调用、改为较少 vector transfer 后，部分结果仍然更慢，说明 pair-count 不是足够的性能模型。`*-copy.jsonl`、`*-optimized.jsonl` 与最终记录都保留了这些正反结果。

临时扩大 Iota/scalar rematerialization、降低整段 scale 启发式阈值曾使 IQ1_M SG standalone 从约 2.188 跌到 1.117；这些实验修改没有进入最终实现。最终只保留有完整供给与 effect 条件的 scalar-repeat edge 改写。所有负例的进一步因果隔离和吞吐恢复尚未完成，不能将本轮称为性能全面验收通过。

## 7. 手工复现

在仓库根目录，先构建当前工具，然后用现有 runner 完成 DSL -> canonical/physical -> C -> 目标机数值与计时：

```bash
cmake --build build -j4
WEFT_KEEP_ARTIFACTS=1 examples/run/weft-quantized-vec-dot.sh sg2044 iq1_m 10
WEFT_KEEP_ARTIFACTS=1 examples/run/weft-mul-mat.sh sg2044 iq1_m decode 10
WEFT_KEEP_ARTIFACTS=1 examples/run/weft-quantized-vec-dot.sh sg2044 tq1_0 10
WEFT_KEEP_ARTIFACTS=1 examples/run/weft-mul-mat.sh sg2044 tq1_0 decode 10
WEFT_KEEP_ARTIFACTS=1 examples/run/weft-quantized-vec-dot.sh k1 iq1_m 10
WEFT_KEEP_ARTIFACTS=1 examples/run/weft-mul-mat.sh k1 iq1_m decode 10
WEFT_KEEP_ARTIFACTS=1 examples/run/weft-quantized-vec-dot.sh k1 tq1_0 10
WEFT_KEEP_ARTIFACTS=1 examples/run/weft-mul-mat.sh k1 tq1_0 decode 10
WEFT_KEEP_ARTIFACTS=1 examples/run/weft-row-dequantize.sh sg2044 tq1_0 10
```

SG 使用 SSH `rvv`、core 48、VLEN 128、Clang 18.1.8；K1 使用 SSH `k1`、core 3、VLEN 256、Clang 18.1.8。运行期读取到的 governor 为 performance，频率分别约 2.6/1.6 GHz；这是环境快照，不是计时全过程频率监测。VLEN 也由实际 CSR 读取核对。工具链、完整 march、ABI 和输入 policy 由现有 runner 输出。

独立验证读取点，以 encoded alias repro 的 SG 运行命令为例：

```bash
read_repro=$(mktemp -d /tmp/weft-read-repro.XXXXXX)
PYTHONPATH=python:examples python -m weft \
  examples/repro/weft/encoded_read_order.py > "$read_repro/kernel.mlir"
build/tools/weft-compile/weft-compile "$read_repro/kernel.mlir" \
  --emit=intrinsic-c --march=rv64gcv --abi=lp64d --vlen-bits=128 \
  -o "$read_repro/kernel.c"
read_remote=$(ssh rvv mktemp -d /tmp/weft-read-repro.XXXXXX)
scp "$read_repro/kernel.c" examples/repro/weft/encoded_read_order_runtime.c \
  "rvv:$read_remote/"
ssh rvv "/opt/tcrv-toolchains/llvm-18.1.8/bin/clang \
  -O3 -std=c11 -Wall -Wextra -Werror -ffp-contract=fast \
  --gcc-toolchain=/opt/tcrv-toolchains/gcc-15.2.0 \
  -B/opt/tcrv-toolchains/binutils-2.46.1/bin -fno-integrated-as \
  -march=rv64gcv -mabi=lp64d \
  $read_remote/kernel.c $read_remote/encoded_read_order_runtime.c \
  -o $read_remote/run"
ssh rvv "taskset -c 48 $read_remote/run"
```

预期 `a=7 b=3 expected_a=7 expected_b=3`，exit 0。另外三个 repro 使用同目录对应 `.py` 和 `_runtime.c`，没有增加 test 框架。

两候选调参并实际应用结果的可执行例子：

```bash
WEFT_TUNE_LMUL_EIGHTHS=16,32 \
WEFT_TUNE_UNROLLS=1 WEFT_TUNE_PIPELINE_DEPTHS=1 \
WEFT_TUNE_SCALAR_LOAD_PRIMES=0 \
WEFT_TUNE_PARTIAL_COMBINE_POLICIES=independent-multilevel \
WEFT_TUNE_RECORD_AXIS_POLICIES=within-record \
WEFT_TUNE_MAX_CANDIDATES=2 WEFT_TUNE_APPLY_WINNER=1 \
  examples/run/weft-kernel-tune.sh sg2044 q8_0_quantize 10
```

完整枚举域和全部候选结果以该次命令输出的外部目录为准；新的测量可能选择不同 winner。
