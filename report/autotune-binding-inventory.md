# Autotune 参数绑定分布与证据边界

2026-09-06，调查与 DSL 表面更新同轮。本文是当前源码与历史测量记录的快照，不是配置设计规范；本轮不迁移参数、不重新选择 production winner。

## 1. 四种来源不能混同

| 来源 | 位置 | 内容与数量 |
|---|---|---|
| 作者 source 参数 | `examples/kernels/` | 111 个 kernel；29 个文件内 117 次 `auto` 调用，只声明名字；实例值由构建侧提供 |
| 目标合法性与默认配置 | `include/Weft/Target/RISCVTargetProfile.h`、`lib/Target/RISCVTargetProfile.cpp`、`tools/weft-compile/weft-compile.cpp` | 4 项公开数值/布尔 physical bindings，另有 3 项结构策略/限制 |
| 具体 target/entry/phase 配方 | `examples/run/weft-kernel.sh`、`weft-mul-mat.sh`、`weft-quantized-vec-dot.sh`、`weft-row-dequantize.sh` | 各自保存下面的静态配方与覆盖逻辑 |
| 调优候选域 | `examples/run/weft-kernel-tune.sh` | 环境变量提供 meta 域和 physical 扫描范围；结果打印后再运行 winner，但不持久写回 production 配方 |

source `auto` 改变作者程序的静态实例；physical binding 实例化同一程序。C++ 的 ISA 合法域、target 默认值、runner 中一次实测选择的值、完整搜索的 winner 是四类不同事实。

## 2. C++ 与命令行

| 字段 | 默认值 | 校验/来源 |
|---|---|---|
| `--auto-lmul-eighths` | 8，即 m1 | profile 合法域 `{1,2,4,8,16,32,64}`，单位为 LMUL 的八分之一；资源仍需检查 |
| `--auto-unroll` | 1 | 正整数；CLI 没有额外统一上限，不代表任意值都能编译 |
| `--auto-pipeline-depth` | 1 | 只接受 1 或 2；depth 2 还需合法 producer/carry 关系 |
| `--auto-scalar-load-prime` | 0 | 0 或 1；开启后仍需存在闭合的适用 memory edge |
| `--max-widening-combine-groups` | 2 | target 对一次 widening combine 的资源策略限制 |
| `--partial-combine-policy` | `independent-multilevel` | 显式结构优先级 |
| `--record-axis-policy` | `within-record` | record-axis placement 优先级 |

见 [CLI](../tools/weft-compile/weft-compile.cpp:40)、[profile](../lib/Target/RISCVTargetProfile.cpp:206)。profile 的 SEW 为 8/16/32/64、vector-register budget 为 32，要求 full V 和显式正 VLEN。IME1 另受 RV64、VLEN256、类型与 fragment 合同约束。这些机器事实不是“某格式测得更快”的参数。

`--meta NAME=INTEGER` 只接收单值正整数且名称不能重复，不定义每个 kernel 的候选域。

## 3. Runner 配方数量

计数单位是脚本接受的 target/entry/phase 配方，不是独立 kernel、合法 artifact 或 winner。四个 runner 存在重复入口，不应相加声称泛化覆盖数。显式字段数不包含 `--meta`、march/ABI/VLEN 或矩阵扩展选择。

| runner | 配方数 | 含 source meta 的配方 | physical 显式字段 | 非默认 physical 字段 |
|---|---:|---:|---:|---:|
| `weft-kernel.sh` | 15 | 5 | 12 | 7 |
| `weft-mul-mat.sh`，标准 26 格式 | 104 | 54 | 254 | 66 |
| `weft-quantized-vec-dot.sh` | 48 | 0 | 91 | 35 |
| `weft-row-dequantize.sh` | 48 | 0 | 6 | 6 |

`weft-kernel.sh` 配方入口见 [58 行](../examples/run/weft-kernel.sh:58)。MUL_MAT 的 104 是 26 格式 × 2 目标 × decode/prefill；显式 staged/persistent 别名还可单独请求，不计作此表新的标准格式。

真实 entry 必须记录，不能只记用户传入的 format：

- `q4_k prefill` 实际 dispatch 到 `production_mul_mat_q4_k_staged`。
- `iq2_s/iq2_xs/iq2_xxs prefill` 实际 dispatch 到相应 staged entry。
- vec-dot 的函数名包含两侧格式，例如 `quantized_vec_dot_q4_k_q8_k`。
- row-dequantize 使用 `examples/kernels/dequantize/<format>.py`，runtime 按 format ID dispatch。

见 [MUL_MAT dispatch](../examples/run/weft-mul-mat.sh:37)、[vec-dot identity](../examples/run/weft-quantized-vec-dot.sh:15)。

## 4. Tuner 覆盖与缺口

默认扫描 LMUL={m1,m2,m4,m8}、unroll={1,2,4}、pipeline={1,2}，即每组 source binding 请求 24 个 physical 配置；不是 24 个必然合法的 artifact。meta 域来自 `WEFT_TUNE_META_CHOICES`，多维按笛卡尔积展开。当前 vec-dot/row 分支不传 meta bindings。

没有独立扫描 scalar-prime、partial-combine-policy、record-axis-policy；也不包含 runner 已经使用的 unroll=8。`WEFT_TUNE_APPLY_WINNER` 默认再次运行选中值，但不保存到下次 production 自动消费的记录。见 [候选域](../examples/run/weft-kernel-tune.sh:64)、[选择与再次运行](../examples/run/weft-kernel-tune.sh:142)。

## 5. 哪些值有实测依据

下表只能称“报告记录的一次实测选定 binding”，不能称当前完整候选域的全局 winner，也不能把历史数值视为本轮重测。

| 输入与目标 | 记录的选择 | 证据范围 |
|---|---|---|
| F32，SG decode/prefill | decode `NC16/MC64/MR2/NR2,m2`；prefill `NC16/MC64/MR4/NR4,m1` | [固定十次测量](riscv-full-account-cluster-closure.md:191) |
| F32，K1 decode/prefill | decode `NC64/MC64/MR2/NR1,m4`；prefill `NC16/MC64/MR4/NR4,m1` | 同一份报告，不证明完整搜索域 |
| Q3_K，SG standalone/decode | scalar-prime 开启；K1 保持关闭 | [局部叶变体对照](riscv-full-account-cluster-closure.md:317) |
| Q1_0，K1 | unroll=4 | [1/2/4/8 的短扫](riscv-full-account-cluster-closure.md:254)，只覆盖该局部选择 |
| IQ2_S row，SG | LMUL=m8，CLI 值为 64 | 当前 [runner](../examples/run/weft-row-dequantize.sh:48) 与[历史选定记录](post-ledger-kernel-closure-consolidated.md:137) |

`report/weft-kernel-performance.csv` 保存 configuration、toolchain、数值和计时等字段，但没有完整候选域、逐项拒绝原因及选择来源。不能从其 202 行性能数据推导“202 个配置全部经过完整 autotune”。`kernel-performance-comparison.csv` 是更简的比较表，也不能承担选择溯源。

结论：已知分布与部分实测来源，不能精确统计“真正测遍完整候选域的 winner 有多少”。本轮保持 C++、runner 和 tuner 的所有绑定值不变。
