# DSL 表面更新与读取顺序复核

2026-09-06。本轮更新作者表面、前端识别/诊断及对应规范，记录调优配置现状；不移动参数，不修改 C++ lowering、leaf、runtime 数值合同或正式性能 CSV。改名前源码、工具与临时 artifacts 保存在 `/tmp/weft-dsl-update.TlgLSV/`，不进入仓库。

## 1. 表面与覆盖

冻结 kernel 语料实际是 111 个 Python 文件，而不是 109 个；全部完成更新。另更新 `examples/formats/ggml.py` 和 `examples/functions/quantization.py`，没有保留公开旧入口或兼容别名。runner 没有引用这些公开 API，因此未改脚本；文件/函数中已有的 entry identity 不因词汇调整而更名。

| 表面 | 当前写法 | 实际调用数，不含 import/定义 |
|---|---|---:|
| 读取/写入 | `load(region)` / `store(region, value)` | 162 / 125 |
| 状态/只读复用 | `state(dtype, shape, init=...)` / `stage(expr)` | 52 / 64 |
| 块乘/一般乘积归约 | `dot(..., over=..., acc_dtype=...)` / `reduce_dot(...)` | 23 / 43 |
| 逻辑层级 | `level.rows/cols/blocks/tiles/subtiles` | 34 / 28 / 84 / 57 / 52 |
| 索引/相邻配对求和 | `arange(start, end, ...)` / `sum_pairs(value)` | 81 / 6 |
| Encoding | `grouped(elements=...)` / `bit_layers(elements=..., order=...)` / `pack_fields(group=..., fields=..., low_bits=..., order=...)` | 27 / 27 / 4 |

计数涵盖 `examples/kernels`、`examples/functions` 与 `examples/formats`，不包含本轮单独添加的读取顺序手工 repro。新词的语义见 `doc/dsl/`；本文不另定义语言合同。

## 2. Canonical 对照

改名前后逐份使用 `python -m weft` 生成 IR，并经 `weft-compile --emit=kernel-ir` 解析验证。比较仅去掉 `loc(...)` 和 kernel `source` 属性中的源码位置，不删除类型、轴、operands、effects 或其它 attributes。

- 111/111 份解析验证成功。
- 105/111 份除源码位置外逐字相同。
- 6 份只有 7 个 operation 名称变化；将这 7 项明确对应后，111/111 份无其它差异。

| 文件 | 节点变化 | 等价依据 |
|---|---|---|
| `gemv/q4_k.py`、`gemv/q4_k_contract.py`、`gemv/q4_k_groups4.py`、`mul_mat/q4_k_persistent.py` | 各 1 个内部 `dot` 改为 `contract` | 显式保留同一最终共同 k 轴，operands/result 类型与默认提升不变；它们是一般 row-vector 乘积归约 |
| `dense/gemm_f32.py` | 1 个 `contract` 改为 `outer_contract` | 两个二维块，各有一条互异 free axis；缩并轴、输出轴序及 dtype 不变 |
| `attention/flash_attention.py` | 2 个 `contract` 改为 `outer_contract` | QK 和 PV 各有一条共同缩并轴和两条互异 free axes，数值树及 state recurrence 不变 |

其它 `contract` 只改公开拼写为 `reduce_dot`，内部 op 不变；已有 `outer_contract` 的公开拼写成为 `dot`。没有展开成窄输入乘法再求和。`state/stage` 的分区顺序和 `subtiles` 的内部 domain relation 也不变。

新 `load` 不再先对 rank-0 region 执行隐式读取，再错误地把读取结果作为 View 输入。这修正了显式标量读取的前端入口；不等于已经补齐 rank-0 后端，也没有改变 111 份既有 IR。

## 3. 验收方法

现有 runner 按其原 target、entry、shape、meta、LMUL、unroll、pipeline、scalar-prime 等绑定执行，10 次冷缓存测量，数值 policy 不变。运行保留临时 canonical/C/assembly，并以日志中的原 compiler 参数重建改名前 C，与本轮 C 做字节比较，不做摘要校验。

每个可编译配置另生成 final Physical IR，执行：

```bash
build/tools/weft-opt/weft-opt physical.mlir \
  --canonicalize --cse --weft-riscv-canonicalize-layouts \
  --weft-riscv-share-layered-windows --weft-riscv-share-layered-windows \
  --weft-riscv-verify-final --verify-each --verify-roundtrip -o checked.mlir
```

再对 checked IR 执行同一命令并比较两次输出。该证据只证明这个 final 样本在所列流程上稳定；`resources_materialized=true` 跳过主要 layout rematerialization 分支。不能据此宣称全部 pass 幂等或数值正确。真实运行是独立验收条件。

环境快照：SG 为 Linux 6.12.66、64 个在线 CPU、core 48、Clang 18.1.8、performance governor，读取到 2600000 kHz；K1 为 Linux 6.6.63、8 个在线 CPU、core 3、Bianbu Clang 18.1.8、performance governor，读取到 1600000 kHz。snapshot 在运行期间采集，不代表每次计时全程监测频率。VLEN/ISA 与 flags 沿用各 runner 的 128/256-bit profile 和完整 march。

## 4. 读取顺序的真实反例

手工入口是 [scalar_read_order.py](../examples/repro/weft/scalar_read_order.py)，对应 [runtime](../examples/repro/weft/scalar_read_order_runtime.c)。它在同一个一元素 Level 里读取 A，写入替换值，再把先前读取结果写入 B。无需两个独立参数互相 alias，同一个 A 的读写就能触发问题。

输入 `A=3, S=7`，正确结果是 `A=7, B=3`。SG2044 与 K1 均实际得到：

```text
a=7 b=7 expected_a=7 expected_b=3
```

两个 runtime 都返回 1。对应 final Physical IR 能通过 verifier 与指定重放，然而 terminal C 将前一个 load 保存成了解引用表达式：

```c
*A = *S;
*B = *A;
```

这已是可执行反例，不再只是性能疑点。它证明本例存在错误，不能外推为所有现有 kernel 的数值都错。公开 API 更新后的同一 repro 生成 C 与改名前完全相同，因此不是这轮改名引入。

SG 手工重现，从仓库根目录执行：

```bash
scalar_work=$(mktemp -d /tmp/weft-read-order.XXXXXX)
PYTHONPATH=python:examples python -m weft \
  examples/repro/weft/scalar_read_order.py > "$scalar_work/kernel.mlir"
build/tools/weft-compile/weft-compile "$scalar_work/kernel.mlir" \
  --emit=intrinsic-c --march=rv64gcv --abi=lp64d --vlen-bits=128 \
  -o "$scalar_work/kernel.c"
scalar_remote=$(ssh rvv mktemp -d /tmp/weft-read-order.XXXXXX)
scp "$scalar_work/kernel.c" examples/repro/weft/scalar_read_order_runtime.c \
  "rvv:$scalar_remote/"
ssh rvv "/opt/tcrv-toolchains/llvm-18.1.8/bin/clang \
  -O3 -std=c11 -Wall -Wextra -Werror -ffp-contract=fast \
  --gcc-toolchain=/opt/tcrv-toolchains/gcc-15.2.0 \
  -B/opt/tcrv-toolchains/binutils-2.46.1/bin -fno-integrated-as \
  -march=rv64gcv -mabi=lp64d \
  $scalar_remote/kernel.c $scalar_remote/scalar_read_order_runtime.c \
  -L/opt/tcrv-toolchains/gcc-15.2.0/lib \
  -Wl,-rpath,/opt/tcrv-toolchains/gcc-15.2.0/lib -o $scalar_remote/run \
  && taskset -c 48 $scalar_remote/run"
```

K1 使用 `--vlen-bits=256`、SSH alias `k1`、`/usr/bin/clang-18`、core 3，去掉 SG 的 GCC/binutils 和 library-path 参数，保留 `-fno-integrated-as -march=rv64gcv -mabi=lp64d`。

## 5. 读取风险范围与本轮边界

- dense scalar 和 scalar tuple 在 `RISCVIntrinsicC.cpp::materializeNumeric` 返回解引用字符串；`compileLocalLoad` 同样保存 local array 访问表达式。这些 use 不能自动拥有 load 点快照语义。
- encoded `compileAdmit` 可以只保存 record descriptor；实际 field 读取延迟至后续消费。部分 physical `FieldOp/ExtractOp` 为 Pure，已有 typed storage-load 物化只覆盖特定路径。修复不能仅在某个最终 scalar consumer 前增加临时变量，还需要闭合原 read/effect 边界。
- scalar lookup、显式 register materialization 和普通 RVV load 已有实际临时变量，不能与上述路径混为一谈。
- 直接 rank-0 dense 例子目前停在 `sequential dense load requires one selected physical element`；独立 encoded 例子停在现有 scalar store/handoff emission 限制。它们不是已完成的运行证明，也没有为了绕过限制改变生产 kernel。

本轮对 scalar 的工作是独立诊断与真实 repro，未修改 emitter，读取错误尚未修复。后续修复应让每个已选 physical memory-read SSA result 在读取点获得稳定绑定；不能要求作者加 `stage` 保正确，也不能用 volatile、额外 alias 承诺或隐藏 reload 掩盖错误。

## 6. 初始化与调优

保留 Level 的 state/stage 初始化区域，不做 region 或生命周期重构。新增诊断分别点名 Level body 值与另一初始化区域的值，初始化可见性在 `doc/dsl/values-and-levels.md` 中说明。两个最小拒绝例分别得到：

```text
Level initialization cannot reference body value 'loaded'; state and stage initializers execute before the Level body
stage initialization cannot reference 'value' from the other initialization region; state and stage initializers have separate scopes
```

诊断保持当前区域的外层环境解析规则，不引入新的 shadowing 语义；inline helper 使用自己的名称环境。

成本模型边界、有限结构比较、第二输入证据范围、普通 for 暂不放宽以及机械验收的证明范围，已写入 `doc/compiler/optimization-principles.md` 并同步相关规范。参数数量与来源单列于 [autotune-binding-inventory.md](autotune-binding-inventory.md)，本轮不搬配置。

partial materializer 的 operation-major 选择、后生成 issue loop 的专用调度 owner、reload part 的 emitter 选择仍是上一调查确认的实现边界。本轮没有改动这些物理算法，也不宣称放宽政策即已完成相关实现。

## 7. 最终运行结果

逐配置记录包含请求、实际 entry、compiler bindings、数值状态、计时、C 对照、重放结果及拒绝原因。独立 CSV 已按用户要求清理，可从清理前的 Git 历史恢复；本节保留验收范围与关键结果。该轮正式性能 CSV 和 baseline CSV 未修改。

| 范围 | 请求数 | 数值运行、C 同一性与指定重放均通过 | 改名前后相同的拒绝 |
|---|---:|---:|---:|
| SG2044 标准 runner 配置 | 107 | 104 | 3 |
| K1 标准 runner 配置，含 IME entry | 108 | 105 | 3 |
| SG2044 显式 persistent/staged 变体 | 6 | 4 | 2 |
| K1 显式 persistent/staged 变体 | 6 | 4 | 2 |
| 合计 | 227 | 217 | 10 |

217 个通过项的生成 C 与改名前同配置逐字一致，217 个 final Physical IR 均通过所列两次重放且文本稳定；数值检查也均通过。这个结论不包含下列拒绝项：

- 双机 Q1_0 prefill：`rvv_storage_window` 不能闭合 byte-aligned contiguous lane window。
- 双机 IQ2_S/IQ2_XS prefill：实际 staged entry 的 nested scaled contraction 缺少资源合同内合法的 full-product carrier。
- 双机显式 IQ2_S/IQ2_XS staged decode：同一 full-product carrier 拒绝。

以上 10 个请求都使用原绑定，改名前 canonical IR 同样失败，去掉源码位置后的错误全文相同。没有换 shape、参数或 entry 绕过失败；它们不是本轮回归，也不是成功运行。

runner 请求共涉及 104 个独立 source 文件，其中 101 个至少有一个真正运行通过的配置。其余 7 个冻结源文件不在这些 runner 请求内：attention、`gemv/q4_k_contract.py`、`mul_mat/iq2_xxs.py`、`mul_mat/q3_k_predecoded_scales.py`、`mul_mat/q3_k_predecoded_scales_decode.py`、`mul_mat/q4_k.py`、Top-K。它们均计入 111 份 canonical 对照，但不能说已全部真机运行。

对其中有 operation 名称变化的 attention 与 Q4_K contract GEMV，另做双机同配置 lowering 对照：attention 在 `BQ=4,BK=16` 停于 `floating stream contract has no closed reduction-lane memory realization`；Q4_K contract GEMV 在默认 m1 配置停于 selected widening-dot shape 合同。改名前同样拒绝，未获得可执行 artifact。未以其它配置的运行代替这两个入口。

### 7.1 配对计时

每个 target 选 Q4_K row-dequantize、Q4_K MUL_MAT decode、dense F32 GEMM，各执行两轮 before→after 交替；每次仍为 10 次冷缓存测量。下表为两轮 median_us 的中位数，正值表示 after 用时增加，不表示统计显著性。

| target / entry | before us | after us | 用时变化 |
|---|---:|---:|---:|
| SG / Q4_K row | 9494.742 | 9468.371 | -0.278% |
| SG / Q4_K decode | 4137.328 | 4066.092 | -1.722% |
| SG / F32 GEMM | 550544.868 | 553008.743 | +0.448% |
| K1 / Q4_K row | 12334.104 | 12331.640 | -0.020% |
| K1 / Q4_K decode | 9392.891 | 9375.780 | -0.182% |
| K1 / F32 GEMM | 1678337.749 | 1676601.654 | -0.103% |

24 次配对运行全部数值通过；12 对生成 C 和远端 Clang 输出的 kernel assembly 均逐字相同。未发现改名改变计算代码或参数；这些小幅计时差异没有对应指令变化，不归因为优化收益或回归，也不宣称已经统计证明性能严格相等。

本轮完成的是表面统一、既有程序对照、当前可执行入口回归、政策文档同步和读取错误诊断。既有拒绝、没有运行入口的源码及 scalar 读取错误均保留上述明确边界。
