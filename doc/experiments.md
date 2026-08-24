# 实验定义

## 1. 实验服务于编译器

Weft 的目标是建立一门语言和一个编译器：

```text
作者用 Weft 写出确定的数值与实现树
    ↓
编译器生成 target-aware RISC-V physical program
    ↓
terminal translation 生成 intrinsic C / local asm
```

实验用于回答三件事：

1. 这份程序是否生成了数值正确的真实 target artifact；
2. 编译器是否把作者已写明的结构实现成了高质量机器程序；
3. 相同语言事实换 format、shape、consumer、VLEN 或 extension 后，编译能力是否仍成立。

实验结果可以否定语言或编译器假设，但不能成为隐藏语言。不能因为某行 CSV 慢，就增加 kernel/format route、whole-kernel leaf、emitter matcher 或 fallback。

## 2. Baseline 的双重作用

### 2.1 Baseline 决定作者比较哪一种实现

`source/` 中的 GGML/llama.cpp baseline 不只是一个最终速度数字。它是作者编写对应 DSL/std 函数时的真实算法实现参照。为了比较同一种实现，作者必须先从 baseline 确认并在 DSL 中自然表达：

- algorithm variant 与数值分解；
- outer traversal 与 loop order；
- cache/blocking 层次；
- staging、workspace 与跨 loop reuse；
- accumulator/state 的诞生层和更新方式；
- persistent repack、derived Encoding 与 ABI；
- activation quantize、decode、prefill 等 phase；
- preprocessing 与 timed region 的边界。

若 baseline 的高性能依赖这些结构，而 DSL kernel 没有写出，应该修改 DSL/std tree；不能要求 target compiler 从普通 SSA 猜回 blocking、staging 或 persistent packing。若当前 DSL 无法无歧义表达它，这是语言缺口，不能改用更容易的 baseline 或缩小 workload 掩盖。

### 2.2 Baseline 不决定 target physical realization

作者树冻结后，baseline 不能指定：

- LMUL、`vl`、lane/register/fragment mapping；
- unit/strided/indexed/segment memory form；
- invocation-local pack schema；
- RVV/IME target leaf；
- unroll、pipeline、spill 与 intrinsic spelling。

这些由 RISC-V physical passes 与 leaf contracts 决定。可以阅读 baseline 或 `materials/` 理解其有效的 microkernel、reuse 和指令组织，但知识必须分别进入 DSL tree、physical compiler 或 local leaf；baseline 函数本身不得被调用、链接、包装或作为 fallback。

### 2.3 Baseline 比较最终结果，不参与编译选择

GGML 吞吐不参与 leaf selection 或 physical-parameter tuner。构建期 tuner 只在编译器自己生成并验证合法的有限参数绑定之间实测；baseline 只在 winner 生成以后评价最终 artifact。

## 3. 可比较 case 的身份

只有下列内容全部一致时，Weft 与 baseline 的比值才有意义：

```text
target machine
algorithm variant 与数值语义
operator / helper boundary
shape、dtype、quant format 与 layout
decode / prefill phase
thread count
input bytes / data generation policy
persistent preprocessing 与 repack 归属
timed region
system compiler 与 flags
correctness policy
measurement protocol
```

production `MUL_MAT`、standalone vec-dot、activation quantize 和 row dequantize 是不同 case。不能用 vec-dot helper 数字代替完整 MUL_MAT，也不能把 persistent repack 排除在一边、计入另一边。

当前 production 矩阵参照固定为 `N=4096, K=4096`；decode 为 `M=1`，prefill 为 `M=128`。vec-dot helper 是独立 case，当前参照 shape 为 `M=1, N=14336, K=4096`。改变 shape 形成新 case，不能覆盖旧结果。

## 4. 正确性定义

### 4.1 Encoding 与离散 storage

bit-exact 只用于验证：

- 同一段 packed storage bytes；
- bit/byte order 与 padding；
- grouped/layered/joined/bit-plane mapping；
- 从随机 bytes 提取出的离散字段；
- 定义为 exact wrap/saturate 的整数结果。

这一判据用于发现 Encoding 错一位、field 拼接错误或 ABI 不一致。

### 4.2 浮点 kernel 输出

浮点输出使用：

- finite / NaN / Inf 政策检查；
- case 预先声明的 absolute tolerance；
- case 预先声明的 relative tolerance；
- 对符号、数量级、全零和明显量化错误的检查。

合法 FMA contraction、reduction 顺序或 closed primitive 内部 partial 可以产生末位差异。不能为了 float bit-exact 而交换作者 scale tree、插入人为物化阻断 contraction，或修改编译器选择的合法指令。

tolerance 必须在运行前随 case 定义，不能看到结果后放宽。数值失败的 case 不记录性能数字。

## 5. Toolchain 与编译口径

Weft generated C、Weft runtime/harness 和 GGML baseline wrapper 必须使用同一套 Clang toolchain 与共同 flags。每次结果至少记录：

```text
Clang executable 与完整版本
-O level
-ffp-contract policy
-march / target extensions
-mabi
LTO、fast-math 与其它影响数值或代码生成的 flags
assembler / linker（IME 需要时）
```

当前统一的公共口径是 `-O3 -ffp-contract=fast -mabi=lp64d`，`-march` 和 extension flags 按 target profile 设置。不能让 Weft 开启 contraction 而 baseline 关闭，也不能用 Clang 编译 Weft、用 GCC 编译 baseline 后给出速度比。

若 Clang 版本、target flags、assembler 或数值 flags 改变，相关 baseline 必须用新口径重编重测。generated C 中的 intrinsic、typed asm 和用于保护已选 physical 形态的局部 pragma 属于编译器输出，不要求 baseline 源码具有相同文本；公平性要求的是相同 toolchain、target capability 和数值政策。

## 6. 目标机器

正式 kernel 性能比较使用两台主目标：

| target | 当前实验环境事实 | 正式路径 | 测量方式 |
|---|---|---|---|
| SG2044 | 48 cores，VLEN128 | 标准 RVV | 单线程，固定目标 core |
| K1/X60 | baseline 环境记录 3 个可用 cores，VLEN256 | 标准 RVV、SpacemiT RVV、IME1 | 单线程，固定目标 core |

两台机器之间不直接比较绝对速度。跨 target 只检查同一 DSL tree 是否根据 profile 形成各自合法的 layout、memory form、leaf 和 schedule；性能比始终是 SG Weft 对 SG baseline、K1 Weft 对 K1 baseline。

`rvv-v100` 是备用 VLEN256 标准 RVV 环境，不是 NVIDIA V100，也不是当前正式性能 baseline。只有在同一 toolchain、source baseline 和完整测量协议在该机重建后，其数据才可进入正式 CSV；当前 toolchain 不能生成 IME 时不得报告 IME 结果。

每次运行还必须记录 OS/kernel、CPU identity、VLEN 与 extension probe、目标 core、frequency/governor 政策和当时 toolchain。机器事实与实测不一致时，以当次 probe 为准，并建立新结果集，不能沿用旧 baseline。

## 7. Kernel 测量协议

正式单线程 kernel 测量采用统一协议：

1. 初始化相同 inputs 与 persistent artifacts；
2. 完成一次不计时 warmup；
3. 每次 timed invocation 前遍历 64 MiB eviction buffer；
4. 记录 10 次 kernel invocation 的 wall time；
5. 报告 median；
6. 编译、动态链接、tensor 初始化、persistent weight quantize/repack、warmup 和 eviction 不计时。

activation quantize、kernel 内 local pack 或其它属于 operator invocation 的工作，若 baseline 在 timed graph 内执行，Weft 也必须计入。只有跨调用 persistent artifact phase 可以在双方都排除。每个 case 必须把 timing scope 写成可读描述，不能只写“kernel”。

矩阵乘与 vec-dot 吞吐按 `2MNK` 计算 GOP/s；quantize/dequantize 按实际 logical elements 计算 MElements/s。其它 operation 优先报告 wall time；没有一致 work 定义时不制造吞吐率。

模型级 `llama-bench` 是另一种实验，沿用其真实 graph、线程和计时协议，不能与上述 kernel microbenchmark 行混在同一分布中。

## 8. 结果文件

两类数字物理分离：

- `report/baseline/ggml-riscv-kernel-performance.csv` 保存当前协议下固定的 GGML baseline；
- `report/weft-kernel-performance.csv` 保存当前 Weft compiler 真实重测结果，重测后直接覆盖对应行。

baseline 只在机器、toolchain、flags、算法/shape 或测量协议改变时重测；Weft compiler 修改不触发无意义的 baseline 重跑。历史 compiler 结果、旧协议行和 `materials/experiments` 不得混入当前表。

每行至少保存 case identity、hardware/ISA、shape、timing scope、compiler/version、flags、physical configuration、repetitions、correctness、absolute/relative error、median 和 throughput。缺少可比较合同中任何一项的行可以保留为运行记录，但不得计算 baseline ratio。

## 9. 性能差距怎样归因

发现差距时，按所有权逐层判断：

1. **DSL/std tree：** baseline 是否具有当前树未表达的 algorithm variant、blocking、staging、state 或 persistent layout；
2. **Canonical IR：** axes、Encoding、Level、effects 和 numerical semantics 是否完整保存；
3. **RISC-V physical passes：** layout、conversion、memory、reuse、pipeline 和 resources 是否正确产生；
4. **Leaf selection：** 是否选择了符合 typed facts 与 target capability 的 RVV/IME operation；
5. **Terminal spelling：** intrinsic/asm 是否忠实，Clang 是否保留了已经选择的 physical 形态；
6. **Experiment protocol：** workload、timing scope、flags 或 cache 状态是否仍匹配。

若第 1 项不一致，修改 DSL/std 并把它作为新的作者程序重新比较。若前四项已经闭合但达到 baseline 仍必须改变 canonical logical Value 集合或 Level 归属，应停止并报告 spec 2.2 可能被证伪；不能让 compiler 暗中改 tree。

性能比、分档和聚类只是定位共享能力缺口的工具，不是编译器架构。单条高性能不能证明泛化，平均数也不能代替逐 case 事实。有效证据是：修改共享 pass 或 leaf 后，多个具有相同 typed 事实的自然程序共同改变，而且陌生输入不需要新增后端 route。

## 10. 禁止把实验变成产品主线

不允许：

- 为追一行数字增加 kernel 名、format 名、exact closure 或 whole-kernel leaf；
- 通过 GGML/materials 调用、legacy path 或 silent scalar fallback 通过测试；
- 数值失败仍记录性能；
- 为 float bit-exact 修改作者 tree 或禁用合法融合；
- 用更小 shape、不同 phase、不同 timing scope 或 helper 替换失败 case；
- 因当前 baseline 慢而降低 Weft 目标，或因当前 baseline 快而把其机器结构硬编码进 emitter；
- 将内部 tuner 结果、generated-C inspection 或 compile success 冒充真实 target 性能。

实验最终回答“当前编译器在这个明确合同下做得怎样”。它不回答“下一条后端特例应该写在哪里”。
