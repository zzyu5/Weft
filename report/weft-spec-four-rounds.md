# Weft 按新 spec 重建的四轮实际结果

本文记录从“第一轮：前端按 spec 重写”到“第四轮：全量重测与性能”的一次性结果。它只描述当前代码、真实生成物和已经执行的命令，不把历史性能表、可打印 IR 或本地编译成功写成真机闭环。

## 总结

四轮完成了三件基础工作：

1. 旧 DSL 和旧 canonical IR 被删除，语言改为 `report/weft-spec.md` 定义的 Level、Encoding、普通 SSA handoff 与硬绑定 engine role；
2. 旧的一次性 lowering 被替换为 MLIR 中的临时 RISC-V 编译问题、约束 pass、联合求解和 assignment；
3. assignment 已能继续生成 intrinsic C / 局部 IME asm，并通过远端系统编译器形成可执行程序。

第四轮没有完成固定 206 行的全量重测。当前新主线真实完成的是 K1 上三条 activation quantize，以及一个不计入固定语料的通用 RVV lookup 探针。继续接入 Q4、vec-dot、row-dequantize 和 MUL_MAT 时，发现 spec 中 Q4 packed-array 的逻辑索引到存储 bit 地址关系不足以描述 GGML 的真实布局。按照本轮约束，编译器没有按格式名恢复该关系，也没有继续制造不可比较的性能数字。

## 第一轮：前端按 spec 重写

### 这一轮解决的问题

这一轮把“作者写什么”从历史前端中重新冻结。结果不是给旧 DSL 增加别名，而是删除旧模型，只保留 spec 能逐项解释的构造。

当前语言的基本边界是：

- Encoding 只声明内存位布局，不携带数值不变量和解码等价关系；
- Level 是带 domain/partition、multiplicity、births 和 handoff 的作用域，不是带名字的 `for`；
- 普通 `for/if/while` 是有序标量控制，不能自动获得 wide realization；
- handoff 在 Python 中仍是普通赋值，在 canonical IR 中则保存层归属和生命周期；
- `@wide`、`@matrix`、`@transfer` 是硬绑定，不是后端提示；
- GEMM、GEMV 和量化计算是 `python/weft/std/` 中用同一门 DSL 写的函数，不存在内置 whole-kernel matmul；
- 派生 encoding 是 build/load 阶段的类型生成，不是运行 kernel 的外层循环。

### 实际代码变化

旧前端、旧 examples 和旧 target-facing surface 被整体删除；本轮代码规模净减少。新的持久表示集中在：

- `python/weft/language/`：dtype、Encoding annotation、Level 和基本局部 op；
- `python/weft/frontend/`：Python AST 到 canonical Kernel IR；
- `include/Weft/Dialect/Kernel/IR/` 与 `lib/Dialect/Kernel/IR/`：唯一 canonical dialect、类型和 verifier；
- `python/weft/std/`：由普通 DSL 函数组成的 dense、quant 和 attention/top-k 示例能力；
- `examples/kernels/`：只保留手工 repro 的入口包装。

代表程序被改写为新模型：

- q4_K vec-dot 压 encoding、层级位宽、代数支路和 handoff；
- GEMM 显式写 NC/KC/MC、accumulator 生命周期、pack/materialize 和局部 contraction；
- GEMV 是同一 dense 关系的退化写法；
- attention 用普通 use-def 表示非归约 handoff；
- top-k 保持普通有序标量控制，不为了 SIMD 使用 Level。

### 本轮证据边界

第一轮的完成物是语言、canonical IR 和 examples。它没有形成 intrinsic C，也没有新增真机性能结果；“前端能够打印 Kernel IR”不能被解释为后端已经闭合。

## 第二轮：建立编译空间

### 编译主链

第二轮把 Kernel IR 到 target assignment 重写成多段 MLIR pass：

```text
canonical Kernel IR
→ construct RISC-V problem facts
→ constrain representations
→ constrain instructions and layout
→ constrain resources and schedule
→ jointly solve one assignment
```

`weft_riscv.problem` 和 `weft_riscv.assignment` 都只在一次编译调用中存在，不是新的长期 IR authority。最终持久表示仍然只有 canonical Kernel IR 和生成的 C/object/header。

### 求解方法

这一轮采用“有限候选枚举 + 约束传播 + 资源过滤 + 最优 assignment”，而不是纯前向 pass 或把回边迭代到一个任意不动点。

原因是 C/D 决定互相制约：LMUL 影响寄存器占用，寄存器占用反过来限制 accumulator、unroll 和 pipeline；decode 形式影响指令和 temporary；instruction 又限制 operand layout。候选来自两处：

- `std/` 中作者写明的函数特化和数值树；
- `auto(...)` 与 target profile 给出的有限取值范围。

求解器不搜索新的 loop、blocking、代数分解、persistent packing 或算法 variant。每个已实例化候选的数值树保持不变。

### C 组与 D 组的实现位置

本轮将 spec 中的 16 项编译决定放进同一个联合 assignment：

| 组 | 决定 | 当前职责 |
|---|---|---|
| C11-C14 | SEW、LMUL、VL、tail | representation constraint 与 per-value assignment |
| C15-C18 | accumulator grouping、资源/spill、partial layout、reduction placement | resource/schedule constraint 与联合求解 |
| C19 | VLEN specialization | target facts 对合法表示的约束，不形成 VLEN 专用路径 |
| D20-D24 | unpack、MAC/instruction、scale broadcast、byte interleave、memory form | instruction/layout constraint 与 per-op assignment |
| D25-D26 | prefetch、pipeline/unroll | 只在存在实际可生成结构时成为候选 |

普通 `for/if/while` 在 representation pass 中保持 `ordered-scalar-control`，不存在“目标看起来适合就自动向量化”的入口。

### 本轮证据边界

第二轮能够为 spec 的 q4_K 程序打印一份完整 assignment，但当时 compiler 的终点就是 assignment；尚不能据此声称 intrinsic C、系统编译或真机数值已经通过。

## 第三轮：走通 assignment 到真机

### 新增的生产路径

第三轮把主链继续闭合为：

```text
DSL source
→ canonical Kernel IR
→ RISC-V problem / assignment
→ intrinsic C 或局部 IME asm
→ 目标机 C/C++ compiler
→ executable
```

`tools/weft-compile` 恢复 `--emit=intrinsic-c`；`lib/Target/RISCVIntrinsicC.cpp` 消费 assignment 生成函数 ABI、普通 C 控制流、地址表达、RVV intrinsic 和 IME 局部 asm。`examples/run/weft-kernel.sh` 是唯一手工执行入口，临时生成 MLIR/C，在目标机编译后用固定 CPU 执行。

本轮接入的代表程序包括：

- F32 GEMV；
- F32 GEMM；
- q4_K × q8_K 的 `mac_pairs` 和 `mac_groups(n=4)` 两棵作者树；
- K1 上同一局部量化关系的 IME realization。

### 本轮证明了什么

- 同一 canonical pipeline 可以落到 VLEN128 RVV、VLEN256 RVV 和 K1 IME 局部指令；
- engine role 在前端硬绑定，matrix 版本不会静默换回 wide；
- assignment 中的 representation 已经能决定 intrinsic 类型和局部 fragment；
- runner 不链接 GGML 或 materials 的实现。

### 本轮没有证明什么

第三轮 q4_K runtime 使用自己构造的线性 synthetic nibble/scale layout，而不是 GGML Q4_K 的真实 half-block 和 packed-scale排列。因此它证明了“新主链能执行这棵 synthetic tree”，没有证明该 q4_K 数字可以与 `source/` 的 GGML baseline 作严格性能比较。

第三轮的 F32 GEMV/GEMM 也只是少量端到端压力点，不等于第四轮 206 行固定语料已经迁移。

## 第四轮：全量重测与性能

### 固定语料

本轮计划覆盖的 production 语料共 206 行：

| 家族 | 逻辑 case | 目标行数 |
|---|---:|---:|
| MUL_MAT | 26 formats × decode/prefill | 104 |
| quantized vec-dot | 24 typed pairs | 48 |
| activation quantize | q8_0/q8_1/q8_K | 6 |
| row dequantize | 24 formats | 48 |
| 合计 |  | 206 |

每一行都应使用相同算法、shape、数据组织、preprocessing 边界和计时范围，与同一目标上的 `source/` baseline 单独比较。旧 CSV 中来自旧 compiler 的数字不算本轮重测。

### 本轮实际扩展的共享能力

为三条 activation quantize 和后续量化族，本轮实现增加了：

- `abs` 与显式 `narrow(dtype, rounding, saturation)`；
- f32 常量的 C 类型保持，避免意外进入 double 运算；
- encoded View 的字段投影与单字段 commit；
- block value 在嵌套 Level 中的普通 SSA subview；
- logical lanes 与 physical strip lanes 分离；
- per-value `physical_lanes`、`stream_parts`、SEW/LMUL 传播；
- `lmul_multiplier={1,2}` 的真实候选和资源过滤；
- 多 consumer value 的 `shared-register` 与 `reload-per-use` 决定；
- widening reduction 的 operation realization；
- max/min co-reduction 的 `separate/shared` 候选；
- 通用 `lookup` 的 `scalar.lookup` / `rvv.indexed-lookup` 决定，以及机械生成的 RVV `vluxei`。

`lookup` 已用临时 DSL source 在 K1 上经完整主链、Clang 和 RVV 执行，4096 个索引结果 bit-exact。它不计入 206 行，只证明 codebook/IQ 路径共同依赖的局部 op 已有真实 artifact。

### K1 activation quantize 的真实结果

当前新主线在 K1/X60、VLEN256 上以 `M=128,K=14336`、10 次 invocation、64 MiB eviction 运行得到：

| Kernel | source baseline | Weft | 时间差 | 吞吐差 |
|---|---:|---:|---:|---:|
| q8_0 | 5.560 ms | 5.878013 ms | +5.72% | -5.40% |
| q8_1 | 6.395 ms | 6.922016 ms | +8.24% | -7.61% |
| q8_K | 5.112 ms | 6.596924 ms | +29.05% | -22.50% |

三条 Weft runtime 都先执行一次数值检查，再做 10 次带 cache eviction 的计时并取 median。它们与 source runtime 的 shape、计时调用和 eviction 一致，但输入初始化序列不同，所以这些数字是当前性能位置，不是严格同输入协议的最终表。

### 性能差距在 16 项中的归因

| Kernel | 已确认的决定 | 剩余差距 |
|---|---|---|
| q8_0 | C12 最小 LMUL 在 K1 实测胜出；C13 为 32 logical lanes；输入 load 可共享 | D21 f32→i8 饱和 RNE 指令链仍长；D26 narrowing 与 store 的局部安排仍弱 |
| q8_1 | 在 q8_0 基础上，C18/D21 已选择 widening reduction 并生成 `vwredsum` | C18 reduction handoff 与 D26 scale/sum/store 安排仍不及手写路径 |
| q8_K | C11-C13 已得到外层 256 与内层 16 的不同物理 strip；D21 使用 widening reduction；K1 实测 separate max/min 胜过 shared | C17 subgroup partial layout、C18 subgroup reduction placement、D26 16-element subview 与外层 256 block 的 handoff/schedule 是主要差距 |

当前没有证据把差距归因于 D20 unpack、D23 byte interleave、D24 stride/alignment 或 D25 prefetch。

### 为什么全量在这里停止

真实 GGML Q4 布局揭示了一个 spec/encoding 问题，而不是候选或 cost model 问题。

当前 spec 声明：

```python
q: u4[256] @ nibble(lo_first)
```

当前 frontend/canonical declaration 只保存 field shape、整体 bit offset、storage bits 和 `nibble:lo_first`。若按线性 packed array 解释，它表示：

```text
logical q[2j]   → byte[j].low
logical q[2j+1] → byte[j].high
```

GGML 的实际关系是：

```text
q4_0 / q4_1:
logical q[j]      → byte[j].low
logical q[16 + j] → byte[j].high

q4_K，每 64 个 logical element:
logical q[64g + j]      → byte[32g + j].low
logical q[64g + 32 + j] → byte[32g + j].high
```

Q4_K 的 `sc/m` 还使用跨字段的 6-bit 重排，不是两个简单线性 `u6[8]` bit 数组。

这意味着当前 Encoding 没有持久表示“逻辑数组索引 → storage bit address”的 group/permutation。让 emitter 根据 Q4_0、Q4_1 或 Q4_K 名称恢复关系会违反本轮禁止的格式 route；把现有 synthetic runtime 的线性布局继续称作 GGML baseline 则会产生数值语义错误的性能结果。

因此第四轮在进入 48 条 vec-dot、48 条 dequantize 和依赖它们的 104 条 MUL_MAT 前停止，没有修改 spec 的树，也没有把旧 CSV 冒充本轮全量结果。

### 当前覆盖状态

| 家族 | 本轮真实新主线运行 | 当前状态 |
|---|---:|---|
| activation quantize | K1 3/3；SG 0/3 | SG2044 SSH 不可达；K1 结果已写入性能 CSV |
| quantized vec-dot | 0/48 | Q4 packed-array语义阻塞；其他 typed encoding/std 函数也尚未迁移 |
| row dequantize | 0/48 | 同一 encoding 阻塞；lookup 局部能力已闭合 |
| MUL_MAT | 0/104 | production outer tree、encoding/std 函数和 vec-dot realization 尚未形成严格对应 |

## 最终判断

前三轮已经把项目从历史 DSL 和一次性 emitter 推进成“canonical Kernel IR → 临时约束问题 → 联合 assignment → intrinsic C/asm”的真实编译主链。第四轮的 q8 数据说明，对于 encoding 已经准确、数值树已经写清的 primitive，当前与手写 intrinsic 的差距约为 6% 到 29%，主要属于 C/D 候选、资源与局部 schedule 的工程问题。

但完整量化语料还不能给出一个统一差距百分比。Q4 packed-array 的索引布局当前无法由 spec 的 Encoding 唯一表达，这是原理性的语言/类型缺口；在该关系被冻结前，继续调 LMUL、microtile 或 emitter 都不能形成可信 baseline。
