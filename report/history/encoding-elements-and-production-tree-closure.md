# Encoding `elements` 与 production 数值树闭合

## 结论

这一轮完成了两个相互依赖的闭口：encoding 现在必须显式声明一条物理记录对应的逻辑元素数，production 所需的 row dequantize、quantized vec-dot 和 `MUL_MAT` 数值树也已补齐。原能力矩阵中 194 条“缺 std 数值树”的目标均已进入唯一主链，矩阵由 `A/B/C/D/E = 0/194/0/0/12` 变为 `0/0/0/0/206`。

这里的 206 是 production 目标行数，不是 206 份独立源码：同一份符号化 DSL kernel 可以覆盖多个 shape、phase 和 target。`E` 只表示该目标可以通过 DSL、Kernel IR 和 RISC-V lowering 生成 intrinsic C；本轮没有重测性能，也没有修改性能 CSV。

## 一、起点与范围

固定 production 语料共 206 条：

| 能力族 | 计数方式 | 目标数 | 本轮开始时 E |
| --- | --- | ---: | ---: |
| `MUL_MAT` | 26 种输入格式 × decode/prefill × SG2044/K1 | 104 | 4 |
| quantized vec-dot | 24 个 typed pair × SG2044/K1 | 48 | 2 |
| activation quantize | Q8_0/Q8_1/Q8_K × SG2044/K1 | 6 | 6 |
| row dequantize | 24 种格式 × SG2044/K1 | 48 | 0 |
| 合计 |  | 206 | 12 |

开始时的 194 条缺口全部落在 B：前端能够表达 encoding 与计算树，后端已有相关通用 op lowering，但 `std/` 尚未给出 production 数值分解。它们不是 194 个互不相关的编译器问题，因此本轮没有按 target 行复制实现，而是补齐共享数值骨架、局部重构片段和各格式的 encoding 声明。

本轮覆盖的 24 种 weight/row 格式为：

```text
Q1_0
Q4_0 Q4_1 Q5_0 Q5_1 Q8_0
Q2_K Q3_K Q4_K Q5_K Q6_K
IQ1_S IQ1_M IQ2_S IQ2_XS IQ2_XXS IQ3_S IQ3_XXS IQ4_NL IQ4_XS
TQ1_0 TQ2_0
MXFP4 NVFP4
```

Q8_1 与 Q8_K 是 activation partner encoding，不作为额外 weight 格式计数。

## 二、Encoding 的逻辑 extent 已成为显式布局事实

### 2.1 新合同

每个 concrete encoding 现在必须显式提供：

```python
elements = N
```

它只回答“一条 encoded record 对应多少个逻辑元素”，与 block、sub、字段宽度、bit/byte order、grouped/layered packing 一样属于布局事实。它不包含 scale 公式、zero/min 修正、codebook、radix 或任何数值解码语义。

### 2.2 删除错误推断

原先前端会从最长字段的布局范围猜逻辑 extent。这个推断对真实 GGML record 不成立：字段可能只覆盖元数据、分组 scale 或 packed 子空间，最长字段长度并不等于记录的逻辑元素数。

这一轮删除了该推断路径：

- encoding 缺少 `elements` 时，前端直接拒绝；
- canonical IR 显式保存 `elements`；
- verifier 和 lowering 只消费这一事实，不再重建；
- 24 个 production encoding 以及 Q8_1/Q8_K activation encoding 全部显式声明自己的逻辑 extent；
- spec 的 Encoding 一节同步明确：`elements` 必填，不能由字段长度推断。

因此，抽象 encoding family 与 concrete layout instance 仍然分离；只有已经实例化的 encoding 才携带确定的 `elements`。

## 三、数值树的组织方式

194 条缺口最终没有形成 194 份复制代码，而是由六个数值骨架、八个局部重构片段和 24 个 encoding 声明组合出来。

### 3.1 六个数值骨架

| 骨架 | 表达的局部数值关系 | 主要使用者 |
| --- | --- | --- |
| `symmetric_integer` | 单 scale 的对称整数值 | Q4_0、Q5_0、Q8_0 等 |
| `min_affine` | scale 与 minimum 的仿射值 | Q4_1、Q5_1 等 |
| `k_superblock` | super-block 内 per-sub scale/min 与外层 scale | Q2_K 至 Q6_K |
| `ternary_radix` | 三值 radix packed 数据 | TQ1_0、TQ2_0 |
| `iq_codebook` | 索引、符号/修正与只读 codebook | IQ 系列 |
| `fp4_codebook` | FP4 codebook 与 exponent scale | MXFP4、NVFP4 |

这些函数只表达局部数值关系，不拥有 outer traversal、workspace、activation quantization 或完整 kernel。

### 3.2 可复用局部片段

以下重构只实现可复用的局部关系，没有按格式复制完整 decode 树：

```text
extract_bits
high_bit_plane
signed_scale
grid_sign
grid_delta
nonlinear_lookup
exponent_scale
radix3_digit
```

例如 Q5_0 与 Q5_1 共用同一 high-bit plane 拼接；IQ 家族复用 lookup、sign 与 delta 片段；FP4 家族复用 exponent-scale 片段。格式差异保留在 encoding 与调用这些片段的普通 DSL 函数中，没有进入 backend route。

Codebook、grid、sign 和 scale table 作为只读 `View` 参数进入 kernel。没有为此增加 DSL 常量数组构造，也没有把上游表隐藏进 emitter。

## 四、三族 production 数值树

### 4.1 Row dequantize

24 个 row-dequantize 入口都由相应 encoding 和共享数值骨架生成。每个入口读取一条真实 packed record，按 encoding 的 logical-index-to-storage mapping 取得字段，再计算对应的浮点逻辑元素。

这一步验证的是布局与解码边界：如果 row dequantize 与 GGML reference 对不上，错误只能位于 encoding mapping 或数值树，不能由 contraction、activation quantize 或 outer traversal 掩盖。

完成后能力矩阵变化为：

```text
B: 194 → 146
E:  12 →  60
```

### 4.2 Quantized vec-dot

24 个 typed pair 均已建立。Q4_K 已有的入口保留在同一模型中，其余 23 个 pair 由相同的“weight decode + activation partner + scalar contraction”结构补齐。

这一轮同时把 vec-dot 的结果改成普通 scalar SSA result，而不是由 helper 内部直接写最终输出。这样同一个结果可以被 outer `MUL_MAT`、pointwise 或其他普通 consumer 使用；写回由调用者决定，不再要求 direct-store closure。

完成后能力矩阵变化为：

```text
B: 146 → 100
E:  60 → 106
```

新增 E 是 46 而不是 48，因为 Q4_K 在本轮开始时已经覆盖 SG2044/K1 两条 vec-dot 目标。

### 4.3 Production `MUL_MAT`

补齐了 F16 和 24 种 quantized weight 的 production `MUL_MAT` DSL kernel。它们不是“vec-dot helper 外面随手套一层”：

- outer M/N traversal 由 DSL 明确写出；
- quantized 路径在 DSL 中显式完成 activation quantization；
- activation workspace 由 caller 提供并具有明确 encoding；
- 每个输出调用返回 scalar SSA 的 typed vec-dot，再显式 commit 到 Y；
- F16 路径显式把 F32 activation staging/convert 为 F16 workspace，再进行 block accumulation；
- codebook 类格式继续通过只读 `View` 参数传入表，不由 backend 私自构造。

同一个符号化 M/N/K kernel 同时覆盖 decode 与 prefill shape，也同时面向 SG2044 和 K1；phase 和 target 只实例化 meta/target facts，不复制算法树。

完成后能力矩阵变化为：

```text
B: 100 →   0
E: 106 → 206
```

新增的 100 条为 25 种原先缺失的 `MUL_MAT` 格式（F16 + 24 quant）× 两种 phase × 两个 target。原有 F32 的四条目标不重复计入。

## 五、组合后暴露并修正的共享 lowering 问题

这些问题只在更大的 production 树中出现，但修复均落在通用语义或 lowering 上，没有按格式建立路径。

### 5.1 Nested encoded record 的地址

嵌套 slice 原先没有完整消费 `SliceInfo.localOffsets`，导致 outer record 与 inner encoded field 组合时地址少了一层局部偏移。修复后，地址生成统一使用 slice 已经携带的各级 offset，Q4/Q5/IQ 等格式不再各自重建 record address。

### 5.2 单元素 slice 与顺序 scalar load

Activation quantization 的有序扫描需要从一元素子域读取、比较并更新 scalar state。lowering 现在能够：

- 对 sequential one-element domain 生成 dense scalar load；
- 对单元素 slice 正确提取值；
- 对未被表达式使用的 Level base 仍生成合法控制流拼写。

Top-level scalar loop仍然是作者写下的有序控制，没有被自动转成 VLA。

### 5.3 F16/F32 value handoff

F16 production 路径需要 F32 activation → F16 staging → F32 accumulation。RVV spelling 补齐了 F16/F32 双向 cast；dense commit 也改为从 value axes 找非-lane 输出轴，而不是从 memory axes 猜轴身份。

### 5.4 Q8_K activation quantization 的精确语义

Q8_K 的绝对最大值扫描改为与 GGML 一致的 signed-absolute-max、first-wins 规则。它影响最终 scale 和量化字节，因此不能用近似相等掩盖。独立 Q8_K activation quantization 与 production `MUL_MAT` 都使用同一 DSL 实现。

### 5.5 浮点 contraction 拼写

GGML reference 在 SG2044 上的相关内层表达实际发生 fused multiply-sub。为保证同一算法树的逐位结果，只有生成 kernel C 使用 `-ffp-contract=fast`；runtime/reference 继续关闭 contraction。没有启用 fast-math，也没有允许 reassociation。

这只是把已选浮点操作忠实拼写为目标编译器可融合的 C 表达，不是按格式选择另一套算法。

## 六、最终能力矩阵

### 6.1 轮次变化

| 节点 | A | B | C | D | E |
| --- | ---: | ---: | ---: | ---: | ---: |
| 本轮开始 | 0 | 194 | 0 | 0 | 12 |
| row dequantize 完成 | 0 | 146 | 0 | 0 | 60 |
| quantized vec-dot 完成 | 0 | 100 | 0 | 0 | 106 |
| production `MUL_MAT` 完成 | 0 | 0 | 0 | 0 | 206 |

分类含义：

- A：前端无法表达；
- B：缺少 std 数值树；
- C：缺少 target op lowering；
- D：physical pass 无合法实现；
- E：唯一主链能够生成 intrinsic C。

### 6.2 最终按能力族分布

| 能力族 | A | B | C | D | E |
| --- | ---: | ---: | ---: | ---: | ---: |
| `MUL_MAT` | 0 | 0 | 0 | 0 | 104 |
| quantized vec-dot | 0 | 0 | 0 | 0 | 48 |
| activation quantize | 0 | 0 | 0 | 0 | 6 |
| row dequantize | 0 | 0 | 0 | 0 | 48 |
| 合计 | 0 | 0 | 0 | 0 | 206 |

这个结果说明本轮定义的 production 范围内没有剩余语言表达缺口、std 树缺口、op-lowering 缺口或资源判非法项；它不等价于“206 条都已达到手写性能”。

## 七、真实 repro 与证据边界

本轮使用既有手工 repro 入口，没有建立 test 目录、fixture 或 case matrix。

| 能力 | 手工命令形式 | 目标 | 输入与 reference | 本轮结果 |
| --- | --- | --- | --- | --- |
| row dequantize | `examples/run/weft-row-dequantize.sh <target> <format>` | SG2044、K1 | 随机 packed record；GGML row-dequant reference | 24 格式逐元素 bit-exact |
| quantized vec-dot | `examples/run/weft-quantized-vec-dot.sh <target> <format>` | SG2044、K1 | weight/activation 随机字节；GGML vec-dot reference | 24 typed pair 的结果 bit-exact |
| production `MUL_MAT` | `examples/run/weft-mul-mat.sh <target> <format>` | SG2044、K1 | M=2、N=2、K=该格式一条完整 record；GGML quantize/vec-dot reference | 25 个新增格式的 workspace 与输出 bit-exact |
| Q8_K activation quantize | `examples/run/weft-kernel.sh <target> q8_K_quantize 1` | SG2044、K1 | M=128、K=14336；GGML quantize reference | packed workspace bit-exact |

这几条命令都实际经过：

```text
DSL source
→ canonical Kernel IR
→ RISC-V physical passes
→ intrinsic C
→ 目标机 C/C++ 编译
→ 目标机执行
→ GGML reference 数值比较
```

证据边界需要明确：

- row dequantize 和 vec-dot 覆盖的是一条完整 record/一次 contraction；
- production `MUL_MAT` repro 的 M/N 很小，用于验证 activation workspace、outer traversal、typed vec-dot composition 与 store，不代表真实模型 shape 的性能；
- 同一符号化 kernel 能为 decode/prefill 与两台 target 生成代码，因此能力矩阵可计入对应目标；本轮没有逐一执行 M=1/128、N=4096、K=4096 的 production shape；
- 本轮没有性能重测，没有产生新的 GOP/s，也没有改写 `weft-kernel-performance.csv`；
- 本轮运行结果是当轮真实执行结果，临时远端产物和日志未保存进仓库。

## 八、设计边界检查

本轮闭合 194 条目标时没有引入以下内容：

- kernel 名、q-format 名或 target 名驱动的 backend route；
- whole-kernel emitter 或完整手写 kernel；
- GGML、`source/` 或 `materials/` runtime 调用；
- legacy、scalar 或旧 emitter fallback；
- emitter 内重新选择 encoding、decode tree、LMUL 或算法结构；
- 为 codebook 特设的语言常量数组；
- 从 record 字段长度猜 logical extent 的兼容路径。

格式差异留在“encoding 声明 + 普通 DSL 数值函数”；outer traversal、workspace 和 activation quantization 留在 production DSL kernel；通用地址、slice、cast 与 value handoff 留在 lowering。这个归属与 spec 的“作者树不可由编译器改写”边界一致。

## 九、仍显得别扭的地方

Q8_K 的 signed-absolute-max 扫描目前使用一元素 `L.subs` 加一元素 reduce，把向量域显式收缩为 scalar state。这是因为当前 scalar-role admit 不能直接承载该路径。它不造成 A/C 缺口，也不影响数值正确性，但源码表达比“从一元素 slice 取得 scalar”更绕。

本轮闭合的是表达、std tree、lowering 和真实小规模执行链。真实模型级 `MUL_MAT` shape 的运行时间、缓存行为、activation workspace 成本以及与手写 intrinsic 的差距仍未在这一轮测量，因此不能从 `E=206` 推导性能结论。
