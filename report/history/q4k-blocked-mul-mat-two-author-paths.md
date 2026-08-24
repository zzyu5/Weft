# Q4_K blocked MUL_MAT：持久派生布局与现场 pack 双路径闭合

## 报告边界

本轮只处理 Q4_K，不把同一修改并行铺到其余量化格式，也不做 206 条全量重测。

本轮要回答的是：production MUL_MAT 能否由作者显式选择两种不同的数据组织，同时让后端复用同一套 encoded materialize、表示传播、资源计算和 widening contraction 能力：

1. 调用前已经完成持久重排的派生 encoding；
2. 输入仍是 canonical Q4_K，在 kernel 内按当前 block 生命周期现场 pack。

`source/` 和 `materials/` 均未进入生成代码或运行路径，没有增加 GGML 调用、历史实现调用或 fallback。

## 一、问题的真实性质

此前 9.214 GOP/s 与 0.327 GOP/s 不是同一作者程序：

| 项目 | 9.214 GOP/s 路径 | 0.327 GOP/s 路径 |
|---|---|---|
| 权重表示 | 跨 16 行持久交错的派生 encoding | 单行 canonical Q4_K View |
| 输出组织 | kernel 内一次处理 16 个输出 | runtime 在 kernel 外逐行调用 |
| 行循环 | kernel 内 | runtime 外 |
| ABI | 接受预先 packed 的权重 | 接受 canonical 权重 |
| 数据复用 | activation 与 decode window 跨输出复用 | 一次只服务一个输出 |

因此，std 函数无权把后一种程序自动改成前一种程序。改变 ABI、逻辑值集合、输出 cohort 和持久布局属于作者决策，符合 spec 2.2 的边界。

同时已经确认，第八轮的 grouped-MAC lowering 读取的是 grouped/layered 派生布局及 typed operand facts，不读取 Q4_K 名称。它具有布局层面的泛化性，只是 canonical base layout 没有提供它需要的数据组织。

## 二、作者侧的两种正式程序

### 2.1 持久派生布局

作者通过 `Q4K_I16` 明确选择跨 16 行的持久派生 encoding。生成结果同时包含：

- packed storage size 查询函数；
- canonical Q4_K 到派生布局的 pack 函数；
- 接受派生布局的 production kernel。

decode 与 prefill 使用两个作者特化：

- decode 是 GEMV 退化形态：普通有序行循环、16 输出 cohort、K block 与 grouped MAC；
- prefill 显式拥有 NC/KC/MC、MR×16 accumulator、activation 跨输出复用和完整 K 生命周期。

两者不是后端按 phase 选择的路径，而是两个明确的 std 函数。runtime 只负责选择要编译、调用哪个作者程序。

持久 pack 在计时循环外完成，因此性能数字表示 packed operand 已经存在时的 kernel 时间。这个边界对应模型加载期或跨调用复用的长期 packed storage。

### 2.2 canonical 输入的现场 pack

现场路径继续接收 canonical `View[Q4_K, (N, K)]`。作者树明确写出：

```text
quantize activation
→ scalar ordered output initialization
→ NC
  → KC
    → materialize(pack(W[nc, kc], along="n"))
    → MC
      → MR × 16 accumulators
      → quantized outer contraction
      → write back partial Y
```

`wp` 在一个 `(nc, kc)` 内创建，位于 MC/MR 循环之外，因此同一个临时 encoded panel 被当前 M cohort 的全部输出复用。Y 是作者明确使用的跨 KC 算法状态；每个 KC tile 从 Y 回读、更新并写回。

零初始化使用普通有序标量 `for`，没有用无意义的 pointwise 操作强迫 wide lowering。现场 pack、初始化及其临时存储全部位于 kernel 调用和计时范围内。

## 三、后端形成的共享能力

### 3.1 Ephemeral encoding 保留逻辑 encoding

`pack` 不再创造一个无法对应源布局的字符串 family。它保留源 encoding family，并以 `ephemeral + packed.along.<axis>` 记录当前作者选择的临时布局身份。

这样，临时 packed View 仍然拥有 Q4_K 的字段、位宽、grouped/layered mapping 与 record 大小；变化的只是 primitive-local 数据组织，不是数值语义。

### 3.2 表示沿普通 use-def、Level handoff 与初始化 edge 传播

表示 pass 现在联合传播：

- lane axis 与 register axis；
- physical lanes、SEW、LMUL；
- stream parts 与 register parts；
- Level births/handoff 后的 carried value；
- `new(init=admit(...))` 的 memory-to-register 初始化链。

此前 lane 表示能到达 loop-carried accumulator，却不能从 handoff 再反向到初始化 admit，因此 Y 回读被 memory pass 错判为 scalar address。修复后，handoff 传播与普通 use-def 传播位于同一个 pass 内的不动点中；memory pass 可以从 transferred value 的最终表示得到 runtime-strided load。

Level 的物理 lane 宽度只读取该 Level 子程序 `level_path` 内真正使用相同 axis 的值。activation quantize 中的 K-lane 不再污染后续 contraction 的 K-block/K-sub Level。

### 3.3 Encoded materialize 的决定在 pass 中闭合

materialize pass 现在为每次 pack 使用明确产生：

```text
source rank
row axis
record axis
lane axis
interleave rows
record storage bits
logical elements per record
source encoding family
```

rank-2、row/record axis 和 multi-row cohort 是 selection legality，不由 emitter 临时猜测。emitter 只核对 IR 与已选决定一致，然后机械生成局部 panel。

同一个 materialize 机制同时保留 dense f32 pack 和 encoded record pack；新增的是 encoded source 的合法输入范围，不是 Q4_K 专用 materializer。

### 3.4 Quantized outer contraction 复用 dense 的 blocked 骨架

量化 outer contraction 由以下事实进入 widening-MAC realization：

- 显式 `outer_contract` primitive；
- operand/result 的 logical SEW、signedness 和 lane/register axis；
- field 的 grouped/layered layout；
- decode 后的 typed widening chain；
- accumulator parts、unroll 与 Level-local pipeline；
- target vector resource budget。

后端没有 Q4_K、kernel symbol、VLEN128/VLEN256 或 target 型号分支。dense 和 encoded operand 使用同一套 axis/register/resource 骨架；格式差异停留在 encoding 声明与局部字段数值关系。

### 3.5 Emitter 只拼写已选结果

Emitter 消费 pass 已经给出的：

- encoded pack axes 和 record facts；
- partial term 数量及 LMUL；
- accumulator/register parts；
- memory form；
- pipeline depth 和 operand buffers；
- grouped/layered extract mapping。

Q4_K hot loop 的 i16/u16 标量字段读取改为显式 little-endian byte 拼接。此前 `memcpy` 拼写使 GCC 在含 RVV 的函数中生成依赖 `vlenb` 的动态栈字节重组；改变拼写后消除了这段无关的 stack traffic。这里改变的是同一个已选 load 的 C 表达，不是 emitter 新增结构判断。

## 四、数值验证边界

所有真机结果均使用随机 canonical Q4_K bytes；activation 由 GGML Q8_K reference quantizer 生成。reference 按 GGML Q4_K storage mapping 取出 scale、minimum、nibble 和 Q8_K bsum，再严格按照作者树的 grouped accumulation 与 f32 step 边界计算。

因此 `bit-exact` 表示：

```text
同一随机 encoded input
同一作者数值树
同一 f32 contraction 设置
→ reference 与 Weft 输出逐 bit 相同
```

它不是对另一种浮点结合顺序作 bitwise 等价声明。kernel 与 runtime 均使用 `-O3 -ffp-contract=fast`；reference 中需要保持作者树 step 边界的位置使用显式 volatile f32 操作，避免系统编译器越过该边界重新结合。

## 五、SG2044 真实结果

固定 shape：`N=4096, K=4096`。decode 使用 `M=1`，prefill 使用 `M=128`。

| 作者路径 | phase | repetitions | cold median | Weft GOP/s | 数值 | pack 计时归属 |
|---|---:|---:|---:|---:|---|---|
| 持久 Q4K_I16 | decode | 7 | 3574.056 us | 9.388334 | bit-exact | kernel 外，不计入 |
| 持久 Q4K_I16 | prefill | 3 | 386639.155 us | 11.108464 | bit-exact | kernel 外，不计入 |
| canonical + local pack | decode | 3 | 26617.460 us | 1.260617 | bit-exact | kernel 内，计入 |
| canonical + local pack | prefill | 3 | 408811.396 us | 10.505987 | bit-exact | kernel 内，计入 |

手工 repro：

```bash
examples/run/weft-mul-mat.sh sg2044 q4_k_i16 decode 7
examples/run/weft-mul-mat.sh sg2044 q4_k_i16 prefill 3
examples/run/weft-mul-mat.sh sg2044 q4_k_local decode 3
examples/run/weft-mul-mat.sh sg2044 q4_k_local prefill 3
```

持久布局的 production decode 达到 9.388 GOP/s，超过本轮给定的 9.214 GOP/s 参照点约 1.9%。这说明第八轮的 grouped-MAC 能力已经进入 production kernel，而不是只能服务独立 vec-dot repro。

现场 pack 在 prefill 达到持久路径约 94.6% 的吞吐，说明 pack 成本可被 M=128 的输出复用摊薄；decode 只有 1.261 GOP/s，说明 M=1 时现场重排成本无法摊薄。两条路径都保留是必要的，选择权属于作者，而不是 runtime fallback 或后端自动改树。

## 六、证据边界与结论

- SG2044 上四个入口均完成真实交叉编译、执行和 bit-exact 对照。
- 相同两种结构在 K1/VLEN256 target profile 下均能生成 intrinsic C；本轮没有 K1 真机性能数字。
- 性能 CSV 未更新。持久派生路径改变了权重 ABI和预处理归属；现场 pack 又把 pack 成本放进 kernel。它们都不能无说明地覆盖旧 canonical 单行 MUL_MAT baseline。
- 本轮没有发现必须让编译器改变作者逻辑值集合或 Level 归属才能达到高性能的反例。spec 2.2 没有被证伪：作者选择 cohort、blocking、persistent layout 与 pack 生命周期；编译器解决这些显式边界内部的 lane/register mapping、memory realization、局部 packing、resource 与 pipeline。
- 本轮完成的是一个格式上的纵向闭合。新增格式是否只需 encoding 声明与对应 std 数值函数，尚未由其他格式迁移证明，本报告不把它写成全量能力。

