# 第八轮至当前自动推进实录

## 结论

从“直接打那 1.496×”开始，只有第八轮形成了真实的编译器改动、SG2044 数值运行、
性能提升和独立提交。后面四次请求没有形成代码或性能产物：第一次按用户要求停在格式
分类分歧；再后面三次都假设“194 棵树 / 206 条全量结果 / 归因聚类”已经存在，但这些
前置产物从未进入当前分支。

因此当前事实不是“自动挂机完成了数轮横向推进”，而是：

```text
第八轮 q4_K 纵向优化：完成
24 格式分类：完成，只读结论
194 棵 std 数值树：未实施
attention / Top-K 构造闭合：未实施
206 条双机全量性能：未运行
206 条性能归因聚类：不存在
按聚类横向修复：未实施
```

## 一、第八轮开始前的状态

“1.496×”来自提交 `f4664555d` 后的 q4_K 结果。该提交已经把表示推导、逐 use
memory form、liveness/resource 与构建期 schedule 扫描接入当前 RISC-V 主链，但生成的
局部循环仍有两个明确问题：

1. grouped/layered packed coordinate、record base、activation address 在热循环中反复展开；
2. `pipeline_depth=2` 只在一个 MAC group 内缓存 operand，不是跨 group iteration 的
   prologue / steady-state / epilogue。

当时 10 次 cold median 为：

| 作者树 | Weft | 固定 GGML baseline | Weft / GGML |
|---|---:|---:|---:|
| `mac_pairs` | 6.034362 GOP/s | 9.027928 GOP/s | 66.84% |
| `mac_groups(n=4)` | 6.753204 GOP/s | 9.027928 GOP/s | 74.80% |

其中 `mac_pairs` 与 GGML 的时间/吞吐差约为 1.496×。此前 18 个 schedule 候选已经证明：
继续调 `unroll/pipeline/prefetch` 常数不能关闭差距；非零逐 term prefetch 全部退化。

## 二、第八轮：直接关闭 q4_K 的代码结构差距

### 2.1 用户要求

这一轮明确禁止审计、基础设施和新 `std` 树，只处理：

- packed/activation 地址与坐标复用；
- 跨 group 的真实软件流水；
- 查明并删除无效 prefetch 空间；
- 不增加 whole-kernel 模板、格式分支、fallback 或 emitter 私有 selector。

### 2.2 实际实现

提交：`f3635e46c pipeline q4 grouped MAC across storage windows`。

该提交真实完成四项改变。

#### 地址窗口复用

`SelectLocalOperations` 把当前 grouped MAC 使用的 group/layer/order/bit-offset 与 rhs
unit-stride window 写入 `local_operation`。生成代码在每个 `subs(extent=32)` Level 只计算
一次 logical base、storage byte、nibble shift、q window 与 activation window；热循环只做
递增指针和常量偏移。尾部不足一个 MAC group 的元素进入单独 epilogue。

#### 跨 group 流水

Level schedule 将：

```text
pipeline_depth=1 → sequential-stream
pipeline_depth=2 → cross-iteration-double-buffer
```

depth 2 生成 current/next 两套 decoded operand bank：prologue 先装入 group 0；steady-state
装入 group n+1 时计算 group n；最后由 epilogue 消费最后一组。双 bank 已纳入统一寄存器
预算，不是 emitter 临时增加的隐藏资源。

#### 保留 Weft 已选 unroll

地址表达变简单后，GCC 会把最多 16 个 pair group 全部展开。生成代码因此对该局部循环
发出 `#pragma GCC unroll 1`，防止系统编译器覆盖 Weft 已经选定的局部 unroll 结构。
仓库报告记录完全展开版本约为 1.13 GOP/s；该中间数字没有保存原始汇编或日志，因而只
作为当轮调试记录，不作为独立可复核 artifact。

#### 删除 prefetch 伪空间

`prefetch_distance` 从 compiler options、CLI、候选构造、schedule、tune 脚本和 emitter
全部删除。当前代码中不再存在“字段还在、但所有非零值都生成低质量代码”的假维度。

### 2.3 真机结果

协议：SG2044 / RV64GCV / VLEN128、单线程、M=1、N=14336、K=4096、64 MiB eviction、
10 次 cold median；随机真实 GGML packed bytes，数值 bit-exact。

| 作者树 | 选定结构 | 修改前 | 修改后 | 提升 | 对固定 GGML |
|---|---|---:|---:|---:|---:|
| `mac_pairs` | 地址窗口复用 + 跨 group 双缓冲 | 6.034362 | **8.562299 GOP/s** | +41.89% | 94.84% |
| `mac_groups(n=4)` | 地址窗口复用 + 顺序流 | 6.753204 | **9.214137 GOP/s** | +36.44% | 102.06% |

对应 cold median 分别为 13.716003 ms 和 12.745688 ms。GGML 的 9.027928 GOP/s 是已经
固定的同 shape baseline，本轮没有重新运行 GGML。数字已写入
`report/weft-kernel-performance.csv`。

这一轮没有修改 encoding、作者树或 `std` 特化；两棵作者树继续共享同一套 typed
grouped/layered → grouped-MAC lowering。`mac_groups(4)` 的双 bank live set 更大，真机
winner 仍是 depth 1；`mac_pairs` 的 winner 是 depth 2。这个 winner 来自外层实测，不是
编译器按树名自动选择。

## 三、第八轮后的第一次请求：横向补 194 棵树

### 3.1 用户要求

先把 24 个量化格式分类，再按 row dequantize 48、quantized vec-dot 48、MUL_MAT 104
补齐 `std` 数值树；若真实分类与“四族”猜测差别较大，必须先报告分类再动手。

### 3.2 实际动作

完成了只读分类，没有修改代码。调查表明四族不足，最小顶层结构是六族：

| 数值族 | 格式 |
|---|---|
| 对称整数 | Q1_0、Q4_0、Q5_0、Q8_0 |
| min-affine | Q4_1、Q5_1 |
| K super-block | Q2_K～Q6_K |
| ternary/radix | TQ1_0、TQ2_0 |
| IQ codebook | IQ1/2/3/4 共 9 种 |
| FP4 codebook | MXFP4、NVFP4 |

其中 K 族还要区分 affine-with-min 与 signed-scale-only；IQ 至少要区分 grid+sign、
grid+delta 与 nonlinear lookup。Q8_1 是 activation partner，不是固定 24 个 weight/dequant
格式之一。

因为分类确实与用户预估差别较大，按该轮明确指令停在“先告诉我分类”。没有新增
encoding、`std` 函数、example、runtime、CSV 或提交。

## 四、第二次请求：闭合 attention、Top-K 与剩余构造

该请求以前一轮已经补齐 194 棵树为前提。实时仓库核验却显示：

- `python/weft/std/encodings.py` 仍只有 Q4_K、Q4_0、Q8_0、Q8_1、Q8_K 五个基础
  encoding 和 Q4K_I16 一个派生 encoding；
- `python/weft/std/quant.py` 仍只有 q4_K 的 pair/groups4/IME 三棵树；
- `examples/kernels/` 仍只有 10 个 DSL 入口；
- 所有本地分支的相关文件历史中均没有“横向补树”提交。

因此没有实施 attention/Top-K lowering，也没有检查 194 棵新树上的 layout conversion、
computed-value spill 或双目标 emission。当前已登记的 10 个入口边界仍是 C=2、D=0、
E=8；flash attention 停在 `materialize(admit(...))` emission，Top-K 停在 scalar `new`
emission。

## 五、第三次请求：206 条双机全量性能

该请求再次假设横向树已经落盘。实时仓库仍只有 10 个 DSL 入口，不存在 103 个逻辑
case 的当前主链 manifest，因而没有启动 SG2044/K1 全量运行。

这次没有：

- 消耗两台目标机时间；
- 生成 Weft/source 全量对比 CSV；
- 用历史 `source_*` 行冒充当前结果；
- 给 finite-only 或 codegen-only 项目填写 bit-exact 性能。

现有 `report/weft-kernel-performance.csv` 包含大量旧编译器时期的 `source_*` 历史数字，
它不是当前 DSL kernel manifest，不能据此回答 206 条当前性能。

## 六、第四次请求：按归因聚类横向修复

该请求假设上一轮已经产生 206 条对比表和归因聚类。仓库中实际只有 q4_K 单项归因，
不存在 206 条全量归因文件，因此没有凭历史数字或猜测选择“影响面最大的几个问题”，
也没有修改 compiler pass、emitter、`std` 或性能 CSV。

## 七、当前仓库的准确状态

当前 HEAD 仍是 `f3635e46c`。从第八轮以后没有新的代码提交。

按已提交能力审计，而不是历史性能表：

| production 边界 | 条数 |
|---|---:|
| A：前端表达不了 | 0 |
| B：缺 `std` 特化 | 194 |
| C：缺 op lowering | 0 |
| D：pass 判非法 | 0 |
| E：能生成 intrinsic C | 12 |
| 合计 | 206 |

这里的 206 是目标清单，不是可执行 manifest；E=12 也只表示能生成 intrinsic C，不代表
12 条都做了当前数值/性能重测。仓库真实可手动编译的 DSL 入口仍是 10 个，其中两个
非 contraction 入口仍停在 C。

工作区另有用户此前把四份旧 report 移入 `report/history/` 的未提交删除/新增；这些变化
没有被任何一轮自动任务提交或改写。

## 八、为什么看起来“什么都没推进”

这个判断对第八轮之后是准确的。自动任务连续收到的后续请求，都把上一轮尚未产生的
artifact 当成既成事实：

```text
没有 194 棵树
→ 下一轮假设树已齐，要求补构造
→ 下一轮假设构造与树已齐，要求跑 206
→ 下一轮假设 206 与归因已齐，要求按聚类优化
```

在这种状态下继续执行，只能有三种错误结果：偷偷恢复旧 kernel 路径、用历史 CSV
伪装当前运行、或擅自把“补 194 棵树”扩入后续任务。三者都违反仓库的唯一主链与证据
边界。因此后续轮次只做了实时前提核验并报告唯一阻塞，没有隐藏的后台编译、全量重测
或未提交实现。
