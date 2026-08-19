# Core-local command DSL 重构报告

## 本轮范围

本轮从 `report/design.md` 给出的方向出发，完成了 Weft 第一阶段的实际切换：冻结新的编程
模型，替换 Python DSL 公开入口，扩展 canonical Kernel IR，迁移现有 DSL examples，并将
`doc/` 全部改写为与新模型一致的唯一规范。

本轮对应提交：`4a3013be8`（`refactor DSL around core-local commands`）。该提交修改 53 个
文件，其中 17 份规范文档、30 份 DSL example，以及 Python frontend、Kernel IR 与 verifier；
总计新增 1864 行、删除 1736 行。

本轮没有继续扩展 kernel 数量，没有修改 `source/` baseline，没有调用或复制 `materials/`
实现，也没有建立兼容层或第二条 lowering。

---

## 最终冻结的编程模型

Weft 被定义为一门面向非 SIMT 处理器的 **single-controller, multi-engine operator DSL**：

> 一个 worker/hart 从入口持续执行到返回。作者用普通有序控制组织完整算子，并向当前 core
> 的 scalar、vector、matrix 与 data-movement 能力发出目标无关的局部命令。作者拥有算法
> traversal、blocking、storage 和生命周期；编译器拥有局部命令内部的物理映射和 ISA 实现。

项目只保留一门 DSL。高层局部计算命令与较细的轴、值、访存构造是同一语言中可混用的两个
抽象高度，不拆成 Kernel DSL 与 Realization DSL：

```text
Core-local blocked Python DSL
→ canonical Kernel IR
→ one target-specific lowering
→ intrinsic C / local asm
→ system C compiler
```

### 用户明确拥有

- worker 的 `for / while / if`；
- outer traversal、algorithmic/cache blocking 与 loop order；
- accumulator 与 state 的创建、更新和跨循环生命周期；
- workspace、staging、persistent packed layout 与算法 variant；
- pointer、index、predicate、effect 与 alias；
- `W.vla`、`W.vdot`、`W.gemm`、reduce/scan、lookup/decode、quant command 的局部语义。

### 编译器明确拥有

- logical axis 到 sequential/lane/register/unroll/fragment 的映射；
- SEW/LMUL、vector shape、register microtile 与 multiple accumulators；
- unit/strided/indexed/segment memory form；
- value handoff、reload、rematerialize 与短生命周期 local packing；
- loop-local pipeline、prefetch 与局部调度；
- RVV、IME 或其他已接入 extension 的局部 realization；
- intrinsic C 与 typed inline asm 拼写。

普通 scalar loop 不会自动成为 VLA，普通 multiply/add/reduce 图不会被识别成 matmul、online
softmax 或量化 primitive。编译器不得创建作者没有写出的 outer loop、blocking、workspace、
persistent packing 或完整算法 variant。

---

## 为什么是一门 DSL，而不是多个 DSL

本轮明确放弃“普通 Kernel DSL + 专家 Realization DSL”的双语言方案。双语言会产生两套类型、
两份控制语义、一条跨层 ABI，并让 realization 很容易重新拥有 outer traversal 和 storage，最终
恢复 whole-kernel template。

同时也没有把 Weft 压成只有 `W.vla/W.axis/load/store` 的低层语言。Matmul、scan、summary、
lookup 和 packed quant relation 不能从普通 SSA 无歧义恢复，必须由作者显式授权。

因此最终采用：

```text
同一 DSL
├── 局部计算命令
│   ├── vdot / gemm
│   ├── reduce / scan / summary / sort
│   ├── lookup / decode
│   └── W.quant.*
└── 可组合构造
    ├── scalar control
    ├── W.vla / W.axis
    ├── load / store / transfer
    ├── pointwise / cast / select
    └── ordinary SSA carry / state
```

应用作者不能继续下降到 RVV intrinsic、LMUL、register group 或 IME asm。它们只属于 target
lowering 与最低层 leaf。

---

## Python DSL 的具体修改

### 旧公开入口被直接删除

| 旧入口 | 新入口 | 语义变化 |
|---|---|---|
| `W.block` | `W.axis` | 强调创建的是有身份的逻辑轴，不是物理 block layout |
| `W.storage` | `W.buffer` | 强调 caller-owned workspace/persistent view，不是分配操作 |
| `W.dot` | `W.vdot` | 明确这是目标无关的局部 vector-dot command |
| `W.matmul` | `W.gemm` | 明确这是局部 `[M,K]×[K,N]+init` command |
| 顶层量化命令 | `W.quant.<name>` | 把 typed packed compute 组织到同一 DSL namespace |

旧名字没有 alias、deprecation、兼容入口或 fallback。Frontend handler 也同步改名，旧 Python
source 会直接失败，不会悄悄进入旧路径。

### 新增 `W.blocks`

```python
for m0 in W.blocks(m_begin, m_end, BM):
    ...
```

`W.range` 继续表示一般有序 traversal；`W.blocks` 表示作者显式选择的 algorithmic/cache block
traversal。二者都是当前 worker 的真实 scalar loop，不是隐式并行轴。

所有现有三参数、非单位步长的 block traversal 都已经迁移为 `W.blocks`，包括 GEMM、Conv、
OutProd、IME projection、quant projection 与 FWHT，共处理 15 处、9 个 example 文件。

### 新增 `W.pipeline`

```python
for k0 in W.pipeline(W.blocks(0, k, BK)):
    ...
```

它只表达作者允许在这个已存在 loop 内进行依赖合法的局部流水化：

- 不指定 stage 数、buffer 数或 prefetch distance；
- 不创建新的算法 pass；
- 不创建跨 loop workspace 或 persistent buffer；
- 顺序执行仍然是合法 realization。

Frontend 把授权保存为 `weft_kernel.for pipeline=true`。本轮没有把 stage、LMUL 或 target 名称
加入 DSL。

### 新增 `W.accumulator`

```python
acc = W.accumulator((mi, ni), W.f32, init=0.0)
```

它为 loop-carried output domain 提供自然源码表达，但 canonical 语义仍是普通 `full` 初值与
普通 SSA carry，不创建特殊寄存器对象。当前正式范围是 rank-one/rank-two f32 engine value。

### 新增 `W.transfer`

```python
W.transfer(input + i, output + i)
```

当前语义严格限定为：

- source 可读、destination 可写；
- element type、logical shape 与 axis identity 一致；
- 全部元素有效；
- frontend 展开成一次 canonical load 与一次 store。

它不表示 async copy、packing、transpose、cache placement 或隐式 allocation。

### 量化命令收敛到 `W.quant`

原有 18 个 typed quant command 继续保持各自真实数值语义，但不再作为语言根层的顶级名字。
本轮迁移 27 处量化调用、8 个 example 文件，例如：

```python
W.quant.affine_i4_i8_dot(...)
W.quant.symmetric_i4_i8_dot(...)
W.quant.iq2_s_i8_dot(...)
W.quant.q6_k_i8_dot(...)
```

`Intrinsic.name` 继续连接已有 frontend handler，`spelling` 只负责新的用户可见名称，因此没有
复制 quant lowering，也没有恢复格式 selector。

当前没有增加通用 `qgemm`。现有 packed、scale/minimum、codebook、correction 关系尚不存在一
个不丢失语义的共同 schema；强行统一只会把差异重新藏进 q-format 分支。

---

## Canonical Kernel IR 的修改

### Source 名称与 canonical 名称分离

新 source surface 进入现有稳定 canonical semantics：

```text
W.axis   → weft_kernel.block_index
W.buffer → weft_kernel.storage
W.vdot   → weft_kernel.dot
W.gemm   → weft_kernel.matmul
```

Canonical `block/storage/dot/matmul` 描述内部语义，不再被当成 Python 公开 API。这样避免为了
改善 source 分层而复制 IR op 或 target lowering。

### `weft_kernel.for` 增加正式语义

ForOp 新增两个 required attribute：

```text
traversal = "ordered" | "blocks"
pipeline = false | true
```

Verifier 检查 traversal 值，并拒绝常量为零的 step 与负数 block extent。Carry 的类型、逻辑轴
和 dynamic extent identity 仍由原 verifier 检查。

这些属性属于 canonical semantics：它们保存作者 traversal 类型与 pipeline 授权。具体 pipeline
schedule、buffer 数和 register organization 仍是 target lowering 的瞬态决定，不进入 IR。

### 没有增加新的持久层

本轮没有增加 Physical IR、Selected IR、provider registry 或 realization bytecode。LMUL、
microtile、packing、resource、fragment 和 candidate 继续只存在于一次 target lowering 内。

---

## Examples 的全仓迁移

本轮修改 30 份 active DSL example，覆盖 attention、dense/dot、gather、IME、permutation、
quantization、rotation、selection 与 vision。

机械迁移规模为：

- `W.block → W.axis`：124 处，19 个文件；
- `W.storage → W.buffer`：63 处，18 个文件；
- `W.dot → W.vdot`：6 处，6 个文件；
- `W.matmul → W.gemm`：2 处，2 个文件；
- 顶层 quant command → `W.quant.*`：27 处，8 个文件；
- 三参数 block traversal → `W.blocks`：15 处，9 个文件。

另外对代表程序做了语义迁移，而不只是改名：

### Blocked GEMM

- M/N traversal 改为 `W.blocks`；
- logical M/N/K domain 改为 `W.axis`；
- accumulator 改为 `W.accumulator`；
- K-loop 从手写 `while` 改为 `W.pipeline(W.blocks(...))`；
- 局部 product 改为 `W.gemm`；
- 原来的 M/N/K blocking、地址、mask、accumulator lifetime 与 epilogue 保持不变。

### Copy

`copy_f32` 的直接 `store(load(...))` 改为 `W.transfer`，证明 data-movement command 可以进入
普通 VLA region，而不需要新 canonical op 或特殊 emitter。

迁移结束后，active examples 中不存在旧 `W.block/W.storage/W.dot/W.matmul`、顶层 quant
调用或三参数 `W.range`。未修改时间戳的其他 example 没有使用被替换的构造，因此已经天然
符合新 DSL；“全仓迁移”不等于要求每个 Python 文件产生无意义 diff。

---

## 文档重构

`doc/` 的 14 份核心规范被完整改写，另同步更新 3 份 kernel 文档。新的阅读结构从
`doc/index.md` 开始：

1. `doc/dsl/model.md`：single-controller/multi-engine 根模型；
2. `doc/dsl/python.md`：真实可写语法；
3. `doc/dsl/vla-memory-and-blocks.md`：scalar、memory view、engine value 与逻辑轴；
4. `doc/dsl/storage-and-lifetime.md`：external/workspace/persistent/private temporary；
5. `doc/dsl/structured-compute.md`：vdot/gemm/state/lookup/quant 命令边界；
6. `doc/dsl/numerics.md`：order、math、validity 与 quant numerics；
7. `doc/dsl/api.md`：只列当前 frontend/IR/target 同时闭合的入口；
8. `doc/compiler/*`：canonical IR、target decisions、emission、tuning 边界与真实 repro。

完整设计论证写入 `report/design.md`，包括：

- 为什么不是 Triton-CPU；
- 为什么不直接复用 TileLang 的 GPU storage/thread 模型；
- 相比直接写 C/intrinsic 省略了哪些机器工作；
- 为什么是一门渐进式 DSL，而不是多 DSL；
- GEMM、vdot、quant 与 extension 的边界；
- compiler 的唯一推导、结构性选择与参数性选择；
- 反方审查、代价与设计失败条件。

`report/design.md` 是本轮推理快照；冻结后的规范已经进入 `doc/`，后续不应把 report 当作第二
authority 持续同步。

---

## 构建与真实执行

### 编译器构建

执行：

```bash
cmake --build build -j2
```

TableGen、Kernel dialect/verifier 与 RISC-V target library 全部重新构建成功。

### 唯一真实 repro

执行：

```bash
./examples/run/weft.sh k1-rvv256 blocked_gemm_f32 decode 1
```

实际主链为：

```text
blocked_gemm_f32.py
→ Python frontend
→ canonical Kernel IR with blocks/pipeline attrs
→ RISC-V intrinsic C
→ K1 clang-18
→ K1/VLEN256 target execution
```

真机结果：

```text
kernel=f32_dense_projection
model_shape=Llama-8B.hidden_projection
phase=decode
M=1
N=4096
K=4096
repetitions=1
max_absolute_error=0
max_relative_error=0
weft_ms=20.764451
weft_gop_s=1.615956
```

这证明代表性的 `W.blocks + W.pipeline + W.axis + W.accumulator + W.gemm` 已经通过唯一主链
生成并执行真实 RISC-V artifact，数值误差为零。

### 为什么没有更新性能 CSV

本轮只使用 `repetitions=1` 做结构重构后的真实 correctness repro。`20.764451 ms` 是本次单次
执行的原始观测，不是按现有正式 warmup/repetition 协议得到的新性能 baseline，因此没有写入
`report/weft-kernel-performance.csv`。

这避免把一次结构验证误写成可与历史 baseline 比较的正式性能数字。

---

## 当前已经成立的能力

- Python public surface 只剩新 DSL，不存在旧 alias；
- 所有新 Intrinsic 都有对应 frontend handler；
- quant namespace 能继续命中唯一 extension handler；
- `W.blocks/W.pipeline` 的授权进入 canonical ForOp；
- `W.axis/W.buffer/W.vdot/W.gemm` 复用唯一 canonical op 与 target lowering；
- `W.transfer` 通过普通 load/store semantics lowering；
- representative GEMM 真机数值执行成功；
- build、doc、examples 与 canonical schema 描述同一编程模型；
- `source/` 与 `materials/` 没有进入 production path；
- 提交后工作区保持干净。

---

## 当前仍未实现或未验证的边界

这些内容没有在本轮报告中伪装成完成：

### Pipeline 物理调度

`pipeline=true` 已经进入 canonical IR，但当前 RISC-V emitter 仍可选择普通顺序 loop。本轮完成
的是语言授权和 IR 合同，不代表已经生成 prologue/steady-state/epilogue 或 double buffering。

### `W.blocks` 的 target 利用

`traversal="blocks"` 已被保存，当前 artifact 仍忠实生成作者的 scalar block loop。Lowering 尚未
利用该属性建立更宽的 loop-local scheduling；它不会因此改变 loop ownership。

### `W.accumulator` 范围

当前只支持 rank-one/rank-two f32 engine value。它没有独立 canonical op，也没有 source-visible
register placement。

### `W.transfer` 范围

当前只支持同 dtype、同 logical domain、全有效 copy。Async copy、transpose、packing、masked
transfer 和 target-specific DMA 不在本轮能力内。

### 自动 compile-and-measure tuner

当前 `weft-compile` 对一组 target facts/meta/backend config 执行一次 lowering；runner 可以外部
重复调用和记录结果，但仓库内没有自动枚举、测量并固化 winner 的长期 tuning pipeline。

### 全量 examples 的真机执行

本轮对 active examples 完成了 source-surface 全仓扫描和迁移，但只对 blocked F32 GEMM 做了一
条真实 K1 repro。不能把它描述成 77 个 source 文件或全部 kernel 的重新执行结果。

### 性能变化

本轮没有按正式协议重测，不能声称新 DSL 提升或保持了所有 kernel 的性能。Canonical compute
semantics 主要复用原 lowering，但 `pipeline` 目前也不构成新的机器流水实现。

---

## 本轮没有做的事情

- 没有增加 kernel-name、q-format、VLEN 或 target-specific source route；
- 没有建立 generic `qgemm`；
- 没有把 RVV/IME selector 暴露给作者；
- 没有把 LMUL、microtile、register、fragment 或 pipeline stage 加入 DSL；
- 没有新增 Physical IR、Selected IR 或第二种 realization language；
- 没有修改 baseline、性能 CSV 或历史 report；
- 没有调用、包装或链接 `source/`、GGML 或 `materials/`；
- 没有保留旧入口、compatibility layer、feature flag 或 fallback。

---

## 最终判断

本轮真正完成的不是一次 API 改名，而是把 Weft 的 source ownership 固定下来：

```text
作者写完整 core-local CPU operator program
        ↓
显式 local command 授权 SIMD / microkernel / extension realization
        ↓
canonical IR 保存 axis、value、memory、lifetime 与 command semantics
        ↓
RISC-V target 在授权边界内选择物理实现
```

因此后续性能工作应发生在 shared axis/value planning、reuse、resource、pipeline scheduler 与
RVV/IME local realization 中。若后续又需要 kernel 名、格式 route、whole-kernel template、旧
emitter 或算法猜测才能工作，那是实现偏离了本轮冻结的模型，而不是继续修改 DSL 的理由。
