# Weft 最终设计：面向非 SIMT 处理器的单控制器、多引擎算子 DSL

## 结论

Weft 冻结为一门 **single-controller, multi-engine operator DSL**：

> 一个 CPU worker/hart 持续执行完整、有序的算子程序；作者明确组织 traversal、blocking、
> staging、workspace、persistent layout、state 和局部计算关系；编译器把每个显式局部关系映射
> 到当前 core 的 scalar、scalable vector、register microtile、局部流水或 extension fragment。

Weft 不是图编译器，不从普通 SSA 图识别完整算子；不是 Triton-CPU，不以 program grid 与
program-owned tile 为根对象；也不是 TileLang-CPU，不把 GPU 风格 shared/fragment/thread
binding 直接暴露给 CPU 作者。

项目只保留：

```text
one Python DSL
→ one canonical Kernel IR
→ one target lowering
→ intrinsic C / local asm
→ system compiler
```

“高层局部命令”和“较细的组合构造”是同一语言的两个使用层次，不是两门 DSL。RVV/IME
realization 是编译器内部候选，不是第二种用户程序。

---

## 一、Weft 要解决的真实问题

### 1. 直接写 C/intrinsic 时，作者被迫同时决定三层内容

以 RISC-V GEMM 为例，手写高性能 C 通常混在一起：

1. 算法组织：M/N/K traversal、BM/BN/BK、packing、workspace、accumulator lifetime；
2. core 内机器组织：向量化 M 还是 N、LMUL、多个 accumulator、K-unroll、load reuse、流水；
3. ISA 拼写：RVV type/intrinsic、vsetvl、mask/tail、IME fragment 与 inline asm constraint。

第一层决定“这份 CPU 算法是什么”，第二层决定“同一算法如何占用 core 资源”，第三层只是
目标 API/ISA 的具体表达。直接写 C 会把三层永久绑定：换 VLEN、寄存器预算、IME realization
或 intrinsic API 时，作者往往复制整个 kernel。

### 2. Weft 省略什么

Weft 不省略算法。作者仍写第一层，并明确哪些局部关系允许编译器重组。Weft 省略的是第二、
三层的重复机器工作：

- 不写 RVV strip/vsetvl；
- 不选 LMUL 和具体 vector type；
- 不手工展开 register accumulator tuple；
- 不为 VLEN128/VLEN256 复制 microkernel；
- 不在算法源码里选择 RVV 或 IME；
- 不手写 decode/load/compute 的局部调度与 intrinsic 拼写；
- 不为每个 consumer 亲自安排 register handoff、reload 或 rematerialize。

代价是作者必须把 axis、pointer relation、lifetime 和局部数值关系写得足够清楚，让编译器能
合法地产生这些机器决定。

### 3. 为什么不能只做一个高性能函数库

预编译 BLAS/quant library 能覆盖固定 ABI 与固定算法，但无法自然嵌入：

- matmul 后紧接自定义 pointwise/state；
- accumulator 跨作者 K-loop 或控制流存活；
- decode、lookup、dot 与不规则 memory 的局部组合；
- 同一 structured result 的多个普通 consumer；
- 新 quant relation 或 extension primitive。

把整个算子包装成库函数会夺走外围 traversal、workspace 和 fusion；把每个 intrinsic 暴露给
作者又失去可移植编译。Weft 的边界因此必须是 **可组合的局部语义命令**。

---

## 二、执行模型

### 1. 一个 kernel 由谁执行

当前 kernel 由一个 worker/hart 从入口执行到返回。并行 runtime 可以把不同工作范围分给多个
worker，但 worker identity 和调度不在 kernel 中隐式出现。Kernel 内没有 grid、program id、
warp、thread block 或 SIMT lane identity。

### 2. 有序 control 是根

`for/while/if` 是当前 worker 的真实有序程序。作者可以顺序处理多个 block，让 accumulator、
state、workspace view 和 packed data 跨循环存在。普通 scalar loop 绝不自动变成 VLA；编译器
也不改变作者的 outer traversal。

### 3. 局部并行来自显式授权

作者通过 `W.vla`、`W.vdot`、`W.gemm`、reduce/scan、typed quant 或其他局部 command，明确
允许编译器在该语义边界内部使用 SIMD、register repetition 或 extension fragment。没有显式
授权，普通 multiply/add/reduce 不会被猜成 matmul 或 tensor primitive。

---

## 三、为什么是一门 DSL，而不是多门 DSL

### 方案 A：Kernel DSL + realization DSL

表面上很诱人：普通用户写 kernel，专家再写 RVV/IME realization。但它会立即产生：

- 两套类型和控制语义；
- kernel value 到 realization value 的新 ABI；
- realization 是否允许拥有 loop/storage/state 的边界争议；
- 每个新 command 都要在两门语言间同步；
- 用户 realization 很容易重新长成 whole-kernel template。

这正是 Weft 想删除的重复 authority。

### 方案 B：一个极低层 DSL

如果所有用户都直接写 `W.vla/W.axis/load/store`，语言虽统一，但 matmul、scan、quant relation
会散成普通 SSA 图，编译器只能重新识别模式。用户也会重复写已知局部高性能结构。

### 冻结方案：一门语言，渐进式抽象

同一 DSL 中同时存在：

1. 高层局部计算命令：`gemm/vdot/reduce/scan/summary/lookup/typed quant`；
2. 组合层构造：有序 control、`W.vla`、`W.axis`、load/store/transfer、pointwise、state；
3. 后端专属 intrinsic/asm：只在 compiler 内部，不是 DSL。

标准 command 不够时，作者下降到组合层写新的局部数据流；不需要切换语言，也不能下降到
RVV/IME intrinsic。这相当于“一个语言内有不同抽象高度”，而不是“多个 DSL 互相调用”。

---

## 四、用户真正操作的三类值

### Scalar value

用于 control、address、loop counter 和 scalar state。它可广播到 engine domain，但不会因普通
arithmetic 自动获得 SIMD 语义。

### Memory view

由 typed pointer、access、alignment、alias、storage class 与地址表达式组成。用户明确写
load/store/transfer，因此 data movement 和 effect 可见；用户不指定 cache level 或 vector load
instruction。

### Engine value

具有逻辑 axis 和 dtype 的普通 SSA value。它可以由 load、pointwise、reduce、gemm 或 quant
command 产生，可以多 use、跨 loop carry、进入 state/memory/另一个 command。

“Engine value”是编程模型概念，不是物理类型：它可能保持在 vector register、多个 scalar、
temporary stack slot 或 IME fragment，也可能因资源需要 reload。Source 不出现物理 placement。

---

## 五、用户可见的命令层级

### 1. Control 与 traversal

```python
W.range(begin, end, step)
W.blocks(begin, end, block)
W.pipeline(W.blocks(...))
```

`range` 是一般有序 loop；`blocks` 显式标记作者选择的 algorithmic/cache block traversal；
`pipeline` 只授权对这个已存在 loop 做依赖合法的局部流水。它不指定 stage 或 buffer count，
更不创建新的算法 pass。

### 2. Logical domain

```python
with W.vla(begin, end) as i:
    ...

mi = W.axis(BM)
ni = W.axis(BN)
```

`W.vla` 是运行时长度 SIMD logical domain；`W.axis` 是有身份的 block axis。二者都只表达逻辑
元素关系，不表达 lane/LMUL/fragment。

### 3. Storage 与 movement

```python
W.buffer(workspace, shape=(...))
value = W.load(pointer, where=...)
W.store(pointer, value, where=...)
W.transfer(source, destination)
```

`W.buffer` 声明 caller-owned workspace/persistent view，不分配内存。`transfer` 当前严格等价于
同域 load+store，不暗示 async、packing 或 cache placement。

### 4. Local collective

```python
W.vdot(...)
W.gemm(...)
W.reduce(...)
W.scan(...)
W.argmax(...)
W.online_softmax_summary(...)
```

用户知道命令的数学语义，不知道它最终使用何种 microkernel。

### 5. Typed packed compute

```python
W.quant.affine_i4_i8_dot(...)
W.quant.iq2_s_i8_dot(...)
W.quant.q6_k_i8_dot(...)
```

Namespace 表明这些是同一语言的一组 typed command，不是各自独立 DSL。格式真正不同的
packed/codebook/scale/correction 关系仍由不同 op 保存；机器 mapping、decode、widen、reuse、
resource 和 pipeline 在编译器中共享。

---

## 六、GEMM：作者与编译器的精确分工

### 作者写的程序

```python
for m0 in W.blocks(m_begin, m_end, BM):
    for n0 in W.blocks(0, n, BN):
        mi = W.axis(BM)
        ni = W.axis(BN)
        acc = W.accumulator((mi, ni), W.f32, init=0.0)

        for k0 in W.pipeline(W.blocks(0, k, BK)):
            ki = W.axis(BK)
            a_block = W.load(a + ...)
            b_block = W.load(b + ...)
            acc = W.gemm(a_block, b_block, init=acc, acc_dtype=W.f32)

        W.store(c + ..., epilogue(acc), where=...)
```

这段源码决定：

- 当前 worker 的 M/N/K block traversal；
- BM/BN/BK 与 loop order；
- accumulator 跨 K-loop 生命周期；
- A/B 地址、mask 与 visible staging；
- pipeline 授权边界；
- epilogue 与 store。

### 编译器生成的内容

对 `W.gemm` 的 M/N/K logical axes，编译器可以生成：

```text
M = sequential × register repetition
N = sequential × RVV lane
K = sequential × unroll
```

或者在合法 target 上：

```text
M = sequential × IME fragment-M
N = sequential × IME fragment-N
K = sequential × IME fragment-K
```

然后联合决定 A/B load window、broadcast、multiple accumulator、handoff、buffering 与 intrinsic/
asm。RVV 和 IME 不是两个 GEMM source API，也不是两条 whole-kernel route。

### `order` 与 `math`

它们不是 LMUL 或 target 配置，而是作者允许的数值等价空间。常见 `relaxed/native` 已是默认，
源码无需反复写；只有严格顺序或精确数学有要求时才显式覆盖。把这两项完全交给 backend 会让
性能优化静默改变算法语义，因此不能删除其语义地位。

---

## 七、为什么 CPU/RISC-V 特别需要 vdot 与 quant 作为重点

LLM CPU kernel 的性能热点并不只有 dense tile GEMM。大量真实路径由 packed weights、block
scale/minimum、codebook lookup、widening dot、GEMV/vec-dot、irregular gather 和 ordered state
组成。若语言只有 generic tile/matmul：

- packed format 的真实数学关系会泄漏成指针魔法；
- 后端按 q-format 名称选择 whole-kernel 实现；
- decode 与 dot 的复用和流水无法跨格式共享；
- IME/RVV 的局部实现边界不清楚。

因此 Weft 把 `vdot` 和 typed quant relation 视为一等局部 command，同时坚持它们只拥有局部
数值关系。它们不能接管 projection/MoE 的 outer traversal、persistent repack 或 ABI。

目前不引入一个泛化 `qgemm`，因为现有格式尚没有一个不丢失 scale/minimum/codebook/correction
语义的统一 schema。为了 API 看起来简洁而强行统一，会把差异重新藏进格式分支；这比多个
typed command 更坏。

---

## 八、编译器内部的正式抽象

### 1. 输入事实

来自 Kernel IR：axis identity/extent、loop nesting、use-def、pointer relation、predicate/effect、
storage/lifetime、dtype、validity 和 explicit command semantics。

来自 target profile：ISA/ABI、VLEN、SEW/LMUL、register budget、segment/gather、标准扩展、IME
fragment 与 intrinsic/asm availability。

VLEN、target 型号和格式名都只是事实，不能成为完整实现身份。

### 2. 唯一合法推导

- free/reduction/broadcast axis；
- unit/strided/indexed/segment memory relation；
- validity、tail、mask；
- cast/widen/narrow element mapping；
- live value 与 resource legality；
- 某个 leaf 是否满足 dtype/shape/ISA 限制。

这些结论只有一个正确答案，应由共享逻辑产生一次。

### 3. 结构性选择

- time/lane/register/unroll/fragment mapping；
- scalar/vector state；
- RVV microkernel 或 IME fragment；
- shared/reload/rematerialize/local-pack handoff；
- memory schedule；
- decode materialize 或 decode-compute fusion；
- sequential/single-buffer/double-buffer local pipeline。

### 4. 参数性选择

- LMUL；
- MR/NR 与 register factor；
- accumulator 数；
- K-unroll；
- buffer count、prefetch distance；
- source 已标为 constexpr 的 block 参数实例。

参数只能实例化结构，不能创造算法。

### 5. 发射

Emitter 只消费已选 mapping、loop、memory schedule、buffering 与 local leaf，机械拼写普通 C、
RVV intrinsic 和 typed IME asm。它不能再看 kernel 名、格式、VLEN、shape 或外围 op closure 重新
选择。

---

## 九、统一扩展机制

### 已有语义的新硬件实现

若新 RISC-V tensor/vector extension 能实现现有 `W.gemm`、`W.vdot`、reduce 或 quant command，
只需增加：

- 接受的 logical mapping/dtype；
- fragment/mask/tail/memory constraints；
- input/output physical shape；
- resource accounting；
- intrinsic/asm spelling。

所有具有同一局部语义的 kernel 自动获得新候选，不新增 source engine name。

### 真正的新可观察语义

只有当前 DSL 无法无歧义表达、且局部数值关系真实独立时才增加 command。新 command 必须有
canonical op、verifier 和至少一个真实 target artifact，且不能拥有 outer control/storage/ABI。

### 非 RISC-V target

模型原则上适用于其他 single-controller、non-SIMT target，但当前正式产品范围仍是 RISC-V。
NPU 只有在它能够遵守同一 worker program、memory/storage contract 和局部 command boundary，
且实际 target lowering 完成后才算支持；不能用“未来可能支持”反证当前抽象正确。

---

## 十、与 Triton 和 TileLang 的精确关系

| 问题 | Triton | TileLang 常见开发者层 | Weft |
|---|---|---|---|
| 根执行对象 | program instance / grid tile | GPU tile + thread/storage scheduling | 一个持续执行的 CPU worker |
| 外层 traversal | 常由 program grid 表示 | kernel/thread binding 与 tile loop | 作者的普通有序 control |
| 主要显式存储 | global/shared/register-like tensor | shared/fragment/local allocation | external/workspace/persistent lifetime |
| 并行授权 | tile/program semantics | tile op + thread binding | `W.vla` 与局部 command |
| 微内核 | GPU compiler/tensorization | library/thread primitive | compiler-generated RVV/IME realization |
| pipeline | GPU stage/copy 管理 | 作者常显式 stage/order | 作者只授权 loop，target 选局部结构 |
| 量化重点 | 通常作为 tile computation | 可用 custom op/library | typed packed relation + shared mapping |

Weft 会借鉴二者的原则：明确的局部语义、逻辑数据域与物理布局分离、可组合 command、target
tensorization。但不复用其根编程模型，因为 GPU 的 program grid、协作线程和显式 shared/
fragment ownership 并不是 CPU worker 算法的自然所有权。

---

## 十一、反方审查

### 反驳 1：这只是“for + intrinsic library”

若 `W.gemm/W.quant.*` 只是 opaque 函数，反驳成立。Weft 要求 command 结果进入普通 SSA，axis、
memory relation 和 lifetime 穿过 command，target 联合规划 producer/consumer，并为同一 command
生成多个机器 realization。因此 command 不是调用预编译库，而是 canonical compiler semantic。

### 反驳 2：为什么不让编译器从 C 自动识别 matmul

因为识别 ordinary multiply/add 无法确定作者是否授权重结合、blocking、packing、state 与
extension semantics，且 source spelling 变化会导致 closure matcher。显式 command 是必要的
语义授权，不是多余 annotation。

### 反驳 3：为什么还让用户写 BM/BN/BK

BM/BN/BK 常影响 cache traffic、workspace、persistent reuse 与 worker traversal，是算法实现的
可观察组织；LMUL/MR/NR/K-unroll 是 command 内机器组织。将前者也隐藏给 compiler 会要求一个
完整 cache/parallel schedule language 或自动算法搜索，改变项目责任边界。`constexpr` 允许构建
期比较不同作者实例，但 target 不暗中发明 block loop。

### 反驳 4：为什么 `W.vla` 与 `W.gemm` 看起来仍是两种模型

它们是不同 semantic command，不是不同执行根。二者都在同一 worker control 中产生普通 SSA
value、使用同一 axis/memory/lifetime facts，并进入同一 target mapping/resource/emission。
统一它们的语义会丢信息；统一它们的编译基础才是正确目标。

### 反驳 5：一个 DSL 会不会过粗

“粗”若指用户看不到 LMUL/fragment，这是有意边界；若指 command 结果无法组合、axis/memory
facts 无法传递，则是编译器缺陷。解决后者应加强 canonical facts 与 target planning，而不是把
物理 layout 泄漏回 source。

### 反驳 6：自动 pipeline 是否偷偷改算法

只有作者显式 `W.pipeline(existing_loop)` 才授权，scheduler 受 effect/alias/state dependence 与
resource 约束，只移动当前 loop 内不可观察 producer，不能创建 workspace、跨 loop stage 或新
算法 pass。顺序 realization 永远合法。这个边界比直接暴露 stage/寄存器组更稳定。

---

## 十二、收益、代价与失败条件

### 收益

- CPU 算法组织在 source 中可审查，不靠 backend 猜；
- RVV/IME/VLEN 差异集中在 target realization；
- dense、quant、state 和 irregular memory 共享 axis/value/resource/scheduler；
- structured result 可自然 fusion、多 use 与跨 control carry；
- 新 extension 扩张局部候选，而不是增加完整算子路径；
- intrinsic API 变化不穿透 DSL 与 compiler decisions。

### 代价

- 作者需要理解 CPU blocking、storage lifetime 与 command boundary；
- 编译器必须做真正的 value-chain planning 和资源合法性，难度高于模板选择；
- typed quant command 数量不会被虚假压到一个；
- 单 worker 模型不直接解决跨 core parallel scheduling。

### 设计失败的可观察条件

出现以下任一情况，说明实现偏离而不是模型需要随意改写：

- 新 kernel 必须靠 kernel/format/target 名 route；
- command 增加一个普通 consumer 就失去 lowering；
- RVV 与 IME 需要两份完整 DSL kernel；
- emitter 再次推断 LMUL/microtile/pipeline/fragment；
- backend 为性能创建作者未写的 blocking/staging/persistent packing；
- 用户必须写寄存器、LMUL、fragment 或 intrinsic；
- `W.quant.*` command 拥有 projection/MoE 的外围 traversal；
- source/examples/materials 被链接为 production fallback。

只有真实 workload 证明“作者必须表达的一项可观察算法结构无处表达”，才修改 DSL；只有真实
target 证明“现有 command 无法描述一项独立局部语义”，才增加 command。性能实现不足应修
physical mapping、reuse、resource、scheduler 或 leaf，不能反复改编程模型。

---

## 十三、本次第一步落地

本次把设计落实为唯一 source surface：

- `W.block → W.axis`：强调逻辑轴而非物理 block layout；
- `W.storage → W.buffer`：强调 caller-owned workspace/persistent view；
- `W.dot → W.vdot`、`W.matmul → W.gemm`：建立用户可读的局部命令层；
- 顶层量化命令 → `W.quant.*`：同一 DSL namespace，不是多个格式 DSL；
- 新增 `W.blocks`、`W.pipeline`、`W.accumulator` 与 `W.transfer`；
- `for` canonical IR 保存 ordered/blocks traversal 与 pipeline authorization；
- 全部现有 DSL examples 使用新 surface；
- canonical `block_index/storage/dot/matmul` 继续保存稳定内部语义；
- 旧 Python 入口和兼容 alias 不保留。

这一步冻结的是用户模型和 canonical semantics。后续性能工作只能深化统一 axis/value/memory/
lifetime planning、局部 realization 与 emission，不能重新把语言改成 target command list、图模式
库或多 DSL 拼接。
