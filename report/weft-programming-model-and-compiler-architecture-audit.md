# Weft 编程模型与编译器架构收敛审计

## 这份报告怎样得出结论

这不是对 `doc/` 的复述，也不把此前“六十个 kernel 能跑”当作编程模型已经成立的证明。
本轮按下面的顺序重新建立事实：

1. 先读作者实际可调用的 Python surface 与 AST frontend；
2. 再读 canonical Kernel IR 的 type、op 与 verifier；
3. 再读 RISC-V physical decision、resource model 与 intrinsic-C emitter；
4. 再检查真实 kernel source、runtime、runner 与性能 CSV；
5. 用临时程序验证自然组合、公开 API 与后端产物之间是否真正闭合；
6. 最后才用 `doc/` 对照哪些是当前事实、哪些只是目标设计。

审计中的“支持”统一指：

```text
自然 Weft source
→ canonical Kernel IR
→ 当前 RISC-V target lowering
→ intrinsic C / local asm
→ system compiler
→ 真实执行
```

只有 Python 能构造、只有 IR 能 parse、只有某个精确 source closure 能发射，均不单独算支持。

本轮实际做了四项探针：

- 仓库现有 `add_bias` 重新走完整 SG2044 主链并真实执行，结果为 `PASS weft add_bias`；
- 现有 `out_product_f32` 可以从 DSL 生成 intrinsic C；
- 只在同一个 `dot` 结果与最终 store 之间加入自然的逐元素 bias，canonical IR 仍可生成，但
  RISC-V lowering 失败为 `load pointer is unavailable`；
- `W.atomic_add` / `W.fence` 可以进入 canonical IR，但 intrinsic-C lowering 在
  `weft_kernel.atomic_add` 明确失败；`W.widen` 会形成一个未注册 verifier 的
  `weft_kernel.widen` generic operation，随后在 physical planning 中间接失败。

另一个临时探针确认：helper 的未知 keyword 会被 frontend 拒绝，不存在此前静态阅读一度怀疑的
“额外 keyword 被静默吞掉”。但 helper 参数 annotation 的确完全不参与类型检查：标成 `W.i32`
的 helper 参数接收 F32 VLA value 后仍生成 F32 RVV intrinsic C。

因此，本报告既不把文档愿景当事实，也不把所有窄实现都误判成架构错误。


## 一、最终判断：已有真实编译主干，但编程模型尚未冻结

### 1.1 当前最准确的定位

当前 Weft 是一个**有真实 RISC-V lowering 的 worker-local AOT kernel compiler prototype**。

它已经不是图编译器，原因很明确：

- source 是一个完整 callable，而不是 graph node 集合；
- 没有 grid、`program_id`、hart ID 或隐式 task identity；
- scalar loop、VLA、pointer effect、state 与 structured primitive 都在同一 Kernel IR 中；
- unsupported 会失败，没有 legacy、GGML 或材料代码 fallback；
- 已有 Python → Kernel IR → intrinsic C → object → 真实 RVV/IME 执行主链。

但它也还不是一个 Triton 级别的 RISC-V kernel compiler。这里的“Triton 级别”不要求复制
GPU grid/SIMT，而要求至少满足：

1. 作者面对一个稳定、单一、可预测的程序模型；
2. 公开语言构造在目标后端上的能力边界真实；
3. primitive 可以通过普通 SSA/use-def 自然组合；
4. physical realization 不依赖精确 source closure；
5. selected physical decision 的 producer 唯一，emitter 不重新决策；
6. 同一算法存在受资源约束且有真实性能差异的候选空间；
7. compiler front door 与 artifact contract 不依赖固定 benchmark catalog。

当前 Weft 在第 1、2、3、4、5、6、7 项上都仍有结构性缺口。

### 1.2 当前实际混合了三种抽象高度

代码中能够看到三个同时存在、但尚未完全统一的作者模型：

```text
A. C-like worker function
   raw pointer/scalar ABI + ordered scalar control + caller-provided workspace

B. lexical VLA program
   one explicit runtime logical axis + masked region SSA + compiler-selected RVV strip

C. backend-shaped local block/extension program
   explicit block axes + fixed dot/matmul envelopes + packed quant/IME leaf
```

这三者放进了同一个 Kernel IR，并不自动意味着它们已经形成同一种模型。真正的统一要求是：

- 相同 value 在三者之间有一致的 extent、validity、storage 与 use-def 语义；
- 一个 structured primitive 的结果可以进入普通 pointwise/state/memory consumer；
- workspace、persistent layout 与 primitive-local temporary 的所有权明确；
- target 从逐实体事实选择 realization，而不是要求作者先写成 emitter 喜欢的 closure。

当前这些条件没有全部成立。最直接的反例就是合法 `dot → pointwise add → store` 不能 lowering，
而 `dot → store` 可以。

### 1.3 “六十个 kernel 能跑”证明了什么

它证明：

- 主链不是空壳；
- 已有一批 VLA、state、block、quant 与 extension 实现；
- 同一份 DSL 可在 SG2044/VLEN128、K1/VLEN256 和两个 IME leaf 上产生真实产物；
- corpus 中没有靠 kernel 名选择完整实现的显式 production fallback。

它没有证明：

- 任意自然使用这些 primitive 的程序都可组合；
- public DSL surface 与后端能力闭合；
- corpus 没有按现有 fast path 预塑形；
- physical candidate space 已经成熟；
- 生成实现普遍具有 GGML 或手写内核竞争力；
- Weft 已经具有类似 Triton 的外部用户编译体验。


## 二、作者实际写下的是什么

### 2.1 Kernel 是受限 Python source，而不是 Python execution

`@weft.kernel` 捕获一个可由 `inspect` 取得源码的同步函数；kernel 本身不能作为 Python callable
运行。Frontend 重新解析 AST，直接打印 generic MLIR assembly：

- `python/weft/api/definitions.py:28-62`；
- `python/weft/frontend/source.py:27-40`；
- `python/weft/frontend/ir.py:48-50`。

Entry ABI 当前只接受 annotated positional pointer/scalar/constexpr 参数，禁止默认参数、
keyword-only、varargs 与 kwargs；return 为 `None` 或一个 terminal scalar：

- `python/weft/frontend/compiler.py:182-203`；
- `python/weft/frontend/compiler.py:263-349`。

这是一个可以成立的 AOT C ABI 选择。Entry 不接受 block/region 参数本身不是 bug，因为当前根模型
本来就是从 pointer/scalar ABI 在 kernel 内构造逻辑域。问题在于这个 ABI 尚未补全 workspace、
descriptor 与 artifact declaration 的统一合同。

### 2.2 Scalar control 的语义基本清楚

`W.range` 是有序 scalar loop，Python `if`/`while` 只接受 scalar `i1`。Frontend 根据 AST assignment
自动形成 loop-carried 和 branch result SSA：

- `for`：`compiler.py:2073-2114`；
- `if`：`compiler.py:2175-2217`；
- `while`：`compiler.py:2219-2261`。

这里的自动 carry 是 Python source 到 region SSA 的正常投影，不应为了“显式 IR”强迫作者手写
tuple state，也不需要增加 normalization pass。真正必须保持的是：ordinary scalar loop 不能被
target 猜成 VLA、reduce 或 dot。当前实现没有发现这种静默替换。

### 2.3 VLA 是当前最完整的核心构造

`with W.vla(begin, end) as i` 明确授权一个一维 runtime logical axis。它禁止嵌套，body 不能修改
outer state，region value 不能逃出 lexical scope：

- `compiler.py:2116-2173`；
- `KernelDialect.cpp:677-750`。

这个构造已经具有一条清楚的作者/target 边界：

```text
作者：逻辑区间、pointer/index、predicate、state primitive
target：strip 划分、vl、LMUL、mask/memory instruction
```

对 pointwise、unit/strided memory、若干 indexed memory、reduce/scan/summary，实际 backend 确实
按该模型工作。Weft 最值得保留的语言核心就是这里。

但当前 VLA 仍只有一个 active logical axis；多个 block axes 只是 value shape，不是第二个 VLA。
这是清楚的能力边界，不应为了显得一般而恢复任意多维 contraction 或隐式 SIMT。

### 2.4 Logical block 目前既是语义值，也常被用来对齐 fast path

`W.block_axis`、broadcast/slicing、load、reshape、transpose 构造 local shaped value；`W.dot`、
`W.matmul`、block reduce、lookup/decode 与 extension op 消费它们。

语言刻意把 contraction 收缩成当前真实需要的：

```text
dot:    [R,K] × [K]       → [R]
        [VLA,K] × [K]     → [VLA]
        [R,K] × [VLA,K]   → [VLA,R]

matmul: [M,K] × [K,N]     → [M,N]
```

不恢复任意轴 contraction 是正确取舍。问题不是 primitive 不够“一般”，而是这些已经公开的合法
形态在 backend 中仍要求非常精确的 producer、consumer、axis 与 store closure。

### 2.5 State 不是一种统一实现，但应是一套统一边界

当前 source 区分：

- ordered sequential carry；
- `reduce`；
- `scan`；
- `argmax`；
- `online_softmax_summary`。

保持这些可观察语义不同是正确的。无需引入任意 `lift/merge/finalize` generic fold，也不应从
普通 SSA 猜 summary。

当前不清楚的不是语义分类，而是 physical state placement 的覆盖宽度：部分 state 有 VLA owner
和明确 physical shape，部分 complex recurrence 仍主要由 source scalar loop + caller scratch
完成；同一套 state facts 尚未统一支撑跨 strip、block state 与 structured primitive handoff。

### 2.6 量化和扩展 primitive 可以专门，但不能成为格式 route

`weft_ext` 的局部 quant/IME primitive 具有真实理由：packed bit ordering、scale/zero-point、
saturation、codebook 与矩阵 fragment 都是普通 float dot 无法完整表达的可观察语义。

因此，名字中出现 `iq2`、`q6_k` 或 `i4_i8` 不自动等于错误。判据应是：

- op 是否完整定义一个局部 numerical relation；
- 是否只消费 explicit typed operands；
- 是否不拥有 outer traversal、public ABI、persistent layout 与 whole-kernel state；
- 多个 kernel 是否能复用同一 local realization。

当前 extension ops 基本满足“局部 leaf”边界，但 surface 与 verifier 仍高度固定为某一 packed tuple，
candidate/resource model 也大量硬编码。它们目前是**合法但很窄的 semantic leaf**，不是一个已经
收敛的通用量化 value/layout 模型。


## 三、当前编程模型最缺的一层：storage、workspace 与 layout ownership

### 3.1 Source 没有 local storage/lifetime 构造

公开 DSL 中没有 local buffer、scratch allocation、workspace declaration 或 storage lifetime
primitive。复杂 kernel 只能把临时空间作为普通 pointer ABI 参数传入。

真实 source 已经依赖这种做法：

- flash attention 的 `query_scratch` 与 `accumulator_scratch`：
  `examples/kernels/attention/online_flash_attention.py:6-13`；
- CSR sparse attention 的 `accumulator_scratch`：
  `examples/kernels/attention/csr_sparse_attention.py:6-18`；
- dense convolution 的 `packed_patches` 与 `packed_weight`：
  `examples/kernels/vision/dense_conv2d.py:5-12`；
- IME projection 的 `activation_scale` 与 `activation_code`：
  `examples/kernels/ime/q4_k_projection.py:8-18`。

这不是简单的“应该让 compiler 自动 staging”。设计边界已经正确规定：algorithmic staging、
persistent packing 和跨 primitive lifetime 属于作者。真正未冻结的是作者用什么语言对象表达它们。

当前只有 `ptr + scalar extent + 约定`，因此 compiler 看不到：

- workspace 是 per-call、per-worker、per-row 还是 persistent；
- shape、alignment、alias 与 lifetime；
- 哪段是算法可观察 staging，哪段只是 primitive-local temporary；
- runtime 需要为 entry 分配多少空间；
- artifact header 应如何向调用者描述 workspace。

如果最终决定“所有显式 scratch 都由调用者作为 raw pointer 提供”，这也可以成为模型，但必须
明确成为 ABI 合同，并由 artifact 暴露 size/alignment/lifetime。当前既没有这种 artifact，也没有
语言级 descriptor，所以这个问题仍是空的。

### 3.2 Persistent packing 的边界在理念上正确，在产品链上未闭合

IME Q4_K runtime 先把 canonical Q4_K 权重手工 repack 成 N16×K32 的 304-byte layout，随后 Weft
kernel 直接消费 packed pointer：

- `examples/repro/weft/ime/q4_k_projection_runtime.cpp:115-172`；
- `examples/repro/weft/ime/q4_k_projection_runtime.cpp:263-275`。

Persistent model layout 本应由模型加载/上游 runtime 拥有，不应由 target 偷偷创造；因此这种
所有权本身合理。但当前 repro 只证明“已有人准备好私有 packed ABI 时 leaf 可运行”，还没有定义：

- canonical weight 到 packed artifact 的正式 builder/contract；
- packed layout 的 identity、size、alignment 与 target compatibility；
- llama.cpp/IntentDSL 如何查询并提供它；
- 同一 semantic primitive 的 RVV 与 IME realization 是否接受同一 logical operand contract。

这也是为什么 IME 两条记录不能外推成通用矩阵扩展支持。

### 3.3 Triton 级别不要求 target 自动发明算法，但要求 storage boundary 可组合

Weft 不需要复制 Triton shared memory 或 GPU program grid。然而，如果作者写下 algorithmic
staging，语言必须有一种比“再加两个无说明 pointer 参数”更稳定的方式保存 storage 事实；如果
temporary 只在 dot/IME primitive 内部存在，则它应只属于 physical plan，不进入 public ABI。

这是当前编程模型中最大的未决项。


## 四、Python frontend：能表达很多，但 public contract 还不可信

### 4.1 Public surface 与唯一后端没有闭合

`python/weft/language/builtins.py:82-141` 暴露了完整 intrinsic surface。实际核对得到：

| 构造 | Frontend/IR 状态 | 当前 RISC-V 状态 | 判断 |
| --- | --- | --- | --- |
| `widen` | frontend 产生 `weft_kernel.widen`，ODS 中无注册 op/verifier | physical planning 间接失败 | 假 canonical 能力 |
| `atomic_add` | frontend + registered IR | dispatcher 无 emitter | 假 target 能力 |
| `fence` | frontend + registered IR | dispatcher 无 emitter | 假 target 能力 |
| `valid` / `fill` | frontend + registered IR | 无一般 emitter，corpus 无消费者 | public 能力未闭合 |
| `permute` | frontend + registered IR | 无一般 emitter | public 能力未闭合 |
| `expand_dims` / `broadcast_to` / `reshape` / `transpose` | frontend + registered IR | 只可能在特定 block closure 中被吸收，无独立一般 realization | surface 比能力宽 |
| `lookup` | 看似任意 block table/index domain | 实际仅 VLA F32 table-16 gather envelope | surface 严重过宽 |
| `matmul` | local rank-2 float block | 当前只有 F16×F16→F32 特定 GEMM envelope | surface 仍过宽 |
| `dot` | 三种明确合法 shape | 必须被预先识别的 store/region decision 吸收 | composition 不闭合 |

`RISCVLowering.cpp:3260-3378` 是当前唯一 operation dispatcher。没有 owner/emitter 的 op 最终
明确报 `RISC-V target lowering does not implement ...`。这比 silent fallback 好，但不能把
“明确失败”写成 public support。

这里应采用很简单的标准：

> public construct 要么能在当前正式 target 上走到真实 artifact，要么明确从 public surface
> 删除；不能靠“未来可能实现”留在语言里。

### 4.2 `W.widen` 暴露了 canonical IR authority 的漏洞

Frontend 在 `compiler.py:1872-1876` 直接打印 `weft_kernel.widen`，但 `KernelOps.td` 没有对应 op。
临时 repro 中 `--emit=kernel-ir` 仍接受了这个 generic operation，意味着它没有正式 op class 和
verifier，却可以混入所谓 canonical Kernel IR。之后 target 没有在 primitive 边界报 unsupported，
而是以 `VLA has no legal LMUL candidate for its entity resources` 间接失败。

这同时是三个问题：

1. public frontend surface 与 registered canonical schema 不一致；
2. canonical parser/verify 没有拒绝同 namespace 的未知 primitive；
3. physical planner 的错误边界晚于语义边界，诊断误导。

### 4.3 Helper 当前只是 AST inline，不是一等 typed function

实际 helper 路径是 `_inline_helper`：`compiler.py:1601-1632`。它：

- 检查参数数量和 keyword 绑定；
- 不读取 helper 参数/return annotation；
- 把 body 直接编译进 caller；
- 只用 decorator effects 约束 effectful helper。

`_compile_helper_region` 与 `_resolve_helper_argument` 存在，但全仓没有调用点：
`compiler.py:1568-1599`。因此 `doc/compiler/kernel-ir.md:118-120` 和
`doc/dsl/python.md:151-153` 所说的“summary helper 编译成 canonical region”不是当前能力。

Helper 不一定需要 annotation；但语言必须二选一：拒绝 annotation，或使 annotation 成为真实
type contract。现在“语法接受、语义忽略”会让作者误判。

### 4.4 不应误修的部分

下列现状不构成本轮架构问题：

- context-only 的 `W.range` / `W.vla` 只能出现在 `for` / `with`，这是 AST DSL 语义；
- scalar control 自动投影 carried SSA，不需要作者手写 IR tuple；
- entry 只接 pointer/scalar/constexpr 是当前 C ABI 选择，不需要为了“更 tensor”强行开放
  region 参数；
- `AnyType` 本身不是 bug，只要每个 op verifier 完整；
- 未知 helper keyword 当前会被拒绝，不存在 silent typo fallback。


## 五、Canonical Kernel IR：骨架正确，局部 contract 仍有孔

### 5.1 值得保留的部分

Kernel IR 已经形成一个真实的 canonical value/control system：

- `ptr / constexpr / block / region / masked / tuple` 持久类型；
- scalar、block 与 region 使用同一 pointwise/value query；
- dynamic extent identity 从 SSA producer、region argument 与 transforms 推导；
- structured `for/while/if/vla` 有 region/terminator/carried type verifier；
- VLA nesting 与 value escape 有明确约束；
- load/store/prefetch/atomic/fence/sort 有 memory effects；
- extension dialect 复用 core block/region/type，而不是第二份 execution IR。

这部分足以作为继续重构的承重层，不需要重新发明另一份 Python IR、schedule IR 或 capability IR。

### 5.2 `AnyType` 不是问题，缺 verifier 才是问题

当前真正的 verifier 缺口包括：

- `FillOp` 只检查 underlying/result element type，不能区分合法 scalar broadcast 与不兼容 shaped
  fill footprint：`KernelDialect.cpp:1069-1074`；
- `AtomicAddOp` 不验证 result footprint 与 pointer/value footprint 一致：
  `KernelDialect.cpp:1039-1051`；
- `LookupOp` / `DecodeOp` 没有闭合 table rank/extent、index/code domain、table element 与 result
  numerical relation：`KernelDialect.cpp:1254-1278`；
- dot/matmul verifier 先 `unwrapMasked`，但没有完整定义 masked operand validity 如何进入 unmasked
  accumulator/result：`KernelDialect.cpp:1171-1203`；
- dynamic output-shaped init 只比较 `-1` shape，没有逐轴验证 extent identity：
  `KernelDialect.cpp:1193-1201`；
- broadcast/reshape 对 shape attr、extent operand 与 source logical domain 的一致性验证仍弱：
  `KernelDialect.cpp:798-817`；
- `InvalidOp` 没有 verifier，却能作为 `none` sentinel 出现在任意结构中。

这里不需要为每个 op 再建一层 contract IR。正确修复位置就是 op verifier 与共用 logical
value/extent/validity query。

### 5.3 Dynamic extent 的“双轨”不是原罪

类型中用 `-1` 表示 dynamic dimension、op operand 保存实际 extent 是合理的 canonical 设计；它允许
类型保持有限，同时由 SSA 定义 identity。问题只在某些 op 没有调用同一套 identity verifier，
不能把整个设计误判成“必须把 runtime extent 塞进 type”。

### 5.4 Stable ID 当前不是必需层

Physical plan 只在一次 lowering 中使用 `mlir::Value/Operation*`，没有持久 facts 或外部 plan 需要
跨序列化引用 block arguments。因此当前没有 stable block/state ID 不是独立缺口。只有未来真的
引入外部可序列化分析消费方时，才有证据要求它；现在增加 ID 只会制造第二份 authority。

### 5.5 Extension schema 与文档已经漂移

当前 Python/Extension dialect 有 9 个量化 extension intrinsic，而
`doc/compiler/kernel-ir.md:109-110` 只列出 5 个。更重要的是，文档列举不能替代每个 extension
op 的 observable numerical contract、masked policy 与 local storage boundary。

这说明 `doc/` 目前只能作为设计意图，不能作为当前能力表。


## 六、RISC-V Realizer：已有 physical core，但仍是 hybrid architecture

### 6.1 已经成立的部分

当前 lowering 不是纯字符串 emitter。代码中确有：

- `PhysicalEntityPlan`：value shape、handoff、temporary、resource budget、schedule；
- VLA access/predicate/state/narrow/dot decisions；
- F32 dot 的 LMUL/K-unroll candidate 枚举；
- target VLEN/register count 参与 resource legality；
- `preparePhysicalDecisions()` 先于 emission；
- emitter 通过 plan maps 查 selected decision；
- 缺少 decision 或 unsupported primitive 时失败，不走旧路径。

因此，不应把现有后端整体称为“if-else 按 kernel 发射器”。它已经具有真实 compiler core。

### 6.2 但 common entity plan 之上仍有多套 family-specific closure recognizer

`preparePhysicalDecisions()` 对不同 op family 分别 walk，并建立不同 map。最明显的窄入口包括：

- local dot 必须 `hasOneUse()`，唯一 consumer 必须是直接 `StoreOp`：
  `RISCVLowering.cpp:1165-1195`；
- VLA dot 同样要求 `store.value == dot.result` 和 one-use，并固定 rank、dtype、order、math、axis 与
  pointer dependency：`RISCVLowering.cpp:1747-2031`；
- VLA physical operation 递归只穿过 nested `ForOp`，不遍历一般 structured region：
  `RISCVLowering.cpp:2057-2066`；
- widening F16 reduction 要求精确 `load → cast → multiply → reduce` 且每级 one-use：
  `RISCVLowering.cpp:2497-2523`；
- F16 FMA 同样依赖固定 producer closure：`RISCVLowering.cpp:2983-3014`；
- block store/reduce 按 producer closure、同 region、单 local axis 与 `getNextNode()` adjacency
  分组：`RISCVLowering.cpp:7887-7934, 8833-8984`。

这些不是 kernel-name route，但也还不是“每个实体从自身 typed facts 独立产生决定”。它们是
**primitive anchor + 精确局部源码闭包**。这解释了为什么 corpus 可以很多，而自然组合仍失败。

### 6.3 `dot → pointwise → store` 是当前最有代表性的反例

探针直接基于已能编译的 `examples/kernels/contraction/out_product.py`，只增加：

```python
value = W.dot(...)
value = value + W.load(bias + column[:, None])
W.store(output_ptr, value, where=...)
```

结果：

```text
Python DSL → canonical Kernel IR       成功
canonical Kernel IR → intrinsic C      失败
diagnostic                              load pointer is unavailable
```

算法没有改变 dot 的授权域、outer traversal、blocking 或 memory organization。唯一变化是 dot
结果经过一个普通 pointwise consumer。一个成熟的 local primitive lowering至少应产生：

```text
dot result physical shape
→ explicit handoff
→ pointwise add realization
→ store realization
```

当前 planner 没有为这个合法 use chain 建立 owner/handoff，于是连上游 load 都失去物理归属。
这不是“新 kernel 不支持”，而是编程模型的组合律没有进入 realizer。

### 6.4 Candidate space 仍大量是固定选择

未配置显式 backend override 时，F32 dot 默认枚举
`LMUL {1,2,4,8} × unroll {1,2,4}`，这是实质进展。但选择仍是固定 penalty 排序，不是
compile-and-measure tuner：`RISCVLowering.cpp:1664-1745`。

其他 family 更窄：

- VLA 取固定 preference 下第一个合法 LMUL；
- block store/reduce 只有少量固定 strip candidate；
- F16 matmul 只有 LMUL 与 row divisor 的窄组合；
- quant/fragment 多数直接选择一个 realization，再写死 resource group 常量；
- `requireEntityPlan()` 当前明确要求 `pipelineGroups == 0`、`pipelineStages == 1`、
  `prefetchDistance <= 1`：`RISCVLowering.cpp:1603-1616`。

因此，文档中“software pipeline、prefetch、multiple accumulators、local reuse 已进入统一 physical
space”的表述目前过强。数据结构里有字段不等于候选真实存在。

### 6.5 Emitter 仍保留部分物理 authority

Emitter 可以复杂，因为 intrinsic spelling、ABI 与 typed inline asm 本来属于它。但当前仍有超出
spelling 的硬编码：

- F16 matmul emitter 重新展开 row/column/K 结构并假设 row-major accumulator linearization：
  `RISCVLowering.cpp:7571-7734`；
- block index/compare/widen/load/store 固定若干 `u16m2`、`i32m4`、`vluxei16`、`f32m1/m4`
  shape：`RISCVLowering.cpp:7967-8614`；
- block decode 固定 table extent 16、u8 code、i8 table/result 与固定 gather：
  `RISCVLowering.cpp:8531-8585`；
- quant finalize 使用固定 primitive/peak groups 与 unroll 常量：
  `RISCVLowering.cpp:6463-6623`。

这些内容应先成为 typed transient decision，再由 emitter机械拼写。否则 planner 声称的 resource
legality 与 emitter 实际使用的寄存器/layout 不是同一份事实。

### 6.6 Target profile 还不是可靠的 RISC-V target contract

当前 parser 只做：

- `march` 是否以 `rv32/rv64` 开头；
- ABI 是否以 `ilp32/lp64` 开头；
- `march` 剩余字符串是否包含字符 `v`；
- VLEN 是否为 8 的倍数。

见 `lib/Target/RISCVTargetProfile.cpp:3-42`。Endianness 固定 little、vector registers 固定 32，
matrix extension 是 CLI 直接赋字符串：`include/Weft/Target/RISCVTargetProfile.h:11-21`。

这不足以支撑 element width、LMUL、Zve/Zvfh、ABI float convention、vendor extension、fragment 与
instruction legality。当前许多 target capability 实际散落在 realization matcher 中。


## 七、Corpus 审计：覆盖面真实，但 selection bias 仍然很强

### 7.1 当前语料的实际构成

`examples/kernels` 当前有 65 个 `.py` 文件、17 个 family；其中一个是 quant helper source。
`examples/repro/weft` 有 61 个 C++ runtime、6 个 quant C runtime 与公共 header。Runner 映射约 70 个
kernel 名称到 62 个 source basename，部分名称共享 source。

有两个真实 DSL source 没有进入 runner/runtime：

- `examples/kernels/reduction/online_softmax_summary.py`；
- `examples/kernels/reduction/predicate_reduce.py`。

它们只能算 source 示例，不能算已运行能力。

### 7.2 简单 kernel 自然，复杂 kernel 常已按当前后端塑形

自然度较高的例子是 `add_bias`：一个 VLA、两次 load、一次 add、一次 store。

复杂例子则大量显式拥有：

- blocked GEMM 的 `BM/BN/BK`、三重 cache loop 与 block materialization：
  `examples/kernels/contraction/blocked_gemm.py:17-55`；
- dense conv 的完整 patch/weight staging，再进入 `[8,K]×[K]` dot：
  `examples/kernels/vision/dense_conv2d.py:33-129`；
- IME projection 的 activation quantization、N16×K32 packed offsets 与 fixed extension primitive：
  `examples/kernels/ime/q4_k_projection.py:19-97`；
- attention 的 explicit scratch 与完整 online state traversal；
- GGML quant block 的 byte offsets、nibble/high-bit unpack 与 format-local dot。

这些结构有些确实属于作者：cache blocking、algorithmic staging、persistent layout 不应由 compiler
自动发明。因此不能简单要求“全部从 source 删除”。审计结论是另一条：

> 如果一种 fast path 只接受在其开发过程中同步塑形出的 source，就不能用这些 source 证明
> realizer 已经理解相同 primitive 的自然组合。

### 7.3 等价写法证据存在，但规模和强度不足

Runner 有 7 组显式 equivalent pair：segmented scan、Top-K、SSM conv、ROI Align、AdamW、
interleaved complex 和 lookup。它们能排除部分无关 SSA/source spelling 依赖。

但它们不能覆盖：

- dot result 的新 consumer；
- matmul 在陌生 outer context；
- block value 多 use；
- state 与 indexed memory 的跨 region handoff；
- workspace/layout ownership 变化；
- shape、stride、alias 与 mask 的合法变化。

本轮 dot 探针已经证明：现有 equivalent pairs 没有排除关键 composition overfitting。

### 7.4 Worker-local 不是统一 range ABI，也不需要被强行统一

有些 kernel 接收 `begin/end` 或 row slice，有些一次处理完整 problem。代码中没有 grid/hart identity，
所有 kernel 都由调用方在当前 hart 上直接调用。

这不违反 worker-local：worker-local 描述“谁执行”，不要求所有 entry 都用统一 `work_begin/end`。
真正缺的是可嵌入的 runtime/artifact contract，而不是再造一个 core grid abstraction。


## 八、Runtime 与性能：真实数字，不是成熟度证明

### 8.1 当前数字能证明的范围

`report/weft-kernel-performance.csv` 当前有 134 条成功记录、64 个 distinct kernel：

| 项目 | 数量 |
| --- | ---: |
| SG2044 / RVV VLEN128 | 66 |
| K1/X60 / RVV VLEN256 | 66 |
| K1/X60 / IME1 | 2 |
| full correctness | 104 |
| sampled correctness | 22 |
| full_rows correctness | 8 |

这些记录证明相应 source、shape、runtime 和 target profile曾成功执行，并给出对应 scope 的 median。

### 8.2 它们不是一个统一实验总体

CSV 中同时存在：

- repetitions 3/5/7/10/30；
- warmup 0 或 3；
- full、sampled、full_rows correctness；
- preprocessing 计时与不计时；
- 不同 throughput unit；
- 不同算法 scope 与 packed ABI。

多数普通 runtime 在 timed loop 前已经做 correctness invocation，因此 CSV 的 `warmup=0` 也不能
简单解释为严格冷启动。仓库没有逐次 raw samples、失败运行清单或统一 target environment snapshot。

这不要求给 CSV 加验证逻辑；它只意味着这张表是数字记录，不能承担编程模型证明。

### 8.3 GGML baseline 不能整表横向相除

GGML baseline 与 Weft CSV 的 scope、preprocessing、shape、implementation family 和 correctness
协议并不普遍一致。尤其 IME projection 的 Weft scope包含 activation quantization，而 baseline
可能只测 packed compute。只有同 target、同算法、同 shape、同 layout/preprocess ownership 的局部
pair 才能作性能结论。

### 8.4 当前性能事实仍显示 candidate space 很窄

CSV 中 VLEN256 并未系统优于 VLEN128，GEMM、vision contraction、state 与多种 quant family 在 K1
仍有明显低吞吐。两个 segment2 kernel 有真实局部收益，两个 IME leaf 也确实比其 K1 RVV realization
快，但这只能证明对应 local realization 有效。

不能据此声称：

- Weft 整体超过 GGML；
- VLEN256 target adaptation 已成熟；
- matmul/quant/state physical search 已成熟；
- Weft 已有 Triton 式 autotuning；
- IME 已成为通用 matmul realization。


## 九、Compiler product 边界也还没有闭合

### 9.1 `weft-compile` 只是 Kernel IR → intrinsic C front door

当前 CLI 能 parse/verify textual MLIR，并输出 `kernel-ir` 或 `intrinsic-c`：
`tools/weft-compile/weft-compile.cpp:85-147`。

它没有：

- install/package/export target；
- stable external construction library；
- C header emitter；
- object/archive driver；
- workspace/layout declaration；
- generic compile-and-tune loop；
- artifact metadata。

这不影响编译核心成立，但影响“像 Triton 一样可被真实用户消费”。

### 9.2 `examples/run/weft.sh` 是 benchmark harness，不是通用 compiler driver

完整 Python→C→remote object→runtime 路径目前只在 `examples/run/weft.sh` 中闭合。它硬编码：

- 三个 profile；
- remote host、CPU、compiler 与 library path；
- kernel 名到 source/runtime 的 catalog；
- quant runtime glue；
- meta/config 特例。

见 `examples/run/weft.sh:15-64, 91-639, 652-765`。

Quant repro 会复制 GGML header，Q4_K 对照还链接 GGML library。这是 benchmark reference/harness
依赖，不是生成 kernel 调用 GGML fallback；但也说明现有 runner 不能被当作独立 production compiler
workflow。

### 9.3 External frontend contract 只有 textual MLIR

IntentDSL 或其他前端理论上可直接生成 canonical Weft IR，当前真实 contract 就是 generic textual
MLIR schema。仓库没有专用 builder、installed dialect package、schema discovery 或 artifact API。

这条路可以先保持很薄，不需要把 IntentDSL 拉进仓库；但“其他 frontend 已可稳定纳入 Weft”目前
只能理解为“它可以手工打印当前 dialect 文本”，不是成熟集成接口。


## 十、与 Triton 级别的准确差距

Weft 不应复制 Triton 的 GPU grid，但可以用 Triton 的成熟度标准检查自己：

| 维度 | Triton 级别应有的性质 | 当前 Weft |
| --- | --- | --- |
| 根程序模型 | 作者知道一个 program instance/worker 对什么负责 | worker-local 已成立，但 storage/workspace 未冻结 |
| 逻辑数据域 | block/tile/value 能自然组合 | VLA 较完整；block/structured result 常被 exact closure 限制 |
| 授权边界 | 显式构造授权 target 重组 | scalar/VLA/dot 边界正确，但 public op 与 backend 不闭合 |
| SSA composition | producer 换普通 consumer 不改变能力类别 | `dot → add → store` 失败 |
| Physical planning | layout/memory/microtile/resource 一次选择 | 已有骨架；仍有 family matcher 与 emitter hardcode |
| Candidate space | shape/target 驱动多个合法实现并可实测选择 | 多数 family 固定或首个合法，pipeline 实际被禁止 |
| Target facts | ISA/ABI/resource facts统一进入 legality | profile 仍是字符串启发式 |
| Public surface | 暴露能力与 backend artifact 闭合 | 存在 widen/atomic/fence/permute 等假能力 |
| Product front door | 普通用户可编译、打包、调用 | 依赖 textual MLIR CLI + hardcoded repro runner |
| 证据 | 自然程序、等价写法、shape/target变化共同验证 | corpus 广但预塑形明显，等价 probe 有限 |

因此，Weft 与 Triton 的最大差距当前不是“少几个 intrinsic”，也不是“没有 GPU SIMT”。最大差距是：

> **作者程序与 physical realizer 之间还没有一个对所有公开 primitive都稳定成立的组合合同。**


## 十一、哪些设计应冻结，哪些必须重新打开

### 11.1 应继续冻结

以下方向经代码与 repro 复核后仍然正确：

- 单 worker/hart callable，不引入 core grid、program ID 或隐式 hart identity；
- scalar ordered loop 与 VLA logical axis明确分开；
- 只有显式 VLA 授权 SIMD strip realization；
- 只有显式 dot/matmul 授权 local reduction/microkernel 重组；
- reduce、scan、typed summary 与 sequential carry保持不同语义；
- 不恢复 arbitrary-axis generic contraction；
- canonical Kernel IR 是唯一持久 algorithm representation；
- physical plan 只在一次 target lowering 中存在；
- system compiler负责最终 register allocation、machine scheduling 和 peephole；
- unsupported 明确失败，不增加 scalar/legacy/GGML/materials fallback。

### 11.2 必须重新打开

当前不能继续当作已解决的问题：

1. **Storage model**：caller workspace、algorithmic staging、persistent layout 与 primitive-local
   temporary 分别如何表达和进入 artifact ABI；
2. **Block/value composition**：structured result 怎样进入 pointwise、多 use、state 与 memory
   consumer，而不要求 direct store；
3. **Public surface truthfulness**：没有完整 canonical schema 或正式 target realization 的构造是否
   应直接删除；
4. **Validity/extent contract**：masked dot、fill、atomic、lookup/decode 与 dynamic init 的 canonical
   语义；
5. **Extension granularity**：哪些 packed/quant relation值得一个 semantic op，哪些只是 layout/leaf
   spelling；
6. **Physical authority**：value shape、handoff、layout、fragment、resource 与 schedule必须全部先由
   planner产生；
7. **Candidate/tuning**：从固定 preference扩展到真实合法候选和外部实测选择；
8. **Target contract**：march/ABI/VLEN/vendor extension/resource facts统一解析；
9. **Artifact/front door**：外部 frontend与runtime如何在不依赖 benchmark catalog 的情况下消费。

### 11.3 不建议推倒全部重写

不需要重新开始的资产已经很明确：

- Python AST→canonical IR 的基本路径；
- scalar/VLA 的语言边界；
- Kernel IR value/control/type骨架；
- 逐实体 VLA access/predicate/state facts；
- transient `PhysicalEntityPlan`；
- intrinsic-C 与 local asm leaf；
- 明确 unsupported、无 fallback 的主链。

真正需要的是一次**围绕编程模型收敛的结构重构**，而不是再加 kernel，也不是在现有 exact matcher
旁边加第二条通用路径。尤其应避免：

```text
natural composition 不支持
→ 增加一个新的 closure matcher
→ 新 corpus 命中
→ 再把 PASS 数量当作模型成立
```


## 十二、问题登记表与关闭判据

### A. 语言与 canonical correctness

| 问题 | 当前证据 | 正确 owner | 关闭判据 |
| --- | --- | --- | --- |
| `W.widen` 无 registered op | frontend emits generic unknown op，IR 路径仍接受 | public DSL + Kernel dialect | 删除 surface，或有正式 op/verifier/target repro；未知同 namespace op 必须在 canonical 边界拒绝 |
| helper annotation 被忽略 | F32 value 可传给标注 `W.i32` 参数 | Python frontend | annotation 被拒绝或成为真实 type contract |
| atomic/fence/permute 等假能力 | frontend/IR 有，RISC-V dispatcher 无 | DSL surface + target | 删除未支持 public API，或各自通过一条真实 DSL→run repro |
| masked dot validity 未闭合 | verifier unwrap masked，result contract 不完整 | Kernel IR verifier | valid/invalid lane对accumulator/result的语义唯一且被 frontend/verifier/target共同消费 |
| fill/atomic footprint不足 | verifier只检查部分element/footprint关系 | Kernel IR verifier | shaped/scalar合法关系统一由logical footprint verifier接受或拒绝 |
| lookup/decode contract过宽 | surface一般，target固定table-16 | DSL/IR semantic op | surface明确真实语义域；越界/shape/dtype policy在canonical层闭合 |

### B. 编程模型

| 问题 | 当前证据 | 正确 owner | 关闭判据 |
| --- | --- | --- | --- |
| workspace/lifetime未定义 | attention/conv/IME通过raw pointer传scratch | DSL + artifact ABI | caller/persistent/local三类storage所有权、shape/alignment/lifetime可从source/artifact唯一得知 |
| dot普通consumer失败 | `dot → add → store` IR成功、target失败 | Physical planner | 不改算法source即可为dot result建立handoff并完成真实run |
| block materialization语义不稳 | transforms只在特定closure中可用 | Kernel IR + planner | 同一block value可在合法pointwise/state/memory consumer间组合，unsupported在primitive边界报错 |
| persistent packed ABI未产品化 | IME runtime手工repack 304-byte layout | 上游/runtime + artifact contract | layout identity、builder、size/alignment与target compatibility明确，不由leaf/runtime私约定 |

### C. Physical realizer

| 问题 | 当前证据 | 正确 owner | 关闭判据 |
| --- | --- | --- | --- |
| exact use/adjacency closure | dot one-use/direct store、block getNextNode | entity analysis/decision | decision读取semantic anchor、axes、typed use relation；无关consumer插入不改变能力类别 |
| region traversal不完整 | VLA只递归nested For | domain/entity analysis | 所有合法structured region中的实体都被同一作用域规则遍历 |
| emitter保留shape/layout决定 | f16 matmul、block index/decode硬编码 | physical decision | emitter只读取selected typed decision；资源模型与实际spelling一一对应 |
| pipeline只是字段 | planner强制0 group/1 stage | candidate/resource model | 至少一个共享primitive有多个合法pipeline/schedule candidate并由同一resource equation选择 |
| quant资源模型写死 | fixed groups/unroll常量 | primitive candidate model | live operand/state/temporary/fragment facts导出resource；常量只表达ISA固定fragment |
| target profile过浅 | contains('v')、固定32 regs | target facts | ISA/ABI/VLEN/extension/resource legality由统一profile提供 |

### D. 证据与产品边界

| 问题 | 当前证据 | 正确 owner | 关闭判据 |
| --- | --- | --- | --- |
| corpus预塑形 |复杂kernel显式适配fast path | kernel corpus选择 | 事后冻结的自然程序和等价consumer组合进入同一realizer，不新增matcher |
| 两个source无runtime | source存在但runner断链 | examples/repro | 要么删除未声明能力，要么各有一条真实手动repro |
| correctness scope不统一 | 30/134不是full | runtime记录 | 每条结论只按CSV scope表述；sampled不再被概括成full |
| 性能scope不可整表比较 | preprocess/repetition/unit混合 | report解释 | 只对同target/算法/shape/scope的pair作性能判断 |
| hardcoded runner | kernel catalog + remote paths | compiler/artifact driver | 普通source/IR可在无catalog分支时生成object/header并被外部runtime调用 |


## 十三、以后继续开发时的固定提问顺序

遇到一个新 kernel 或性能问题时，按下面顺序判断，可以避免重新滑回图编译器或模式库：

### 第一步：作者是否已经写下算法结构

Traversal、blocking、algorithmic staging、persistent layout、state 与 algorithm variant若未写下，
target不得猜。若这些是算法必需，修 DSL/source contract。

### 第二步：canonical primitive 是否完整表达局部语义

检查 axis、extent、validity、effect、numerical policy、storage lifetime与typed operands。缺的是可观察
语义才加 primitive；不要为 format名或kernel名加op。

### 第三步：失败是否只来自无关 source closure

把 primitive结果换一个普通consumer、增加等价SSA临时值、移动到合法structured region。若因此
失去realization，修 entity facts/handoff，不加新的kernel matcher，也不规范化作者程序。

### 第四步：这是派生事实、selected decision还是spelling

```text
axis/use/effect relation          → derived entity fact
LMUL/microtile/memory/fragment    → selected physical decision
intrinsic/asm/ABI text            → emitter spelling
```

三者不得重复拥有同一事实。

### 第五步：candidate 是否真实

数据结构里有字段不算候选。必须存在多个合法实现、资源方程、target/shape差异和可选的构建期实测，
且失败候选不会变成kernel-specific永久规则。

### 第六步：真实 artifact 边界在哪里

最终只用一条可手动执行的 DSL→IR→backend code→system compiler→run repro判断支持。停在IR或C时，
就如实写停在哪，不增加测试框架或假成功记录。


## 最终结论

此前报告中“一个清楚且唯一的 worker-local 语言/IR 模型已经成立”的判断需要收回一半。

准确说法是：

> **Weft 已经有一个正确方向的 worker-local 根模型、一个有实质内容的 canonical Kernel IR，
> 以及一个开始形成逐实体 physical planning 的 RISC-V compiler core；但 public DSL、storage
> ownership、block/structured composition、physical candidate 与 artifact contract 尚未形成同一套
> 闭合编程模型。**

当前最危险的退化方向不是重新变成 graph compiler，而是变成：

```text
作者先把程序写成backend喜欢的局部closure
→ compiler识别该closure
→ 生成高性能leaf
→ 用越来越多成功kernel证明“已经泛化”
```

这仍然是高性能模式库，只是入口从 kernel name 换成了 source closure。

Weft 不需要推倒重写，也不需要恢复 general contract 或 GPU SIMT 根模型。它需要保留已经正确的
worker-local/VLA/Kernel IR/physical-plan 主干，同时停止扩 kernel 数量，先完成三件结构性收敛：

```text
公开能力 = canonical语义 = 正式target artifact

structured primitive结果可通过普通SSA/use-def自然组合

storage、handoff、resource与schedule由一个physical authority产生
```

只有这三条对自然程序成立之后，“能跑很多 kernel”才开始接近 Triton 所代表的编译器能力，而不只是
一批经过共同设计的成功案例。
