# DSL 表面、编译贡献与调优组织调查

调查基线：`57bf55370`。本轮仅阅读源码、设计文档、Git 与既有实验记录，没有修改编译器、kernel、runner 或 CSV，没有运行性能。历史测量只用于说明当时的变化，不代表当前快照的消融结果。

## 一、DSL 语法与命名

### 覆盖与结论

完整审阅 `examples/kernels/` 的实际 111 个 Python 文件、111 个 `@weft.kernel`，共 3819 行：mul_mat 52、vec_dot 24、dequantize 24、quantize 3、dense 3、gemv 3、attention 1、selection 1；另读取 Encoding 与 quantization helpers。prompt 中的 109 已不是当前数量。

命名空间、kernel 装饰器和普通 helper 的骨架自然，不需要为模仿 Triton 再换抽象。主要阅读障碍是隐藏的符号、birth 与调用配置。接口建议如下；本轮均未实施。

| 当前表面 | 建议 | 原因与影响面 |
|---|---|---|
| `admit(region)` | 改叫 `load(region)` | 实际为 View→Value 读取；84 个 kernel 文件、162 处调用 |
| `commit(value, region)` | 改叫 `store(region, value)` | 只有写 effect，没有事务或 barrier；111 个文件、125 处调用 |
| `contract(..., acc=i32)` | 保留 contract，参数改成 `acc_dtype=` | Triton 的 acc 是 accumulator value，本项目是 dtype；48 个文件、57 处显式 acc 调用 |
| `joined(4,2,4,lo_first)` | 支持命名参数 | `group/fields/low_bits/order` 更可读；Encoding 文件中 4 处声明 |
| `L.subs(extent=...)` | 示例统一显式父点 | 7 处依赖隐式当前域；风格改动即可 |

依据：[前端读写](../python/weft/frontend/compiler.py:1754)、[contraction 解析](../python/weft/frontend/compiler.py:2116)、[Encoding 关键字参数目前被拒绝](../python/weft/frontend/compiler.py:798)。以上文件数是 kernel 表面影响，不包含 frontend、导出 API、文档和 runtime 符号的连带改动。

不建议把 contract 全改为 dot：前者支持显式多轴缩并；Triton `tl.dot` 主要接受二维/三维矩阵块。当前 outer_contract 比 contract 多一个“双方 free axes 不相交”的 verifier 条件，并非完全等价别名。应讲清三者区别，不能让用户依据 backend 偏好挑名字。[canonical verifier](../lib/Dialect/Kernel/IR/KernelDialect.cpp:1526)

`new` 表示可更新 state，`materialize` 表示 staged value，区别应保留。grouped/layered 的布局含义也成立；目前没有充分理由只为美观改名。

### 更重要的隐藏约定

1. **隐式符号。** Shape annotation 中未声明的 M/N/K 被前端自动建立为 shape symbols；29 个 kernel 中的 117 处 auto 全部只声明名字，候选域不在 kernel 旁。`auto("MR")` 还允许后文使用 MR。Triton 的 constexpr 参数和 TileLang 示例的 `T.const` 更容易找到声明处。[shape symbols](../python/weft/frontend/compiler.py:424)、[auto](../python/weft/frontend/compiler.py:2432)
2. **birth 与书写顺序。** 前端把 Level 顶层 new/materialize 赋值抽到独立 birth regions。因此先写 `w=admit(...)`，随后 `materialize(f(w))` 不一定合法。仅改名解决不了；需要让 birth 可见域与诊断更明确，具体表面方案尚未定。[实现](../python/weft/frontend/compiler.py:2309)
3. **调用不在程序旁。** KernelDefinition 捕获 AOT source，直接调用会报错；参数、entry dispatch、ABI runtime 在 shell 中。问题不是 AOT 本身，而是一个用户无法从 kernel 文件找到完整 build/call 关系。[API](../python/weft/api/definitions.py:36)

代码长度没有支持“Weft 普遍更冗长”的结论。两次读取加一次 contraction，三者都约三条主体语句；额外代码承担不同职责。Weft 的 Level、Encoding、有限位宽，不能直接对比 Triton 的 pointer/mask/launch 或 TileLang 的 storage scopes。当前六行 dense GEMM 没有显式 KC staging，不能拿它与完整 Triton GEMM 作同算法 LOC 胜负比较；Top-K 的 heap 与参考 repeated-argmax 也不是同一算法。

参考：[Triton vector add 的 load/store 与 launch](/home/kingdom/phdworks/ref/triton/python/tutorials/01-vector-add.py:29)、[TileLang GEMM 的参数与编译调用](/home/kingdom/phdworks/ref/tilelang/examples/gemm/example_gemm.py:6)。

## 二、性能贡献与编译结构

### 可以归因到什么程度

139/202 超过 source 是整套作者程序、参数、编译器和系统编译器共同结果。202 行含 94 行旧全量、108 行后续定向测量，不能由此计算 pass 贡献百分比。

| 历史证据 | 可支持的归因与限制 |
|---|---|
| `c29d835ac` TQ2 partial：SG 1.564→19.977，K1 3.681→6.245 | 作者树、runner 未改，支持编译器 partial 物理组织收益；并非将 pass 与新增 leaf 的贡献进一步分离 |
| `d4e8525b0` IQ2_S row：SG 359.274→673.905 | 仅 runner LMUL 改动，属于参数绑定利用既有 compiler 能力 |
| `35a9711cf` IQ2_XXS row：SG 345.542→692.395 | 作者 entry/payload 改写与 LMUL 绑定共同作用，不能全归 pass |
| `f822bd54e` IQ2_XS full-product | product/slide 的实际消除成立，但 SG unroll 同时改变，不是纯 pass 消融 |
| `00afcbbce` reduction-lane | 同树跨格式大涨，但 MR/NR 同提交变化，是 carrier 与参数联合证据 |
| `43a772aad` Q3 scalar-prime | 隔离实验中其它 IR 基本不变，增加局部 scalar read 有显著收益；属于 selected memory leaf 变体 |
| `a9437b00e` two-stream combine | 同树改变 combine/reduction 指令图，target policy 与 leaf 同时扩展，属于联合 compiler mechanism |

依据：[partial 历史结果](history/riscv-typed-partial-accumulation-and-q2-tree-boundary.md:150)、[阶段合并报告](post-ledger-kernel-closure-consolidated.md)、[scalar-prime 隔离实验](riscv-full-account-cluster-closure.md:287)。历史 source 口径不能用于替换当前 source 比值。

### 每个 pass 实际承担的工作

[当前主链](../lib/Target/RISCVCompiler.cpp:24) 为 33 次调用、24 种 pass；下表是源码职责核验，不是本轮逐入口 instrumentation 的命中数量。

| pass | 次数 | 实际输出 |
|---|---:|---|
| ConvertWeftToRISCV | 1 | Canonical→Physical ops/types/control/memory descriptors |
| SelectRISCVOperations | 4 | 依据 typed facts 写 implementation |
| PropagateRISCVLayouts | 1 | 改 value layout/type，插 conversion/broadcast |
| PlanRISCVMemory | 3 | 改 access/leaf，物化 repeat/indexed 等 memory ops |
| CanonicalizeRISCVLayouts | 3 | 消 conversion，受限 producer rematerialization、conversion CSE |
| LowerRISCVComposites | 1 | composite→RVV/IME/local ops 与 reduction 程序 |
| MaterializeRISCVPrograms | 1 | macro→真实 scf、load/step/use-def |
| VectorizeRISCVRecordLoops | 1 | 可证明的 record cohort、主循环与 tail |
| PlanRISCVNestedMemory | 1 | 物化 point 创建后才闭合的 nested memory relation |
| FuseRISCVBitplanes | 1 | typed bitmask/plane merge，替换 decode 链 |
| HoistRISCVLoopInvariants | 2 | 移动不变 producer，冻结部分 scalar supply |
| ScheduleRISCVLevels | 1 | 写 stage/order |
| PipelineRISCVLevels | 1 | 消费 schedule，生成版本化 SSA 与三段循环 |
| SCCP | 1 | 常量传播 |
| ShareRISCVLayeredWindows | 2 | raw window 共享及显式 layer decode |
| PlanRISCVPartialTopologies | 1 | 冻结 carrier/partial/combine/supply/resource plan |
| MaterializeRISCVPartialAccumulators | 1 | 生成 partial ops、issue loops、windows |
| UnrollRISCVLevels | 1 | 机械复制迭代 |
| MaterializeRISCVReplicaStorageLoads | 1 | 最终 field→register memory ops |
| FinalizeRISCVLeaves | 1 | 普通 operation/conversion 的 exact leaf |
| SelectRISCVScalarLoadPrimes | 1 | 选择局部 memory leaf 变体 |
| MaterializeRISCVResources | 1 | live-set 分析、显式 spill/reload 与资源汇总 |
| EliminateDeadRISCVLayouts | 1 | DCE、清理无用 loop carry |
| VerifyFinalRISCV | 1 | 验证 terminal 合同 |

写属性不等于假 pass：implementation、schedule、partial plan 都有后续真实消费者。问题在信息是否完整、是否被另一 owner 重选。

### 这次确认的剩余问题

- **materializer 仍选 schedule。** nested materializer 根据 windowExtent、unroll、资源数决定 operation-major，而 plan 没有冻结这一选择。[代码](../lib/Target/MaterializeRISCVPartialAccumulators.cpp:5890)
- **通用流水看不到后生成程序。** Schedule/Pipeline 在 partial materialization 之前；后来新建的 issue loops、index/load/partial use-def 没有再经过通用 scheduler。不是简单放宽 pipeline 条件就能覆盖。
- **scalar load binding 是延迟表达式。** compileAdmit 保存的是解引用表达式，不是读取点的 C 变量；local load 同样如此。重复 use 可能重复读取，跨别名写入时有顺序风险。源码行为已确认，本轮没有运行数值反例，不能宣称现有某个 kernel 已算错。[dense scalar](../lib/Target/RISCVIntrinsicC.cpp:1758)、[local scalar](../lib/Target/RISCVIntrinsicC.cpp:7147)
- **reload emitter 按 consumer 决定加载 part。** handler 扫 uses，针对 PartialSet 筛选实际 reload；这个决定应在 Physical IR 中可见，否则 IR 工作账与发出的加载数不一致。[代码](../lib/Target/RISCVIntrinsicC.cpp:7412)
- **中间态独立重放不足。** weft-opt 只注册三个项目 pass；compiler 没接完整 pass instrumentation/replay 入口。[注册](../tools/weft-opt/weft-opt.cpp:12)

grouped-MAC 固定 slots/terms、IME 固定 fragment packing 和已计入 local_bytes 的有限 scratch，不能仅因产生多条语句就判越权。Triton FMA lowering 同样按已选 operand 结构生成多条 LLVM op；其 load lowering 产生实际 SSA result，后续 consumers 复用结果。[FMA](/home/kingdom/phdworks/ref/triton/lib/Conversion/TritonGPUToLLVM/DotOpToLLVM/FMA.cpp:19)、[load](/home/kingdom/phdworks/ref/triton/third_party/nvidia/lib/TritonNVIDIAGPUToLLVM/LoadStoreOpToLLVM.cpp:322)

此前“parse/verify+CSE 二次 diff=0 证明所有 pass 幂等/正确”的说法应收窄。它只证明该样本在所重放流程上稳定；final resource marker 还会跳过 layout canonicalization 的主要 rematerialization 分支。[代码](../lib/Target/CanonicalizeRISCVLayouts.cpp:55)

要量化 pass 贡献，必须比较同树同 binding 下两份都合法的 Physical program。关闭必需 lowering 后编译失败不能当性能消融。当前没有完整消融数据，所以不提供贡献百分比。

建议优先修上述 owner、SSA effect 与观察接口，影响 Physical passes/emitter/tools，不必修改 DSL 抽象。

## 三、autotune 配置的位置

### 当前分布

C++ 保存 ISA 合法域、默认值、结构策略与算法；kernel 保存 auto 名称；四个 runner 保存实际 target/entry/phase 配方；tuner 从环境变量取候选域。

下表通过只解释 runner 参数赋值段核对，没有执行编译、远程命令或性能。组合包括脚本接受但目前可能编译失败的请求，不代表合法 artifact 数；各 runner 之间有重复入口，不能相加作独立 winner 数。

| runner | 调用组合 | 有非默认 physical 参数 | 有 source meta |
|---|---:|---:|---:|
| weft-kernel.sh | 15 | 6 | 5 |
| weft-mul-mat.sh 标准 26 格式 | 104 | 53 | 54 |
| weft-quantized-vec-dot.sh | 48 | 26 | 0 |
| weft-row-dequantize.sh | 48 | 6 | 0 |

默认 physical 值为 LMUL=m1、unroll=1、pipeline=1、scalar-prime=0。显式赋值字段数依次为 12/254/91/6；其中非默认字段为 7/66/35/6。这个数量说明配置散布程度，不能证明每个绑定都由完整扫描得出。[CLI](../tools/weft-compile/weft-compile.cpp:31)

[tuner](../examples/run/weft-kernel-tune.sh:64) 默认扫 LMUL={1,2,4,8}、unroll={1,2,4}、pipeline={1,2}；source meta 域由环境变量提供。默认会再次运行 winner，但不持久化 winner 配置。下次 production 仍读 runner 手填值。默认扫描也不含部分已用的 unroll=8；scalar-prime、结构 policy 没有独立扫描维度。

当前不能精确给出“多少个绑定是真正测遍候选后的 winner”：CSV configuration 和历史报告没有为每个绑定保存完整候选域及选择来源。

### 建议组织

1. **kernel 旁的 Python 配置**：source 参数声明、有限候选域、entry 与 build/call 关系。
2. **target profile 数据**：机器事实和可调整策略参数可用 TOML；合法性算法、layout 传播和指令合同仍留代码。
3. **集中实测绑定**：关联精确 entry、shape、target、编译配置、候选域和结果，由 production 消费。

这会修改四个 runner、tuner 与 build 配置入口，规模中等；不需要改变数值树。不能只把 if 逐行搬到一个大配置文件，就宣称调优已接通。

Triton Config 与 autotuner 保存作者配置、按 key 选择并可缓存结果；TileLang 同时支持 kernel 旁 decorator 与程序化 autotuner。[Triton](/home/kingdom/phdworks/ref/triton/python/triton/runtime/autotuner.py:217)、[TileLang](/home/kingdom/phdworks/ref/tilelang/examples/gemm/example_gemm_autotune.py:155)

还有入口身份缺口：runner 的 q4_k prefill 自动选择 staged，而表中仍叫 mul_mat_q4_k。底层记录应保存真实 DSL entry/输入表示，简表可继续保持简洁。[dispatch](../examples/run/weft-mul-mat.sh:37)

## 四、哪些规则过度保守

### 建议放宽的工程政策

- **绝对禁止 cost model。** Triton rematerialization 比较 conversion 与重算成本；TileLang 枚举 inference roots 并按 memory/register 成本选择。Weft 自己的 spill victim 也在用 groups×lifetime 排序。这些不是合法性证明，但可以是可解释、可反证的决策依据。[Triton](/home/kingdom/phdworks/ref/triton/lib/Dialect/TritonGPU/Transforms/RemoveLayoutConversions.cpp:1044)、[TileLang](/home/kingdom/phdworks/ref/tilelang/src/transform/layout_inference/layout_inference.cc:1081)、[Weft spill](../lib/Target/MaterializeRISCVResources.cpp:343)
- **绝对禁止有限结构比较。** 可允许对少量已闭合 Physical structures 排序或实测，固定 canonical numerical program。有限选择不会增加第三层 IR，也不必依赖隐藏 fallback。
- **tuner 只能扫数字参数。** TileLang config 可以带 pass_configs，GEMM 还扫 rasterization；Triton meta 也不限于 tile 大小。可以允许显式策略选项，但必须记录候选来源与真实 entry。[TileLang 合并配置](/home/kingdom/phdworks/ref/tilelang/tilelang/autotuner/tuner.py:493)
- **第二输入作为实现禁令。** 保留“一个正例不宣称泛化”；不要扩大成“没有第二个现成格式就不能实现合同完整的关系”。独立 half-byte projection 与为了某个 donor 捆成的大组合 op，需要分开判断。

放宽前两项主要影响 model/non-goals、compiler candidate API、target 策略和调优记录。应限制搜索规模、让估计可观测，不能让成本分数覆盖数值/effect/资源合法性。

### 需要单独确认的语言约定

“普通 for 永不向量化”不是保持可观察顺序的必然推论。独立迭代可以在正确依赖/effect 证明下物理向量化；TileLang 有这类 strip-mine 机制，但不能据此声称它自动向量化任意循环。[实现](/home/kingdom/phdworks/ref/tilelang/src/transform/loop_vectorize.cc:1003)

Weft 可以继续要求作者显式 shaped axes，作为项目的可预测编程约定；若放开，则涉及 iteration identity、birth/handoff、alias 和 numerical boundary，属于较大语义/实现工作，本轮不建议作为小修顺手实施。

### 应保留的边界

- 不偷偷改变作者数值语义、Level/lifetime 和跨调用 ABI；
- 不以 kernel/格式名接管 whole-kernel lowering；
- 物理选择进入可观察的 IR；emitter 不临时决定共享、spill、调度；
- 无足够证据时不宣称泛化，负结果照实保留。

target backend 按机器能力分支是正常职责。Triton AccelerateMatmul 按 compute capability 选择 MMA 版本，也按 shape/use relation 使用启发式；这不等于 compiler 可以按量化格式偷偷替换作者程序。[源码](/home/kingdom/phdworks/ref/triton/lib/Dialect/TritonGPU/Transforms/AccelerateMatmul.cpp:42)

调查没有证明需要推翻核心抽象。已确定的问题集中在：用户看不见完整参数/调用合同；部分 materializer/emitter 决策仍未归位；归因证据被写得过满；工程政策禁止了参考实现实际依赖的局部比较。性能贡献百分比、scalar 延迟读取的具体运行反例，以及新策略的实际收益，本轮均未测，保持未定。
