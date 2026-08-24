# 编程模型非目标

本附录列出不属于 Weft 编程模型的构造。每项都给出语义原因，不记录设计过程。

## 1. Worker、hart 或 program tile 作为根对象

Weft kernel 没有隐式 worker/hart identity，也不以“一个 program instance拥有一个tile”为语义。并行调用和任务分配属于调用方。

若把worker设为根，Level lifetime、Encoding和numerical tree会被迫依附于一个并不存在于所有target上的任务分解；同一kernel也无法自然表示一次调用内的完整有序控制。

## 2. Grid/SIMT ownership 作为根布局

语言不提供program id、thread id、warp/CTA owner或logical coordinate→thread owner映射。目标后端可以把shaped axis映射到它的物理执行资源，但该mapping不是source semantic。

否则CPU/vector/matrix目标必须先模拟一个SIMT ownership，再恢复单控制器的register/lifetime结构。

## 3. VLA region作为整门语言的根

宽执行只能来自显式shaped Value、Level cohort或closed primitive。语言不把所有程序包进一个VLA region，也不把普通scalar loop自动向量化。

VLA是RVV实现能力，不足以表达Encoding、state birth、staged lifetime、非归约handoff或普通不规则控制。

## 4. Block/tile作为统一源类型

Value有logical shape和axis，但不是目标tile：它没有owner、register layout、shared scope或静态2的幂限制。

把tile设为根会把某一种target物理对象提升为所有kernel必须采用的source结构；Weft只在target physical program中选择这些对象。

## 5. Stream + resident state作为根模型

stream、resident state、summary state和ordered carry可以描述部分kernel能力，但不能统一Encoding、persistent artifact、GEMM staging、ordinary control和多支路numeric tree。

Weft直接使用Level births/staged/handoff和普通SSA，不把其中一种常见组合命名为特权kernel类别。

## 6. 硬件storage scope进入DSL

不提供：

```text
alloc_l1
alloc_cache
alloc_vector_register
alloc_ime_fragment
```

CPU cache通常由硬件管理，物理register由下游compiler分配，fragment属于target realization。作者表达的是logical lifetime、materialization、workspace和persistent layout，不是它们最终落在哪级硬件。

## 7. Permission bit

不提供：

```text
allow_reassociation
allow_repack
allow_low_precision
allow_matrix
```

permission没有独立语义。若一种变化会改变logical tree，它必须成为另一个operation、type或stdtree；若不改变，它本来就在编译器表示权限内。精度和结合性必须写进具体operation语义，不使用全局许可。

这不等于成熟 DSL 不能有 assumption、schedule metadata 或 target hint。Weft 的选择是把 target requirement 放进 build config，把 physical organization 放进 target machine；core DSL 不用一个 `allow_*` 位混淆这些层次。

## 8. 通用 `ordered` 属性

ordinary control本身有序；state/attention等顺序由SSA use-def和effect明确；closed primitive需要的结合/顺序由primitive定义。

一个没有消费者、只用于阻止本来就禁止的tree rewrite的`ordered`属性不增加语义。

## 9. 从普通SSA识别高级primitive

编译器不从：

```text
mul + add + reduce
max + exp + sum
固定数量lookup
相邻同构operation
```

识别dot、matmul、online softmax、codebook axis或量化格式。需要这些语义时，作者调用显式basic op或std函数。

这避免kernel-name、exact-op-count、source-closure matcher成为隐藏语言。

## 10. Invocation-local `pack(along=...)`

core DSL 不提供只保持数值、shape、axes 与 Level 不变的 local-pack operation。`materialize(expr)` 已经表达 staged Value 的诞生层、物化次数、lifetime 和复用域；local pack 的存在、连续方向、carrier 和 schema 属于 target physical representation。

若重排改变 logical axes，作者写真实的 reshape/transpose/index operation；若改变跨调用 bytes 与 ABI，作者写 derived Encoding。两者都不需要模糊的 pack 授权点。

## 11. Source engine role 与 `stage_handoff`

core DSL 不提供 `@wide`、`@matrix`、`@transfer` 或通用 `stage_handoff`。它们不改变 canonical operation/value/effect graph 或 Level 归属，只限制 target realization，因此属于 build config 和 physical machine。

`scalar/wide/matrix/transfer` 仍是 target engine 类别。若 build 必须使用 IME，写 `require=uses_extension(IME)`，不满足则失败。若 IME 需要另一棵 materialization/Level tree，作者选择另一份 std 函数；跨 engine 的 register/fragment/local-storage 交接是 physical conversion。

## 12. Built-in whole-kernel matmul或quant op

GEMM/GEMV/quantized MUL_MAT不是opaque language op。它们由同语言std函数组织outer traversal、Level、staged lifetime、state和local contract。persistent packing 通过 derived Encoding 表达；invocation-local packing 由 target 物理化。

允许的closed primitive只拥有局部operand/result numerical relation，不能拥有kernel ABI、outer traversal、persistent layout或workspace。

## 13. 两套用户DSL

不区分“普通用户语言”和“专家kernel语言”。std作者、应用作者和新格式作者使用相同Encoding、Value、Level、control和basic ops。

库可以提供更高层普通函数和重载，但它们没有隐藏IR、特殊verifier或后端捷径。

## 14. 自动规范化作者tree

编译器不把作者的blocking、staging、accumulator lifetime、persistent packing、普通control或algorithm variant重写成标准模板。

等价source spelling可以通过相同typed facts得到相同物理类别；实现这一点依赖SSA/axis/use-def分析，不依赖把source tree改成统一形状。

## 15. 全局物理开关或全局约束求解器

SEW/LMUL属于value，instruction属于op，memory form属于memory edge，pipeline属于Level/local cluster。它们不是kernel-global C/D字典。

唯一事实由实体的typed use推导；存在多个合法物理实现时，target按固定规则与优先级作结构选择；物理参数由有限实测调优选择。冲突用显式physical conversion表示；资源不足时spill或拒绝当前candidate。编译器不通过带回边的全局solver修改作者tree。

这条属于语言与target compiler的职责边界；具体physical IR和pass结构不在本部分定义。

## 16. 静态 cost model

Weft 不为结构性选择建立预测执行时间、带宽、cache 命中或综合得分的静态 cost model。当前目标的机器行为、编译器和 runtime 还不足以让这类模型成为可靠的规范组成部分；把未经验证的估计写成公共选择权威只会隐藏硬编码。

结构性选择由 target 的确定规则和固定优先级完成。规则可以读取 typed use-def、Encoding mapping、target legality 和资源上限，但不能用一个预测分数比较两个都合法的结构。

## 17. 编译器生成结构后竞赛

目标编译器不为同一个 source candidate 生成多种 lane/register/fragment、memory、pack 或 pipeline 结构，再通过静态排序或真机运行挑 winner。这样做会把编译时间、实现复杂度和结果可解释性绑定到无限增长的结构空间。

允许实测的只有有限参数绑定：作者显式声明的 source `auto`，以及 target 为已经固定的物理结构声明的 LMUL、schema 内 physical microtile extent、unroll、pipeline depth 和 buffer count 等参数。tuner 不生成新的结构，也不改变结构优先级。
