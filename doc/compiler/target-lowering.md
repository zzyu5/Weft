# RISC-V Target Lowering

## Target profile

每次 lowering 必须绑定一个 `RISCVTargetProfile`。当前 public profile由 `--march`、`--abi`、
`--vlen-bits` 与 `--matrix-extension` 构造，保存：

```text
target triple / march / ABI / XLEN / endianness
strict ABI kind、full RVV availability与fixed VLEN bits
scalar/vector f16、supported SEW与legal LMUL
indexed/segment memory与integer/float widening capability
architectural vector register count
matrix/vendor extension identity
```

Parser只为它能证明的ISA事实赋能：当前backend要求full `V`，不会把`Zve`子集冒充完整RVV；
`spacemit-ime1`要求RV64与精确VLEN256。Profile可保留原始march作诊断，但lowering legality读取
typed facts而不是散落的字符串判断。当前`intrinsic-c` target要求显式fixed VLEN；0只可作为
尚未绑定的profile值，不能进入codegen。Cache、throughput、latency与preferred unroll只能
作为hint排序合法物理配置，不能让非法实现变合法。

当前 `RISCVLoweringOptions` 的public输入包括target profile、source meta bindings与显式的
build-time backend config。后者只约束本次lowering中的LMUL、microtile、unroll与radix
candidate；它不进入canonical IR，也不成为持久表示或第二份语义authority。

## 内部 realization families

Target lowering 可以为同一 semantic primitive 实现多种局部 realization family，例如：

```text
scalar control/math   -> ordinary C / target libm spelling
VLA pointwise       -> RVV f32m2 / f32m4
reduce              -> scalar ordered / RVV tree / widening RVV
dot / matmul        -> RVV FMA microtile / RVV dot / IME fragment
lookup/decode       -> scalar / indexed RVV / table gather
memory permutation  -> RVV segment / indexed / extension permute
```

每个 family 的内部定义可以包含 capability predicate、物理参数空间、legality、resource
equation、local fusion envelope 和 emission hook。当前实现可以先只有一个固定合法配置；
当它扩展成多个candidate时，candidate也只存在于本次lowering，不拥有独立IR schema、
verifier、pipeline front door或长期provider registry。

所有transient decision由一个短生命周期的physical planning core联合形成。Planner读取每个
VLA、value/use、memory access、state与local primitive自身的typed facts，在一个共同resource
budget下建立：

```text
value/register physical shape
+ memory and handoff
+ primitive realization
+ loop-local schedule/pipeline
+ peak live resource
```

Load、cast、state、dot/matmul、extension result与store之间的SEW、LMUL、mask/index relation不能
各选一遍；冲突必须明确选择share、convert、rematerialize、reload、shuffle或primitive-local
pack。Peak resource同时计入live value、memory/index、predicate、state、primitive-private
temporary和pipeline footprint。Physical prefetch是source不可观察的schedule candidate，不要求
也不产生source prefetch op，且不得创造algorithmic staging。这个plan只存在于一次lowering
调用中，不是语言抽象、新IR或artifact contract。

这些decision不进入Kernel IR。每个decision由对应canonical primitive的唯一owner
产生。Intrinsic/asm emitter消费已经选定的mode、LMUL、vector shape和fragment；后续store、
cast或leaf不能再根据use count、周围op数量或完整kernel source重新选择一次。

一个 family 只能绑定：

```text
primitive/interface + typed operands + local use relation + target/config facts
```

禁止绑定：

```text
kernel name + operator name + model name + q-format route string
```

Scalar realization若存在，也是明确的局部实现，不是兜底。目标要求 RVV/IME 而局部结构没有
合法实现时，lowering 必须失败。

当前 `intrinsic-c` backend在module入口整体要求RVV；其中的scalar control/memory仍生成普通
C，显式scalar `floor`/`log`/`exp`等math primitive使用对应target spelling；仓库尚无独立
scalar-only target backend。缺少RVV时直接unsupported，不会把整个kernel切换到scalar
fallback。

## 算法与物理自由度

作者和 Kernel IR 拥有：

- worker-local ABI；
- scalar loop、outer traversal 与 cache blocking；
- staging、recomputation 与 persistent packed storage；
- external/persistent/workspace ownership、shape与lifetime；
- VLA logical domain；
- pointer/index/mask/effect；
- structured primitive、axes、state 与 numerical policy；
- source meta-parameter 和显式算法 variant。

Target lowering 拥有：

- dynamic `vl` placement；
- LMUL、register repeat、microtile 与 K unroll；
- instruction/fragment family；
- primitive-private packing/temporary 与 software pipeline；
- local pure producer/consumer fusion；
- intrinsic/asm spelling。

Target lowering不得改变source-visible iteration/effect semantics、algorithmic loop boundary、
staging、ABI、logical predicate、state algebra或observable numerical mode。它只能在显式
`W.vla`、dot/matmul或其他local primitive授权的domain内部执行strip-mining、K-unroll、
software pipeline与primitive-local fusion，不能交换或替换作者的ordinary scalar traversal。

Target不得分配或隐藏source-visible workspace。External、persistent与workspace都是普通entry
pointer并由caller分配；target只消费其pointer/storage facts。只有local primitive内部不可观察的
temporary可由physical plan创建，且不得进入canonical IR、generated header或public ABI。

这里的strip-mining只适用于source已经显式授权的VLA logical axis，或dot/matmul-local
realization中的局部轴；普通canonical `for` / `while`不得因此被改写成新的VLA axis。

## 新语义与新硬件

若新硬件只是更快实现现有语义，例如用矩阵fragment实现普通 `W.matmul`，只增加
target-local realization。

若硬件引入可观察的新语义，例如 block-scaled accumulation、特有 codebook、saturation、
rounding或 lane permutation，则增加一个局部 canonical primitive。禁止增加
`softmax_kernel`、`q4_K_gemm_kernel` 等整算子 op。

若扩展要求改变outer traversal、cache blocking、persistent packing、多阶段staging或跨
primitive state，作者或上游必须提供完整source variant。Target lowering不自动发明算法
variant。当前 `weft_ext.affine_i4_i8_contract`、`grouped_affine_i4_i8_dot`、
`sign_bit_i8_dot` 与 `e2m1_e8m0_i8_dot` 都只定义局部observable quantized relation；
它们周围的packing、pointer和loop仍属于Kernel IR。

## 组合判据

同一kernel中必须自然允许多个VLA、reduce、summary、irregular relation、dot/matmul与扩展
fragment同时存在。增加能力时扩展局部primitive lowering，而不是增加互斥的`KernelKind`、
format route或whole-kernel emitter。一个local envelope可以跨多个相邻pure producer/
consumer，甚至联合安排相邻primitive，但选择必须从明确canonical anchor出发，不能把整个
loop nest的形状当作operator identity。

同样的规则覆盖selection中的coordinate summary、ordered recurrence中的VLA memory、indexed
base与unit-stride region、scalar coordinate/window与VLA channel，以及source-owned grouping
中的local dot/matmul。它们是已有实体性质的新组合关系，不构成Top-K、SSM、MoE或vision
kernel family。

## 正式 target 能力边界

Public `intrinsic-c` backend绑定一个RVV target profile，并可选择性绑定`spacemit-ime1`矩阵扩展。
它只接受同一份canonical core-local program：scalar ordered control以普通C realization嵌在RVV
module中；VLA、memory、state、dot/matmul、ordering/decode/quant与extension primitive分别按自身
facts查询local capability。某个profile没有合法realization时，对该local structure明确报
unsupported，不能切换到另一种kernel模型、scalar-only backend或whole-kernel emitter。

同一canonical kernel换到SG2044/VLEN128、K1/VLEN256或K1+IME时，改变的只能是physical plan、
intrinsic/asm leaf与artifact target flags；ordered traversal、block/state/storage lifetime、entry ABI
与local primitive semantics保持不变。实现阶段的具体已覆盖case、窄closure与性能数字只记录在
当轮`report/`，不进入本规范。
