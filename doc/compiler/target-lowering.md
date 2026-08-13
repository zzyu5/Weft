# RISC-V Target Lowering

## Target profile

每次 lowering 必须绑定一个 `RISCVTargetProfile`。当前 public profile由 `--march`、`--abi`、
`--vlen-bits` 与 `--matrix-extension` 构造，保存：

```text
target triple / march / ABI / XLEN / endianness
RVV availability
fixed VLEN bits，或 0 表示 runtime-unknown
architectural vector register count
matrix/vendor extension identity
```

`march` 是ISA extension事实的当前载体。随着realization family扩展，profile可以解析出element
width、LMUL、mask/tail、fragment、rounding与memory instruction等派生capability；这些仍是
target facts，不形成新IR。Cache、throughput、latency与preferred unroll只能作为hint排序
合法物理配置，不能让非法实现变合法。

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

当前实现已经存在的transient physical decision包括：

- 每个VLA predicate、load/store、reduce/scan/summary与narrow各自的lane relation、memory
  mode、activity、state realization和LMUL；F16/F32 cast与F16 fused multiply-add也拥有逐op
  vector shape decision；位于nested scalar control中的access仍逐实体决策；
- local F32 dot/matmul的operand shape、row microtile、pointer/stride relation、LMUL和resource
  footprint；含dot/matmul的
  VLA region直接消费该LMUL来决定strip width、mask ratio与index LMUL，不保留第二份选择；
- block store/reduce的strip family、byte-vector shape、active length和lane-index需求；closure内
  每个value、decode code/result与materialized store的vector shape只存在于entity plan，typed
  payload只保留local realization和source provenance；
- symmetric i4×i8 primitive的typed operand closure、288-byte local block relation与IME1
  N16×K32 realization；
- sign-bit×i8 primitive的block bases、scale/init operands与VLEN128 widening-sign-sum
  realization；
- E2M1/E8M0×i8 primitive的packed-code/activation bases、exponent/scale/init operands与
  VLEN128 table-dot realization。

这些decision由一个短生命周期的physical planning core联合形成。每个显式VLA region或
local primitive都拥有同样的typed entity plan，并在其中建立value/use
physical shape、memory handoff、primitive realization、loop-local schedule和resource budget，
再把selected fields投影到各operation owner。Load、cast、state、dot和store之间的SEW、LMUL、
mask/index relation不能各选一遍；handoff必须明确为share、convert、rematerialize、reload、
shuffle或primitive-local pack。候选的peak resource同时计入live value、memory/index、predicate、
state、primitive和pipeline footprint。当前没有显式pipeline语义的实体会明确选择单stage；
没有显式prefetch primitive时选择零prefetch-distance，存在该primitive时只记录它授权的局部
handoff，不能被解释成编译器发明了staging。这个plan只存在于一次lowering调用中，不是新IR。

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
- VLA logical domain；
- pointer/index/mask/effect；
- structured primitive、axes、state 与 numerical policy；
- source meta-parameter 和显式算法 variant。

Target lowering 拥有：

- dynamic `vl` placement；
- LMUL、register repeat、microtile 与 K unroll；
- instruction/fragment family；
- primitive-local packing、scratch 与 software pipeline；
- local pure producer/consumer fusion；
- intrinsic/asm spelling。

Target lowering不得改变source-visible iteration/effect semantics、algorithmic loop boundary、
staging、ABI、logical predicate、state algebra或observable numerical mode。在这些语义保持
不变且legality成立时，可以改变物理loop形态，执行strip-mining、unroll、interchange、
software pipeline与primitive-local fusion。

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

## 当前实现边界

逐实体VLA memory/predicate/state/narrow、F16/F32 cast、F16 arithmetic、codebook decode、
sign-bit/E2M1 local dot与symmetric IME fragment已经按上述模型工作。此前F16 fill、F32→F16、
F16 weighted update和F16→F32 normalize四条whole-region realization已经由generic VLA
access/cast/binary decision替代。Block store/reduce的e8mf4/e8m1 strip选择及closure内每个结果
shape也已移出emitter，成为entity-owned physical decision；block emitter不再保存或重建第二份
vector shape。

Local F32 dot/matmul根据block/VLA axis、typed operand、pointer/access与predicate projection形成
resource model，再结合target寄存器数、显式config与resource headroom从局部candidate中选择LMUL；
当前local-row候选包含随row extent变化的LMUL4/2/1，
VLA free-axis候选也共用同一selector。LMUL与K-unroll只物化在entity plan中，typed primitive
payload不保存第二份选择；当前合法K-unroll winner仍为1。multi-axis
microtile、pointer schedule、prefetch、reuse和pipeline候选仍窄。这些固定内部candidate不是
开放注册表或通用搜索承诺。

Dot的physical owner是`weft_kernel.dot`本身；当前store只是已知的downstream handoff，不再拥有
dot decision；store到dot的索引只把output handoff路由到已经选定的dot decision。VLA内dot仍与
enclosing VLA entity联合选择LMUL/resource，但canonical授权anchor和物理decision identity保持
在dot上。Core verifier、extension verifier与target复用同一组logical value/validity queries；
core dot/matmul按自身语义处理masked operand，而未定义validity语义的extension primitive必须要求
作者先显式`fill`。

当前仍有一个精确的局部VLA fusion envelope：F16 widening dot将显式load/cast/multiply/reduce
闭包实现为widening MAC。Online-softmax summary只拥有自身typed state，后续normalize VLA由
其中的memory/pointwise实体独立lower。F16 GEMM nested loop和affine Q4_K IME N/K closure仍
要求较精确的局部loop关系。
它们都从typed source relation产生decision，不读取kernel名，也不接管public ABI或外围
traversal；但尚未获得更宽的等价source接受范围。新增能力应扩展局部decision的合法语义输入
与physical candidate，不能重新增加whole-kernel try-emitter。
