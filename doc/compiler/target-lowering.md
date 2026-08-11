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

当前 `RISCVLoweringOptions` 的public输入只有target profile与source meta bindings。LMUL、
microtile、unroll、fragment等物理参数由lowering内部选择；以后若暴露build-time backend
config，它必须是显式、短生命周期的lowering option，不能进入canonical IR。

## 内部 realization families

Target lowering 可以为同一 semantic primitive 实现多种局部 realization family，例如：

```text
VLA pointwise       -> RVV f32m2 / f32m4
reduce              -> scalar ordered / RVV tree / widening RVV
contract            -> RVV FMA microtile / RVV dot / IME fragment
lookup/decode       -> scalar / indexed RVV / table gather
memory permutation  -> RVV segment / indexed / extension permute
```

每个 family 的内部定义可以包含 capability predicate、物理参数空间、legality、resource
equation、local fusion envelope 和 emission hook。当前实现可以先只有一个固定合法配置；
当它扩展成多个candidate时，candidate也只存在于本次lowering，不拥有独立IR schema、
verifier、pipeline front door或长期provider registry。

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
C，但仓库尚无独立scalar-only target backend。缺少RVV时直接unsupported，不会把整个kernel
切换到scalar fallback。

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

## 新语义与新硬件

若新硬件只是更快实现现有语义，例如用矩阵fragment实现普通 `W.contract`，只增加
target-local realization。

若硬件引入可观察的新语义，例如 block-scaled accumulation、特有 codebook、saturation、
rounding或 lane permutation，则增加一个局部 canonical primitive。禁止增加
`softmax_kernel`、`q4_K_gemm_kernel` 等整算子 op。

若扩展要求改变outer traversal、cache blocking、persistent packing、多阶段staging或跨
primitive state，作者或上游必须提供完整source variant。Target lowering不自动发明算法
variant。当前 `weft_ext.affine_i4_i8_contract` 与 `grouped_affine_i4_i8_dot` 都只定义局部
observable quantized relation；它们周围的packing、pointer和loop仍属于Kernel IR。

## 组合判据

同一kernel中必须自然允许多个VLA、reduce、summary、irregular relation、contract与扩展
fragment同时存在。增加能力时扩展局部primitive lowering，而不是增加互斥的`KernelKind`、
format route或whole-kernel emitter。一个local envelope可以跨多个相邻pure producer/
consumer，甚至联合安排相邻primitive，但选择必须从明确canonical anchor出发，不能把整个
loop nest的形状当作operator identity。
