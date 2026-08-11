# RISC-V Target Lowering

## Target profile

每次 lowering 必须绑定一个 RISC-V target profile。它保存 architectural facts：

```text
XLEN / ABI / endianness
ISA extension set
supported scalar and vector element widths
available LMUL set
architectural vector register count
VLEN fixed value、范围或运行时未知标记
mask/tail capability
matrix/quant/vendor extension identity
fragment shapes and accumulator classes
rounding/saturation capability
memory instruction classes
```

可以附加 cache、instruction throughput、latency、preferred unroll 等 microarchitecture hint。
Hint 只能排序合法物理配置，不能让非法实现变合法。

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
equation、local fusion envelope 和 emission hook。它们共同属于一次 target lowering，不拥有
独立 IR schema、verifier、pipeline front door 或长期 provider registry。

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

Target lowering不得改变 source outer loop、staging、ABI、logical predicate、state algebra 或
observable numerical mode。

## 新语义与新硬件

若新硬件只是更快实现现有语义，例如用 IME fragment 实现 `W.contract`，只增加 target-local
realization。

若硬件引入可观察的新语义，例如 block-scaled accumulation、特有 codebook、saturation、
rounding或 lane permutation，则增加一个局部 canonical primitive。禁止增加
`softmax_kernel`、`q4_K_gemm_kernel` 等整算子 op。

若扩展要求改变 outer traversal、cache blocking、persistent packing、多阶段 staging 或跨
primitive state，作者或上游必须提供完整 source variant。Target lowering不自动发明算法
variant。

## 组合判据

同一 kernel 中必须自然允许多个 VLA、reduce、summary、irregular relation、contract 与扩展
fragment同时存在。增加一种能力时扩展局部 primitive lowering，而不是增加互斥的
`KernelKind`、format route 或 whole-kernel emitter。
