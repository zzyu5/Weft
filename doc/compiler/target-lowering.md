# RISC-V Target Lowering

## 输入

一次 lowering 只读取：

- canonical Kernel IR；
- `--march`、ABI、endianness、VLEN、vector register 数和扩展能力；
- 显式 backend config 与已绑定 meta values。

它不读取 kernel 名对应的实现表，也不调用 `source/`、GGML 或 `materials/`。

## 逐实体 facts

每个实体独立产生 lowering 所需事实：

```text
VLA: logical extent, coordinate, predicates, carried state
memory: pointer relation, element type, alignment, access/effect, validity
block value: shape, axis identity, dtype, definition/last use, ordinary consumers, control carry
state: reduce/scan/summary algebra and numerical policy
dot/matmul: operand domains, reduction axis, init/result, dtype
quant/extension: complete local numerical and byte relation
```

递归收集某个 operand 的 producer 只用于投影它的普通 use-def 依赖。高性能能力不能以完整
producer 集合、精确 operation 数量、one-use、direct-store、邻接关系或外围 loop 形状为入口。
一个纯 producer 是否可安全 fusion/rematerialize 可以读取其普通 uses，但增加另一个 consumer
不能让原 primitive 失去合法实现。

## 瞬态物理决定

Lowering 联合决定：

- VLA strip schedule、SEW/LMUL、mask 与 state placement；
- unit/strided/indexed/segment memory；
- value/register shape 与 share/reload/rematerialize/local-pack handoff；
- F32 dot 的 LMUL/K-unroll 与 F16 matmul 的 row microtile、input LMUL、K-unroll、load schedule；
- F16 matmul 已实现的单阶段 stream 与双阶段 batch-load/compute pipeline；
- quant decode/register organization；
- RVV 或 IME fragment 与局部 asm leaf；
- primitive-private temporary 的大小、alignment 与 resource budget。

Resource budget按同一时间活跃的value、memory operand、predicate、loop-carried state、primitive
temporary与extension fragment组成。多个vector carry会累加占用；只有互不同时存活的瞬态工作
取峰值。资源不足或target没有等价实现时立即返回unsupported，不切换到旧emitter、标量实现或
外部函数。

每项决定只有一个 producer。对 fixed-lane codebook dot，semantic lanes、vector shape与最终
32-lane/64-lane leaf在选择阶段一起确定；对grouped affine i4×i8，VLEN128、VLEN256与scalable
leaf同样是不同的selected realization。生成阶段不能再从target VLEN推一次helper版本。

## Intrinsic C 与局部 asm

生成阶段只读取已完成的物理决定，并负责：

- C ABI、typed local variable 与 intrinsic 名称；
- `vsetvl`、RVV intrinsic、mask/inactive policy 的拼写；
- extension register class、clobber 与 inline-asm constraint；
- C header metadata。

高度专门的 helper 或 asm 只能实现一个明确局部运算。它不能拥有 entry ABI、outer traversal、
blocking、staging、persistent storage 或完整 state machine。

## Target profile

SG2044 与 K1 使用同一套 Kernel IR 和 lowering。VLEN128/VLEN256、F16 widening、vector register
budget 与 IME 能力来自各自 profile，因此可以产生不同 LMUL、microtile、memory 和 fragment
选择。跨机器不直接比较速度；每台机器只与同算法、同 shape 的本机 baseline 比较。
