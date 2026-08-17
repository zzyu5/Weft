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

递归收集某个 operand 的 producer 只用于投影普通 use-def、reload source 与 address dependency。
高性能能力不能以完整 producer 集合、精确 operation 数量、one-use、direct-store、源码邻接或
外围 loop 形状为入口。增加普通 consumer 只改变 lifetime、handoff 与资源，不能让原 primitive
失去合法实现。

每个显式 op 先贡献自己的语义约束：pointwise 保持轴映射，load/store 给出 pointer relation，
cast 保持逻辑元素并改变 SEW，state 给出 order/carry，dot/matmul 给出 free/reduction/broadcast
关系，quant/lookup 给出普通 SSA 无法无歧义恢复的局部数值关系。共享 mapping 再把逻辑轴分解为：

```text
sequential inner work
+ scalable RVV lanes
+ register repetitions / accumulators
+ reduction unroll
+ optional extension fragment coordinates
```

`RVV strip`、register microkernel 与 IME fragment 只是 mapping 的最终投影，不是进入 lowering 的
结构类别。VLEN、target 型号和量化格式也不能成为实现身份。

## 瞬态物理决定

Lowering 联合决定：

- VLA strip schedule、SEW/LMUL、mask 与 state placement；
- unit/strided/indexed/segment memory；
- value/register shape 与 share/reload/rematerialize/local-pack handoff；
- F32 dot 与 F16 matmul 的 lane axis、register microtile、multiple accumulators、K-unroll、operand
  window 和 load/compute schedule；
- column-lane 与 reduction-lane F16 matmul 都消费相同的 pipeline actions；
- quant/codebook 的 packed-axis decomposition、decode、gather、widen、accumulator 与 operand window；
- RVV 或 IME fragment 与局部 asm leaf；
- primitive-private temporary 的大小、alignment 与 resource budget。

`RISCVKernelFacts` 先按 pointer、index SSA、block scope、effect 顺序和 element bytes 产生 indexed
address group 与 interleaved memory group。Target planning只对这些事实判断 indexed/segment
legality；selected segment record保存access kind、field count、element SEW、coordinate scale与
最终vector shape。Emitter不再扫描相邻access重建分组。

`RISCVReuseAnalysis` 对 dot/matmul 的 operand 统一推导 reduction advancement、memory mode、
accumulator-dependent address/predicate、consumer count 与 control crossing。Physical planning据此
生成operand window和loop-local pipeline；dense、VLA dot与matmul不各自维护依赖规则。

VLA state直接保存同一个`SelectedVLAStatePhysical`，其中包含carry representation、strip update、
finalize与whole-VLA lifetime；entity resource planning与emitter都消费这份结果，不再拆出第二份
state resource schema。

Resource budget按同一时间活跃的value、memory operand、predicate、loop-carried state、primitive
temporary、pipeline buffer与extension fragment组成。每个lifetime snapshot绑定真实operation；
外层跨`for/while/if`存活的值会与nested temporary联合计算，primitive临时资源只叠加在它实际
执行的位置。多个vector carry会累加占用，互不同时存活的工作才取峰值。资源不足或target没有
等价实现时立即返回unsupported，不切换到旧emitter、标量实现或外部函数。

每项决定只有一个 producer。Quant与codebook的typed rule只描述数值差异、value shapes、decode
和最小硬件operation；轴mapping、microkernel schedule与resource selection由一份共享candidate
builder完成。生成阶段不能再从target VLEN、格式名或table width推一次helper版本。

每个consumer对每个value只能有一条handoff记录。若handoff kind、source shape或result shape
发生冲突，physical plan直接非法；emitter不从C value当前拼写反推新的转换方式。

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
