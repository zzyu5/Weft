# Core-local blocked DSL 结构重构

## 结论

本轮把 Weft 的 source model 收敛为一份由单个 worker/hart 持续执行的
Core-local blocked program，并把这个定义落实到了 Python surface、canonical type/op、verifier
和三条真实 RISC-V artifact 路径中。

本轮没有增加 kernel、没有扩展调优空间、没有增加 Physical Plan 字段，也没有保留
`W.block_axis` 兼容入口。旧 catalog 没有为了维持表面通过率被机械批量迁移；只有三份能够分别
检验 GEMM、普通 block SSA composition、workspace/persistent storage 与 RVV/IME local
realization 的代表程序进入新语言。

## 1. 冻结后的作者程序

一个 Weft entry 是当前 worker 从入口到返回持续执行的普通 C ABI kernel。作者显式拥有：

- scalar `for / while / if` 的 traversal、顺序、carry 与 effect；
- M/N/K 等 outer loop 与 BM/BN/BK 的使用位置；
- logical block domain、accumulator、state 与它们的 lexical/control lifetime；
- pointer/index、predicate、blocking、algorithmic staging、workspace 与 persistent layout；
- dot/matmul、reduce/scan/summary、quant/extension 等局部可观察语义。

Compiler 只在显式授权点内部决定：

- `W.vla` 的 strip、`vl`、LMUL 与 memory form；
- `W.dot/W.matmul` 的 register microtile、multiple accumulators、K-unroll、局部 packing 与
  RVV/IME realization；
- local state/memory/extension primitive 的寄存器组织与 intrinsic/asm spelling。

普通 scalar loop 不会被改写成 VLA，普通 multiply/add 不会被猜成 dot/matmul；target 也不能
替作者创建 outer loop、staging、persistent repack 或另一种算法。

## 2. Source surface 的实际重构

### 2.1 `W.block` 取代 `W.block_axis`

每次：

```python
axis = W.block(extent, offset=0)
```

同时完成两件事：

1. 创建一个 source-owned logical axis identity；
2. 产生该轴上的 rank-one index block。

因此三个长度相等的 `m = W.block(16)`、`n = W.block(16)`、`k = W.block(16)` 仍然是三条
不同轴。Matmul 的 reduction relation 来自 lhs/rhs 显式共享同一个 `k`，不再来自“两个维度
碰巧都是 16”。

`W.block_axis` 已从 public Python surface 和 canonical op 中删除，没有 alias、compatibility
layer 或 legacy parse path。

### 2.2 Block constructor 消费真实 axis

新的 constructor 写法是：

```python
m = W.block(BM)
n = W.block(BN)
acc = W.zeros((m, n), dtype=W.f32)
bias = W.full((n,), W.f32(1.0))
```

`W.full/W.zeros` 不再接受裸整数 shape。Accumulator 的 domain identity 在创建时就已闭合，
frontend、verifier 和 target 都不能用另一个同长 axis 替代它。

### 2.3 Pointwise 与 singleton broadcast

`value[:, None]` 和 `value[None, :]` 只插入显式 singleton broadcast axis。Pointwise join 只在
以下情况合法：

- 两侧是同一个 source axis；
- 一侧是显式 singleton axis；
- 一侧是 scalar。

这关闭了旧模型中“相同 shape 被当作相同 domain”的歧义。当前没有加入任意 reshape、
transpose 或 broadcast surface。

### 2.4 Block state 使用普通 control carry

Python 变量在 `for/while/if` 中的更新继续机械 lowering 为 region argument、yield 和 result。
Block accumulator 不包装成新的 state object，也不进入 opaque tuple。

`W.tuple` 已收窄为 scalar state tuple。Block/state 的跨循环生命周期由普通 SSA control carry
直接表达。

### 2.5 删除无权威的旧 surface

- 删除 `W.block_axis`；
- 删除 frontend 中未进入正式 source model 的备用 region helper；
- 删除 `Intrinsic.canonical_name` 这一未被 canonical lowering 消费的第二命名字段；
- helper 仍只按调用点 inline，不形成 helper IR 或第二份 typed authority。

## 3. Canonical Kernel IR 的变化

Block/region type 现在同时保存 shape 和 axis identity：

```text
!weft_kernel.block<[D0, D1, ...], [a0, a1, ...], T>
!weft_kernel.region<[-1, D0, ...], [-1, a0, ...], T>
```

其中：

- positive axis ID 必须由 kernel 内唯一的 `weft_kernel.block_index` 定义；
- `0` 只表示显式 singleton broadcast；
- region 首轴的 `-1` 只表示当前 active VLA axis；
- dynamic extent 的 shape 可以是 `-1`，但 identity 仍不会丢失。

新的 canonical op 是：

```text
weft_kernel.block_index(extent, offset) {axis = ...}
```

`weft_kernel.full` 直接消费一组 `block_index` result，不再保存一份独立 shape attribute 和一组
可与之失配的裸 extent operands。

Verifier 现在统一检查：

- block/region rank、shape 与 axis identity 一致；
- axis ID 在一个 kernel 内唯一，所有 positive reference 都有定义；
- pointwise、select、pointer、predicate、load/store footprint 的 domain join；
- cast/bitcast/narrow、reduce/scan/argmax/lookup/decode 对 axis identity 的保持；
- dot/matmul 的共享 reduction identity、result domain 与 init domain；
- loop/branch carried value 的 dynamic extent identity；
- tuple field 只能是 scalar，block state必须使用普通control carry。

这些检查都位于 source/canonical semantic boundary；没有新增 Physical IR 或 Plan 字段。

## 4. GEMM 如何体现模型

迁移后的 F16 blocked GEMM 明确写出：

```python
for m0 in W.range(m_begin, m_end, BM):
    for n0 in W.range(0, n, BN):
        m = W.block(BM)
        n = W.block(BN)
        acc = W.zeros((m, n), dtype=W.f32)

        for k0 in W.range(0, k, BK):
            k_block = W.block(BK)
            a_block = W.load(... m ..., ... k_block ...)
            b_block = W.load(... k_block ..., ... n ...)
            acc = W.matmul(a_block, b_block, init=acc, ...)

        acc = acc + W.f32(0.0)
        W.store(... m ..., ... n ..., acc)
```

从 source 本身可以读出：

- 当前 worker 遍历 M/N/K；
- BM/BN/BK 属于哪些 loop；
- accumulator 由 worker 持有并跨 K-loop carry；
- A/B 分别复用 M/N 轴并共享同一个 K 轴；
- source 没有 staging 或 persistent packing；
- matmul 只授权当前 local block product；
- matmul result 在 loop 后仍可 pointwise 后 store。

Target 只选择这一 local product 的 RVV microkernel；它没有接管 outer M/N/K traversal。

## 5. 三份代表程序证明的组合

| Source | 本轮检查的 source property | 实际 target artifact |
| --- | --- | --- |
| `blocked_gemm_f32.py` | dot result 同时进入 add 与 multiply 两个普通 consumer，再经 maximum/store | SG2044 RVV intrinsic C |
| `blocked_gemm.py` | `[M,N]` block accumulator 跨 K-loop carry，matmul result 再 pointwise/store | SG2044 RVV intrinsic C |
| `q4_k_projection.py` | scalar summary、worker workspace、persistent packed input、block accumulator 与 local quant primitive 组合 | K1 RVV intrinsic C 与 K1 IME local asm leaf |

这些程序没有对应的 direct-store、one-use、kernel-name 或 exact-op-count 入口。

## 6. 迁移暴露的真实缺口

### 已关闭的语言/IR缺口

1. **Block 只有 shape、没有 axis identity。** 同长 M/N/K 会被误当成同一 domain。通过
   `W.block`、axis-bearing canonical type 和 verifier 关闭。
2. **Accumulator constructor 只有整数 shape。** Source 无法声明它由哪些 axis 持有。通过
   constructor 直接消费 axis value 关闭。
3. **Tuple 与 ordinary block carry 权责重叠。** Block 可以被包装进另一种 state 容器。通过
   tuple scalar-only、block direct carry 关闭。
4. **Structured result 的语言地位被 terminal closure 暗中限制。** 代表程序现在显式包含
   multi-consumer 与 pointwise consumer，frontend/canonical 不再把 one-use/direct-store 当作合法性。

### 已关闭的 target ownership 缺口

F16 GEMM 首次生成 artifact 时，Realizer 报告：

```text
block producer closure crosses a region boundary
```

原因不是 source 非法，而是“丢弃 private block producer tree”的分析把 K-loop 外定义的 M/N axis
也当成 consumer-private closure。现在该分析只收集当前 region 内的 private definition；captured
block producer 保持普通 SSA handoff。真正需要局部 producer closure 的 dot/store/reduce 路径仍
维持各自限制，没有把跨 region 捕获泛化成新的 fast path。

### 当前迁移边界

`examples/kernels/` 中仍有 15 个 source 文件、54 处旧 `W.block_axis`。它们没有 compatibility
fallback，因此在正式迁移前会明确失败。这是本轮“只迁移少量代表 kernel”的刻意边界，不是
另一条可用 DSL。

本轮真实 artifact 只证明了：

- `for` 中的 block accumulator carry；
- dot multi-use/pointwise composition；
- matmul pointwise composition；
- workspace/persistent storage进入RVV与IME local primitive。

其他 source 中的 `if/while` block carry、更多 structured-result consumer 和更复杂 workspace
lifetime 尚未由本轮真实 target artifact 覆盖，不能根据 canonical 可表达性宣称已经全部后端闭合。

## 7. 真实 artifact 与执行结果

本轮使用现有手工 target runner，完整经过：

```text
Python DSL
→ canonical Kernel IR
→ RISC-V intrinsic C / local asm + header
→ target C compiler
→ SSH target execution
```

### SG2044 / RVV128 / F32 dot composition

```sh
./examples/run/weft.sh sg2044-rvv128 blocked_gemm_f32 decode 1
```

```text
kernel=f32_dense_projection
M=1, N=4096, K=4096
max_abs=0, max_rel=0
14.822386 ms, 2.263767 GOP/s
```

### SG2044 / RVV128 / F16 matmul carry

```sh
./examples/run/weft.sh sg2044-rvv128 blocked_gemm decode 1
```

```text
max_abs=0, max_rel=0
13.114619 ms, 2.558552 GOP/s
```

### K1 / RVV256 / persistent Q4_K projection

```sh
./examples/run/weft.sh k1-rvv256 q4_k_projection_ime 1
```

```text
persistent=q4_k_n16_k32_304b
activation mismatch=0
scale error=0
output max_abs=0, max_rel=0
42.914816 ms, 0.781885 GOP/s
```

### K1 / IME256 / 同一 source primitive

```sh
./examples/run/weft.sh k1-ime256 q4_k_projection_ime 1
```

```text
activation mismatch=0
scale error=0
output max_abs=3.33786011e-06
output max_rel=3.28104943e-06
3.967554 ms, 8.457209 GOP/s
```

这些单次数字只证明新模型能够生成并执行真实 artifact；本轮不是性能调优轮，因此没有把它们
写入 `weft-kernel-performance.csv`，也没有建立阈值、同步或校验逻辑。

## 8. 本轮没有引入的 authority

本轮没有新增：

- Physical IR / Selected IR；
- provider/capability registry；
- kernel-name、q-format、exact-op-count 或 whole-region route；
- GGML/materials runtime调用；
- legacy/scalar fallback；
- normalization pass；
- 任何测试、fixture、case matrix 或兼容层。

当前唯一语言定义是 `WEFT_CORE_LOCAL_BLOCKED_PROGRAM_MODEL.md`；其余 `doc/` 文件只投影对应
模块，canonical IR 保存唯一持久语义，target physical facts 仍只存在于一次 lowering 内。
