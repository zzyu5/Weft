# RISC-V Target 与 Primitive Provider

## 15. RISC-V target model

### 15.1 Target profile

Weft 编译输入必须绑定一个 RISC-V target profile。Profile 保存 architectural facts，例如：

```text
XLEN / ABI / endianness
ISA extension set
supported scalar and vector element widths
available LMUL set
architectural vector register count
VLEN fixed value、范围或运行时未知标记
mask/tail architectural能力
matrix/quant/vendor extension identity
fragment shapes 与 accumulator classes
rounding/saturation capability
memory instruction classes
```

### 15.2 Optional microarchitecture profile

可以另外提供非语义 microarchitecture hints：

```text
cache capacity / line size
preferred unroll
instruction throughput / latency hint
load-store bandwidth hint
```

这些 hint 只能用于候选排序或搜索缩减，不能让 architectural-illegal candidate 变合法。

### 15.3 Target profile 不是 cost model

Target profile 是 typed machine fact table。Provider 可以使用解析 resource constraints；构建期 tuner 使用实测处理未建模的 residual performance。

---

## 16. Primitive realization provider

### 16.1 Provider 的职责

一个 provider 为某类 semantic primitive 定义参数化物理实现族：

```text
supported semantic interface
capability predicate
physical parameter space
legality constraints
resource equations
selected record schema
lowering/emission
optional local fusion envelope
```

Provider 绑定：

```text
primitive/interface + typed operands + target facts
```

禁止绑定：

```text
kernel name + operator name + model format + route string
```

### 16.2 Core primitive interfaces

规范至少定义以下接口族：

- VLA pointwise；
- scalar / vector memory；
- reduce；
- scan；
- summary fold；
- contract；
- permute / gather / table lookup；
- widen / narrow / convert；
- decode / dequantize；
- math primitive；
- atomic / fence。

一个扩展只需要为它真正加速的接口注册 provider。

### 16.3 Baseline 与扩展组合

同一个 target 可以同时拥有：

```text
baseline scalar lowering
baseline RVV provider
Zvfh provider
matrix extension provider
vendor quant provider
vendor permute/lookup provider
```

它们是可组合的 primitive providers，不是互斥 whole-kernel backend。

普通 scalar control 直接由 scalar lowering 处理；普通 VLA pointwise/memory 通常由 RVV provider 处理；structured primitive 可以选择更专用 provider。

### 16.4 纯 realization 扩展

若扩展不改变 source-observable semantics，它可以实现现有 primitive，无需修改 Python DSL 或 canonical IR。

例如：

```text
W.contract
  → RVV FMA microkernel
  → RVV dot extension
  → IME fragment
  → future matrix extension
```

Provider 可以改变：

- LMUL；
- register microtile；
- fragment；
- packing；
- local scratch；
- instruction family；
- local unroll / pipeline。

不得改变 source axes、outer loops、state、mask、ABI 或 numerical mode。

### 16.5 新语义扩展

若扩展引入可观察的新语义，例如：

- block-scaled accumulation；
- 特有 codebook / decode；
- 特殊 saturation / rounding；
- persistent architectural state；
- 新的 lane permutation semantics；
- 不同 output relation；

则必须增加一个局部 canonical primitive，例如：

```python
W.block_scaled_contract(...)
W.table_decode(...)
W.saturating_dot(...)
```

它仍然必须是局部 primitive，不得增加 `softmax_kernel`、`q4_K_gemm_kernel` 等整算子 op。

### 16.6 算法 variant

若扩展要求改变：

- outer traversal；
- cache blocking 层次；
- persistent packed storage；
- 多阶段 staging；
- 跨 primitive 共享状态；

则作者或上游必须显式提供另一份 kernel variant。Compiler 不自动发明该 variant。

Build system 可以在作者提供的 variants 中选择，但每个 variant 都必须是独立完整的 Weft kernel。

---

## 17. Compiler、provider 与 tuner 的权限边界

### 17.1 作者/source 拥有

- worker-local ABI；
- scalar loop 与 control；
- VLA region 的逻辑范围；
- algorithm/cache blocking 的存在和位置；
- staging / recomputation skeleton；
- pointer/index/mask/effect；
- structured primitive 是否存在；
- contract outer loops 与 operand logical blocks；
- source meta-parameter；
- explicit algorithm variant；
- observable numerical policy。

### 17.2 Provider 拥有

- primitive-local physical parameter family；
- LMUL 候选；
- register tile / repeat；
- instruction / fragment family；
- local packing / scratch；
- local unroll / pipeline；
- capability 与 resource constraints；
- lowering。

### 17.3 Compiler 拥有

- 从 canonical facts 重算 shape、axis、stride 与 use relation；
- 绑定 target profile；
- 构造 provider 候选；
- 解析 legality；
- 删除寄存器、fragment、dtype、mask、memory 不合法实例；
- 推导动态 `vl` mechanics；
- 生成 selected execution；
- 进行 semantics-preserving local optimization；
- 生成 artifact。

### 17.4 Tuner 拥有

- 在 compiler 已证明合法的候选中测量；
- 为 source meta-parameter 与 provider physical knobs 选值；
- 选择作者提供的 kernel variant；
- 生成 specialization / dispatch decision。

Tuner 禁止：

- 创造 source 中不存在的 loop 或 primitive；
- 让非法 candidate 合法；
- 改变 numerical policy；
- 根据测量绕过 verifier。

---
