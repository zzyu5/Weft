# 构建期 Tuning

## 18. Tuning 规范

### 18.1 三类变量

#### Source structural knobs

例如：

```text
BM / BN / BK
作者显式声明的 prefetch distance
作者显式声明的 staging depth
显式 algorithm variant
```

其存在与语义位置属于 source，候选值由 build specification 提供。

#### Provider physical knobs

例如：

```text
LMUL
register microtile mr×nr
register repeat
unroll
fragment strategy
local packing strategy
```

由 provider 声明参数空间与约束。

#### Compiler-derived mechanics

例如：

```text
每次 strip 的 actual vl
physical tail
vsetvl placement
已唯一决定的 mask realization
pointer strength reduction
```

这些不是 tuner knob。

### 18.2 AOT build-time search

Tuning 流程发生在构建期：

```text
候选绑定
  → selected execution
  → emit source/object
  → compile
  → benchmark
  → 固化最快合法变体
```

运行期不带 compiler，不即时生成代码。

### 18.3 Shape specialization

Tuning key 是：

```text
(target profile, specialization predicate)
```

Specialization predicate 可以是：

- exact shape；
- shape range；
- alignment / stride class；
- quant format；
- model-specific constant；
- generic fallback。

### 18.4 单变体与多变体

Weft artifact 必须支持两种 AOT 形态：

1. **generic entry**：runtime shape 动态，单个 VLEN-agnostic kernel；
2. **multiversion entry**：构建期生成多个 shape / target specialization，并生成轻量 runtime dispatch。

Dispatch 只在已生成的 AOT variants 中选择，不进行 JIT。

### 18.5 Build specification

Python source 不需要包含完整搜索集合。外部 build specification 提供：

```text
entry
source meta domains
target profile
specialization predicates
measurement harness
allowed providers
optional provider constraints
artifact options
```

概念示例：

```toml
[entry.gemm_worker]
meta.BM = [16, 32, 64]
meta.BN = [16, 32, 64]
meta.BK = [16, 32]

[[entry.gemm_worker.specialization]]
when = "N % 64 == 0"

[[entry.gemm_worker.specialization]]
when = "true" # fallback
```

具体配置格式可以变化，职责边界不得变化。

---
