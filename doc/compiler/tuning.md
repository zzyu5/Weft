# 构建期 Tuning

## Tuning 不是 compiler stage

Tuning 是外部构建循环：用不同 source meta binding 或 backend config 重复调用同一条编译
主链，生成 object，真机测量，再保留最快合法 artifact。

```text
DSL meta / backend config candidate
  -> Kernel IR
  -> target lowering
  -> intrinsic C / asm
  -> system compiler
  -> object
  -> benchmark
  -> retain fastest artifact
```

Tuning 不形成新 IR、不向 target lowering 插入第二份 legality，也不成为运行期依赖。

## 三类变量

### Source structural knobs

例如 `BM/BN/BK`、显式 prefetch distance、staging depth 和算法 variant。它们的存在与使用
位置属于 source，候选值由 build specification 绑定。

### Backend physical config

例如 LMUL、register microtile、unroll、fragment、local packing 与 instruction family。Target
lowering定义合法值和资源关系；build loop只枚举并实测，不改变 Kernel IR 算法。

### Compiler-derived mechanics

每次 strip 的 actual `vl`、physical tail、唯一决定的 mask realization、pointer strength
reduction 等由 target lowering直接推导，不是 tuning knob。

## 允许的 specialization

构建期可以对 target、exact/range shape、alignment、stride class、persistent data layout、fixed
VLEN 和 extension capability 生成多个普通 AOT object。Runtime dispatcher 只在这些现成
artifact 中选择。

Tuning 禁止：

- 创造 source 中不存在的 loop、primitive 或 persistent layout；
- 让 architectural-illegal config 合法；
- 改变 numerical policy；
- 将 benchmark winner 物化成新的 compiler IR authority；
- 为失败候选启用 scalar、legacy 或 GGML fallback。
