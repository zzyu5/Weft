# 构建期 Tuning

## Tuning 不是 compiler stage

Tuning若被使用，就是可选的外部构建循环：用不同source meta binding或显式backend config
重复调用同一条编译主链，生成object，真机测量，再保留最快合法artifact。

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

例如 `BM/BN/BK`、显式prefetch distance、staging depth和算法variant。它们的存在与使用
位置属于source，候选值由build specification绑定；当前CLI通过 `--meta=NAME=INTEGER`
materialize source meta。

### Backend physical config

例如LMUL、register microtile、unroll、fragment、local packing与instruction family。Target
lowering定义合法值和资源关系；没有external knob时可以选择唯一合法配置。若将这些候选
暴露给build loop，必须是显式backend config，不能伪装成source meta或新的IR。

### Compiler-derived mechanics

每次strip的actual `vl`、physical tail、内部mask realization与pointer strength reduction等
由target lowering直接推导，不是tuning knob。

## 允许的 specialization

构建期可以对target、exact/range shape、alignment、stride class、source-declared persistent
data layout、fixed VLEN和extension capability生成多个普通AOT object。Runtime dispatcher
只在这些现成artifact中选择；shape specialization不能发明source中不存在的loop、state或
primitive identity。

Tuning 禁止：

- 创造 source 中不存在的 loop、primitive 或 persistent layout；
- 让 architectural-illegal config 合法；
- 改变 numerical policy；
- 将 benchmark winner 物化成新的 compiler IR authority；
- 为失败候选启用default scalar、legacy或GGML fallback。
