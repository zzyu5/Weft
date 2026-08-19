# 构建期选择

Tuning 不是语言 stage，也不产生持久 IR。DSL 中没有 `W.tune`、target 名、LMUL、microtile 或
candidate list。

## 允许的构建期选择合同

```text
target lowering constructs semantically legal structures
→ resource model filters illegal combinations
→ parameters instantiate remaining structures
→ system compiler builds candidates
→ target execution measures them
→ build selects one artifact/config
```

实测不能让非法 candidate 合法，也不能定义一个新的 whole-kernel implementation。这段顺序是
构建期选择必须遵守的边界，不表示仓库当前已经集成自动测量器。

## 当前实现

当前 `weft-compile` 对一组 target facts、meta 与 backend config 执行一次 lowering，runner 可用
不同 config 重复调用并在真机记录结果；仓库内没有自动枚举、测量并固化 winner 的长期 tuning
pipeline。没有真实生成器支持的维度不能被文档或 config 宣称为已可调。

## 可选择内容

只有真实存在至少两个可生成实现的维度才进入选择，例如 lane/LMUL、register microtile、
accumulator count、unroll、local buffering、pipeline depth、prefetch distance 或 RVV/IME local
realization。没有第二个实现的字段应删除，而不是伪装成 tuning space。

`constexpr` 表示作者允许构建期实例化的算法参数。若改变 traversal、blocking、staging、
persistent layout 或算法 variant，候选必须是不同的显式 DSL kernel/meta instance，而不是 target
暗中修改程序。

## Target 与 shape

Winner 只对当前 target profile、shape、dtype、storage organization 与计时范围有意义。SG2044
与 K1 的数字不能直接横比；同一 DSL kernel 可以因 VLEN、register budget 或 extension facts
选择不同的合法 realization。

性能 CSV 只是实测数字记录，不附带阈值、同步或验证逻辑，也不成为 compiler authority。
