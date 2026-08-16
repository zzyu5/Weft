# 构建期选择

Weft 不在 Kernel IR 中保存 measurement 或 winner。作者可以通过 `W.constexpr` 暴露算法级
block size；当前 target lowering 的 backend config 只暴露已经具有多个可生成实现的 VLA/dot/
matmul LMUL、reduction state placement、row/column microtile、K-unroll、register-load buffer count、narrow LMUL 与
sort radix 候选。Extension fragment 由 target profile 在同一局部 primitive 内选择，不伪装成
尚不存在的可调维度。

构建系统可以重复执行：

```text
bind meta values / backend config
→ generate intrinsic C
→ compile object
→ run on the target
→ retain the fastest legal object
```

候选必须先由 typed facts、target profile 与 resource budget 判定合法。某次编译或运行失败只删除
该候选，不能删除整个物理维度，也不能创建 kernel-name、format-name 或 whole-kernel route。

选择结果属于当前 build，不反写 DSL kernel 或 Kernel IR，也不改变作者写下的 traversal、blocking、
staging、persistent layout 或算法 variant。
