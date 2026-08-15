# 构建期选择

Weft 不在 Kernel IR 中保存 measurement 或 winner。作者可以通过 `W.constexpr` 暴露算法级
block size；target lowering 可以通过 backend config 暴露 LMUL、microtile、unroll、prefetch、
packing 与 fragment 候选。

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
