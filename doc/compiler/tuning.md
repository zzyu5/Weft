# 构建期选择

Weft 不在 Kernel IR 中保存 measurement 或 winner。作者可以通过 `W.constexpr` 暴露算法级
block size；当前 target lowering 的 backend config 只暴露已经具有多个可生成实现的 VLA/dot/
matmul LMUL、F16 lane axis、row/column microtile、K-unroll、register-load buffer count、narrow LMUL
与sort radix候选。Extension fragment由target profile在同一局部primitive内选择，不伪装成尚不
存在的可调维度。

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

仓库中的实际入口是`examples/run/tune.py`。它不保存kernel/格式最优表；调用者显式给出backend
维度和runtime metric，它对参数做笛卡尔实例化，并逐项调用同一个`weft.sh`真实路径。例如：

```text
python3 examples/run/tune.py sg2044-rvv128 blocked_gemm \
  --metric weft_ms \
  --dimension f16-lane-axis=column,reduction \
  --dimension f16-k-unroll=1,2,4 \
  --dimension f16-load-buffer-count=1,2 \
  -- decode 5
```

`tune.py`只组合并转发调用者给出的dimension；option是否属于正式backend config、组合是否合法仍由
`weft-compile`和target planning唯一判断。只有compiler判定合法、system compiler成功且runtime
数值检查成功的实例才产生metric并参与选择。
`--winner-config PATH`可把winner backend config写入当前build使用的文件；该文件不是Kernel IR、
不进入仓库，也不改变compiler默认值。
