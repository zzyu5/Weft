# 真实运行复核

Weft 只保留一类复核：把一个 DSL kernel 生成 intrinsic C/C header，在目标机编译成 executable，
真实执行并对照数值。

```bash
./examples/run/weft.sh sg2044-rvv128 add_bias
./examples/run/weft.sh k1-rvv256 add_bias
./examples/run/weft.sh k1-ime256 q4_0_projection_ime prefill 3
```

每个 examples 分支指定唯一 DSL kernel、runtime、target profile 与真实 workload shape。Runtime
可以在同一进程调用 GGML baseline 做数值/性能对照，但 Weft 生成代码不得调用 GGML。

目标机暂不可用时，只能报告已经生成到 Kernel IR、intrinsic C、object 中的哪一层；不能把未执行
写成通过。

仓库不建立 unit test、boundary test、fixture、lit/pytest、版本兼容检查或验证矩阵。解析器、
verifier 和 target unsupported error 是编译器本身的一部分，不形成另一套测试系统。
