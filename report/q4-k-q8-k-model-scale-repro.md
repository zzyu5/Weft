# Q4_K × Q8_K 模型级单 Kernel Repro

## 测量对象

这是 `q4_K_q8_K` 自己的 runtime，不是通用算子 harness，也不是 `llama-bench`
端到端测量。它使用 DeepSeek-R1-Distill-Llama-8B-Q4_K_M 的真实 tensor
角色与完整投影尺寸：

| tensor | M | N | K | phase |
|---|---:|---:|---:|---|
| `blk.0.attn_q.weight` | 1 | 4096 | 4096 | decode |
| `blk.0.ffn_up.weight` | 1 | 14336 | 4096 | decode |
| `blk.0.attn_k.weight` | 128 | 1024 | 4096 | prefill |

输入值由 runtime 确定性构造，但 buffer、Q4_K/Q8_K block layout、输出矩阵和
调用次数均为上述完整模型级规模。初始化不计时。每个输出元素分别调用一次 Weft
生成的 `q4_K_q8_K` 和 GGML RISC-V 的 `ggml_vec_dot_q4_K_q8_K`；二者共享
同一输入、外层投影循环和单核 CPU 绑定。

## RVV 实测

目标机入口：

```sh
./examples/run/weft.sh q4_K_q8_K attn_q decode 3
./examples/run/weft.sh q4_K_q8_K ffn_up decode 3
./examples/run/weft.sh q4_K_q8_K attn_k prefill 3
```

| tensor / phase | Weft | GGML | Weft / GGML | 数值 |
|---|---:|---:|---:|---|
| attn_q / decode | 264.829 ms | 3.465 ms | 0.013083× | max abs/rel error = 0 |
| ffn_up / decode | 926.882 ms | 12.290 ms | 0.013259× | max abs/rel error = 0 |
| attn_k / prefill | 8473.426 ms | 109.332 ms | 0.012903× | max abs/rel error = 0 |

表中时间是完整 warmup 后三次串行运行的进程内均值。对应有效整数 dot 吞吐为
Weft `0.1267 GOP/s`，GGML `9.56–9.82 GOP/s`。当前发射代码约慢 `75–77×`；差距在 decode、prefill 和不同
输出宽度上稳定存在。

## 结论边界

这个结果只刻画单线程、完整投影调用次数下的 Q4_K × Q8_K block-dot 实现质量。
它不包含 GGML graph dispatch、activation quantization、线程池、repack 或模型其他
算子，也不与 `llama-bench` 的 token/s 共用分母。当前问题在 Weft 的该 kernel
发射本身：数值语义已经闭合，但 RVV lowering 产生了大量逐元素索引、gather 和宽整数
运算，没有接近 GGML 手写 kernel 的数据布局与指令组织。
