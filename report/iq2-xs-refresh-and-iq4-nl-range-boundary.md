# IQ2_XS 测量刷新与 IQ4_NL 融合边界

日期：2026-09-05

本快照只记录当前 `main` 上的定向重测与静态诊断，不改变编译器或作者程序。

## 当前数字

所有数字均为当前源码生成、10 repetitions、与固定 source 输入合同配对的结果。

| kernel | target | phase | Weft | source | ratio |
|---|---|---|---:|---:|---:|
| IQ2_XS vec-dot | SG2044 | decode | 4.171873 | 4.164948 | 100.17% |
| IQ2_XS MUL_MAT | SG2044 | decode | 4.283233 | 4.091731 | 104.68% |
| IQ4_NL vec-dot | K1/X60 | decode | 2.018567 | 2.821807 | 71.53% |
| IQ4_NL MUL_MAT | K1/X60 | decode | 1.978564 | 2.762856 | 71.61% |
| IQ4_NL MUL_MAT | K1/X60 | prefill | 2.058851 | 2.870430 | 71.73% |

IQ2_XS 的两条 SG 旧值不是当前实现的真实性能；定向重测后 standalone 与
MUL_MAT decode 同步过线。`report/kernel-performance-comparison.csv` 已用上述结果覆盖。

## IQ4_NL 的工作账

K1 standalone/decode 的最终 RISC-V IR 每个 32-element record 含：

- 一次 16-byte packed-weight load；
- 两次 16-byte Q8 load；
- 两次 codebook lookup；
- 两次 widening MAC；
- 两次 widening reduction 和两次 scalar extract。

donor 的 load 和 product 数量相同，但 low/high 两条 stream 共用一个 i16 product
carrier，只做一次最终 widening reduction 和一次 extract。因此当前可数的额外工作是
每 record 一次 reduction、一次 seed/extract 以及随后的 scalar combine；不是 read-CSE，
也不是多载入了一份 payload。

## 为什么不能直接打开现有 fused 路径

`LowerRISCVComposites.cpp` 中的 `fusedStreamsLegal` 只有在 operand 的整数范围能证明
窄 carrier 不溢出时才允许 stream 融合。IQ4_NL codebook 在作者程序中是普通
`View[i8, (16,)]`，编译器只能使用类型全范围，不能假设其内容一定是 GGML 固定表中的
`[-127, 113]`。GGML 的合法 Q8 payload 同样具有比裸 `i8` 更窄的实际输入域，但这个事实
目前也没有进入 canonical program。

直接照 donor 生成单一窄 carrier 会扩大当前程序允许的输入集合上的溢出风险，因而不是
合法的 physical rewrite。

可闭合的两条路归属不同：

1. 作者程序携带可支配 use 的数值范围前提，使现有窄 fused carrier 可被证明合法；
2. 编译器形成更宽的 typed partial-combine topology，在不依赖输入窄范围的情况下延迟收敛。

前者类似 Triton `tl.assume` 产生的显式 assumption 及其 dominance-sensitive range
analysis，而不是 target pass 根据格式名猜表内容。后者必须成为可观察的 RISC-V op/use-def
和资源合同，不能留给 emitter 临时重排。

MXFP4 提供了相近的 grouped/layered codebook 输入，但其表值范围、scale 关系和 use-def
不同；它能作为 issue grouping 的第二个观察输入，不能证明 IQ4_NL 的窄融合数值合法。

## 当前结论

IQ4_NL 的低值已经定位到具体 carrier 与收敛边界，但当前没有足够依据在“作者输入范围合同”
与“通用宽 partial combine”之间替语言作选择。该项标记为待定；不保留 emitter 特判、格式
分支或无证明的窄融合。
