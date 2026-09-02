# Row-dequant shaped-axis 聚类

本报告记录 `ff2862475` 之后的 row-dequant 工作。设计权威仍是 `doc/`；source 数字来自
`report/baseline/ggml-riscv-kernel-performance.csv`，当前测量写入
`report/weft-kernel-performance.csv`。真机结果统一使用 Clang 18、`-O3
-ffp-contract=fast`、相同 ISA/ABI 和 10 repetitions。下表所列数值结果的最大绝对误差与最大
相对误差均为 0。

## 1. 本轮先处理的遗留项

Q2_K vec-dot 在本轮开始前已经闭合，当前 standalone/decode 分别为：

| target | standalone | source | decode | source |
|---|---:|---:|---:|---:|
| SG2044 | 9.510 GOP/s | 9.308 | 8.953 GOP/s | 8.815 |
| K1/X60 | 3.495 GOP/s | 2.446 | 3.465 GOP/s | 2.404 |

因此本轮没有继续改 Q2_K，也没有把已经超过 source 的路径换成另一套实现。

## 2. 第九节工作账

旧的 IQ1_M、IQ2_S 和 IQ3_XXS row tree 都把一个 256-element block 展开成 256 次
Python 标量循环。静态 Physical IR/C 的工作账分别约为每个输出 61、42、37 个物理操作，
没有 RVV carrier；group metadata、codebook entry 和 payload 关系随输出重复出现。这里首先缺的
不是新的 physical pass，而是作者程序没有表达原本就存在的 `group × entry × payload` 逻辑轴。

三棵树现在显式表达：

- 8 个 32-element group；
- 每组 4 个 codebook entry；
- 每个 entry 的 8-element payload lane；
- 各格式真实存在的 scale、high bits、sign 或 delta 数值支路。

编译器仍从 Encoding、typed axes、use-def 和 target facts选择 carrier 与 memory form。没有从展开
后的标量 SSA 猜回轴，也没有增加格式名 matcher。

## 3. 结果

单位为 MElements/s。

| format | target | 改前 | 当前 | source | 当前/source |
|---|---|---:|---:|---:|---:|
| IQ1_M | SG2044 | 71.665 | 709.896 | 700.561 | 101.33% |
| IQ1_M | K1/X60 | 29.490 | 316.873 | 122.728 | 258.19% |
| IQ2_S | SG2044 | 108.565 | 688.171 | 418.088 | 164.60% |
| IQ2_S | K1/X60 | 40.870 | 225.420 | 179.633 | 125.49% |
| IQ3_XXS | SG2044 | 109.572 | 348.244 | 312.528 | 111.43% |
| IQ3_XXS | K1/X60 | 35.469 | 198.197 | 174.039 | 113.88% |

IQ1_M 的 `grid_delta` 先把 i8 codebook value 显式提升到 i32，再转 f32；这与数值树的
arithmetic domain 一致，避免依赖后端接受未定义的 i8→f32 复合转换。

更新后的 24-format × 2-target row-dequant 分布为：`>= source` 37 条、`90%–100%`
0 条、`70%–90%` 6 条、`50%–70%` 5 条、`<50%` 0 条。仍低于 source 的 11 条是：

- SG2044：Q1_0、Q4_0、Q4_1、Q5_0、Q5_1、Q4_K、Q5_K、IQ3_S、TQ1_0；
- K1/X60：IQ3_S、TQ1_0。

## 4. 机制对照

Triton `AxisInfo.cpp:802-945` 与 `CoalesceUtils.cpp:17-95` 先传播 contiguity、divisibility
和 axis facts，再选择 typed layout；`RemoveLayoutConversions.cpp:259-347` 只沿已有 shaped
SSA 传播或重物化，不从八份标量 op 发明一个逻辑轴。TileLang
`loop_vectorize.cc:1115-1170` 同样先证明 loop index 能成为 ramp 或 broadcast，再做机械
vector rewrite。本轮三棵作者树提供的是这些分析所需的 logical axes，不是把物理 layout 写进 DSL。

## 5. 负结果与已定缺口

TQ1_0 的 radix-3 输出由三个不同长度、不同 stride 的逻辑段组成。尝试把三个段写成 shaped value
时，frontend 在对 local Value 使用 shaped destination index 处明确拒绝：

```text
shaped gather currently indexes a local Value
```

该实验没有保留。当前 DSL 文档已经定义 canonical reshape/transpose/index，但实现尚不能表达
这种 segmented shaped projection/scatter。继续保留 Python 标量循环是合法程序，却不会给
physical compiler 一个完整 shaped output axis；用后端 matcher 识别 radix-3 拼写会越过 IR 边界。

SG2044 的 Q4/Q5 family 形成另一个聚类：LMUL 扫描后仍稳定在 source 的 80.9%–84.3%。
生成汇编显示 Weft 在单个 record 内向量化，并用 slide 重新组织 payload；donor 则让 record axis
骑 lane，用跨 record strided load/store 处理多个 block。现有 Physical IR 已能表示 strided field
load，但没有一个 pass 将 Level 的多个动态 record iteration strip-mine 成 RVV lane cohort。
TileLang `loop_vectorize.cc:1003-1049` 的 loop rewrite 与 Triton AxisInfo 的 strided pointer facts
是直接参照；这是待实现的通用 physical loop-vectorization 能力，不是 Q4/Q5 leaf 特判。

Q5_0/Q5_1 还共同暴露了 high-bit plane 的第二个缺口：同一 qh metadata 在四个 physical parts
中被分别 `vlm + vmerge + shift/or`，当前 IR 没有“一次 metadata carrier，多个 mapped consumer”
这一 typed relation。Q5_0 与 Q5_1 的 q/qh geometry 相同而数值尾部分别为 symmetric 与 affine，
因而它们可作为该关系的两个独立输入。

## 6. 机械验收

24 个 row-dequant entry 在 SG2044 与 K1 上形成 48 份 RISC-V Physical IR。每份均执行：

```text
weft-compile --emit=riscv-ir
weft-opt --canonicalize --cse
         --weft-riscv-canonicalize-layouts
         --weft-riscv-share-layered-windows
         --weft-riscv-share-layered-windows
         --weft-riscv-verify-final --verify-each --verify-roundtrip
```

结果为：生成 48/48，parse/verify/canonicalizer/CSE 48/48，同一管线第二次运行后的文本 diff
为 0 的入口 48/48；没有失败入口。
