# Q3_K shaped tree 与 physical representation 清理

## 结论

这一节点完成了 Q3_K 从逐元素标量展开到显式 shaped 作者树的迁移，并补齐了它暴露的两项通用 physical compiler 能力：自然仿射轴合并，以及宽 RVV operand 的逻辑 slice carrier。Q3_K vec-dot 和 blocked MUL_MAT 已在 SG2044 与 K1 真机数值通过。

它还暴露并修复了一个真实的两层 IR 合同问题：composite replacement 后，已经失去消费者的纯 representation graph 仍留在最终 RISC-V IR，导致 intrinsic C 出现未使用变量。最终实现使用一个独立 pass 对 typed `pure` layout conversion 和 trivially-dead producer 做固定点删除；它不合并仍有使用的 physical op，也不让 emitter 判断 liveness。

Q3_K 还没有达到 source 性能。尤其 blocked prefill 虽然已经是正确的作者树，但生成的 partial/load 组织仍远低于 donor。这是当前明确的 compiler-side 性能缺口，不能用“迁移完成”掩盖。

## 作者程序

### Encoding

Q3_K 的 canonical record 现在显式描述：

- `hmask`: 256 个逻辑 `u1`，`grouped(256) + layered(32, lo_first)`；
- `q`: 256 个逻辑 `u2`，`grouped(128) + layered(32, lo_first)`；
- `scale_low`: 16 个逻辑 `u4`；
- `scale_high`: 16 个逻辑 `u2`；
- `d`: `f16`。

这使 storage mapping 位于 Encoding，而不是继续散落在 Python 的 byte index、shift 和 mask 中。

### Vec-dot

每个 256-element record 显式保留四条逻辑关系：

```text
half        = 2
plane       = 4
scale_part  = 2
lane        = 16, reduction axis k
```

作者树构造 Q3 value、16 个 signed scale，并对 `plane × scale_part × lane` 做 contraction 和 reduction。compiler 不再从 16×16 个 Python scalar operation 猜回这些轴。

### MUL_MAT

prefill 使用 `NC/MC` output tile、`MR/NR` accumulator cohort、256-element K block，以及 activation materialization。Q3 的局部数值分解与 vec-dot 一致；decode 保留单行 GEMV 特化。

当前可运行参数是 `NC=32, MC=16, MR=2, NR=1`。此前脚本中的 `MR=5, NR=2` 会产生 33 个同时存活的 vector groups，资源 pass 正确判非法，现已从正式 repro 中移除。

## 编译器改动

### 自然仿射轴合并

layout propagation 现在能从 index use-def 的系数链识别自然线性坐标：若相邻轴系数满足前一轴 extent 的连续乘积，则这些轴可共同进入 lane mapping。这个规则读取 axis、extent 与 affine coefficient，不读取 Q3_K 或任何格式名。

它解决的是 shaped activation 与 encoded weight 在 contraction 前的物理域闭合；没有这一步，二者会得到不一致的 lane/register 分解。

### 宽 operand 的逻辑 slice carrier

VLEN256 下，一个逻辑 reduction slice 可能小于 RVV 可直接 `vget` 的最小 register group，而 source operand 本身更宽。lowering 现在保留：

- source vector group；
- logical reduction lanes；
- selected m1 carrier；
- carrier 内 lane offset。

emitter 只按这些 typed facts 发 `vget` 和必要的 `vslidedown`。这使同一 Q3 tree 在 VLEN128 和 VLEN256 都能生成合法 widening-dot，不需要 target 或格式分支。

### 最终 representation DCE

`root_point` 与 `physical_point` 是 final IR 的结构节点，不能一边标记为可任意删除的 `Pure`，一边又由 verifier 要求每个 Level 必须存在。它们现在保留结构身份。

新增的 final cleanup pass 只做两件事并迭代到不动点：

1. 删除结果无使用且 effect 为 `pure` 的 typed layout conversion；
2. 删除 MLIR 判定为 trivially dead 的 numerical producer。

曾尝试使用通用 canonicalizer；它把算术改写成 emitter 合同之外的 `arith.select`。曾尝试使用 CSE；它把 TQ2_0 的独立 partial 合并，SG2044 prefill 从约 20.47 降到约 10.13 GOP/s。两项尝试均已撤掉。最终 pass 不改写或合并仍然 live 的 physical graph，TQ2_0 恢复到 20.468 GOP/s。

## 真机结果

以下都是本节点提交前的真实 production-shape 单次运行，用于确认代码状态；不是正式重复次数，因此没有覆盖 `weft-kernel-performance.csv`。

| kernel | target | result | GOP/s |
|---|---|---:|---:|
| Q3_K vec-dot | SG2044 | within tolerance | 6.234 |
| Q3_K vec-dot | K1/X60 | within tolerance | 2.089 |
| Q3_K prefill MUL_MAT | SG2044 | within tolerance | 0.886 |
| Q3_K prefill MUL_MAT | K1/X60 | within tolerance | 0.429 |
| Q2_K prefill regression | SG2044 | within tolerance | 10.114 |
| Q2_K prefill regression | K1/X60 | within tolerance | 2.961 |
| TQ2_0 prefill regression | SG2044 | exact zero observed error | 20.468 |

相对当前 CSV 中旧 scalar Q3 tree：

- SG2044 vec-dot 从 0.335 提高到 6.234 GOP/s，约 18.6 倍；
- K1 vec-dot 从 0.097 提高到 2.089 GOP/s，约 21.4 倍；
- SG2044 prefill 从 0.312 提高到 0.886 GOP/s，约 2.8 倍；
- K1 prefill 从 0.096 提高到 0.429 GOP/s，约 4.5 倍。

这些倍率只说明新作者树与 physical mechanism 相对旧 Weft 路径的变化，不是对 source 的正式性能结论。

## 负实验

以下改动都经过生成 C、汇编或真机结果核验后撤销，没有残留在提交中：

- 将 scale 提前 materialize 到父 Level：生成 16×i32 local array，带来约 1 KiB/block 的 stack/vector traffic，SG2044 vec-dot 降到约 2.28 GOP/s；
- 将 Q3 bitplane 写成普通 `or/shl` 以等待 fusion：现有 fusion 无法覆盖 `[4,2,16]` projected geometry，性能基本不变；
- 强制 free axis 进入更宽 lane placement：SG2044 略降；
- 把 raw Q3 window 扩成更宽 m2 load：静态热指令减少，但依赖链和寄存器组织恶化，SG2044 降到约 2.57 GOP/s；
- 最终使用通用 canonicalizer：产生不在 intrinsic-C 合同内的 `arith.select`；
- 最终使用 CSE：TQ2_0 的独立 partial 被合并，吞吐近乎减半。

## 尚未闭合的事实

Q3_K prefill 的作者树已经含 blocking、activation reuse 与 accumulator cohort，但当前资源可行实例只能使用 `MR=2, NR=1`，生成代码仍没有形成 donor 风格的多输出 partial/load reuse。因此 0.886/0.429 GOP/s 是正确但低效的 compiler 结果；这里不能靠再改 ABI 或退回 row×column 树掩盖。

Q6_K 与 TQ1_0 没有被判定为“应撤回的工作”。Q6_K 的 donor/encoding/axis 关系已经查清，下一完整节点必须把 `half/plane/scale/lane` 写成 shaped 作者树后再处理物理缺口；TQ1_0 当前 canonical row-dequant、vec-dot 和 blocked MUL_MAT 已能生成 VLEN128/VLEN256 intrinsic C，先前崩溃对应的 loop-carried local/window binding 已由当前 emitter 状态闭合。两者都不冒充本节点已经完成。
