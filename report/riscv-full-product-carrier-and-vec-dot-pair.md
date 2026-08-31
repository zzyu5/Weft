# RISC-V 完整 Product Carrier 与 Vec-dot/Decode 配对收口

日期：2026-08-31  
实现提交：`f822bd54e`（`Plan complete RVV product carriers`）

## 1. 结论

本轮只处理 IQ2_XS vec-dot 中的完整 widened product carrier，没有铺新格式，也没有改作者
tree。此前后端先把 reduction input 切成 16-lane slices，再逐片 multiply/reduce，最后用
slide 拼回逻辑结果；现在 physical planner 先选择一个完整 `i16m8` product carrier，再以
typed repack 切成最终 reduction windows。

结果同时出现在 standalone vec-dot 与 MUL_MAT decode：

| target | entry | 改前 GOP/s | 改后 GOP/s | source GOP/s | 改后/source |
|---|---:|---:|---:|---:|---:|
| SG2044 | standalone IQ2_XS×Q8_K | 2.530 | 3.392 | 3.483 | 97.4% |
| SG2044 | MUL_MAT decode IQ2_XS | 2.517 | 3.121 | 3.490 | 89.5% |
| K1 | standalone IQ2_XS×Q8_K | 1.393 | 2.176 | 1.784 | 122.0% |
| K1 | MUL_MAT decode IQ2_XS | 1.409 | 2.175 | 1.760 | 123.6% |

K1 两条超过 source；SG 两条提升但 decode 仍低于 source 10.5%。因此本轮证明了 product
carrier 的 P1/P4 归因，却没有证明 SG 的 vec-dot 已完全收口。

## 2. 第九节七步账本

### 2.1 Hot Level、轴与资源

IQ2_XS 的 hot contraction 保持作者已经写出的 entry、payload、group 与 reduction axes。
修改前后 logical values、Level、Encoding、widening 与最终 reduction 都不变。

新的 planner 在每个 issue 中选择：

```text
SG2044: 64  i8 lanes → one i16m8 product → 4 × 16-lane reductions
K1:     128 i8 lanes → one i16m8 product → 8 × 16-lane reductions
```

完整 product 占 8 个 vector groups；lhs/rhs issue operands 各占 4 groups，planner 的联合峰值
为 17/32 groups。该合同在 materialization 前已经闭合，不依赖 emitter 缩小 LMUL。

### 2.2 Work ledger

| 项目 | 旧 SG | 新 SG | 旧 K1 | 新 K1 |
|---|---:|---:|---:|---:|
| widened product / issue | 4 × i16m2 | 1 × i16m8 | 16 × i16m2 | 1 × i16m8 |
| final 16-lane reduction / issue | 4 | 4 | 16 | 8 |
| product reassembly | 3 `vslide1up` | 0 | 30 `vslide1up` + 18 `vslidedown` | 0 |
| scale supply | vector gather + 4 extracts | 同左 | vector gather + 8 extracts | 4 scalar loads，SSA replica |

K1 原有 48 条 slide/slidedown 不参与数值计算，只重建过早切碎的 product。新 C 中该组
指令归零。SG 的 product 拼装同样归零，但 scale 的 vector-to-scalar conversion 仍保留
1 次 gather 和 4 次 extract。

### 2.3 Supply identity 与 memory edge

weight payload 与 Q8 activation 各在 issue window 中载入一次。indexed codebook entry 与
regular-repeat scale 仍由 typed storage relation产生，不按 consumer 重建。

regular-repeat scale 有两种合法供应形态：

- 4 replicas：保留 vector gather，再显式 convert 为 scalar replicas；
- 8 replicas：按 source identity 只载入 4 个 scalar bytes，SSA 复用成 8 replicas。

这不是按 VLEN 或格式名分支。planner 读取 replica 数与 rematerialization use-def 能力，写入
`NestedPartialPlanAttr.scale_supply`；materializer 只执行 `vector-convert` 或
`scalar-rematerialize`。曾尝试在 SG 的 4-replica 情况也标量化整条 scale 链，standalone
由约 3.45 降至 2.96 GOP/s，因此该负路径没有作为统一规则保留。

### 2.4 Contraction、reduction 与 spill

最终 IR 的核心链是：

```text
rvv_partial_set<i16m8>
→ rvv_partial_repack<4 or 8 windows>
→ rvv_partial_reduce<i32>
→ rvv_partial_scale_combine
→ rvv_partial_finalize
```

两台机器的生成 C 都只有一个 widened multiply。standalone 路径没有新增 vector spill；
MUL_MAT decode 多出的 reduction 来自 activation quantization，不属于 vec-dot product。

### 2.5 Pipeline

该 kernel 的收益来自 P1 carrier 与 P4 convergence，不来自 P5。pipeline depth 保持 1；没有
把 prefill 的 output-cohort 或 software pipeline 机制带进 M=1 decode。

### 2.6 Donor 对照与责任归属

GGML donor 先构造完整 `i16m8` product，再按 16-lane segment reduction。该差异不改变作者
tree，只改变同一 product 的物理载体和收敛点，因此属于 compiler。

## 3. 与 Triton/TileLang 的机制对照

Triton 的 canonical `tt.dot` 只保存数值关系；
[`AccelerateMatmul.cpp`](../../ref/triton/lib/Dialect/TritonGPU/Transforms/AccelerateMatmul.cpp#L441)
根据 target、dtype、shape、warps 与 instruction shape 选择 result/operand carrier，随后
[`FMA.cpp`](../../ref/triton/lib/Conversion/TritonGPUToLLVM/DotOpToLLVM/FMA.cpp#L10)
只对已经 unpack 的 A/B/C 拼写局部乘加。Reduction lowering 则从已选 layout bases 决定何时
做 register/lane reduction，见
[`ReduceOpToLLVM.cpp`](../../ref/triton/lib/Conversion/TritonGPUToLLVM/ReduceOpToLLVM.cpp#L228)。

TileLang 同样先由 backend `infer_layout` 选择 fragment carrier，再由 `lower` 机械实例化，见
[`gemm_mma.py`](../../ref/tilelang/tilelang/cuda/op/gemm/gemm_mma.py#L20)；reducer 的 pass 顺序是
LayoutInference 后再 plan/materialize，见
[`transform/__init__.py`](../../ref/tilelang/tilelang/transform/__init__.py#L409)。

Weft 复用的是“typed carrier owner 先冻结、materializer 后实例化”这一机制，不复用 GPU 的
thread/warp/CTA ownership。Weft 的 carrier coordinates 仍是 issue time、RVV lane、register
replica 与 fragment。

## 4. 实现边界

`PlanRISCVPartialTopologies` 与 materializer 在主链中相邻，见
[`RISCVCompiler.cpp`](../lib/Target/RISCVCompiler.cpp#L41)。planner 的
[`planNestedPartialCarrier`](../lib/Target/MaterializeRISCVPartialAccumulators.cpp#L2425)
一次写入 issue types、完整 product type、split/reduced types、partial-set types、scale supply、
multiply leaf 与 resource peak。materializer 对 clone 后的类型做等值检查，再实例化这些 op；
不再调用 lane-slice planner，也不再创建临时 sequential dot。

regular-repeat scalar supply 是真实 physical op，定义和 verifier 分别位于
[`RISCVOps.td`](../include/Weft/Dialect/RISCV/IR/RISCVOps.td#L1280) 与
[`RISCVDialect.cpp`](../lib/Dialect/RISCV/IR/RISCVDialect.cpp#L5078)。terminal emitter
[`RISCVIntrinsicC.cpp`](../lib/Target/RISCVIntrinsicC.cpp#L9793) 只把已选 scalar load 展开为 C，
不选择何时使用它。

提交前独立审计仍发现旧的 scalar-add-tree、independent 与 level-scaled materialization 路径会
现场构造部分 combine tree。这些不是本轮新增 nested 路径，但说明不能把“nested owner 已
收口”扩大成“所有 partial materializer 已纯机械化”。

## 5. Q1_0 反例与横向扫描

Q1_0 不是同一种病。K1 最终代码没有 product slice/reassembly slide；它是 128 个独立
`vwmul + vwredsum`，并有 32 对 stack spill/reload。donor 则对 sign-merged i8 直接做 i16
reduction，没有 product vector。它与 IQ2 只共享宽泛的 P1/P4 分类，不能作为本机制的第二个
正例。

Q1_0 配对回归：

| target | standalone | decode | standalone/source | decode/source |
|---|---:|---:|---:|---:|
| SG2044 | 4.679 | 4.646 | 206.6% | 199.1% |
| K1 | 2.799 | 2.806 | 74.2% | 75.9% |

对 24 格式 × 2 target 的静态扫描中，47 个组合生成成功；只有 IQ2_XS 出现“product 先切片、
随后 slide 拼回”这一精确模式。SG TQ2_0 在该静态扫描的默认 binding 下资源非法，未纳入
横向判断。其余 slide 分别属于 bit/index、radix 或 packed-layout conversion。

因此，本轮有跨 target 和 standalone/decode 两类外部证据，但没有第二个跨格式正例。

## 6. CSV 脏数据清理

旧 CSV 的 MXFP4 与 K1 TQ2_0 配对来自不同快照。用当前 HEAD、相同 Clang/flags、10 次中位数
重测后：

| target/format | standalone | decode | decode/standalone |
|---|---:|---:|---:|
| SG MXFP4 | 6.774 | 6.716 | 99.1% |
| K1 MXFP4 | 2.338 | 2.317 | 99.1% |
| K1 TQ2_0 | 6.287 | 6.292 | 100.1% |

原来的 `143%/154%/110.7%` outlier 消失，说明它们是 CSV 快照污染，不是 GEMV wrapper 的
独立优化。

## 7. 预测对答案与未闭合项

优化原则原预测 IQ2 的首要问题是 P2/P4。实际顺序是：P1 issue/product carrier 与 P3
indexed-entry 先暴露，P2 regular-repeat supply 随后生效，最后才是本轮 P4 完整 product。
该预测记为 **MISS**，不因 P2/P4 后来确实出现而改写解释。当前累计为
`0 HIT / 1 MISS / 5 pending`。

仍未闭合：

- SG IQ2_XS standalone/decode 为 source 的 97.4%/89.5%；最终 IR 仍有 1 次 regular-repeat
  gather 和 4 次 scale extract，但尚无证据证明全部剩余差距都来自这里。
- Q1_0 K1 为 source 的 74.2%/75.9%，其病灶是独立 product/reduction 与 spill，不是本轮
  product reassembly。
- 没有第二个格式触发相同完整-product修复，因此不能宣称该规则已经跨格式泛化。
- 旧 partial materialization 路径仍存在现场 combine-tree owner；本轮只收口 nested 路径。

## 8. 手工复现

```bash
examples/run/weft-quantized-vec-dot.sh sg2044 iq2_xs 10
examples/run/weft-mul-mat.sh sg2044 iq2_xs decode 10
examples/run/weft-quantized-vec-dot.sh k1 iq2_xs 10
examples/run/weft-mul-mat.sh k1 iq2_xs decode 10
```

四条均为 `numeric=within-tolerance`；没有建立测试目录或性能脚手架。
