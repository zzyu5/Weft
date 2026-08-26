# F16、blocked quant 与物理 pass 联合推进报告

## 1. 本轮问题与验收方式

本轮同时修改作者树和物理编译器，但用不同证据判断两侧责任：

- F16 已经具有 `NC/KC/MC/MR/NR/KB`、staging、多个 accumulator 和 `outer_contract`，因此它用于隔离物理编译器问题；
- Q4_0、Q5_0 原先的 production prefill 只是逐行逐列调用一次 vec-dot，因此它们用于验证作者树改成 blocked/cohort 结构后，编译器能否消费 shaped values；
- Q5_0 的 high-bit plane 用来检验 pass 是否能透过真实 `convert_layout` 传播 typed mapping，而不是只匹配相邻 SSA closure；
- SG2044/VLEN128 与 K1/VLEN256 分别真机运行；正确性使用现行浮点容差，不要求 bit-exact；性能只与同目标、同 phase、同 shape 的 baseline 比较。

本轮没有跑全量 206 条，也没有据此声称其他量化格式已经获得相同实现。

## 2. 先保留 generated C 与汇编

`examples/run/weft-mul-mat.sh` 新增 `WEFT_KEEP_ARTIFACTS=1`：

```bash
WEFT_KEEP_ARTIFACTS=1 ./examples/run/weft-mul-mat.sh sg2044 f16 prefill 3
```

开关关闭时，runner 仍删除本地和远端临时目录；开关打开时，它打印精确的本地、远端目录，并在远端额外生成 `kernel.s`。这个入口只用于当轮分析，不向仓库保存 generated artifact。

观察到的汇编事实：

- 旧 F16 Weft 候选在 reduction 内形成多个 vector accumulator，内层可见成组 F16 load 与 `vfwmacc.vv`；
- 同一候选在 SG2044 上有 416-byte 栈帧，在 K1 上有 480-byte 栈帧，K1 的 spill/reload 更多；
- GGML 的 F16 vec-dot leaf 是两次连续 `vle16`、一次 `vfwmacc.vv` 和一次末尾 reduction，无栈帧；
- 因此旧结果不能只归因为 F16 转换指令。真正的差异还包含 reduction 轴是否直接成为 RVV lane、多个输出的寄存器组织、load reuse、LMUL 和 KB。

这份证据促使修复发生在 physical lowering，而不是在 emitter 中增加一个 F16 分支。

## 3. F16：从普通 outer contract 到 typed stream contract

### 3.1 物理 IR 新结果

F16 `outer_contract` 现在可以 lowering 为真实的 `weft_riscv.rvv_stream_contract`。该 op 在最终物理 IR 中携带：

- reduction axis；
- operand layout：F16 SEW、LMUL、VL 与 lane axis；
- accumulator layout：F32 SEW、翻倍后的 LMUL、free-axis register replicas；
- stationary operand；
- lhs/rhs memory access；
- unroll；
- 已选的 `rvv.stream-widen-contract` leaf。

`LowerRISCVComposites` 从 typed operand、result axes、memory edge、target LMUL 集合和 schedule 产生这些字段；`RVVStreamContractOp::verify` 检查 F16→F32 widening、reduction lane、free-axis replicas、memory relation 和 LMUL 合法性；intrinsic-C emitter 只读取这些字段写出 load、`vfwmacc.vv` 与最终 reduction。

同时修正了每种 SEW 的 fractional LMUL 合法性：`LMUL * ELEN >= 8 * SEW`。这阻止 K1 为 F32 value 产生非法的 `mf4` 形态。

### 3.2 目标相关实例

同一 F16 DSL 树最终使用了不同的目标实例：

- SG2044：`NC=32, KC=4096, MC=16, MR=4, NR=2, KB=4096, LMUL=m1`；
- K1：`NC=32, KC=4096, MC=16, MR=2, NR=2, KB=4096, LMUL=m2`。

这不是由 target 名选择 leaf；runner 为固定 production shape 绑定构建期 `auto` 参数，物理 legality 仍由同一目标 profile 和同一 lowering 检查。

### 3.3 10 次真机结果

| target | phase | 旧 Weft GOP/s | 新 Weft GOP/s | baseline GOP/s | 新/基线 |
|---|---:|---:|---:|---:|---:|
| SG2044 | decode | 0.403 | 5.021 | 4.542 | 1.105× |
| SG2044 | prefill | 3.222 | 18.356 | 14.358 | 1.278× |
| K1 | decode | 0.320 | 3.378 | 3.429 | 0.985× |
| K1 | prefill | 0.248 | 5.849 | 5.488 | 1.066× |

K1 decode 与 baseline 的差距约 1.5%；本轮没有用 emitter 私判去追这一个数字。

## 4. Q4_0 与 Q5_0：作者树改成 blocked/cohort 程序

### 4.1 新 prefill 树

两个 production prefill 函数现在都明确写出：

```text
activation quantize
→ N tile / M tile
→ materialize weight tile 与 activation tile
→ NR output cohort / MR row cohort
→ [MR, NR] accumulator
→ 32-element K block
→ outer_contract
→ commit output cohort
```

Q4_0 内层使用 shaped `x.q × centered(w.q)`；Q5_0 额外使用一个显式 K-axis iota 和 high-bit plane reconstruction。activation 在多个输出之间的共享、accumulator 的 Level 归属和 output cohort 都由作者树表达，编译器没有从逐行 vec-dot 猜出 blocking。

### 4.2 decode 与 prefill 是两个作者特化

把 blocked Q5_0 树直接用于 M=1，SG2044 只有 2.352 GOP/s；原先的单行 vec-dot 树为约 4.99 GOP/s。原因是 M=1 无法摊销 blocked materialization 和空 row replicas。

因此 Q4_0、Q5_0 decode 使用逐输出的 GEMV 特化，prefill 使用 blocked GEMM 特化。两者具有相同 ABI 和数值语义，但不同的 Level/value graph；这是作者选择，不是 target 在 lowering 中改树。

### 4.3 Encoding 审查结论

本轮逐项核对 Q4_0、Q5_0 与 IQ2 代表格式后，没有修改 Encoding：

- Q4_0/Q5_0 的低 nibble 已由 `grouped(32) + layered(16, lo_first)` 完整描述；
- Q5_0 的 `qh: u8[4]` 是自然 byte array，high-bit 拼接属于 numerical decode；
- IQ2 的 `u16` word 与独立 scale 字段已与 GGML ABI 对应，word 内 grid/sign/scale 分割属于 numerical decode，不是 storage byte mapping。

为了“看起来更统一”把这些数值 bitfield 塞入 Encoding，反而会混淆纯 storage layout 与数值解码。

## 5. 两项横向编译器修复

### 5.1 encoded materialize 的坐标原点

blocked Q4_0 首次运行时，N tile 后半部分数值错误。生成 C 显示父 `NC` base 和子 Point base 被重复加入地址。

修复后：

- nested `domain/group_index` Point 替换该 logical axis 的绝对 selector；
- 普通 scalar `index` 仍作为相对 local offset；
- materialized encoded record 保存每条保留轴的绝对 origin；
- 后续 extract 先减 record origin，再形成 record 内坐标。

该规则同时用于 dense 与 encoded nested slices，没有按 Q4/Q5 格式补偿地址。

### 5.2 Q5 bitplane fusion 穿过 typed conversion

旧 Fuse pass 只接受：

```text
iota → div/mod → gather → shift/and → shift/or
```

且要求 iota 与 consumer 相邻、结果是静态一维 value。Q5 blocked tree 的真实物理 IR 是：

```text
iota → convert_layout(tuple) → div/mod
qh[N,4] → extract[all,gather]
→ shift/and/or → q[N,32]
```

本轮将匹配依据改为 typed geometry：

- 透过 effect=`pure` 的 `convert_layout` 找到同一坐标源；
- 允许一个 gather selector 与其余 all selectors；
- 要求 gather axis 的 result extent 正好是 plane extent 的 8 倍，其余 axes 完全一致；
- 替换为真实 `rvv_bitplane_merge` op，并删除已经无 consumer 的纯旧链。

最终 Q5 物理 IR 中出现 1 个 `rvv_bitplane_merge`；原 extract/div/mod/shr/and/shl/or 链消失。该 pass 不读取 Q5 名称。

直接 A/B：

| target | fusion 前 GOP/s | fusion 后 10 次 GOP/s | 提升 |
|---|---:|---:|---:|
| SG2044 | 6.551 | 9.676 | +47.7% |
| K1 | 1.890 | 2.198 | +16.3% |

## 6. 当前 10 次结果

| target | format | phase | Weft GOP/s | baseline GOP/s | 比值 | 备注 |
|---|---|---:|---:|---:|---:|---|
| SG2044 | F16 | decode | 5.021 | 4.542 | 1.105× | RVV 对 RVV |
| SG2044 | F16 | prefill | 18.356 | 14.358 | 1.278× | RVV 对 RVV |
| SG2044 | Q4_0 | decode | 7.632 | 7.415 | 1.029× | GEMV 作者特化 |
| SG2044 | Q4_0 | prefill | 7.768 | 7.580 | 1.025× | blocked tree |
| SG2044 | Q5_0 | decode | 4.991 | 6.164 | 0.810× | 仍慢 19.0% |
| SG2044 | Q5_0 | prefill | 9.676 | 6.307 | 1.534× | blocked + bitplane fusion |
| K1 | F16 | decode | 3.378 | 3.429 | 0.985× | RVV 对 RVV |
| K1 | F16 | prefill | 5.849 | 5.488 | 1.066× | RVV 对 RVV |
| K1 | Q4_0 | decode | 2.239 | 11.409 | 不可直接比较 | baseline 是 IME1 asm，Weft 是 RVV |
| K1 | Q4_0 | prefill | 2.217 | 28.431 | 不可直接比较 | baseline 是 IME1 asm，Weft 是 RVV |
| K1 | Q5_0 | decode | 2.004 | 1.789 | 1.120× | RVV 对 RVV |
| K1 | Q5_0 | prefill | 2.198 | 1.850 | 1.188× | RVV 对 RVV |

Q4_K persistent grouped 路径的回归运行仍然正确：SG2044 prefill 10.123 GOP/s，K1 prefill 3.225 GOP/s（各 3 次）；本轮的通用地址与 bitplane 修改没有使它失效。

全部当前数字已写入 `report/weft-kernel-performance.csv`；baseline 文件未修改。

## 7. 没有完成的能力

### 7.1 Pipeline 仍不是通用 local scheduler

`PipelineRISCVLevels` 仍只接受 depth=2、buffer=2、一个 carry，且 loop body 必须恰好是一个既有 window-load 加一个对应 window-step。F16 typed stream contract 的 reduction loop目前封装在 physical op 中，Q4/Q5 blocked outer contract 也没有统一 typed window interface。

因此本轮 F16 的提升来自 reduction-lane mapping、typed widening stream、load reuse、microtile/LMUL/KB，而不是一个已经推广到任意 cluster 的软件流水 pass。若只把 depth 写成 2 而没有显式 window/cluster op，只会产生假 pipeline；本轮没有这样做。

### 7.2 Share 仍只覆盖相邻两层

`ShareRISCVLayeredWindows` 仍要求相邻两个一维 `grouped_layered` extract、严格相同的 parent/index 关系和两层几何。Q5 的成功来自独立的 typed bitplane fusion，不代表任意 layered/codebook window 已能跨 consumer 共享。

### 7.3 横向覆盖仍有限

- 本轮只把 Q4_0、Q5_0 两棵 MUL_MAT prefill 树改成 blocked；其余量化 MUL_MAT 仍有大量 row×column 树；
- 其余 row-dequant/vec-dot 中逐元素 Python 展开尚未全量迁移；
- K1 Q4_0 尚无从同一 blocked tree 得到 IME fragment 的闭合路径；
- SG2044 Q5_0 decode 仍只有 baseline 的 81.0%。

这些事实意味着，本轮证明了两种不同量化格式能消费 shared outer-contract 与 typed bitplane 能力，但没有证明 24 个格式或完整 206 条语料已经横向闭合。

## 8. 设计边界观察

本轮没有出现“必须由编译器改变作者 logical value 集合或 Level 归属才能得到上述结果”的证据：

- output cohort、materialize lifetime、MR/NR accumulator 和 decode/prefill 特化由 std 作者树写出；
- reduction axis 的 lane mapping、free axes 的 register replicas、F16 widening stream、memory form、LMUL legality 与 bitplane leaf 由物理编译器产生；
- emitter 消费最终 typed op。

这只说明 Q4_0/Q5_0/F16 三条路径没有证伪该边界；Pipeline、Share、其余 quant 与 IME 仍未提供足够横向证据。
