# RVV 性能缺口收敛：本 change 汇总

日期：2026-09-09。范围从 `22dcd8506` 起，包含本 change 的实现、定向测量与收尾。
设计规范仍以 [doc/index.md](../doc/index.md) 为准；本文是一份结果快照。

## 结果与统计口径

当前 [对比表](kernel-performance-comparison.csv) **202/202 行达到 Weft/baseline ≥ 0.95**，
且与 [当前性能表](weft-kernel-performance.csv) 的吞吐逐行一致。
[固定 baseline](baseline/ggml-riscv-kernel-performance.csv) 未改动。
最低 ratio 为 K1 Q6_K prefill 的 **0.951779**；不存在未达标或待定性能条目。

这不是一次全量 202 项重跑。按起点对比表列出的 26 个缺口启动，先刷新 23 条旧记录，
对已有 3 条证据核对复用条件，随后按实际影响分批运行、即时更新两张可变表。
未受影响的历史有效记录继续保留；达标不代表所有运行都有很大余量，也不代表硬件性能下界。

## 保留了什么

### 编译器主链

- **partial、scale 与资源。** `Partial/` 的 planner/materializer、`MaterializeRISCVResources`
  保留已归约 i32 partial、合并有序 scale windows、使用合同闭合的寄存器切片，并缩短私有
  reload 与纯整数链的驻留。多路 i16 独立加树若不能证明安全，在第一次 combine 前 widen，
  不以固定码表上的零误差代替完整整数范围证明。窗口搜索、成本比较和资源关闭均有边界。
- **索引、符号和共享供应。** `FuseRISCVBitplanes`、`ShareRISCVLayeredWindows`、
  memory/leaf passes 选择 typed mask、unsigned narrow-shift 和局部 byte-window 关系。
  joined metadata 共享连续 raw window，冗余 mask 由位宽范围证明后删除；不是格式专用模板。
- **layout 与重物化。** `PropagateRISCVLayouts` / `CanonicalizeRISCVLayouts` 区分
  编码存储窗口和 lookup 数值结果，保持必要的 pack/resize 边；同型整数 pointwise 继续
  传播 widening 派生宽度，不在 add/mul 处退回基础 LMUL。
  Q4_K SG 修正项由两组 4 元素变为一组 8 元素，减少一组读取、加宽、乘法及分片操作；
  Q5_K 是同关系的第二个实际输入。
- **标量化与调度。** 已完整归约的 singleton i32 scale 可在 i32 范围证明后用 scalar mul，
  系数仍在原 combine 点消费。纯浮点因子的提前准备限制为 8 个节点、64 个操作窗口及每
  block 4 个 share；不得跨多系数 GPR/scalar tuple consumer，且必须提供完整 issue 覆盖。
  成本估计不覆盖数值、effect 或最终资源合法性。
- **terminal 修正也如实保留。** typed halfword 的 LE load 使用定长 memcpy，读取点提供已经
  证明的 alignment，避免丢失真实的半字访问机会。它是 terminal 翻译质量修正，不冒充高层 pass。
  layout、memory form、leaf 与调度仍由 Physical IR 选择，emitter 不按格式接管 whole kernel。

### 作者表达和有限绑定

- IQ1_S / IQ1_M 使用可证明范围内的 metadata 表达；Q2_K scale 保持标量读取和 mask。
- IQ2_XXS prefill 显式表达完整 packed-entry 与多行复用；Q6_K prefill 把纯解码放到
  64 元素窗口内，内部仍保留原 16 元素 dot、scale、i32 累加和浮点结合顺序。
- F32、Q4_1、IQ4_NL 等选择已有有限 source/physical bindings，并落回 kernel 旁的
  `tuning.json`。绑定收益不单独归成编译器 pass 收益。

参考方式沿 [优化原则第九节](../doc/compiler/optimization-principles.md) 的七步工作账：
先看物理轴、动态工作、供应身份、访问、partial/reduction/spill，再看调度并与 donor 对照。
参考只读 Triton 的 layout conversion、rematerialization 与指令重排，
TileLang 的 layout/向量化候选处理；具体量化程序逐段对照 `source/` 中 GGML donor。
未链接、包装或 fallback 到参考实现。

## 初始 26 个缺口的收尾结果

“初始”来自 change 起点的对比表，是清单起点，不是所有条目统一重测后的配对基线。
完整吞吐、配置与误差见既有两张性能表，不另建 CSV。

| kernel | target | phase | 初始 ratio | 当前 ratio |
| --- | --- | --- | ---: | ---: |
| iq2_xxs_q8_k | SG2044 | decode | 0.780844 | 0.960852 |
| mul_mat_iq2_xxs | SG2044 | decode | 0.794455 | 0.978554 |
| mul_mat_iq1_s | SG2044 | prefill | 0.686181 | 1.098730 |
| mul_mat_iq1_m | K1/X60 | decode | 0.765177 | 0.960282 |
| iq1_m_q8_k | K1/X60 | decode | 0.773803 | 0.955284 |
| dequantize_row_iq3_s | SG2044 | tensor | 0.729545 | 1.991438 |
| iq3_xxs_q8_k | SG2044 | decode | 0.818003 | 1.318486 |
| iq1_s_q8_k | K1/X60 | decode | 0.791664 | 1.046012 |
| iq1_s_q8_k | SG2044 | decode | 0.805320 | 0.987472 |
| mul_mat_iq1_s | K1/X60 | decode | 0.808147 | 1.070133 |
| mul_mat_iq1_s | SG2044 | decode | 0.812287 | 0.992905 |
| q2_k_q8_k | SG2044 | decode | 0.802374 | 0.986577 |
| mul_mat_q1_0 | SG2044 | prefill | 0.825704 | 1.022794 |
| mul_mat_q2_k | SG2044 | decode | 0.825936 | 1.006809 |
| iq4_nl_q8_0 | K1/X60 | decode | 0.840755 | 0.956597 |
| mul_mat_iq4_nl | K1/X60 | decode | 0.858867 | 0.969841 |
| mul_mat_iq2_xxs | K1/X60 | prefill | 0.867087 | 1.545433 |
| q4_k_q8_k | SG2044 | decode | 0.882827 | 1.012646 |
| mul_mat_q6_k | K1/X60 | prefill | 0.895851 | 0.951779 |
| mul_mat_q4_1 | K1/X60 | prefill | 0.911246 | 1.050367 |
| mxfp4_q8_0 | K1/X60 | decode | 0.918302 | 0.972097 |
| mul_mat_mxfp4 | K1/X60 | decode | 0.931623 | 0.987717 |
| iq3_xxs_q8_k | K1/X60 | decode | 0.942819 | 1.167406 |
| mul_mat_q4_k | SG2044 | decode | 0.938670 | 1.066336 |
| mul_mat_iq4_nl | K1/X60 | prefill | 0.789381 | 1.245393 |
| mul_mat_f32 | SG2044 | prefill | 0.764323 | 0.988693 |

## 最后一项：IQ1_S 行反量化的非原地输出合同

用户明确确认该入口不需要支持输出覆盖输入。正式定义采用
`@weft.kernel(alias_groups={"Y": "output"})`：Y 与 W/grid 不重叠，两个只读输入仍可互相 alias。
这项改变仅属于该作者入口，不改变语言默认 may-alias 规则、任意 i8 码表、公式或数值容差。

既有 read-snapshot pass 可以删除每 256 元素的 50 字节权重快照，C 参数携带 `restrict Y`
供系统编译器使用。收益是内存供应与后续调度的整体变化，不声称全部来自单个复制操作。

| 实际版本 | SG Melem/s | SG ratio |
| --- | ---: | ---: |
| 原 may-alias 合同，同源码/绑定隔离实跑 | 422.697002 | 0.581347 |
| 确认前的缓存诊断，不作为正式 winner | 716.564213 | 0.985511 |
| 用户批准后的普通入口，10 次计时 | 731.477618 | 1.006022 |

K1 同一正式入口为 **595.858120 Melem/s，ratio 4.590372**。
双机数值误差均为 0。违反不重叠承诺的调用不属于该入口的合法输入；没有添加自动备用路径。

## 负结果与修正

- RVVWidenAdd-only 的宽度传播假说没有改变 C，已撤回；实际断点在后续整数 pointwise。
  把同一传播扩大到浮点曾使 SG Q1_0 row 降到 ratio 0.842428；收回浮点覆盖后恢复 0.995734。
- Q6_K SG 的过早浮点准备使 standalone/decode 降到 4.332599/4.017536 GOP/s。
  同源码同绑定旧 compiler 配对实跑为 4.712145；最终多系数 scalar/tuple 约束使 standalone C
  与该实跑 C 一致，decode 重测 5.132102。ASM 没有新增 hot-loop spill，不能把早期
  “GPR 压力”候选解释写成已证明的 spill 因果。
- IQ4_NL K1 的更早浮点准备为 2.628480，按普通 op 数估计的窗口为 2.667125；
  register-issue 最近窗口保留 2.699332。另一种 pairwise i16 reduction 为 2.392501，撤回。
- Q4_K SG 分开的 unit 窗口为 8.870636；合并后 9.248587；删除已证明冗余 mask 后
  9.584062；最终完整 8-lane 整数链为 10.271419。另一棵 min 修正作者表达 8.435602、
  unroll4 9.200728 均不保留。
- Q1_0 K1 prefill 的 mask-only 初版因系统编译器额外展开而退到 3.201917；
  physical pass 保留已选 unroll 后为 3.996595。没有增加格式或机器名分支。
- IQ1_S row 的 unroll4 为 415.671895，i16→f32 中间路径为 413.634063，
  精确整数分子表达为 386.674275，均无 SG 收益；相关作者树和临时 leaf 已完整撤回。
- typed halfword 修正并非全项加速：K1 IQ2_XXS row 同源配对约 412.581729→360.623538，
  有真实退步但仍达标；保留这个 tradeoff，不用更旧的表值夸大或掩盖差异。

## 表格历史错位处理

change 起点的 current/comparison 已有 138 行不同步；收尾时仍有 95 行。
其中若干高值只更新了 comparison，缺少可恢复完整配置/计时记录的原 stdout。
没有从吞吐反推配置或 median：先以完整 current 记录对齐，标 `record-reconciled`，
随后对影响门槛的 7 条用普通入口重新实跑并同步两表，全部通过。
这类对齐不是新性能优化；未重测记录不标成新测。始终只有 baseline/current/comparison 三张 CSV。

## 验证、复现与边界

- SG2044 VLEN128、K1 VLEN256 沿既有真实 runner、原数值与计时协议分批运行；
  两机可并行，同机计时串行，编译和 JIT 加载不计入 kernel throughput。
- 最终编译器节点的 80 份保留测量加一个非 winner 的独立第二输入，共 81 份实际 C
  与重新 emit 一致，final 清 marker/统计及 pre-resource rematerialization/resume 两条路径
  的二次 diff 都为 0。本次 alias 收尾另核双机两份最新产物，同样通过。
  这是所重放流程的稳定性证据，不是所有 pass 的普遍幂等证明，也不是任意输入数值等价的形式证明。
- 更新后的 compiler 已在 SG 原生构建并同步双机；两机已有 native JIT 的 dense/encoded
  数值误差为 0，同绑定复用、不同绑定专门化通过。最终 IQ1_S 作者源码同步两机，
  本机前端/compiler 重新生成的 C 与对应 10 次计时的实际 C 一致。
- 相关 VLEN1024 样本及此前 IQ4_NL VLEN2048 已到合法 RISC-V object；IQ1_S 独立输出
  收尾也核了 VLEN1024。没有这些 VLEN 的硬件运行证据，不宣称已经实机覆盖。
- 原始 runner 日志、IR/C/ASM 在仓库外
  `/home/kingdom/.cache/weft/performance-gap.Uy1EUL/` 与
  `/home/kingdom/.cache/weft/nested-scale.lqTRQo/`。
  收尾日志为 `allpass-final-approved-iq1s-{sg2044,k1}.log`；
  对应 `.verify.*` 为重放产物，`allpass-approved-native-{rvv,k1}.log` 为本机 C 核对。

可从仓库根目录手动复现最后的正式修改：

```bash
WEFT_KEEP_ARTIFACTS=1 examples/run/weft-row-dequantize.sh sg2044 iq1_s 10
WEFT_KEEP_ARTIFACTS=1 examples/run/weft-row-dequantize.sh k1 iq1_s 10
```

其它代表性主链修改使用既有入口，例如
`examples/run/weft-quantized-vec-dot.sh sg2044 q4_k 10`、
`examples/run/weft-mul-mat.sh k1 q6_k prefill 10`。
不增加 test/fixture、兼容矩阵或新的验证脚手架；正式 Comet 验收由独立 Verifier 另行作出，
不以本文的自评替代。
