# 从 op 驽物理映射到 baseline 全量闭环：四轮推进报告

## 报告范围

本报告记录以下四轮实际推进：

1. **纠正编译模型，建立唯一的 op 驽物理映射主干**；
2. **从物理映射生成高性能微内核、数据复用和自动流水**；
3. **完成全部现有 DSL kernel 的统一高性能编译空间**；
4. **只对有 baseline 的项目全量运行，并修正全部暴露问题**。

对应代码范围为：

```text
cf78782c3 select local hardware operations in op rules
...
48b891abb Record final source baseline run
```

报告只描述上述提交、当前源码以及第四轮真机日志已经形成的事实。它不是新的规范；当前规范仍是
`doc/index.md`及其链接文档，当前性能数字仍以`report/weft-kernel-performance.csv`为准。

## 总结论

这四轮完成的核心变化不是“又支持了一批 kernel”，而是把 Weft 的 RISC-V 编译方式从：

```text
primitive/family
→ 识别一份人工高性能结构
→ 填入LMUL、microtile等参数
→ emitter继续补充判断
```

重构为：

```text
显式 DSL op 的语义规则
+ logical axis / use-def / memory / lifetime facts
+ target facts
→ shared physical mapping
→ reuse-driven local schedule
→ unified resource filtering
→ selected local hardware operation
→ mechanical intrinsic C / local asm emission
```

最终状态有三项可以确认：

- 生产代码只保留一条 Kernel IR 到 RISC-V intrinsic C / IME asm 的主链；
- 第二次 baseline 全量运行在 SG2044 RVV、K1 RVV 和 K1 IME 上完成 **250/250**；
- F32 matmul、Q3 vector partial、Q4/Q5 decode work 等共享能力产生了明显性能改善。

同时也有三项不能夸大：

- F16 matmul、Q2、部分 codebook quant、IME prefill、FlashAttention 和 K1 transpose仍有明显差距；
- 当前 candidate space 与 cost model 仍不等价于 Triton 级成熟度；
- 第三轮完成的是 production authority 的全 family 迁移，最终真机全量只覆盖第四轮定义的
  baseline 子集，不等于所有无 baseline example 都在第四轮重新运行。

---

## 四轮之前的问题

Core-local blocked DSL、轴身份、普通 block SSA、storage ownership 与作者/target 边界此前已经建立。
本轮开始时，真正的问题位于 target lowering：

1. local primitive 已经拥有 typed axis facts，但 planner 仍会在 mapping 形成后，用一组
   `is*LocalImplementationMapping` 函数反向识别 leaf；
2. dense、quant、state 与 memory 各自维护部分 structure selection、schedule 和 resource 逻辑；
3. register microkernel、RVV strip、IME fragment 虽然是 target 结果，却仍容易被当成上层实现入口；
4. reuse、operand window、pipeline 与 lifetime 没有形成一条贯穿 value chain 的共同信息流；
5. emitter仍承担部分 shape、unroll、decode 或 leaf 完整性判断。

因此，旧实现虽然已经不再按 kernel 名选择 whole-kernel emitter，但仍没有完全摆脱“先匹配一个
高性能结构，再填参数”的方法。

---

## 第一轮：纠正编译模型，建立唯一的 op 驽物理映射主干

### 本轮提交

| 提交 | 作用 |
|---|---|
| `cf78782c3` | 让显式 op rule 直接选择 local hardware operation |
| `a495e5e8c` | 让 typed op rule 直接贡献 physical mapping 与 quant rule |

### 1. 明确“统一什么、不统一什么”

这一轮没有建设一个从普通 SSA 图自动恢复算法的万能求解器。

保留的 op-specific semantics 包括：

- pointwise 保持逻辑元素与轴映射；
- load/store 提供 pointer-axis relation；
- cast/widen/narrow 保持逻辑元素并改变 SEW；
- reduce/scan/state 提供消去轴、顺序与 carry 约束；
- `W.dot/W.matmul`提供 free/reduction axis 与数值关系；
- quant、lookup、decode 与 extension primitive保留普通SSA无法无歧义恢复的局部语义。

真正统一的是：

```text
sequential
lane
register
unroll
fragment
```

这组 core-local 物理坐标，以及沿 value chain 的 shape、schedule、resource 与 handoff 表达。

### 2. Hardware operation 改由 op rule 显式产生

`cf78782c3`删除了 planner 中大量“拿最终 mapping 反查 leaf”的入口，包括旧的 packed dot、
nibble codebook、signed codebook、IQ1/IQ2/IQ3 与 Q6 mapping predicate。

新主干由以下职责组成：

```text
selectLocalHardwareOperation(...)
finalizeLocalHardwareOperation(...)
LocalImplementation.operation
```

每个 typed primitive rule 在构造合法 candidate 时同时产生：

- local primitive identity；
- local hardware operation kind；
- physical mapping；
- projection；
- value shapes。

Emitter 不再根据 mapping 形状反向猜 leaf，而只消费已经写入
`LocalImplementation.operation`的结果。

当前保留的：

```text
LocalHardwareOperationKind::RVVIntrinsic
LocalHardwareOperationKind::RVVInlineAsm
LocalHardwareOperationKind::MatrixFragment
```

是瞬态 target leaf，不是 DSL 构造，也不是 whole-kernel route。

### 3. Quant rule 由局部数值语义贡献 mapping 约束

`a495e5e8c`把 packed quant 的选择整理为 typed rule builder，例如 packed I4/I5、packed I3、
IQ1/IQ2/IQ3 与 Q6。Rule只描述：

- 当前 primitive 的 logical axis role；
- packed/decode 数值关系；
- value shape 与 live temporary；
- 可接受的 operation/projection。

Shared planner 再枚举 axis mapping、检查 target legality并形成 implementation。旧的做法是先生成
mapping，再在另一个 switch 中重新推断 operation/projection；这份重复 authority 被删除。

同一轮还用 `LogicalAxisRole` 查找 free/reduction axis，减少依赖固定轴编号或 operand 位置的逻辑。

### 4. 第一轮形成的编译判据

本轮之后，一个 local implementation 必须能回答：

```text
哪个显式 primitive 授权了它；
各 logical axis 映射到 sequential/lane/register/unroll/fragment 的哪一部分；
使用哪个局部硬件 operation；
输入输出采用什么 physical value shape。
```

它不能通过以下信息回答：

```text
kernel 名；
example 名；
q-format route；
VLEN128/VLEN256 实现身份；
完整外围 loop closure。
```

### 5. 本轮没有删除的必要差异

Quant emitter仍按显式 primitive 分发到最小 typed leaf。这不是旧 family route：不同 primitive
确实拥有不同局部数值语义与 intrinsic/asm spelling。被删除的是“一个格式拥有完整 traversal、
microkernel skeleton与资源公式”，不是 primitive 本身。

---

## 第二轮：从 physical mapping 生成 microkernel、复用与流水

### 本轮提交

| 提交 | 作用 |
|---|---|
| `c6130260a` | 从 mapping 生成统一 local microkernel schedule |
| `db1056d5e` | 按 operation lifetime 计算 VLA entity resource |
| `1aeb3e64f` | 共享 quant mapping 与 resource candidate builder |
| `4a0a5d7e0` | F16 matmul 的 column/reduction lane pipeline |
| `fd8e3950f` | 统一 indexed 与 segment memory groups |
| `e9ffc9a1f` | 建立真实 compile-and-measure 入口 |

### 1. Mapping 开始生成真实 microkernel schedule

`c6130260a`增加并统一使用：

```text
LocalOperandWindow
LocalPipelineAction
LocalPipelineSchedule
LocalDecodeSchedule
LocalMicrokernelSchedule
```

`buildLocalMicrokernelSchedule()`从最终 axis mapping 直接产生：

- sequential iteration 数；
- SIMD lane factor；
- register repetition 与 accumulator 数量；
- K-unroll；
- operand window及每次load服务多少accumulator；
- buffer count与prefetch distance；
-明确的Load/Compute action序列；
- decode chunks/groups/reduction segments。

这使得 VLA dot、local F32 dot、matmul 与 quant leaf 不再各自手写一份 unroll/load loop。
Emitter消费同一份 pipeline action，而不是看到 GEMM 或 quant 名字后进入专用流水。

这里的 pipeline 是 author-written loop 内部的 register load/compute schedule。它不创建新的算法
pass、跨 loop workspace、异步 DMA 或 persistent buffer。

### 2. Resource 与 operation lifetime 对齐

`db1056d5e`把 VLA resource 从粗粒度 phase 汇总改为 operation-ordinal snapshot：

- 每个 value 保存 definition/last-use 与 control carry；
- narrow、lookup、state、indexed/segment memory 和 local primitive绑定实际 operation位置；
- nested scalar control继承外层仍存活value；
- primitive temporary只在其真正执行的 snapshot 计入峰值。

这删除了 `hasNarrow`、`lookupCount` 等无精确生命周期来源的粗粒度预算。

### 3. Quant family 共用 candidate construction

`1aeb3e64f`建立共享 candidate builder：

```text
typed quant rule
→ enumerate axis mappings
→ build local schedule
→ finalize local operation
→ calculate axis-mapped resources
→ legal candidate ordering
```

格式差异只留下：

- packed bit/codebook 数值规则；
- value shape；
- projection；
- minimal local operation。

旧 selector 中重复的 candidate struct、mapping enumeration、schedule generation与resource wiring
被集中删除。

### 4. F16 matmul 获得两种 lane mapping 与真实 register pipeline

`4a0a5d7e0`让同一个 F16 matmul planner可以生成：

- column-lane mapping；
- reduction-lane mapping；
- 1/2/4 K-unroll；
- 单 buffer或合法的register multi-buffer；
- RHS/LHS register bank与Load/Compute交错。

`f16-lane-axis`、`f16-k-unroll`与`f16-load-buffer-count`只实例化已经合法的结构，不进入 DSL。

### 5. Indexed 与 segment memory 由 facts 统一分组

`fd8e3950f`把原来 compiler decision 内部的两两扫描移到 `RISCVKernelFacts`：

- indexed accesses 按 coordinate、offset SSA、element bytes 与 block scope 分组；
- interleaved accesses 按 invariant base、field、scale与effect顺序形成 segment candidate；
- planner只消费这些 facts 判断 indexed/segment legality；
- emitter只消费最终 memory form与共享offset emission。

跨 block、复杂 alias/effect或无法证明的 non-affine relation仍不会被错误合并。

### 6. 构建期实测选择成为真实入口

`e9ffc9a1f`增加 `examples/run/tune.py`：

```text
显式backend dimensions
→ 笛卡尔实例化
→ 同一weft.sh主链生成intrinsic C
→ 目标C编译器编译
→ SG2044/K1真实运行
→ 只在合法且runtime成功的候选中选择metric winner
```

Tuner不生成新结构，不把 winner写入 Kernel IR，不保存格式最优表。它只是对 compiler 已经能够
生成并证明合法的 candidate重复编译和实测。

### 7. 第二轮仍有的边界

- Lifetime snapshot仍是 operation-level resource class，不是完整 SSA live interval；
- pipeline深度受实际K-unroll和register budget限制；
- 没有真实第二个实现的preload/prefetch维度没有被伪装成已支持；
- quant rule仍需逐 primitive 提供真实局部数值关系。

---

## 第三轮：完成所有 family 的统一编译空间

### 本轮提交

| 提交 | 作用 |
|---|---|
| `176ab2572` | 抽出共享 reuse/schedule dependence analysis |
| `8283223ce` | 文档同步统一 physical planning |
| `828bb8b9d` | 删除 quant resource 旁路 |
| `216772070` | 形成 VLA invariant address hoist schedule |
| `419094630` | extension fragment/asm 共用 resource calculator |
| `ea6fa3de1` | 记录并冻结统一 planning 结果 |

### 1. Reuse 不再由各 dense emitter 分别推导

`RISCVReuseAnalysis`从同一份 `KernelPhysicalFacts` 统一分析：

- operand是否随reduction axis推进；
- memory mode是unit/strided/indexed中的哪一种；
- address/predicate是否依赖accumulator；
- result与operand的consumer count；
- value是否跨control或loop carry。

F32 dot、VLA dot与F16/F32 matmul共同消费这份 dependence facts。外围程序从 GEMM 换成 Conv、
MoE、Attention或OutProd时，只要 primitive与use/memory事实相同，就不会因为源码邻接改变能力类别。

### 2. Quant resource 旁路被删除

`828bb8b9d`删除 quant candidate 的独立 `directResources` 与重复预算公式。Packed Q4/Q5/Q6、
IQ1/IQ2/IQ3和codebook gather都通过：

```text
AxisMappedResourceFacts
→ calculateAxisMappedResources
→ calculatePhysicalResources
```

固定 asm register group也作为 `AxisMappedLiveValue` 或 reserved group进入同一预算，不再隐藏在
emitter常数中。

### 3. Extension fragment进入同一 resource authority

`419094630`把 I4×I8 RVV/IME fragment与 grouped-affine I4×I8 的 accumulator、operand、temporary、
fixed fragment/register set纳入同一个axis-mapped resource calculator。

IME因此是相同 local primitive的一种合法 operation，不拥有outer traversal、workspace、persistent
layout或entry ABI。

### 4. VLA invariant address hoist 成为 facts 与 schedule

`216772070`识别 VLA body 中只依赖外层值的纯地址链：

```text
constant / binary / cast / expand-dims / ptr-add
```

Facts证明其不依赖coordinate且不跨越nested control后，planning才形成hoist schedule。Emitter在
保留空extent guard的条件下先发射这些operation，再发射strip loop；它不重新扫描相邻地址表达式。

### 5. 第三轮完成后的模块责任

```text
RISCVKernelFacts
  axis / value / use / memory / effect / lifetime

RISCVReuseAnalysis
  operand advancement / sharing / control crossing

RISCVAxisMapping
  sequential / lane / register / unroll / fragment decomposition

RISCVPhysicalPlanning
  schedule / value shape / local operation / resource / candidate selection

RISCVKernelCompiler
  transient value-chain decisions与handoff，随后生成普通C控制流

RISCVRVVIntrinsicC / RISCVQuant*IntrinsicC / RISCVIMEIntrinsicC
  只拼写已选intrinsic或local asm
```

### 6. “全部现有 kernel 统一”具体指什么

这句话指 production authority：dense、state、memory、quant/codebook与RVV/IME extension不再保留
第二套 family selector、resource公式或旧emitter主干。

它不表示第三轮重新全量运行了所有 example。第三轮按共享能力运行受影响的少量真实repro；第四轮
才对有 `source/` baseline 的固定集合进行两次完整运行。没有 baseline 的 example没有在第四轮占用
目标机时间。

### 7. 第三轮结束时仍明确的限制

- local dot/matmul/quant峰值尚未与外围VLA snapshot合成全局live interval；
- nested VLA仍明确unsupported；
- runtime-extent二维block通用materialized load与二维到一维ordered reduction没有artifact；
- candidate合法性已统一，但候选宽度与cost排序仍不均衡。

---

## 第四轮：只对有 baseline 的项目全量运行并修正问题

### 本轮提交

| 提交 | 作用 |
|---|---|
| `7ff2b72a4` | 统一 block value-chain lowering，关闭第一遍42个真实失败 |
| `2d6ccf671` | 平衡 wide-VLEN F16 matmul reuse |
| `2de0c7332` | grouped quant partial保存在vector state中 |
| `72c1de027` | F16/F32 matmul共用一份mapping/planner/emitter skeleton |
| `2a1eb98f3` | quant candidate优先减少重复decode工作 |
| `48b891abb` | 写回最终250项性能并同步现行文档 |

### 1. Baseline 对应集合

第四轮没有运行所有 examples，只运行 CSV 中能够和 `source/` baseline准确对应的项目：

| 类别 | 条目数 | 说明 |
|---|---:|---|
| `source_mul_mat` | 104 | SG/K1，decode/prefill，dense与quantized projection |
| `source_mul_mat_ime` | 6 | K1 IME，Q4_0/Q4_1/Q4_K decode/prefill |
| `source_vec_dot` | 48 | SG/K1量化row-dot microkernel |
| `source_quantization` | 6 | SG/K1 Q8_0/Q8_1/Q8_K activation quantize |
| `source_dequantization` | 48 | SG/K1完整row dequantize集合 |
| `source_forward` | 38 | pointwise、normalization、layout、gather、RoPE、attention |
| **总计** | **250** | SG RVV 121 + K1 RVV 123 + K1 IME 6 |

每个条目都固定：

- algorithm与DSL kernel；
- hardware与target profile；
- shape、dtype/quant format与phase；
- preprocessing是否在timed region；
- repetitions、warmup与计时范围。

`source/`不进入Kernel IR lowering、生成的`kernel.c`或Weft production entry。部分repro runtime会从
`source/`复制`ggml-common.h`并链接目标机GGML库，在timed region之外计算数值reference；这只是
correctness对照，不是Weft实现、fallback或被计时的kernel路径。

### 2. 运行清单自身先暴露了两类非编译器问题

第一次 orchestration 中出现：

- CSV 中 `iq1_m/iq2_s/iq3_s` 与 runner 中大小写 entry 不一致，在两台机器形成6次错误调用；
- 6个IME条目最初使用了错误profile。

处理方式是修正一次性manifest映射，而不是在`weft.sh`增加alias或compatibility layer：

```text
iq1_m_q8_K → iq1_M_q8_K
iq2_s_q8_K → iq2_S_q8_K
iq3_s_q8_K → iq3_S_q8_K

mul_mat_q4_0 → q4_0_projection_ime
mul_mat_q4_1 → q4_1_projection_ime
mul_mat_q4_K → q4_k_projection_ime
```

同样，CSV中的上游名字 `mul_mat_f32/mul_mat_f16`只映射到仓库已有
`blocked_gemm_f32/blocked_gemm`运行入口；它们不是新的编译器route。

### 3. 第一遍完整运行暴露42个真实编译器失败

去除上述manifest错误后，标准RVV集合为：

```text
202 PASS
42 real compiler/codegen FAIL
```

再加上正确profile下已经通过的6个IME条目，第一遍整体状态为208/250通过。

42个失败在两台目标上以21+21对称出现，总计仍是42条；根因不是42个kernel case，而是四组共享缺陷：

| 共享根因 | 数量 | 典型现象 |
|---|---:|---|
| Block reduction读取了独立SEW/mapping | 6 | Q2 matrix decode/prefill与vec-dot报告input mapping不一致 |
| Grouped affine leaf identity与selected mapping脱节 | 6 | Q4_K找不到已选register leaf定义 |
| Decode/index handoff存在第二份authority | 6 | NVFP4生成空index operand或非法表达式 |
| Block lane、decode与indexed value-chain未统一 | 24 | 多种dequant出现undeclared lane或no legal block implementation |

### 4. `7ff2b72a4`用一项共享修复关闭42个失败

这次修复没有增加 dequant/Q2/Q4/NVFP4 特例，而是重建 block entity 的单一决定关系：

- block decode不再保存独立 `physicalPlan.blockDecodes`；
- decode decision进入所在 `BlockOperationDecision`；
- store/reduce/decode共享同一value-chain entity与physical value records；
- block reduction先探测producer-selected input shape，再决定reduction realization；
- lane vector只在实际需要它的operation位置按selected shape物化；
- data-derived index relation明确成为`Indexed`，不再退化成无法发射的`NonAffine`；
- decode与store/reduce resource合并到同一entity budget。

因此，q2 reduction、q4_K leaf、NVFP4 index与24个dequant失败一起消失。Emitter只查找
value-chain中的operation decision，不再从当前C spelling补建lane或decode shape。

### 5. 第一遍性能推动的四项共享改进

#### 5.1 Wide-VLEN F16 matmul reuse

`2d6ccf671`根据target lane capacity与register reuse重新平衡F16 candidate preference。K1最终F16
prefill由上一份记录的4267.736 ms降到3755.370 ms；改动只位于共享matmul planner。

#### 5.2 Grouped Q3 partial保持vector state

`2de0c7332`不再在每个decode segment后立即做scalar reduction，而是：

```text
decode/widen product
→ scale-weighted i32 vector accumulation
→ primitive末尾一次reduction
```

新增vector accumulator同时进入resource budget。最终Q3 row-dot：

| 目标 | 旧记录 | 最终 | source baseline |
|---|---:|---:|---:|
| SG2044 | 40.647 ms | 22.452 ms | 15.894 ms |
| K1 | 66.766 ms | 44.194 ms | 48.107 ms |

K1已经略快于source baseline；SG仍慢约41%。

#### 5.3 F16/F32 matmul共用同一 mapping

`72c1de027`完成以下统一：

- `W.matmul`允许两侧dtype相同的F16或F32 multiplicand，accumulator保持F32；
- `F16Matmul*` decision/planner泛化为 `MatmulDecision/MatmulCandidateFacts`；
- F16使用widening FMA，F32使用native FMA；
- mapping、microtile、K-unroll、pipeline、resource与emitter skeleton共用；
- `blocked_gemm_f32.py`从逐列rank-one dot改为显式BM/BN/BK二维`W.matmul`程序。

最明显结果是K1 F32 prefill：

```text
旧记录 5205.112 ms
→ 最终 1563.137 ms
→ K1 source baseline 1606.776 ms
```

即最终比source baseline快约2.8%，而不是依赖F32 kernel专用路径。

#### 5.4 Quant mapping优先减少重复decode

`2a1eb98f3`在合法 mapping 中优先更少register chunks与重复decode工作，同时保留资源合法性。

代表结果：

| Kernel | 目标 | 旧记录 | 最终 | source baseline |
|---|---|---:|---:|---:|
| Q4_0×Q8_0 | SG | 81.730 ms | 44.659 ms | 47.504 ms |
| Q4_0×Q8_0 | K1 | 119.486 ms | 59.084 ms | 52.794 ms |
| Q5_0×Q8_0 | K1 | 120.114 ms | 59.947 ms | 64.756 ms |

这不是所有quant都同步改善。Q2在修复正确性后选择的合法实现仍很差：

| 目标 | 旧记录 | 最终 | source baseline |
|---|---:|---:|---:|
| SG Q2_K×Q8_K | 34.466 ms | 73.851 ms | 12.617 ms |
| K1 Q2_K×Q8_K | 136.091 ms | 164.273 ms | 48.013 ms |

这说明 shared candidate space已经统一，但candidate质量和cost排序仍未完成。

### 6. 第二遍最终全量运行

第二遍使用同一个 commit、同一份DSL、同一shape与协议，目标内严格顺序执行，SG与K1两台机器
之间并行；K1 IME在K1 RVV结束后单独执行。

最终结果：

```text
SG2044 RVV: 121 / 121 PASS
K1 RVV:     123 / 123 PASS
K1 IME:       6 /   6 PASS
--------------------------------
total:       250 / 250 PASS
```

仓库外的临时运行目录`/tmp/weft-r4-second.8AcmAL`中，250份日志均具有唯一CSV row marker与
`R4_EXIT=0`；没有compile failure、unsupported、runtime failure或数值mismatch。日志没有进入Git；
其最终动态数字写回`weft-kernel-performance.csv`第151–400行。前149条历史pressure记录和所有
静态protocol列保持不变。

---

## 最终性能分析

### 1. Dense F32/F16

| Kernel | 目标/phase | 旧记录 | 最终 | source baseline | 判断 |
|---|---|---:|---:|---:|---|
| F32 | SG decode | 14.902 ms | 16.383 ms | 20.612 ms | Weft约1.26× baseline速度 |
| F32 | SG prefill | 847.535 ms | 771.829 ms | 574.193 ms | 仍慢34% |
| F32 | K1 decode | 22.107 ms | 20.256 ms | 16.715 ms | 仍慢21% |
| F32 | K1 prefill | 5205.112 ms | 1563.137 ms | 1606.776 ms | Weft约快2.8% |
| F16 | SG decode | 12.058 ms | 14.036 ms | 7.387 ms | 约1.90× baseline时间 |
| F16 | SG prefill | 743.040 ms | 702.321 ms | 299.138 ms | 约2.35× baseline时间 |
| F16 | K1 decode | 39.967 ms | 39.518 ms | 9.785 ms | 约4.04× baseline时间 |
| F16 | K1 prefill | 4267.736 ms | 3755.370 ms | 782.657 ms | 约4.80× baseline时间 |

结论：F32统一matmul主线已经证明有效，尤其K1 prefill；F16虽然共享planner与pipeline，但显式F32→F16
workspace staging、operand reuse、microtile与load/compute overlap仍远未闭合。

### 2. Quant vec-dot

| Kernel | 目标 | 最终 | source baseline | 判断 |
|---|---|---:|---:|---|
| Q3_K×Q8_K | SG | 22.452 ms | 15.894 ms | 仍慢41% |
| Q3_K×Q8_K | K1 | 44.194 ms | 48.107 ms | 快约8% |
| Q4_0×Q8_0 | SG | 44.659 ms | 47.504 ms | 快约6% |
| Q4_0×Q8_0 | K1 | 59.084 ms | 52.794 ms | 慢约12% |
| Q5_0×Q8_0 | SG | 44.050 ms | 18.514 ms | 约2.38× baseline时间 |
| Q5_0×Q8_0 | K1 | 59.947 ms | 64.756 ms | 快约7% |
| Q4_K×Q8_K | SG | 12.687 ms | 13.009 ms | 略快 |
| Q4_K×Q8_K | K1 | 52.561 ms | 45.462 ms | 慢约16% |

同一共享改动在不同target/format上的收益不同。性能不再由“有没有对应leaf”决定，而由decode chunk、
lane/register factor、vector partial、gather与reduction候选质量决定。

### 3. K1 IME

| Kernel | Phase | Weft | source IME baseline | 判断 |
|---|---|---:|---:|---|
| Q4_0 | decode | 3.181 ms | 2.941 ms | 慢约8% |
| Q4_0 | prefill | 226.679 ms | 151.069 ms | 慢约50% |
| Q4_1 | decode | 3.989 ms | 3.519 ms | 慢约13% |
| Q4_1 | prefill | 245.305 ms | 174.724 ms | 慢约40% |
| Q4_K | decode | 4.011 ms | 3.436 ms | 慢约17% |
| Q4_K | prefill | 245.344 ms | 174.758 ms | 慢约40% |

IME leaf已经完全通过唯一主链生成并执行，decode接近baseline；prefill仍缺fragment输入复用、局部packing
与pipeline质量。IME没有拥有完整GEMM traversal或另一条compiler主干。

### 4. Quantization、dequantization 与 forward

已经达到或超过source baseline的代表项：

- SG `quantize_q8_0`：2.776 ms，对比3.292 ms；
- K1 `quantize_q8_0`：5.024 ms，对比5.560 ms；
- SG `dequantize_q4_K`：5.683 ms，对比5.837 ms；
- K1 `dequantize_q4_K`：10.136 ms，对比25.133 ms；
- SG/K1 `get_rows`：0.764/0.920 ms，对比0.825/3.362 ms；
- SG/K1 softmax：4.107/6.339 ms，对比4.170/8.349 ms。

仍明显落后的代表项：

- SG SiLU：6.903 ms，对比4.113 ms；
- K1 contiguous transpose：20.522 ms，对比4.251 ms；
- SG/K1 FlashAttention：32.267/61.429 ms，对比21.667/44.422 ms。

这些差距没有用kernel特例遮住。它们分别指向math helper、memory schedule/window reuse、summary/state
placement与attention local reuse等共享能力。

---

## 四轮后实际保留的唯一主干

当前生产信息流为：

```text
Core-local blocked DSL
→ canonical Kernel IR
→ RISCVKernelFacts
→ op-specific compile rules
→ RISCVAxisMapping
→ RISCVReuseAnalysis
→ RISCVPhysicalPlanning
   mapping + schedule + resource + selected operation
→ transient value-chain plan与handoff
→ ordinary C control/address generation
→ RVV intrinsic / quant leaf / IME local asm spelling
→ system C compiler
```

生产代码中已经不存在：

- `RVVStrip`、`RegisterMicrokernel`、`IMEFragment`作为高层入口；
- lhs-VLA/rhs-VLA/local-row作为whole-structure selector；
- kernel-name、format-name、VLEN-name驱动的完整实现route；
- legacy/scalar/GGML/materials fallback；
- emitter阶段重新选择LMUL、microtile、decode chunk或fragment。

仍存在的 `LocalPrimitiveKind`、`LocalHardwareOperationKind`与最小quant/IME emitter是必要的局部typed
leaf。Persistent `storageFormat`也仍是caller与kernel共享的ABI identity；它不参与whole-kernel route
选择。

---

## 尚未闭合的问题

### 1. Dense candidate 与 cost

F16在两台机器上都远落后baseline。当前已有column/reduction lane、microtile、unroll与buffer候选，
但缺少更强的cross-row/cross-output reuse、load scheduling、workspace staging amortization与目标相关cost。

### 2. Quant candidate质量不均衡

Q3、Q4_0与K1 Q5_0证明共享mapping可以明显改善；Q2、SG Q5_0、部分IQ/MXFP4则证明共享框架本身不
等于高质量candidate。Decode/gather/widen/scale/correction/accumulator之间仍需更完整的组合空间。

### 3. IME prefill

IME decode已接近baseline，但prefill仍慢40%–50%。问题位于primitive-local fragment input reuse、
packing与pipeline，不应通过恢复完整IME GEMM路径解决。

### 4. Attention 与 irregular memory

FlashAttention仍慢38%–49%，K1 transpose仍慢约4.8倍。当前facts已经能表达state与memory relation，
但window reuse、coordinate reuse、load ordering与state placement候选还不够成熟。

### 5. Resource model

VLA entity与local dot/matmul/quant各自拥有真实峰值预算，但尚未合成完整全局live interval。
Nested VLA、runtime-extent二维materialized block与部分ordered block reduction仍明确unsupported。

### 6. 代码与文档残留

`RISCVKernelCompiler.cpp`仍然很大，虽然authority已前移，普通C控制流、value-chain plan消费与多种
local spelling仍集中在同一实现文件。

此外，当前`doc/kernels/gemm.md`的F32小节仍保留“先按lhs-VLA/rhs-VLA/local-row确定structure”的
旧描述，与本轮op-driven shared mapping的现行代码不一致。这是文档残留，不代表生产代码仍保留
旧selector。

---

## 最终判断

这四轮之后，Weft 已经不再是一组按 family 选择高性能模板的 emitter。它现在能够从作者显式写下的
Core-local blocked program中，依据op语义、logical axis、value chain、memory relation与lifetime，
组合地产生RVV/IME局部实现；所有250个baseline条目已经通过这条唯一主链生成、编译并真机执行。

但“250/250能跑”只证明主链与正确性覆盖已经闭合，不证明性能空间已经成熟。真正完成的部分是编译
方法与authority的重构；仍需继续成熟的是candidate质量、resource/cost、reuse与pipeline，尤其是
F16、Q2、IME prefill、attention和K1 irregular memory。
