# Production 能力、IME 资源与 pass 候选空间

## 结论

本轮不做性能比较。审计得到的主要横向问题不是某个后端 pass 把大量候选判掉，
而是 **206 条 production 目标中有 194 条尚无对应的 `std` 数值树**。当前前端可以用
真实 storage 字段、现有布局关系、普通位运算和 `lookup` 写出这些格式；缺的是把每种
有限位宽数值分解写成同语言函数，而不是新增 194 条后端路径。

原始 206 条目标按当前代码归类为：

| 最早边界 | 条数 |
|---|---:|
| A：前端表达不了 | 0 |
| B：缺 `std` 特化 | 194 |
| C：缺 op lowering | 0 |
| D：pass 判非法 | 0 |
| E：能生成 intrinsic C | 12 |
| **合计** | **206** |

这里的 E 只表示当前唯一主链能够从 DSL source 生成 intrinsic C；本轮没有用历史 CSV
替代当前编译，也没有进行性能重测。206 条不是仓库中的可执行 manifest：它来自四个
production 族的目标清单，仓库当前只有 10 个手工 DSL kernel 入口。因此下文同时给出
“206 条目标覆盖”和“10 个当前入口的真实编译结果”，两种口径不能混用。

IME 的 28 也已经查清并修正。28 不是硬件公式，而是 target profile 中没有来源的
literal；它还被错误地永久叠加到整段程序的 value-liveness 峰值。当前 Weft M1 leaf
实际只固定占用 `v0/v14/v16` 三个物理向量寄存器。修正资源合同、时间域计费以及 packed
layout 消费后，`q4_k_q8_k_gemv_ime` 已经在 K1 上由唯一主链 bit-exact 执行。

## 一、矩阵口径

### 1.1 206 条是什么

固定 production 目标由以下四族组成：

| 家族 | 逻辑 case | 两台硬件后的目标行 |
|---|---:|---:|
| `MUL_MAT` | 26 formats × decode/prefill | 104 |
| quantized vec-dot | 24 typed pairs | 48 |
| activation quantize | 3 | 6 |
| row dequantize | 24 formats | 48 |
| **合计** |  | **206** |

当前 `report/weft-kernel-performance.csv` 还有 38 条 forward 与 6 条历史 IME alternative，
所以其中 `source_*` 一共有 250 行；这不改变原始 206 条 production 四族的定义。
CSV 是历史数字记录，不是当前 DSL manifest。当前可手动编译的 DSL source 只有
`examples/kernels/` 下 10 条。

### 1.2 A 与 B 的分界

A 只表示语言本身无法无歧义写出真实 storage 或数值树。一个 encoding declaration
尚未存在，不等于 A；能用现有语言写、但 `std` 中还没人写，就是 B。

当前公开且可 lowering 的 encoding 关系是：

```text
natural
grouped
layered
joined
```

对于 Q5/Q6/IQ 等分离 bit planes，当前不需要伪造一个未实现的 `bit_planes` 表面：
各 plane 可以先作为真实 storage 字段出现，`std` 函数再显式组合并执行 lookup/decode。
这正是 spec 3.1 在 `bit_planes` 尚未闭合时规定的表示。因此本轮调查的 24 个量化格式
没有一个必须落 A；它们都能写，但绝大多数尚未写。

## 二、206 条 production 能力矩阵

### 2.1 按家族展开

| 家族 | A | B | C | D | E | E 的当前来源 |
|---|---:|---:|---:|---:|---:|---|
| `MUL_MAT` 104 | 0 | 100 | 0 | 0 | 4 | f32 decode/prefill × SG/K1 |
| quantized vec-dot 48 | 0 | 46 | 0 | 0 | 2 | q4_K × q8_K × SG/K1 |
| activation quantize 6 | 0 | 0 | 0 | 0 | 6 | q8_0/q8_1/q8_K × SG/K1 |
| row dequantize 48 | 0 | 48 | 0 | 0 | 0 | — |
| **合计** | **0** | **194** | **0** | **0** | **12** | |

12 条 E 已分别在 SG2044/VLEN128 与 K1/VLEN256 profile 下重新走过：

```text
DSL source
→ canonical Kernel IR
→ current RISC-V pass sequence
→ intrinsic C
```

没有执行计时。

### 2.2 为什么 `MUL_MAT` 只有 4 条 E

`python/weft/std/dense.py` 当前只有 f32 GEMV/GEMM。它覆盖：

```text
f32 decode  × SG/K1 = 2
f32 prefill × SG/K1 = 2
```

f16 没有同语言的 staging/convert 特化。24 个量化 weight format 也没有完整 production
树：当前 q4_K 函数接收已经量化的 Q8_K activation，只对应 vec-dot helper，不包括
production `MUL_MAT` 中 activation quantize 和 outer traversal。因此其 production 行仍是 B，
不能借 q4_K helper 冒充 E。

典型 B：

- `mul_mat_f16`：树可写，但 `std` 没有 f16 dense 特化；
- `mul_mat_q4_0`：Q4_0 encoding 已有，完整 production 树没有；
- `mul_mat_iq2_s`：storage、bit composition 与 lookup 可写，encoding/function 均未落地。

### 2.3 quantized vec-dot

当前只有 `q4_K × q8_K` 的 wide 树进入主链；它在 SG 与 K1 target profile 下都能 emit。
其余 23 个 typed pair 都落 B。

典型 B：

- `q4_0_q8_0`：Q4_0/Q8_0 encoding 已有，缺数值分解函数；
- `q5_0_q8_0`：真实 `qs`/`qh` storage 可表达，缺组合与 MAC 树；
- `iq3_s_q8_K`：真实 planes 与 codebook 可表达，缺 lookup/decode/accumulate 树。

### 2.4 activation quantize 与 row dequantize

q8_0/q8_1/q8_K 三条 activation quantize 已有 `std` 函数，两个 target profile 均能 emit，
所以 6 条都是 E。

row dequantize 当前没有任何 `std` 函数。24 个格式在两台硬件上的 48 条目标全部先停在
B，尚未进入 op lowering 或资源检查；不能把 GGML 的 scalar reference 当作 Weft 路径。

## 三、当前 10 个 DSL 入口的真实编译边界

206 条目标没有 manifest，所以 C/D 需要从仓库里真实存在的 10 条 DSL source 观察。
修复前后如下：

| 当前 DSL kernel | 修复前 | 修复后 | 最早边界 |
|---|---|---|---|
| dense GEMM | E | E | — |
| dense GEMV | E | E | — |
| q4_K GEMV `mac_pairs` | E | E | — |
| q4_K GEMV `mac_groups(4)` | E | E | — |
| q4_K GEMV IME | D | E | 资源合同与 leaf layout 已修复 |
| q8_0 quantize | E | E | — |
| q8_1 quantize | E | E | — |
| q8_K quantize | E | E | — |
| flash attention | C | C | `materialize(admit(...))` 没有 target emission |
| Top-K | C | C | scalar `new` value 没有 target emission |
| **计数** | **C=2,D=1,E=7** | **C=2,D=0,E=8** | |

这里 E 从 7 变为 8。原始 206 的 E 仍是 12，因为修复的 IME specialization 是另行记录
的 K1 alternative，不属于原始四族中的独立目标行。

## 四、IME 的 28 来自哪里

### 4.1 修复前 target 声明与实际候选

修复前 `RISCVTargetProfile.cpp` 声明两个 capability：

| dtype | shape | fixed resource groups | emitter leaf |
|---|---|---:|---|
| i4 × i8 → i32 | M1 × N16 × K32 | 28 | 有 |
| i4 × i8 → i32 | M4 × N16 × K32 | 28 | 无 |

28 是直接写入 profile 的 literal，没有从 fragment shape、LMUL、clobber 或硬件描述计算，
仓库中也没有可支撑它的说明。selector 只按 operand/result SEW 返回首个匹配 fragment，
所以两个 shape 也没有形成两个可比较候选；emitter 又只接受 M1。修复前名义 candidate=2，
实际可发射 candidate=1。

修复后只发布当前真实存在的 M1×N16×K32 leaf，candidate=1。不存在的 M4 leaf 不再伪装
成 target capability。

### 4.2 当前 Weft leaf 的真实资源

当前 local asm 每次 `vmadot` 固定使用：

| register | vector shape | 物理寄存器数 | 角色 |
|---|---|---:|---|
| `v0` | e8,m1 | 1 | 4×8 packed weight window |
| `v14` | e8,mf4 | 1 | 1×8 activation |
| `v16` | e32,mf2 | 1 | 1×4 accumulator |
| **合计** |  | **3** | primitive-private clobber |

16 项 i32 output 先存在 local memory；helper 返回的 `i32m2` 是普通 result value，已经由
value liveness 另行计 2 groups，不能再塞进 fixed fragment cost。

### 4.3 修复前为什么得到非法

旧 resource pass 先在整个程序寻找 value 峰值，再无条件加上所有 matrix fragment：

```text
ordinal 39 的 epilogue value 峰值：
    h17  i32 accumulator / handoff chain       2 groups
    h45  min_term（dot result）                 2 groups
    h48  widened d                             2 groups
    h49  widened i32_acc                       2 groups
                                              --------
                                               8 groups

旧计算：8 + 28 fragment + 2 reserved = 38 > 32
```

但 matrix op 位于 ordinal 24；ordinal 39 时 fragment 已经结束。旧计算把不相交的两个
时间点叠在一起。若一个 kernel 有多个顺序 matrix op，旧实现还会把所有 fragment 永久相加。

在 matrix op 自己的 ordinal，真实 live set 是：

```text
h33  ui4[16,32] packed operand   1 group
h36  i32[16] result              2 groups
leaf fixed clobber               3 groups
reserved                         2 groups
                                --------
                                 8 groups
```

修复后资源按 operation ordinal 计算。全程序真实峰值仍是 ordinal 39 的 8 个 live-value
groups，加 2 个 reserved，共 10；该处 fragment cost 为 0。候选因此合法。

### 4.4 GGML donor 用的是什么

K1 GGML Q4_0 production kernel 不是当前 Weft local leaf：

- IME1 基础 `vmadot` fragment 是 4×4×8；
- M1 path 把多个基础 fragment 组合成 N16，并在一个整 kernel asm 中使用从 `v0` 到
  `v31` 的多组固定寄存器；
- M4 path 的局部主体覆盖 M4×N16×K32，同样把 decode、scale、累加和 store 包进整段 asm；
- 28.4 GOP/s 对应 Q4_0、M=128 的 M4 prefill donor，不是当前 Weft M1 local leaf。

因此不能把 donor 整 kernel 的近满寄存器占用直接写成每次 local primitive 的 fixed cost。
donor 证明 IME 可用，也提供指令组织知识；它不提供当前 Weft leaf 的资源合同。

### 4.5 数值错误与 layout 决策归属

资源放通后的第一次 K1 repro 数值错误。原因是 emitter 中的 IME helper 又把 Q4_K q
当成线性 `logical * 4` nibble，而前序 memory-edge 已经决定了真实
`grouped(64) + layered(32, lo_first)` mapping。

修复后：

1. `SelectLocalOperations` 沿普通 def-use 穿过 `extract` 找到 packed field；
2. 它把 `lhs_group_size=64`、`lhs_layer_size=32`、`lhs_layer_order=lo_first` 写入当前
   matrix local-operation decision；
3. emitter 只读这些值，按已选 mapping 生成 local leaf，不按格式名判断。

K1 手工 repro 最终报告 `numeric=bit-exact`。本轮没有记录或比较该运行产生的时间数字，
性能 CSV 也没有改动。

## 五、每个 pass 的真实候选数

“代码里有一个字段”不等于“编译器正在选择”。当前实际情况是：除最外层 `auto`
笛卡尔积机制外，所有 physical pass 都是单候选确定化。

| pass / 决定 | 理论输入数 | 当前 production/std 的实际候选数 | 是否真的选择 |
|---|---:|---:|---|
| `ConstructProblems`: `std` specialization | 由前端选定 | 1 | 否；当前调用点已展开为一棵树 |
| `ConstructProblems`: `auto` bindings | `∏ choices_i` | 1 | 当前 std 只有符号 auto，每次命令只绑定一个值 |
| encoding mapping | 每个 field-use | 1 | deterministic projection |
| lane axis | 多个逻辑轴可能存在 | 1 | rows/cols 优先，取一个 |
| SEW | target 支持 8/16/32/64 | 1 | `max(8, logical_sew)` |
| LMUL | 7 个合法 LMUL facts | 1 | 取第一个能容纳 lanes 的值 |
| `vl` / stream parts | shape/target facts | 1 | 公式直接产生 |
| materialization | use/liveness facts | 1 | shared 或 reload 的固定判据 |
| ordinary local op instruction | target facts | 1 | 每个 op/engine 一个固定 realization |
| memory form | target 可支持多种 | 1 | admit/commit 固定 unit-stride；field unpack 固定映射 |
| matrix fragment | 修复前 profile 2、可发射 1 | **修复后 1** | 无枚举/排序 |
| Level mapping | axis/value facts | 1 | lane/stream/sequential 的确定投影 |
| unroll | — | 1 (`1`) | 硬编码 |
| pipeline depth | — | 1 (`1`) | 硬编码 |
| prefetch distance | — | 1 (`0`) | 硬编码 |
| resource check | 1 个完整 candidate | 接受或拒绝 | 不产生候选 |
| winner | 外层 source candidates | 当前为 1 | 有排序代码，但当前没有可比较项 |

所以当前项目已经有“前向 pass 序列”，但还没有真实的 physical candidate space：

```text
7 个 legal LMUL facts ≠ 7 个 LMUL candidates
target 支持 unit/strided/indexed/segment ≠ memory pass 正在四选一
schedule 有 unroll/pipeline/prefetch 字段 ≠ schedule pass 正在选择
```

206 条矩阵给出的优先级也很直接：当前最宽的阻塞是 B（194 条），而不是某个单候选
physical pass 判掉大量已经存在的树。补 physical candidates 能改善现有 12 条 E 的质量，
但不能让缺失的 194 棵 `std` 数值树凭空出现。

## 六、本轮代码收敛

本轮只改与 IME 审计直接相关的五处：

- target profile 只发布真实可发射的 M1×N16×K32 fragment，并用实际 leaf clobber 计 3 groups；
- resource pass 按 operation ordinal 计 fragment，不再永久叠加；
- physical assignment 打印 handoff class 与 live interval；
- local-op selection 物化 IME packed operand 的 grouped/layered mapping；
- IME emitter 机械消费该 mapping，删除线性 nibble 假设。

没有新增 fallback、格式 route、whole-kernel emitter 或第二条编译路径。
