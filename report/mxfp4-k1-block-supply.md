# MXFP4 K1 block supply

日期：2026-09-05

## 实现前七步账

1. 作者程序已有 32-element K block、packed nibble Encoding、codebook lookup、i32
   contraction 和每 block 独立的 f32 scale。相邻 block 的 scale 不同，不能跨 block 合并
   reduction 或改变浮点加法顺序。
2. 当前 standalone/decode/prefill 分别为 source 的 76.67%/79.36%/80.53%。三条共享
   同一 32-element contraction；donor 每次展开两个相邻 block，但仍分别形成 `sum1/sum2`、
   两次 reduction 和两次 f32 scale accumulation。
3. codebook 是调用期不变的 16-element输入。prefill 已在 K loop 外 materialize；standalone/
   decode 没有显式 lifetime，因而先验证它是否在每 block 重载。合法删除上限是每次调用
   从 `K/32` 次 table supply 降到 1 次；Q8/weight block load、两次独立 reduction均不能删。
4. memory form 已是 packed unit load + nibble split + table gather；本轮不新增格式 leaf。
5. `unroll=2` 只机械展开两个独立 block issue，保持各自 reduction/scale和 K 顺序。若当前
   generic unroll 能形成 donor 的相邻 issue schedule，它属于参数绑定；不能把两次 reduction
   合成一次。
6. 可见预测：table materialize 后 standalone/decode 的 block-loop table load消失；unroll2
   减少 loop/control overhead，但数学 MAC/reduction不变；prefill只可能从 unroll受益。
7. 若 C/IR table supply计数不变，materialize不保留；若 unroll2不涨或破坏资源，维持现有
   binding。第二个 memory-lifetime输入是 IQ4_NL（已显式 materialize同一 16-entry table），
   但它不能证明跨 block reduction 合并。

参考边界：Triton 的 backward rematerialization把 expensive memory op当 anchor，不把 table
load复制给每个 consumer；TileLang 用 producer/consumer access region与 last-use决定 buffer
lifetime。Weft 在 Canonical Level 中由作者声明跨 K 的 materialize，Physical pass只选择其表示。

## 预实验对答案

- 把 16-entry codebook 显式 materialize 后，K1 standalone 从 `1.975292`
  变成 `1.979554 GOP/s`；远端汇编文本没有变化。Clang 已把 table load 提出
  block loop，因此这不是 P2 缺口，代码不保留。
- `unroll=2/4/8` 的单次运行分别为 `2.098855/2.064096/2.033976 GOP/s`。
  `unroll=2` 只消掉少量 loop overhead，仍不足以解释相对 source 的差距；prefill
  的单次结果还从约 `2.117` 降至 `2.053 GOP/s`，因此不把参数改动当作闭合方案。

重新逐段计数后，真正的动态工作差异在 contraction 内部：当前每个 32-element
block 生成两条独立 i16 lane partial，各做一次 `vwredsum`、一次 scalar extract，
最后 scalar add；donor 依赖固定 MXFP4 table 的窄值域，把两半留在一条 i16 carrier
中，只做一次 reduction。Canonical ABI 接受任意 i8 codebook，编译器不能假定 donor
的窄值域，因此直接复用 donor 的 i16 accumulation 会改变溢出语义。

## widening combine 实现前七步账

1. Canonical contraction 是一个 32-element i32 reduction；两个 16-lane issue 是同一
   reduction axis 的物理切片，不是两个作者值。
2. 当前每 block 的相关计数是两条 i16 product、两次 widening reduction、两次 extract、
   一次 scalar add。MXFP4 与 IQ4_NL 都有相同的 two-stream typed relation。
3. 合法的减少方式是保留两条 i16 product，先用一次 signed widening vector add 合成
   一条 i32 lane carrier，再做一次 i32 reduction。预期删除一次 reduction、一次 extract
   和 scalar add，增加一次 `vwadd.vv`；数学乘法数不变。
4. carrier 从 `2 × i16m1` 变为 combine 时同时存活的 `2 × i16m1 + i32m2`，峰值为
   4 vector groups；两台 target 都在 32-group 合同内。lane 数和 logical reduction axis
   不变。
5. 选择条件只能读取 typed facts：恰好两个 reduction-time streams、窄 fusion 的完整
   范围证明失败、source partial 为 signed i16、结果为 signed i32、target 支持 widening、
   资源闭合。已证明可安全 fused 的 two-stream（历史 Q5_1 负例）仍走原路径。
6. planner 冻结 i16→i32 combine set type 和 exact `rvv.partial-combine.widen` leaf；
   materializer 只实例化，emitter 只拼写 `vwadd.vv`。最终 IR 应直接出现这个 use-def。
7. 若两次 reduction 没有塌成一次，或 MXFP4/IQ4_NL 任一没有受益，则这个合同没有减少
   预计的动态工作，撤销实现；不能靠格式、kernel 或 target 分支保住单点数字。

参考机制不是 donor 的窄值域技巧本身。Triton 的 reduction lowering按 source layout 的
register/lane bases组织 partial，并保持 accumulator dtype；TileLang 的 reducer plan要求
各 update site 得到结构相等、dtype 闭合的 partial storage。Weft 不能直接照搬 GPU thread
layout或 Tile buffer，但同样必须先冻结 partial carrier 与 dtype，再让终端 lowering消费。

## 结果

实现后的 RISC-V IR 显式包含 `rvv_partial_set → rvv_partial_combine`（selected leaf
为 `rvv.partial-combine.widen`）`→ rvv_partial_finalize`。K1 生成 C 的相关动态工作由

```text
2 × vwmul + 2 × vwredsum + 2 × extract + scalar add
```

变为

```text
2 × vwmul + vwadd + vredsum + extract
```

K1 十次正式结果如下；三种入口数值均在容差内：

| entry | Weft 前 | Weft 后 | source | 后/source |
|---|---:|---:|---:|---:|
| MXFP4 standalone | 1.978456 | 2.372602 | 2.580476 | 91.94% |
| MXFP4 MUL decode | 2.011957 | 2.313262 | 2.535316 | 91.24% |
| MXFP4 MUL prefill | 2.117196 | 2.884714 | 2.629021 | 109.73% |
| IQ4_NL standalone | 2.018567 | 2.379864 | 2.821807 | 84.34% |
| IQ4_NL MUL decode | 1.978564 | 2.327479 | 2.762856 | 84.24% |
| IQ4_NL MUL prefill | 2.058851 | 2.762285 | 2.870430 | 96.23% |

直接在 VLEN128 上选择同一 wide carrier 是负结果：MXFP4 decode/prefill 分别从
`6.522843/7.658325` 降到 `6.003909/7.095727 GOP/s`，IQ4_NL decode/prefill
从 `6.529264/9.126334` 降到 `5.861625/7.470585 GOP/s`。两种格式的 widened
result 都占 4 个 vector groups；VLEN256 上只占 2 个。最终固定结构规则因此要求
widened result 不超过 2 groups，VLEN128 继续用原有 per-stream reduction。规则读取
typed carrier 的资源合同；上限由 target profile 的
`max-widening-combine-groups` 提供，不读取 target、format 或 kernel 名。恢复后 SG 单次
MXFP4/IQ4_NL standalone 为 `6.411937/6.488570 GOP/s`，与原路径一致。

本轮没有把四条仍未过 source 的 K1 行解释成编译器还能无条件消掉的工作。donor 的
剩余优势来自固定 codebook 值域允许 `vwmul + vwmacc` 留在 i16；Weft 的公开 ABI 接受
任意 i8 table，只能用多一条 `vwadd` 的 exact i32 combine。要继续闭合必须由作者程序
表达固定 table/值域，或改变 ABI；把该事实写进 pass 会让合法输入溢出，不能做。

K1 全部 vec-dot 的静态扫描还找到了两个没有参与规则设计的输入。十次正式结果均保持
数值容差：

| entry | Weft 前 | Weft 后 | source | 后/source |
|---|---:|---:|---:|---:|
| IQ4_XS standalone | 2.095060 | 2.624126 | 1.713500 | 153.14% |
| IQ4_XS MUL decode | 2.127253 | 2.613976 | 1.692141 | 154.48% |
| IQ4_XS MUL prefill | 2.102341 | 2.393512 | 1.717705 | 139.34% |
| NVFP4 standalone | 0.762450 | 0.945007 | 0.158325 | 596.88% |
| NVFP4 MUL decode | 0.759401 | 0.945292 | 0.161658 | 584.75% |
| NVFP4 MUL prefill | 0.786240 | 1.065293 | 0.162006 | 657.56% |

因此 widening combine 有四个独立格式输入，其中两个是实现后才由全格式静态扫描发现；
四者在 VLEN256 上全部正向，不能解释为针对 MXFP4/IQ4_NL 的格式 matcher。96 份
vec-dot/row-dequant Physical IR 均通过 parse、verify、canonicalizer、CSE、Share 连跑
两次与 final verifier，第二轮文本 diff 为零。
