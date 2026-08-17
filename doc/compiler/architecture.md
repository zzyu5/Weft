# 编译器结构

## 唯一主链

```text
Weft Python DSL kernel
→ canonical Weft Kernel IR
→ one RISC-V target lowering
→ intrinsic C / local inline asm + C header
→ system C compiler
→ object / executable
```

Python 前端只把作者程序转换成 Kernel IR。仓库外的前端也可以直接生成相同 Kernel IR；
Weft 不解析或链接 IntentDSL。

长期保存的编译器表示只有 Kernel IR。LMUL、memory form、state placement、microtile、packing、
fragment、resource legality 与候选选择只存在于一次 target lowering 调用中，不形成可输入的
第二份 IR。

## 三方职责

作者拥有：

- worker 的 `for / while / if` 与 effect 顺序；
- outer traversal、blocking、staging 与 persistent layout；
- block axis、pointer/index、predicate、state 与 accumulator 生命周期；
- 显式 `W.vla`、`W.dot/W.matmul`、state 和扩展局部运算。

RISC-V target lowering 拥有：

- VLA strip、`vl`、LMUL、mask 与 memory instruction；
- value/register shape、handoff、reload、rematerialize 与短期 local pack；
- dot/matmul microtile、multiple accumulators、K-unroll 与已实现的 load/pipeline schedule；
- RVV、IME 和其他扩展的局部实现；
- intrinsic C 与 typed inline asm 的生成。

系统 C 编译器拥有最终寄存器分配、最终机器调度、peephole、常量折叠与机器码生成。

## 组合原则

Target lowering 逐个读取 VLA、memory、predicate、state、dot/matmul、decode/lookup 与扩展运算
自身的 typed operands、axis、validity、effect、普通 use relation、backend config 和 target facts。
它不能先判断完整 kernel 属于哪一类，也不能用 kernel 名、格式 route、外围 loop 数量或精确
producer/use 形状选择整段实现。

一次 target lowering 内部只有以下信息流，不产生新的持久 IR：

```text
typed program/target facts
→ 唯一合法性与轴关系推导
→ structural implementation candidates
→ parameter instances 与 resource filtering
→ selected physical decisions
→ intrinsic C / typed local asm
```

一个 value 的 physical shape 只有一个决定来源；每个 `(consumer, value)` 也只有一个 handoff。
重复记录若在 Share/Convert/Reload/Rematerialize/LocalPack、source shape 或 result shape 上冲突，
lowering 必须在 emission 前失败。生成 intrinsic C 时只能读取 selected decisions，不能再次推导
LMUL、microtile、layout、fragment、table width 或 pipeline。

## 当前实现的信息流

`RISCVKernelFacts` 从 Kernel IR 记录 axis identity、ordered parent、普通 consumers、definition/last
use、control carry，以及每次 memory access 相对各轴的 unit/strided/indexed/non-affine relation。
它还根据pointer、index SSA、element bytes和effect顺序形成indexed-address与interleaved-memory
group。这些是后续所有实现共同读取的程序事实，不按example或量化格式分组。

`RISCVReuseAnalysis` 从同一份facts推导operand是否随reduction推进、是否直接供给primitive、
address/predicate是否依赖accumulator、consumer数量和control crossing。它是F32 dot、VLA dot和
F16 matmul共同的reuse/pipeline输入，不产生第二份IR。

`RISCVAxisMapping` 与 `RISCVPhysicalPlanning` 把显式op约束组合成sequential/lane/register/unroll/
fragment轴分解，生成operand window、pipeline actions与统一resource budget，再按target profile和
显式backend config选择合法实例。Dense与quant都使用同一mapping和resource机制；quant格式只在
typed numerical rule与最底层local operation中保留差异。

`RISCVKernelCompiler` 把相连value的selected shape、memory form、state placement、primitive
realization与handoff组成一次瞬态physical plan。VLA lifetime按真实operation位置计算；普通nested
`for/while/if`会继承外层仍存活的值，nested VLA仍明确unsupported。所有决定准备完成后才生成
kernel body，并同时收集实际使用的exact intrinsic/asm leaf。

`RISCVIntrinsicCPrelude`、`RISCVRVVIntrinsicC`、`RISCVQuant*IntrinsicC`与`RISCVIMEIntrinsicC`
只按selected local operation拼写helper；它们不读取VLEN、kernel名、格式名或外围IR重新选择实现。
`examples/run/weft.sh`中的selector只选择DSL source与相邻runtime，不是production lowering route。

## 仓库边界

- `source/` 只保存 GGML baseline 与对应 runtime，不进入 Weft 编译主链。
- `materials/` 只提供历史实现知识，不进入 CMake、include、import、link 或 runtime。
- `examples/kernels/` 保存 DSL kernel；`examples/repro/weft/` 保存与其相邻的真实 runtime。
- `report/` 保存一次性工作记录与当前性能 CSV，不定义语言或编译器。
