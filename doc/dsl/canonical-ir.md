# Canonical Kernel IR

Canonical Kernel IR 是 Weft 源程序的唯一算法与数值 authority。它保存作者写下的树，供任何目标后端读取；它不保存某次 target lowering 的机器表示。

本文只规定 canonical 边界。两层 lowering 与 target-aware physical IR 见[编译主干](../compiler.md)。

## 1. 必须承载的内容

### 1.1 Kernel 与普通控制

- kernel 参数、View 类型、access effect 与 alias boundary；
- 普通 `for/while/if` 及其有序语义；
- loop/branch carried SSA values；
- source operation 的 use-def、region nesting 和 dominance。

### 1.2 Encoding 与内存语义

- base/dense/derived family/derived instance identity；
- bit order、byte order、elements、alignment和padding；
- field storage span；
- grouped/layered/bit-plane/joined mapping；
- View/Slice 的 logical shape 和 axis identities；
- derive builder、persistent interleave 与 pinned artifact layout identity。

### 1.3 Logical Value

- element type；
- logical shape；
- axis identity；
- pointwise/broadcast/reduction/contraction axis relation；
- 显式cast/widen/narrow及其中间数值类型。

### 1.4 Level

- domain axis、relation和parent identity；
- partition、multiplicity、tail；
- 尚未实例化时的 source `auto` 参数 identity；进入 target lowering 的 candidate 则携带具体绑定；
- carried operands/results；
- `births.state` 与 `births.staged`；
- body region；
- handoff operand、result type和层归属；
- logical cohort width。

Level 不能在canonical化时退化成“一个loop加几个可丢弃attribute”。births和handoff必须是typed region/SSA结构。

### 1.5 Numerical operations

- pointwise operation；
- grouped MAC 的group width与accumulator type；
- reduce/fold/dot/contract的输入、输出与轴；
- lookup/index relation；
- state update与effect；
- operation精度、顺序、overflow/wrap/saturate语义；
- reduce identity、空域政策、NaN/signed-zero与允许的结合自由度。

## 2. 按设计不承载的内容

Canonical Kernel IR 不包含：

- target ISA、机器 ABI（例如 `lp64d`）、VLEN或resource budget；
- SEW、LMUL、`vl`或tail loop实现；
- physical lane、register tuple或fragment layout；
- machine register编号；
- selected RVV/IME instruction；
- selected scalar/wide/matrix/transfer engine；
- unit/strided/indexed/segment memory form；
- invocation-local unpack/packing、schema、axis orientation 与 parameterized extents；
- physical layout conversion；
- unroll、prefetch、local pipeline和buffer version；
- live interval、spill、reload、rematerialization；
- intrinsic C或inline asm spelling；
- kernel/operator/quant-format专用backend route。

这些事实属于target candidate的物理程序，不得反向成为语言annotation。

## 3. Canonical verifier 的职责

canonical verifier必须检查程序是不是一份合法Weft程序：

- encoding mapping与storage span完整合法；
- View/Value shape、axis和element type一致；
- domain parent/partition/multiplicity/tail合法；
- Level births/body/handoff签名一致；
- 普通控制carry与branch result类型一致；
- admit/materialize/new/commit的domain与value关系一致；
- primitive operand/result shape和axis relation合法；
- derive builder result与declared family一致；
- pinned layout identity在ABI边界明确。

verifier不负责：

- 证明程序等价于某个上游数学contract；
- 证明作者选择的i16/i32不会溢出；
- 证明`bsum`由正确公式生成；
- 证明某个candidate高性能；
- 选择另一棵stdtree；
- 证明浮点结果与某个reference bit-exact。

## 4. 作者树与closed primitive

canonical IR必须区分：

```text
开放SSA树中的logical values
closed primitive内部不可观察的physical temporaries
```

例如：

- 把一个source accumulator拆成四个canonical state再合并，改变logical value集合，必须由作者写；
- `reduce`后端使用四个register partial并在primitive内合并，不改变canonical operand/result contract，可以由编译器实现。

这个边界由IR可观察性决定，不由“代码里是否出现四个临时变量”决定。

## 5. 与target compiler的交接

目标编译器从canonical IR读取：

```text
typed SSA
logical axes
Level/domain/lifetime
memory/encoding relation
operation semantics
effects
```

并与独立的 target profile/build config 一起，为每个已实例化 std/auto candidate 形成一份 target-aware RISC-V IR module。target profile 提供 engines、representations、legality 与结构规则；build config 提供 target requirement，例如最终 physical program 必须使用 IME。requirement 在结构性选择时参与 legality 过滤，并在 selected program 上验证；二者都不进入 canonical Value identity。

```text
Canonical Kernel IR
    ↓ ConvertWeftToRISCV
RISC-V Physical IR
    ↓ terminal translation
intrinsic C / local asm
```

这是两层 MLIR。target profile、build config、tuner和各个physical pass都是转换输入或对第二层程序的改写，不构成额外IR层。canonical module保持不变；每个candidate从它独立建立一份瞬态RISC-V module。

唯一合法事实必须从这些输入唯一推导；存在多个合法物理结构时，target 按固定规则和优先级选择；结构固定后的有限物理参数可以由构建期实测选择。若物理实现需要改变 canonical values、Level 归属、artifact ABI 或 numerical operation，它不能在 target lowering 中完成；必须返回到作者 tree 或另一个 std overload。

canonical IR本身不承担目标物理candidate的持久authority。一个candidate被拒绝或替换，不得修改源程序语义；选定layout、conversion、memory、schedule和target operations必须存在于RISC-V IR本身，不能只保存在side record中。
