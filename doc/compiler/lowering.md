# Lowering 与发射

## 一次 target invocation

一次 lowering 内按以下信息流工作：

```text
Kernel IR
→ typed program facts
→ op-specific requirements
→ connected value/axis/memory/lifetime analysis
→ legal structural candidates
→ parameter instances and resource filtering
→ one selected physical decision set
→ intrinsic C / local asm
```

中间对象是短生命周期实现数据。它们不能被序列化为另一份 IR，也不能被其他阶段当作第二真理。

## 普通控制与地址

作者的 scalar `for/while/if`、pointer arithmetic、workspace access 与 effect 原样投影为普通 C。
`W.blocks` 仍是作者的 block traversal；target 不替换其 loop order。`W.pipeline` 仅把
当前已存在 loop 标记为允许局部重排；当前 RISC-V lowering 尚未实现通用 loop
prologue/steady/epilogue scheduler，因而对该构造采用合法的顺序 realization。

Dot/matmul leaf 已有的 K-unroll 和 register load buffering 属于 explicit primitive 内部的
realization，由 primitive 本身授权；它不等于对作者 `for` loop 实现了通用流水。

## Selected local operation

每个 selected RVV/IME operation 已确定 input/output physical shape、memory form、mask/tail、
LMUL/microtile、unroll、buffering、resource 与 instruction family。Leaf 不再读取外围 Kernel IR
寻找结构。

高度专门的 intrinsic 或 inline asm 是允许的，但只实现当前局部 command；它不能接管 outer
traversal、persistent layout、workspace、state algorithm 或函数 ABI。

## Emitter 分层

发射代码按真实职责分离：

- ordinary C control 与 address expression；
- RVV type/intrinsic spelling；
- RVV API adaptation；
- typed quant/decode local sequence；
- IME local asm；
- function declaration、header 与 metadata。

RVV intrinsic API 变化只修改 spelling/adaptation，不穿透到 axis analysis、resource 或 candidate。

## 明确失败

如果 canonical semantics 完整但当前 target 没有合法 realization，lowering 返回明确 unsupported，
并停在 Kernel IR 或已生成 artifact 边界。禁止静默 scalar、旧 RVV、legacy emitter、GGML 或
materials fallback。

## Artifact

`compileRISCVModule` 的成功结果是可编译 intrinsic C/局部 asm 与调用所需 header，
header 包含已能从 canonical storage contract 投影的 workspace/persistent metadata。System C
compiler 再生成 object/library/executable；这些不是当前 compiler API 返回的 artifact 字段。
Artifact 不链接 `source/` 或 `materials/`。
