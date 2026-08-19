# 真实 Repro

Weft 唯一允许的验证是可手工执行的一条真实主链：

```text
DSL source
→ canonical Kernel IR
→ target lowering
→ intrinsic C / local asm
→ system compiler
→ target execution
→ compare numerical result with the same algorithm
```

不建立 pytest/lit、单测、边界测试、fixture、case matrix、版本兼容测试或验收状态机。

## Repro 必须说明

- DSL kernel symbol 与 constexpr/meta instance；
- target profile、ISA/ABI 与必要 extension；
- shape、dtype、storage format 和 input organization；
- 生成与编译命令；
- 真机执行命令；
- 数值 reference 与比较方式。

性能结论还必须保证同硬件、同算法、同 shape、同数据组织、同 preprocessing 归属和相近计时
范围。跨 target 只用于观察选择差异，不直接给速度优劣结论。

## Artifact 边界

若 target 不可用或 command 没有合法 realization，应准确报告停在：frontend、canonical IR、
target decision、intrinsic C、system compile 或 target execution 的哪一处。构建成功不能替代数值
执行，明确 unsupported 也不能写成已支持。
