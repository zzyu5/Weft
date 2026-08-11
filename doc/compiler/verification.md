# Legality 与错误边界

Weft只验证编译所需、局部可判定的结构事实，不建立证明、审批、认证或测试合同。

## Frontend 与 canonical IR

Frontend和dialect verifier负责拒绝自相矛盾的IR，例如：

- entry ABI、argument kind、return与type不一致；
- pointer arithmetic、shape、broadcast、axis或tuple type不合法；
- nested VLA、active VLA value逃逸或VLA body任意修改outer state；
- masked value进入不理解validity的consumer；
- reduce/scan/summary/contract的operand、state、axis、predicate或result shape不闭合；
- summary region含memory effect或不以正确type yield；
- extension dialect未链接，或extension operand不满足其typed local schema。

Verifier只检查IR内部已经存在的事实，不从tensor shape推导algorithm identity，也不检查它
是否“像”GEMM、Softmax、Attention或某种q-format。

## Target lowering

Target lowering针对本次target/profile/config检查：

- 所需RVV、fixed VLEN、element width或matrix extension是否存在；
- 当前primitive closure是否有合法realization；
- register/fragment/scratch/alignment与local fusion是否可实现；
- meta/backend binding是否完整且合法。

任何一项不满足都直接返回明确unsupported/error。不存在legacy、GGML、旧emitter或默认
scalar fallback；正式scalar primitive的普通C lowering不属于fallback。

## 不由 verifier 证明的事实

以下是source/caller语义前提，不建立额外证明系统：

- `noalias`、alignment、bounds与pointer lifetime；
- VLA non-atomic iteration effect independence；
- custom summary algebra的identity/associativity及声明的commutativity；
- external worker slices之间的数据竞争与同步；
- persistent packed storage确实符合source声明的格式。

错误输入的行为由对应source contract决定；compiler不添加try/catch、default value或防御性
fallback来伪装支持。

## 工程验证边界

实现质量使用手工可运行的真实repro检查：同一DSL source生成target代码，在目标机执行并
对照数值与性能。Repro不形成新的IR authority、capability database、compatibility layer或
长期测试框架。

## 架构退化判据

以下行为直接违反规范：

- 在canonical language重新引入program grid、implicit worker/hart identity或physical lane ID；
- target从普通SSA graph、完整shape、kernel/operator/q-format名字猜algorithm skeleton；
- target创建source中不存在的algorithmic loop、staging、persistent layout或state algebra；
- logical block、register microtile与ISA fragment被合并成同一层；
- capability、legality、physical configuration或target-local state成为第二份持久authority；
- extension primitive接管完整kernel outer loops，或inline asm变成whole-kernel emitter；
- tuner改变observable numerical semantics或使architecturally illegal config变合法；
- Python成为generated artifact的runtime依赖。
