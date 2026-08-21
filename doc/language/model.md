# 语言模型

Weft 作者编写一棵有层归属的有限位宽计算树。树的叶子是带 Encoding 的内存 `View`，树内的普通 SSA 值是没有物理布局的 `Value`；`Level` 记录哪些值在某个逻辑域层诞生、被后代复用，以及哪些外层值由该层交回。普通 Python `for / while / if` 则保持有序标量语义。

这不是命令序列，也不以 worker、tile、VLA、stream、state machine 或某种 RISC-V 引擎作为语言根。一次 kernel invocation 最终在 CPU 上执行，但 source 中没有 grid、program id、hart id 或隐式任务身份。

## 当前公开对象

```text
@weft.kernel                 一个前端可降为 canonical Kernel IR 的入口函数
@weft.encoding               逻辑字段坐标到 storage bits 的纯布局映射
@weft.derive                 一个 build/load 阶段的编码类型生成器
View[Encoding, shape]        带编码的内存对象
L.rows/cols/tiles/blocks/subs
                             一个 domain/partition 关系
new                          本层诞生、可被后代更新的值
materialize                  本层物化一次、被后代只读复用的值
admit / commit               View 区域与本层 Value 的边界
普通函数                     调用点 inline 的同语言程序
基本 op                      局部、完整、可观察的数值关系
```

`Value` 只有逻辑 element type、shape 和 axis identity。LMUL、物理 lane、寄存器组、microtile、fragment 和 intrinsic 类型都不进入语言或 canonical Kernel IR。

## 作者决定与前端保留

作者通过程序树决定：层级、cohort、分块、值诞生位置、物化位置、显式数据重排、累加类型、有限位宽运算和普通控制依赖。前端只把这些事实无损变成 canonical IR；它不从普通 multiply/add 猜 contraction，不把普通循环改成宽执行，也不把一种标准库函数替换成另一种算法树。

## 唯一语言层级

`python/weft/std/` 中的 GEMM、GEMV、attention、q4_K GEMV 与 Top-K 都是普通 Weft 函数。它们不是后端 route，也没有比调用者更高的语言权限。调用在前端 inline，展开后的值链与调用者共同进入同一份 Kernel IR。
