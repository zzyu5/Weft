# 模块 H — 正确性 / 证据

## 职责

定义新架构唯一允许的活验证：按当前被推进的一个 Weft kernel，尽可能走到真实
RISC-V 运行和数值对照的手动 repro；若缺目标/工具链，必须明确停点，不能把
selected MLIR 或 source 检查写成真机验证。

## 不迁移旧测试体系

旧 test、lit、CTest、fixture、catalog coverage、certification 和战役账本不迁入
新主干，也不建立替代它们的新测试目录或覆盖率系统。

## 唯一 repro

每次只选一个手写 Weft kernel，执行：

```text
Weft Python DSL → canonical Weft Kernel MLIR → selected RISC-V execution
→ RISC-V source/object → 真实目标运行 → source reference 数值对照
```

不建立 kernel corpus，不为边界情况增加 fixture，不累计通过数量。某个语言构造
没有进入这条主链，就直接视为未实现。

当前 RVV 代表 kernel 已经走到 object、真实机器和数值对照。第一条 IME
`si8×si8→si32` fragment-tiled 路径目前只走到 canonical→`weft_ime_execution`
selected MLIR→source；逻辑 `8×12×16`、非整除 `5×6×9` 与多 contract site 已完成
selected IR round-trip 或 C++ syntax check。由于当前环境没有
SpacemiT IME toolchain/目标机，这不是 IME runtime correctness 证据。旧 IME plugin 的
K1 seal 只能证明复用的 `vmadot` hardware leaf 事实，不能代替新 canonical/selected
主干的端到端运行。

同一 kernel 的两个独立 RVV reduction component 已完成 canonical→selected→object 与
selected IR round-trip，但由于共享 RVV 目标机当时已有持续硬件扫描负载，没有追加
数值运行；因此它只证明组合选择和 artifact 闭合，不计作新的 runtime correctness 证据。

本轮新增的 rowwise max、三阶段 softmax 与 rank-3 affine 都已完成 canonical MLIR、
selected verifier/round-trip、RVV source 和 RISC-V object。softmax 的 source 可见真实
`vfredmax`、`vfredusum` 与两个 `exp_poly_v1` 序列；rank-3 source 可见由 layout order
推导出的三层循环。这仍是 artifact correctness，不是真机数值结论：实时预检发现共享
RVV 主机上已有两个长期占满 CPU 的 sysfs 扫描进程，本轮没有部署、运行或终止它们。

## 与旧测量纪律的关系

性能只在同一条真实 repro 上、同一目标和同一调用边界下测量。没有真实硬件结果
就不报告性能；不建立 master/runs/certification 等累计账本。
