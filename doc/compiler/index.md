# 编译器总览

## 1. 两层 MLIR

Weft 的程序 IR 只有两层：

```text
Weft Kernel IR
    logical Value、axis、Level、Encoding、operation、effect
        ↓ ConvertWeftToRISCV
Weft RISC-V IR
    layout、conversion、memory、schedule、RVV/IME leaf
        ↓ TranslateToIntrinsicC
intrinsic C / local asm
        ↓
system C compiler
```

第一层对应 Triton TTIR，第二层对应 TTGIR。非 SIMT 物理机器是第二层的语义，不是第三层 IR；target profile、build config、tuner 和各个 pass 也不是 IR 层。RISC-V module 可以混用 `func`、`scf`、`arith` 与 `weft_riscv` operations，dialect namespace 不增加层次。

当前 RISC-V physical compiler 的正式目标范围是带完整 `V` 扩展且构建时显式给出正 `VLEN` 的 RV32/RV64 profile；`Zve` 子集、纯标量 target 和未知 VLEN 会在创建 RISC-V IR 前明确拒绝。scalar physical value/leaf 是同一 RVV kernel 内的表示与操作，不构成另一条 scalar backend。

## 2. 两层的 authority

[Canonical Kernel IR](../dsl/canonical-ir.md) 是唯一持久作者程序，保存 logical tree、Level/lifetime、Encoding/ABI、ordinary control 与数值语义。target pass 不能增加 canonical Value、Level、effect、persistent artifact 或算法阶段。

[RISC-V IR](riscv-ir.md) 是一份瞬态、target-aware、可改写的 physical program。layout、conversion、memory form、local object、pipeline、resource handling 和 [RVV/IME leaf](leaves.md) 必须存在于这份程序中，不能只保存在 side record。

## 3. Candidate 与 tuner

一份 RISC-V module 对应：

```text
一个已实例化的 canonical/std candidate
× 一个 target profile
× 一组 physical parameter bindings
× 一个按固定优先级选定的 structural realization
```

source `auto` 在进入 lowering 前绑定。结构固定后的 LMUL、microtile extent、unroll、pipeline depth 等有限参数，每组绑定各自建立并编译一份 RISC-V module。depth=1表示顺序执行；当前两阶段 pipeline 只接受 depth=2/buffer=2，并且必须能从一个带 carry 的 physical Level 中推导出 pure/read producer 与 carry-dependent consumer cluster，buffer count不是独立 binding。当前 compiler API 每次只接受一组单值 binding；外部 tuner 可以枚举这些 module 并比较可执行 artifacts，但不共享 `assignment` 字典，也不生成新的物理结构。

结构性选择按 target 固定规则和优先级进行；一个 module 内不保存备用路线。选定结构在后续 legality/resource pass 中失败时该 module 直接失败，不在 pass 或 emitter 中回退。需要另一 target/configuration 时，调用者从同一 canonical candidate 重新建立一份 module。

## 4. 编译器文档

- [RISC-V IR](riscv-ir.md)：physical values、memory descriptors、conversion、local objects 与 schedule entities。
- [编译 Pass](passes.md)：同一份 RISC-V IR 上的分析、选择与真实 rewrite。
- [非 SIMT 物理优化方法](optimization-principles.md)：从生成 IR/C/汇编分类动态工作、定位 carrier、supply、memory、reduction 与 pipeline 缺口。
- [Local Leaf](leaves.md)：RVV intrinsic、IME/opaque asm leaf 的合同与选择。
- [Terminal Emission](emission.md)：已选 leaf 和已物化程序怎样确定地写成 C/asm。
- [非 SIMT 物理机器](../machine/physical-machine.md)：time/lane/register-replica/fragment/local-storage 的机器语义。

## 5. 明确排除

Weft 不采用：

- `weft_phys → weft_riscv` 两级 physical lowering；
- `weft_riscv.problem` / `weft_riscv.assignment` decision dictionaries；
- canonical program 加 side records，再由 emitter 首次构造 physical program；
- 同一语义的新旧 physical 路径、compatibility layer 或 fallback；
- final emitter 中的 layout/default/selector/source-closure reconstruction。

未来 target 实现同一非 SIMT 机器合同时，替换第二层 target-aware physical IR，不插入第三层，也不要求另一份 source DSL。
