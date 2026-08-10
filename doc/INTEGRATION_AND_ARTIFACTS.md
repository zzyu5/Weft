# 前端接入、Runtime 与 Artifact

> 本文定义系统之间的工程边界，不改变最终规范中的 kernel ABI 或 artifact 语义。

## 1. Weft 是独立 DSL，不消费 Intent IR

Weft core 的正式输入是 canonical Weft Kernel MLIR，而不是 Intent MLIR。IntentDSL 可以
把 Weft 作为 target DSL/backend 纳入自身，但依赖方向必须是：

```text
Intent algorithm/source
        │
        └─ Intent-side target plugin or standalone bridge
                 │
                 └─ canonical Weft Kernel IR
                          │
                          └─ weft-compile
```

Bridge 位于 IntentDSL 一侧或独立 integration package。Weft repository 不包含
`intent.*` dialect、Intent parser、Intent schedule classifier 或 Intent runtime。这样 Weft
可以独立编写/编译 kernel，IntentDSL 也能把同一算法 lowering 给其他 target DSL。

## 2. 上游 lowering 合同

任何上游生成 canonical Weft IR 时，都必须显式决定 source/algorithm 层事实：

- worker-local ABI 与显式 work descriptor；
- scalar outer control 和 VLA logical range；
- algorithm/cache blocking、staging/recomputation skeleton；
- pointer/index、logical predicate、effects；
- state algebra 与 structured primitives；
- source meta-parameters、algorithm variants 和 numerical policy。

上游不得把全局 worker/program axis 直接改名为 `task_id` 塞进 Weft。它必须在上游或
runtime 层完成 work slicing，并把 source 选择的普通 scalar/pointer 参数或 descriptor
传给 kernel。Weft 不规定统一 slice 形状；row range、tile index、expert/quant-block range 与
ragged descriptor 都是可能的 ABI。
同样，上游不应指定 LMUL、register microtile 或 IME fragment；这些是 Weft provider 的
物理选择。

Bridge 生成的 IR 与 Python frontend 生成的 IR 走完全相同的 parse、canonical verifier、
selection 和 artifact pipeline。不存在“Intent 专用 emitter”或按来源选择的 lowering。

## 3. `weft-compile` 边界

Compiler driver 的概念输入为：

- canonical Weft Kernel MLIR；
- RISC-V target profile；
- 可选 build specification：source meta domains、specialization predicates、measurement
  harness 与作者提供的 variants；
- 所需 semantic extension dialect/provider registrations。

概念输出为：

- canonical MLIR（检查/规范化后的可读边界）；
- Selected Execution MLIR；
- 可选 readable generated source；
- relocatable object；
- static library；
- C-compatible public header；
- 可选 multiversion dispatcher。

Driver 不接受 kernel/format route，不允许 CLI 选择 whole-kernel owner。Target 选项只构造
typed target profile；provider discovery 根据 primitive 自动发生。

## 4. 普通 C ABI

生成 entry 使用普通 C-compatible ABI。Worker 的工作范围由 source 定义的普通参数或
descriptor 携带，例如：

```c
void kernel(const float *x,
            float *y,
            int64_t begin,
            int64_t end,
            int64_t stride);
```

具体参数由 canonical kernel ABI 决定；以上只展示边界形态。Artifact/header generator
必须从 canonical ABI 生成声明，不能从 target emitter 的 operand 顺序另造签名。

Pointer qualifiers、alignment、address space 与 alias contract 要在 ABI metadata 中可追溯。
Selected IR 可以选择 memory strategy，但不能复制或改变 ABI。

## 5. Runtime 责任

Weft artifact 不创建线程。外部 runtime 负责：

- worker 数量、affinity、NUMA、线程池；
- 全局 iteration/domain 的切分；
- 按 kernel ABI 给每个 worker 构造 source-defined slice arguments/descriptors；
- feature/shape/stride 条件下选择已构建的 AOT variant；
- 调用 kernel 并管理输入输出生命周期。

Runtime 不负责：

- 解释 Python DSL 或 MLIR；
- 在运行期调用 compiler/JIT；
- 决定每次 RVV `vl` 或 strip；
- 修改 provider-selected microtile/fragment；
- 用隐藏的 thread id 改变 kernel 语义。

## 6. Multiversion dispatcher

Build-time tuning 可以生成多个已验证、已编译的 variants。轻量 dispatcher 可以检查
shape predicate、alignment/stride class、target extension 或 fixed-VLEN guard，然后调用
已存在的 entry。Dispatcher 不管理线程，不生成代码，也不把一次 dispatch 选择写回
compiler。

每个 variant 都对应完整 canonical kernel variant + selected execution + artifact provenance。
Compiler 只能在作者明确提供的 algorithm variants 中选择，不能为了某 target 自动发明
新的 outer traversal 或 staging skeleton。

## 7. Artifact 生成纪律

Artifact pipeline 是单向的：

```text
canonical + selected + target facts
    -> transient provider lowering
    -> generated source / LLVM or toolchain input
    -> object
    -> library + public header + optional dispatcher
```

Packaging 层可以处理 symbol、temporary source、toolchain flags、object bytes、header 与
bundle layout，但不能重新运行 selection。Artifact handoff 必须 fail closed：selected record
缺失、ABI 不一致、未注册 extension 或 toolchain target 不匹配都应明确报错。

## 8. 集成验收

对任一前端，最小真实链路是：

```text
source kernel
  -> canonical Weft MLIR
  -> selected execution
  -> RISC-V source/object
  -> external runtime on real target
  -> numerical comparison
```

这是一条按功能增量维护的手工 repro，不扩张成独立测试 corpus、覆盖率工程或兼容矩阵。
若目标机/extension toolchain 不可用，应准确声明链路停在 source、object 还是 link 边界，
不能把 source-only 当成可运行 artifact。
