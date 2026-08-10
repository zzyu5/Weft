# 编译器总架构

## 27. 最终架构总图

```text
Python Weft DSL / other frontend
                │
                ▼
Canonical worker-local Weft Kernel IR
  ├─ scalar control / pointer / effect
  ├─ VLA iteration regions
  ├─ logical predicates / masked values
  ├─ logical block values
  ├─ reduce / scan / summary fold
  ├─ contract / lookup / decode / extension primitives
  └─ source meta-parameters
                │
                ▼
RISC-V target profile
  architectural facts + optional microarchitecture hints
                │
                ▼
Primitive-local realization providers
  capability + parameter family + legality + resource + lowering
                │
                ▼
Compiler-derived legal execution space
                │
        build-time AOT tuner
                │
                ▼
Selected Execution IR
  vl mechanics / LMUL / memory plan / microtile / fragment / strategy
                │
                ▼
Mechanical scalar + RVV + extension lowering
                │
                ▼
object / static library / C header / optional AOT dispatcher
                │
                ▼
llama.cpp / ggml / framework / application runtime
  owns threads, work partition and multi-core scheduling
```

---

## 28. 定位一句话

> **Weft 是一门面向单个 RISC-V worker/hart 的 AOT kernel DSL：作者用普通控制流、一等 VLA iteration region、logical block、显式 predicate/state algebra 与 structured compute primitive 编写完整 worker-local 算法；compiler 从 target facts 与 primitive provider 中构造合法的动态 `vl`、LMUL、register microtile、memory 与扩展 realization，构建期 tuner 选择性能点，多核调度由外部 runtime 负责。**
