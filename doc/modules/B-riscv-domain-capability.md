# 模块 B — RISC-V Domain Binding 与 Capability

## 职责

建立唯一的 selection/deployment domain（RISC-V target/profile）与其 typed
capability environment `C_d`，供域内每个 owner（RVV/IME/Scalar，未来任何新增
RISC-V 扩展）投影自己的 `c_o=π_o(C_d)`。

这部分大量复用旧架构已经验证过的设计（`BindDomain(t)=(d,C_d)`、owner-local
capability projection、不建跨 owner 的 giant optional struct、显式
domain-identity membership gate）——这套东西本身没有被这次重构否定，需要
调整的是**粒度**：旧架构里一个 kernel 绑定一个 domain 后整体交给一个 owner；
新架构下，同一个 Intent kernel 内部的**不同结构化原语实例**可能需要在同一
domain 内选择不同 owner（例如某个 `contract` 交给 IME，另一个 `reduce`
交给 RVV）。

## 与模块 C 的边界

模块 B 只负责"这个 domain 里有哪些 owner、每个 owner 的能力事实是什么"，
不负责"某个具体结构化原语实例该选哪个 owner"——后者是模块 C 的职责。

## 待裁问题

- 旧架构的 capability 模型（`[S-1]` 结构化事实、`kind` 闭合枚举、
  `provides`/`implies`/`conflicts` 关系）是否原样保留，还是需要因为
  "细粒度、按原语实例选 owner"这个新需求而调整？
