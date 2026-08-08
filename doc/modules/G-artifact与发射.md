# 模块 G — Artifact / 发射

## 职责

把 typed extension body 机械降级为 EmitC/C/object 等最终 artifact。

## 与旧架构的关系

这是复用程度最高的模块。旧架构已经完成"artifact-neutral construction
lifecycle"横向重构——owner 先构造/资格化 final typed body，artifact
driver 只做机械 lowering，`emitc.func` 只是 EmitC 自己的完成门，不是
跨 owner 的 construction 完成门。这条边界与新方向不冲突，预期基本原样
保留。

## 待裁问题

- 无（本模块预期改动量最小；如果模块 F 的 typed body 结构发生大改，
  需要回头核实本模块的假设是否仍然成立）。
