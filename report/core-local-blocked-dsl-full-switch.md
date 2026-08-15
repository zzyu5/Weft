# Core-local blocked DSL 全仓切换

## 结论

当前活动代码、文档与 examples 已只保留一套 Weft 编程模型：一份 DSL kernel 由一个
CPU worker/hart 持续执行；作者用普通 `for/while/if` 组织有序控制，用 `W.block` 创建带
逻辑轴身份的局部 block，并通过普通 SSA 保存、更新、合并和复用 block/state。`W.vla`、
`W.dot/W.matmul`、state 与 extension primitive 只授权各自局部域内的 RISC-V 物理实现，
不接管外围 traversal、blocking、staging、persistent layout 或函数 ABI。

活动主链只有：

```text
DSL kernel
→ canonical Kernel IR
→ RISC-V target lowering
→ intrinsic C / local asm + C header
→ system C compiler
→ object / executable
```

不存在第二套 DSL reader、旧 block lowering、备用 emitter、scalar/GGML/materials fallback
或按 kernel 名选择整段实现的路径。`materials/` 仍保存历史代码与硬件知识，但没有进入
Python import、CMake、include、link 或 runtime；它不属于当前可运行 DSL。

## 当前唯一语言模型

### Worker 与控制

- 一个 kernel invocation 对应一个 worker/hart 的持续有序程序；
- scalar `for/while/if` 的顺序、边界和 nesting 由作者拥有；
- `for/while/if` 可以同时 carry scalar、block 与 state；
- compiler 不创建新的 outer loop，不把 scalar loop 自动变成 VLA，也不替换作者算法。

### Block 与普通 SSA

`W.block(extent, offset=...)` 同时建立逻辑坐标和新的 axis identity。Kernel IR 的 block type
保存 shape、axis IDs 与 element type；相同 extent 不代表同一逻辑轴。Pointwise、predicate、
load/store 与 structured primitive 都按轴身份检查 domain，不按张量尺寸猜 broadcast。

Block 是普通 SSA value：

- 可以跨 `for`、`while` 与 `if` 存活；
- 可以有多个 consumer；
- 可以进入 pointwise、memory、state 或另一个合法 structured primitive；
- structured result 不要求 one-use、direct-store、邻接 consumer 或固定 producer 数量。

当前 examples 中真实存在并执行了：

- `dot → pointwise → store` 与 dot result 多 consumer；
- `matmul → pointwise → store`；
- matmul accumulator 跨 K `while` carry，并再次进入下一次 `matmul`；
- block 与 scalar 同时作为 loop carry；
- block 在 `if` 两个分支产生并在分支后继续消费；
- 同一个 packed block 经不同 transform 进入两个 store consumer。

### Structured primitive 的授权边界

Weft 不再提供任意轴乘加入口，只保留现有 kernel 真正需要的：

- `W.dot`：明确的局部 block dot relation；
- `W.matmul`：明确的局部 `[M,K] × [K,N]` block product；
- reduce、scan、argmax 与 online summary：各自保留独立 state semantics；
- lookup、decode、quant 与 IME primitive：只定义局部、可观察数值关系。

作者仍显式拥有 operand block、pointer/index、outer loop、blocking、staging、persistent
packing 与 accumulator lifetime。Target 只在已授权的 primitive 内选择 VLEN/LMUL、memory
form、microtile、local packing、fragment 与 intrinsic/asm spelling。

### Storage 与 lifetime

当前只有四类 storage：

| 类别 | DSL / Kernel IR 表达 | 生命周期 |
|---|---|---|
| ordinary input/output/state | external pointer | caller 至少持有一次调用 |
| persistent packed object | `W.persistent(format)` + `W.storage(shape)` | caller 跨调用保存 |
| worker-local workspace | `W.workspace, W.noalias` + `W.storage(shape)` | 当前 worker 一次调用 |
| primitive-private temporary | 只存在于 target lowering | 当前 primitive realization |

Pointer type保存 element type、access、alias/alignment、storage class 与 persistent format。
当前 target只有一种可公开寻址的 pointer 类，因此已经删除恒为 `global` 的 `address_space`
字段。生成的 C header 保存 entry declaration、pointer metadata，并为 workspace/persistent
storage生成 rank、extent 与 element-count query。

Attention、convolution、IME 与量化 projection 的 workspace/persistent object 已全部用
`W.storage` 声明。对应 runtime 通过生成的 query 分配容量，不再复制 scratch shape 公式。
Q1_0 与 MXFP4 model-format weight 是 ordinary external packed input，不伪装成 target-specific
persistent representation。

## 编译器切换

### Python DSL 与 Kernel IR

- 全部 DSL kernel 已从 `W.block_axis` 迁移到 `W.block`；
- `BlockType` / `RegionType` 保存 axis identity，pointwise 与 memory verifier按 axis验证；
- `for/while/if` 的 block carry 在 Kernel IR 与 intrinsic C 中均有实际 storage/handoff；
- unknown `weft_kernel.*` / `weft_ext.*` 不能混入 module；
- extension primitive 在 Python 边界检查 dtype、shape、validity 与 scalar/block domain，IR
  verifier再次检查 canonical semantics；
- affine/symmetric i4×i8 dot直接消费 typed persistent `u8*` 与显式 local layout，不再暴露
  只能被特定 consumer 使用的 packed-view value；
- unary `math` 参与 target legality；当前没有 exact VLA realization 的 `strict exp/tanh`
  会明确失败，不会使用近似实现假成功。

### RISC-V lowering

Target lowering从 block axes、typed operands、validity、effects、普通 use-def 与 target facts
建立物理决定。Emitter只读取决定并拼写 intrinsic C / local asm。

本轮关闭的 closure 问题包括：

- packed `and/shr` realization不再要求“所有 consumer 都是 F32 cast”；直接 u8 value与延迟
  widening fact可以同时服务不同 consumer；
- F16 mul-add fusion只用通用 liveness判断是否可吸收 private producer；共享 multiply仍可被
  其他 consumer使用，FMA能力本身不依赖 one-use；
- block tuple projection不再扫描其它 field并擅自 discard；
- rematerialized block tree包含 Region 与跨 region captured coordinate，shared block axis可在
  dot/matmul、store和外围控制中共同使用；
- block store/reduce按各自 primitive closure产生局部 realization，不要求源码邻接。

Target仍可以用 user/liveness事实做 DCE或决定一个 producer能否被某次 fusion吸收；这种判断
只改变是否多发一条等价 intrinsic，不决定 DSL program是否合法，也不作为 primitive能力入口。

## 全仓迁移规模

- `examples/kernels/`：72 个 Python 文件；
- 其中 71 个文件定义 89 个 `@weft.kernel` entry；
- `ggml_k.py` 只提供同一 DSL 中使用的 helper，不是独立 kernel；
- `examples/run/weft.sh`：79 个用户可选 case；
- 固定 phase 展开后，每个标准 RVV target执行 90 条命令；
- 10 个等价写法 case 每条命令生成两个 entry，因此每个标准 RVV target共进行 100 次
  DSL frontend/compiler invocation。

旧乘加 example 分类已并入 `dot/`。`source/` 继续只保存 GGML baseline，不含 Weft
DSL kernel；Weft runtime位于相邻 `examples/repro/weft/`。

## 真实执行结果

### 本地生成

`weft-compile` 重建成功。全部 89 个 entry 分别以 SG2044/VLEN128 与 K1/VLEN256 target facts
执行 DSL → Kernel IR → intrinsic C/C header，共 178/178 次成功。

### SG2044 / RVV VLEN128

- 固定语料：90/90 条真实命令正常退出；
- compiler invocation：100 次；
- 每条命令均在 SG2044 上用系统 C/C++ compiler生成 object/executable并执行；
- 无 compile error、unsupported、mismatch 或 runtime failure。

### K1 / RVV VLEN256

- 与 SG2044 相同 DSL kernel、shape与 runtime：90/90 条真实命令正常退出；
- compiler invocation：100 次；
- 每条命令均在 K1 上生成 object/executable并执行；
- 无 compile error、unsupported、mismatch 或 runtime failure。

### K1 / IME

同一 DSL kernel另外选择 IME1 local fragment realization：

| DSL kernel | 执行结果 | max absolute error |
|---|---:|---:|
| `q4_0_projection_ime` | 正常退出 | `0.0312509537` |
| `q4_k_projection_ime` | 正常退出 | `3.33786011e-06` |
| `q4_k_mul_mat_id` | 正常退出 | `1.90734863e-06` |

三条均生成 typed local asm fragment；outer traversal、activation quantization、workspace、
persistent weight与ABI仍来自同一 DSL kernel，没有 IME whole-kernel入口。

本轮用 repetition=1 完成结构性真实执行，未把单次时间写入性能数字表。现有性能数字未重算；
只统一了 DSL/dot 相关标签，数值保持不变。

## 已完整迁移并真实执行

- `W.block` axis identity与block pointwise/domain规则；
- scalar `for/while/if`，以及scalar/block混合carry；
- VLA pointwise、predicate、unit/strided/indexed memory；
- block/VLA load-store与masked validity；
- reduce、scan、argmax、online summary与sequential state；
- local f32 dot、f16 matmul、loop-carried accumulator与普通 structured result composition；
- lookup、decode、packed narrow/widen与现有quant local primitive；
- external/workspace/persistent storage及生成header query；
- SG2044 RVV、K1 RVV，以及K1 IME local fragment。

## 已从当前 DSL / Kernel IR 删除

- `W.block_axis`及对应旧block op/parser/verifier/lowering；
- 一般化任意轴乘加入口；
- `packed_i4_view`及其只能服务两个dot consumer的特殊路径；
- block形式的 `load_f16_le`；当前只保留完整闭合的scalar little-endian F16 load；
- `exp2`；它没有完整scalar/VLA target实现，也没有现有kernel消费者；
- pointer `address_space`；当前唯一地址类别无需伪参数；
- unary `exceptional` annotation；它没有独立可选语义；
- 未闭合的独立 widen、atomic、fence、permute、reshape、transpose与fill入口；
- 旧报告、旧lowering命名、旧example分类和兼容读取路径。

`W.cast`仍覆盖当前真实使用的逐元素 widening；`W.full/W.zeros`仍是有正式IR/verifier/target
实现的显式block constructor，不等同于已删除的假 `fill` 入口。

## 当前明确不支持

以下能力不会 fallback或假成功：

- nested VLA；
- grid/task/hart identity与GPU SIMT根模型；
- arbitrary reshape、transpose、axis permutation与隐式同-extent broadcast；
- 任意轴乘加；
- atomic与fence；
- strict VLA `exp/tanh`；
- 超出当前 typed shape/domain 的 lookup、decode、dot、matmul或extension组合；
- 一般 block reduction；当前非VLA block reduction只支持all-active rank-one i32 add；
- 没有合法RVV/IME target facts、resource candidate或instruction family的primitive；
- 非RISC-V target。

这些情况在 Python、Kernel IR verifier或RISC-V lowering中最早可知的位置明确失败，不进入
legacy、scalar、GGML、materials或旧emitter路径。

## 可手动执行的真实 repro

```bash
./examples/run/weft.sh sg2044-rvv128 add_bias
./examples/run/weft.sh k1-rvv256 add_bias
./examples/run/weft.sh k1-ime256 q4_0_projection_ime 1
```

每条命令都从 DSL kernel重新生成 Kernel IR、intrinsic C与C header，在指定RISC-V机器编译
object/executable并执行数值对照。
