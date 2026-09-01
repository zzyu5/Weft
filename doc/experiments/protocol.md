# 正确性与性能测量协议

## 1. Correctness

### Encoding 与离散 Storage

bit-exact只用于同一packed bytes、bit/byte order、padding、grouped/layered/joined/bit-plane mapping、离散field，以及定义为exact wrap/saturate的整数结果。它用于发现Encoding错位、field拼接和ABI错误。

### 浮点 Kernel

浮点输出检查finite/NaN/Inf政策、case预先声明的absolute/relative tolerance，以及符号、数量级、全零和明显量化错误。合法FMA contraction、reduction顺序或closed primitive partial可以产生末位差异。

不能为了float bit-exact交换作者scale tree、插入人为物化阻断contraction或修改合法leaf。tolerance必须在运行前定义，不能看到结果后放宽；数值失败不记录性能。

## 2. Toolchain

Weft generated C、runtime/harness和GGML baseline wrapper使用同一套Clang toolchain与共同flags。每次记录：

```text
Clang executable与完整版本
-O level
-ffp-contract policy
-march / target extensions
-mabi
LTO、fast-math及其它数值/代码生成flags
assembler / linker（IME需要时）
```

正式公共口径为`-O3 -ffp-contract=fast -mabi=lp64d`，`-march`与extension flags按target profile设置。不能让Weft开启contraction而baseline关闭，也不能用Clang编译Weft、用GCC编译baseline后给出比值。

Clang版本、target flags、assembler或数值flags改变时，相关baseline必须重编重测。generated C中的intrinsic、typed asm和保护已选physical形态的局部pragma属于编译器输出；公平性要求相同toolchain、target capability与数值政策，不要求两份source文本相同。

## 3. Target Machines

| target | 实验环境事实 | 正式路径 | 测量方式 |
|---|---|---|---|
| SG2044 | 48 cores，VLEN128 | 标准RVV | 单线程，固定目标core |
| K1/X60 | baseline环境记录3个可用cores，VLEN256 | 标准RVV、SpacemiT RVV、IME1 | 单线程，固定目标core |

两台机器不直接比较绝对速度。性能比只能是SG Weft对SG baseline、K1 Weft对K1 baseline；跨target只检查同一DSL tree是否形成各自合法layout、memory form、leaf与schedule。

`rvv-v100`是备用VLEN256标准RVV环境，不是NVIDIA V100，也不是正式performance baseline。只有重建相同toolchain、source baseline与测量协议后才可进入正式CSV；toolchain不能生成IME时不得报告IME。

每次运行记录OS/kernel、CPU identity、VLEN/extension probe、目标core、frequency/governor和toolchain。事实变化时建立新结果集，不能沿用旧baseline。

## 4. Kernel Timing

正式单线程kernel协议：

1. 初始化相同inputs与persistent artifacts；
2. 一次不计时warmup；
3. 每次timed invocation前遍历64 MiB eviction buffer；
4. 记录10次wall time；
5. 报告median；
6. 编译、动态链接、tensor初始化、persistent weight quantize/repack、warmup和eviction不计时。

activation quantize、kernel内local pack或其它operator-invocation工作，若baseline在timed graph内执行，Weft也必须计入。只有双方都属于跨调用persistent artifact phase时才可排除。每个case必须写明timing scope。

matrix/vec-dot按`2MNK`计算GOP/s；quantize/dequantize按logical elements计算MElements/s；没有一致work定义时只报告wall time。

模型级`llama-bench`沿用真实graph、线程和自身计时协议，不与kernel microbenchmark混在同一分布。

## 5. Result Files

- `report/baseline/ggml-riscv-kernel-performance.csv`：当前协议下固定的GGML baseline；
- `report/weft-kernel-performance.csv`：由当前checkout完整重测固定manifest后得到的Weft结果快照。

baseline只在机器、toolchain、flags、算法/shape或测量协议变化时重测；Weft compiler修改不触发baseline重跑。历史compiler结果、旧协议行和`materials/experiments`不得混入当前表。

Weft结果表只能由一次完整快照运行整体替换，不能按受影响case局部覆盖旧行。快照运行必须从
干净worktree执行默认`cmake --build build`，使用本次生成的`weft-compile`完成整个固定manifest，
先写临时文件，全部case成功后再原子替换正式CSV；任一case编译、数值或字段合同失败时正式表保持
不变。单轮定向回归只写入当轮`report/`报告，不写正式CSV。这样“当前表”始终表示同一checkout、
同一build和同一协议下的完整结果，而不是不同compiler快照的逐行拼接。

每行至少保存case identity、hardware/ISA、shape、timing scope、compiler/version、flags、physical configuration、repetitions、correctness、absolute/relative error、median与throughput。合同字段缺失的运行记录不得计算baseline ratio。
