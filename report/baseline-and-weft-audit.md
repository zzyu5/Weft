# GGML Baseline 与 Weft 端到端现状审查

## 结论

当前仓库里已经形成了一套可运行的 GGML 单算子性能基线，也已经存在一条真实的 Weft 编译与执行链。但是，这两件事的成熟度完全不同：

- GGML baseline 覆盖了 123 个逻辑 case、244 条硬件实测，包含真实 `GGML_OP_MUL_MAT` graph、RVV intrinsic、K1/X60 上的 IME1 汇编路径以及只有 scalar 实现的路径。它可以作为后续性能比较的数字底座，但仍有输入真实性、跨硬件覆盖和实现归因方面的缺口需要审查。
- SG2044 上确实测了 GEMM 语义的完整矩阵乘，decode 为 `M=1`，prefill 为 `M=128`；但 VLEN128 构建没有选择 GGML 的 packed `16x1` repack microkernel。它执行的是完整 `MUL_MAT` graph，核心计算却仍是逐行 RVV vec-dot，不能称为已经具备高性能 RVV GEMM microkernel。
- K1/X60 上的 `Q4_0`、`Q4_1`、`Q4_K` 确实进入了 production repack + IME1 asm 路径。当前没有 IME2 实测数据；源码里出现 IME2 文件不等于当前硬件上已经执行。
- Weft 不是伪端到端：Python DSL 可以经过 canonical MLIR、Selected IR、生成的 RVV C++、外部交叉编译，最终在 RISC-V 上执行并得到正确结果。但是它目前只是一个 correctness skeleton，不是已经成形的高性能 RISC-V 算子编译器。
- Weft 的生成端确实写出了 RVV intrinsic，但只覆盖很窄的模式，输出是 C++17 而不是独立的 intrinsic C 后端；当前没有 IME provider、没有 IME asm lowering，也没有把 `materials/` 中已有的量化、repack、fragment、exp 等高性能方法吸收到 provider 与发射器中。

本轮只整理和审查现状，没有修改 Weft 编译器实现，也没有改动 baseline 数字。

## 产物位置

| 文件 | 含义 |
|---|---|
| `report/baseline/ggml-riscv-kernel-baseline.md` | 按统一语义分类组织的可读性能报告 |
| `report/baseline/ggml-riscv-kernel-performance.csv` | 与报告对应的 244 条逐硬件性能记录 |
| `report/baseline-and-weft-audit.md` | 对 baseline 可信边界和 Weft 当前实现的审查 |

两份 baseline 产物现在固定保留在 `report/baseline/`，不再作为临时 report 清理。

## 一、GGML baseline 实际做了什么

### 1. 覆盖范围

baseline 按执行层次而不是按文件名混排：

| 执行层次 | 逻辑 case | 硬件实测 | 测量对象 |
|---|---:|---:|---|
| Production matrix multiplication | 52 | 104 | 26 种 weight format，各自包含 decode 与 prefill 的完整 `GGML_OP_MUL_MAT` graph |
| Quantized vec-dot projection helper | 24 | 48 | 直接调用 GGML vec-dot helper 的完整 decode projection |
| Activation quantize helper | 3 | 6 | `q8_0`、`q8_1`、`q8_K` 的整块 activation quantize |
| Row dequantize helper | 24 | 48 | 24 种格式的整块 row dequantize |
| Forward primitive | 20 | 38 | elementwise、normalization、layout、indexing、RoPE、flash attention 等单 op graph |
| **合计** | **123** | **244** | SG2044 与 K1/X60 的实际执行数据 |

这里的 123 是逻辑 case 数，244 是“case × 实际可运行硬件”的数据行。不是每个 case 都有两条：`get_rows_f32` 和 `concat_dim0` 当前只有 K1/X60 数据。

### 2. 统一测量语义

所有表格数字使用同一协议：

- 单线程、固定核心：SG2044 core 48，K1/X60 core 3。
- 先执行一次 warmup。
- 每个计时样本前遍历 64 MiB eviction buffer。
- 计时一次完整 kernel invocation，重复 10 次，记录 wall-time median。
- 编译、动态链接、tensor 初始化、weight quantize、weight repack、warmup 和 eviction 不计时。
- 矩阵乘和 vec-dot 使用 `2MNK` 计算 GOP/s；quantize/dequantize 使用 MElements/s。

因此，表里的时间不是进程冷启动时间，也不是第一次调用时间。runtime 当前打印的 `cold_median_us` 更准确的含义是“已经 warmup、经过 64 MiB cache displacement 后的一次 invocation”。报告正文已经统一描述了真实协议，但 runtime 字段中的 `cold` 命名容易造成误读。

64 MiB 遍历只是统一的 cache displacement 方法，不是硬件 cache flush 指令，也不保证每级 cache 的状态完全相同。

### 3. 输入规模

矩阵和张量 shape 采用真实 LLM 层级规模，而不是玩具小矩阵：

- `MUL_MAT`：Llama-8B hidden projection，`N=4096, K=4096`；decode `M=1`，prefill `M=128`。
- vec-dot helper：DeepSeek-R1-Distill-Llama-8B FFN-up decode，`M=1, N=14336, K=4096`。
- activation quantize：`M=128, K=14336`。
- row dequantize：`N=1024, K=4096`。
- forward primitives 使用 hidden、FFN、attention、embedding 等模型级 shape。

“真实模型级别”在当前 baseline 中指 shape 和工作量真实；输入内容仍是 runtime 合成的数据，不是从 GGUF tensor 原样读取的数据。

## 二、SG2044 上到底有没有 RVV GEMM

答案需要分成算子语义、GGML 调度和最终内核三个层次：

| 层次 | SG2044 当前事实 |
|---|---|
| 算子语义 | 有。测量对象是完整 `GGML_OP_MUL_MAT` graph，不是只测一个标量 dot |
| 矩阵规模 | 有 decode `1×4096×4096`，也有 prefill `128×4096×4096` |
| ISA | 核心 vec-dot 对绝大多数量化格式使用真实 RVV intrinsic；`nvfp4` 仍是 scalar |
| GGML packed repack | 没有被当前 SG2044 VLEN128 build 选择 |
| 高性能 GEMM microkernel | 当前证据没有。prefill 仍通过逐行 vec-dot 完成，而不是 packed multi-row/multi-column microkernel |

所以可以说“SG2044 有完整 RVV 矩阵乘执行”，不能说“SG2044 baseline 已经包含高性能 packed RVV GEMM kernel”。例如 `q4_K` 的 SG2044 数据是：

| phase | time | throughput | 实际路径 |
|---|---:|---:|---|
| decode `M=1` | 3.760 ms | 8.923 GOP/s | RVV quantized vec-dot |
| prefill `M=128` | 444.056 ms | 9.672 GOP/s | 完整 `MUL_MAT` graph，内部重复 RVV vec-dot |

K1/X60 则是另一条物理实现：

| format | decode | prefill | 实际路径 |
|---|---:|---:|---|
| `Q4_0` | 2.941 ms / 11.409 GOP/s | 151.069 ms / 28.431 GOP/s | production repack + IME1 asm |
| `Q4_1` | 3.519 ms / 9.535 GOP/s | 174.724 ms / 24.581 GOP/s | production repack + IME1 asm |
| `Q4_K` | 3.436 ms / 9.766 GOP/s | 174.758 ms / 24.577 GOP/s | production repack + IME1 asm |

这些 K1 数字不包含 weight repack 时间，但 production graph 内部需要的 activation quantize 属于被计时的计算路径。因此它们适合作为 steady-state 单层计算基线，不是“从原始 tensor 到最终输出”的全流程延迟。

## 三、baseline 中已经可靠的部分

1. **真实执行而不是源码盘点。** 每一条保留数据都来自 SG2044 或 K1/X60 上的实际 runner；仅仅在源码中发现一个 RVV/IME 文件不会进入性能表。
2. **production graph 与 helper 分开。** `MUL_MAT` 表回答最终 GGML graph 执行速度；vec-dot、quantize、dequantize 表回答可独立比较的叶子 primitive，二者没有混成一个“kernel”口径。
3. **两种矩阵工作区间分开。** decode 与 prefill 使用相同 `N,K`，只改变 `M`，不会把 GEMV 和矩阵批量计算混成一个数字。
4. **scalar 没有被隐藏。** 没有 RVV 实现的 `nvfp4`、若干 dequantize、forward path 明确标为 scalar，而不是笼统写成 RISC-V optimized。
5. **IME1 是真实运行路径。** K1 的三个 Q4 production `MUL_MAT` 记录的是 special buffer/repack 后进入 IME1 asm 的执行，不是把普通 RVV 结果改标签。
6. **硬件归属清楚。** SG2044 是 VLEN128 标准 RVV；K1/X60 是 VLEN256，并同时存在标准 RVV、SpacemiT RVV/inline asm 和 IME1。

## 四、baseline 需要审查的问题

以下问题不会让现有数字失效，但决定了以后能怎样解释和比较它们。

### 1. 两台机器不是同一份 GGML 实现的纯硬件对比

`examples/run/ggml-kernel.sh` 在 SG2044 上链接 `/home/ubuntu/llama.cpp-upstream-native`，在 K1 上链接 `/home/bianbu/tcrv-k1-llama`。后者包含 SpacemiT 专有 RVV、repack 和 IME 路径。

因此横向表的含义是“每台目标机当前最好且可运行的 GGML 路径”，不是“相同二进制只更换 CPU”的硬件微架构对比。这个口径适合作为 Weft 各目标后端要追赶的 target-specific baseline。

### 2. 部分 IQ production 输入是零填充合法块

`examples/repro/ggml/mul_mat_runtime.cpp` 中 `iq1_s`、`iq1_m`、`iq2_s`、`iq2_xs`、`iq2_xxs`、`iq3_s`、`iq3_xxs` 没有对应 quantize function；runtime 初始化的 weight row 保持为零，再复制到全部输出行。

这些结果证明相应 GGML dispatch 和内核能够执行，也能测出固定控制路径成本，但不能代表真实 IQ GGUF 权重的 codebook、scale、sign 分布。它们应被理解为“模型级 shape 的路径性能”，不是“真实模型 tensor 数据分布上的性能”。

### 3. 其余权重也重复同一行

有 quantizer 的格式使用一个合成 `K=4096` source row 量化，然后把同一量化行复制为全部 `N=4096` 行。工作量和内存规模真实，但数据重复性可能影响 cache、压缩元数据和分支行为。当前 baseline 的首要用途是固定可重复的单 kernel 比较，不是模拟完整 GGUF 的统计分布。

### 4. 两个 K1-only forward case 还不能解释为 SG2044 不支持

`get_rows_f32` 和 `concat_dim0` 只被放进 `GGML_BASELINE_K1` 的 case table；构图代码本身并没有限制它们只能在 K1 运行。当前缺少 SG2044 数字，更像 runner coverage 缺口，而不是已经证明 SG2044 无法执行。

因此当前 244 条数据完整对应“已经登记并跑过的 manifest”，但严格按照“能在两台硬件运行就都记录”的标准，这两项仍未闭合。

### 5. correctness 检查较弱

production `MUL_MAT`、forward、vec-dot 和 dequantize runtime 主要检查 graph 成功以及输出为 finite；没有统一与参考实现逐元素比较。baseline 的性能数据是真实执行时间，但不能单独作为数值正确性的充分证据。

这不是要求新增测试体系，而是对数字含义的边界说明：它是性能 baseline，不是 correctness certification。

### 6. implementation 标签不是统一的运行时 introspection

多数 `implementation` 字段来自 runtime case table 和已知 target build，不是每次从 GGML dispatch 返回实际选中函数。IME1 repack 日志提供了更强的运行证据；普通 RVV/scalar 标签主要依据编译目标和源码分派。以后比较 Weft 时，应以对应源码路径与实际执行日志解释，不能只依赖字符串标签。

### 7. 没有 IME2 数据是有意的事实空缺

本地 GGML source 中存在 `ime2_kernels.cpp`，但当前 K1/X60 实测目标是 IME1，SG2044 没有 IME。baseline 只记录真实运行过的 IME1，不把 source availability 当作硬件 performance。

## 五、Weft 当前端到端到底是什么

### 1. 实际链条

当前真实路径是：

```text
Python kernel DSL
  -> Python AST frontend
  -> canonical Weft Kernel MLIR
  -> C++ selection
  -> Selected IR
  -> generated C++17 source
  -> remote clang++ cross compilation
  -> RISC-V relocatable object / executable
  -> target runtime
```

其中 `weft-compile` 自身只支持 `canonical-mlir`、`selected-mlir` 和 `source` 三种输出；object 不是编译器当前直接产物，而是 `examples/run/weft.sh` 把生成源码送到目标机后调用 clang++ 得到的。

这条链不是直接调用 GGML 来冒充 Weft。以 `q4_K_q8_K` runtime 为例，`run_weft` 调用生成的 Weft symbol，`run_ggml` 只作为独立参考路径；两者链接在同一个 repro 中是为了正确性与性能对比。

### 2. 已经实际跑通的代表性语义

| kernel 类别 | 当前实证 | 结论 |
|---|---|---|
| `add_bias` VLA elementwise | 生成 RVV source、RISC-V object，运行 PASS | 基础 VLA elementwise 链闭合 |
| `rms_norm` | 运行 PASS | reduce + elementwise 的窄路径闭合 |
| `online_softmax` | 运行 PASS | summary/reduction 组合存在可执行路径 |
| blocked GEMM | `M=17,N=18,K=19` correctness PASS | contract 能生成并执行，但只是玩具 correctness repro |
| `q4_0/q4_1/q5_0/q5_1/q8_0` block-dot | 运行 PASS | DSL 能表达若干 unpack、widen、reduce 组合 |
| `q4_K_q8_K` | 真实 `4096×4096` decode，误差为零 | 量化端到端是真的，但性能远未达到 GGML |

本轮重新执行 `q4_K_q8_K`、DeepSeek-R1-Distill-Llama-8B `attn_q` decode，单线程、`M=1,N=4096,K=4096`、3 次重复，得到：

| implementation | time | throughput | max abs / rel error |
|---|---:|---:|---:|
| Weft generated path | 264.777 ms | 0.127 GOP/s | 0 / 0 |
| GGML RVV reference | 3.473 ms | 9.663 GOP/s | — |

Weft 吞吐约为 GGML 的 1.31%，GGML 约快 76 倍。这个结果最准确地说明了当前状态：算法语义和执行链可以闭合，但没有吸收高性能量化发射方法。

## 六、为什么当前 Weft 仍然不是目标中的算子编译器

### 1. selection 仍是集中式模式分类器

设计要求是 operation-local provider：每个 primitive 根据 target、type、layout、mask 和资源约束提供候选，selection 只在合法候选中做物理选择。当前 `lib/Compiler/Selection.cpp` 仍集中识别整段 VLA/body/contract 形状，再构造 scalar 或 RVV candidate，最后用 `candidates.back()` 选择后加入的 RVV candidate。

它没有按 kernel 名称 dispatch，这是好的一面；但它仍按“整个 region 长什么样”分类，而不是让 load、decode、dot、reduce、store 等 primitive 各自拥有可组合 provider。新量化格式或混合结构仍容易演化成新的 recognizer 分支。

### 2. Selected IR 中的物理决策大多是占位值

- RVV VLA 当前固定 `SEW=32`、`LMUL=m1`、`unroll=1`。
- contract 的 `micro_m/micro_n/micro_k` 固定为 `1×1×1`。
- source emitter 没有真正消费 microtile 属性来形成寄存器 tile。
- target profile 只包含基础 triple、march、ABI、XLEN、endianness、RVV、VLEN 和 vector register 数，尚不足以表达量化 block、LMUL 压力、fragment、repack、IME 能力和资源约束。

因此 canonical/selected 两层形式已经存在，但“Selected IR 只保存多个合法方案中真正选出的物理决策”这一核心还没有落实。

### 3. blocked GEMM 只是正确性 lowering

当前 RVV contract emitter 对每个输出元素 `(m,n)`：

1. 创建两个 `std::vector<float>` 临时 slice；
2. 用标量循环把 lhs/rhs 对应 K 轴复制进去；
3. 对临时 slice 使用 e32m1 `vfmul` 与 `vfredusum`；
4. 把单个结果写回。

它没有 MR×NR 寄存器 tile、没有 packed panel、没有多 accumulator、没有按 LMUL/寄存器预算做选择。虽然内部出现 RVV intrinsic，但这不是高性能 GEMM 发射，甚至每个输出元素的动态临时容器都引入了不应存在的开销。

### 4. intrinsic C 的判断需要更准确

“完全没有写 intrinsic”并不准确：`lib/Compiler/SourceEmitter.cpp` 已经生成 `<riscv_vector.h>` 调用，部分 VLA、block op、reduce 和 contract 会输出真实 RVV intrinsic。

但用户担心的核心是成立的：

- 输出是依赖 C++ runtime wrapper 和 `std::vector` 的 C++17 translation unit，不是清晰独立的 intrinsic C kernel backend。
- RVV 路径主要限制在 all-active e32m1 或少量固定 block/type 模式。
- `while`、`scan`、lookup/decode/widen/narrow/permute、`block_scaled_contract` 等已经进入 DSL/schema 的语义没有完整 source lowering。
- 没有 IME provider，没有 inline asm/IME leaf emitter。
- 没有把 microtile、LMUL、packing、fragment 作为可搜索和可消费的物理选择。

所以现在有“能证明 lowering 发生的 RVV intrinsic”，没有“针对真实算子的高性能 intrinsic C 发射体系”。

### 5. object 与普通 artifact 还没有成为编译器合同

架构文档要求 source/object 都是 Selected IR 的机械产物。当前 `weft-compile` 不支持 `--emit=object`，object 由 repro shell 额外调用远端编译器生成。对于当前手工 repro 足够，但还不能声称 Weft 已经独立完成普通 AOT artifact pipeline。

## 七、`materials/` 中高性能方法的利用情况

`materials/` 的正确定位是只读知识 donor：不能在当前 build/link/runtime 中调用旧实现，但应该把其中已经验证过的算法分解、layout 事实、资源公式和 intrinsic/asm spelling 重新长进当前 provider。现状是这些关键方法几乎尚未进入 Weft：

| donor | 已有高性能方法 | 当前 Weft 缺口 |
|---|---|---|
| `materials/legacy-source/lib/Conversion/RVV/RVVToEmitCFlatBlockDotPrimitives.cpp` | Q1/Q4 block-dot、mask-packed sign、i8 运算、widening reduction、VLEN/LMUL 与 boolean ratio | 当前只有基础 unpack/reduce 表达，没有同等级 quant block-dot provider |
| `materials/legacy-source/lib/Conversion/RVV/RVVToEmitCRepackKQuantQ4.cpp` | q4_K ×16 repack GEMV、16-column interleave、8×32 subblock、scale/min 分离累加、i16→i32 | 没有 repack layout/provider，也没有多列寄存器 kernel |
| `materials/legacy-source/lib/Conversion/RVV/RVVToEmitCKQuant.cpp` | q6_K 6-bit unpack、aux8/aux32、按 LMUL 选择 strip、`vslidedown` fold | 没有对应量化 unpack 与资源选择 |
| `materials/legacy-source/lib/Conversion/RVV/RVVToEmitCRepackGrid.cpp` | IQ codebook/sign table、`vluxei16` gather、显式 repack layout | 没有 lookup/gather 的完整 source lowering，更没有 IQ repack provider |
| `materials/legacy-source/lib/Target/IME/SelectedExecutionIMESource.cpp` | fragment-tiled loop、IME `vmadot`/widening inline asm leaf | 没有 IME target capability、provider 或 emitter |
| `materials/legacy-source/lib/Target/RVV/SelectedExecutionRVVSource.cpp` | RVV exp polynomial、mask 与特殊值修复 | softmax 仍依赖通用/标量数学路径，没有高性能 RVV exp provider |

此外，当前 `source/c/ggml/llama.cpp` 中也保存了实际 baseline 使用的 RISC-V 路径，包括标准 RVV quants/repack 和 SpacemiT RVV/IME 实现。它们应该作为目标性能与实现方法的参照，但不应被 Weft 直接调用或包装成后端。

问题不是旧代码没有可复用内容，而是当前重建只复用了“canonical -> selected -> source”的编译器外形，没有把旧实现中最有价值的 primitive 分解、layout、资源选择和 intrinsic/asm leaf 纳入新 provider 架构。

## 八、最终判断

| 对象 | 当前可接受的说法 | 当前不能接受的说法 |
|---|---|---|
| GGML baseline | 已有覆盖两台 RISC-V 机器、RVV、scalar 和 IME1 的 244 条统一协议性能数据 | 所有数据都来自相同 GGML 实现；所有输入都是真实 GGUF tensor；所有 case 都已双机闭合 |
| SG2044 GEMM | 已运行完整 `MUL_MAT` graph，decode/prefill 都有真实 RVV 计算 | 已拥有 packed、多行多列的高性能 RVV GEMM microkernel |
| K1 IME | Q4 production graph 已进入 repack + IME1 asm，并有实测性能 | 当前已有 IME2 实测或 Weft IME lowering |
| Weft end-to-end | DSL 到 RISC-V source/object/runtime 的 correctness 链真实存在 | 已经是接近 GGML 性能的 RISC-V kernel compiler |
| RVV emission | 生成端已能写出一部分真实 RVV intrinsic | 已有完整、高性能、纯 intrinsic C backend |
| materials 复用 | 旧代码可作为 provider、layout、资源公式和 leaf spelling 的知识 donor | 当前 Weft 已经吸收了这些高性能方法 |

最核心的审查结论是：baseline 已经是一份有明确测量口径的性能底表，但有四个需要明确接受或补齐的边界——双机不是同一 GGML 实现、部分 IQ 是零填充输入、两个 forward case 未双机闭合、correctness 只做了弱检查。Weft 则不是“完全没做”，而是先闭合了正确性链；它目前真正错误的地方在 selection/provider 与发射质量，尚未把设计中的物理选择和历史高性能知识变成可组合的 RISC-V lowering。
