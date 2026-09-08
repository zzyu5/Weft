# Weft

Weft 将数值 DSL 编译为可执行的 RISC-V kernel。编译主链只有 Canonical Kernel IR 与
RISC-V Physical IR 两层，随后将已选定的物理操作翻译为 intrinsic C / local asm，交给
系统 C 编译器。语言和编译器合同见 [`doc/index.md`](doc/index.md)。

应用可以在 RISC-V 本机通过 `weft.compile` / `weft.jit` 编译并调用 kernel；开发机生成
代码、目标机编译运行的显式部署方式也继续可用。两者共用同一编译主链。

## 构建编译器

需要 CMake 3.20 以上、Ninja、支持 C++17 的编译器，以及同一 LLVM 安装中的 LLVM/MLIR
CMake packages。配置时显式给出两者的位置：

```bash
cmake -S . -B build -G Ninja \
  -DLLVM_DIR=/path/to/llvm/lib/cmake/llvm \
  -DMLIR_DIR=/path/to/llvm/lib/cmake/mlir
cmake --build build
```

例如系统安装 LLVM 20 时，常见配置是：

```bash
cmake -S . -B build -G Ninja \
  -DLLVM_DIR=/usr/lib/llvm-20/lib/cmake/llvm \
  -DMLIR_DIR=/usr/lib/llvm-20/lib/cmake/mlir
cmake --build build
```

默认构建会生成：

```text
build/tools/weft-compile/weft-compile
build/tools/weft-opt/weft-opt
```

在 RISC-V 本机使用相同的构建命令；LLVM/MLIR 库、TableGen 工具与宿主 C++ 编译器
须能在该机运行。开发机上的 LLVM/MLIR 安装不能直接作为 RISC-V 原生构建依赖。

## RISC-V 本机编译与 JIT

原生运行需要 little-endian RISC-V Linux、系统已启用的完整 RVV、Python 3，以及能编译
RVV intrinsic 的本机 C 编译器。在目标机的仓库根目录查询实际硬件：

```bash
build/tools/weft-compile/weft-compile --query-native-target
```

该入口查询可用 ISA、进程 ABI、实际 `vlenb` 和允许执行的 CPU 集合；JIT 会自动调用它，
不需要用户手填 `--vlen-bits`。同一执行集合必须具有一致 VLEN，调用时还会检查当前线程
的向量状态和 CPU affinity。必要事实不可确认时明确报错。

Python API 使用现有 `@weft.kernel` 定义：

- `weft.compile(definition, options=..., toolchain=...)` 立即编译并返回可调用的产物。
- `weft.jit(definition, options=..., toolchain=...)` 在首次调用时编译，随后复用进程内产物。
- `weft.runtime.CompileOptions` 保存 source `meta` 与 physical bindings；`Toolchain`
  指定本机 `weft-compile`、C 编译器及 flags。
- `weft.runtime.Buffer` 借用连续 buffer；packed storage 要显式给出逻辑 shape 与 Encoding。
  指针和动态 shape 通过编译器生成的 ABI 传入，runtime 不隐式复制、repack 或创建 workspace。

原生服务在本机完成 DSL → 两层 IR → C → shared object → 加载 → 调用。同一编译身份
复用产物，改变代码生成绑定会生成对应产物；`CompiledKernel.close()` 释放加载句柄及
临时文件。缓存当前限于进程内，不是跨进程磁盘缓存。

现有 [原生 JIT repro](examples/repro/weft/native_jit.py) 使用 GEMM F32 和 Q4_0/Q8_0
两个真实 kernel，检查数值、同绑定复用及不同绑定专门化。以下命令直接在 RISC-V 本机运行，
将 C 编译器和 GGML 库路径替换为本机安装位置：

```bash
PYTHONPATH=python:examples python3 examples/repro/weft/native_jit.py \
  --compiler build/tools/weft-compile/weft-compile \
  --cc /path/to/clang \
  --ggml-lib-dir /path/to/ggml/build/bin
```

工具链需要额外参数时可重复传入 `--cflag=...`，例如 `--cflag=-fno-integrated-as`、
`--cflag=--gcc-toolchain=/path/to/gcc` 或本机动态库的 `-L` / rpath。这里的 GGML 仅供
repro 生成输入和数值对照；Weft runtime 本身不依赖 GGML、SSH 或远端 runner。

### VLEN 与能力边界

原生入口先发现实际 VLEN，再针对该 profile 编译；这不是一份二进制跨任意 VLEN 运行。
编译器按 VLEN、SEW、LMUL、lane 窗口和寄存器资源处理物理切片，不改变作者的数值分组。
完整 V profile 接受 128–65536 bits 范围内的二次幂，但这只是 target 合法域，不代表
所有 kernel 在所有 VLEN 上都已支持或实跑。SG2044/128 与 K1/256 已有原生 JIT 执行依据；
IQ2_XXS standalone/decode、TQ1_0 prefill 和第二输入 IQ2_XS 的 1024 配置已通过系统
编译器 object 编译，尚无 1024 数值执行依据。原生 vendor matrix 能力发现目前明确
unsupported，不猜测 IME 能力。完整合同见 [编译器总览](doc/compiler/index.md#6-原生-runtime-与编译产物)。

## 离线产物与显式远端运行

下面的入口供观察编译产物和显式远端部署使用；它们不是原生 JIT 失败后的自动 fallback。

### 生成 IR 与 C

先生成 Canonical Kernel IR，再生成并验证 RISC-V Physical IR：

```bash
PYTHONPATH=python:examples python -m weft \
  examples/kernels/dense/gemv_f32.py \
  > /tmp/weft-gemv.mlir

build/tools/weft-compile/weft-compile \
  /tmp/weft-gemv.mlir \
  --emit=riscv-ir \
  --march=rv64gcv_zfh_zfhmin_zvfh_zvfhmin_zfa_zba_zbb_zbs_zicbom_zicboz_zicbop_zicond_zawrs_zihintpause \
  --abi=lp64d \
  --vlen-bits=128 \
  --meta MR=4 \
  --meta KB=64 \
  -o /tmp/weft-gemv.riscv.mlir

build/tools/weft-opt/weft-opt \
  --canonicalize \
  --cse \
  --weft-riscv-verify-final \
  --verify-each \
  --verify-roundtrip \
  /tmp/weft-gemv.riscv.mlir \
  -o /tmp/weft-gemv.verified.mlir
```

生成 intrinsic C：

```bash
build/tools/weft-compile/weft-compile \
  /tmp/weft-gemv.mlir \
  --emit=intrinsic-c \
  --march=rv64gcv_zfh_zfhmin_zvfh_zvfhmin_zfa_zba_zbb_zbs_zicbom_zicboz_zicbop_zicond_zawrs_zihintpause \
  --abi=lp64d \
  --vlen-bits=128 \
  --meta MR=4 \
  --meta KB=64 \
  -o /tmp/weft-gemv.c
```

### 真机运行与性能对照

真机 runner 会完成 DSL lowering、intrinsic C 生成、远端 Clang 编译、数值检查与计时。
需要预先配置 SSH alias `rvv`（SG2044）和 `k1`（K1/X60），以及 runner 中记录的远端
GGML source/build 与 Clang 路径。正式性能运行使用 10 次重复：

```bash
cmake --build build

./examples/run/weft-kernel.sh sg2044 gemv_f32 10
./examples/run/weft-kernel.sh k1 ime_i8_contract 10

./examples/run/weft-quantized-vec-dot.sh sg2044 q4_k 10
./examples/run/weft-quantized-vec-dot.sh k1 q4_k 10

./examples/run/weft-row-dequantize.sh sg2044 q4_k 10
./examples/run/weft-row-dequantize.sh k1 q4_k 10

./examples/run/weft-mul-mat.sh sg2044 q4_k decode 10
./examples/run/weft-mul-mat.sh sg2044 q4_k prefill 10
./examples/run/weft-mul-mat.sh k1 q4_k decode 10
./examples/run/weft-mul-mat.sh k1 q4_k prefill 10
```

GGML baseline 使用相同 target、输入政策、Clang flags 和重复次数单独运行，例如：

```bash
./examples/run/ggml-kernel.sh sg2044 dequantize q4_K 10
./examples/run/ggml-kernel.sh k1 dequantize q4_K 10
```

生成 IR 或 C 成功不等于运行通过；结果必须在对应目标机运行，并按
[`doc/experiments/`](doc/experiments/index.md) 的数值与测量合同解释。

性能只维护三张表：[固定 baseline](report/baseline/ggml-riscv-kernel-performance.csv)、
[当前 Weft 性能](report/weft-kernel-performance.csv) 与
[明确对照表](report/kernel-performance-comparison.csv)。对照表的 `ratio = weft / source`，
受影响条目重测后标记为 `targeted-rerun`；未重测条目保留其原测量身份。
