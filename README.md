# 编译

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

# 测试

先生成 Canonical Kernel IR，再生成并验证 RISC-V Physical IR：

```bash
PYTHONPATH=python:examples/kernels python -m weft \
  examples/kernels/dense/gemv.py \
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

生成 IR 或 C 成功不等于测试通过；结果必须在对应目标机运行，并按
[`doc/experiments/`](doc/experiments/index.md) 的数值与测量合同解释。
