# 编译与生成结果

## Python DSL kernel 到 Kernel IR

```bash
PYTHONPATH=python python3 -m weft \
  examples/kernels/elementwise/binary.py --kernel add_f32
```

包含多个 `@weft.kernel` 时使用 `--kernel NAME` 选择一个 entry。`W.constexpr` 参数在生成
intrinsic C 前用 `--meta=NAME=INTEGER` 绑定。

## Kernel IR 到 intrinsic C

```bash
PYTHONPATH=python python3 -m weft \
  examples/kernels/elementwise/binary.py --kernel add_f32 |
  build/tools/weft-compile/weft-compile \
    --emit=intrinsic-c \
    --march=rv64gcv_zfh_zvfh \
    --abi=lp64d \
    --vlen-bits=256 \
    --matrix-extension=none \
    --header=/tmp/kernel.h \
    -o /tmp/kernel.c
```

`weft-compile`解析并验证Kernel IR，收集typed facts，完成唯一合法性推导、结构候选、参数实例、
target/resource过滤与selected physical decisions，然后生成intrinsic C与C header。没有旧IR
reader、兼容入口、备用emitter或silent scalar路径。

Kernel body生成前，lowering已经选定每个value shape、每个`(consumer,value)` handoff、segment
field/SEW/shape、VLA state与structured primitive的物理实现，并收集本模块实际使用的exact leaf。
Prelude只输出这些leaf所需的RVV helper或typed local asm；
例如32-lane与64-lane codebook dot、VLEN128与VLEN256 grouped dot是不同leaf，不由C生成阶段
检查VLEN后再分派。

## C header

C header 直接给出：

- entry declaration；
- pointer 的 access、alignment、alias、storage class 与 persistent format metadata；
- workspace/persistent storage 的 rank、每维 extent 和 element-count 查询函数。

Runtime 用查询函数分配 workspace 或 persistent data；不能再复制一份 shape 公式。Header 不定义
第二套 kernel 语义。

## object 与 executable

Intrinsic C 交给目标 C 编译器：

```bash
riscv64-linux-gnu-gcc -O3 -std=c11 -march=rv64gcv_zfh_zvfh \
  -mabi=lp64d -include /tmp/kernel.h -c /tmp/kernel.c -o /tmp/kernel.o
```

调用者再把 object 与自己的 runtime 链成 executable 或 library。Python、MLIR 与 Weft package
不参与目标机运行。

`examples/run/weft.sh` 把 DSL kernel、生成的 intrinsic C/C header 和对应 runtime 放进临时目录，
在指定 RISC-V 机器编译并执行；临时文件不会进入仓库。
