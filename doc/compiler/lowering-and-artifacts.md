# Lowering 与 Artifact

## 21. Lowering 与 artifact

### 21.1 瞬态 owner IR

RVV、矩阵扩展、vendor extension 的 typed lowering IR 可以存在，但只能：

- 从 canonical + selected execution 单向生成；
- 不新增 selection；
- 不成为 source front door；
- 不反向驱动 canonical IR；
- 不成为第四份长期 authority。

### 21.2 Emitter 只是 Selected Execution 的投影

Emitter 的完整事实来源只有：

```text
Canonical Kernel IR + Selected Execution IR + static target facts
```

它可以按 canonical op kind 和已经选定的 provider/strategy 分派 target spelling，但不得：

- 根据 kernel 名、算子名、q-format 或整体 shape 选择完整模板；
- 从 tensor shape 猜测 source loop count、state boundary 或 contraction identity；
- 在 emission 时重新选择 LMUL、microtile、memory strategy、packing 或 fragment；
- 从 analysis cache、旧 route registry 或 target helper 读取第三份算法/计划事实；
- 对未支持的 selected primitive 使用 catch-all 或默认 emitter。

遇到无法投影的合法 selected record，emission 必须明确 unsupported；不能退回另一条旧路径
或临时重新做 selection。

### 21.3 Artifact 类型

Weft compiler 至少支持：

```text
canonical MLIR
selected MLIR
readable generated source（可选）
relocatable object
static library
C-compatible public header
```

### 21.4 Runtime ABI

生成 entry 必须具有普通 C-compatible ABI。

```c
void weft_rms_norm(
    const float *x,
    const float *weight,
    float *y,
    int64_t row_begin,
    int64_t row_end,
    int64_t cols,
    int64_t stride);
```

以上只是示例；实际 work descriptor 由 kernel source 决定。

Weft artifact 不创建线程。外部 runtime 可以：

```cpp
parallel_for(worker_ranges, [&](auto range) {
    weft_rms_norm(..., range.begin, range.end, ...);
});
```

### 21.5 Multiversion dispatch

可选 AOT dispatcher 可以检查：

- shape predicate；
- alignment / stride class；
- target extension；
- fixed VLEN guard；

并调用已编译 variant。

Dispatcher 不管理线程，也不调用 compiler。

---
