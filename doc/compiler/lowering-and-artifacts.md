# Lowering 与 Artifact

## 21. Lowering 与 artifact

### 21.1 瞬态 owner IR

RVV、矩阵扩展、vendor extension 的 typed lowering IR 可以存在，但只能：

- 从 canonical + selected execution 单向生成；
- 不新增 selection；
- 不成为 source front door；
- 不反向驱动 canonical IR；
- 不成为第四份长期 authority。

### 21.2 Artifact 类型

Weft compiler 至少支持：

```text
canonical MLIR
selected MLIR
readable generated source（可选）
relocatable object
static library
C-compatible public header
```

### 21.3 Runtime ABI

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

### 21.4 Multiversion dispatch

可选 AOT dispatcher 可以检查：

- shape predicate；
- alignment / stride class；
- target extension；
- fixed VLEN guard；

并调用已编译 variant。

Dispatcher 不管理线程，也不调用 compiler。

---
