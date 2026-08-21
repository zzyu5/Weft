# q4_K 地址复用与跨 group 流水

本轮只修改 q4_K 已有 `mac_pairs` / `mac_groups(n=4)` 局部物理实现的生成质量，
没有修改 encoding、作者树或 `std` 特化。两棵树都继续从同一组 grouped/layered
field facts 进入同一个 RVV grouped-MAC lowering。

## 生成结构

### 地址与坐标复用

instruction pass 现在把每次 grouped MAC 实际使用的 packed field 映射写入
`local_operation`：group 大小、layer 大小、layer 顺序、field bit offset，以及 rhs
的自然 unit-stride window。Emitter 不再为热循环中的每一个 term 展开：

```text
logical / 64
logical % 64
logical % 32
record base
activation base
```

每个 `subs(extent=32)` Level 只计算一次 logical base、packed byte base、nibble shift
和 activation base。完整 group 的热循环只使用两个递增指针和编译期常量偏移；尾部
不足一个 MAC group 的元素走单独 epilogue。

### 跨 group 软件流水

`pipeline_depth=2` 不再表示“同一个 group 先全部 load/decode、再全部 MAC”。Level
schedule pass 现在明确产生 `cross-iteration-double-buffer`，其生成形态是：

```text
prologue:     load/decode group 0 到 current bank
steady state: load/decode group n+1 到 next bank
              同时 MAC current bank 的 group n
              current/next 交换
epilogue:     MAC 最后一个 current bank
```

`pipeline_depth=1` 对应 `sequential-stream`。两种结构均由 schedule attribute 唯一决定，
emitter 只投影该决定。双银行的 current/next decoded operands 已进入统一寄存器预算。

Weft 已选择 `unroll=1`，因此 intrinsic C 在该局部循环前显式禁止系统 C 编译器再次
完全展开。若不保留这一决定，GCC 会把最多 16 个 pair group 全部展开，代码膨胀并把
吞吐降到约 1.13 GOP/s；这不是合法的“最终机器调度”，而是覆盖了 Weft 已选的局部
unroll 结构。

### 删除 prefetch 候选

`prefetch_distance` 已从 compiler options、CLI、候选构造、Level schedule、tune 入口和
emitter 一并删除。上一轮所有非零候选都退化，SG2044 的 GGML VLEN128 内层实现也没有
逐 term software prefetch；保留这一维只会生成已知低质量实现。

## SG2044 真实结果

硬件与计时协议保持为 SG2044 / RV64GCV / VLEN128、单线程、M=1、N=14336、K=4096、
64 MiB eviction、10 次 cold median。两项均为随机真实 GGML packed bytes 上的 bit-exact
结果。GGML 数字沿用已固定 baseline 的 9.027928 GOP/s，本轮没有重新运行 baseline。

| 作者树 | 本轮物理结构 | 上轮 Weft | 本轮 Weft | 相对上轮 | 相对 GGML |
|---|---|---:|---:|---:|---:|
| `mac_pairs` | address window reuse + cross-group double buffer | 6.034362 | **8.562299 GOP/s** / 13.716003 ms | +41.89% | 94.84% |
| `mac_groups(n=4)` | address window reuse + sequential stream | 6.753204 | **9.214137 GOP/s** / 12.745688 ms | +36.44% | 102.06% |

`mac_pairs` 相对 GGML 只剩 1.054× 的吞吐差距；`mac_groups(n=4)` 在同一计时范围内比
固定 GGML baseline 高 2.06%。更宽的 group 已占用更多 decoded vector value，给它增加
current/next 双银行会提高寄存器压力，因此该树的合法 winner 仍是顺序流；这不是格式或
kernel 分支，而是两份作者树经过同一 schedule/resource 机制得到的不同实例。

当前性能数字已写回 `report/weft-kernel-performance.csv`。
