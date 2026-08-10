# 第一里程碑：完整 Python DSL 与 Canonical Kernel IR

> 本文定义并记录第一项实现工作的工程边界。该里程碑已经落地；语言语义仍以
> `WEFT_FINAL_SPEC.md` 为唯一权威。

## 1. 裁决

第一里程碑已经完成完整的 Weft Python reference frontend；“完整 Python DSL”同时包含
与它一一对应的 canonical Kernel dialect、types 和 verifier：

```text
Python source
  -> transient AST / source environment / frontend values
  -> canonical Weft Kernel MLIR
  -> parse + canonical verification
```

第一里程碑不做 Selected Execution IR、provider、Scalar/RVV/IME lowering、artifact、tuning
或 Intent bridge。它先把作者能够表达的算法语义完整钉住，阻止 backend 反向塑造语言。

这不是“先写一层 Python API，未来再找 IR 接住”。每个 public construct 只有在能够生成
对应 canonical op/type/region，并由 canonical verifier 检查时，才算实现。

## 2. 为什么这样排

- Python DSL 是规范化参考 frontend，其他 frontend 未来也要生成同一 canonical IR；先把
  source/canonical 合同做全，才能得到稳定的外部编译边界。
- 当前最危险的错误不是缺少 RVV intrinsic，而是再次把 grid、physical tile、mask tail 或
  whole-kernel route 写进算法表示。
- 完整 frontend 可以用最终规范中的六个完整示例直接验收，不需要先决定 LMUL、microtile
  或 IME fragment。
- Provider 与 emitter 只应该消费已稳定的算法语义；让它们与语言同时生长，会诱使实现
  为了某条现成硬件路径删减 masked value、state algebra 或 numerical attributes。

## 3. “完整”的精确定义

第一里程碑同时满足以下条件才完成：

1. `weft` 与 `weft.language` 暴露最终规范要求的完整核心 source surface；
2. 每个 API 都完成 Python AST/type checking 到 canonical MLIR 的 lowering；
3. canonical dialect 能独立 parse、print、verify，不依赖 Python runtime；
4. source location、ABI、types、regions、effects、numerical attributes 全部进入 canonical facts；
5. 不存在 Python 侧长期 op schema、长期 typed IR、独立 verifier 或第二份算法 authority；
6. 最终规范的六个完整示例均能生成并通过 canonical verifier；
7. 规范列出的非法 VLA、masked value、state、shape、constexpr 和 extension 情况能明确失败；
8. canonical IR 中不存在 grid/task identity、`vl`、LMUL、layout、register tile、provider ID
   或 instruction spelling。

“完整”只表示 source→canonical 的语言闭环，不表示已经能生成 RISC-V object。

### 3.1 当前实现证据

- `python/weft/`：AOT definition、public language、source compiler 与 generic assembly builder；
- `include/Weft/Dialect/{Kernel,Extension}/IR/`：TableGen schema；
- `lib/Dialect/{Kernel,Extension}/IR/`：canonical verifier；
- `tools/weft-opt/`：独立方言注册、parse、print、verify；
- `source/weft/weft/` 下对应最终规范的六个 source 文件：六个规范程序。

本机 LLVM/MLIR 20.1.8 已完成 CMake/TableGen/C++ build；六个示例均由活动 Python package
生成 MLIR，并由活动 `weft-opt` 独立解析与验证。没有使用 materials 中的旧 executable。

## 4. Public Python surface

### 4.1 Kernel、types 与 annotations

必须实现：

```python
@weft.kernel
def kernel(...): ...

W.i1
W.i8, W.i16, W.i32, W.i64
W.u8, W.u16, W.u32, W.u64
W.f16, W.bf16, W.f32, W.f64
W.index
W.ptr[T]
W.constexpr[T]
```

Pointer annotations 需要表达 address space、readonly/writeonly、noalias、minimum alignment
和 restrict-like ownership。`constexpr` 必须与 runtime scalar 在 canonical 类型上区分。

### 4.2 Scalar control 与 helpers

必须实现普通 Python expression、scalar `if`、有序 `W.range`、受限 `while`/early exit、
`W.select` 和 `@W.pure` helper。Effectful helper 必须有显式 effect contract；reflection、
文件 I/O、异常、generator 和动态 Python object 不属于语言。

普通 sequential carry 只通过 scalar control 表达，不能被自动改写成 reduce。

### 4.3 VLA

```python
with W.vla(begin, end) as i:
    ...
```

`i` 是 logical VLA coordinate，不是 scalar induction variable 或 lane ID。Frontend 与
canonical verifier 必须共同保证 lexical scope、单一活跃 VLA axis、value escape、外层
state mutation 和 effect independence 规则。

### 4.4 Memory、effect 与 validity

```python
W.load(ptr, *, where=True, other=W.invalid, alignment=None)
W.store(ptr, value, *, where=True, alignment=None)
W.prefetch(ptr, *, where=True, locality="default")
W.atomic_add(ptr, value, *, where=True, order="relaxed")
W.fence(order="acq_rel")

W.valid(masked_value)
W.fill(masked_value, fill_value)
W.select(predicate, a, b)
```

不带显式 `other` 的 masked load 必须生成一等 validity-carrying value。Frontend 不得把它
提前压成 RVV mask 或普通 filled block；consumer legality 由 canonical verifier 负责。

### 4.5 Logical block

```python
W.block_axis(extent, offset=0)
W.full(shape, value, dtype=None)
W.zeros(shape, dtype)
W.expand_dims(value, axis)
W.broadcast_to(value, shape)
W.reshape(value, shape)
W.transpose(value, permutation)
```

同时支持规范示例使用的 `value[:, None]` / `value[None, :]` slicing sugar。Block 只携带
logical shape/axis identity，不携带 physical layout、microtile 或 fragment。

### 4.6 State algebra

必须分别实现：

- `W.reduce`；
- `W.scan`；
- `W.summary_fold`；
- scalar `W.range` 中的 ordinary sequential carry。

它们必须拥有不同 canonical op/region contract。`summary_fold` 的 `lift/merge/finalize`、
tuple state、purity 和 capture 必须在 canonical IR 中可见；不能用一个字符串 kind 塞进
通用 loop。

### 4.7 Structured compute、pointwise 与 numerics

必须实现 `W.contract`、`W.dot` sugar、`W.permute`、`W.lookup`、`W.decode`、`W.widen`、
`W.narrow`，以及规范完整示例所使用的 cast、tuple、maximum、exp、rsqrt、negative infinity
等 typed pointwise/math helpers。

Canonical facts 必须保存 input/output/accumulator dtype、`order`、`math`、rounding、
saturation/wrap 和适用的 exceptional-value policy。`W.dot` 必须统一 lowering 成 contract，
不能拥有独立算法 authority。

### 4.8 Extension surface

第一里程碑已经通过 dialect registry 装载 typed local semantic extension，并让规范中的
`W.block_scaled_contract` 示例形成独立 sibling canonical op。它验证 extension 可以增加
局部新语义，但不能注册 whole-kernel/operator API 或只有 Python callback 的无 schema op。

## 5. 当前 Python signature ↔ canonical 裁决

以下是最终规范允许工程选择之处的当前实现事实，不构成第二份语言规范：

| Source surface | Canonical 形态 |
|---|---|
| `W.ptr[T, W.readonly/writeonly, W.noalias, W.aligned(N), W.restrict, W.address_space(S)]` | `!weft_kernel.ptr` 保存 element/address/access/noalias/alignment/restrict-like |
| `W.constexpr[T]` | ABI 使用 `!weft_kernel.constexpr<T>`；body 由 `meta_value` 显式物化 specialization value |
| Python scalar `if` / 受限 `while` / `W.range` | `weft_kernel.if` / `while` / `for`，全部使用显式 SSA carry region |
| `@W.pure` | helper AST 在使用点或 state region 内联；不得产生 effect |
| `@W.helper(effects=(...))` | effect set 只允许 `read/write/atomic/fence`；helper 内联后保留真实 canonical effect ops |
| `W.load(..., other=W.invalid)` | `!weft_kernel.masked<value-type>`；显式 `other` 返回普通 value |
| `W.tuple` 与 destructuring | `!weft_kernel.tuple<[...]>` + `tuple`/`tuple_get` |
| `W.cast/maximum/minimum/exp/exp2/log/rsqrt/neg_inf` | typed pointwise、math 或 special-value canonical op |
| `W.permute(value, permutation)` | `weft_kernel.permute`，与 block transform `transpose` 分开命名 |
| `W.lookup(table, indices, *, where=True)` | validity-aware typed `weft_kernel.lookup` |
| `W.decode(codes, table, *, where=True, out_dtype=...)` | typed `weft_kernel.decode`，输出 dtype 显式保存 |
| `W.widen(value, dtype)` | `weft_kernel.widen` |
| `W.narrow(value, dtype, *, rounding="rne", saturation=False)` | `weft_kernel.narrow` 保存 rounding/saturation |
| `W.block_scaled_contract(...)` | 已注册的 sibling `weft_ext.block_scaled_contract`；不是普通 contract provider alias |

Python compiler 与 C++ dialect 不各自维护 op schema：Python 只保存一次 lowering 所需的
type spelling/shape facts 并打印 generic op；TableGen schema 和 C++ verifier 决定最终合法性。
新的 semantic extension 必须先增加并注册 sibling canonical dialect/op，不能只注册一个
Python 回调或 whole-kernel 名称。

## 6. Canonical dialect 必须同期完成的内容

### 6.1 Types

- scalar/index；
- pointer + qualifiers；
- constexpr/meta；
- logical region/block；
- predicate；
- validity-carrying masked value；
- tuple/state types；
- 必要的 extension types。

### 6.2 Op/region families

- kernel ABI、return/yield、constant/meta value；
- scalar control、range、condition/while；
- VLA region 与 logical coordinate；
- pointer/index、pointwise/compare/cast/select；
- memory、prefetch、atomic、fence；
- block constructors/transforms；
- valid/fill/masked propagation；
- reduce、scan、summary fold；
- contract 与 local extension primitives。

### 6.3 Verifier

至少验证 ABI/type、pointer arithmetic、VLA nesting/escape/state mutation、masked consumer、
block shape/broadcast、dynamic extent identity、contract axes/output、state type/purity、
constexpr runtime misuse、effect/order 和 extension registration。

Python frontend 可以提前给出更好的 source diagnostic，但 C++ canonical verifier 是最终
语义判定者。

## 7. Python 内部表示纪律

允许的瞬态对象：

- Python AST、source locations、static bindings；
- symbol environment；
- 指向 canonical MLIR type spelling/handle 的 frontend value；
- generic `Operation/Region` assembly node；
- lowering 期间的常量与 shape facts。

禁止：

- 与 TableGen 并行维护的 Python op class hierarchy；
- 独立持久化的 Python block/type/loop IR；
- Python verifier 成为 MLIR verifier 的替代品；
- 在 frontend 中缓存 canonical 之外的算法计划；
- 为尚未实现的 API 生成 placeholder op 或假成功文本。

第一实现采用 donor 中的 generic one-way MLIR assembly builder，而不是先引入一套新的
Python IR framework。Builder 只负责拼装 generic MLIR；registered Kernel dialect 才拥有
schema 和 legality。

## 8. 新根目录

第一里程碑只创建以下活动实现：

```text
python/weft/
  api/
  language/
  frontend/
  diagnostics.py
  __main__.py                  # source -> canonical MLIR AOT entry

include/Weft/Dialect/Kernel/IR/
lib/Dialect/Kernel/IR/
include/Weft/Dialect/Extension/IR/
lib/Dialect/Extension/IR/
tools/weft-opt/                 # parse/print/verify canonical IR
source/weft/weft/               # 规范 source 与算子语料
examples/repro/                 # 只保存手工 repro 入口
```

不创建 `Execution`、RVV、IME、artifact 或 Intent 目录的空壳。

## 9. 已执行的实现依赖顺序

1. 建立 Python signature ↔ canonical schema ledger，闭合第 5 节列出的工程选择；
2. CMake/TableGen/dialect registration、source locations、core scalar/index/pointer/constexpr types；
3. `@weft.kernel`、ABI annotations、constants、expressions、scalar control；
4. VLA region、logical predicate、masked memory 和 effects；
5. logical block、broadcast/shape transforms、dynamic extent identity；
6. reduce、scan、summary fold 与 sequential carry；
7. contract、pointwise/math/numerical attributes；
8. local semantic extension hook；
9. 六个规范示例全部 lower→parse→verify；
10. 对规范 verifier 义务做少量直接诊断 repro，不建立测试矩阵。

该顺序只记录依赖关系。各项均由 Python API、canonical schema 和 verifier 同期落地，
没有保留 placeholder API 或旧 fallback。

## 10. Donor 复用

可以抽取：

- `materials/legacy-source/python/weft/frontend/source.py`；
- generic `frontend/ir.py`；
- `diagnostics.py`；
- definitions 的 source/signature capture 与 AOT-only call guard；
- annotations/dtypes 的机械 descriptor；
- TableGen/CMake/registry 脚手架；
- KernelDialect 中局部 broadcast、shape 和 extent-provenance helper。

必须重写：

- `frontend/compiler.py`；
- public builtins/export surface；
- KernelOp 与全部 canonical schema/verifier；
- pointer qualifiers、masked types、VLA 和 state algebra。

禁止抽取：`grid_rank`、`task_id`、旧 `arange` 根模型、whole-kernel selector/emitter、旧
typed Python IR 作为 authority，或任何从活动 Python package import materials 的路径。

## 11. 验收

已经使用 `WEFT_FINAL_SPEC.md` 第 23 节的六个完整程序作为固定 acceptance：

1. elementwise add-bias；
2. RMSNorm worker；
3. predicate + reduce identity；
4. online softmax summary；
5. worker-local blocked GEMM；
6. semantic extension primitive。

六份 source 已生成 canonical MLIR，并被独立 `weft-opt` parse/print/verify。第一里程碑
仍不以 RISC-V source/object 或性能数字作为完成条件；这些产物目前也尚未实现。

## 12. 第二里程碑

语言闭环完成后，第二里程碑才建立最小 executable vertical slice：typed target profile、
primitive provider interface、Scalar baseline、一个 RVV VLA memory/pointwise provider、
Selected Execution IR，以及 source/object/C header。

这样 provider 从第一天就消费完整且稳定的 canonical semantics，而不是反向决定 DSL。
