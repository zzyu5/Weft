# RISC-V Physical IR

## 1. Physical Value

shaped physical value 使用带 target layout 的类型：

```text
tensor<logical-shape x element,
       #weft_riscv.layout<
           axes = logical-axis identities,
           map = logical coordinates -> time/lane/replica/fragment,
           carrier = scalar|rvv|ime,
           validity = mask/tail relation>>
```

layout 必须保存所有未被 numerical operation 消去的 logical axes。`reduce [M,K] -> [M]` 后 M 轴仍在 result type；register replica 是 M 的物理映射，不是新的 canonical Value。

每个 physical operation/result 携带 source origin，但真实 use-def、control、memory 与 schedule 必须存在于 RISC-V IR 本身，不能靠 origin 回查 canonical closure 补齐。

## 2. Memory Descriptor

```text
!weft_riscv.memdesc<
    logical shape,
    Encoding/storage mapping,
    extents/strides/origin/alignment,
    address class,
    alias/effect/lifetime>
```

kernel 参数 descriptor 完整保存 pinned Encoding 与 ABI。动态 extent、stride 和 origin 是显式 SSA operand 或 function argument。local descriptor 只表示 invocation-local object，不能越过 pin boundary 或替代 caller-visible workspace。

## 3. Explicit Conversion

representation conflict 必须成为真实 operation：

```mlir
%b = weft_riscv.convert_layout %a
    : tensor<..., #layout_a> -> tensor<..., #layout_b>
```

conversion 保持 logical value、shape、axes 与 Level identity，只改变 physical representation。它有 SSA result、verifier 和 rewrite pattern；多个 consumers 可以共享同一个 result。conversion elimination 必须重写或删除这个 operation，不能修改旁路表。

final RISC-V IR 中，通用 conversion 已降低为确定的 RVV slide/gather/splat、tuple split/merge、register/local transfer 或 RVV/IME handoff。

## 4. Physical Operations

RISC-V IR 使用真实 operations 表示：

```text
load / store
encoded_load / encoded_extract
local_alloc / local_load / local_store / local_pack
spill / reload
convert_layout
RVV operations
IME operations
```

memory op 必须携带 descriptor、logical region、selected form、validity、alignment、effect 与 result layout。local pack 必须有 source、schema、destination object、lifetime 和 consumers；它不能伪装成 source operation 或 caller-visible workspace。

## 5. Level、Control 与 Schedule

canonical Level 转入 RISC-V IR 后保留一一对应的 origin、domain、partition、multiplicity、births、carried values 与 handoff。ordinary `for/while/if` 保持有序标量语义，不因进入 physical IR 获得 shaped axis 或 lane mapping。

schedule 必须物化成真实结构：

- strip 与 tail control；
- prologue、steady state 与 epilogue；
- versioned local buffers；
- load/compute ordering、wait/barrier；
- loop-carried physical values。

只有 attribute 而没有这些结构，不算已经支持 pipeline。

## 6. Final IR 不变量

terminal emission 前必须满足：

- 所有 shaped values 都有完整 layout；
- 每条不兼容 use edge 都有 typed conversion；
- memory descriptor/access 与 pinned Encoding 一致；
- source axes、Level、control、effects 与 handoff identity 保持；
- target op、mask/tail、fragment 和 resources 合法；
- 不存在未展开的 composite、Level、cluster 或 schedule op；
- 每个 target op 已经落到合同闭合的 [local leaf](leaves.md)。
