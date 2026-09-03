# Row-dequant 同字节账与 storage-supply 收口

本报告记录 `21ddcf673` 之后这一轮的定向结果。设计规范仍以 `doc/` 为准；这里仅保存测量、
静态工作账、负结果与尚未闭合的事实。

## 1. 测量口径

此前 source row-dequant 对可量化格式先从 float quantize，对 IQ 格式则使用零填充，而 Weft
读取固定 seed 的随机 encoded record；两边不是同一输入。本轮统一为：每个格式用固定 seed
生成一条能够有限解码的随机 record，并把同一 record 复制成 `N=1024, K=4096` tensor。
source 与 Weft 读取完全相同的 bytes。两机都使用 Clang 18、`-O3 -ffp-contract=fast`、同一
ISA/ABI、64 MiB cache eviction 和 10 次 median。source 的 48 行已整体更新到 baseline CSV；
定向 Weft 结果只写本报告，不局部覆盖正式全量 Weft CSV。

该口径变化解释了旧报告与本报告中 source 数字的差异；它没有改变任何 dequant 算法或 timed
region。

## 2. 第九节七步得到的工作账

row-dequant 没有 contraction，因而 partial topology、product carrier 和 combine tree 不适用。
本轮先按 logical carrier 与 memory form 统计，再看供应 identity 与 donor：

1. 24 个格式均在两台机器生成 RISC-V Physical IR；按 typed storage relation 可分为 bitmask、
   layered stream、projected stream、storage window、indexed-entry、regular-repeat、普通 contiguous
   load、lookup 和 scalar-expanded radix 九类。
2. 当前明显低档不是一个共同瓶颈。IQ3_S 是 entry/sign memory relation；TQ1_0 是作者程序没有
   表达三段 radix shaped output；Q4_K 则是 joined scale/min 已有 regular-repeat relation，但原
   physical path把两个 source scalar扩成向量后再参与 pointwise。
3. Q4_K donor 每个 64-element group只构造两组 `d*sc` 与 `dmin*m`，再分别服务两个 32-element
   payload；见 `source/c/ggml/llama.cpp/ggml/src/ggml-quants.c:1471-1491`。改动前，同一棵 64-group
   作者树在每个 physical part产生向量 broadcast/move；改成 scalar supply 后，生成 C 直接使用
   `vfmul.vf`/`vfsub.vf`。
4. IQ3_XXS donor每个 32-element group有 4 个 sign group，每组两个独立 4-byte grid entry；见
   `ggml-quants.c:2503-2531`。旧作者树把两个 entry重新拼成一个 8-element lookup。新树显式保留
   `code×payload = 4×4`，每个 code只取自己的 entry，不再先拼后拆。
5. TQ1_0 donor是三个不同的 radix domain：`5×32`、`5×16`、`4×4`，见
   `ggml-quants.c:2356-2392`。当前 std 仍是 256 次 Python scalar展开。已有 DSL 能构造三组
   shaped value，但没有带输出 offset 的 sub-Level projection，也没有把 `[5,32]/[5,16]/[4,4]`
   映射到三个扁平输出区间的 canonical reshape/offset projection；因此不能由 physical pass从
   展开 SSA 中猜回轴。

## 3. 24 格式双机结果

单元格为 `Weft / source（比值）`，吞吐单位为 MElements/s。除特别注明外，Weft 为本轮完整
row ledger 的 10 次 median；Q5_K 是恢复旧树后的 3 次定向回归。

| format | SG2044 | K1/X60 |
|---|---:|---:|
| Q1_0 | 743.441 / 747.541（99.5%） | 585.476 / 150.276（389.6%） |
| Q4_0 | 737.587 / 426.623（172.9%） | 432.128 / 171.620（251.8%） |
| Q4_1 | 737.081 / 372.007（198.1%） | 276.919 / 150.617（183.9%） |
| Q5_0 | 733.562 / 168.147（436.3%） | 274.802 / 160.433（171.3%） |
| Q5_1 | 733.940 / 157.928（464.7%） | 293.096 / 147.935（198.1%） |
| Q8_0 | 727.036 / 333.989（217.7%） | 502.947 / 183.911（273.5%） |
| Q2_K | 664.459 / 630.211（105.4%） | 300.407 / 137.808（218.0%） |
| Q3_K | 513.992 / 378.330（135.9%） | 281.247 / 114.229（246.2%） |
| Q4_K | 712.790 / 684.531（104.1%） | 362.154 / 167.580（216.1%） |
| Q5_K | 396.228 / 410.225（96.6%，3 reps） | 244.147 / 112.132（217.7%，3 reps） |
| Q6_K | 423.501 / 188.425（224.8%） | 246.618 / 101.166（243.8%） |
| IQ1_S | 727.544 / 728.975（99.8%） | 452.182 / 129.929（348.0%） |
| IQ1_M | 708.122 / 712.919（99.3%） | 316.946 / 121.641（260.6%） |
| IQ2_S | 643.082 / 428.963（149.9%） | 225.294 / 147.167（153.1%） |
| IQ2_XS | 710.996 / 473.533（150.1%） | 252.540 / 136.814（184.6%） |
| IQ2_XXS | 432.616 / 444.218（97.4%） | 207.601 / 148.885（139.4%） |
| IQ3_S | 259.730 / 350.612（74.1%） | 156.685 / 152.032（103.1%） |
| IQ3_XXS | 484.880 / 445.011（109.0%） | 294.181 / 148.733（197.8%） |
| IQ4_NL | 738.593 / 183.304（402.9%） | 441.617 / 154.773（285.3%） |
| IQ4_XS | 721.952 / 180.925（399.0%） | 344.454 / 158.302（217.6%） |
| TQ1_0 | 312.469 / 668.762（46.7%） | 67.561 / 110.511（61.1%） |
| TQ2_0 | 734.597 / 684.362（107.3%） | 603.277 / 173.383（347.9%） |
| MXFP4 | 738.107 / 160.297（460.5%） | 261.343 / 167.845（155.7%） |
| NVFP4 | 735.371 / 151.589（485.1%） | 253.491 / 99.495（254.8%） |

双机 48 行中，40 行达到 source，5 行位于 90%–100%，1 行位于 70%–90%，2 行低于
70%。按格式看，只有 IQ3_S 与 TQ1_0 仍有明显低于 source 的 target。

## 4. 本轮保留的改动与外部结果

### 4.1 Joined regular-repeat scalar supply

`PlanRISCVMemory` 现在从 typed regular index relation得到 dynamic source base、source count、
repeat 和最终 lane/time/replica mapping。只有 pure pointwise 链到达一个 RVV binary/compare
boundary时，才物化 `rvv_regular_repeat_scalar_load`。该 op保存每个 physical part 的
`part_bases`；verifier从layout与repeat关系重算并逐项核对。emitter只读取选定 source，随后按
`part_bases`绑定 scalar tuple，不再选择 broadcast/gather topology。

Q4_K 的受控对照：同一 64-group 作者树在 SG2044 的旧 vector supply约为 262 MElements/s；
scalar supply在相同较宽 binding下约为 515–520，最终 `LMUL=m1, unroll=1` 为 712.790。
K1 从约 333 提至 362.154。SG 参数扫描还显示 `unroll=2` 会降到约 499，因而没有把该
schedule写死进机制。

该机制目前只有 Q4_K row 一个生产正例。对 Q5_K 临时改成相同 64-group relation后，scalar
supply确实形成，但 SG `206.676→207.891`、K1 `297.507→295.898`，最终汇编工作没有实质减少；
而该作者树使 SG 相对原 32-sub tree大退，因此整棵实验树已撤销。96 个 row/vec-dot target-entry
静态扫描中，也只有 Q4_K row产生此 op。结论只能写成“Q4_K 的 closed relation有效”，不能宣称
跨格式泛化。

### 4.2 IQ3_XXS shaped entry/payload

std 把每个 32-element group改为 4 个 code point，每个 code有 4-element payload；sign index仍按
两个 code共享的 7-bit group计算。它与 donor的 `grid1/grid2` 两个独立 entry一致，没有修改 ABI
或 Encoding。SG 从约 346 提至 484.880，K1 从约 198 提至 294.181，两机都超过 source。

## 5. 负结果与未保留代码

| 实验 | 静态/真机结果 | 结论 |
|---|---|---|
| Q5_K 64-group tree + scalar repeat | SG约 207，低于恢复后的 396；scalar on/off几乎相同 | 同一 joined metadata relation不等于同一最终工作；没有保留实验树 |
| Q4_K `unroll=2` | SG从约 705降至约 499 | row decode没有由该复制形成有效 schedule；保留 unroll=1 |
| Q4_K 较宽 LMUL | m1/m2约 705，m4/m8约 518–521 | 更宽 carrier增加time/part组织而不删工作；选 m1 |
| IQ3_S 4-element sign window | verifier拒绝 half-byte起点 | 不能把未闭合的半字节 relation伪装成 byte-aligned window |

## 6. 结构验收与参考实现对照

当前 24 个 row-dequant入口在 SG2044/VLEN128 与 K1/VLEN256 上共 48 份 RISC-V Physical IR，
全部通过独立 parse、final verifier、标准 canonicalizer/CSE、两次 Share pass与 round-trip verifier；
同一整套流水再次运行后 48/48 文本 diff为 0。

Triton `Coalesce.cpp:77-119` 由 pointer contiguity/order与consumer shape决定memory-op layout并
改写真实 IR；TileLang `loop_vectorize.cc:289-305` 把 loop-invariant non-local load明确分类为
“一次 scalar load，向所有 lanes broadcast”。Weft不能照搬其 thread/warp owner或TVM vectorized
loop，但本轮采用同一责任边界：storage relation与consumer layout在physical pass闭合，selected
scalar load/broadcast relation成为真实 op，terminal C只拼写。

## 7. 尚未闭合的事实

- IQ3_S 的数学工作量已经和 donor一致，剩余 SG 差距落在 P3：两个 4-byte entry当前仍作为一个
  8-lane indexed lookup组织，qh/sign 的 half-byte relation没有形成可供两个 entry共享的闭合
  physical window。这里需要 typed two-entry payload relation及合法的半字节 sign projection；
  当前没有第二个独立生产输入证明合同边界，因此本轮未新增 op。
- TQ1_0 缺的是 canonical shaped output projection，不是 RISC-V leaf。三组 shaped radix值可以用
  现有 `iota` 表达，但现有 `L.subs`没有输出 offset，frontend也禁止对 View做 shaped gather；
  在新增明确的 offset projection/reshape 语言语义以前，只能保留 scalar tree。
- Q5_K/SG 为 source 的 96.6%，IQ2_XXS/SG 为 97.4%，没有静态证据支持为这几个百分点新造 pass。
- 本轮没有验证 software pipeline；row-dequant 数字不能作为 P5 证据。
