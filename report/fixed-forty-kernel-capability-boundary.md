# 固定四十个 Kernel 的能力边界收敛

## 本轮范围

本轮没有增加kernel，也没有为了迁就backend改写四十项DSL。固定压力语料仍是四十个
distinct kernel；两个GEMM分别有decode与prefill，所以实际执行四十二条phase。

本轮只处理一件事：冻结作者与编译器之间的授权边界，并把已经暴露的物理选择从region
matcher和emitter继续收敛为共享、显式、短生命周期的physical decision。

固定GGML baseline与此前的累计Markdown报告不属于本轮产物，也没有被本轮更新。
`weft-kernel-performance.csv`作为当前Weft数字表已写入本轮四十二条phase的真机重测结果。
`source/`与`materials/`仍只作baseline和知识供体，没有进入production link、runtime或
fallback。

## 最终授权边界

| DSL / Kernel IR构造 | 作者拥有 | Target得到的授权 | Target不得做 |
|---|---|---|---|
| scalar `for` / `while` | 有序traversal、carry、effect、outer control | 保语义unroll、schedule、pipeline | 自动变成VLA、scan或contract |
| `W.vla` | SIMD logical axis、logical bounds、predicate、memory relation | `vl`、SEW/LMUL、strip、mask/index、memory form | 创建第二VLA轴或替换外围blocking |
| `W.contract` | operands、axes、init、dtype、predicate、numerical policy | microtile、LMUL、unroll、fragment、短生命周期packing | 从普通乘加猜contract或创建outer K/staging |
| reduce | 只观察最终聚合值 | reduction tree与跨strip aggregate | 生成prefix或补猜顺序状态 |
| scan | 每个位置的有序prefix | scan instruction sequence与strip carry | 降成普通reduce |
| summary fold | `lift/merge/finalize`代数与coordinate | summary state在RVV上的物理实现 | 从use graph补猜summary代数 |
| sequential carry | source写下的严格状态顺序 | 保序scalar/RVV realization | 自动改成reduce、scan或summary |
| blocking / staging / persistent layout | 算法variant与data organization | 消费显式关系做local reuse/fusion | 发明、替换或跨越这些关系 |

这条边界已经写入稳定DSL与target-lowering文档。Weft仍是作者编写完整worker-local kernel
program、target实现显式授权局部结构的算子编译器；不是从SSA图猜算法的图编译器。

## 本轮实际重构

### VLA 物理决定成为唯一来源

`VLARegionDecision`现在显式持有：

- data与index的SEW、LMUL和mask ratio；
- 每个load/store的element shape、memory mode、activity与store-value mode；
- reduce、scan、summary的state realization与跨strip shape；
- narrow的source/intermediate/result LMUL；
- contract、cast与local FMA各自的typed physical decision。

没有predicate的VLA不再无条件创建index-vector配置，因此窄数据region可以合法使用更高LMUL。
VLA strip、mask/index relation、access emitter和contract emitter读取同一份decision，不再各自
选择物理宽度。

### 四条 F16 region matcher 被通用 lowering 替代

本轮删除：

- F16 fill whole-region realization；
- F32→F16 load/cast/store realization；
- F16 weighted-update realization；
- F16→F32 normalize realization。

替代它们的不是新的kernel分类，而是逐operation能力：

- generic F16 load/store与scalar broadcast；
- `VLACastDecision`描述F16↔F32两向转换的source/result SEW与LMUL；
- generic F16 binary arithmetic；
- `VLABinaryDecision`识别局部multiply-add use relation，并选择F16 RVV FMA。

纯F16 VLA当前可选择e16/LMUL8；混合conversion可生成f16m4↔f32m8。FlashAttention中的fill、
query conversion、weighted update与normalize因此都回到同一条generic VLA主链。

当前specialized VLA只剩两个局部融合：

- widening F16 dot，把显式双load/cast/multiply/reduce闭包实现为`vfwmacc`加延迟reduction；
- online-softmax envelope，联合显式summary producer和相邻normalize consumer。

它们都不读取kernel名、不生成ABI、不接管outer traversal，但仍依赖较精确的local use
closure。

### F32 contract 获得共享资源候选

Local-row contract与VLA-free-axis contract现在共用F32 physical selector。Decision显式记录：

- resource model类别；
- row microtile；
- LMUL；
- K-unroll；
- vector register groups；
- operand/output memory relation。

当前local-row候选在LMUL4/2/1中按row extent和live resource选择：row6选择LMUL2，row8选择
LMUL1；VLA free-axis row6选择LMUL4。Emitter只消费selected config。

本轮探索性真机测量曾尝试K-unroll=2，F32 decode从约16 ms退化到约77 ms，因此完整回退，
没有保留死分支或备用路径。这组被回退的数字只作为当轮取舍记录；当前源码与CSV只保留
`kUnroll=1`的winner。

### Quant、decode 与 IME decision 收敛

- grouped affine Q4_K从“发射时解析operand closure”拆为`GroupedAffineI4I8Decision`，typed
  block bases、scale/minimum、activation sum与init只有一个来源；
- Q8 saturating narrow扩大为f32m8→i16m4→i8m2，保持DSL显式写下的RNE与saturation语义，
  没有改成GGML的non-saturating sequence；
- Q4_0与Q4_K IME的N16/K32及288/304-byte local layout从source的显式axis/layout relation
  进入decision，不再依赖emitter default；
- specialized VLA vector shape、tail policy和memory form均已物化，intrinsic spelling不再反推
  这些选择。

### Block emitter 不再决定微向量形态

此前block store/reduce会在emitter内检查extent、axis users和closure，再现场选择e8mf4或
e8m1。现在由：

- `BlockStorePhysicalDecision`；
- `BlockReducePhysicalDecision`

唯一产生strip family、byte-vector shape、strip VL与lane-index需求。Block operation emitter
只消费这些字段。IQ4 decode中的e8mf4 micro-strip和Q8/Q1/MXFP4 reduction路径不再拥有第二套
物理选择。

这些decision都是一次target lowering内的C++ transient，不进入Kernel IR，也没有新增
Physical IR、provider registry、validator、tuning stage或长期authority。

## 真实执行结果

主要lowering改动完成后，四十个kernel的四十二条phase都由原有单-kernel runtime执行并重新
记录性能；其中各runtime沿用CSV所列`full`或`sampled` correctness scope，不能把sampled项
解释成全输出验证。最后的block-decision责任拆分完成后，又定向执行IQ4、Q8、Q1与MXFP4，
这些定向runtime报告的误差和时间仍在全量记录的相同区间。

### 明显变化

| Kernel / phase | 本轮前 Weft ms | 本轮 Weft ms | 变化 | 相对固定GGML baseline |
|---|---:|---:|---:|---:|
| F16 GEMM decode | 7.221713 | 6.682329 | 快7.5% | 快9.5% |
| F16 GEMM prefill | 330.751278 | 341.452654 | 慢3.2% | 慢14.1% |
| F32 GEMM decode | 17.742197 | 14.532404 | 快18.1% | 快29.5% |
| F32 GEMM prefill | 854.746405 | 761.052490 | 快11.0% | 慢32.5% |
| ConvTranspose2D | 16.333010 | 13.593719 | 快16.8% | 快20.9% |
| Dense Conv2D | 779.800666 | 777.616083 | 基本不变 | 慢23.6% |

F16 generic化没有靠退回scalar维持正确性：FlashAttention为23.543003 ms，SwiGLU为
5.845516 ms，均保持此前性能区间。Q8扩大narrow vector后为4.140899 ms，也没有获得可确认的
性能改善。

量化专用primitive的当前结果：

| Kernel | 本轮 ms | Throughput | 相对固定GGML手写对照 |
|---|---:|---:|---:|
| Q4_K×Q8_K RVV | 12.890796 | 9.110 GOP/s | 快0.9% |
| Q4_K IME | 3.254460 | 10.310 GOP/s | 快5.3% |
| Q4_0 IME | 3.192927 | 10.509 GOP/s | 慢8.6% |
| Q1_0×Q8_0 | 56.161806 | 2.091 GOP/s | 慢8.3% |
| MXFP4×Q8_0 | 19.522045 | 6.016 GOP/s | 慢11.4% |

这些数字同时写入当前`weft-kernel-performance.csv`；CSV只记录数字，不附带阈值、同步或
检查逻辑。

## 四十项在本轮逼出的结论

1. scalar、VLA与contract不能共享一个“compiler发现并向量化”的模糊入口。只有显式VLA和
   contract分别授权SIMD logical axis与local contraction decomposition。
2. 同一项物理事实必须只有一个producer。特别是VLA LMUL、cast shape、contract LMUL、IME
   layout与block strip，已经不能再由emitter根据周围结构重建。
3. Whole-region matcher确实可以被逐实体能力替代。四条F16路径删除后，FlashAttention仍由
   普通access/cast/arithmetic decision组合生成，而不是换成另一条FlashAttention emitter。
4. Candidate space必须由资源关系约束，而不是“LMUL越大越好”。F32 row6从LMUL4改为LMUL2
   改善decode与ConvTranspose；K-unroll=2则真实退化并被移除。
5. 扩展leaf仍只能实现local semantic primitive。Q4/Q1/MXFP4/IME的outer traversal、ABI、
   persistent layout与activation preparation继续由source拥有。

## 当前没有完成的部分

### Physical candidate space仍窄

- F32 prefill仍比固定GGML baseline慢32.5%，Dense Conv慢23.6%；
- contract尚无成熟multi-axis microtile、K-unroll、pointer schedule、prefetch、local reuse与
  software-pipeline候选；
- Q8慢25.8%，Q1与MXFP4仍落后手写RVV 8.3%和11.4%；
- Top-K仍为固定GGML baseline的3.115倍；
- state placement、indexed-memory prefetch与vision sliding-window reuse仍未形成physical
  decision space。

### 合法输入范围仍窄

- widening F16 dot与online-softmax仍是精确local fusion envelope；
- F16 GEMM nested N/K关系与affine Q4_K IME N16/K32仍要求较精确的局部loop closure；
- generic VLA尚不支持lane-varying masked load；
- nested VLA analysis目前只递归已支持的scalar `for`，不覆盖任意`if/while`中的所有
  memory/state组合；
- local F32 contract与VLA free-axis contract仍只覆盖当前少数rank/axis关系；
- 第二个active VLA axis继续明确unsupported。

### Decision 代码组织尚未完全统一

常规VLA、contract和block路径已经明确做到analysis/selection产生decision、emitter消费。
部分quant/IME decision虽然也只有一次typed选择，但producer仍贴近具体emit入口，尚未形成
统一的primitive decision组织。这是当前代码结构成熟度边界，不是legacy route或fallback。

## 本轮判断

这一轮没有让Weft多“支持”kernel，而是把四十个既有kernel当成固定语料，证明并收紧了三条
编译器原则：显式DSL构造才授予物理重组权；逐实体physical decision是唯一事实；emitter不再
重新识别算法或选择物理策略。

当前Weft的能力边界已经比上一轮清楚：generic VLA和F32 contract拥有更宽的共享实现空间，
四条F16 region matcher已经消失，block emitter的物理选择也已移出。但candidate richness、
合法source envelope和部分extension decision组织仍不成熟，因此本轮结论是“能力边界已经
冻结，编译器内部继续收敛”，不是“RISC-V Triton已经完成”。
