# 图结构与融合机会分析

> 组 1。等级 **B**（底层能力在通用生态里成熟，需要确认你的栈提供哪一个）。

## 一句话结论

**从计算图里挖重复子图的算法是完全通用的、现在就能实现的；卡住的只有两件事——能不能导出一份计算图 IR，以及 IR 的节点能不能对应到一个有耗时的 kernel。** 前者决定分析能不能跑，后者决定结果能不能排序，而排序才是这件事的全部价值。

## 当前状态

| 现在能做 | 还差什么 |
|---|---|
| 本文「算法」一节是完整的、与硬件和框架都无关的实现说明，照着写就能跑 | 没有任何一份 IR 可以喂进去。`hardware-profiles/` 的 `graph` 一节全是 `null` |
| 拿到 IR 后，可以立刻给出"哪些算子组合重复出现了多少次" | **排不了序**。没有「IR 节点 → kernel 名 → 耗时」这条链，只能按出现次数排，而出现次数多的往往是最便宜的 |
| 收益公式（本文「融合收益怎么估」）可以手算验证单个候选 | 公式里的达成带宽与启动开销需要实测，依赖[性能数据采集](perf-profiling.md) |

---

## 目标

从计算图里找出**重复出现的子图**，按 `重复次数 × 单次耗时` 排序，定位最值得写融合算子的位置。

它解决的是一个具体的决策问题：写一个融合算子的成本是确定的（手写 kernel、注册到框架、验证数值、长期维护），收益却不确定。盲目地"看起来相邻就融一下"会产出大量收益接近零的算子。本分析的产出是一张排好序的候选表，让"先融哪个"有依据。

边界：[性能拆解](../../skills/model-infer-perf-breakdown/SKILL.md)按**单个算子**回答"谁最耗时"，本分析按**结构**回答"哪个组合最值得合并"；真正写出融合算子并替换进去是组 3 的[融合算子替换](fused-op-substitution.md)，本分析只负责告诉它替换哪里。

---

## 算法

全部步骤与硬件、框架无关。输入是一张 DAG，每个节点有 `id`、`op_type`、`inputs`（前驱 id 列表）；输出是按收益排序的模式列表。

### 1. 拓扑排序

```python
def topo_order(nodes):
    indeg = {n.id: len(n.inputs) for n in nodes}
    succ = {n.id: [] for n in nodes}
    for n in nodes:
        for p in n.inputs:
            succ[p].append(n.id)

    # tie-break 必须确定：同一张图每次跑要得到同一个序，否则结果不可复现
    ready = heapify([n.id for n in nodes if indeg[n.id] == 0])   # 按原始声明序
    order = []
    while ready:
        u = heappop(ready)
        order.append(u)
        for v in succ[u]:
            indeg[v] -= 1
            if indeg[v] == 0:
                heappush(ready, v)
    assert len(order) == len(nodes), "图里有环，输入不是 DAG"
    return order
```

**唯一的坑是 tie-break。** 拓扑序不唯一，并行分支的相对次序由出队顺序决定。不固定 tie-break 的话，同一张图两次分析会产出不同的 n-gram，结果无法复现、也无法对比两个版本的模型。用原始声明序（节点在 IR 里的出现顺序）做小顶堆的键是最简单的确定性选择。

### 2. 过滤 trivial 节点

不产生实际计算的节点会把真正相邻的计算算子隔开，必须收缩掉：

```python
TRIVIAL_OPS = {
    "placeholder", "output", "get_attr", "constant",       # 图的边界与常量
    "view", "reshape", "permute", "transpose", "contiguous",
    "expand", "squeeze", "unsqueeze", "detach", "alias",   # 纯视图变换
}

def contract_trivial(graph):
    # 必须按拓扑序收缩：连续多个 trivial 节点串在一起时，乱序收缩会漏缝
    for t in [u for u in topo_order(graph) if graph[u].op_type in TRIVIAL_OPS]:
        for p in graph.pred(t):
            for s in graph.succ(t):
                graph.add_edge(p, s)      # 缝合：前驱直连后继
        graph.remove(t)
```

**缝合这一步不能省。** 只删节点不连边，后面的连通性检查会把本来相连的两个算子判成不相连，所有跨越视图算子的模式都会丢掉。

**`TRIVIAL_OPS` 的名单取决于你的后端。** 视图类算子是否真的零成本，要看后端会不会为它生成一次实际拷贝——有些后端对 `permute` 后接非连续访问的算子会插入一次物化。保守做法是先按上表过滤出候选，最后用 profiler 的真实 kernel 列表反查：如果某个被当作 trivial 的算子在 kernel 列表里出现了，就把它从名单里拿掉。

### 3. 对算子类型序列做 n-gram

```python
N_MIN, N_MAX = 2, 8          # 超过 8 的子图几乎写不成单个融合算子，没必要挖
for n in range(N_MIN, N_MAX + 1):
    for i in range(len(order) - n + 1):
        window = order[i : i + n]
        ...                   # 进入第 4 步
```

**为什么用拓扑序滑窗，而不是做一般的频繁子图挖掘。** 一般子图枚举是指数复杂度；拓扑序滑窗是 `O(节点数 × N_MAX)`，在几万节点的图上是毫秒级。代价是会漏掉"图上相邻但在拓扑序里被排远了"的子图。这个取舍对当前目的是划算的：真正值得融合的子图本来就是一条生产-消费链，在任何合理的拓扑序里都相邻或接近相邻。**这是用完备性换可计算性的明确决定，不是疏忽**，报告里应当写明。

### 4. 窗口连通性检查

拓扑序上相邻不代表图上连通。两个毫无关系的分支完全可能被排到一起。

```python
def window_is_chain(window, graph):
    """严格档：窗口内除第一个节点外，每个节点至少有一个前驱也在窗口内。"""
    inside = set(window)
    for u in window[1:]:
        if not (set(graph.pred(u)) & inside):
            return False
    return True

def window_is_weakly_connected(window, graph):
    """宽松档：忽略边的方向后是一个连通块。"""
    inside = set(window)
    seen, stack = {window[0]}, [window[0]]
    while stack:
        u = stack.pop()
        for v in (set(graph.pred(u)) | set(graph.succ(u))) & inside:
            if v not in seen:
                seen.add(v)
                stack.append(v)
    return len(seen) == len(inside)
```

**推荐用严格档。** 宽松档会放进"同一个输入的两个并行分支"这类窗口——它们确实连通，但彼此之间没有中间张量，融合省不下任何访存，收益估算会虚高。严格档保证子图是一条真实的生产-消费链，这正是融合算子能省掉中间结果的前提。

### 5. 按多重集签名去重

```python
def signature(window, graph):
    types = tuple(sorted(graph[u].op_type for u in window))
    internal_edges = sum(
        1 for u in window for v in graph.succ(u) if v in set(window)
    )
    entries = sum(1 for u in window if not (set(graph.pred(u)) & set(window)))
    exits   = sum(1 for u in window if not (set(graph.succ(u)) & set(window)))
    return (len(window), types, internal_edges, entries, exits)
```

**为什么签名用多重集而不是序列**：同一个模式在图的不同位置可能被排成不同顺序（并行分支的相对次序不确定），用有序序列做签名会把一个模式拆成好几条，重复次数被摊薄，排序结果失真。

**多重集签名的已知缺陷**：它会把"算子组成相同但连接结构不同"的子图合并成一个模式。例如 `a→b→c` 和 `a→b` 加 `a→c`，多重集完全一样。这是误合并，会高估重复次数。

上面的签名里额外带了 `internal_edges` / `entries` / `exits` 三个结构量，用很低的成本挡掉绝大多数误合并。要完全精确需要子图同构判定，成本陡增且对本目的不必要——因为误合并的后果是候选表里混进一两条假模式，而候选表本来就要人工复核。

**shape 要不要进签名**：建议不进。shape 进签名会让重复次数大幅下降（同一个 FFN 在不同 batch 下就不是一个模式了），而融合算子通常本来就要支持多种 shape。正确做法是签名不含 shape，但在模式的**实例列表**里保留每个实例的 shape，排序后再按 shape 分组看单次耗时。

### 6. 跨长度取极大模式

长度为 n 的模式，必然让它的每一个长度 n−1 的连续子串也成为一个模式。不处理的话，候选表会被同一结构的各种长度版本塞满。

```python
# 从长到短扫。若 P 的每个实例都被某个更长模式 Q 的实例覆盖，则 P 被 Q 吸收
for n in range(N_MAX - 1, N_MIN - 1, -1):
    for P in patterns[n]:
        for Q in patterns[n + 1]:
            if all(any(inst_p <= inst_q for inst_q in Q.instances) for inst_p in P.instances):
                P.absorbed_by = Q
                break
patterns = [p for group in patterns.values() for p in group if p.absorbed_by is None]
```

**用实例集合的包含关系判断吸收，而不是签名的包含关系。** 签名级判断（"P 的多重集是 Q 的子集且出现次数相同"）会在存在误合并时误吸收。实例级判断（P 的每个实例的节点集合都被某个 Q 实例的节点集合包含）更准，代价只是要保留每个模式的实例列表——这份列表后面排序和复核都要用，本来就得存。

### 7. 按收益排序

见下一节的公式。输出每个模式时至少带：`签名`、`出现次数`、`单次耗时`、`估计总收益`、`占整体墙钟的比例`、`前三个实例的节点 id 与 shape`。**最后一项不要省**——自动挖出来的模式总有一部分是算法产物而非真实机会，没有实例清单就无法人工复核。

---

## 卡在哪：IR 从哪来

算法不挑 IR，但不同来源的信息富度差别很大，直接决定分析能回答到哪一步。

| IR 来源 | 节点粒度 | 带 shape/dtype | 能否对应到 kernel 名 | 典型获取失败原因 | 适合回答什么 |
|---|---|---|---|---|---|
| `torch.fx` 静态 trace | 模块/函数级（`call_module` / `call_function` / `call_method`） | 默认无，需另跑一次形状传播补上 | 不能直接对应 | 遇到依赖张量值的控制流就 trace 不下去 | 模块级的重复结构，例如"一个 decoder layer 由哪些块组成" |
| Dynamo 捕获的 graph | 分解后的算子级，比 fx 细得多 | 有（捕获时带 fake tensor 的 meta） | 不能直接对应，但可通过编译产物间接关联 | graph break 会把图切成多段，每段单独分析会丢掉跨段模式 | **粒度最接近 kernel，最适合找融合机会** |
| 编译器生成的 kernel 代码 | 已经融合之后的 kernel 级 | 有 | **能**，生成的 kernel 名可直接与 profiler 匹配 | 不走这条编译路径就没有产物 | 编译器"没融上"的部分，也就是手写融合算子的**增量**机会 |
| 自研编译器的 IR dump | 取决于你的编译器分几层 IR | 取决于实现 | 最有可能能——如果 dump 里带了最终 kernel 名 | — | 最贴近真实执行；但算子词表是你自己的，需要一张归一化表 |

**决定分析能做到什么的不是节点数量，而是"节点能不能对应到一个有耗时的 kernel"。** 没有这条映射，算法照样跑完，但输出只有"这个模式出现了 N 次"——而出现次数最多的模式往往恰恰是最便宜的那些（逐元素加法能出现几千次），按次数排序会把人引向错误的方向。

**暂时没有映射时的两条降级路线**：按算子类型做粗映射（复用 [perf-breakdown](../../skills/model-infer-perf-breakdown/SKILL.md) 的分桶思路，用名称正则把 profiler 的 kernel 归到 IR 的 `op_type`，再把桶的平均耗时摊到节点上——粗，但量级对）；或者只按出现次数排序、人工复核 Top 20，第一次做这件事时这一步就能捞出明显的机会。

如果你的编译器**已经在做融合**，分析目标要换一个：不是"找可以融的地方"，而是"找编译器漏掉的地方"。做法是拿编译后的 kernel 列表反过来对照融合前的 IR，看哪些本应合并的链路最终仍是多个 kernel。

---

## 融合收益怎么估

收益只有两个来源：**少启动几次 kernel**，和**少读写几次中间张量**。两者都可以算。

### 符号

```
n          子图的节点数
occ        该模式的出现次数
t_now      单次出现的现状耗时 = Σ_{v ∈ 子图} kernel_duration(v)
L          单次 kernel 启动开销（微秒），需实测
BW_ach     达成带宽（字节/微秒），用实测值，不要用规格峰值
F_ach      达成算力（FLOP/微秒），同上
```

`L` 的实测方法：连续下发一串极短的空 kernel，测平均间隔；或者从 perf-breakdown 的气泡占比反推（气泡时长 / 气泡期间的下发次数）。
`BW_ach` / `F_ach` 来自 T3 的硬件计数器，或[无计数器时的解析法](perf-profiling.md#硬件计数器的关键性)。**用峰值规格会系统性高估收益**，实际达成值常常只有峰值的一半上下。

### 省下的启动次数

```
saved_launches        = occ × (n − 1)              # n 个 kernel 合成 1 个
t_launch_saved_per_occ = (n − 1) × L
```

这一项在 decode 阶段权重很高（kernel 本身只有几微秒，与一次启动同量级），在 prefill 阶段几乎可以忽略。

### 省下的中间张量读写

```
internal_edges = { 边 (u,v) | u 和 v 都在子图内
                             且 u 的输出没有任何子图外的消费者 }

saved_bytes_per_occ = Σ_{e ∈ internal_edges} 2 × bytes(tensor_e)
                                              ↑ 一次写出 HBM + 一次读回
t_mem_saved_per_occ = saved_bytes_per_occ / BW_ach
```

**第二个条件（没有子图外的消费者）绝对不能漏。** 如果中间张量还要被子图外的节点用到——最常见的是残差连接要用到 norm 之前的值——那它必须被物化到显存，融合省不掉它。漏掉这个条件会系统性高估收益，而且恰好在 Transformer 这种残差密集的结构里高估得最厉害。

### 收益的上下夹逼

单独用"省掉的部分"估会乐观，因为融合算子自身也要干活。用下界夹一下：

```
乐观估计   gain_opt   = occ × ( t_launch_saved_per_occ + t_mem_saved_per_occ )

融合后的耗时下界（物理上不可能更快）
          t_floor    = max( flops(子图) / F_ach,
                            (external_in_bytes + external_out_bytes) / BW_ach )
上界约束   gain_cap   = occ × max(0, t_now − t_floor)

最终估计   gain       = min(gain_opt, gain_cap)
相对收益   gain / 整体墙钟
```

`external_in_bytes` / `external_out_bytes` 是子图的入口张量与出口张量字节数——这些无论如何都要读写，融合省不掉。

**排序用 `gain`，决策用 `gain / 整体墙钟`。** 一个模式估出来能省 50ms 听起来很多，但如果整体墙钟是 5s，那就是 1%，不值得为它写一个要长期维护的 kernel。这与 perf-breakdown 的对比纪律是同一条："看绝对值不要只看百分比"的反面——这里是"看百分比不要只看绝对值"。

---

## 什么值得融合，什么不值得

### 值得

| 特征 | 为什么 |
|---|---|
| 访存受限的逐元素/归一化算子链 | 收益直接来自省掉的中间张量读写，`internal_edges` 大、计算量小，`t_floor` 远低于 `t_now` |
| 单次很短但出现次数极多的小算子序列 | 收益主要来自省启动。decode 路径上这类机会最多 |
| 中间张量大、且**没有**子图外消费者 | 省下的字节数直接乘以 2 |
| 归一化 + 残差相加、激活 + 逐元素乘这类结构固定的组合 | 结构稳定意味着融合算子只需支持有限的几种 shape，写和维护都便宜 |
| 编译器已经证明能融、但你的后端还没融上的链路 | 收益已被生态验证过，风险最低 |

### 不值得

| 特征 | 为什么 |
|---|---|
| 子图里有一个计算受限的大矩阵乘 | `t_floor` 已经接近 `t_now`，融合救不了；而且把矩阵乘圈进自定义 kernel 会丧失算子库的调优成果，常常更慢 |
| 中间张量很小 | `saved_bytes` 小，收益被启动开销的噪声淹没 |
| 出现次数少 | 总收益小，维护成本却一样 |
| 中间张量还有子图外的消费者 | 必须物化，访存省不掉，只剩启动收益 |
| 子图内含依赖张量值的动态分支 | 融合算子要把分支编进 kernel，难写也难验；留给图外的主机侧更合理 |
| 跨越流边界或包含通信算子 | 融合会把本来可以重叠的部分串起来，可能**反而变慢**。与[多流优化](../../skills/model-infer-multi-stream/SKILL.md)的收益直接冲突，两者必须一起评估 |
| 子图的入口或出口数量很多 | 出入口多意味着大部分张量都要物化，`internal_edges` 占比低，省不下什么 |

一条跨项的纪律：**任何候选在写 kernel 之前，先用现有算子手工拼一个"假融合"版本测一次**（比如把两个算子换成一个已有的融合算子，或在 Python 层合并计算）。如果假融合的收益就已经远低于估计值，说明收益模型里某个参数错了，应该先修模型而不是先写 kernel。

---

## 需要你拍板的问题

| # | 问题 | 回答后能解锁什么 |
|---|---|---|
| 1 | 你的栈能从哪一层导出计算图——`torch.fx`、Dynamo 捕获的 graph、还是自研编译器的 IR dump？ | 决定本分析能不能跑起来，以及节点粒度。这是唯一的硬阻塞 |
| 2 | IR dump 里的节点能否带上它最终对应的 kernel 名（或任何可与 profiler 对齐的标识）？ | 决定候选表能不能按收益排序。为否则只能按出现次数排，价值大打折扣 |
| 3 | IR 里有没有每个节点的输出 shape 和 dtype？ | 决定 `saved_bytes` 能不能算，也就是收益公式的第二项能不能用 |
| 4 | 你的编译器现在做不做算子融合？做到什么粒度（只融逐元素？还是能融进矩阵乘的 epilogue）？ | 决定分析目标是"找融合机会"还是"找编译器漏掉的机会"，这两者的候选表完全不同 |
| 5 | 融合算子最终怎么接回去——注册成自定义算子、写一个编译器 pass、还是在模型代码里手工替换？ | 决定候选表要输出到什么粒度（节点 id 够不够，还是要给出可被 pass 匹配的模式描述） |
| 6 | 单次 kernel 启动开销实测是多少微秒？ | 决定收益公式第一项的权重，也决定"小算子融合"这条路值不值得走 |
| 7 | 你的算子类型有没有一张规范词表（同一个计算在 IR 里只有一个名字）？ | 决定要不要先写一张归一化表。词表不规范时，同一个模式会因为算子别名被拆成多条 |
| 8 | 视图类算子（permute / reshape / contiguous）在你的后端会不会产生实际拷贝？ | 决定 `TRIVIAL_OPS` 名单怎么定。判错会让收缩步骤删掉真实开销，收益估计偏低 |

**回答顺序建议**：1 → 2。第 1 题为否则整件事不成立；第 1 题成立但第 2 题为否，仍可以先做出一张按出现次数排序的粗表，边用边补。

---

## 知识来源

- 拓扑排序（Kahn 算法）、弱连通分量、n-gram 统计、多重集签名去重：标准算法，无特定来源。极大模式与子模式吸收的判别思路来自频繁模式挖掘中闭合/极大模式的通用概念
- 用拓扑序滑窗替代一般子图枚举、以完备性换可计算性：本文的工程取舍，依据是子图枚举的指数复杂度这一公开事实
- `torch.fx` 的节点种类与静态 trace 对数据依赖控制流的限制、TorchDynamo 的图捕获与 graph break、fake tensor meta 传播：PyTorch 官方 `torch.fx` 与 `torch.compile` 文档
- 算子融合的收益来源（减少 kernel 启动、减少中间结果往返 HBM）与逐元素算子链融合的常规做法：深度学习编译领域的公开共识，可见于 TVM、XLA、TorchInductor 的公开设计文档
- 达成带宽/达成算力与峰值规格的区别、roofline 式下界估计：Williams, Waterman, Patterson, "Roofline" (2009)。融合与多流重叠的冲突（融合消除了可重叠的边界）是并发执行模型的通用推论，与具体硬件无关
