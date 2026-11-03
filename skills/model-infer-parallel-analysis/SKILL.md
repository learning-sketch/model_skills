---
name: model-infer-parallel-analysis
description: 为 LLM 推理做多卡并行策略选型分析，输出 2-3 个候选配置及其定量依据。按模块分别选型（Embedding / Attention / Dense FFN / MoE / LMHead），用参数量、单卡权重显存、KV cache 显存、激活显存、通信代价的完整公式做估算，检查整除与拓扑硬约束，并给出拿到 profiling 之后的校准循环。所有带宽、显存、算力数字一律从硬件 profile 读取。本 skill 只分析不改代码：不负责切分层的实现、不负责权重切分与加载、不负责通信算子调优、不负责单卡跑通与基线采集。
---

# 并行策略选型分析

本 skill 的产出是**一份分析报告**，不是代码改动。
它要回答的问题只有一个：**这个模型在这批卡上，应该怎么切，为什么这么切，以及怎么用实测数据推翻它。**

两条不可退让的纪律：

1. **不给单一答案。** 至少给 2 个、最多 3 个候选配置，每个都附定量依据和明确的权衡代价。只给一个配置等于把选择的责任藏起来了。
2. **不写死任何卡型规格。** 显存、带宽、算力全部从 profile 读。profile 里没有的，标注"待确认"，不要用业界常见值顶替。

## 硬件依赖

| profile 字段 | 用途 | 缺失时的降级行为 |
|---|---|---|
| `devices.models[].memory_gb` | 单卡显存预算，决定最小卡数 | **阻断定量分析**。向用户索取或用 `diagnostics.smi_command` 实测，并标注来源 |
| `devices.models[].memory_bandwidth_gb_s` | decode 阶段的访存上限，估 TPOT 下界 | 跳过 roofline 判断，只给显存与通信维度的结论 |
| `devices.models[].compute.{bf16,fp16,fp8,int8}_*` | prefill 阶段的算力上限，估 TTFT 下界 | 同上 |
| `devices.models[].model` | 报告中的卡型标识 | 填 `"unknown"` |
| `interconnect.intra_node.bandwidth_gb_s` | 节点内集合通信时间估算 | **阻断通信估算**。可用实测 all-reduce 带宽反推，标注为实测值 |
| `interconnect.intra_node.topology` | 判断 ring / 全互联，影响通信步数模型 | 按 `ring` 从保守估计 |
| `interconnect.intra_node.cards_per_node` | TP 组不跨节点的硬约束 | **阻断多节点分析**，向用户确认 |
| `interconnect.inter_node.bandwidth_gb_s` | 跨节点通信时间估算 | 单节点方案照常，多节点方案标注"跨节点代价未知" |
| `collective.primitives` | 候选方案依赖的原语是否存在 | 按最小集（`all_reduce` / `broadcast`）处理，不假定 `all_to_all` 可用 |
| `collective.torch_backend` / `library` | 报告中注明通信后端 | 填 `"unknown"` |
| `collective.fused_comm_compute_ops.available` | 通信能否被计算掩盖，影响通信代价排序的最后一档 | 按 `false` 处理：通信时间全额计入 |
| `collective.env_vars.*` | 校准阶段排查带宽不达标 | 跳过该排查项 |
| `kernels.moe.grouped_gemm` / `expert_parallel_dispatch` | 专家并行是否有可用实现 | 为 null 时把 EP 方案标为"需先建设算子"，不作为首选 |
| `kernels.attention.mla_support` | 压缩潜在表示模型的 KV 切分方式 | 按"潜在向量在各 TP rank 上复制"估算 KV 显存（偏保守） |
| `profiler.task_type_values.communication` | 校准阶段识别通信 kernel | 按 kernel 名前缀匹配通信库名 |
| `profiler.kernel_table_columns.duration_us` / `name` / `device_id` | 校准阶段统计通信占比与 rank 间方差 | 退化为用主机侧打点统计，精度下降 |
| `pytorch.api.max_memory_allocated` | 校准阶段核对显存峰值预估 | 用 `diagnostics.smi_command` 采样 |
| 各节的 `_status` | `assumed` / `unknown` 字段先验证再用 | 在报告的 `assumptions` 中逐条登记，并标注该结论的置信度 |

### profile 未覆盖、必须向用户确认的信息

- **集群规模**：可用节点数（`interconnect.intra_node.cards_per_node` 只说明单节点卡数）
- **集合通信的实效带宽与延迟**：`interconnect.*.bandwidth_gb_s` 是链路标称值，不等于 all-reduce 的实效带宽。需要一张"消息尺寸 → 实测耗时"的表，或至少给出带宽效率系数与小消息延迟下限
- **每节点网卡数与每卡可用的跨节点带宽**
- **`bandwidth_gb_s` 的口径**：单向还是双向、每链路还是每卡聚合
- **驱动/运行时的常驻显存**（`memory_gb` 标注为"可用"，但是否已扣除常驻未说明）
- **各模块可用的并行实现**：推理框架实际支持哪些并行维度

## 适用场景

- 模型已在单卡跑通（或已确认单卡放不下），要决定用几张卡、怎么切
- 要在时延优先和吞吐优先之间做取舍，需要定量依据
- 上下文变长，单卡 KV cache 容量不够，要判断是加 TP 还是做序列维切分
- MoE 模型，要在专家并行、张量并行、两者混合之间选择
- 已有一个并行配置在跑，但性能不达预期，需要重新评估选型是否合理

## 不适用场景

| 情形 | 应该去哪 |
|---|---|
| 单卡还没跑通，没有基线 | `model-infer-migrator`。**没有单卡基线的并行分析缺少最重要的锚点** |
| 要实际改代码做切分 | 并行实施相关 skill。本 skill 不改一行代码 |
| KV cache 的块池与分页接入 | `model-infer-kvcache` |
| 通信算子本身慢，要调参 | 运行时调试 / 性能拆解相关 skill |
| 训练的并行选型（含优化器状态与梯度切分） | 训练系列 skill。本 skill 的显存模型不含优化器状态与梯度 |

## 工作流

### Step 1 · 收集输入

| 输入 | 来源 | 缺失时 |
|---|---|---|
| 模型结构参数 | `config.json` | 阻断。无法做任何定量估算 |
| 单卡基线（时延、显存、吞吐） | `model-infer-migrator` 的 `baseline.json` | 可继续，但激活显存只能估算不能实测，置信度下调 |
| 硬件规格 | profile | 见上表 |
| 目标 SLO | 用户 | **必须问**：首 token 时延上限、每 token 时延上限、目标并发数、目标上下文长度。没有目标就没有选型 |
| 可用卡数与节点数 | 用户 | 必须问 |

需要从 `config.json` 提取的符号（全文统一使用）：

```
L        = 层数                      h       = 隐藏维度
n_q      = query 头数                n_kv    = key/value 头数
d_head   = 单头维度                  d_ffn   = FFN 中间维度
V        = 词表大小                  b       = 权重/激活的字节数（bf16 = 2）
E        = 专家总数                  k       = 每 token 激活的专家数
d_exp    = 单专家的中间维度          S_sh    = 共享专家数
tie      = 是否 embedding 与 lm_head 权重共享
```

### Step 2 · 按模块分别选型

**并行方式不是整模型一个选择，而是每类模块各有最适合的切法。**

| 模块 | 首选切法 | 通信代价 | 硬约束 | 什么时候不这么切 |
|---|---|---|---|---|
| **Embedding** | 词表维切分，各 rank 查自己的分片，out-of-range 置零后 `all_reduce` | 每次前向 1 次 `all_reduce`，字节量 `tokens × h × b` | `V % TP == 0`（否则按 TP 补齐 padding） | `V` 不大时直接复制，省掉一次通信；`V × h` 占总参数比 < 2% 时复制的显存代价可忽略 |
| **Attention** | 按 query 头切（`W_q`、`W_o`），KV 头随之切 | `W_o` 之后 1 次 `all_reduce`，字节量 `tokens × h × b` | `n_q % TP == 0`；理想情况 `n_kv % TP == 0` | `TP > n_kv` 时 KV 头必须复制，每卡 KV cache 不再随 TP 下降，收益打折 |
| **Dense FFN** | 第一层列切、第二层行切（SwiGLU 的 gate/up 均列切，down 行切） | 行切之后 1 次 `all_reduce`，字节量 `tokens × h × b` | `d_ffn % TP == 0` | 几乎总是这么切，这是通信量最小的组合（中间维度不需要通信） |
| **MoE 专家** | 专家维切分（EP）：每卡持有 `E / EP` 个专家 | 每层 2 次 `all_to_all`（dispatch + combine），字节量各 `tokens × k × h × b` | `E % EP == 0`；需要 `all_to_all ∈ collective.primitives`；需要 `kernels.moe.*` 有实现 | 专家数少于卡数、或 `all_to_all` 不可用时改用专家内 TP；专家负载严重不均时考虑混合 |
| **MoE 路由器** | 复制（参数量 `h × E` 极小） | 无 | 无 | 无 |
| **LMHead** | 词表维切分 | 直接 `all_gather` logits 的字节量是 `tokens × V × b`，**往往是全模型最大的单次通信**。应在本 rank 先取局部 top-k 或 argmax，再只 gather 候选 | `V % TP == 0` | 需要完整 logits（打分、对数概率输出）时无法规避，按完整字节量计 |
| **Norm / RoPE / 激活** | 不切（逐元素，参数极小）；启用序列并行时随序列维切 | 序列并行把 1 次 `all_reduce` 换成 `reduce_scatter + all_gather`，**总字节量不变**，但激活显存按 TP 下降 | 需要 `reduce_scatter` 与 `all_gather` 两个原语 | 激活显存不紧张时不必引入 |
| **流水并行（PP）** | 按层切成若干 stage | 每个 stage 边界 1 次 `send_recv`，字节量 `micro_batch_tokens × h × b`，是所有方案里**单位通信量最小的**，适合跨节点 | `L % PP == 0`（不整除时首尾 stage 要为 embedding / lm_head 留预算） | 低时延场景慎用：流水气泡会直接抬高单路时延 |
| **副本并行（DP）** | 整模型复制，请求级分流 | 推理前向**无通信** | 单卡（或单 TP 组）必须放得下全部权重 | 放不下时不可用 |

### Step 3 · 决策树

**第一层：看模型架构**

```
是 MoE 吗？
├─ 是 → 算专家参数占比 r_exp = P_experts / P_total
│        r_exp > 0.7  → EP 为主维度，attention 用较小的 TP
│        r_exp ∈ [0.4, 0.7] → EP 与 TP 混合，两套方案都估一遍
│        r_exp < 0.4  → 按 Dense 处理，专家用 TP 切
└─ 否 → 进入下一层

n_kv 是多少？
├─ n_kv ≥ 目标 TP            → TP 可以按目标设，KV cache 随 TP 线性下降
├─ n_kv < 目标 TP            → KV 头需复制：每卡 KV = 单卡 KV × (TP / n_kv) / TP = 单卡 KV / n_kv，
│                               再加 TP 不再降低 KV 显存，改用 PP 或 DP 扩容
└─ 压缩潜在表示（MLA 类）     → KV 本就很小，TP 对 KV 的收益低；
                                 切分方式受算子约束，查 kernels.attention.mla_support
```

**第二层：看优化目标**

| 目标 | 主导维度 | 理由 | 代价 |
|---|---|---|---|
| 时延优先（TTFT / TPOT 低） | TP 尽量大，上限为 `interconnect.intra_node.cards_per_node` | 单请求的计算被摊到更多卡，decode 的访存压力也被摊薄 | 通信次数 `2 × L` 每 token，TP 越大单次 all-reduce 的延迟项越重；超出单节点后收益通常转负 |
| 吞吐优先（总 tokens/s 高） | DP 副本数尽量多，TP 取"刚好放得下权重 + 目标 KV"的最小值 | 副本之间零通信，扩展效率最高 | 单请求时延不改善 |
| 两者兼顾 | TP 取满足时延 SLO 的最小值，剩余卡数做 DP | —— | 需要实测确认 TP 的边际收益拐点 |
| 跨节点扩容 | 优先 PP 跨节点、TP 留在节点内 | `send_recv` 的字节量远小于 `all_reduce`，且只在 stage 边界发生 | 气泡率 `(PP − 1) / (m + PP − 1)`（`m` 为并发微批数），低并发下气泡严重 |

**第三层：看序列长度**

```
单条最长序列的 KV cache 字节数 kv_one_seq = 2 × L × n_kv × d_head × b × S_max
有效 KV 切分度 tp_kv = min(TP_attn, n_kv)          （超过 n_kv 的 TP 不再降低 KV）
若 kv_one_seq / (tp_kv × PP) > 单卡可用于 KV 的显存：
    → 仅靠 TP 放不下一条序列，必须引入序列维切分，或进一步加 PP / 加卡
若 prefill 的 S 很大且 attention 不是 flash 形态：
    → 注意 O(B × n_q × S²) 的中间矩阵，这项常常先于 KV cache 爆掉
```

### Step 4 · 定量估算

**参数量**

```
每层 attention：
  P_attn = h × d_head × (2 × n_q + 2 × n_kv)
           └ W_q: h×(n_q×d_head)，W_o: (n_q×d_head)×h，W_k/W_v: 各 h×(n_kv×d_head)

每层 Dense FFN（SwiGLU 三矩阵；双矩阵结构把 3 换成 2）：
  P_ffn  = 3 × h × d_ffn

每层 MoE：
  P_moe  = h × E                       （路由器）
         + E × 3 × h × d_exp           （路由专家）
         + S_sh × 3 × h × d_ffn        （共享专家，没有则为 0）

每层归一化（RMSNorm 两处）：
  P_norm = 2 × h

整模型：
  P_total = L × (P_attn + P_ffn_or_moe + P_norm) + V × h × (1 if tie else 2) + h

MoE 的单 token 激活参数量（用于估算计算量，不是显存）：
  P_active = P_attn + h × E + k × 3 × h × d_exp + S_sh × 3 × h × d_ffn
```

**单卡权重显存**

```
W_card_bytes = P_shard × b

P_shard 的算法：对每个模块，按它实际的切分方式除以对应并行度
  attention 部分：  P_attn / TP_attn，但 W_k/W_v 在 TP_attn > n_kv 时不再除
                    有效 KV 切分度 tp_kv = min(TP_attn, n_kv)
                    P_attn_shard = h × d_head × (2 × n_q / TP_attn + 2 × n_kv / tp_kv)
  Dense FFN：       P_ffn / TP
  MoE 专家：        E / EP 个专家，每个再按 TP_expert 切 → (E / EP) × 3 × h × d_exp / TP_expert
  Embedding/LMHead：V × h / TP（词表切）或 V × h（复制）
  流水并行：        以上结果再乘以 (本 stage 的层数 / L)
```

**KV cache 显存（单卡）**

```
kv_bytes_per_card = 2 × L_stage × n_kv_per_card × d_head × b × T_inflight

  L_stage       = L / PP                          本 stage 持有的层数
  n_kv_per_card = max(1, n_kv / TP_attn)          TP > n_kv 时为 1（且发生复制）
  T_inflight    = Σ 当前在飞的所有序列的长度之和（不是最大长度）
  b             = KV 存储 dtype 的字节数

压缩潜在表示（MLA 类）把前半段换成：
kv_bytes_per_card = L_stage × (d_latent + d_rope) × b × T_inflight
  （无 KV 头维度；是否能随 TP 切分由算子决定，查 kernels.attention.mla_support）
```

**激活显存（单卡，峰值）**

```
decode（每步仅 1 token/序列）：
  A_decode ≈ c_d × B × h × b          c_d 为实现相关的小常数（量级 10），须实测标定

prefill（flash 形态 attention）：
  A_prefill ≈ c_p × B × S × h × b / sp   c_p 量级 10–20，须实测标定
                                          sp 为序列维切分度，未启用时为 1

prefill（非 flash，存在显式注意力矩阵）：
  额外一项 B × (n_q / TP_attn) × S² × b   —— 这一项随 S 平方增长，通常是第一个爆点

MoE 额外：dispatch/combine 的收发缓冲 ≈ 2 × B × S × k × h × b / EP
```

`c_d`、`c_p` **必须用单卡基线实测标定**：用 `baseline.json` 里的 `non_weight_memory_gb` 减去该用例的 KV cache 理论值，再除以 `B × S × h × b`。不要用默认常数出报告。

**显存总账（每卡）**

```
Total_card = W_card_bytes + kv_bytes_per_card + A_peak + comm_buffer + runtime_reserved
约束：      Total_card ≤ devices.models[].memory_gb × 1e9 × 0.90

反推最小卡数：
  TP_min = ceil( P_total × b / (mem_bytes × 0.90 − kv_target − A_peak − comm_buffer − runtime_reserved) )
  再向上取整到满足 Step 5 全部整除约束的值
```

**通信时间**

```
环状 all-reduce（N 个参与方，消息 M 字节）：
  传输量/rank = 2 × (N − 1) / N × M
  T ≈ 2 × (N − 1) × latency + (2 × (N − 1) / N × M) / BW_eff

all_gather 或 reduce_scatter：
  传输量/rank = (N − 1) / N × M
  T ≈ (N − 1) × latency + ((N − 1) / N × M) / BW_eff

all_to_all（MoE dispatch 或 combine）：
  传输量/rank = (N − 1) / N × M

  BW_eff   = interconnect.<intra|inter>_node.bandwidth_gb_s × η
             η 为带宽效率，profile 无此字段 → 实测标定，未标定时取 0.8 并标注为假设
  latency  = 单次集合通信的固定开销，profile 无此字段 → 实测小消息耗时标定
  全互联拓扑（topology 为 all-to-all）可用更优算法，上式是 ring 的保守上界
```

**热路径上的通信频次与字节量**

```
张量并行，decode 每 token：
  次数 = 2 × L_stage            （attention 输出后 + FFN 输出后）
  单次 M = B × h × b            （B 为批内序列数，每序列 1 token）

张量并行，prefill 每次请求：
  次数 = 2 × L_stage
  单次 M = B × S × h × b

专家并行，每 MoE 层：
  次数 = 2（dispatch + combine）
  单次 M = tokens × k × h × b

流水并行，每微批每 stage 边界：
  次数 = 1，单次 M = micro_batch_tokens × h × b

通信时间占比的估算：
  comm_ratio_est = Σ T_comm / (Σ T_comm + T_compute_single_card)
  T_compute_single_card 取自单卡基线，按并行度折算
```

### Step 5 · 硬约束检查清单

**任何候选配置在写进报告之前，必须逐项通过：**

```
[ ] 卡数自洽：TP × PP × DP == world_size
               MoE 侧：EP × TP_expert × DP_moe == world_size
[ ] n_q  % TP_attn == 0
[ ] n_kv % TP_attn == 0        不整除 → KV 头复制，重算 kv_bytes_per_card 后再判显存
[ ] d_ffn % TP == 0
[ ] E % EP == 0
[ ] V % TP == 0                不整除 → 词表 padding，重算 Embedding/LMHead 显存
[ ] L % PP == 0                不整除 → 给出不均衡切分方案，首尾 stage 为 embedding/lm_head 留预算
[ ] TP 组不跨节点：TP ≤ interconnect.intra_node.cards_per_node
                   若必须跨节点，用 inter_node.bandwidth_gb_s 重估并在报告中突出标注
[ ] 所需原语都在 collective.primitives 中（EP 方案必须有 all_to_all，序列并行必须有 reduce_scatter）
[ ] EP 方案的算子存在：kernels.moe.grouped_gemm 与 expert_parallel_dispatch 非 null
[ ] 显存总账：Total_card ≤ memory_gb × 1e9 × 0.90
```

任一项不过，该配置不进候选表；在报告里单列"被排除的方案及原因"。

### Step 6 · 通信代价排序原则

当两个方案在显存上都可行时，按以下优先级比较通信代价。**顺序本身就是结论的一部分**——排在前面的因素，量级差异通常大到让后面的因素无关紧要：

| 优先级 | 因素 | 判据 | 典型量级差 |
|---|---|---|---|
| 1 | **是否跨节点** | `inter_node.bandwidth_gb_s` 与 `intra_node.bandwidth_gb_s` 的比值 | 常见为 5–10 倍。把高频通信放到节点间，其他优化都白做 |
| 2 | **在热路径上的频次** | 每 token 发生 vs 每请求发生 vs 每批发生 | decode 的每 token 通信会被生成长度放大数百倍 |
| 3 | **原语类型** | `all_to_all` 的扩展性最差（点对点全连接），`all_reduce` 次之，`reduce_scatter`/`all_gather` 约为 `all_reduce` 的一半，`send_recv` 最轻 | 同等字节量下可差 2–4 倍 |
| 4 | **字节量** | 按 Step 4 的公式算 | 线性影响 |
| 5 | **能否被计算掩盖** | `collective.fused_comm_compute_ops.available` 为 true，或可用多流重叠 | 最多把通信时间掩盖掉，不会变成负数 |

**第 5 项永远放在最后考虑。** 用"以后可以重叠掉"来为一个高频跨节点通信的方案辩护，是选型阶段最常见的错误。

### Step 7 · 构造候选配置

至少 2 个、至多 3 个，推荐覆盖**时延优先**、**吞吐优先**、**显存余量最大**三类。构造步骤：

```
1. 先算 TP_min（Step 4 的反推公式），得到"能放下"的下界
2. 时延优先候选：TP = min(intra_node.cards_per_node, 满足 n_q/n_kv 整除的最大值)
                 PP = 1，DP = world_size / TP
3. 吞吐优先候选：TP = 满足整除约束的最小可行值（≥ TP_min）
                 DP = world_size / TP，PP = 1
4. 显存余量候选：在 2、3 之间取一档 TP；若需跨节点，PP 跨节点、TP 留节点内
5. MoE 模型：以上每个候选再加一行 EP 的取法（EP 通常取 world_size / TP_expert）
6. 每个候选跑一遍 Step 5 的硬约束清单，不过的就地淘汰并记录原因
7. 每个候选按 Step 4 填满显存总账与通信估算，按 Step 6 的优先级横向比较
```

候选之间必须**有实质差异**：若两个候选的显存占用和通信量都在 10% 以内，合并成一个，再找一个真正不同的维度（例如换 PP 跨节点）作为第二候选。

### Step 8 · 拿到 profiling 之后的校准循环

选型是估算，必须用实测推翻或确认。采集一轮 profiling（用 `profiler.*` 定义的产物与列名），按下表校准：

| 指标 | 算法 | 阈值 | 超标后的动作 |
|---|---|---|---|
| 通信时间占比 | `Σ(通信 kernel 时长) / 总 step 时长`。通信 kernel 用 `profiler.task_type_values.communication` 识别，字段名用 `profiler.kernel_table_columns.duration_us` | < 15% 健康；15–30% 关注；**> 30% 重新选型** | 降 TP、把高频通信移回节点内、改用 PP 跨节点、检查是否可重叠 |
| 集合通信实效带宽 | `传输量 / rank ÷ 实测耗时`，与 `interconnect.*.bandwidth_gb_s` 比 | < 标称的 50% | 先查环境变量（`collective.env_vars.*`：网卡选择、缓冲区大小、调试级别），再怀疑拓扑或选型 |
| 显存峰值 | `pytorch.api.max_memory_allocated` 与 Step 4 预估比 | 偏差 > 20% | 回头标定 `c_d` / `c_p`；检查是否漏算通信缓冲或临时工作区 |
| rank 间耗时方差 | `max(rank 耗时) / min(rank 耗时)`，按 `profiler.kernel_table_columns.device_id` 分组 | > 1.2 | **MoE 首先怀疑专家负载不均**：统计各专家被路由到的 token 数，极差大则考虑专家重排、容量因子、冗余热专家；Dense 模型怀疑 PP 切分不均或批内长度不均 |
| 流水气泡率 | `1 − (Σ 各 stage 实际计算时长) / (PP × 端到端时长)`，与理论值 `(PP − 1) / (m + PP − 1)` 比 | 实测显著高于理论 | 增加并发微批数 `m`；检查 stage 间负载是否均衡 |
| decode 访存效率 | `每步实际读取字节 / 每步耗时` 与 `devices.models[].memory_bandwidth_gb_s` 比 | < 50% | decode 是访存受限的，低于此值说明瓶颈不在带宽而在 kernel 启动或通信，并行度不是正确的调节旋钮 |

**校准循环的出口条件**：候选配置中实测最优的那个，其通信占比 < 30%、显存峰值与预估偏差 < 20%、rank 间方差 < 1.2。三条都满足才算选型收敛；否则带着实测数据回到 Step 2 重新选。

## 输出要求

### 分析报告结构

**1. 结论摘要**：推荐哪个候选、为什么、最大的不确定性是什么。

**2. 输入与假设**：模型结构参数表；用的哪份 profile；SLO 目标；所有取自 `assumed` / `unknown` 字段的值，逐条列出并标注对结论的影响。

**3. 模块选型表**：Embedding / Attention / Dense FFN / MoE / LMHead / Norm 各自的切法与理由。

**4. 候选配置对比表**（每个候选都必须填满）：

| 项 | 候选 A（时延优先） | 候选 B（吞吐优先） | 候选 C（显存余量优先） |
|---|---|---|---|
| TP / PP / DP / EP | | | |
| 节点数 × 每节点卡数 | | | |
| 单卡权重显存（GB） | | | |
| 单卡 KV cache 显存（GB，按目标并发与长度） | | | |
| 单卡激活峰值（GB，标明实测标定或估算） | | | |
| 单卡显存合计 / 上限 | | | |
| decode 每 token 通信次数与字节量 | | | |
| 估算通信时间占比 | | | |
| 硬约束检查 | 全部通过 | | |
| 主要优势 | | | |
| 主要代价 | | | |
| 最大风险与验证方法 | | | |

**5. 被排除的方案**：列出来并写清楚卡在哪条硬约束上。这部分的价值不亚于候选本身——它能阻止别人再走一遍同样的死路。

**6. 验证计划**：先跑哪个候选、采什么数据、按 Step 8 的哪些阈值判定。

### 报告纪律

- 每个数字都要能追溯到公式和输入；公式里的每个符号都在 Step 1 的符号表里
- 取自 profile 的数字标注字段路径；实测值标注测量方法；估算值标注估算依据与置信度
- profile 中为 `null` 的字段，报告里写"待确认"并说明它影响哪个结论，**不要用业界常见值顶替**
- 不输出任何代码改动建议的细节实现，只说"该模块应按 X 维切分"

## 知识来源

- Megatron-LM 系列公开论文：《Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism》的张量并行切分方式（FFN 列切-行切组合、attention 按头切、词表维嵌入并行），《Efficient Large-Scale Language Model Training on GPU Clusters Using Megatron-LM》的流水并行与气泡率模型，《Reducing Activation Recomputation in Large Transformer Models》的序列并行（以 reduce-scatter + all-gather 替代 all-reduce）
- DeepSpeed 公开文档与论文中关于推理并行、专家并行与通信代价的论述
- GShard / Switch Transformer 公开论文中的专家并行与 all-to-all dispatch/combine 机制，以及专家负载不均衡问题
- 集合通信的标准代价模型：环状 all-reduce 的 `2(N−1)/N` 传输量、`α + β` 形式的延迟-带宽时间模型（公开教科书与 MPI 文献中的通用结论）
- vLLM 与同类开源推理框架公开的并行配置约定：张量并行度与注意力头数、KV 头数的整除关系，以及 KV 头数小于并行度时的复制行为
- DeepSeek-V2 / V3 公开技术报告中关于压缩潜在表示 KV 缓存的描述
- PyTorch 官方文档：`torch.distributed` 的集合通信原语语义、显存统计 API
