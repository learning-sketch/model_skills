---
name: model-infer-kvcache
description: 为目标加速卡上的 LLM 推理设计并接入 KV cache，重点是分页式（paged）缓存：物理块池的显存规划、逻辑块到物理块的映射、block_table 与 slot_mapping 的职责划分、变长与定长两种批布局下的序列元数据构造、不同 attention 形态（MHA/GQA、滑动窗口、压缩潜在表示、混合层）的缓存策略，以及初始化、运行期、多轮复用三条验证链路。当模型已能单卡跑通、需要支持多请求并发或长上下文、或 KV cache 显存成为瓶颈时使用。不覆盖首次迁移与基线采集、并行切分选型、量化、图模式与调度器策略。
---

# KV cache 与分页 attention 接入

前置条件：模型**已经单卡 eager 跑通并有性能基线与参考输出**（见 `model-infer-migrator`）。
本 skill 改变的是显存组织方式和 attention 的调用契约，判断"改对了"的唯一依据，就是用基线里的参考输出做逐 token 比对。

## 硬件依赖

| profile 字段 | 用途 | 缺失时的降级行为 |
|---|---|---|
| `kernels.attention.paged_attention` | decode 阶段带分页 KV 的 attention 算子 | 为 `null` 时走 Step 8 的三级降级路径（朴素 → 索引层 → 接算子），并在 `unknown_or_unsupported` 登记 |
| `kernels.attention.flash_attention_varlen` | prefill 阶段变长拼包 attention | 为 `null` 时 prefill 走定长填充布局 + SDPA，`input_len` 受 O(S²) 显存限制 |
| `kernels.attention.sdpa_available` | 所有降级路径的兜底 | 为 false 时手写 `matmul + mask + softmax + matmul` |
| `kernels.attention.kv_cache_layout` | 决定 cache 张量的维度顺序与写入索引方式 | **阻断**。不要猜布局，向用户索取算子文档；猜错的表现是输出乱码而非报错 |
| `kernels.attention.mla_support` | 是否有压缩潜在表示（MLA 类）的专用算子 | 非 `true` 时把潜在表示展开成常规 K/V 再走 MHA 路径，显存收益归零但功能正确 |
| `kernels.attention.supported_mask_modes` | causal / 滑动窗口由算子内建还是调用方构造 | 不含所需模式时由调用方显式构造 additive mask |
| `kernels._status` | `assumed` / `unknown` 的算子先验证再用 | 用到了就在输出的 `assumptions` 中登记，并先跑单算子对拍 |
| `pytorch.device_type` | cache 张量的设备 | **阻断**，向用户确认 |
| `pytorch.api.memory_allocated` / `max_memory_allocated` | 权重与激活占用，用于推算可分配块数 | 用 `diagnostics.smi_command` 采样代替，并把安全系数从 0.90 下调到 0.80 |
| `pytorch.api.empty_cache` | 测空闲显存前清理缓存池 | 跳过，改为在独立进程里测 |
| `devices.models[].memory_gb` | 显存总量 | 用 smi 实测值 |
| `graph.graph_capture.constraints.static_address_required` | 若后续要进图，cache 必须一次性分配且地址不变 | 按 `true` 从严设计：块池一次分配，运行期不扩容 |
| `diagnostics.error_signatures.illegal_memory_access` | 索引越界的典型症状 | 按"非法内存访问"模糊匹配 |
| `diagnostics.sanitizer.command` | 排查越界写 | 跳过，改用主机侧的索引断言兜底 |

### profile 未覆盖、必须向用户确认的信息

以下信息本 skill 必须有，但当前 profile schema 没有对应字段。**逐条问用户或查算子文档，不要猜。**

- 分页算子支持的 `block_size` 取值集合或整除约束
- 分页算子支持的 KV cache dtype（是否支持低于 16 bit 的 KV）
- 分页算子对 `head_dim` 的取值限制
- `block_table` / `slot_mapping` / 序列长度数组要求的整数位宽与设备位置
- K 与 V 是否使用同一种布局（部分实现里两者不同）
- 是否支持"带已有 KV 的 prefill"（分块 prefill / 前缀复用）
- 空闲显存查询接口（PyTorch 侧用于推算块数的 free/total 查询）

## 适用场景

- 模型已跑通，要从"一次一条序列"走向多请求并发
- 上下文变长，按最大长度预留显存已经放不下目标并发数
- 需要前缀复用、多轮对话、分块 prefill 等依赖块级管理的能力
- KV cache 已经成为显存瓶颈，需要定量规划块池大小

## 不适用场景

| 情形 | 应该去哪 |
|---|---|
| 模型还没在本卡跑通 | `model-infer-migrator` |
| 要决定 TP/PP/EP 怎么切 | `model-infer-parallel-analysis`（KV cache 的切分维度在那里定） |
| 要做调度策略：抢占、换出、优先级、前缀树 | 调度器设计，不在本 skill 范围 |
| KV cache 量化到低比特 | 量化相关 skill。本 skill 只负责布局正确 |
| 要把 attention 进图或融合 | 图模式 / 融合相关 skill |

## 工作流

### Step 0 · 读 profile，判定能力等级

| 等级 | 判据 | 本次目标 |
|---|---|---|
| **P2 完整** | `paged_attention` 与 `flash_attention_varlen` 均非 null | 直接做分页接入 |
| **P1 部分** | 只有其一 | 有的那一侧用算子，另一侧走对应的降级实现 |
| **P0 空白** | 两者都是 null，`sdpa_available` 为 true | 走 Step 8 的三级路径，先用朴素实现跑通并产出参考输出 |

无论哪一级，先记录模型的缓存形状参数：`num_layers`、`num_key_value_heads`、`head_dim`、KV 的 dtype、是否 GQA、是否有滑动窗口层、是否为压缩潜在表示、是否存在层间异构。

### Step 1 · 为什么要分页：先把浪费算出来

按"每条序列预留一段连续显存"分配时，有两类浪费：

```
预留浪费（reserved waste）
  = Σ_i (L_reserve − L_i) × per_token_bytes
  其中 L_reserve 是按最大上下文预留的长度，L_i 是序列 i 的实际长度
  浪费比例 = 1 − mean(L_i) / L_reserve

外部碎片（external fragmentation）
  长短序列反复进出后，空闲显存被切成放不下新序列的碎片。
  表现为"总空闲够、但分配失败"。
```

分页之后，浪费只剩**最后一块的内部碎片**，有确定的上界：

```
内部碎片 tokens = Σ_i (ceil(L_i / block_size) × block_size − L_i)
                ≤ B × (block_size − 1)
内部碎片比例    ≤ B × (block_size − 1) / Σ_i L_i
```

**决策判据**：若 `1 − mean(L) / L_reserve > 0.3`，或已经出现"空闲够但分配失败"，分页的收益明确；否则先确认瓶颈真的在 KV cache 上，不要为了分页而分页。

每 token 的缓存字节数（后续所有公式的基石）：

```
per_token_per_layer_bytes = 2 × num_key_value_heads × head_dim × dtype_bytes
                            ↑ K 与 V 各一份
per_token_bytes           = num_layers × per_token_per_layer_bytes

符号：num_key_value_heads 为单卡持有的 KV 头数（张量并行下已按 TP 切分）；
      head_dim 为单头维度；dtype_bytes 为 KV 存储 dtype 的字节数（bf16/fp16 = 2）。
```

### Step 2 · 逻辑块到物理块：两张表，各管一头

分页的核心是**让逻辑上连续的序列，落在物理上离散的块里**。两张元数据表职责严格分离：

| 表 | 形状 | 谁用 | 管什么 | 何时变化 |
|---|---|---|---|---|
| `block_table` | `[num_seqs, max_blocks_per_seq]` 整型 | attention 算子**读**历史 KV | 序列 i 的第 j 个逻辑块落在哪个物理块 | 序列跨过块边界时追加一项 |
| `slot_mapping` | `[num_tokens_this_step]` 整型 | cache **写入**本步新 KV | 本步第 t 个 token 的 K/V 写到哪个物理位置 | 每步都重建 |

**为什么要分成两张表**：写是"本步的少量新 token，一个 token 一个位置"，天然是逐 token 的扁平索引；读是"整段历史，按块批量取"，天然是逐块的索引。合成一张表会让其中一侧退化（要么写入时做块内偏移换算，要么读取时逐 token 寻址）。

地址计算：

```
输入：序列在批次中的下标 seq_index，token 在序列中的绝对位置 position（从 0 开始）

logical_block_index = position // block_size
block_offset        = position %  block_size
physical_block_id   = block_table[seq_index][logical_block_index]

当块内按 token 连续存放时（例如 [num_blocks, block_size, num_kv_heads, head_dim]）：
    slot = physical_block_id × block_size + block_offset
    写入：cache.view(num_blocks × block_size, num_kv_heads, head_dim)[slot] = kv_vector

当块内按头优先存放时（例如 [num_blocks, num_kv_heads, block_size, head_dim]）：
    扁平 slot 不成立，必须拆成两级索引：
    写入：cache[physical_block_id, :, block_offset, :] = kv_vector
```

**布局从 `kernels.attention.kv_cache_layout` 读，不要写死。** 扁平化 `slot` 只在"块内 token 连续"的布局下成立，这是接入时最容易出错的一处：索引算错不会报错，只会让输出变成语法通顺但语义错乱的文本。

`block_table` 的填充项（行内超出 `ceil(L_i / block_size)` 的部分）必须永远不被读到——由序列长度元数据界定，见 Step 3。填充值用 0 还是 −1 取决于算子约定，向用户确认。

### Step 3 · 两种批布局下的序列元数据

| 维度 | 变长拼包（packed / varlen） | 定长填充（padded） |
|---|---|---|
| query 张量形状 | `[total_tokens, num_heads, head_dim]` | `[batch, max_len, num_heads, head_dim]` |
| 长度信息载体 | 前缀和数组 | 长度数组 + mask |
| 计算浪费 | 无 | padding 位置白算，长短混批时浪费显著 |
| 实现难度 | 高，元数据多 | 低 |
| 依赖 | `flash_attention_varlen` 非 null | 仅需 SDPA |

**统一约定（整篇遵守）：先把本步的 K/V 写入 cache，再调用 attention。** 因此下文所有 `seq_len` 都**包含当前 token**。若实现顺序相反，所有 `seq_len` 要减 1，必须在代码里写明。

**Prefill 阶段**（`query_len_i` 为本步要算的 token 数，`context_len_i` 为该序列此前已缓存的 token 数，无前缀复用时为 0）：

```
query_start_loc[0] = 0
query_start_loc[i+1] = query_start_loc[i] + query_len_i        # 长度 batch+1，int32
seq_len_i          = context_len_i + query_len_i               # 长度 batch
max_query_len      = max(query_len_i)                          # 主机侧标量
max_seq_len        = max(seq_len_i)                            # 主机侧标量
slot_mapping       长度 = Σ query_len_i，逐 token 按 Step 2 公式算
block_table        每行有效项数 = ceil(seq_len_i / block_size)
```

注意 `context_len_i > 0` 的情形（分块 prefill / 前缀复用）：query 的前缀和与 key 的前缀和**不是同一个数组**，causal mask 必须**右下对齐**——query 的最后一个位置对齐 key 的最后一个位置。这是分块 prefill 最常见的错误。

**Decode 阶段**（每条序列恰好 1 个新 token）：

```
query_len_i        = 1
query_start_loc    = [0, 1, 2, ..., batch]                     # 即 arange(batch+1)
seq_len_i          = 该序列已缓存的 token 总数（含本步写入的这一个）
slot_mapping       长度 = batch，每条序列一个位置
block_table        同 prefill
```

**定长填充布局**额外需要：有效位置 mask `[batch, max_len]`；`slot_mapping` 在 padding 位置填算子约定的忽略值（若算子不支持忽略值，则先把有效 token 压实再写）。

元数据的硬性检查（每步都做，用环境变量控制开关）：

```
query_start_loc 单调不减，首项 == 0，末项 == total_tokens，dtype 为算子要求的整型
seq_len_i ≥ query_len_i > 0
max(seq_len_i) ≤ block_table 行内有效项数 × block_size
slot_mapping 内所有有效值 ∈ [0, num_blocks × block_size)，且本步内两两不重复
block_table 内所有有效值 ∈ [0, num_blocks)
```

### Step 4 · 按 attention 形态选缓存策略

| 形态 | 缓存什么 | 每层每 token 字节 | 关键注意 |
|---|---|---|---|
| 标准 MHA | 全部 K、V 头 | `2 × num_heads × head_dim × b` | `num_key_value_heads == num_heads` |
| GQA / MQA | 仅 KV 头 | `2 × num_key_value_heads × head_dim × b` | **绝不能在写入 cache 之前把 KV 头复制到 query 头数**，否则显存乘以分组数。复制只能发生在 attention 内部或由算子隐式处理 |
| 滑动窗口 | 仅窗口内的 K、V | 同上，但块数上限为 `ceil(window_size / block_size) + 1` | 块可环形复用；mask 下界为 `position − window_size + 1`；`block_table` 只挂当前窗口覆盖的块 |
| 压缩潜在表示（MLA 类） | 压缩后的潜在向量 + 解耦的位置编码分量 | `(d_latent + d_rope) × b`，**无 KV 头维度** | 需 `kernels.attention.mla_support == true`。否则在层内把潜在向量展开回 K/V 再走 MHA 路径：功能正确、显存收益归零，须在输出中明确标注 |
| 固定状态层（线性 attention / 状态空间类） | 固定大小的状态，与序列长度无关 | `state_bytes`，与 `L` 无关 | 按 batch 分配，**不进块池**；生命周期与序列绑定 |
| 层间异构（部分层全局、部分层滑窗或无 cache） | 分组 | 各组求和 | **按组建独立块池与独立 `block_table`**；不要用一个全局 `block_size` 硬套所有层 |

`b` = KV 存储 dtype 的字节数。

层间异构模型的分组纪律：先按"每 token 字节数 + 最大需保留长度"给层分组，组内共用块池；调度器为每条序列维护"每组一个 `block_table`"。混在一起会出现"滑窗层的块被全局层的长序列挤占"这类难以定位的容量问题。

### Step 5 · 显存规划

```
【块大小】
block_bytes_per_layer = block_size × per_token_per_layer_bytes
block_bytes_all_layers = num_layers × block_bytes_per_layer

【可用于 cache 的显存】
available_bytes = total_device_bytes × utilization
                  − weight_bytes
                  − peak_activation_bytes
                  − comm_buffer_bytes
                  − runtime_reserved_bytes

  total_device_bytes     取自 devices.models[].memory_gb（若 profile 缺失则 smi 实测）
  utilization            安全系数，默认 0.90；显存读数来自 smi 采样时降到 0.80
  weight_bytes           权重加载完成后实测（pytorch.api.memory_allocated）
  peak_activation_bytes  用最大 batch × 最长 prefill 跑一次实测峰值减去 weight_bytes
  comm_buffer_bytes      多卡时的集合通信缓冲，单卡为 0
  runtime_reserved_bytes 驱动与运行时常驻，实测总占用减去上述各项

【块数】
num_blocks = floor(available_bytes / block_bytes_all_layers)

【并发能力】
blocks_per_seq(L)   = ceil(L / block_size)
concurrent_seqs(L)  = floor(num_blocks / blocks_per_seq(L))

【反推：要支撑 N 条长度 L 的序列，需要多少显存】
required_bytes = N × ceil(L / block_size) × block_bytes_all_layers
```

`peak_activation_bytes` 必须**实测**，不要估算：它随 prefill 的 chunk 大小、attention 实现（是否 O(S²) 中间矩阵）、是否有临时工作区而剧烈变化。实测方法是在不分配块池的情况下跑一次最大负载的 prefill。

**`block_size` 选型权衡**：

| `block_size` | 内部碎片 | `block_table` 长度与索引开销 | 算子效率 | 前缀复用粒度 |
|---|---|---|---|---|
| 小（如 16） | 最低 | 最长 | 单块 token 少，可能喂不饱计算单元 | 最细 |
| 中（如 32–64） | 中 | 中 | 通常是折中点 | 中 |
| 大（如 128） | 最高（上界 `B × 127` tokens） | 最短 | 最高 | 最粗 |

**硬约束优先于上述权衡**：取值必须落在分页算子支持的集合内。profile 当前没有这个字段，向用户确认后再选。选定后用两组真实长度分布各跑一次，比较 `内部碎片比例` 与 `decode 时延`，再定稿。

### Step 6 · 生命周期与所有权

**所有权规则（违反它会产生最难查的 bug）**：

- 序列长度、`block_table`、块的分配与回收，**由调度器唯一维护**
- attention 层**只读**这些元数据，不修改、不缓存、不自行推断长度
- 层内不得从张量形状反推序列长度（变长布局下形状不含这个信息）

块池操作与不变量：

| 操作 | 动作 | 不变量 |
|---|---|---|
| `allocate(seq, n_tokens)` | 从空闲表取 `ceil(n_tokens / block_size)` 块 | 取不到则拒绝入队，不要部分分配 |
| `append(seq)` | 仅当 `seq_len % block_size == 0` 时取新块 | 其余步不分配 |
| `free(seq)` | 块引用计数减一，归零才回空闲表 | `已分配块数 + 空闲表长度 == num_blocks` 恒成立 |
| `fork(seq)` | 共享块、引用计数加一；写时复制最后一块 | 被共享的块不可原地写 |

**脏数据的处理**：块回收后内容不清零（清零的带宽开销在高吞吐下不可忽略）。正确性由两道防线保证：

1. **第一道（必须有）**：`seq_len` 严格界定可读范围，attention 永不读超出 `seq_len` 的位置。
2. **第二道（按算子行为决定）**：若算子以**整块为粒度**读取（即会读到最后一块 `seq_len` 之后的槽位），则必须保证这些位置被 mask 覆盖，或在分配该块时清零其尾部。向用户确认算子是否整块读取——这个问题不问清楚，表现为"偶发的、与并发历史相关的输出异常"。

### Step 7 · 三条验证链路

**链路 1：初始化分配正确**

```
[ ] num_blocks > 0；若为 0，先查 available_bytes 的各项减法是不是哪一项估大了
[ ] 每层 cache 张量的 shape 与 kernels.attention.kv_cache_layout 约定逐维一致
[ ] K 与 V 分别断言（两者布局可能不同）
[ ] dtype 与算子要求一致；张量 contiguous
[ ] 实际分配字节数与 num_blocks × block_bytes_all_layers 的偏差 < 1%
[ ] 分配完成后设备剩余显存 ≥ runtime_reserved_bytes，且再跑一次最大 prefill 不 OOM
```

**链路 2：运行期 shape 与 dtype 正确**

```
[ ] Step 3 末尾的全部元数据检查，在每步执行（调试开关打开时）
[ ] 写入抽查：随机取若干 (seq_index, position)，按 Step 2 公式从 cache 读回，
    与该 token 本步的 k/v 向量 allclose（bf16 下 atol=1e-2, rtol=1e-2）
[ ] attention 输入输出的 shape / dtype / 设备在每层入口断言
[ ] 越界写的排查：报出 diagnostics.error_signatures.illegal_memory_access 时，
    优先怀疑 slot_mapping 与 block_table，再用 diagnostics.sanitizer.command 复查
```

**链路 3：多轮对话不串数据**

```
[ ] A/B/A 测试：跑 prompt A 记 out_A1 → 跑 prompt B → 再跑 A 记 out_A2，
    要求 out_A1 与 out_A2 的 token id 序列完全一致
[ ] 交错批测试：A 与 B 同批跑，各自结果与单独跑时完全一致
[ ] 回收压力测试：反复 allocate / free 1000 轮后再跑 A，结果不变
[ ] 长短混批：把最长与最短序列放同一批，验证短序列不被长序列的残留污染
[ ] 与 model-infer-migrator 产出的 accuracy_reference 逐 token 比对，必须完全一致
```

三条链路有任何一条不过，不要进入性能测量——分页接入的性能数字只有在正确性成立时才有意义。

### Step 8 · 降级路径（`paged_attention` 为 null 时）

**不要为了等算子而停工，也不要一步跨到分页算子。** 分三级推进，每一级的验收条件都是"输出 token id 与上一级完全一致"：

| 级别 | 做什么 | 验收 | 产出 |
|---|---|---|---|
| **L0 朴素连续** | 每条序列一段连续 cache `[batch, num_kv_heads, max_len, head_dim]`，定长填充，用 SDPA 或手写 attention | 与 migrator 的 `accuracy_reference` 逐 token 一致 | 参考输出；显存与并发上限的现状数据 |
| **L1 索引层先行** | 引入块池、`block_table`、`slot_mapping`，但 attention 之前先按 `block_table` 把块 gather 成连续张量，再喂给 L0 的 attention | 与 L0 逐 token 一致 | **索引逻辑已被单独验证过**，这是最大的价值 |
| **L2 接真算子** | 把 L1 的 gather + dense attention 换成分页算子 | 与 L1 逐 token 一致；逐层 hidden state 偏差在阈值内 | 最终实现 |

L1 是关键一级：它把"索引算错"与"算子用错"这两类问题彻底分开。跳过 L1 直接做 L2，一旦输出乱码，无法判断是地址公式错了还是算子参数传错了。

L1 的 gather 实现本身很慢（多一次完整的 KV 拷贝），这是预期内的，不要在这一级做性能优化。

同时在 profile 的 `unknown_or_unsupported` 中登记：

```yaml
- capability: kernels.attention.paged_attention
  status: unknown          # 或 not_supported
  reason: <为什么现在没有>
  affects: ["model-infer-kvcache"]
```

## 输出要求

### 1. 缓存规划表（`kv_cache_plan.json`）

```json
{
  "schema_version": 1,
  "model": {
    "num_layers": 0, "num_key_value_heads": 0, "head_dim": 0,
    "attention_form": "mha | gqa | sliding_window | latent_compressed | hybrid",
    "layer_groups": [{ "name": "global", "layers": [0], "per_token_per_layer_bytes": 0 }]
  },
  "kv_dtype": "bfloat16", "dtype_bytes": 2,
  "layout": { "source": "profile.kernels.attention.kv_cache_layout", "value": "", "k_and_v_same": true },
  "block_size": 0,
  "block_size_rationale": "算子约束集合 + 碎片与效率权衡的实测依据",
  "memory": {
    "total_device_bytes": 0, "utilization": 0.9,
    "weight_bytes": 0, "peak_activation_bytes": 0,
    "comm_buffer_bytes": 0, "runtime_reserved_bytes": 0,
    "available_bytes": 0, "block_bytes_all_layers": 0, "num_blocks": 0,
    "measured_not_estimated": ["weight_bytes", "peak_activation_bytes"]
  },
  "capacity": [{ "seq_len": 4096, "blocks_per_seq": 0, "concurrent_seqs": 0 }],
  "fragmentation": { "internal_waste_tokens_upper_bound": 0, "internal_waste_ratio_at_measured_lengths": 0.0 },
  "capability_level": "P0 | P1 | P2",
  "degradation_stage": "L0 | L1 | L2 | n/a",
  "assumptions": [{ "field": "", "status": "assumed | unknown", "impact": "", "action": "" }]
}
```

### 2. 接口契约文档

写清楚 attention 层从调用方收到哪些元数据、每个的形状与 dtype、`seq_len` 是否含当前 token、`block_table` 的填充值约定、prefill 与 decode 的差异。**这份契约是后续所有人改这段代码的依据，口头约定一定会丢。**

### 3. 验证报告

Step 7 三条链路的逐项结论 + 证据（日志片段、比对输出）。链路 3 的 A/B/A 结果必须附完整的 token id 序列对比。

### 4. 性能对照

与 migrator 基线的同口径对比：相同用例下的 TTFT、TPOT、峰值显存、可并发序列数。分页的收益主要体现在**并发数和显存**，单路时延可能因为索引开销**略有退化**，这是正常的，如实报告。

## 知识来源

- vLLM 的 PagedAttention 公开论文《Efficient Memory Management for Large Language Model Serving with PagedAttention》（SOSP 2023）：逻辑块/物理块映射、块表、内部碎片上界、写时复制共享等概念
- vLLM 与 FlashInfer 开源实现中公开的接口约定：块表与 slot 映射的分工、cache 布局的维度顺序命名、块级元数据的组织方式
- FlashAttention 开源实现公开的变长接口约定：累积序列长度数组的语义（长度为 `batch+1`、首项为 0、整型）、causal mask 在 `q_len != kv_len` 时的右下对齐规则
- HuggingFace Transformers 公开的建模约定：GQA/MQA 的 KV 头组织、滑动窗口 attention 的掩码定义
- DeepSeek-V2 / V3 公开技术报告中描述的压缩潜在表示 KV 缓存思路：缓存低维潜在向量与解耦的位置编码分量，而非完整 K/V
- PyTorch 官方文档：`scaled_dot_product_attention`、显存统计 API、张量索引与视图语义
