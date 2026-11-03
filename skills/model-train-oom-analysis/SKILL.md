---
name: model-train-oom-analysis
description: 诊断训练时的显存不足（OOM）。当用户说「训练跑不起来，显存爆了」「out of memory」「显存不够怎么调」「跑了几百步之后才 OOM，是不是漏了」「显存一直涨」「batch 只能开到 1」「碎片是不是太多了」，或者贴出一段显存分配失败的报错、给出一份 memory snapshot 文件时使用。覆盖从报错分类、静态容量估算、快照深度分析到泄漏定位与优化顺序的完整流程。
---

# 训练显存不足（OOM）诊断

OOM 不是一个问题，是三个问题共用了一条报错：**配置本身就超了**、**某条路径的峰值超了**、**有东西在漏**。三者的处置完全不同，强行用同一套手段去试（先减 batch，再开重计算，再减序列长）会浪费大量时间，而且常常把真正的病因掩盖掉。

所以这个 skill 的第一步不是动手调参，是**分类**。

---

## 硬件依赖

| profile 字段 | 用途 | 缺失时的降级行为 |
|---|---|---|
| `diagnostics.oom_sources.caching_allocator` | Step 0 判定是缓存分配器 OOM | 退回通用关键字（`out of memory` / `Tried to allocate`），可能误判 |
| `diagnostics.oom_sources.workspace_allocator` | 判定是算子临时工作区 OOM | 空列表则认为该芯片没有独立的 workspace 分配器，这类 OOM 并入缓存分配器 |
| `diagnostics.oom_sources.collective_buffer` | 判定是通信缓冲 OOM | 无法区分，只能从调用栈里有没有通信库帧来猜 |
| `diagnostics.oom_sources.host_memory` | 判定其实是主机内存不够 | 容易把主机侧 OOM 误诊成显存问题，要人工确认 |
| `diagnostics.error_signatures.out_of_memory` | 从日志里捞出 OOM 行 | 退回通用关键字 |
| `pytorch.api.record_memory_history` | Step 2 开启快照采集 | **Step 2 整步不可用**，只能靠静态估算 + 水位探测 |
| `pytorch.api.dump_memory_snapshot` | Step 2 导出快照 | 同上 |
| `pytorch.memory_snapshot_format_compatible` | 决定 `analyze_memory_snapshot.py` 能不能直接吃 | 为 `false` 时脚本结果不可信；为 `null` 时先拿一份真实快照核对结构 |
| `pytorch.api.memory_allocated` / `max_memory_allocated` | 轻量水位探测，不用导快照就能看峰值 | 退回 `smi_query_memory`，粒度粗（只有进程级 reserved） |
| `pytorch.api.reset_peak_memory_stats` | 分阶段测峰值（前向 / 反向 / 优化器各一段） | 只能测全程峰值，定位不到阶段 |
| `pytorch.api.empty_cache` | 碎片缓解与「到底是不是真不够」的判别 | 无法区分「真不够」与「缓存占着不还」 |
| `pytorch.api.set_per_process_memory_fraction` | 设内存水线，把偶发 OOM 变成稳定复现 | 只能靠加大 batch 硬复现，代价高 |
| `pytorch.device_namespace` | 所有示例代码的命名空间 | 示例需要手工改写 |
| `devices.models[].memory_gb` | Step 1 静态估算的预算上限 | 估算只能给绝对值，给不出「超了多少」 |
| `collective.library` / `collective.torch_backend` | 注入到语义分类的通信关键字，使自研通信库的分配能被归类 | 只能靠默认关键字（`nccl` / `gloo` / `c10d` 等）识别，自研库的缓冲会落进未分类 |
| `diagnostics.memory_tools.snapshot_visualizer` | 快照的图形化查看入口 | 只能看脚本的文本报告 |
| `diagnostics.memory_tools.deep_analysis_tool` | 更重的离线分析工具 | 跳过 |
| `kernels.memory.host_offload_api` | Step 4 的卸载方案能不能走通 | 卸载一项从优化矩阵中划掉 |
| `kernels.memory.unified_memory` | 是否有统一内存/页迁移兜底 | 当作没有 |
| `graph.graph_capture.available` | 图捕获会把内存池固定住，诊断时要先关掉 | 不提示，可能得到被图捕获污染的快照 |
| `training_framework.name` / `config_style` | Step 4 的措施落到哪个配置项 | 只能给出通用措施名，用户自己找配置入口 |
| `collective.fused_comm_compute_ops.available` | 影响 MoE 场景能否靠融合算子省掉中间缓冲 | 当作没有 |

---

## 适用场景

- 训练启动后很快报显存不足，想知道是配置超了还是哪里有问题
- 跑了一段时间后才 OOM，怀疑有泄漏
- 显存占用曲线单调上涨
- 显存总量看起来够，但一次大分配就失败（碎片型 OOM）
- 想在改配置之前先算清楚「这个模型这套并行到底需要多少显存」
- 多卡训练中只有个别 rank OOM

## 不适用场景

- **推理时的显存规划**。KV cache 的容量规划、分页策略、批大小上限是另一套模型，不要套用训练的公式
- **主机内存 OOM**。报错里出现 `DefaultCPUAllocator` 一类字样时，问题在主机侧（dataloader worker 数、pinned memory、数据集缓存），和加速卡显存无关
- **性能调优**。省显存的手段几乎都在拿时间换空间。「怎么更快」是另一个问题，两者的最优解经常相反
- **精度问题**。改了并行切分之后 loss 对不上，那是数值问题，不在这里
- **显存够但进程被 OOM Killer 杀掉**。那是系统内存，看 `dmesg`

---

## 工作流

### Step 0：先分类，不要先动手

这一步最关键，也最容易被跳过。**两件事：OOM 的来源，以及 OOM 的发生时机。**

#### 0.1 来源分类

拿报错原文去对 `{{profile.diagnostics.oom_sources}}` 里的特征串：

| 来源 | 特征 | 含义 | 下一步 |
|---|---|---|---|
| 缓存分配器 | `{{profile.diagnostics.oom_sources.caching_allocator}}` | 框架的显存池要不到内存。报错里通常带「请求了多少、已分配多少、已预留多少」 | 走下面完整流程 |
| 算子工作区 | `{{profile.diagnostics.oom_sources.workspace_allocator}}` | 某个算子要的临时工作区拿不到。这一类的量通常远小于主占用，往往是**碎片**而不是总量问题 | 直接跳到碎片分析与 Step 4 的分配器配置 |
| 通信缓冲 | `{{profile.diagnostics.oom_sources.collective_buffer}}` | 集合通信库自己的缓冲要不到内存。这部分通常**不在**框架显存池的统计里 | 查通信库的缓冲配置（`{{profile.collective.env_vars.buffer_size}}`），同时注意它和框架显存池在抢同一块物理显存 |
| 主机内存 | `{{profile.diagnostics.oom_sources.host_memory}}` | 根本不是显存问题 | 退出本流程，查 dataloader / pinned memory / 数据集缓存 |

profile 里这几项为空时，不要假装能分类。明确告诉用户「这颗芯片的 OOM 来源特征串还没填，下面按缓存分配器 OOM 处理」。

同时记录报错里的三个数字——**本次请求量、已分配量、已预留量**。它们是后面所有判断的锚点：

- 请求量很大（> 1 GiB）→ 多半是某个具体张量开太大，去 Top-N 大块里找
- 请求量很小（< 100 MiB）但失败了 → 几乎可以肯定是碎片问题，不是总量问题
- 已预留量远大于已分配量 → 显存池里囤着大量碎片

#### 0.2 时机分类

**这一步决定了后面走哪条路，两条路完全不同。**

| 时机 | 判断 | 走哪条路 |
|---|---|---|
| **前 1~3 个 step 就 OOM** | 配置本身超限。显存需求是静态的，跑第一步就超说明这套配置在这块卡上根本放不下 | **Step 1 静态估算 → Step 4 优化**。不要采 snapshot，算一下就知道超了多少 |
| **几十到几百步之后才 OOM** | 疑似泄漏，或者长尾样本 | **Step 3 泄漏检测**。先排除「变长序列的长尾 batch」这个常见误诊 |
| **第一次 eval / 第一次存 checkpoint 时 OOM** | 不是训练主路径的问题，是那条路径自己的峰值高 | 单独给那条路径做 Step 2，常见原因是 eval 忘了 `no_grad`、checkpoint 把整个 state_dict 先搬到一张卡上 |
| **某些 batch 上 OOM、换个数据顺序就好了** | 动态 shape。变长序列让峰值随 batch 内容波动，叠加碎片 | Step 2 看碎片 + Step 4 的分配器配置；考虑按 token 数而不是样本数组 batch |
| **只有个别 rank OOM** | 负载不均 | 流水并行的第一个 stage 要存最多的中间激活；MoE 的热点专家吃掉额外显存；数据并行下某个 rank 拿到了最长的序列 |
| **换了芯片/框架版本后才开始 OOM** | 对照实验。先确认是不是默认分配策略、默认重计算开关、默认通信缓冲大小变了 | 直接对比两边的配置，不要先从代码里找 |

**判据要用数据而不是印象。** 「跑了很久才 OOM」要用日志里的显存曲线确认：如果显存从第 1 步到 OOM 前一直在涨，是泄漏；如果一直平着、只在某一步突然跳起来，是那一步的输入特殊。这一步可以直接借
[`model-train-log-visualization`](../model-train-log-visualization/) 画出 `memory` 曲线。

### Step 1：静态估算（先算，再采）

**先算再采。算出来就超了，就不必采 snapshot。** 采集快照要改代码、重跑、等它再 OOM 一次，成本不低；而静态估算只需要几个超参数，几分钟就能给出结论。

符号：`P` 参数量，`L` 层数，`h` 隐藏维，`a` 注意力头数，`s` 序列长，`b` micro-batch size，`N` 数据并行度，`T` 张量并行度，`PP` 流水并行度。

#### 1.1 参数

```
M_param = P × bytes_per_weight
```

- 纯 fp32 训练：`bytes_per_weight = 4`
- 纯 bf16/fp16 训练：`2`
- **混合精度要算两份**：计算用的低精度副本 `2P` ＋ 优化器用的 fp32 主权重 `4P`。主权重这笔钱经常被漏掉，它通常算在优化器状态里（见下）
- 参数被分片时（ZeRO-3 / FSDP full shard，分片数 `N`）：`M_param = P × bytes / N`，但**峰值要加上一次 all-gather 的量**：`+ (最大一个 FSDP unit 的参数量) × bytes`
- 张量并行 `T`：`M_param = P × bytes / T`（注意 embedding / norm 等不切的部分要单独算）

#### 1.2 梯度

```
M_grad = P × bytes_per_grad
```

- fp32 梯度 `4P`，bf16 梯度 `2P`
- **梯度累积不增加显存**（原地累加到同一块 buffer）。若实现上为每个 micro-batch 单独开 buffer，那是实现问题，不是必须的开销
- ZeRO-2 及以上，梯度分片：`/ N`。但 reduce-scatter 之前仍需要完整的一份或至少一个 bucket
- 梯度与参数共用同一分片度时，`M_grad` 与 `M_param` 同比例缩小

#### 1.3 优化器状态

**这是最大的一块，也是最容易低估的一块。**

| 优化器 | 每参数字节 | 说明 |
|---|---|---|
| SGD（无动量） | 0 | |
| SGD + momentum | 4 | 动量 fp32 |
| **AdamW（fp32 状态）** | **8** | `exp_avg` 4 ＋ `exp_avg_sq` 4 |
| AdamW + fp32 主权重 | **12** | 上面 8 ＋ 主权重 4。混合精度下这是默认形态 |
| 8-bit Adam | 2 | 两个状态各 1 字节，另有少量分块缩放因子 |
| Adafactor | ≈ `4(hidden + ffn)` 量级 | 二阶矩按行列因子化，远小于 `8P` |

```
AdamW 混合精度总计：M_optim = 12P
```

一个 7B 模型纯 AdamW 混合精度：`12 × 7e9 = 84 GB`。**光优化器状态就放不进一张 80 GB 的卡**——这就是为什么优化器分片几乎是大模型训练的必选项，而不是优化项。

分片与卸载：

```
优化器分片（ZeRO-1，分片数 N）： M_optim = 12P / N
优化器卸载到主机：               M_optim(device) = 0，改为占用主机内存 12P
                                 代价：每步要搬 O(P) 字节过 PCIe，且需要 pinned memory
```

#### 1.4 激活

激活是唯一随 `b` 和 `s` 变化的项，也是调参时手里最大的旋钮。

Transformer 单层、不做重计算、bf16 存储的激活量（经典估算，常数项随实现有出入）：

```
A_layer ≈ s·b·h·(34 + 5·a·s/h)   字节
          └─ 34·s·b·h ─┘ └─ 5·a·s²·b ─┘
          逐元素与线性层的中间结果    注意力分数矩阵
```

```
M_act = L × A_layer / T          （张量并行切 h 方向）
```

关键是**第二项带 `s²`**：序列长度翻倍，注意力部分的激活涨四倍。长序列场景下它会压倒一切，这也是序列并行和 FlashAttention 类算子存在的理由——后者不物化完整的注意力矩阵，直接把 `5·a·s²·b` 这一项抹掉。

重计算（gradient checkpointing）：

```
全量重计算：  M_act ≈ 2·s·b·h·L / T  ＋  单层重算时的瞬时激活 A_layer
              只存每层边界的输入，反向时现算。显存从 O(L·s·b·h·s²) 降到 O(L·s·b·h)
选择性重计算：M_act ≈ 34·s·b·h·L / T
              只重算注意力部分（那个 s² 项），保留其余。时间代价远小于全量重计算
```

**先确认用的是哪一档。** 「开了重计算还是 OOM」在全量和选择性两档之间差一个量级，不问清楚就没法估。

#### 1.5 临时缓冲与余量

```
M_workspace = 算子工作区 ＋ 通信缓冲 ＋ 分配器碎片余量
```

- **算子工作区**：GEMM 的 split-k 缓冲、注意力算子的中间量。单个不大，但并发多了会累积
- **通信缓冲**：数据并行的梯度 bucket（典型默认 25 MB × 2 组，用于计算通信重叠）；FSDP 的 all-gather / reduce-scatter 缓冲（约等于最大 unit 的参数量 × 2）。注意通信库自己的缓冲（`{{profile.collective.env_vars.buffer_size}}`）**不在框架显存池的统计里**，但一样占物理显存
- **碎片余量**：按总量的 5%~10% 留。这不是浪费，是分配器正常工作所需

#### 1.6 汇总与判定

```
M_total = M_param + M_grad + M_optim + M_act + M_workspace
预算    = {{profile.devices.models[].memory_gb}} × 1024³
```

- `M_total > 预算` → **配置本身超限**。不用采 snapshot，直接进 Step 4
- `M_total ≈ 预算`（90%~100%）→ 卡在边界上，碎片和瞬时峰值随时能把它推过去。进 Step 2 看峰值和碎片
- `M_total < 预算 × 0.8` 却还是 OOM → **估算解释不了**。这才是需要 Step 2 的场景

把估算过程和每一项的数值写出来给用户看。不要只给结论——用户比你清楚自己的实现有没有额外的副本。

### Step 2：快照深度分析（只在估算解释不了时做）

#### 2.1 采集

采集前先关掉图捕获（`{{profile.graph.graph_capture.available}}` 为真时），图捕获会把内存池固定住，快照会反映被污染的状态。

```python
# 开始记录分配历史（带调用栈）。max_entries 控制 trace 环形缓冲的大小
{{profile.pytorch.api.record_memory_history}}(max_entries=100_000)

try:
    train_a_few_steps()
except Exception:
    # 关键：在 OOM 的异常处理里导出，拿到的才是失败那一刻的状态
    {{profile.pytorch.api.dump_memory_snapshot}}("oom_at_failure.pickle")
    raise
```

两个注意点：

1. **要在 OOM 当下导出**，不是事后。OOM 之后框架通常会释放缓存重试，事后导出的快照已经不是失败现场
2. **记录调用栈很贵**。它会显著拖慢训练并吃掉额外内存，只在诊断期开，别留在生产配置里

#### 2.2 分析

```bash
python scripts/analyze_memory_snapshot.py oom_at_failure.pickle \
    --profile <你的 profile.yaml> --top-n 15
```

> **安全提示**：snapshot 是 pickle 文件，反序列化可以执行任意代码。**只分析你自己采集的或来源完全可信的文件。** 脚本默认用受限 Unpickler（只还原普通容器类型），遇到需要 import 的对象会直接拒绝并报错。

五个维度按这个顺序读：

1. **总览**：`allocated / reserved` 的比值。低于 80% 说明显存池里囤了大量用不上的碎片；`active - allocated` 这一项是「已释放但还在等跨流事件」的量，它大说明有跨流生命周期管理问题，**直接对应下面泄漏 checklist 的第一条**
2. **语义分类**：看哪个桶最大，和 Step 1 的估算逐项对照。对不上的那一项就是突破口。未分类占比超过 30% 说明默认关键字表不贴合你的代码，用 `--category-rules` 换一张
3. **峰值时刻**：峰值是多少、哪次分配把它推上去的、调用栈是什么。trace 里如果有分配失败记录，会直接给出「申请了多少、当时底层还剩多少」——这一对数字能立刻区分「真不够」和「碎片」
4. **碎片化**：**看最大空闲块，不是空闲总量。** 空闲总量 20 GiB 但最大空闲块只有 200 MiB 时，一次 1 GiB 的分配照样失败。外部碎片率高（> 80%）且报错请求量不大，基本就可以定性为碎片型 OOM
5. **Top-N 大块**：最大的那几块的调用栈。经常能直接看出某个不该这么大的中间张量

### Step 3：泄漏检测

只有时机分类落在「跑了很久才 OOM」时才做这一步。

#### 3.1 采样

在训练循环里按固定间隔导出快照。**关键是每次都在同一个阶段采**——比如都在优化器 step 之后、下一次前向之前。在不同阶段采的快照之间做差值没有意义，前向中途和优化器之后的占用本来就不一样。

```python
if step % 50 == 0 and step > 0:
    {{profile.pytorch.api.dump_memory_snapshot}}(f"snap_step{step:05d}.pickle")
```

至少采 4 份。2 份分不出「一次性峰值」和「持续泄漏」——这两者在只有两个点时长得一模一样。

#### 3.2 趋势分析

```bash
python scripts/analyze_memory_snapshot.py snap_step*.pickle --mode trend --no-trace
```

判据是**斜率为正且基本单调**，两条都要满足：

- 只看首末差值 → 会把一次性的峰值当成泄漏
- 只看斜率 → 会把剧烈抖动但整体略涨的曲线算进来
- 两条合用才能挑出「每步稳定涨一点」这种真泄漏

脚本会把在用的块按调用栈签名分组，给出每个签名随快照的增长序列。排在最前面的那条签名就是泄漏点。

还要分清两种增长：

- **allocated 在涨** → 真的有对象没释放，是引用泄漏。往下走 checklist
- **allocated 平、reserved 在涨** → 是碎片在累积，不是对象泄漏。走 Step 4 的分配器配置

#### 3.3 泄漏排查 checklist

按出现频率排序。前两条占了实际案例的绝大多数。

1. **异步通信没有 wait**
   `all_reduce(..., async_op=True)` 返回的 handle 没有 `.wait()`，通信对象会一直持有输入输出缓冲的引用。循环里每步都发一次、都不 wait，缓冲就按步数线性累积。
   **症状**：快照里通信类的占用随步数线性上涨；`active - allocated` 这一差值持续变大。
   **查法**：搜 `async_op=True`，确认每一处的 handle 都在当前迭代内被 wait 掉；用 `_coalescing_manager` 一类的批量接口时，确认退出上下文时真的同步了。

2. **全局变量 / 日志持有张量引用**
   `loss_history.append(loss)` 存的是张量不是 `loss.item()`。这一条的破坏力远超它的体积——张量带着 `grad_fn`，会把**整张计算图连同所有中间激活**一起挂住，一步就可能是几百 MiB。
   **症状**：快照里激活类或未分类的占用线性上涨，调用栈指向日志/指标模块。
   **查法**：所有进了容器、全局变量、回调、进度条、TensorBoard 的标量，都必须是 `.item()` / `.detach().cpu()` 之后的值。

3. **自定义 autograd Function 的 `save_for_backward`**
   - 用 `ctx.saved_tensor = x` 而不是 `ctx.save_for_backward(x)`：绕过了框架的生命周期管理，引用不会在反向后被清掉
   - 存了反向其实不需要的大张量：很多时候存一个形状、一个索引就够，不必存整个输入
   - 在 `forward` 里存了输出张量，又在别处持有它，形成环

4. **没有清理的 hook**
   `register_forward_hook` / `register_full_backward_hook` 返回的 handle 没有 `.remove()`。hook 闭包捕获的张量会跟着模块的生命周期一直在。诊断期临时加的 hook 尤其容易忘——**调试显存问题时加的 hook 本身造成了显存泄漏**，这种事比想象的常见。

5. 其余不那么常见但确实出现过的：
   - **异常栈帧持有张量**：`except` 块里的 traceback 对象引用了栈帧，栈帧引用了局部张量。长时间持有异常对象会把它们一起挂住
   - **无上限的缓存字典**：位置编码缓存、mask 缓存按 `(batch, seqlen)` 做 key，变长序列下 key 空间无限大
   - **`create_graph=True`**：`torch.autograd.grad(..., create_graph=True)` 会为二阶导保留整张图。不做二阶优化就不要开
   - **`record_stream` 用法不当**：把块的释放无限期推迟到某个永远不会完成的流事件上。症状同样是 `active - allocated` 持续变大

### Step 4：优化矩阵

**按「对训练结果的影响从小到大」依次尝试，不要跳着来。** 跳过前面直接减 batch 或减序列长，是在用改变训练语义的代价去换本可以免费拿到的显存。

#### 4.0 先把不改变任何计算的手段用满

这一档是纯粹的白捡，没有任何理由不先试。

| 措施 | 预期效果 | 代价 |
|---|---|---|
| 开启可扩展显存段（分配器把段按需扩展而不是按固定档位切） | 碎片型 OOM 最直接的解法，碎片率常能从 30%+ 降到 10% 以内 | 几乎没有；部分实现与图捕获不兼容 |
| 调大分配器的分段上限 / 关闭过度切分 | 减少大块被切碎 | 可能略微增加 reserved |
| 在阶段边界调用 `{{profile.pytorch.api.empty_cache}}` | 把缓存还给底层，给通信库或其他进程腾地方 | 下次分配要重新向底层申请，有延迟；**不要放在训练主循环里** |
| 检查有没有该用 `no_grad` 的地方没用（eval、指标计算、数据预处理） | 消除一整条路径的激活 | 无 |
| 检查 checkpoint 保存路径是不是把整个 state_dict 聚到了一张卡上 | 消除保存时的尖峰 | 无 |

#### 4.1 激活重计算（对训练结果零影响）

| 问题 | 措施 | 预期效果 | 代价 |
|---|---|---|---|
| 激活占比最大 | **选择性重计算**（只重算注意力部分） | 消掉 `5·a·s²·b` 那一项，长序列下往往就是大头 | 约 5%~15% 的额外计算 |
| 还是不够 | **全量重计算** | 激活从 `O(L·s·b·h·(34+5as/h))` 降到 `O(2·L·s·b·h)`，常见能降一个量级 | 约 30%~40% 的额外计算 |
| 注意力矩阵物化 | 换用不物化完整注意力矩阵的融合算子 | 直接去掉 `s²` 项 | 需要算子库支持；数值上不完全逐位一致（规约顺序不同） |

重计算是数学等价的——同样的输入出同样的梯度，只是把一部分中间结果丢掉再算一遍。它应该是第一个被考虑的真正手段。

#### 4.2 优化器状态分片 / 卸载（数学等价，但分布式规约顺序可能变）

| 问题 | 措施 | 预期效果 | 代价 |
|---|---|---|---|
| 优化器状态占比最大（AdamW 下几乎必然） | **优化器状态分片**（ZeRO-1 式，分片数 `N`） | `12P → 12P/N`。8 卡数据并行下直接省掉 87.5% | 每步多一次 all-gather 参数；通信量增加 |
| 还不够 | **梯度也分片**（ZeRO-2 式） | 再省 `4P × (1 − 1/N)` | all-reduce 换成 reduce-scatter + all-gather，总通信量相当 |
| 还不够 | **参数也分片**（ZeRO-3 / FSDP full shard） | 常驻 `P/N`，但峰值要加一个 unit 的 all-gather | 通信量显著上升，小模型上可能得不偿失 |
| 通信带宽足够、但显存实在不够 | **优化器状态卸载到主机**（需要 `{{profile.kernels.memory.host_offload_api}}`） | 设备侧优化器状态归零 | 每步 O(P) 字节过 PCIe；必须用 pinned memory，否则慢到不可用 |

这一档数学上等价，但分片会改变规约的顺序和分组，**可能带来 bit 级的数值差异**。做精度对照实验时要把这一点记进变更清单。

#### 4.3 调整并行切分（等价，但通信模式变化更大）

| 问题 | 措施 | 预期效果 | 代价 |
|---|---|---|---|
| 单卡放不下模型本体 | **提高张量并行度 `T`** | 参数、梯度、优化器状态、激活全部 `/T` | 每层两次 all-reduce，对卡间带宽（`{{profile.interconnect.intra_node.bandwidth_gb_s}}`）非常敏感，一般不跨节点 |
| 层数很多 | **提高流水并行度 `PP`** | 参数类 `/PP` | 引入流水气泡；**第一个 stage 要存最多的中间激活，是 OOM 高发点** |
| 序列特别长 | **序列/上下文并行** | 激活的 `s` 方向被切开，直接削 `s²` 项 | 注意力需要跨卡交换 KV |
| MoE 模型 | **专家并行** | 专家权重按卡分布 | all-to-all 通信；热点专家导致负载不均，个别 rank 仍可能 OOM |

#### 4.4 调整批大小（分两档，影响完全不同）

| 措施 | 是否改变训练结果 | 说明 |
|---|---|---|
| **降 micro-batch、同比例升梯度累积步数**（全局 batch 不变） | **不变** | 激活线性下降，是这一档里应该先试的。注意 loss 的归一化要跟着改，否则等效学习率会变 |
| 降低**全局** batch size | **改变** | 梯度噪声、等效学习率、收敛曲线都会变。要改就得连学习率一起重新调 |
| 缩短序列长度 | **改变** | 直接改变了模型能看到的上下文，训练目标都不一样了。**最后手段** |

把这一档排在最后是有原因的：前面所有手段都只影响「怎么算」，这一档影响的是「算什么」。拿它换显存，省下的那点工程时间会在后面的精度对齐上十倍地还回去。

#### 4.5 验证

每改一项都要重新测一次峰值，确认省下来的量和预期相符：

```python
{{profile.pytorch.api.reset_peak_memory_stats}}()
run_one_step()
peak = {{profile.pytorch.api.max_memory_allocated}}()
```

省下来的量和预期差很多，说明对这套实现的理解有偏差，回到 Step 1 重算。

偶发 OOM 难以复现时，用内存水线把它变成稳定复现：

```python
{{profile.pytorch.api.set_per_process_memory_fraction}}(0.8)
```

把进程可用显存压到 80%，原本几百步才出现的 OOM 会在前几步就稳定出现，调试循环从小时级缩到分钟级。

---

## 输出要求

给用户的结论必须包含这几项，缺一项都会让人没法往下走：

1. **分类结论**：OOM 来源 ＋ 发生时机，以及**依据是什么**（哪条报错特征、显存曲线长什么样）
2. **静态估算的逐项数值**：参数 / 梯度 / 优化器状态 / 激活 / 临时缓冲各占多少，合计多少，预算多少，超了多少。不要只给合计
3. **估算与实测的差值**（做了 Step 2 的话）。差值大的那一项就是真正的问题所在
4. **结论的确定性**：「确定是配置超限」和「怀疑是泄漏，但只有 2 份快照，证据不足」必须说清楚区别
5. **按顺序给出的措施清单**，每条都要带：预期省多少、代价是什么、**会不会改变训练结果**
6. **profile 缺口**：哪些字段为空导致哪一步降级了。这是让用户知道「补上什么能让下次诊断更准」

不要做的事：

- 不要在没有分类的情况下直接给「把 batch 调小试试」。这是把问题推回给用户
- 不要把估算值说成实测值。估算的常数项随实现变化，要讲明这是估算
- 不要在只有 1~2 份快照时给出泄漏结论
- 不要建议在训练主循环里调 `empty_cache`。它只是把问题从「OOM」变成「慢」，并且掩盖了真实的占用

---

## 脚本

| 脚本 | 职责 |
|---|---|
| `scripts/analyze_memory_snapshot.py` | 快照解析与七个维度的分析。既是 CLI 也是库：`load_snapshot()` / `analyze_snapshot()` / `compare_snapshots()` / `detect_leak_trend()` 是四个主入口 |

模式按文件数量自动选择：1 份走单快照分析，2 份额外做对比，3 份及以上做泄漏趋势。可用 `--mode` 强制。

常用参数：

```bash
--profile P.yaml          # 读单卡显存（算占用率）与集合通信库名（注入分类关键字）
--category-rules R.json   # 换一张语义分类关键字表，顺序即优先级
--device N                # 只看某张卡
--top-n 15                # Top-N 大块与增长榜的条数
--no-trace                # 跳过 trace 解析。大快照里 trace 常占九成体积
--max-trace-events 50000  # 给超大快照的 trace 设上限（保留末尾，OOM 在末尾）
--json -o report.json     # 结构化输出
```

退出码：`0` 成功、`2` 用法错误、`5` 输入文件或 profile 有问题、`7` 文件不是可识别的 snapshot。

**反序列化安全**：脚本默认用受限 Unpickler，只还原普通容器类型，任何需要 import 的对象一律拒绝。这能挡住「打开别人发来的快照就被执行代码」这条路。确需加载含自定义类型的文件时用 `--allow-arbitrary-pickle`，但那等于你为这个文件的来源背书。

---

## 知识来源

- 显存快照的结构、采集接口与分析方法：[PyTorch 官方显存分析文档](https://docs.pytorch.org/docs/stable/torch_cuda_memory.html) 与 [Understanding GPU Memory 博客系列](https://pytorch.org/blog/understanding-gpu-memory-1/)
- 缓存分配器的分段 / 块 / 碎片模型：PyTorch `CUDACachingAllocator` 的公开设计文档与 [Memory management 文档](https://docs.pytorch.org/docs/stable/notes/cuda.html#memory-management)
- 激活显存的估算公式：Korthikanti 等，[*Reducing Activation Recomputation in Large Transformer Models*](https://arxiv.org/abs/2205.05198)（选择性重计算的来源）
- 优化器状态分片的分级（状态 / 梯度 / 参数三档）：Rajbhandari 等，[*ZeRO: Memory Optimizations Toward Training Trillion Parameter Models*](https://arxiv.org/abs/1910.02054)
- 不物化注意力矩阵的算子思路：Dao 等，[*FlashAttention*](https://arxiv.org/abs/2205.14135)
- 并行切分对显存的影响：[Megatron-LM](https://github.com/NVIDIA/Megatron-LM) 论文系列与 PyTorch FSDP 文档
- pickle 反序列化风险：[Python 官方 pickle 文档的安全警告](https://docs.python.org/3/library/pickle.html)
- 本文件及配套脚本为独立编写，不含任何来自第三方受限许可仓库的代码或文本，依据见 [`docs/LICENSE-CONSTRAINT.md`](../../docs/LICENSE-CONSTRAINT.md)
