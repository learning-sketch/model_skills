---
name: model-infer-runtime-debug
description: 诊断 LLM 推理在加速卡上的运行时故障——进程报错退出、设备侧非法内存访问、显存不足、集合通信超时、以及完全没有报错的卡死。核心方法是利用同步检查点做二分定位，把"异步执行下报错的位置"还原成"真正出错的位置"，再按根因选择回退、修参、统一路径或降级配置。适用于已经能跑起来但跑挂了的场景。不覆盖：结果能跑出来但数值不对（属精度诊断）、能跑完但跑得慢（属性能分析）、模型首次接入框架时的结构适配（属模型迁移）、训练侧的梯度与优化器问题。
---

# 推理运行时错误诊断

把一次"推理跑挂了"收敛为一条可复现的最小用例、一个确定的根因、一个带验证的修复。

## 硬件依赖

本技能从 `hardware-profiles/` 下的 active profile 读取硬件信息。若字段为 `null` 或所在大节的 `_status` 为 `unknown` / `assumed`，**先提示用户确认再执行**，不要按 CUDA 语义自行推定。

| profile 字段 | 用途 | 缺失时的降级行为 |
|---|---|---|
| `pytorch.device_type` | 构造设备字符串、判断张量所在设备 | 阻塞。这是最低要求，让用户补齐 |
| `pytorch.device_namespace` | 所有设备侧 API 调用的前缀 | 阻塞 |
| `pytorch.extension_module` | 执行前是否需要 import 设备扩展包 | 按 `null` 处理（设备为框架原生后端），并在报告中标注该假设 |
| `pytorch.api.synchronize` | 二分定位的同步检查点，本技能的核心手段 | 整个 Step 4 失效。退化为只能看原始报错栈，需提示用户这会大幅降低定位精度 |
| `pytorch.api.device_count` / `current_device` / `set_device` | 多卡场景核对 rank 与物理卡的绑定 | 多卡分支跳过绑定核对，只做日志比对 |
| `pytorch.api.memory_allocated` / `max_memory_allocated` | 显存水位采样，判定是否 OOM 类根因 | 改用 `diagnostics.smi_command` 从外部观测，精度变粗（看不到框架内已分配 vs 缓存） |
| `pytorch.api.empty_cache` | 释放缓存块，区分"真的不够"与"碎片化" | 跳过碎片化判定，OOM 结论标注为未区分 |
| `diagnostics.error_signatures.*` | 错误归类表的唯一数据源 | 归类表失效，改为让用户粘贴完整报错文本，由人工判断归属哪一类症状 |
| `diagnostics.smi_command` / `smi_query_memory` | 查设备占用、残留进程、设备健康 | 跳过"残留进程占显存"这一常见根因的排查，需在报告中声明该项未排除 |
| `diagnostics.sanitizer.available` / `command` / `tools` | 非法内存访问的精确定位 | 非法访问类只能靠二分缩小到算子级，无法定位到具体越界地址 |
| `diagnostics.oom_sources.*` | 区分 OOM 来自缓存分配器 / 临时工作区 / 通信缓冲 / 主机内存 | 全部按缓存分配器处理，可能给出错误的处置方向 |
| `collective.torch_backend` | 初始化进程组的后端名 | 多卡分支阻塞 |
| `collective.env_vars.debug_level` / `connect_timeout` / `exec_timeout` / `interface_select` | 打开通信库日志、调整超时、排查网卡选错 | 通信类问题只能从框架侧 watchdog 日志判断，看不到通信库内部状态 |
| `graph.modes` | 判断当前是否跑在图模式、能否切回 eager 做分流 | 假定只有 eager，跳过 Step 2 的图模式分流 |
| `devices.models[].memory_gb` | 判断显存需求是否从一开始就不可能满足 | 跳过容量可行性预检 |
| `inference_framework.name` | 定位框架侧的日志位置与配置开关 | 按裸 PyTorch 脚本处理 |

**profile 未覆盖但本技能需要的能力**（见文末「已知的 profile 缺口」）：同步下发开关、设备侧断言开关、残留进程查询命令、算子输入约束表。这四项目前只能让用户口头确认。

## 适用场景

- 推理进程抛出设备侧错误并退出（非法内存访问、显存不足、kernel 下发失败）
- 推理进程无任何输出地卡住，设备利用率为 0 或恒定不降
- 多卡推理中部分 rank 报集合通信超时，但看不出是谁先出的问题
- 单卡能跑、扩到多卡就挂；或小 batch 能跑、放大 batch 就挂
- 换了模型权重 / 换了 tokenizer / 改了并行配置之后开始挂

## 不适用场景

| 现象 | 应转向 |
|---|---|
| 能跑完，但输出乱码或与标杆模型不一致 | 推理精度诊断 |
| 能跑完，但吞吐/时延不达标 | 性能拆解分析 |
| 报错明确指向图编译失败或图捕获约束违反 | `model-infer-graph-mode` |
| 报错只在开启多流优化后出现 | `model-infer-multi-stream` 的五类问题分诊 |
| 模型第一次接入推理框架，结构尚未跑通 | 模型迁移接入 |
| 训练中的 NaN / 梯度爆炸 / 优化器状态异常 | 训练精度系列 |

## 工作流

### Step 0 — profile 门禁

**输入**：active profile 路径。
**动作**：读取并检查上表中标为"阻塞"的字段，以及 `pytorch._status` 与 `diagnostics._status`。
**决策路由**：

| 检查结果 | 动作 |
|---|---|
| 阻塞字段为 `null` | 停止。列出缺失字段，要求用户补齐后再来 |
| `_status` 为 `unknown` 或 `assumed` | 继续，但在每一处引用该节字段的地方标注「该值未经真机验证」，并在最终报告的「残留风险」里汇总 |
| 相关能力登记在 `unknown_or_unsupported` 中 | 该条排查路径直接标记为不可用，不要给出基于它的建议 |

### Step 1 — 固定现场

**输入**：用户的故障描述。
**动作**：按清单收齐，缺一项就明确问一次。不要在信息不全的情况下开始猜。

- [ ] 完整报错文本（从第一行到最后一行，含所有 rank 的，不要只给最后几行）
- [ ] 可复现的完整命令行与配置文件
- [ ] 并行拓扑：总卡数、张量并行度、流水并行度、专家并行度、数据并行度
- [ ] 执行模式：eager 还是图模式；若图模式，是编译路径还是图捕获路径
- [ ] 复现稳定性：必现 / 概率性（给出概率）/ 只在长跑后出现
- [ ] 变更点：上一次正常运行与这次之间，改了什么（权重、tokenizer、框架版本、驱动、配置）
- [ ] 输入特征：是否与特定请求相关；最长输入长度、batch 大小

**决策路由**：若"概率性"且与特定输入相关 → 优先抓那一条输入做必现用例，后续所有步骤都在这条用例上做。若"长跑后出现" → 优先怀疑显存碎片化、KV cache 泄漏、通信缓冲累积，把显存水位采样加进 Step 4。

### Step 2 — 两路分流

**输入**：Step 1 的报错文本。
**决策路由**：

```
有错误文本？
├─ 有 ──> Step 3A 错误归类  ──> Step 4 二分定位 ──> Step 6
└─ 无（卡死/静默退出）──> Step 3B 卡死诊断 ──> Step 4 二分定位 ──> Step 6
```

另有一条正交的分流，两路都要先做：

| 当前执行模式 | 动作 |
|---|---|
| 图模式（见 `profile.graph.modes`） | **先关图模式用 eager 复现一次**。eager 下不复现 → 问题在图捕获/编译路径，转 `model-infer-graph-mode`，本技能到此为止。eager 下同样复现 → 继续本技能，并在后续全程用 eager，因为图模式会打乱同步检查点的粒度 |
| eager | 直接继续 |

### Step 3A — 错误归类

**输入**：报错文本 + `profile.diagnostics.error_signatures`。
**动作**：把报错文本与 profile 中登记的特征串逐条匹配，归入下表的症状类。**不要硬编码任何厂商的错误码或错误串**；表的右列只写从哪个 profile 字段取特征。

| 症状类 | 特征串来源 | 典型根因（按先验概率排序） | 下一步 |
|---|---|---|---|
| 显存不足 | `diagnostics.error_signatures.out_of_memory`，再用 `diagnostics.oom_sources.*` 细分来源 | 1. 批量/序列长度超预算 2. KV cache 预留过大 3. 碎片化 4. 残留进程占卡 5. 通信缓冲未计入预算 | Step 5 的 OOM 支线 |
| 非法内存访问 | `diagnostics.error_signatures.illegal_memory_access` | 1. 嵌入层索引越界（词表未对齐）2. 索引张量含负值或越界 3. 算子输入 shape 超出约束 4. 跨流张量生命周期问题 | Step 4 二分到算子，再用 `diagnostics.sanitizer` 定位 |
| 设备超时 | `diagnostics.error_signatures.device_timeout` | 1. 单个 kernel 执行过久 2. 设备已挂起，后续下发全部超时 3. 看门狗阈值设得过紧 | 先看设备健康（Step 5 支线），再 Step 4 |
| 集合通信超时 | `diagnostics.error_signatures.collective_timeout` | 1. 某个 rank 已经失败但没能让其他 rank 感知 2. 各 rank 进入集合通信的顺序不一致 3. 各 rank 的张量 shape/dtype 不一致 4. 网卡选错导致带宽极低 | Step 5 的多卡支线 |
| kernel 下发失败 | `diagnostics.error_signatures.kernel_launch_failure` | 1. 下发配置超出硬件上限 2. 设备上下文已损坏（往往是前一个错误的次生现象）3. 驱动/运行时版本不匹配 | 往回找**第一个**异常，当前错误很可能是次生的 |
| 编译失败 | `diagnostics.error_signatures.compile_error` | 转 `model-infer-graph-mode` | — |
| 无匹配 | — | profile 的特征串不全 | 让用户确认该报错属于哪一类，并建议把新特征串补进 profile |

> **以 CUDA 为例**：`cuda-reference.yaml` 里 `illegal_memory_access` 登记的是 `"CUDA error: an illegal memory access was encountered"` 和 `"device-side assert triggered"` 这类串。你的芯片上对应的串必然不同，一律从 profile 取，不要套用这里的例子。

**关键纪律**：异步执行下，报错文本指向的调用栈**不是**出错的位置。设备侧错误在后续某次同步时才被主机捕获，栈顶往往停在一个无辜的同步点上。永远不要直接相信第一个栈。

### Step 3B — 卡死诊断

**输入**：卡死的进程。
**动作**：三路观测同时做，再查表。

1. 设备侧：用 `{{profile.diagnostics.smi_command}}` 连续采样若干次，记录利用率与显存占用是否变化
2. 主机侧 CPU：`top` / `pidstat` 看进程 CPU 占用
3. 主机侧调用栈：抓 Python 栈。进程外抓用 `py-spy dump --pid <PID>`；进程内可提前埋 `faulthandler.dump_traceback_later(<秒数>, repeat=True)`。两者都是框架无关的通用工具

**决策路由**：

| 设备利用率 | 主机 CPU | 判定 | 下一步 |
|---|---|---|---|
| ~0% | 接近单核满载 | 主机侧 CPU 密集，不是死锁 | 看 Python 栈停在哪：分词、采样后处理、图编译、权重加载、Python 侧循环。这是性能问题不是故障，转性能分析 |
| ~0% | 很低 | 真阻塞 | 看 Python 栈：停在集合通信 → 多卡支线；停在 event/stream 等待 → 多流依赖环；停在锁/队列 → 框架调度层 |
| 持续高且不变 | 很低 | 设备侧 kernel 跑不完 | 可能是死循环 kernel 或超长 kernel。用 Step 4 定位到是哪个算子，检查其输入是否异常（例如序列长度为 0 或极大值） |
| 波动但无进展 | 中等 | 在做无效重试 | 查通信库日志（`{{profile.collective.env_vars.debug_level}}`）与框架重试逻辑 |

**多卡卡死的额外动作**：对每个 rank 都抓一次 Python 栈，然后比对。绝大多数 rank 停在同一个集合通信调用上、只有一两个 rank 停在别处——**停在别处的那个就是源头**。

### Step 4 — 同步检查点二分定位

本技能的核心。目的是把异步执行拉平成同步执行，让错误在真正出错的地方抛出。

**动作**：先尝试全局同步化。若 profile 登记了设备运行时的同步下发开关（CUDA 以环境变量 `CUDA_LAUNCH_BLOCKING=1` 为例），打开它再跑一次，报错栈通常会直接指向真正的算子。**profile 当前没有这个字段，必须向用户确认你的运行时是否提供等价开关**；没有就用下面的手工同步检查点。

```python
# 手工同步检查点。CUDA 参考值见注释。
import torch
# {{profile.pytorch.extension_module}} 非 null 时需要先 import，CUDA 为 null 不需要
# import {{profile.pytorch.extension_module}}

_sync = {{profile.pytorch.api.synchronize}}          # CUDA 参考: torch.cuda.synchronize
_mem  = {{profile.pytorch.api.memory_allocated}}     # CUDA 参考: torch.cuda.memory_allocated

def checkpoint(tag: str):
    """在此处排空异步队列。若此前有设备侧错误，异常会在这一行抛出。"""
    _sync()
    print(f"[ckpt-ok] {tag} mem={_mem() / 1024**3:.2f}GiB", flush=True)
```

**四级下钻顺序**，每一级定位到后再进入下一级，不要一上来就逐算子插桩：

| 级别 | 切分粒度 | 插桩位置 | 产出 |
|---|---|---|---|
| 1 阶段 | 权重加载 / 预热 / Prefill / Decode / 采样 / 后处理 | 各阶段首尾 | 哪个阶段最后一个 `ckpt-ok` 之后挂的 |
| 2 层 | 第 0..N-1 个 decoder layer | 每层入口 | 哪一层。若在第 0 层挂 → 怀疑输入侧（嵌入、位置编码）；若在最后一层后挂 → 怀疑输出头、采样 |
| 3 模块 | 注意力 / 前馈 / 归一化 / 路由 / 通信 | 模块入口 | 哪个模块。若各层表现一致 → 结构性问题；若只有某一层 → 该层配置或权重特殊 |
| 4 算子 | 模块内逐个调用 | 调用前后 | 具体算子。记录它的全部输入：shape、dtype、device、数值范围（min/max，索引类张量必查） |

**加速技巧**：层级用二分而非顺序扫描——先只在第 N/2 层插桩，据结果决定往前还是往后，`log2(N)` 次即可定位。顺序插桩在 80 层模型上会慢到不可接受。

**决策路由**：定位到算子后，把该算子的输入特征与 profile 的算子约束比对（`profile.kernels.*`）。**profile 目前没有登记各算子的输入约束**（支持的 dtype 组合、head_dim 取值集合、最大序列长度、对齐要求），需要让用户提供你的算子库文档或直接确认。

### Step 5 — 分症状支线

#### OOM 支线

1. 用 `{{profile.diagnostics.smi_command}}` 确认**开跑前**卡上就是干净的。残留进程占显存是高频根因，尤其在上一次异常退出之后。profile 没有登记查询占卡进程的命令（CUDA 以 `nvidia-smi --query-compute-apps=pid,used_memory --format=csv` 为例），需用户确认等价命令
2. 用 `profile.diagnostics.oom_sources.*` 区分来源：缓存分配器 / 临时工作区 / 通信缓冲 / 主机内存。四者的处置方向完全不同——缓存分配器 OOM 调批量与 KV 预留，通信缓冲 OOM 调 `{{profile.collective.env_vars.buffer_size}}`，主机内存 OOM 与设备显存无关
3. 容量可行性预检：用 `profile.devices.models[].memory_gb` 对照「权重 + KV cache + 激活 + 通信缓冲 + 框架开销」的粗估。若粗估本就超出单卡容量，不是故障而是配置不可行，直接给扩并行度或降批量的结论
4. 碎片化判定：在 OOM 前一刻调 `{{profile.pytorch.api.empty_cache}}`（CUDA 参考: `torch.cuda.empty_cache`）后重试。能过 → 碎片化；仍不过 → 真的不够
5. 深入分析转 `model-train-oom-analysis`（显存快照方法在推理侧同样适用）

#### 嵌入越界支线（非法内存访问中最常见的一类）

按顺序核对四个数：tokenizer 的词表大小、模型配置里的 `vocab_size`、嵌入权重张量的第 0 维、以及实际输入 token id 的最大值。任意两者不一致都会让嵌入查表越界，表现为设备侧非法内存访问而非一个友好的 Python 异常。

快速验证：在嵌入层之前插一行 `assert int(input_ids.max()) < embedding.weight.shape[0]`（这是主机同步，仅用于排查，定位完要去掉）。高频诱因：扩了词表的权重配了原始 tokenizer、加了特殊 token 但没 resize 嵌入、量化导出时截断了词表。

#### 多卡支线

| 顺序 | 动作 | 判读 |
|---|---|---|
| 1 | **先降维到单卡复现**：并行度全设为 1，必要时裁层或换小模型 | 单卡复现 → 与分布式无关，回 Step 4 在单卡上定位，成本低一个数量级。单卡不复现 → 确为分布式问题，继续 |
| 2 | 收齐**所有** rank 的日志，按时间戳归并排序 | 找最早的异常。不要只看 rank 0——失败的往往是别的 rank，rank 0 只是后来卡在集合通信上 |
| 3 | 在每个集合通信调用前插进度标记 + `torch.distributed.barrier()` | 比对各 rank 打印出的最后一个标记。停在不同标记上 = 进度不一致，落后的那个 rank 是源头 |
| 4 | 核对各 rank 传入集合通信的张量 shape、dtype、以及调用顺序 | 任一不一致都会导致挂起或超时。MoE 的 all-to-all 尤其容易因各 rank token 分布不同而 shape 不一致 |
| 5 | 打开通信库日志 `{{profile.collective.env_vars.debug_level}}`，必要时放宽 `{{profile.collective.env_vars.exec_timeout}}` | 放宽超时只是为了拿到更多信息，**不是修复**。若放宽后能跑完，说明是慢不是死，转性能分析 |
| 6 | 检查网卡/链路选择 `{{profile.collective.env_vars.interface_select}}` | 选到管理网口会让带宽掉一两个数量级，表现为间歇性超时 |

**警惕部分 rank 静默失败**：某个 rank 的工作进程崩溃或被 OOM killer 杀掉后，其余 rank 会一直等在集合通信里直到超时。此时所有可见的报错都是"通信超时"，真正的根因在那个已经消失的 rank 上。核对办法：比对实际存活进程数与预期 rank 数；检查内核日志里是否有 OOM killer 记录。

#### 设备健康支线

当多个不相关的任务在同一张卡上都挂，或报错指向"设备上下文损坏"时，用 `{{profile.diagnostics.smi_command}}` 检查设备是否掉卡、是否报硬件错误计数。确认是硬件问题就换卡复测，不要继续在软件层面定位。

### Step 6 — 根因确认

**判定标准**：不满足全部三条就不算定位到根因，回 Step 4 继续。

1. 能写出一个**最小复现**：去掉无关层、无关请求后仍必现
2. 能**解释**为什么这个输入会触发（而不只是"换个输入就好了"）
3. 能**预测**：说出一个改动，并准确预言它会让问题消失或仍然存在，然后验证成功

### Step 7 — 选择修复模式

| 修复模式 | 适用根因 | 做法 | 代价 |
|---|---|---|---|
| **算子回退** | 融合算子在特定 shape/dtype 下有缺陷或约束不满足 | 用等价的朴素 PyTorch 组合替换该算子，加 enable 开关可切回 | 性能下降，需量化评估 |
| **参数修正** | 输入违反算子约束（对齐、上限、dtype 组合） | 调整 shape（补齐到对齐边界）、改 dtype、分块调用绕开上限 | 低。首选 |
| **路径统一** | 多 rank 之间配置/权重/执行顺序不一致；或同一逻辑有两条实现路径而只有一条正确 | 统一到单一路径，加启动时一致性校验（各 rank 广播配置哈希后比对） | 低，且能防复发 |
| **配置降级** | 资源不足或能力不支持 | 降批量、降序列长度上限、关图模式、关多流、降并行度 | 性能下降。**只作为临时缓解**，必须同时记录待办 |

**纪律**：修复必须作用在根因上。"重试一次就好了"、"sleep 一下就过了"、"把超时调大"都不是修复，是掩盖。若只能临时缓解，在报告里明确写成缓解措施并保留根因待办。

### Step 8 — 回归验证

- [ ] 原最小复现用例通过
- [ ] 原始完整用例通过
- [ ] 概率性问题：连续跑足够多次（至少让原故障概率下的期望发生次数 > 5 次）无复现
- [ ] 输出正确性未被破坏：与修复前的正常用例做输出比对
- [ ] 修复若涉及算子回退或配置降级，给出性能变化数字
- [ ] 所有排查用的临时插桩（同步检查点、`assert`、同步下发开关）全部移除，并重测一次——同步检查点本身会掩盖异步问题

## 输出要求

完成后必须汇报以下七项，缺项要写明为什么缺：

1. **现象**：一句话描述 + 原始报错的关键行（不要全文粘贴）
2. **归类**：落在 Step 3A 的哪个症状类，依据是 profile 的哪个 `error_signatures` 字段匹配
3. **定位路径**：二分走到了哪一级，每级的结论。给出最终定位到的阶段/层/模块/算子
4. **根因**：一句话因果链，必须满足 Step 6 的三条判定
5. **证据**：支撑根因的具体数据（算子输入 shape/dtype/数值范围、显存水位、各 rank 进度标记、设备状态）
6. **修复**：采用了哪种修复模式、改了什么、代价多大。若为临时缓解须明确标注
7. **残留风险**：未排除的可能性、因 profile 字段缺失而跳过的检查项、`_status` 为 `assumed` 的字段清单

## 已知的 profile 缺口

`_TEMPLATE.yaml` 当前没有以下字段，而本技能的对应步骤需要它们。遇到时**向用户确认，不要臆造**，并把确认结果建议补进 profile。

| 建议字段路径 | 本技能哪一步要用 | 没有时的替代做法 |
|---|---|---|
| `diagnostics.launch_blocking_env` | Step 4 的全局同步化。以 CUDA 为例是 `CUDA_LAUNCH_BLOCKING=1` 这类让下发变同步的环境变量 | 退化为手工插同步检查点，定位慢一个数量级 |
| `diagnostics.device_assert_enable` | 非法内存访问支线。设备侧断言的开关，开启后越界能给出可读信息而非笼统的内存错误 | 只能靠二分 + 手工 `assert` 复现 |
| `diagnostics.smi_query_processes` | Step 5 的残留进程排查。查询占用设备的进程及其显存 | 让用户自己用设备管理工具看，该项在报告中标为未排除 |
| `diagnostics.device_health_query` | 设备健康支线。查掉卡、硬件错误计数 | 无法区分软件故障与硬件故障 |
| `kernels.*.constraints` | Step 4 第 4 级。各算子的输入约束：dtype 组合、head_dim 取值集合、序列长度上限、对齐要求 | 定位到算子后无法判断输入是否越界，只能让用户查算子文档 |
| `collective.default_timeout_seconds` | 多卡支线第 5 步。集合通信的默认超时值 | 无法判断"超时阈值设得过紧"这一根因 |

## 知识来源

本技能的方法论全部来自公开技术资料与通用工程实践，不含任何厂商专有信息：

- **PyTorch 官方文档**：CUDA semantics（异步执行与同步点语义）、`torch.distributed` 文档（集合通信一致性要求、超时与 watchdog 机制）、显存管理与缓存分配器文档、`torch.distributed` 调试环境变量说明
- **NVIDIA 公开文档**：CUDA Programming Guide 关于错误报告为异步、错误状态黏滞（sticky error）的描述；CUDA Runtime API 错误处理章节；Compute Sanitizer 用户手册的工具分类
- **分布式系统通用方法**：故障源头定位中的"最早异常原则"、屏障同步做进度核对、降维复现（先单机后分布式）
- **开源推理框架的公开实践**：vLLM、SGLang 等项目 issue 中反复出现的故障模式（词表不对齐导致嵌入越界、残留进程占显存、各 rank shape 不一致导致 all-to-all 挂起）
- **通用调试工具的公开文档**：`py-spy`（进程外 Python 栈采样）、Python 标准库 `faulthandler`
- **二分查找定位法**：计算机科学基础方法，无特定来源
