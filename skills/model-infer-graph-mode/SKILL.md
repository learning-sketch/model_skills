---
name: model-infer-graph-mode
description: 把 LLM 推理从逐算子下发（eager）改造成图模式执行，消除主机侧下发开销。覆盖上图范围选择（为什么通常只把 Decode 上图而 Prefill 保持 eager）、静态化改造清单（KV cache 预分配原地更新、消除主机同步、消除依赖张量值的控制流、固定输入输出地址、shape 分桶）、Graph Break 与重编译的定位修复、用最简后端做前后端定界、以及编译通过/输出一致/性能收益三段验证。适用于已经能在 eager 下正确跑完、现在要降低单步时延的场景。不覆盖：eager 下就跑不通的故障、图模式无关的算子级性能优化、训练侧的编译。
---

# 图模式适配

把推理的热路径从"每个算子都由主机下发一次"改成"一次下发整张图"，收益来自消除下发开销，不来自算子本身变快。

## 硬件依赖

本技能从 `hardware-profiles/` 下的 active profile 读取硬件信息。若字段为 `null` 或所在大节的 `_status` 为 `unknown` / `assumed`，**先提示用户确认再执行**，不要按 CUDA 语义自行推定。

| profile 字段 | 用途 | 缺失时的降级行为 |
|---|---|---|
| `graph.modes` | 判断可走哪条落地路径（编译 / 图捕获 / 都不支持） | 阻塞。这是本技能成立的前提，必须让用户确认 |
| `graph.torch_compile_supported` | 能否走编译路径 | 按 `false` 处理，只走图捕获路径 |
| `graph.torch_compile_backend` | 编译路径的后端名 | 若 `torch_compile_supported` 为真却缺后端名，阻塞并询问 |
| `graph.graph_capture.available` | 能否走手工图捕获路径 | 按 `false` 处理，只走编译路径 |
| `graph.graph_capture.api` | 图捕获的入口类/上下文管理器 | 图捕获路径阻塞。**绝不猜 API 名** |
| `graph.graph_capture.torch_compile_mode` | 用编译器自动接管图捕获时传的 mode 字符串 | 不使用自动模式，改为手工捕获 |
| `graph.graph_capture.constraints.static_shape_required` | 决定是否必须做 shape 分桶 | 保守按 `true` 处理（做分桶），并标注该假设 |
| `graph.graph_capture.constraints.static_address_required` | 决定是否必须用静态输入输出缓冲区 | 保守按 `true` 处理 |
| `graph.graph_capture.constraints.host_sync_forbidden` | 决定静态化清单第 2 项是否为硬约束 | 保守按 `true` 处理 |
| `graph.graph_capture.constraints.dynamic_control_flow_forbidden` | 决定静态化清单第 3 项是否为硬约束 | 保守按 `true` 处理 |
| `graph.graph_capture.constraints.collective_ops_supported` | 多卡下集合通信能否进图，决定上图边界要不要在通信处切开 | 按"不支持"处理，把通信点作为图的切分边界，并标注该假设 |
| `graph.kernel_fusion_codegen.available` | 是否还有"子图编成单 kernel"这条更激进的路径 | 不提这条路径 |
| `pytorch.device_namespace` / `api.synchronize` | 计时与同步 | 阻塞 |
| `pytorch.api.stream_class` | 部分图捕获实现要求在非默认流上捕获 | 若图捕获路径需要而该字段为 `null`，向用户确认捕获所需的流上下文 |
| `pytorch.uses_privateuse1_backend` | 为真时，编译器后端与若干框架原生特性的可用性需逐项确认，不能假定与原生后端等同 | 为 `null` 时按需确认，不默认为 `false` |
| `pytorch.api.memory_allocated` / `max_memory_allocated` | 衡量静态缓冲区与多桶图带来的显存增量 | 显存代价只能估算，不能实测 |
| `kernels.attention.paged_attention` / `kv_cache_layout` | KV cache 原地更新的写入方式与布局 | 需用户说明 KV cache 的实际组织方式，否则无法判断能否原地更新 |
| `profiler.api` / `profiler.output.trace_file` | 第三段验证（性能收益）与下发间隙度量 | 只能用端到端墙钟时间验证，无法区分"收益来自消除下发"还是别的因素 |
| `profiler.kernel_table_columns.duration_us` / `start_time_us` | 计算下发间隙占比 | 跳过 Step 1 的收益预判，直接试做 |
| `diagnostics.error_signatures.compile_error` | 识别编译失败的报错特征 | 让用户粘贴完整报错，人工判断 |
| `collective.torch_backend` | 多卡场景下通信入图的后端确认 | 多卡分支需用户确认 |
| `inference_framework.name` | 分桶配置与开关挂在框架的哪一层 | 按裸 PyTorch 脚本处理 |

**profile 未覆盖但本技能需要的能力**（见文末「已知的 profile 缺口」）：图捕获的内存池要求、可捕获图的数量/规模上限、捕获期多流支持、动态 shape 编译支持程度、可用的调试后端名清单、编译缓存位置、重编译次数上限配置项、捕获前的预热次数要求。这些目前只能让用户确认。

## 适用场景

- 单步 Decode 时延中，主机下发与间隙占比高（Step 1 给出量化判据）
- 小 batch、小模型、高并发短请求——这类负载的 kernel 普遍很小，下发开销占比最高
- 已经在 eager 下跑通且结果正确，现在要压时延
- 已经开了图模式但收益不及预期，或频繁触发重编译

## 不适用场景

| 现象 | 应转向 |
|---|---|
| eager 下就跑不通 | `model-infer-runtime-debug`。**先把 eager 跑通，图模式不会修复正确性问题，只会让它更难定位** |
| 瓶颈在单个大算子的计算本身（例如长序列 Prefill 的注意力） | 算子级优化 / 性能拆解。上图对此无效 |
| 瓶颈在集合通信带宽 | 并行策略分析 / `model-infer-multi-stream` 的通信计算重叠 |
| `profile.graph.modes` 只有 `eager` | 本技能不成立。先确认你的栈是否真的没有图能力 |
| 需要在图里做依赖输出内容的提前终止、动态束搜索 | 这类逻辑本质动态，应留在图外的主机侧 |

## 工作流

### Step 0 — profile 门禁与路径选择

**输入**：active profile。
**动作**：读 `graph.modes`、`graph.torch_compile_supported`、`graph.graph_capture.available`。
**决策路由**：

| profile 状态 | 落地路径 | 说明 |
|---|---|---|
| 支持编译且支持图捕获 | **优先编译路径**，由编译器顺带接管图捕获（传 `{{profile.graph.graph_capture.torch_compile_mode}}`，CUDA 参考值 `"reduce-overhead"`） | 改造量最小，Graph Break 有现成工具定位 |
| 只支持编译 | 编译路径 | 能消除 Python 解释开销与部分下发，但不能把整步合成一次下发，收益低于图捕获 |
| 只支持图捕获 | 手工图捕获路径 | 改造量大，静态化清单必须逐条做到 |
| 都不支持 / `_status: unknown` | 停止并询问 | 不要猜 API |

若 `graph._status` 为 `assumed`，先让用户在真机上确认 `graph.modes` 与 `graph_capture.constraints` 的真实取值，再继续；整套静态化改造的工作量完全由这几个约束决定，猜错会白做。

### Step 1 — 确定上图范围

**结论先行**：LLM 推理的常规做法是**只把 Decode 上图，Prefill 保持 eager**。理由是两个阶段的特征正好相反：

| 维度 | Decode（每步生成 1 个 token） | Prefill（一次处理整段 prompt） |
|---|---|---|
| 单步算子数量 | 多（每层十几到几十个 kernel × 几十层，单步常达数千次下发） | 同样多 |
| 单个算子规模 | 很小。矩阵乘的 M 维只有 `batch`，注意力只算 1 个 query 位置 | 很大。M 维是 `batch × 序列长度` |
| 单个 kernel 执行时间 | 常在微秒量级，与一次主机下发的开销同量级甚至更短 | 远大于下发开销 |
| 下发开销占比 | **高**，这是上图的全部收益来源 | 低，上图收益接近零 |
| shape 稳定性 | 高。batch 固定时每步 shape 不变，只有 KV 长度随步数增长，可用分块对齐吸收 | **低**。prompt 长度任意，分桶会爆炸或 padding 浪费严重 |
| 上图性价比 | 高 | 低 |

**量化判据**（需要 profiler；无 profiler 时跳过本判据直接试做）：
取一个稳态 Decode 步，令 `kernel_busy_us` = 该步内设备侧所有 kernel 执行时间的并集长度（用 `profile.profiler.kernel_table_columns.start_time_us` 与 `duration_us` 求区间并集，不要直接求和，否则多流下会重复计数），`wall_us` = 该步端到端墙钟时间。定义

```
gap_ratio = 1 - kernel_busy_us / wall_us
```

| `gap_ratio` | 判读 | 动作 |
|---|---|---|
| > 40% | 设备大半时间在等下发 | 强烈建议上图，预期收益接近 `gap_ratio` |
| 20% ~ 40% | 下发开销显著 | 建议上图 |
| 10% ~ 20% | 收益有限 | 先做更便宜的优化（算子融合、减少主机同步），再评估 |
| < 10% | 设备基本跑满 | **不要上图**。瓶颈在计算或访存，转性能拆解 |

**上图边界的切分点**：

- 采样/后处理若含依赖输出内容的分支（提前终止、重复惩罚的动态逻辑），留在图外
- 若 `profile.graph.graph_capture.constraints.collective_ops_supported` 为 `false` 或 `null`，在每个集合通信处切开，图退化为多段；此时要重新评估收益是否还值得
- 权重加载、KV cache 分配一定在图外（它们只执行一次）

### Step 2 — 静态化改造清单

逐条检查，每条给出"是否满足 / 不满足时怎么改"。前五条是硬骨头。

| # | 检查项 | 不满足的典型写法 | 改法 | 对应 profile 约束 |
|---|---|---|---|---|
| 1 | **KV cache 预分配并原地更新** | 每步 `torch.cat([past_kv, new_kv], dim=seq)` 产生新张量 | 按最大序列长度一次性分配 KV 缓冲区，每步用索引写入（`index_copy_` / `scatter_` / 切片赋值）原地更新 | `static_address_required` |
| 2 | **消除主机同步** | `.item()`、`.cpu()`、`.tolist()`、`.numpy()`、`int(t)`、`bool(t)`、`print(t)`、`assert t > 0`、基于张量值的 `if` | 把需要的标量改成设备上的张量参与运算；判断逻辑用 `torch.where` / 掩码表达；日志与断言移出热路径 | `host_sync_forbidden` |
| 3 | **消除依赖张量值的 Python 控制流** | `if seq_len_tensor > threshold:`、`while not finished.all():`、按 token 内容走不同分支 | 两条分支都算再用掩码选（代价可接受时）；或把分支判断上提到图外，按分支编译多张图 | `dynamic_control_flow_forbidden` |
| 4 | **固定输入输出地址** | 每步新建输入张量传进去；输出直接返回图内张量 | 预分配静态输入缓冲区，每步用 `copy_` 把新数据拷进去；输出从静态缓冲区读，若调用方要持有需 `clone()`（图下一次重放会覆盖该缓冲区） | `static_address_required` |
| 5 | **shape 分桶** | batch 随请求数任意变化 | 选一组桶（如 1/2/4/8/16/32），实际 batch 向上取整到最近的桶并 padding；KV 长度按块大小对齐。每个桶一张图 | `static_shape_required` |
| 6 | **图内不触发新的显存分配** | 第一次执行时才惰性分配中间缓冲 | 正式捕获前做若干次预热执行，让所有缓冲都已分配。**预热次数要求 profile 未登记，向用户确认** | 捕获期分配器行为 |
| 7 | **随机性可捕获** | 采样里的随机数生成器状态在主机侧推进 | 优先把采样整体移出图（最稳妥）。若必须进图，确认你的运行时是否提供可捕获的随机数生成器状态管理 | profile 未登记，需确认 |
| 8 | **集合通信可进图** | 多卡下图内含 all-reduce / all-to-all | 查 `{{profile.graph.graph_capture.constraints.collective_ops_supported}}`。为真也要确认通信域在捕获期已完成初始化（先跑几次预热通信）；为假则按 Step 1 切图 | `collective_ops_supported` |

**分桶的代价权衡**：桶数 × 每张图的静态缓冲 = 显存增量，桶数 × 捕获耗时 = 启动变慢。用 `{{profile.pytorch.api.max_memory_allocated}}`（CUDA 参考: `torch.cuda.max_memory_allocated`）实测每加一个桶的显存增量。桶太密浪费显存，太稀则 padding 浪费算力——先用 1/2/4/8/16/32 这样的 2 的幂起步，再按实际请求 batch 分布调整。

**代码骨架（图捕获路径）**：

```python
# 占位符均来自 profile，CUDA 参考值见注释。不要把注释里的 CUDA 名字当成你的 API。
graph_cls = {{profile.graph.graph_capture.api}}        # CUDA 参考: torch.cuda.CUDAGraph
sync      = {{profile.pytorch.api.synchronize}}        # CUDA 参考: torch.cuda.synchronize

static_input  = torch.empty(bucket_shape, dtype=..., device="{{profile.pytorch.device_type}}")
static_output = None

for _ in range(WARMUP_ITERS):     # 预热次数需向用户确认，profile 未登记
    static_output = decode_step(static_input)
sync()

graph = graph_cls()
# 捕获上下文的具体写法随运行时而异：是上下文管理器还是 begin/end 成对调用，
# 以及是否必须在非默认流上捕获，均须以你的运行时文档为准。profile 只登记了类名。
# CUDA 参考: with torch.cuda.graph(graph): static_output = decode_step(static_input)

def run(real_input):
    static_input.copy_(real_input)   # 地址不变，只换内容
    graph.replay()                   # 方法名以你的运行时为准
    return static_output.clone()     # 不 clone 会被下一次 replay 覆盖
```

### Step 3 — Graph Break 定位与修复（编译路径）

**输入**：改造后的模型。
**动作**：

1. 先用 `fullgraph=True` 编译。它会把每一个本来会静默切图的地方变成显式报错，是最快的清单生成器
2. 不想中断时，用 `torch._dynamo.explain(fn)(*args)` 拿到 break 列表与原因
3. 日志开关 `TORCH_LOGS="graph_breaks"`（PyTorch 上游环境变量，与设备无关）

**常见 break 原因与修复**：

| break 原因 | 修复 |
|---|---|
| 调用了编译器前端不认识的 Python 内置或第三方函数 | 换成张量运算表达；或把该段包成自定义算子注册给编译器 |
| 依赖张量值的分支（即静态化清单第 3 项） | 同清单第 3 项 |
| 原地修改了被追踪的 Python 对象属性 | 把可变状态改成显式的张量输入输出 |
| 数据类型/设备在图中途变化 | 统一到单一 dtype 与设备，转换提到图外 |
| 日志、打点、异常处理混在热路径里 | 全部移出 |
| 某个算子没有对应的后端实现 | 查 `profile.kernels.*` 是否有替代；没有则在该处主动切图，让这一段回退 eager |

**判据**：把 break 数压到 0 之前，不要去看性能数字。一次 break 就意味着图被拆成两段、多一次下发往返，性能结论没有意义。

### Step 4 — 重编译排查

症状：首次调用慢可以接受，但**每次输入变化都重新编译**，或运行一段时间后突然变慢。

**动作**：打开 `TORCH_LOGS="recompiles"`（PyTorch 上游环境变量），它会打印每次重编译失败的 guard 及其期望值与实际值。

| guard 失败原因 | 修复 |
|---|---|
| 输入 shape 变化 | 做分桶（清单第 5 项）。若确实需要动态维，用 `torch._dynamo.mark_dynamic` 把该维标为动态，让编译器只编一次——**但要先确认你的后端是否支持动态 shape，profile 未登记此项** |
| 作为参数传入的 Python 标量在变（如 `seq_len=37` 下一次变成 `38`） | 改成传张量，或把它纳入分桶 |
| dtype / device 变化 | 统一 |
| 模块属性在推理过程中被改写 | 改成不可变，或作为张量输入传入 |
| 全局变量/配置对象被读取且在变 | 固化到编译时常量，或改为显式参数 |

**重编译次数上限**：编译器通常对同一个函数的编译版本数设上限，超过后整体回退 eager 并打 warning——表现为"跑着跑着突然慢回原样"。这个上限的配置项名与默认值随框架版本变化，用户若遇到此症状，需查其 PyTorch 版本的 `torch._dynamo.config` 中与 cache size limit 相关的项。

### Step 5 — 前后端定界

**什么时候用**：图模式下出错或结果不对，但不知道该怪前端（图的捕获与变换）还是后端（代码生成与硬件执行）。

**动作**：逐级加码，用 PyTorch 上游自带的调试后端做阶梯。这些后端名是框架级的、与设备无关：

| 顺序 | 后端 | 覆盖范围 | 此级失败说明 |
|---|---|---|---|
| 1 | `eager`（Dynamo 前端 + 原样执行） | 只测图捕获 | 问题在前端：Python 字节码追踪、guard、break 处理 |
| 2 | `aot_eager`（加 AOT 追踪与算子分解） | 测分解与 functionalization | 问题在算子分解或原地操作的 functionalization，常见于自定义算子未正确注册 |
| 3 | `{{profile.graph.torch_compile_backend}}`（CUDA 参考: `inductor`） | 测代码生成与硬件执行 | 前两级都过、这一级挂 → **后端问题**，收集最小复现报给你的编译器团队 |

**判读**：第 1 级就失败的，所有排查都该在 Python 层做，不要去看生成的 kernel。第 3 级才失败的，不要再改模型代码，改了也只是绕过。

`profile.graph` 当前只登记了一个 `torch_compile_backend`，没有登记可用的调试后端清单——若你的栈不提供上面两个上游后端，需向用户确认等价的最简后端。

### Step 6 — 三段验证

三段全过才算完成，缺一段结论不成立。

**第一段：编译/捕获通过**

- [ ] `fullgraph=True` 无 break（编译路径）；或捕获过程无报错（图捕获路径）
- [ ] 运行中无重编译日志
- [ ] 没有静默回退到 eager（检查 warning，并用 profiler 确认 kernel 序列确实来自图重放）

**第二段：输出一致性**

固定随机种子，用同一组输入分别跑 eager 与图模式，按下表比对。**比对必须在图模式回退开关关闭的情况下做**，否则可能在比较两次 eager。

| 指标 | 判据 |
|---|---|
| 贪心解码下的 token 序列 | **必须逐 token 完全一致**。这是最严的也是最易执行的判据，优先用它 |
| 输出 logits 的最大绝对/相对误差 | 应落在该 dtype 的数值噪声量级内。基线由用户给出：先跑两次 eager（不同执行顺序）测出 eager 自身的抖动幅度，图模式的误差不应显著超过它 |
| 采样解码下的输出分布 | 同种子应一致；若因随机数生成器被移出/移入图而无法对齐，改用 logits 比对 |
| 长输出的误差累积 | 生成足够长（≥512 token）后再比，短输出掩盖累积偏差 |

不一致时的排查顺序：先查清单第 4 项（输出缓冲被下一次重放覆盖，表现为"结果像上一步的"）→ 再查清单第 1 项（KV cache 写入位置错，表现为前几个 token 对、之后发散）→ 再查清单第 5 项（padding 的位置参与了计算而未被掩码屏蔽）。

**第三段：性能收益**

| 指标 | 怎么测 | 判据 |
|---|---|---|
| 单步 Decode 时延 | 预热后取稳态多步的中位数，不要用首次调用 | 应下降，且降幅与 Step 1 预估的 `gap_ratio` 同量级 |
| `gap_ratio` | 重新 profiling 后按 Step 1 的公式算 | 应显著低于改造前。若 `gap_ratio` 没降，说明图没真正生效 |
| 端到端吞吐 | 完整压测 | 需同步上升；若单步降了而吞吐没升，瓶颈已转移（常见转移到 Prefill 或采样后处理） |
| 显存峰值 | `{{profile.pytorch.api.max_memory_allocated}}` | 记录多桶静态缓冲带来的增量，确认未逼近 `profile.devices.models[].memory_gb` 的上限 |
| 启动耗时 | 从进程启动到第一个请求可服务 | 记录捕获/编译的一次性成本。桶多时这项会明显变长 |

**收益不及预期的三种情况**：编译通过但 `gap_ratio` 没降 → 图没生效（有静默回退或 break）；`gap_ratio` 降了但端到端没降 → 瓶颈本就不在下发，Step 1 的判据没做或做错；单步降了但吞吐没升 → 瓶颈转移，重做性能拆解。

### Step 7 — 开关与回退

图模式必须可一键关闭。

- [ ] 提供显式开关（配置项或环境变量），默认值由用户决定，并在日志中打印当前生效模式
- [ ] 遇到未覆盖的 shape（超出最大桶）时，自动回退 eager 而不是报错
- [ ] 捕获/编译失败时，打印原因并回退 eager，不让服务起不来
- [ ] 回退路径要有测试覆盖——回退分支长期不走，最容易腐坏

## 输出要求

完成后必须汇报：

1. **选定的落地路径**（编译 / 图捕获 / 两者结合）及依据的 profile 字段
2. **上图范围**：哪些阶段进了图、哪些留在 eager、切分点在哪、为什么
3. **Step 1 的收益预判**：改造前的 `gap_ratio` 及其测量方法；无 profiler 时明确说明未做预判
4. **静态化改造清单的逐条结论**：满足 / 改了什么 / 因 profile 约束未知而按保守假设处理
5. **分桶方案**：桶的取值、覆盖率、显存增量、超桶的回退行为
6. **Graph Break 与重编译**：最终 break 数（应为 0）、修复过哪些、是否仍有重编译
7. **三段验证的完整数据**：三段各自的判据与实测值，一段不过就不得给出"完成"结论
8. **开关与回退**：开关名、默认值、回退触发条件
9. **残留风险**：因 profile 字段缺失而采用的保守假设清单、`_status` 为 `assumed` 的字段、未覆盖的输入形态

## 知识来源

本技能的方法论全部来自公开技术资料与通用工程实践，不含任何厂商专有信息：

- **PyTorch 官方文档**：`torch.compile` 与 TorchDynamo 文档（Graph Break 的产生条件、guard 与重编译机制、`fullgraph` 与 `explain` 的用法、`TORCH_LOGS` 调试开关、调试后端的分级 `eager` / `aot_eager`）；CUDA Graphs 集成文档（静态输入输出地址、捕获期禁止主机同步、预热与内存池要求、`reduce-overhead` 模式）
- **NVIDIA 公开文档**：CUDA Programming Guide 的 CUDA Graphs 章节（图捕获的约束、重放语义、下发开销模型）；关于 kernel launch 开销量级的公开材料
- **开源推理框架的公开实现与设计文档**：vLLM 的 CUDA Graph 捕获策略（只捕获 Decode、按 batch 分桶、预分配 KV cache）、SGLang 与 TensorRT-LLM 公开资料中关于静态图与分桶的讨论
- **PagedAttention 论文与 vLLM 开源实现**：KV cache 按块预分配、原地写入的组织方式
- **通用工程方法**：前后端定界的阶梯式排除法、三段验证（功能 / 正确性 / 性能）的验收结构
