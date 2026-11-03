# 能力地图与建设状态

本文档是整个仓库的主索引。

## 先读这个

任务最初的设想是把 [hicann/cannbot-skills](https://github.com/hicann/cannbot-skills) 的 `model/` 目录迁移过来。实际执行时发现**该项目的许可证明文禁止把它的内容用于非华为芯片**，详见 [LICENSE-CONSTRAINT.md](LICENSE-CONSTRAINT.md)。

所以本仓库改为**按相同的能力领域独立编写**。源仓库的作用退化为一份能力清单的参照——"做 LLM 推理与训练优化大致需要覆盖哪些事"，这部分是行业共识，不是谁的专有资产。所有具体内容来自公开技术文献与通用生态实践。

下文仍沿用"已建设 / 待建设"的分法来组织，但请注意：这里说的不是搬运进度，而是**你的芯片上这套能力建到了哪一步**。

---

## 一句话结论

21 个能力领域中，**9 个已经写出可用的 skill**，**12 个待定**。

待定的原因不是"方法论难写"——方法论基本都是公开知识。待定是因为这些领域的 skill 要能真正干活，**必须先有一个你的芯片上还不存在、或我还不知道是否存在的底层能力**：一个能导出流水线计数器的 profiler、一套张量级 dump 比对工具、一个成规模的融合算子库。没有这些，skill 写出来也只是一篇读起来合理但跑不动的文档。

这 12 个领域具体缺什么、各有哪些建设选项，是 [DISCUSSION.md](DISCUSSION.md) 要和你讨论的内容。

---

## 分级标准

| 等级 | 判据 |
|------|------|
| **A** | 底层能力是 PyTorch 原生的，你的芯片只要接上 PyTorch 就有 |
| **B** | 底层能力在通用生态里成熟（vLLM / NCCL / torch.compile），需要确认你的栈提供了哪一个 |
| **C** | 需要先建设一个专门的底层能力，建设成本明确但不小 |
| **D** | 依赖特定硬件特性，你的芯片没有的话这个领域就不成立 |

---

## 已建设（9 个）

这些 skill 现在就能用。它们读 [硬件 profile](../hardware-profiles/_TEMPLATE.yaml)，填完 profile 即可生效。

| Skill | 等级 | 底层依赖 | 知识来源 |
|-------|------|---------|---------|
| [model-train-log-visualization](../skills/model-train-log-visualization/) | A | 只需要训练日志是文本 | 通用日志解析与 matplotlib |
| [model-train-oom-analysis](../skills/model-train-oom-analysis/) | A | PyTorch memory snapshot | PyTorch 官方显存分析文档 |
| [model-infer-parallel-analysis](../skills/model-infer-parallel-analysis/) | B | 只需要知道硬件规格 | Megatron-LM / DeepSpeed 并行策略文献 |
| [model-infer-runtime-debug](../skills/model-infer-runtime-debug/) | B | 设备同步 API + 错误信息 | 异步执行调试的通用方法 |
| [model-infer-migrator](../skills/model-infer-migrator/) | B | 目标推理框架的接入契约 | HuggingFace / vLLM 模型接入实践 |
| [model-infer-kvcache](../skills/model-infer-kvcache/) | B | 分页 attention 算子 | vLLM PagedAttention 论文与实现 |
| [model-infer-graph-mode](../skills/model-infer-graph-mode/) | B | 图捕获或 torch.compile | PyTorch CUDA Graph / Dynamo 官方文档 |
| [model-infer-multi-stream](../skills/model-infer-multi-stream/) | B | Stream / Event | CUDA 流并发模型 |
| [model-infer-perf-breakdown](../skills/model-infer-perf-breakdown/) | B | 任一能导出 kernel 时间线的 profiler | 自定义，配 [统一 kernel schema](../tools/profiler_adapter/README.md) |

---

## 待建设（12 个）

按"缺的是什么"分组。每组的详细分析在 [`pending/`](pending/) 下。

### 组 1：缺一个够用的 profiler（4 个）

这四个领域的输入都是性能数据。你的 profiler 能导出什么，直接决定它们能做到什么程度。

| 领域 | 等级 | 最低要求 | 理想要求 |
|------|------|---------|---------|
| [性能数据采集](pending/perf-profiling.md) | C | kernel 名 + 时长 + 流 ID | 外加流水线利用率计数器 |
| [精度问题诊断（推理）](pending/infer-precision-debug.md) | C | 能逐层 dump 张量 | 外加 attention 算子的参数契约文档 |
| [图结构与融合机会分析](pending/graph-fusion-analysis.md) | B | 能导出计算图 IR | 外加 kernel 耗时，用于给子图排序 |
| [量化接入](pending/quantization.md) | B | 知道支持哪些量化格式 | 外加量化算子的生效验证手段 |

### 组 2：缺一套精度 dump 与比对工具（3 个）

训练精度诊断三件套。方法论（随机性契约、复现门禁、证据分级）是公开可得的工程实践，但日常执行离不开一个能按调用实例对齐的 dump + 比对工具。CUDA 生态**没有**开源的对等物，各家都是自建。

| 领域 | 等级 | 没有工具时 | 有工具后 |
|------|------|-----------|---------|
| [训练结果不可复现](pending/train-precision-determinism.md) | C | 能查随机性契约，查不出首个差异点 | 完整可用 |
| [训练出现 NaN / Inf](pending/train-precision-nonfinite.md) | C | `detect_anomaly` 能兜住一部分 | 完整可用 |
| [训练与标杆数值不一致](pending/train-precision-mismatch.md) | C | 只能看 loss 曲线级别 | 完整可用 |

三者共用同一套底座，建议**一次性建设**。见 [精度工具栈建设方案](pending/_precision-toolchain.md)。

### 组 3：缺特定硬件能力（3 个）

| 领域 | 等级 | 需要的硬件能力 | 没有的话 |
|------|------|--------------|---------|
| [权重预取](pending/weight-prefetch.md) | C | 以数据依赖为栅栏的片上缓存预取 | 领域不成立，退化为 roofline 分析 |
| [子图二进制融合](pending/subgraph-fusion.md) | C/D | 编译器把标注范围编成单个 kernel | 退化为图捕获，与 graph-mode 重复 |
| [融合算子替换](pending/fused-op-substitution.md) | C | 一个成规模的融合算子库 | 流程可用但无算子可换 |

### 组 4：建议直接对接生态而非自建（2 个）

| 领域 | 等级 | 理由 |
|------|------|------|
| [并行切分实施](pending/parallel-impl.md) | C | Megatron-LM 和 vLLM 已有成熟实现，自己写一套切分层投入产出比低。建议把精力放在让你的通信库适配它们 |
| [端侧部署](pending/edge-deployment.md) | D | 源仓对应的是手机 NPU 的离线模型转换与打包，与云侧训推芯片无共同基础。除非你有端侧产品线，否则不必建 |

---

## 四个贯穿性判断

审读完整个能力清单后，有四件事影响的不止一个领域，值得单独拿出来说。

**一、profiler 产物格式是最大的单点阻塞。** 组 1 的四个领域、加上已建设的 perf-breakdown 和 multi-stream 的验证环节，输入都是性能数据。一个格式不定，六个地方都要打补丁。本仓库为此先定义了[统一 kernel schema](../tools/profiler_adapter/README.md)：skill 只认这个 schema，你的 profiler 怎么导出由一个适配器负责。先把适配器写出来，六个领域一起解锁。

**二、方法论的价值远高于实现的可移植性。** 这类工程 skill 真正值钱的是判据和纪律，比如"流 ID 切对了不等于真的并行，要用时间窗重叠率验证"，"异步执行下报错的位置不是出错的位置，要用同步点二分"。这些话在哪个芯片上都成立。不值钱的是算子名、列名、错误码——它们必然要重建，也必然只能靠你自己填。所以本仓库的 skill 一律把前者写进正文，把后者赶进 profile。

**三、要警惕架构假设被悄悄继承。** 有些优化规则的前提是特定的硬件结构，换个架构就不成立。典型例子是"按流限制计算核数量"这类手段——它预设了计算单元可以被细粒度划分给不同的流，SIMT 架构上没有对等的一等公民 API。本仓库凡是涉及这类规则的地方都标注了前提条件，不要机械套用。

**四、通信与计算的融合是通用生态的真空地带。** 把 all-to-all 和 grouped GEMM 编进同一个算子，可以省掉中间结果的往返。CUDA 侧没有一等公民支持，只能用多流手工重叠，或依赖 DeepEP 这类专用库。MoE 相关的优化都会碰到这堵墙，是自研芯片有机会做出差异化的地方。

---

## 怎么开始

1. **填 [硬件 profile](../hardware-profiles/_TEMPLATE.yaml)**。已建设的 9 个 skill 全部读它。参照 [cuda-reference.yaml](../hardware-profiles/cuda-reference.yaml) 填写。填的过程会自然暴露出能力缺口
2. **读 [DISCUSSION.md](DISCUSSION.md)**，回答里面的问题。问题按"回答后能解锁几个领域"排了序
3. 从 **profiler 适配器**开始建设，它的杠杆率最高
