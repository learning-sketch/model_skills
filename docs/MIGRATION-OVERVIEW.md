# 迁移总览

本文档记录 [hicann/cannbot-skills](https://github.com/hicann/cannbot-skills) 的 `model/` 目录（21 个 skill）向类 CUDA 自研芯片的迁移分级与状态。

源仓库快照：commit `03cbbb9`，21 个 skill，132 个文件，约 3.2 万行。

---

## 一句话结论

21 个 skill 中，**9 个已迁移**、**12 个保留待议**。

分界线不是"方法论是否通用"——绝大多数方法论都通用。真正的分界线是**这个 skill 的交付价值有多大比例落在厂商专有资产上**：算子目录、profiler 产物 schema、错误码本、dump 工具链。方法论可以改写，这些资产只能重建。

---

## 迁移分级标准

| 等级 | 判据 | 处置 |
|------|------|------|
| **A** | 方法论与工具都基本可用，改名即可 | 直接迁移 |
| **B** | 方法论通用，需替换 API 与工具链引用 | 迁移，厂商部分收敛到 profile |
| **C** | 深度绑定硬件架构或专有工具栈，需重新设计 | 保留，写迁移分析文档 |
| **D** | 依赖目标芯片没有的硬件特性，迁移无意义 | 保留，记录技术本质备查 |

"保留价值"一栏是**迁移后仍然有效的内容占原 skill 的比例估计**，依据是逐文件审读，不是行数统计。

---

## 已迁移（9 个）

| Skill | 等级 | 保留价值 | 迁移时做了什么 |
|-------|------|---------|---------------|
| [model-train-log-visualization](../skills/model-train-log-visualization/) | A | ~90% | 日志正则改为 profile 驱动，脚本逻辑未动 |
| [model-train-oom-analysis](../skills/model-train-oom-analysis/) | B | ~75% | OOM 来源分类表改为 profile 驱动；去掉 workspace allocator 这一昇腾独有分支的硬假设 |
| [model-infer-parallel-analysis](../skills/model-infer-parallel-analysis/) | B | ~75% | 互联带宽、显存规格改为读 profile；保留全部决策树与估算公式 |
| [model-infer-runtime-debug](../skills/model-infer-runtime-debug/) | B | ~65% | 保留同步二分定位骨架；错误码表整表换成 profile 的 `error_signatures` |
| [model-infer-multi-stream](../skills/model-infer-multi-stream/) | B | ~60% | 保留 DAG 拆解与 `overlap_pct` 真假并行判据；实现层对位到标准 Stream/Event；控核部分降级为可选能力 |
| [model-infer-perf-breakdown](../skills/model-infer-perf-breakdown/) | B | ~60% | 定义统一 kernel schema + 适配层，脚本算法未动 |
| [model-infer-migrator](../skills/model-infer-migrator/) | B | ~55% | 保留场景分流与基线采集；框架接入契约改为 profile 的 `inference_framework` |
| [model-infer-kvcache](../skills/model-infer-kvcache/) | B | ~50% | 保留 PagedAttention 地址映射与 metadata 生命周期；FA 调用改为能力探测 |
| [model-infer-graph-mode](../skills/model-infer-graph-mode/) | B | ~45% | 保留 Decode 上图与静态化方法论；GE/npugraph_ex 双后端收敛为单一图捕获抽象 |

---

## 保留待议（12 个）

按"迁移阻塞的性质"分组。每个 skill 的详细分析在 [`pending-migration/`](pending-migration/) 下。

### 组 1：等一个前置能力就能迁（4 个）

这些 skill 的方法论没问题，卡在一个明确的前置条件上。前置条件满足后迁移是机械工作。

| Skill | 等级 | 卡在哪 | 前置条件 |
|-------|------|--------|---------|
| [model-infer-profiling](pending-migration/model-infer-profiling.md) | C | 采集 API 与产物 schema 全是昇腾专有 | 确定你的 profiler 能否产出含流水线计数器的 kernel 表 |
| [model-infer-precision-debug](pending-migration/model-infer-precision-debug.md) | C | 排查流程通用，但附录是昇腾 FA 参数契约 | 确定你的 attention 算子参数契约 |
| [model-recommend-analysis](pending-migration/model-recommend-analysis.md) | B | 子图挖掘算法完全通用，解析层绑 MindStudio + GE pbtxt | 同 profiling；外加确定你的图 IR 导出格式 |
| [model-infer-quantization](pending-migration/model-infer-quantization.md) | B | 工作流通用，量化 runtime 与权重布局绑昇腾 | 确定你的量化格式与是否需要特殊权重布局 |

### 组 2：需要自建工具栈（3 个）

训练精度三件套。方法论（随机性契约、门禁、证据分级）质量很高且完全通用，但日常执行路径几乎等同于 msprobe 操作手册。

| Skill | 等级 | 不建工具的价值 | 建了工具的价值 |
|-------|------|--------------|--------------|
| [model-train-precision-determinism](pending-migration/model-train-precision-determinism.md) | C | ~50% | ~75% |
| [model-train-precision-nonfinite](pending-migration/model-train-precision-nonfinite.md) | C | ~55% | ~80% |
| [model-train-precision-numerical-mismatch](pending-migration/model-train-precision-numerical-mismatch.md) | C–D | ~40% | ~73% |

三者共用一套 dump/比对底座，建议**合并评估、一次性建设**。详见 [精度三件套合并分析](pending-migration/_train-precision-trio.md)。

### 组 3：依赖目标芯片可能没有的硬件能力（3 个）

| Skill | 等级 | 依赖的能力 | 若你的芯片没有 |
|-------|------|-----------|--------------|
| [model-infer-prefetch](pending-migration/model-infer-prefetch.md) | C | 以 tensor 依赖为栅栏的片上缓存预取 | 整个 skill 失去落点，只剩 roofline 判据 |
| [model-infer-superkernel](pending-migration/model-infer-superkernel.md) | C/D | 用 scope 标注、编译器把子图编成单个二进制 kernel | 退化为图捕获，与 graph-mode 重复 |
| [model-infer-fusion](pending-migration/model-infer-fusion.md) | C | 一个成规模的融合算子目录 | 流程仍可用，但需先建自己的算子目录 |

### 组 4：建议不迁（2 个）

| Skill | 等级 | 原因 |
|-------|------|------|
| [model-infer-parallel-impl](pending-migration/model-infer-parallel-impl.md) | C | MoE 并行主路径建在 MC2 通信计算融合算子上，这是昇腾独有能力。切分抽象本身在 Megatron/vLLM 里已有更成熟的实现，重写不如直接对接 |
| [model-infer-harmony](pending-migration/model-infer-harmony.md) | D | 麒麟 9030 端侧 ASR 的 omg/omc/CANNPAK 全链路，与云侧类 CUDA 芯片无共同基础。仅抽象 PTQ 方法论有参考价值 |

---

## 跨 skill 的共性发现

审读 21 个 skill 后，有四件事反复出现，它们决定了迁移的真实成本。

**一、方法论的质量远高于实现的可迁移性。** 几乎每个 skill 的决策树、检查清单、验证纪律都是厂商中立的，写得也扎实——比如 multi-stream 的"Stream ID 切对不等于物理并行，要用 `overlap_pct` 验证"，runtime-debug 的"异步执行下报错点不等于出错点，要用同步检查点二分"。这些是真正的资产。被绑死的是算子名、列名、错误码这些查表项。所以迁移的正确做法是**抽出方法论、重建查表项**，而不是逐行改写。

**二、profiler 产物格式是最大的单点阻塞。** `model-infer-profiling` 产出的 47 列 `kernel_details.csv` 是 `model-infer-perf-breakdown`、`model-recommend-analysis`、`model-infer-multi-stream` 验证环节、`model-infer-prefetch` 判据的共同输入。这一个格式不解决，四个 skill 都半残。所以本次迁移把它单独抽成了 [统一 kernel schema](../tools/profiler_adapter/README.md)。

**三、达芬奇架构的 Cube/Vector 分离是若干规则的隐藏前提。** 控核（`limit_core_num`）、AIC/AIV 利用率判据、MIX_AIC/MIX_AIV 任务类型，这些在 SIMT 架构上没有对应物。涉及这些的规则不能机械改名，必须判断在你的架构上是否还成立。

**四、"通信计算融合"（MC2）是 CUDA 生态的真空地带。** 昇腾把 all-to-all 与 grouped GEMM 编进一个算子，CUDA 侧只能靠多流手工重叠或 DeepEP 这类专用库。涉及 MoE 并行的 skill 都受此影响。

---

## 下一步

需要你拍板的问题集中在 [MIGRATION-DISCUSSION.md](MIGRATION-DISCUSSION.md)。那里列了 12 个待议 skill 的具体选项和各自代价，按"回答后能解锁多少 skill"排了序。

最省力的起点是**填一份 [硬件 profile](../hardware-profiles/_TEMPLATE.yaml)**。已迁移的 9 个 skill 全部读它，填完即可用；填的过程本身也会暴露出哪些能力缺口会卡住剩下 12 个。
