# model_skills

面向自研 AI 加速卡的大模型训练与推理优化 Agent Skills。

正文只写厂商中立的方法论，所有与具体芯片相关的 API 名、工具命令、规格数字、错误码，一律从一份**硬件 profile** 读取。换一款芯片只需要换一份 profile，不需要改 skill。

---

## 三步上手

**第一步，填硬件 profile。**

```bash
cp hardware-profiles/_TEMPLATE.yaml hardware-profiles/my-chip.yaml
```

照着 [`cuda-reference.yaml`](hardware-profiles/cuda-reference.yaml) 填。填不出来的留 `null`，并把所在小节的 `_status` 标成 `unknown`——skill 会走降级路径并提示缺什么。**不要为了填满而猜**，一个错的带宽值会让所有 roofline 判定系统性偏向一边，比留空危害大。

**第二步，看能力地图。** [`docs/MIGRATION-OVERVIEW.md`](docs/MIGRATION-OVERVIEW.md) 列出 21 个能力领域、各自的状态和依赖。

**第三步，回答决策问题。** [`docs/DISCUSSION.md`](docs/DISCUSSION.md) 按「一个答案能解锁几个领域」排序。12 个待建设领域的下一步取决于这些答案。

---

## 目录

| 路径 | 内容 |
|---|---|
| [`skills/`](skills/) | 可用的 Agent Skills，每个一个目录，含 `SKILL.md` 和可选的 `scripts/` |
| [`hardware-profiles/`](hardware-profiles/) | 硬件能力描述。`_TEMPLATE.yaml` 是空白模板，`cuda-reference.yaml` 是填写示例 |
| [`tools/profiler_adapter/`](tools/profiler_adapter/) | 统一 kernel 事件 schema 与各家 profiler 的适配层 |
| [`docs/`](docs/) | 能力地图、决策清单，以及 `pending/` 下 12 个待建设领域的分析 |

---

## 已建成的 skills

| Skill | 做什么 | 前置条件 |
|---|---|---|
| [model-infer-perf-breakdown](skills/model-infer-perf-breakdown/) | 把 kernel 时间线拆成「下一步该改哪里」 | 有性能数据，且已接入适配层 |
| [model-infer-kvcache](skills/model-infer-kvcache/) | KV cache 布局与分页 attention 适配 | 有分页 attention 算子 |
| [model-infer-graph-mode](skills/model-infer-graph-mode/) | 图捕获 / `torch.compile` 静态化改造 | 支持图捕获或 Dynamo |
| [model-infer-multi-stream](skills/model-infer-multi-stream/) | 多流并发，含「是否真的并行」的验证方法 | 有 Stream / Event |
| [model-infer-migrator](skills/model-infer-migrator/) | 把新模型接进推理框架 | 目标框架的接入契约 |
| [model-infer-runtime-debug](skills/model-infer-runtime-debug/) | 异步执行下的报错与卡死定位 | 设备同步 API |
| [model-infer-parallel-analysis](skills/model-infer-parallel-analysis/) | TP / EP / PP 策略选型，给可计算的显存与通信估算 | 只需硬件规格 |
| [model-train-log-visualization](skills/model-train-log-visualization/) | 训练日志解析与曲线对比 | 日志是文本 |
| [model-train-oom-analysis](skills/model-train-oom-analysis/) | 训练显存不足诊断与泄漏检测 | PyTorch memory snapshot |

12 个待建设领域各自缺什么、怎么建，见 [能力地图](docs/MIGRATION-OVERVIEW.md)。

---

## profiler 适配层

性能数据是六个以上能力领域的共同输入，而每家 profiler 的产物格式都不一样。这里的做法是定义一个统一的 kernel 事件 schema，分析侧只认这个 schema，每种 profiler 写一个适配器。

```bash
# torch.profiler 导出的 chrome trace
python -m profiler_adapter.normalize --input trace.json --format chrome-trace --output kernels.jsonl

# 通用 CSV，列名映射从 profile 读
python -m profiler_adapter.normalize --input table.csv --format csv \
    --profile hardware-profiles/my-chip.yaml --output kernels.jsonl

# 检查数据够不够后续分析用，会直接告诉你哪些分析因缺字段做不了
python -m profiler_adapter.normalize --input kernels.jsonl --format jsonl --validate-only
```

接新格式只需实现一个 `(路径, 配置) -> Iterator[KernelEvent]` 的函数并注册。详见 [适配层说明](tools/profiler_adapter/README.md)。

---

## 这些 skill 怎么写的

两条贯穿全仓的原则。

**方法论进正文，参数进 profile。** 这类工程 skill 真正值钱的是判据和纪律——「流 ID 切对了不等于真的并行，要用时间窗重叠率验证」「异步执行下报错的位置不是出错的位置，要用同步点二分」。这些话在哪个芯片上都成立。不值钱的是算子名、列名、错误码，它们必然要重建，也只能由你填。

**不确定就说不确定，不猜。** 缺硬件计数器时，受限类型一律报「未知」而不是按耗时猜一个；profile 字段为 `null` 时走降级路径并说明降级了什么。一个标着「未知」的结论比一个猜错的结论有用。

---

## 来源与许可

本仓库内容为独立编写，知识来源是公开技术文献与通用生态实践（PyTorch 官方文档、vLLM / FlashInfer 等开源实现、Megatron-LM 与 ZeRO 等论文、NVIDIA 公开文档）。每份 skill 和分析文档末尾都标注了自己的知识来源，便于核查。

本项目最初的设想是迁移 [hicann/cannbot-skills](https://github.com/hicann/cannbot-skills) 的 `model/` 目录，执行中发现其许可证明文禁止把内容用于非华为处理器，因此改为按相同能力领域独立编写。依据、边界与影响记录在 [`docs/LICENSE-CONSTRAINT.md`](docs/LICENSE-CONSTRAINT.md)。
