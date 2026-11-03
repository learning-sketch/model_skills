---
name: model-train-log-visualization
description: 解析训练日志并画成曲线图。当用户说「把这次训练的 loss 画出来」「grad_norm 怎么走的」「这两次训练对比一下」「新改的实现和基线差多少」「误差曲线看一眼」「训练日志里 memory/吞吐的走势」，或者给出一个训练日志文件路径并要求可视化/对齐/比对时使用。也用于定位 loss 从第几个 step 开始变 NaN。
---

# 训练日志解析与可视化

把纯文本训练日志变成以 step 为横轴的曲线图；给两份日志时，额外把它们按 step 对齐并画出误差。

目标是回答三类问题：**这次训练正常吗**（单日志看趋势）、**这次改动动了什么**（双日志看误差）、**什么时候开始坏的**（找首个 NaN / 突变点）。

---

## 硬件依赖

本 skill 对硬件的依赖只有一条：训练日志是文本。所有芯片相关的信息都从硬件 profile 读。

| profile 字段 | 用途 | 缺失时的降级行为 |
|---|---|---|
| `training_framework.log_format.metric_keys.step` | step 在日志中的字面字段名，决定横轴怎么抓 | 退回内置别名表（`global_step` / `iteration` / `iter` / `step`）。全都抓不到时报错退出，不猜 |
| `training_framework.log_format.metric_keys.loss` | loss 字段名 | 退回内置别名表（`global_avg_loss` / `train_loss` / `lm_loss` / `lm loss` / `loss`） |
| `training_framework.log_format.metric_keys.grad_norm` | grad_norm 字段名 | 退回内置别名表（`grad_norm` / `grad norm` / `gnorm`） |
| `training_framework.log_format.metric_keys.memory` | 显存占用字段名 | 退回内置别名表；抓不到则该面板留空并告警 |
| `training_framework.log_format.metric_keys.throughput` | 吞吐字段名 | 同上 |
| `training_framework.log_format.metric_keys.tflops` | 算力字段名 | 同上 |
| `training_framework.log_format.metric_keys.mfu` | 算力利用率字段名 | 同上 |
| `training_framework.log_format.metric_keys.step_time` | 单步耗时字段名 | 同上 |
| `training_framework.log_format.style` | 日志风格标识，用于向用户说明「按哪套格式在解析」 | 不影响解析，只影响提示文案 |
| `training_framework._status` | 字段名是否已在真机验证 | 非 `verified` 时告警，提示先用一份真实日志核对字段名 |
| `training_framework.name` | 训练框架名，用于提示用户去哪找日志 | 不提示具体路径 |

**关键设计：字段名一律不写死。** 优先级是「内置默认别名表 < profile < 命令行 `--metric-key`」。换框架、换芯片、自研训练脚本改了打印格式，都只需要改 profile 或加一个 `--metric-key`，不需要动脚本。

`metric_keys` 的每一项既可以填单个字符串，也可以填字符串列表。**同一份日志里同一指标有多种写法是常态**——不同 rank、训练与评估阶段、框架升级前后都可能不一致。填列表时脚本会逐个尝试，不必为此改代码或跑两遍。

---

## 适用场景

- 画单次训练的 loss / grad_norm / 显存 / 吞吐 / MFU 曲线
- 对比两次训练：换了并行配置、换了算子实现、换了 dtype、换了芯片，想知道曲线差多少
- 定位发散：loss 从第几个 step 开始变 NaN、grad_norm 什么时候炸的
- 给精度问题做第一层筛查：先看 loss 曲线级别有没有分叉，再决定要不要上更重的逐层比对

## 不适用场景

- **结构化指标后端**。日志已经进了 TensorBoard / W&B / MLflow 的，直接用它们的 UI，别绕回文本解析
- **逐层/逐算子的数值比对**。本 skill 的粒度是「每个 step 一个标量」。要定位首个出现差异的算子，需要张量级 dump 与比对工具，不在这里
- **性能瓶颈拆解**。日志里的 MFU / 吞吐只能告诉你「慢了」，不能告诉你「慢在哪个 kernel」。那是 profiler 的活
- **二进制或结构化日志**（protobuf / Parquet / 自定义 binary）。先用你的工具导出成文本再来
- **单份日志只有一两个 step**。画不出趋势，直接看数字更快

---

## 工作流

### Step 0：确认输入

按顺序问清这四件事，问完再动手。**不要在信息不全时猜测**，尤其是日志路径。

1. **主日志路径**（必需）。如果用户只说了「这次训练」，先问清文件在哪。顺带确认是单卡还是多卡日志——多卡场景通常只看 rank 0，若日志是所有 rank 混在一起的，需要先 `grep` 出一个 rank
2. **对比日志路径**（可选）。给了就进双日志模式，画误差；没给就是单日志模式
3. **要看的指标**（可选）。`loss` 和 `grad_norm` 无论如何都会画，用户可以追加 `memory` / `throughput` / `tflops` / `mfu` / `step_time` / `learning_rate` / `loss_scale`，或者直接要全部
4. **输出路径**（可选）。默认写到当前目录的 `training_log.png` / `training_log_compare.png`

同时确认有没有可用的硬件 profile。有就带上 `--profile`，没有就靠内置别名表，并在结果里说明用的是哪一套字段名。

### Step 1：先解析，确认字段名抓对了

先只跑解析，不要直接出图。出图之前必须确认横轴和指标抓的是对的东西——一张用错字段画出来的图比没有图更糟。

```bash
python scripts/parse_training_log.py <主日志> --profile <profile.yaml>
```

重点看三处输出：

- **命中的字段名**（`grad_norm<-grad norm` 这种）。确认每个指标是从你以为的那个字段抓来的。日志里同时有 `lm loss` 和 `val loss` 时，笼统的 `loss` 别名可能抓错，需要用 `--metric-key "loss=lm loss"` 钉死（字段名含空格时记得加引号）
- **有效 step 数与区间**。和你预期的训练步数对不上，多半是字段名抓错或日志被截断
- **告警区**。缺失的指标、重复的 step、没有 step 的指标行、不一致的单位，都在这里

字段名不对就加 `--metric-key`，可重复：

```bash
python scripts/parse_training_log.py <主日志> \
    --metric-key "loss=lm loss" --metric-key grad_norm=gnorm --metric-key step=iteration
```

一个指标可以给多个候选别名（逗号分隔，按顺序试）：`--metric-key loss=lm_loss,train_loss`。
注意 `--metric-key` 对该指标是**整组替换**而不是追加——这是刻意的，为的是能彻底摆脱默认别名的干扰。

`--show-aliases` 可以打印最终生效的完整别名表，用来确认优先级合成的结果。

### Step 2：单日志出图

```bash
python scripts/plot_training_log.py <主日志> --profile <profile.yaml> \
    --metrics memory,throughput,mfu -o curves.png --no-show
```

读图顺序：

1. **loss 整体形状**。单调下降、带噪声下降是正常；平台期、反弹、突然垂直上升都要追
2. **grad_norm**。和 loss 一起看。grad_norm 先炸、loss 随后变 NaN，是典型的梯度爆炸链条；grad_norm 长期贴近 0 则可能是梯度消失或学习率过小
3. **有没有断点**。曲线上出现断口说明那段是 NaN/Inf，图上会用红色竖线标出首个 NaN 的 step，并在右上角给出 NaN/Inf 计数
4. **显存曲线的斜率**。单调上升不收敛是显存泄漏的信号，接着走 OOM 诊断那条路

### Step 3：双日志对比

```bash
python scripts/plot_training_log.py <主日志> --compare <对比日志> \
    --profile <profile.yaml> -o compare.png --no-show
```

误差定义固定为**有符号**，方向是「对比 − 基线」：

- 绝对误差 `= candidate - baseline`
- 相对误差 `= (candidate - baseline) / baseline`

带符号是刻意的。只看绝对值会丢掉最有用的信息：**偏差是单边的还是双向抖动的**。单边偏差（误差曲线整体在 0 的一侧，且随 step 累积）指向系统性问题——算子实现不等价、累加顺序不同、dtype 不同；双向抖动且幅度稳定则更像随机性差异——数据顺序、dropout、集合通信的规约顺序。

读图顺序：

1. **先看相对误差的量级**。训练 loss 的相对误差在 1e-3 以下通常是噪声级别；到了 1e-2 量级并持续，就得当成真问题查
2. **再看它随 step 的走向**。放大、收敛、还是平稳，决定了是「误差在累积」还是「一次性偏移」
3. **最后看绝对误差的符号分布**。整段同号 → 系统性；围绕 0 抖动 → 随机性
4. 脚本会报出**偏离最大的 step**，直接拿这个 step 号去做更细粒度的排查

需要把对比日志当基线时加 `--baseline-is-compare`。

### Step 4：把数据交出去

要进一步做统计或喂给别的工具时，用 JSON：

```bash
python scripts/parse_training_log.py <主日志> --compare <对比日志> --json -o result.json
```

默认输出是严格合法的 JSON（NaN/Inf 写成 `null`，数量另由 `nan_count` / `infinite_count` 给出）。要保留 `NaN` / `Infinity` 字面量就加 `--json-non-finite literal`。日志很长时加 `--omit-points` 只留统计量。

---

## 输出要求

**图的硬约束：**

1. **横轴统一为 step**。不用墙钟时间、不用行号、不用 epoch。step 是唯一能让两份日志对齐的坐标。日志确实不打印 step 时，必须显式 `--synthesize-steps` 才会用行序号顶上，并且要在结论里讲明「横轴是合成的」
2. **loss 和 grad_norm 必绘**。即使日志里没有，也画占位面板并写明「未找到该指标」。少画一张图容易被忽略，画一张写着「没找到」的图不会
3. **双日志必须额外画误差**。布局是每个指标一行、三列：曲线叠加 | 绝对误差 | 相对误差。只叠加曲线不画误差是不够的——两条几乎重合的曲线，肉眼分不出 1e-3 还是 1e-5 的差距
4. **非有限值必须可见**。NaN/Inf 处断线而不是跨过去连直线，首个 NaN 标红色竖线，角标给出 NaN/Inf 计数
5. **纵轴带单位**。从日志里抓到的单位后缀（`GiB` / `%` / `s`）要显示在轴标签上

**结论的硬约束：**

- 必须说明**用的是哪一套字段名**，以及来自 profile 还是内置默认
- 必须转述解析阶段的全部告警，尤其是缺失指标和重复 step
- 双日志必须给出**共同 step 的数量与区间**。共同 step 只覆盖一小段时，结论的适用范围就只有那一段
- 不要把相对误差的量级直接翻译成「精度有问题 / 没问题」。给出数值和走向，判据交给用户的精度标准

---

## 边界与处置

这几条是硬规则，不要绕过。

| 情况 | 处置 | 为什么 |
|---|---|---|
| 两份日志**没有共同 step** | **报错退出（退出码 4），不出图** | 两条来自不相交区间的曲线画在同一张图上，看起来像一次有效对比，实际毫无意义。常见原因：续训日志的 step 从 checkpoint 处继续、某一侧的 step 字段抓错 |
| 同一 step **重复出现** | 默认**保留最新**（`--on-duplicate-step last`），并告警 | 日志里重复 step 通常来自重试、断点续训覆写、或 eval 分支重打了一次。最新的一条最接近当时的真实状态。需要别的策略时有 `first` / `merge` |
| 某指标在日志中**一次都没出现** | **告警**，并在图上画占位面板 | 静默跳过会让人误以为「这个指标一切正常」。真实原因通常是字段名不匹配，必须提示用户用 `--metric-key` 指定 |
| 某指标**只在部分 step 出现**（覆盖率 < 95%） | 告警，曲线照画但会有断点 | 常见于每 N 步才打印一次的指标，本身正常，但需要让用户知道断点不是数据丢失 |
| 日志里有**指标但没有 step** | 计数并告警，默认丢弃这些行 | eval / summary 行常常不带 step。强行给它们编号会污染横轴 |
| **空文件 / 一个指标都没抓到** | 报错退出（退出码 3），并列出尝试过的全部别名 | 直接报「解析失败」没有信息量，要让用户看到脚本在找什么 |
| 同一指标出现**两种单位** | 告警 | `GiB` 和 `MiB` 混排时，数值不能直接比较，误差更是没有意义 |
| 两份日志同一指标**单位不同** | 告警，误差照算但标注无物理意义 | |
| **超大日志** | `--max-lines N` 设上限，截断时告警 | 解析本身是流式的（逐行处理，不 `readlines`），内存不是瓶颈；但几 GB 的日志解析时间会很长，先截一段看看格式对不对更划算 |
| 日志带 **ANSI 颜色**或 **tqdm 进度条** | 自动处理 | 颜色转义会被剥离；进度条用 `\r` 覆写导致整份日志只有一「行」，解析时会按 `\r` 再切一刀 |
| **非 UTF-8 编码** | `--encoding` 指定；解码错误一律 `replace` | 不能因为日志中间混进几个坏字节就整份解析失败 |
| 环境**没有图形显示** | `--no-show`，或让脚本自动检测 | 自动检测会在没有 `DISPLAY` / `WAYLAND_DISPLAY` 时切到 Agg 后端 |
| 环境**没有中文字体** | 图面文案自动退回英文并告警 | matplotlib 默认字体不含中文字形，强画会得到一排豆腐块 |

---

## 数值解析的容错范围

训练日志里的数字写法五花八门，解析时统一按下面的规则处理，不需要预处理：

| 写法 | 例子 | 解析结果 |
|---|---|---|
| 千位分隔符 | `tps: 12,703.0` | `12703.0` |
| 科学计数法 | `lr: 3.0e-4`、`lm loss: 9.993670E+00` | `0.0003`、`9.99367` |
| 百分号 | `mfu: 41.29%` | 值 `41.29`，单位 `%` |
| 单位后缀 | `memory: 61.41GiB` | 值 `61.41`，单位 `GiB` |
| 复合写法 | `memory: 61.41GiB(76.76%)` | 值 `61.41`，单位 `GiB` |
| 非有限值 | `loss: nan`、`grad_norm: -inf` | `nan`、`-inf`，单独计数 |
| 分隔符变体 | `loss: 1.2`、`loss=1.2`、`loss 1.2` | 都能抓 |
| 字段名含空格 | `lm loss: 9.99`、`grad norm: 2.86` | 都能抓（别名内部空白放宽为 `\s+`） |

匹配时两侧都有边界断言：`step` 不会命中 `steps_per_second`，`loss` 不会命中 `loss_scale`。同一指标配多个别名时按「从具体到笼统」依次尝试，第一个命中的胜出。

**NaN 感知统计**是刻意的设计：NaN 不参与 min/max/mean/std，但单独计数；某指标全是 NaN 时统计量返回 NaN 而不是 0。把 NaN 当 0 处理会让「指标全坏了」在报表上长得像「指标恒为 0」，这是最危险的一种静默失败。

---

## 脚本

| 脚本 | 职责 |
|---|---|
| `scripts/parse_training_log.py` | 解析、对齐、统计、JSON 输出。既是 CLI 也是库：`parse_log_file()` / `compare_parsed_logs()` 是两个主入口 |
| `scripts/plot_training_log.py` | 绘图。复用上面的解析能力，不重复实现任何解析逻辑 |

依赖：解析脚本只用标准库（`--profile` 需要 PyYAML）；绘图脚本需要 matplotlib。

退出码两个脚本共用一套：`0` 成功、`1` 未预期错误、`2` 命令行用法错误、`3` 没解析到指标、`4` 没有共同 step、`5` 输入文件或 profile 有问题；`6`（绘图或写文件失败）只有绘图脚本会返回。在自动化流水线里应当按退出码分支，而不是去 grep 输出文本。

---

## 知识来源

- 日志解析、ANSI 转义处理、流式读取：通用文本处理工程实践
- 绘图：[matplotlib 官方文档](https://matplotlib.org/stable/)，子图布局用 `pyplot.subplots`，NaN 自动断线是 matplotlib 的原生行为
- NaN 语义：IEEE 754，以及 NumPy 的 `nan*` 系列函数在全 NaN 输入时返回 NaN 并告警的约定
- 日志字段命名样例：[torchtitan](https://github.com/pytorch/torchtitan) 与 [Megatron-LM](https://github.com/NVIDIA/Megatron-LM) 的公开训练日志格式
- 本文件及配套脚本为独立编写，不含任何来自第三方受限许可仓库的代码或文本，依据见 [`docs/LICENSE-CONSTRAINT.md`](../../docs/LICENSE-CONSTRAINT.md)
