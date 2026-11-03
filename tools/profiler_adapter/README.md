# Profiler 适配层

## 为什么需要这一层

性能数据是多个能力领域的共同输入：

- 性能拆解（哪一层慢、慢在哪个算子）
- 多流优化的效果验证（切了流有没有真的并行）
- 访存瓶颈识别（哪些算子值得做预取或融合）
- 图结构分析（哪些子图重复出现、值得融合）

这些分析本身与硬件无关——算的都是时间区间的交叠、聚合与分组。但每家 profiler 的产物格式都不一样：列名不同、时间单位不同、kernel 分类词表不同、有没有硬件计数器也不同。

如果每个分析脚本各自去适配各家格式，适配逻辑会被复制很多遍，而且加一种新 profiler 就要改所有脚本。

这一层的做法是：**定义一个统一的 kernel 事件 schema，所有分析只认这个 schema；每种 profiler 写一个适配器把产物转成它。** 加新 profiler 只需加一个适配器。

```
你的 profiler 产物 ──┐
torch.profiler trace ─┼── 适配器 ──→ 统一 kernel 事件 ──→ 各类分析
通用 CSV ────────────┘
```

---

## 统一 Schema

一条记录代表设备上的一次 kernel 执行。

### 必需字段

| 字段 | 类型 | 含义 |
|------|------|------|
| `name` | str | kernel 名称。用于分组与模式匹配，要求同一个算子在不同次执行中名称稳定 |
| `start_us` | float | 起始时间戳，微秒。同一份数据内需同源可比，绝对原点不限 |
| `duration_us` | float | 执行时长，微秒 |
| `stream_id` | str | 执行流标识。单流场景填固定值即可 |

### 可选字段

| 字段 | 类型 | 缺失的影响 |
|------|------|-----------|
| `step_id` | int | 无法按 step 切分，只能整体统计 |
| `device_id` | str | 多卡数据无法分开看 |
| `category` | str | 无法区分计算与通信，气泡分析会把通信算进计算 |
| `block_dim` | int | 无法判断并行度是否打满 |
| `input_shapes` | str | 无法按 shape 聚类，相同算子不同 shape 会被混在一起 |
| `counters` | dict | 无法判断算子是访存受限还是计算受限，roofline 分析失效 |

### category 的取值

统一为四类。适配器负责把各家的分类词表映射过来。

| 取值 | 含义 |
|------|------|
| `compute` | 设备上的计算 kernel |
| `communication` | 集合通信或点对点通信 |
| `memory` | 显存拷贝、填充 |
| `host` | 主机侧执行的算子。这类事件的存在通常意味着某个算子缺少设备实现，是性能告警信号 |

分不清的填 `unknown`，分析侧会按保守方式处理。

### counters 的键名

硬件计数器各家命名完全不同，这里只约定**归一化后的语义键**，取值统一为 0 到 1 的比例：

| 键 | 含义 |
|----|------|
| `compute_util` | 计算单元利用率 |
| `memory_util` | 访存带宽利用率 |
| `tensor_core_util` | 张量核利用率 |

适配器从 `profile.profiler.hardware_counters` 读取你的 profiler 实际指标名，归一化后填进来。没有计数器就留空，依赖它的分析会降级为只看时间。

---

## 用法

### 转换

```bash
# torch.profiler 导出的 chrome trace
python -m profiler_adapter.normalize \
  --input trace.json --format chrome-trace \
  --output kernels.jsonl

# 通用 CSV，列名映射从 profile 读
python -m profiler_adapter.normalize \
  --input kernel_table.csv --format csv \
  --profile ../../hardware-profiles/my-chip.yaml \
  --output kernels.jsonl

# 不确定格式时让它自己判断
python -m profiler_adapter.normalize --input <路径> --format auto --output kernels.jsonl
```

### 校验

转换完先检查数据够不够后续分析用：

```bash
python -m profiler_adapter.normalize --input kernels.jsonl --format jsonl --validate-only
```

会报告记录数、时间跨度、字段完整度，以及**哪些分析因为字段缺失做不了**。

### 接新 profiler

实现一个函数，签名为 `(路径, 配置) -> Iterator[KernelEvent]`，注册到 `adapters.ADAPTERS` 即可。参考 `adapters.py` 里已有的两个实现。

---

## 已知的适配难点

**时间戳原点。** chrome trace 的 `ts` 是进程启动以来的微秒数，有些 profiler 给的是绝对时间或时钟周期数。适配器只保证同一份数据内部可比，不做跨文件对齐。要对比两次运行，用相对时间或 step 边界对齐，不要直接比绝对值。

**step 边界。** torch.profiler 的 chrome trace 没有原生的 step 列，需要从 `ProfilerStep#N` 标注事件的时间区间反推。如果采集时没有调用 `prof.step()`，这个信息就丢了，只能整体分析。

**通信 kernel 的识别。** chrome trace 里通信 kernel 的 category 仍是 `kernel`，要靠名称前缀二次识别。适配器用一组可配置的名称模式来判断，默认模式覆盖常见的通信库命名，你的通信库命名不同的话需要在 profile 里补充。

**同一个算子的多条记录。** 有些 profiler 会把一个逻辑算子拆成多条 kernel 记录（比如通信算子同时有一条汇总记录和若干条实际执行记录）。直接求和会重复计算。适配器提供了去重钩子，但去重规则依赖具体 profiler 的语义，需要你确认后配置。
