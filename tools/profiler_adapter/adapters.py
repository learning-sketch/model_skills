"""把各家 profiler 的产物转成统一 kernel 事件。

新增一种 profiler 时，实现一个 (路径, 配置) -> Iterator[KernelEvent] 的函数并注册到
ADAPTERS 即可，不需要改动任何分析代码。
"""

from __future__ import annotations

import csv
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterator, Sequence

from .schema import COUNTER_KEYS, EventCategory, KernelEvent

# 通信 kernel 的默认识别模式。各家通信库的 kernel 命名不同，
# 你的库命名不在此列时，通过 profile 的 collective.library 或 --comm-pattern 补充。
DEFAULT_COMMUNICATION_PATTERNS: tuple[str, ...] = (
    r"\bnccl",
    r"\bncclDevKernel",
    r"\brccl",
    r"\ballreduce\b",
    r"\ballgather\b",
    r"\breducescatter\b",
    r"\balltoall\b",
)


@dataclass
class AdapterConfig:
    """适配器的可调参数，大部分可以从硬件 profile 推导。"""

    # kernel 表的列名映射：统一字段名 -> 你的 profiler 列名
    column_mapping: dict[str, str] = field(default_factory=dict)
    # category 词表：统一分类 -> 你的 profiler 中属于该分类的取值
    category_values: dict[str, list[str]] = field(default_factory=dict)
    # 硬件计数器：统一语义键 -> 你的 profiler 指标列名
    counter_mapping: dict[str, str] = field(default_factory=dict)
    # 计数器原始值的量纲。百分比需除以 100 才能落到 0~1
    counter_scale: float = 1.0
    communication_patterns: Sequence[str] = DEFAULT_COMMUNICATION_PATTERNS
    # chrome trace 中主机侧的 op 记录是调用方记录而非设备执行，默认不纳入
    include_host_events: bool = False

    @classmethod
    def from_profile(cls, profile: dict[str, Any]) -> AdapterConfig:
        """从硬件 profile YAML 构造配置。字段缺失时回落到默认值。"""
        profiler_section = (profile.get("profiler") or {}) if profile else {}
        raw_columns = profiler_section.get("kernel_table_columns") or {}
        column_mapping = {
            unified: actual
            for unified, actual in raw_columns.items()
            if actual
        }

        raw_categories = profiler_section.get("task_type_values") or {}
        category_values = {
            unified: [str(value).lower() for value in (values or [])]
            for unified, values in raw_categories.items()
        }

        counters_section = profiler_section.get("hardware_counters") or {}
        counter_mapping = {}
        for unified_key, profile_key in (
            ("compute_util", "compute_utilization"),
            ("memory_util", "memory_utilization"),
            ("tensor_core_util", "tensor_core_utilization"),
        ):
            actual = counters_section.get(profile_key)
            if actual:
                counter_mapping[unified_key] = actual

        patterns = list(DEFAULT_COMMUNICATION_PATTERNS)
        library = ((profile.get("collective") or {}).get("library")) if profile else None
        if library:
            patterns.append(re.escape(str(library).lower()))

        return cls(
            column_mapping=column_mapping,
            category_values=category_values,
            counter_mapping=counter_mapping,
            communication_patterns=tuple(patterns),
        )

    def compiled_communication_matcher(self) -> re.Pattern[str]:
        return re.compile("|".join(self.communication_patterns), re.IGNORECASE)


def _coerce_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    if isinstance(value, (int, float)):
        return float(value)
    # 容忍千位分隔符、百分号、单位后缀
    cleaned = re.sub(r"[,%\s]", "", str(value))
    cleaned = re.sub(r"(us|ms|ns|GiB|MiB|KB|MB|GB)$", "", cleaned, flags=re.IGNORECASE)
    try:
        return float(cleaned)
    except ValueError:
        return None


def _coerce_int(value: Any) -> int | None:
    parsed = _coerce_float(value)
    return int(parsed) if parsed is not None else None


# ---------------------------------------------------------------------------
# chrome trace（torch.profiler export_chrome_trace 产物）
# ---------------------------------------------------------------------------

# 设备侧事件的 cat 取值。cpu_op 是主机侧的算子下发记录，不是设备执行，默认排除。
_CHROME_DEVICE_CATEGORIES = {
    "kernel": EventCategory.COMPUTE,
    "gpu_memcpy": EventCategory.MEMORY,
    "gpu_memset": EventCategory.MEMORY,
    "gpu_user_annotation": EventCategory.COMPUTE,
}

_STEP_ANNOTATION_RE = re.compile(r"ProfilerStep#(\d+)")


def _extract_step_windows(trace_events: list[dict[str, Any]]) -> list[tuple[float, int]]:
    """从 ProfilerStep#N 标注事件抽出 (起始时间, step 号)，按时间升序。

    返回起始点而非闭区间，是因为设备 kernel 相对主机侧标注有异步滞后，
    用"落在哪个 step 的起始之后"来归属比用闭区间包含更稳。
    """
    windows: list[tuple[float, int]] = []
    for event in trace_events:
        if event.get("ph") != "X":
            continue
        match = _STEP_ANNOTATION_RE.search(str(event.get("name", "")))
        if not match:
            continue
        start = _coerce_float(event.get("ts"))
        if start is None:
            continue
        windows.append((start, int(match.group(1))))
    windows.sort()
    return windows


def _locate_step(step_windows: list[tuple[float, int]], timestamp: float) -> int | None:
    """二分查找 timestamp 落在哪个 step。"""
    if not step_windows:
        return None
    low, high = 0, len(step_windows) - 1
    if timestamp < step_windows[0][0]:
        return None
    while low < high:
        mid = (low + high + 1) // 2
        if step_windows[mid][0] <= timestamp:
            low = mid
        else:
            high = mid - 1
    return step_windows[low][1]


def from_chrome_trace(path: Path, config: AdapterConfig) -> Iterator[KernelEvent]:
    """解析 torch.profiler 导出的 chrome trace。

    只产出设备侧事件。主机侧的 cpu_op 是算子下发记录而非设备执行，
    把它当成设备事件会让时间统计重复计算。
    """
    with path.open("r", encoding="utf-8") as handle:
        try:
            payload = json.load(handle)
        except json.JSONDecodeError as error:
            raise ValueError(
                f"{path} 不是合法 JSON。profiler 异常退出会产生截断的 trace，"
                f"确认采集过程正常结束。原始错误: {error}"
            ) from error

    trace_events = payload.get("traceEvents")
    if not isinstance(trace_events, list):
        raise ValueError(f"{path} 缺少 traceEvents 数组，不像 chrome trace 格式")

    step_windows = _extract_step_windows(trace_events)
    communication_matcher = config.compiled_communication_matcher()

    for event in trace_events:
        if event.get("ph") != "X":
            continue
        raw_category = str(event.get("cat", "")).lower()

        if raw_category in _CHROME_DEVICE_CATEGORIES:
            category = _CHROME_DEVICE_CATEGORIES[raw_category]
        elif raw_category == "cpu_op" and config.include_host_events:
            category = EventCategory.HOST
        else:
            continue

        start = _coerce_float(event.get("ts"))
        duration = _coerce_float(event.get("dur"))
        if start is None or duration is None:
            continue

        name = str(event.get("name", "<unnamed>"))
        # 通信 kernel 在 chrome trace 里的 cat 仍是 kernel，只能靠名字识别
        if category is EventCategory.COMPUTE and communication_matcher.search(name):
            category = EventCategory.COMMUNICATION

        args = event.get("args") or {}
        stream_id = args.get("stream")
        if stream_id is None:
            # 没有 stream 字段时退回用线程号区分，至少能把并发关系保住
            stream_id = event.get("tid", "unknown")

        block_dim = None
        grid = args.get("grid")
        block = args.get("block")
        if isinstance(grid, list) and isinstance(block, list):
            try:
                grid_size = 1
                for dimension in grid:
                    grid_size *= int(dimension)
                block_size = 1
                for dimension in block:
                    block_size *= int(dimension)
                block_dim = grid_size * block_size
            except (TypeError, ValueError):
                block_dim = None

        input_shapes = args.get("Input Dims") or args.get("Input dims")

        yield KernelEvent(
            name=name,
            start_us=start,
            duration_us=duration,
            stream_id=str(stream_id),
            step_id=_locate_step(step_windows, start),
            device_id=str(args["device"]) if "device" in args else None,
            category=category,
            block_dim=block_dim,
            input_shapes=json.dumps(input_shapes, ensure_ascii=False) if input_shapes else None,
        )


# ---------------------------------------------------------------------------
# 通用 CSV（列名映射由 profile 提供）
# ---------------------------------------------------------------------------

def from_csv(path: Path, config: AdapterConfig) -> Iterator[KernelEvent]:
    """解析 kernel 表 CSV。列名映射来自 profile.profiler.kernel_table_columns。"""
    mapping = config.column_mapping
    required = ("name", "duration_us", "start_time_us", "stream_id")
    missing = [key for key in required if not mapping.get(key)]
    if missing:
        raise ValueError(
            f"profile 的 profiler.kernel_table_columns 缺少必需的列名映射: {missing}。"
            "必须至少提供 name / duration_us / start_time_us / stream_id 四项"
        )

    # 反查表：你的列名属于哪个统一分类
    value_to_category: dict[str, EventCategory] = {}
    for unified_name, raw_values in config.category_values.items():
        try:
            category = EventCategory(unified_name)
        except ValueError:
            if unified_name == "host_fallback":
                category = EventCategory.HOST
            else:
                continue
        for raw_value in raw_values:
            value_to_category[raw_value.lower()] = category

    communication_matcher = config.compiled_communication_matcher()

    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"{path} 没有表头")
        absent = [
            mapping[key] for key in required
            if mapping[key] not in reader.fieldnames
        ]
        if absent:
            raise ValueError(
                f"{path} 的表头里找不到这些列: {absent}。"
                f"实际表头: {reader.fieldnames}"
            )

        for row in reader:
            start = _coerce_float(row.get(mapping["start_time_us"]))
            duration = _coerce_float(row.get(mapping["duration_us"]))
            if start is None or duration is None:
                continue

            name = str(row.get(mapping["name"], "<unnamed>"))

            category = EventCategory.UNKNOWN
            task_type_column = mapping.get("task_type")
            if task_type_column:
                raw_value = str(row.get(task_type_column, "")).lower()
                category = value_to_category.get(raw_value, EventCategory.UNKNOWN)
            if category in (EventCategory.UNKNOWN, EventCategory.COMPUTE):
                if communication_matcher.search(name):
                    category = EventCategory.COMMUNICATION
                elif category is EventCategory.UNKNOWN and task_type_column is None:
                    # 没有分类列时，默认按计算处理，并由校验报告提示覆盖率不足
                    category = EventCategory.COMPUTE

            counters: dict[str, float] = {}
            for unified_key in COUNTER_KEYS:
                column = config.counter_mapping.get(unified_key)
                if not column:
                    continue
                value = _coerce_float(row.get(column))
                if value is None:
                    continue
                scaled = value * config.counter_scale
                # 容忍 profiler 给的是 0~100 的百分比
                if scaled > 1.0:
                    scaled = scaled / 100.0
                counters[unified_key] = min(max(scaled, 0.0), 1.0)

            yield KernelEvent(
                name=name,
                start_us=start,
                duration_us=duration,
                stream_id=str(row.get(mapping["stream_id"], "unknown")),
                step_id=_coerce_int(row.get(mapping["step_id"])) if mapping.get("step_id") else None,
                device_id=str(row[mapping["device_id"]]) if mapping.get("device_id") in row else None,
                category=category,
                block_dim=_coerce_int(row.get(mapping["block_dim"])) if mapping.get("block_dim") else None,
                input_shapes=row.get(mapping["input_shapes"]) if mapping.get("input_shapes") else None,
                counters=counters,
            )


def from_jsonl(path: Path, config: AdapterConfig) -> Iterator[KernelEvent]:
    """读取已经归一化过的数据，用于单独跑校验。"""
    from .schema import read_jsonl

    del config
    yield from read_jsonl(path)


AdapterFunc = Callable[[Path, AdapterConfig], Iterator[KernelEvent]]

ADAPTERS: dict[str, AdapterFunc] = {
    "chrome-trace": from_chrome_trace,
    "csv": from_csv,
    "jsonl": from_jsonl,
}


def detect_format(path: Path) -> str:
    """按扩展名与文件内容猜测格式。"""
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return "csv"
    if suffix == ".jsonl":
        return "jsonl"
    if suffix == ".json":
        # chrome trace 与 jsonl 都可能是 .json，看第一个非空字符区分
        with path.open("r", encoding="utf-8") as handle:
            head = handle.read(4096).lstrip()
        return "chrome-trace" if head.startswith("{") and '"traceEvents"' in head else "jsonl"
    raise ValueError(
        f"无法从 {path.name} 判断格式，请用 --format 显式指定。"
        f"可选: {sorted(ADAPTERS)}"
    )
