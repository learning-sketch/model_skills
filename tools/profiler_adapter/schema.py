"""统一 kernel 事件 schema 及其校验。

所有性能分析只消费本模块定义的 KernelEvent，不直接读各家 profiler 的原始产物。
schema 的设计约定见同目录 README.md。
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Iterable, Iterator


class EventCategory(str, Enum):
    """kernel 事件的四种归一化分类。

    各家 profiler 的分类词表不同，适配器负责映射到这里。
    """

    COMPUTE = "compute"
    COMMUNICATION = "communication"
    MEMORY = "memory"
    # 主机侧执行。出现这类事件通常意味着算子缺少设备实现，是性能告警信号。
    HOST = "host"
    UNKNOWN = "unknown"


# 硬件计数器的归一化语义键。取值统一为 0.0~1.0 的比例。
COUNTER_KEYS = ("compute_util", "memory_util", "tensor_core_util")


@dataclass(slots=True)
class KernelEvent:
    """设备上的一次 kernel 执行。

    时间戳只保证同一份数据内部可比，不保证跨文件对齐。
    """

    name: str
    start_us: float
    duration_us: float
    stream_id: str

    step_id: int | None = None
    device_id: str | None = None
    category: EventCategory = EventCategory.UNKNOWN
    block_dim: int | None = None
    input_shapes: str | None = None
    counters: dict[str, float] = field(default_factory=dict)

    @property
    def end_us(self) -> float:
        return self.start_us + self.duration_us

    def __post_init__(self) -> None:
        if self.duration_us < 0:
            raise ValueError(f"kernel {self.name!r} 的 duration_us 为负: {self.duration_us}")
        if isinstance(self.category, str):
            # 容忍适配器直接传字符串；无法识别的归到 UNKNOWN 而不是抛错，
            # 因为 category 缺失只影响部分分析，不应阻断整条流水线。
            try:
                self.category = EventCategory(self.category)
            except ValueError:
                self.category = EventCategory.UNKNOWN
        for key, value in self.counters.items():
            if not 0.0 <= value <= 1.0:
                raise ValueError(
                    f"kernel {self.name!r} 的计数器 {key} 取值 {value} 不在 0~1，"
                    "适配器需先做归一化"
                )

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["category"] = self.category.value
        return payload

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> KernelEvent:
        known_fields = {
            "name", "start_us", "duration_us", "stream_id",
            "step_id", "device_id", "category", "block_dim",
            "input_shapes", "counters",
        }
        unexpected = set(payload) - known_fields
        if unexpected:
            raise ValueError(f"未知字段: {sorted(unexpected)}")
        return cls(**payload)


def write_jsonl(events: Iterable[KernelEvent], output_path: Path) -> int:
    """逐条写出，避免大 trace 在内存里堆成一个大 list。"""
    written = 0
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for event in events:
            handle.write(json.dumps(event.to_dict(), ensure_ascii=False) + "\n")
            written += 1
    return written


def read_jsonl(input_path: Path) -> Iterator[KernelEvent]:
    with input_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield KernelEvent.from_dict(json.loads(line))
            except (json.JSONDecodeError, ValueError, TypeError) as error:
                raise ValueError(f"{input_path}:{line_number} 解析失败: {error}") from error


# ---------------------------------------------------------------------------
# 校验
# ---------------------------------------------------------------------------

# 各类分析对字段的依赖。校验时据此告诉用户"数据到手了能做什么、不能做什么"。
ANALYSIS_REQUIREMENTS: dict[str, tuple[str, ...]] = {
    "按 step 切分": ("step_id",),
    "多流重叠率": ("stream_id",),
    "计算与通信分离统计": ("category",),
    "气泡（空隙）分析": ("stream_id",),
    "访存受限判定": ("counters.memory_util",),
    "并行度是否打满": ("block_dim",),
    "按 shape 聚类": ("input_shapes",),
    "多卡分别统计": ("device_id",),
}


@dataclass
class ValidationReport:
    total_events: int
    time_span_us: float
    distinct_streams: int
    distinct_steps: int
    field_coverage: dict[str, float]
    category_breakdown: dict[str, int]
    available_analyses: list[str]
    blocked_analyses: dict[str, list[str]]
    warnings: list[str]

    def render(self) -> str:
        lines = [
            "统一 kernel 事件校验报告",
            "=" * 60,
            f"事件总数      : {self.total_events}",
            f"时间跨度      : {self.time_span_us / 1000:.2f} ms",
            f"流数量        : {self.distinct_streams}",
            f"step 数量     : {self.distinct_steps if self.distinct_steps else '未标注'}",
            "",
            "字段完整度",
            "-" * 60,
        ]
        for field_name, coverage in sorted(self.field_coverage.items()):
            bar = "#" * int(coverage * 20)
            lines.append(f"  {field_name:<16} {coverage * 100:5.1f}%  {bar}")

        lines += ["", "事件分类", "-" * 60]
        for category, count in sorted(
            self.category_breakdown.items(), key=lambda item: -item[1]
        ):
            share = count / self.total_events * 100 if self.total_events else 0.0
            lines.append(f"  {category:<16} {count:>8}  ({share:5.1f}%)")

        lines += ["", "可做的分析", "-" * 60]
        for analysis in self.available_analyses:
            lines.append(f"  [可用] {analysis}")
        for analysis, missing in self.blocked_analyses.items():
            lines.append(f"  [缺失] {analysis}  ← 需要 {', '.join(missing)}")

        if self.warnings:
            lines += ["", "注意", "-" * 60]
            lines += [f"  {warning}" for warning in self.warnings]

        return "\n".join(lines)


def _has_field(event: KernelEvent, field_path: str) -> bool:
    if field_path.startswith("counters."):
        return field_path.split(".", 1)[1] in event.counters
    value = getattr(event, field_path, None)
    if field_path == "category":
        return value is not EventCategory.UNKNOWN
    return value is not None


def validate(events: Iterable[KernelEvent]) -> ValidationReport:
    """统计字段完整度，并据此判断哪些分析可做、哪些做不了。"""
    tracked_fields = (
        "step_id", "device_id", "category", "block_dim",
        "input_shapes", "counters.compute_util", "counters.memory_util",
        "counters.tensor_core_util",
    )
    present_counts = dict.fromkeys(tracked_fields, 0)
    category_breakdown: dict[str, int] = {}
    streams: set[str] = set()
    steps: set[int] = set()
    total = 0
    earliest = float("inf")
    latest = float("-inf")
    zero_duration = 0

    for event in events:
        total += 1
        streams.add(event.stream_id)
        if event.step_id is not None:
            steps.add(event.step_id)
        earliest = min(earliest, event.start_us)
        latest = max(latest, event.end_us)
        if event.duration_us == 0:
            zero_duration += 1
        category_breakdown[event.category.value] = (
            category_breakdown.get(event.category.value, 0) + 1
        )
        for field_path in tracked_fields:
            if _has_field(event, field_path):
                present_counts[field_path] += 1

    if total == 0:
        return ValidationReport(
            total_events=0, time_span_us=0.0, distinct_streams=0, distinct_steps=0,
            field_coverage={}, category_breakdown={}, available_analyses=[],
            blocked_analyses={name: list(reqs) for name, reqs in ANALYSIS_REQUIREMENTS.items()},
            warnings=["没有解析到任何事件，检查输入文件与 --format 是否匹配"],
        )

    coverage = {name: count / total for name, count in present_counts.items()}
    # stream_id 是必需字段，解析成功即存在
    coverage["stream_id"] = 1.0

    available: list[str] = []
    blocked: dict[str, list[str]] = {}
    for analysis, required in ANALYSIS_REQUIREMENTS.items():
        # 覆盖率低于一半视为不可用：部分缺失的数据做出来的结论会误导人
        missing = [name for name in required if coverage.get(name, 0.0) < 0.5]
        if missing:
            blocked[analysis] = missing
        else:
            available.append(analysis)

    warnings: list[str] = []
    if zero_duration:
        warnings.append(
            f"有 {zero_duration} 条事件 duration 为 0，可能是时间精度不足或解析错位"
        )
    if len(streams) == 1:
        warnings.append("只有一个流，多流相关分析无意义")
    if not steps:
        warnings.append(
            "没有 step 标注，无法按 step 切分。采集时需要调用 profiler 的 step() 并保留标注事件"
        )
    if coverage.get("counters.memory_util", 0.0) < 0.5:
        warnings.append(
            "缺少访存利用率计数器，无法区分访存受限与计算受限。"
            "若你的 profiler 支持，需在采集时开启硬件计数器"
        )

    return ValidationReport(
        total_events=total,
        time_span_us=latest - earliest,
        distinct_streams=len(streams),
        distinct_steps=len(steps),
        field_coverage=coverage,
        category_breakdown=category_breakdown,
        available_analyses=available,
        blocked_analyses=blocked,
        warnings=warnings,
    )
