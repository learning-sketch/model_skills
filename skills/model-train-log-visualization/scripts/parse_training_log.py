#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""训练日志指标解析器。

从纯文本训练日志里按行提取指标（step / loss / grad_norm / ...），并支持把两份
日志按 step 对齐后计算误差。设计上既是命令行工具，也是库——``plot_training_log.py``
直接复用本模块的解析能力。

核心特性：

* **字段名不写死**。内置一套常见命名作为默认别名表；可用 ``--metric-key`` 覆盖；
  也可以用 ``--profile`` 从硬件 profile 的
  ``training_framework.log_format.metric_keys`` 读取。优先级：
  内置默认 < profile < 命令行。
* **文本容错**。剥离 ANSI 颜色转义；按 ``\\r`` 切分进度条覆写产生的片段；
  数字解析容忍千位分隔符、科学计数法、百分号与单位后缀（``12.34GiB`` / ``1,234.5`` /
  ``3.0e-4`` / ``45.6%``）。
* **NaN 感知统计**。NaN 不参与均值/极值计算，但会被单独计数；若某指标全是 NaN，
  统计量返回 NaN 而不是 0。
* **缺失告警**。请求了但日志里一次都没出现的指标会产生告警，不静默跳过。
* **双日志对齐**。按 step 求交集，计算**有符号**的绝对误差与相对误差；
  交集为空时报错退出，避免生成误导性的对比结论。
* **流式读取**。逐行处理，不做 ``readlines()``，大日志不会把内存打满。

命令行示例::

    python parse_training_log.py train.log
    python parse_training_log.py train.log --compare baseline.log
    python parse_training_log.py train.log --metric-key loss=lm_loss --json
    python parse_training_log.py train.log --profile hardware-profiles/my-chip.yaml
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterator, List, Mapping, Optional, Sequence, TextIO, Tuple

__all__ = [
    "DEFAULT_METRIC_ALIASES",
    "MANDATORY_METRICS",
    "LogComparison",
    "MetricComparison",
    "MetricPoint",
    "MetricStatistics",
    "NoCommonStepError",
    "ParsedLog",
    "build_metric_aliases",
    "compare_parsed_logs",
    "compute_statistics",
    "load_metric_aliases_from_profile",
    "parse_log_file",
    "parse_metric_key_overrides",
    "parse_numeric_token",
    "strip_ansi_escape_sequences",
]


# ------------------------------------------------------------------------------
# 退出码
# ------------------------------------------------------------------------------
EXIT_SUCCESS = 0
EXIT_GENERAL_ERROR = 1
EXIT_USAGE_ERROR = 2  # argparse 自身使用
EXIT_NO_METRIC_PARSED = 3
EXIT_NO_COMMON_STEP = 4
EXIT_INPUT_ERROR = 5


# ------------------------------------------------------------------------------
# 默认别名表
# ------------------------------------------------------------------------------
# 键是 skills 内部使用的统一指标名（不要改），值是日志里可能出现的字面写法。
# 同一指标的多个别名按「从具体到笼统」排列：第一个在行内命中的别名胜出，
# 这样 "lm_loss" 不会被更笼统的 "loss" 抢先匹配到别的数值上。
DEFAULT_METRIC_ALIASES: Dict[str, Tuple[str, ...]] = {
    "step": ("global_step", "global step", "iteration", "iter", "step"),
    "loss": ("global_avg_loss", "train_loss", "training_loss", "lm_loss", "lm loss", "loss"),
    "grad_norm": ("global_grad_norm", "grad_norm", "grad norm", "grad-norm", "gnorm"),
    "learning_rate": ("learning_rate", "learning rate", "lr"),
    "memory": ("max_memory_reserved", "max_memory", "memory", "mem"),
    "throughput": ("tokens_per_second", "samples_per_second", "throughput", "tps", "tgs"),
    "tflops": ("tflops", "tflop/s", "tfs"),
    "mfu": ("model_flops_utilization", "mfu"),
    "step_time": ("elapsed_time_per_step", "iteration_time", "step_time", "time_per_step"),
    "loss_scale": ("loss_scale", "loss scale", "grad_scale"),
}

# 可视化约定要求必绘的指标；解析阶段也默认总是尝试提取它们。
MANDATORY_METRICS: Tuple[str, ...] = ("loss", "grad_norm")

# step 是横轴，任何情况下都必须解析。
STEP_METRIC = "step"


# ------------------------------------------------------------------------------
# 文本与数字处理
# ------------------------------------------------------------------------------
_ANSI_ESCAPE_PATTERN = re.compile(r"\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])")

# 数值主体：允许符号、千位分隔符（, 或 _）、小数、纯小数（.5）、科学计数法。
_NUMBER_BODY_PATTERN = r"[+-]?(?:(?:\d{1,3}(?:[,_]\d{3})+|\d+)(?:\.\d+)?|\.\d+)(?:[eE][+-]?\d+)?"
# 非有限值：nan / inf / infinity，带可选符号。
_NON_FINITE_PATTERN = r"[+-]?(?:nan|inf(?:inity)?)"
_NUMERIC_TOKEN_PATTERN = rf"(?:{_NUMBER_BODY_PATTERN}|{_NON_FINITE_PATTERN})"
# 紧跟数值、中间无空白的单位后缀：百分号，或字母开头的短串（GiB、GB/s、tok/s、ms）。
_UNIT_SUFFIX_PATTERN = r"(?:%|[A-Za-z][A-Za-z/]{0,11})"


def strip_ansi_escape_sequences(text: str) -> str:
    """移除 ANSI 颜色/光标控制序列。

    Args:
        text: 原始文本。

    Returns:
        去掉转义序列后的文本。
    """
    return _ANSI_ESCAPE_PATTERN.sub("", text)


def parse_numeric_token(token: str) -> float:
    """把日志里抓到的数值串转成 float。

    容忍千位分隔符（``1,234`` / ``1_234``）、科学计数法、以及 ``nan`` / ``inf``
    这类非有限值字面量。单位后缀与百分号由正则单独捕获，不会出现在这里。

    Args:
        token: 数值串，例如 ``"1,234.5"`` / ``"3.0e-4"`` / ``"-inf"``。

    Returns:
        解析出的浮点数。

    Raises:
        ValueError: 字符串无法解析为数值。
    """
    cleaned = token.strip().replace(",", "").replace("_", "")
    if not cleaned:
        raise ValueError("空数值串")
    return float(cleaned)


def _build_alias_pattern(alias: str) -> re.Pattern[str]:
    """为单个别名编译匹配正则。

    别名内部的空白被放宽为 ``\\s+``，这样 ``"lm loss"`` 能匹配 ``"lm   loss"``。
    前后加边界断言，避免 ``step`` 命中 ``steps_per_second``、``loss`` 命中
    ``loss_scale``。分隔符（``:`` 或 ``=``）可选，以兼容 ``loss 1.23`` 这种写法。

    Args:
        alias: 日志中的字面字段名，例如 ``"grad_norm"``。

    Returns:
        编译好的正则，group(1) 是数值串，group(2) 是可选单位。
    """
    escaped = r"\s+".join(re.escape(part) for part in alias.split())
    pattern = (
        rf"(?<![A-Za-z0-9_.]){escaped}(?![A-Za-z0-9_])"
        rf"\s*[:=]?\s*({_NUMERIC_TOKEN_PATTERN})({_UNIT_SUFFIX_PATTERN})?"
    )
    return re.compile(pattern, re.IGNORECASE)


# ------------------------------------------------------------------------------
# 数据结构
# ------------------------------------------------------------------------------
@dataclass(frozen=True)
class MetricPoint:
    """单个 step 上抓到的一组指标。"""

    step: int
    values: Dict[str, float]
    units: Dict[str, Optional[str]]
    line_number: int

    def to_dict(self) -> Dict[str, object]:
        """转为可序列化字典。"""
        return {
            "step": self.step,
            "line_number": self.line_number,
            "values": dict(self.values),
            "units": {name: unit for name, unit in self.units.items() if unit},
        }


@dataclass(frozen=True)
class MetricStatistics:
    """NaN 感知的单指标统计量。

    NaN 不参与 min/max/mean/std，但会被 ``nan_count`` 记下来。若所有取值都是
    NaN，则 ``minimum`` / ``maximum`` / ``mean`` / ``standard_deviation`` 全部
    返回 NaN——不是 0，以免把「指标全坏了」误读成「指标恒为 0」。
    ``first`` / ``last`` 保留原始取值（可能就是 NaN），因为训练里「第几步开始变
    NaN」本身是关键信息。
    """

    count: int
    nan_count: int
    infinite_count: int
    first: float
    last: float
    minimum: float
    maximum: float
    mean: float
    standard_deviation: float

    def to_dict(self) -> Dict[str, object]:
        """转为可序列化字典。"""
        return {
            "count": self.count,
            "nan_count": self.nan_count,
            "infinite_count": self.infinite_count,
            "first": self.first,
            "last": self.last,
            "minimum": self.minimum,
            "maximum": self.maximum,
            "mean": self.mean,
            "standard_deviation": self.standard_deviation,
        }


@dataclass
class ParsedLog:
    """一份日志的解析结果。"""

    source: str
    points: List[MetricPoint] = field(default_factory=list)
    requested_metrics: List[str] = field(default_factory=list)
    alias_hits: Dict[str, str] = field(default_factory=dict)
    units: Dict[str, Optional[str]] = field(default_factory=dict)
    total_lines: int = 0
    matched_lines: int = 0
    duplicate_steps: int = 0
    metric_lines_without_step: int = 0
    truncated: bool = False
    warnings: List[str] = field(default_factory=list)

    @property
    def steps(self) -> List[int]:
        """升序的 step 列表。"""
        return [point.step for point in self.points]

    @property
    def present_metrics(self) -> List[str]:
        """至少出现过一次的指标名（按 requested_metrics 的顺序）。

        不含 ``step``——它是横轴而不是曲线，单独存在 ``MetricPoint.step`` 上。
        """
        seen = {name for point in self.points for name in point.values}
        return [name for name in self.requested_metrics if name != STEP_METRIC and name in seen]

    @property
    def missing_metrics(self) -> List[str]:
        """请求了但一次都没抓到的指标名（不含 ``step``）。"""
        present = set(self.present_metrics)
        return [
            name
            for name in self.requested_metrics
            if name != STEP_METRIC and name not in present
        ]

    def series(self, metric: str) -> Tuple[List[int], List[float]]:
        """取某指标的 (step, value) 序列，只包含该指标存在的点。

        Args:
            metric: 统一指标名。

        Returns:
            ``(steps, values)`` 两个等长列表。
        """
        steps: List[int] = []
        values: List[float] = []
        for point in self.points:
            if metric in point.values:
                steps.append(point.step)
                values.append(point.values[metric])
        return steps, values

    def value_map(self, metric: str) -> Dict[int, float]:
        """取某指标的 ``{step: value}`` 映射。

        Args:
            metric: 统一指标名。

        Returns:
            step 到取值的字典。
        """
        return {point.step: point.values[metric] for point in self.points if metric in point.values}

    def statistics(self, metric: str) -> MetricStatistics:
        """计算某指标的 NaN 感知统计量。

        Args:
            metric: 统一指标名。

        Returns:
            统计量；指标不存在时各项为 0 / NaN。
        """
        _, values = self.series(metric)
        return compute_statistics(values)

    def coverage(self, metric: str) -> float:
        """某指标在所有 step 上的覆盖率。

        Args:
            metric: 统一指标名。

        Returns:
            0.0~1.0 的比例；没有任何 step 时返回 0.0。
        """
        if not self.points:
            return 0.0
        hit = sum(1 for point in self.points if metric in point.values)
        return hit / len(self.points)

    def to_dict(self, *, include_points: bool = True) -> Dict[str, object]:
        """转为可序列化字典。

        Args:
            include_points: 是否包含逐点数据。大日志下关掉可大幅缩小输出。

        Returns:
            结构化字典。
        """
        payload: Dict[str, object] = {
            "source": self.source,
            "total_lines": self.total_lines,
            "matched_lines": self.matched_lines,
            "step_count": len(self.points),
            "step_range": [self.points[0].step, self.points[-1].step] if self.points else None,
            "duplicate_steps": self.duplicate_steps,
            "metric_lines_without_step": self.metric_lines_without_step,
            "truncated": self.truncated,
            "requested_metrics": list(self.requested_metrics),
            "present_metrics": self.present_metrics,
            "missing_metrics": self.missing_metrics,
            "alias_hits": dict(self.alias_hits),
            "units": {name: unit for name, unit in self.units.items() if unit},
            "coverage": {name: self.coverage(name) for name in self.present_metrics},
            "statistics": {name: self.statistics(name).to_dict() for name in self.present_metrics},
            "warnings": list(self.warnings),
        }
        if include_points:
            payload["points"] = [point.to_dict() for point in self.points]
        return payload


@dataclass
class MetricComparison:
    """单个指标在两份日志之间的逐 step 误差。

    绝对误差与相对误差都是**有符号**的，定义为：

    * ``absolute_error = candidate - baseline``
    * ``relative_error = (candidate - baseline) / baseline``

    基线为 0 时相对误差无定义，记为 NaN 并计入 ``undefined_relative_count``。
    """

    metric: str
    steps: List[int]
    baseline_values: List[float]
    candidate_values: List[float]
    absolute_errors: List[float]
    relative_errors: List[float]
    undefined_relative_count: int
    baseline_unit: Optional[str] = None
    candidate_unit: Optional[str] = None

    @property
    def absolute_error_statistics(self) -> MetricStatistics:
        """绝对误差的 NaN 感知统计量。"""
        return compute_statistics(self.absolute_errors)

    @property
    def relative_error_statistics(self) -> MetricStatistics:
        """相对误差的 NaN 感知统计量。"""
        return compute_statistics(self.relative_errors)

    def worst_step(self) -> Tuple[Optional[int], float]:
        """绝对误差幅值最大的 step。

        Returns:
            ``(step, signed_error)``；全为 NaN 时返回 ``(None, nan)``。
        """
        worst_step: Optional[int] = None
        worst_error = math.nan
        worst_magnitude = -1.0
        for step, error in zip(self.steps, self.absolute_errors):
            if math.isnan(error):
                continue
            magnitude = abs(error)
            if magnitude > worst_magnitude:
                worst_magnitude = magnitude
                worst_step = step
                worst_error = error
        return worst_step, worst_error

    def to_dict(self, *, include_points: bool = True) -> Dict[str, object]:
        """转为可序列化字典。

        Args:
            include_points: 是否包含逐点误差序列。

        Returns:
            结构化字典。
        """
        worst_step, worst_error = self.worst_step()
        payload: Dict[str, object] = {
            "metric": self.metric,
            "common_step_count": len(self.steps),
            "undefined_relative_count": self.undefined_relative_count,
            "baseline_unit": self.baseline_unit,
            "candidate_unit": self.candidate_unit,
            "absolute_error_statistics": self.absolute_error_statistics.to_dict(),
            "relative_error_statistics": self.relative_error_statistics.to_dict(),
            "worst_absolute_error": {"step": worst_step, "error": worst_error},
        }
        if include_points:
            payload["steps"] = list(self.steps)
            payload["baseline_values"] = list(self.baseline_values)
            payload["candidate_values"] = list(self.candidate_values)
            payload["absolute_errors"] = list(self.absolute_errors)
            payload["relative_errors"] = list(self.relative_errors)
        return payload


@dataclass
class LogComparison:
    """两份日志按 step 对齐后的整体对比结果。"""

    baseline: ParsedLog
    candidate: ParsedLog
    common_steps: List[int]
    baseline_only_steps: List[int]
    candidate_only_steps: List[int]
    metrics: Dict[str, MetricComparison]
    warnings: List[str] = field(default_factory=list)

    def to_dict(self, *, include_points: bool = True) -> Dict[str, object]:
        """转为可序列化字典。

        Args:
            include_points: 是否包含逐点数据。

        Returns:
            结构化字典。
        """
        return {
            "baseline": self.baseline.to_dict(include_points=include_points),
            "candidate": self.candidate.to_dict(include_points=include_points),
            "common_step_count": len(self.common_steps),
            "common_step_range": (
                [self.common_steps[0], self.common_steps[-1]] if self.common_steps else None
            ),
            "baseline_only_step_count": len(self.baseline_only_steps),
            "candidate_only_step_count": len(self.candidate_only_steps),
            "baseline_only_step_sample": self.baseline_only_steps[:10],
            "candidate_only_step_sample": self.candidate_only_steps[:10],
            "metrics": {
                name: comparison.to_dict(include_points=include_points)
                for name, comparison in self.metrics.items()
            },
            "warnings": list(self.warnings),
        }


class NoCommonStepError(RuntimeError):
    """两份日志的 step 没有交集。

    这种情况下任何对比图都是误导性的（两条曲线画在同一张图上但来自不相交的区
    间），因此直接报错而不是降级出图。
    """


# ------------------------------------------------------------------------------
# 统计
# ------------------------------------------------------------------------------
def compute_statistics(values: Sequence[float]) -> MetricStatistics:
    """计算 NaN 感知的统计量。

    NaN 被排除在 min/max/mean/std 之外；±inf 参与计算（发散本身是有效信号），
    但单独计数。全 NaN 或空输入时各聚合量返回 NaN。

    Args:
        values: 原始取值序列。

    Returns:
        统计量对象。
    """
    total = len(values)
    nan_count = sum(1 for value in values if math.isnan(value))
    infinite_count = sum(1 for value in values if math.isinf(value))
    usable = [value for value in values if not math.isnan(value)]

    if not usable:
        return MetricStatistics(
            count=total,
            nan_count=nan_count,
            infinite_count=infinite_count,
            first=values[0] if values else math.nan,
            last=values[-1] if values else math.nan,
            minimum=math.nan,
            maximum=math.nan,
            mean=math.nan,
            standard_deviation=math.nan,
        )

    mean = math.fsum(usable) / len(usable) if all(math.isfinite(v) for v in usable) else _loose_mean(usable)
    if len(usable) >= 2 and math.isfinite(mean):
        variance = math.fsum((value - mean) ** 2 for value in usable if math.isfinite(value))
        finite_count = sum(1 for value in usable if math.isfinite(value))
        standard_deviation = math.sqrt(variance / (finite_count - 1)) if finite_count >= 2 else math.nan
    else:
        standard_deviation = math.nan

    return MetricStatistics(
        count=total,
        nan_count=nan_count,
        infinite_count=infinite_count,
        first=values[0],
        last=values[-1],
        minimum=min(usable),
        maximum=max(usable),
        mean=mean,
        standard_deviation=standard_deviation,
    )


def _loose_mean(values: Sequence[float]) -> float:
    """含 ±inf 时的均值。

    ``math.fsum`` 遇到 ``+inf`` 与 ``-inf`` 同时存在会抛 ValueError，这里退回
    普通加法让结果自然变成 NaN/inf。

    Args:
        values: 非 NaN 的取值序列。

    Returns:
        均值，可能是 ±inf 或 NaN。
    """
    total = 0.0
    for value in values:
        total += value
    return total / len(values)


# ------------------------------------------------------------------------------
# 别名解析
# ------------------------------------------------------------------------------
def build_metric_aliases(
    *,
    profile_aliases: Optional[Mapping[str, Sequence[str]]] = None,
    override_aliases: Optional[Mapping[str, Sequence[str]]] = None,
    selected_metrics: Optional[Sequence[str]] = None,
) -> Dict[str, Tuple[str, ...]]:
    """按优先级合成最终的别名表。

    优先级：内置默认 < profile < 命令行覆盖。同一指标被更高优先级来源定义时，
    整组别名被**替换**而不是追加，这样用户能彻底摆脱默认命名的干扰。

    Args:
        profile_aliases: 从 YAML profile 读到的别名。
        override_aliases: 命令行 ``--metric-key`` 提供的别名。
        selected_metrics: 只保留这些指标（``step`` 与必绘指标始终保留）。

    Returns:
        ``{统一指标名: (别名, ...)}``。
    """
    aliases: Dict[str, Tuple[str, ...]] = {
        name: tuple(values) for name, values in DEFAULT_METRIC_ALIASES.items()
    }
    for source in (profile_aliases, override_aliases):
        if not source:
            continue
        for name, values in source.items():
            normalized = tuple(_normalize_alias(value) for value in values if _normalize_alias(value))
            if normalized:
                aliases[name] = normalized

    if selected_metrics is not None:
        keep = set(selected_metrics) | {STEP_METRIC} | set(MANDATORY_METRICS)
        keep |= set(override_aliases or {})
        aliases = {name: values for name, values in aliases.items() if name in keep}

    if STEP_METRIC not in aliases:
        aliases[STEP_METRIC] = DEFAULT_METRIC_ALIASES[STEP_METRIC]
    return aliases


def _normalize_alias(alias: str) -> str:
    """规范化别名字面量。

    profile 里的写法通常带尾随分隔符（``"step:"``），这里统一剥掉，因为分隔符由
    匹配正则负责。

    Args:
        alias: 原始别名。

    Returns:
        规范化后的别名；不可用时返回空串。
    """
    return alias.strip().rstrip(":=").strip()


def load_metric_aliases_from_profile(
    profile_path: Path,
) -> Tuple[Dict[str, List[str]], List[str]]:
    """从硬件 profile 读取日志字段名。

    读取路径为 ``training_framework.log_format.metric_keys``。字段状态不是
    ``verified`` 时产生告警，提醒先在真机日志上核对。

    Args:
        profile_path: profile YAML 路径。

    Returns:
        ``(别名表, 告警列表)``。profile 中为 null 的项会被跳过。

    Raises:
        RuntimeError: PyYAML 未安装、文件不存在或内容不是映射。
    """
    try:
        import yaml  # 可选依赖：只有用 --profile 时才需要
    except ImportError as error:
        raise RuntimeError(
            "读取 --profile 需要 PyYAML，请先 `pip install pyyaml`，"
            "或改用 --metric-key 在命令行指定字段名"
        ) from error

    if not profile_path.is_file():
        raise RuntimeError(f"profile 文件不存在: {profile_path}")

    try:
        document = yaml.safe_load(profile_path.read_text(encoding="utf-8"))
    except yaml.YAMLError as error:
        raise RuntimeError(f"profile 不是合法 YAML: {profile_path}: {error}") from error

    if not isinstance(document, Mapping):
        raise RuntimeError(f"profile 顶层不是映射: {profile_path}")

    warnings: List[str] = []
    framework = document.get("training_framework")
    if not isinstance(framework, Mapping):
        warnings.append(f"profile 缺少 training_framework 小节，继续使用内置默认字段名: {profile_path}")
        return {}, warnings

    status = framework.get("_status")
    if status != "verified":
        warnings.append(
            f"profile 的 training_framework._status = {status!r}（非 verified），"
            "日志字段名可能与真机不符，建议先用一份真实日志核对"
        )

    log_format = framework.get("log_format")
    if not isinstance(log_format, Mapping):
        warnings.append("profile 缺少 training_framework.log_format，继续使用内置默认字段名")
        return {}, warnings

    metric_keys = log_format.get("metric_keys")
    if not isinstance(metric_keys, Mapping):
        warnings.append("profile 缺少 training_framework.log_format.metric_keys，继续使用内置默认字段名")
        return {}, warnings

    aliases: Dict[str, List[str]] = {}
    null_fields: List[str] = []
    for name, value in metric_keys.items():
        if value is None:
            null_fields.append(str(name))
            continue
        if isinstance(value, str):
            candidates = [value]
        elif isinstance(value, Sequence):
            candidates = [str(item) for item in value]
        else:
            warnings.append(f"profile 中 metric_keys.{name} 类型异常（{type(value).__name__}），已忽略")
            continue
        normalized = [_normalize_alias(item) for item in candidates]
        normalized = [item for item in normalized if item]
        if normalized:
            aliases[str(name)] = normalized

    if null_fields:
        warnings.append(
            "profile 中以下 metric_keys 为 null，退回内置默认字段名: " + ", ".join(sorted(null_fields))
        )
    return aliases, warnings


# ------------------------------------------------------------------------------
# 解析
# ------------------------------------------------------------------------------
def _iter_log_fragments(handle: TextIO) -> Iterator[Tuple[int, str]]:
    """流式产出日志片段。

    逐行读取（不做 readlines），并按 ``\\r`` 再切一刀——进度条覆写会把很多次输出
    塞进同一「行」里，不切开会漏掉绝大部分指标。

    Args:
        handle: 已打开的文本句柄。

    Yields:
        ``(行号, 去掉 ANSI 的片段)``。
    """
    for line_number, raw_line in enumerate(handle, start=1):
        cleaned = strip_ansi_escape_sequences(raw_line)
        if "\r" in cleaned:
            for fragment in cleaned.split("\r"):
                if fragment.strip():
                    yield line_number, fragment
        elif cleaned.strip():
            yield line_number, cleaned


def parse_log_file(
    log_path: Path,
    metric_aliases: Mapping[str, Sequence[str]],
    *,
    max_lines: Optional[int] = None,
    on_duplicate_step: str = "last",
    synthesize_steps: bool = False,
    encoding: str = "utf-8",
) -> ParsedLog:
    """解析一份训练日志。

    Args:
        log_path: 日志文件路径。
        metric_aliases: ``{统一指标名: (别名, ...)}``。
        max_lines: 最多读取的行数，``None`` 表示不限。用于给超大日志设上限。
        on_duplicate_step: 同一 step 重复出现时的处理方式。
            ``"last"`` 保留最新（默认）、``"first"`` 保留最早、
            ``"merge"`` 按指标逐项用最新值覆盖。
        synthesize_steps: 某行抓到了指标但没抓到 step 时，是否用递增序号补一个
            合成 step。默认 False——宁可告警，也不要造出一根假横轴。
        encoding: 文件编码。解码错误一律 ``replace``，避免二进制噪声中断解析。

    Returns:
        解析结果。

    Raises:
        RuntimeError: 路径不存在、不是文件，或读取失败。
    """
    if on_duplicate_step not in {"last", "first", "merge"}:
        raise ValueError(f"不支持的 on_duplicate_step: {on_duplicate_step!r}")
    if not log_path.exists():
        raise RuntimeError(f"日志文件不存在: {log_path}")
    if not log_path.is_file():
        raise RuntimeError(f"日志路径不是文件: {log_path}")

    compiled: Dict[str, List[Tuple[str, re.Pattern[str]]]] = {
        name: [(alias, _build_alias_pattern(alias)) for alias in aliases]
        for name, aliases in metric_aliases.items()
    }
    value_metrics = [name for name in compiled if name != STEP_METRIC]

    result = ParsedLog(source=str(log_path), requested_metrics=list(compiled))
    collected: Dict[int, MetricPoint] = {}
    synthetic_step = 0
    non_integer_step_reported = False

    try:
        handle = log_path.open("r", encoding=encoding, errors="replace")
    except OSError as error:
        raise RuntimeError(f"无法读取日志文件 {log_path}: {error}") from error

    with handle:
        for line_number, fragment in _iter_log_fragments(handle):
            if max_lines is not None and line_number > max_lines:
                result.truncated = True
                result.warnings.append(f"已达到 --max-lines={max_lines} 上限，日志尾部未解析")
                break
            result.total_lines = line_number

            values: Dict[str, float] = {}
            units: Dict[str, Optional[str]] = {}
            for name in value_metrics:
                extracted = _extract_metric(fragment, compiled[name], result)
                if extracted is None:
                    continue
                alias, value, unit = extracted
                values[name] = value
                units[name] = unit
                result.alias_hits.setdefault(name, alias)
                _record_unit(result, name, unit)

            step_extracted = _extract_metric(fragment, compiled[STEP_METRIC], result)
            if step_extracted is not None:
                result.alias_hits.setdefault(STEP_METRIC, step_extracted[0])

            if not values:
                continue

            if step_extracted is not None:
                raw_step = step_extracted[1]
                if math.isnan(raw_step) or math.isinf(raw_step):
                    result.warnings.append(f"第 {line_number} 行的 step 不是有限值，已跳过该行")
                    continue
                if not float(raw_step).is_integer() and not non_integer_step_reported:
                    result.warnings.append("检测到非整数 step，已向下取整；若日志用 epoch 做横轴请改用 --metric-key step=<字段名>")
                    non_integer_step_reported = True
                step = int(raw_step)
            elif synthesize_steps:
                synthetic_step += 1
                step = synthetic_step
            else:
                result.metric_lines_without_step += 1
                continue

            result.matched_lines += 1
            point = MetricPoint(step=step, values=values, units=units, line_number=line_number)
            if step in collected:
                result.duplicate_steps += 1
                collected[step] = _resolve_duplicate(collected[step], point, on_duplicate_step)
            else:
                collected[step] = point

    result.points = [collected[step] for step in sorted(collected)]

    if result.metric_lines_without_step:
        result.warnings.append(
            f"有 {result.metric_lines_without_step} 行抓到了指标但没抓到 step，已忽略。"
            "若日志确实不打印 step，可加 --synthesize-steps 用行序号充当横轴"
        )
    if result.duplicate_steps:
        result.warnings.append(
            f"有 {result.duplicate_steps} 个重复 step，按 --on-duplicate-step={on_duplicate_step} 处理"
        )
    for missing in result.missing_metrics:
        result.warnings.append(
            f"指标 {missing!r} 在日志中一次都没出现（尝试过的别名: "
            + ", ".join(metric_aliases[missing])
            + "）。若字段名不同请用 --metric-key 指定"
        )
    for name in result.present_metrics:
        coverage = result.coverage(name)
        if 0.0 < coverage < 0.95:
            result.warnings.append(
                f"指标 {name!r} 只在 {coverage:.1%} 的 step 上出现，曲线会有断点"
            )
    if not result.points:
        result.warnings.append("没有解析到任何 step，请检查字段名是否匹配（--show-aliases 可打印当前别名表）")
    return result


def _extract_metric(
    fragment: str,
    patterns: Sequence[Tuple[str, re.Pattern[str]]],
    result: ParsedLog,
) -> Optional[Tuple[str, float, Optional[str]]]:
    """在一个片段里按别名顺序抓取第一个命中的数值。

    Args:
        fragment: 已去 ANSI 的文本片段。
        patterns: ``(别名, 正则)`` 列表，按优先级排列。
        result: 解析结果对象，用于登记解析失败的告警。

    Returns:
        ``(命中的别名, 数值, 单位)``；没命中返回 None。
    """
    for alias, pattern in patterns:
        match = pattern.search(fragment)
        if match is None:
            continue
        try:
            value = parse_numeric_token(match.group(1))
        except ValueError:
            if len(result.warnings) < 50:
                result.warnings.append(f"别名 {alias!r} 命中了无法解析的数值: {match.group(1)!r}")
            continue
        unit = match.group(2)
        return alias, value, unit.strip() if unit else None
    return None


def _record_unit(result: ParsedLog, metric: str, unit: Optional[str]) -> None:
    """登记指标单位，发现同一指标出现两种单位时告警。"""
    if unit is None:
        return
    known = result.units.get(metric)
    if known is None:
        result.units[metric] = unit
    elif known.lower() != unit.lower():
        message = f"指标 {metric!r} 出现了不一致的单位: {known!r} 与 {unit!r}，数值不可直接比较"
        if message not in result.warnings:
            result.warnings.append(message)


def _resolve_duplicate(existing: MetricPoint, incoming: MetricPoint, policy: str) -> MetricPoint:
    """按策略合并同一 step 的两个采样点。"""
    if policy == "first":
        return existing
    if policy == "last":
        return incoming
    merged_values = dict(existing.values)
    merged_values.update(incoming.values)
    merged_units = dict(existing.units)
    merged_units.update(incoming.units)
    return MetricPoint(
        step=incoming.step,
        values=merged_values,
        units=merged_units,
        line_number=incoming.line_number,
    )


# ------------------------------------------------------------------------------
# 对比
# ------------------------------------------------------------------------------
def compare_parsed_logs(
    baseline: ParsedLog,
    candidate: ParsedLog,
    *,
    metrics: Optional[Sequence[str]] = None,
) -> LogComparison:
    """按 step 对齐两份日志并计算误差。

    Args:
        baseline: 基线日志的解析结果。
        candidate: 待对比日志的解析结果。
        metrics: 参与对比的指标；``None`` 表示两份日志都出现过的指标取交集。

    Returns:
        对比结果。

    Raises:
        NoCommonStepError: 两份日志的 step 没有交集。
    """
    baseline_steps = set(baseline.steps)
    candidate_steps = set(candidate.steps)
    common_steps = sorted(baseline_steps & candidate_steps)

    if not common_steps:
        raise NoCommonStepError(
            f"两份日志没有共同 step，无法对齐：\n"
            f"  基线 {baseline.source}: {_describe_step_range(baseline)}\n"
            f"  对比 {candidate.source}: {_describe_step_range(candidate)}\n"
            "常见原因：两次训练的 step 计数起点不同、日志字段名被解析成了别的字段、"
            "或其中一份其实是续训日志。请先确认 --metric-key step=<字段名> 是否正确。"
        )

    if metrics is None:
        shared = [name for name in baseline.present_metrics if name in set(candidate.present_metrics)]
    else:
        shared = list(metrics)

    warnings: List[str] = []
    comparisons: Dict[str, MetricComparison] = {}
    for metric in shared:
        baseline_map = baseline.value_map(metric)
        candidate_map = candidate.value_map(metric)
        steps = [step for step in common_steps if step in baseline_map and step in candidate_map]
        if not steps:
            warnings.append(f"指标 {metric!r} 在共同 step 上没有成对取值，已跳过对比")
            continue

        baseline_values = [baseline_map[step] for step in steps]
        candidate_values = [candidate_map[step] for step in steps]
        absolute_errors = [
            candidate_value - baseline_value
            for baseline_value, candidate_value in zip(baseline_values, candidate_values)
        ]
        relative_errors: List[float] = []
        undefined_relative_count = 0
        for baseline_value, absolute_error in zip(baseline_values, absolute_errors):
            if baseline_value == 0.0 or math.isnan(baseline_value) or math.isnan(absolute_error):
                relative_errors.append(math.nan)
                if baseline_value == 0.0:
                    undefined_relative_count += 1
                continue
            relative_errors.append(absolute_error / baseline_value)

        baseline_unit = baseline.units.get(metric)
        candidate_unit = candidate.units.get(metric)
        if baseline_unit and candidate_unit and baseline_unit.lower() != candidate_unit.lower():
            warnings.append(
                f"指标 {metric!r} 两侧单位不同（{baseline_unit} vs {candidate_unit}），误差值没有物理意义"
            )

        comparisons[metric] = MetricComparison(
            metric=metric,
            steps=steps,
            baseline_values=baseline_values,
            candidate_values=candidate_values,
            absolute_errors=absolute_errors,
            relative_errors=relative_errors,
            undefined_relative_count=undefined_relative_count,
            baseline_unit=baseline_unit,
            candidate_unit=candidate_unit,
        )

    for metric in sorted(set(baseline.present_metrics) ^ set(candidate.present_metrics)):
        owner = baseline.source if metric in baseline.present_metrics else candidate.source
        warnings.append(f"指标 {metric!r} 只在 {owner} 中出现，无法对比")

    overlap_ratio = len(common_steps) / max(len(baseline_steps | candidate_steps), 1)
    if overlap_ratio < 0.5:
        warnings.append(
            f"共同 step 只占两份日志 step 并集的 {overlap_ratio:.1%}，对比区间很窄，结论请谨慎"
        )

    return LogComparison(
        baseline=baseline,
        candidate=candidate,
        common_steps=common_steps,
        baseline_only_steps=sorted(baseline_steps - candidate_steps),
        candidate_only_steps=sorted(candidate_steps - baseline_steps),
        metrics=comparisons,
        warnings=warnings,
    )


def _describe_step_range(parsed: ParsedLog) -> str:
    """生成人读的 step 区间描述。"""
    if not parsed.points:
        return "未解析到任何 step"
    steps = parsed.steps
    return f"{len(steps)} 个 step，区间 {steps[0]}..{steps[-1]}"


# ------------------------------------------------------------------------------
# 报告渲染
# ------------------------------------------------------------------------------
def _format_number(value: float, width: int = 12) -> str:
    """把浮点数格式化成定宽字符串，NaN/inf 原样显示。"""
    if math.isnan(value):
        return "nan".rjust(width)
    if math.isinf(value):
        return ("inf" if value > 0 else "-inf").rjust(width)
    if value != 0 and (abs(value) < 1e-4 or abs(value) >= 1e7):
        return f"{value:.4e}".rjust(width)
    return f"{value:.6g}".rjust(width)


def render_log_report(parsed: ParsedLog) -> str:
    """渲染单份日志的文本报告。

    Args:
        parsed: 解析结果。

    Returns:
        多行文本。
    """
    lines: List[str] = []
    lines.append(f"日志: {parsed.source}")
    step_range = f"{parsed.steps[0]}..{parsed.steps[-1]}" if parsed.points else "-"
    lines.append(
        f"  读取行数 {parsed.total_lines} | 命中 step 的行 {parsed.matched_lines} | "
        f"有效 step {len(parsed.points)} (区间 {step_range}) | 重复 step {parsed.duplicate_steps}"
    )
    if parsed.alias_hits:
        hits = ", ".join(f"{name}<-{alias}" for name, alias in sorted(parsed.alias_hits.items()))
        lines.append(f"  命中的字段名: {hits}")

    if parsed.present_metrics:
        header = (
            f"  {'指标':<16}{'点数':>7}{'NaN':>6}{'Inf':>6}{'覆盖率':>9}"
            f"{'首值':>14}{'末值':>14}{'最小':>14}{'最大':>14}{'均值':>14}{'标准差':>14}  单位"
        )
        lines.append(header)
        for metric in parsed.present_metrics:
            stats = parsed.statistics(metric)
            unit = parsed.units.get(metric) or ""
            lines.append(
                f"  {metric:<16}{stats.count:>7}{stats.nan_count:>6}{stats.infinite_count:>6}"
                f"{parsed.coverage(metric):>9.1%}"
                f"{_format_number(stats.first, 14)}{_format_number(stats.last, 14)}"
                f"{_format_number(stats.minimum, 14)}{_format_number(stats.maximum, 14)}"
                f"{_format_number(stats.mean, 14)}{_format_number(stats.standard_deviation, 14)}  {unit}"
            )
    else:
        lines.append("  未解析到任何指标")

    if parsed.warnings:
        lines.append("  告警:")
        for warning in parsed.warnings:
            lines.append(f"    - {warning}")
    return "\n".join(lines)


def render_comparison_report(comparison: LogComparison) -> str:
    """渲染双日志对比的文本报告。

    Args:
        comparison: 对比结果。

    Returns:
        多行文本。
    """
    lines: List[str] = []
    lines.append("=" * 100)
    lines.append("双日志对比（误差定义: 对比值 - 基线值，带符号）")
    lines.append(f"  基线: {comparison.baseline.source}")
    lines.append(f"  对比: {comparison.candidate.source}")
    common = comparison.common_steps
    lines.append(
        f"  共同 step {len(common)} 个，区间 {common[0]}..{common[-1]} | "
        f"仅基线有 {len(comparison.baseline_only_steps)} 个 | "
        f"仅对比有 {len(comparison.candidate_only_steps)} 个"
    )

    for metric, metric_comparison in comparison.metrics.items():
        absolute_statistics = metric_comparison.absolute_error_statistics
        relative_statistics = metric_comparison.relative_error_statistics
        worst_step, worst_error = metric_comparison.worst_step()
        lines.append(f"  指标 {metric}（{len(metric_comparison.steps)} 个成对点）")
        lines.append(
            f"    绝对误差  均值 {_format_number(absolute_statistics.mean)}"
            f"  最小 {_format_number(absolute_statistics.minimum)}"
            f"  最大 {_format_number(absolute_statistics.maximum)}"
            f"  NaN {absolute_statistics.nan_count}"
        )
        lines.append(
            f"    相对误差  均值 {_format_number(relative_statistics.mean)}"
            f"  最小 {_format_number(relative_statistics.minimum)}"
            f"  最大 {_format_number(relative_statistics.maximum)}"
            f"  基线为零而无定义 {metric_comparison.undefined_relative_count}"
        )
        if worst_step is not None:
            lines.append(
                f"    最大偏离出现在 step {worst_step}，偏离量 {_format_number(worst_error).strip()}"
            )

    if comparison.warnings:
        lines.append("  告警:")
        for warning in comparison.warnings:
            lines.append(f"    - {warning}")
    return "\n".join(lines)


# ------------------------------------------------------------------------------
# JSON 输出
# ------------------------------------------------------------------------------
def _sanitize_non_finite(payload: object) -> object:
    """递归把 NaN/±inf 换成 None，使输出是严格合法的 JSON。

    非有限值的数量不会因此丢失：统计量里有 ``nan_count`` 与 ``infinite_count``。

    Args:
        payload: 任意可序列化结构。

    Returns:
        替换后的结构。
    """
    if isinstance(payload, float):
        return None if (math.isnan(payload) or math.isinf(payload)) else payload
    if isinstance(payload, dict):
        return {key: _sanitize_non_finite(value) for key, value in payload.items()}
    if isinstance(payload, (list, tuple)):
        return [_sanitize_non_finite(item) for item in payload]
    return payload


def _dump_json(payload: Dict[str, object], non_finite_policy: str) -> str:
    """序列化 JSON。

    Args:
        payload: 待序列化的结构。
        non_finite_policy: ``"null"`` 把 NaN/inf 换成 null（严格 JSON）；
            ``"literal"`` 保留 Python 的 ``NaN`` / ``Infinity`` 字面量（非标准
            JSON，但肉眼更好读）。

    Returns:
        JSON 文本。
    """
    if non_finite_policy == "null":
        return json.dumps(_sanitize_non_finite(payload), ensure_ascii=False, indent=2)
    return json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=True)


# ------------------------------------------------------------------------------
# 命令行
# ------------------------------------------------------------------------------
def parse_metric_key_overrides(raw_overrides: Sequence[str]) -> Dict[str, List[str]]:
    """解析 ``--metric-key name=alias1,alias2``。

    Args:
        raw_overrides: 原始命令行取值列表。

    Returns:
        ``{统一指标名: [别名, ...]}``。

    Raises:
        ValueError: 格式不是 ``name=alias``。
    """
    overrides: Dict[str, List[str]] = {}
    for raw in raw_overrides:
        if "=" not in raw:
            raise ValueError(f"--metric-key 需要 name=alias 形式，收到: {raw!r}")
        name, _, alias_text = raw.partition("=")
        name = name.strip()
        aliases = [_normalize_alias(item) for item in alias_text.split(",")]
        aliases = [item for item in aliases if item]
        if not name or not aliases:
            raise ValueError(f"--metric-key 取值不完整: {raw!r}")
        overrides.setdefault(name, []).extend(aliases)
    return overrides


def build_argument_parser() -> argparse.ArgumentParser:
    """构造命令行解析器。

    Returns:
        配置好的 ``ArgumentParser``。
    """
    parser = argparse.ArgumentParser(
        prog="parse_training_log.py",
        description="解析训练日志中的指标，可选地与第二份日志按 step 对齐并计算误差。",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "退出码:\n"
            f"  {EXIT_SUCCESS}  成功\n"
            f"  {EXIT_GENERAL_ERROR}  未预期错误\n"
            f"  {EXIT_USAGE_ERROR}  命令行用法错误\n"
            f"  {EXIT_NO_METRIC_PARSED}  日志里没解析到任何指标\n"
            f"  {EXIT_NO_COMMON_STEP}  两份日志没有共同 step\n"
            f"  {EXIT_INPUT_ERROR}  输入文件或 profile 有问题\n"
        ),
    )
    parser.add_argument("log", type=Path, help="主训练日志路径")
    parser.add_argument(
        "--compare",
        type=Path,
        default=None,
        metavar="LOG",
        help="对比日志路径。给定后按 step 对齐并计算有符号的绝对/相对误差",
    )
    parser.add_argument(
        "--baseline-is-compare",
        action="store_true",
        help="把 --compare 的日志当作基线（默认主日志是基线）",
    )
    parser.add_argument(
        "--profile",
        type=Path,
        default=None,
        help="硬件 profile YAML，从 training_framework.log_format.metric_keys 读字段名",
    )
    parser.add_argument(
        "--metric-key",
        action="append",
        default=[],
        metavar="NAME=ALIAS[,ALIAS...]",
        help="覆盖某指标在日志中的字段名，可重复。例: --metric-key loss=lm_loss,train_loss",
    )
    parser.add_argument(
        "--metrics",
        default=None,
        metavar="A,B,C",
        help=f"只解析这些指标（{'/'.join(MANDATORY_METRICS)} 与 step 始终包含）",
    )
    parser.add_argument(
        "--on-duplicate-step",
        choices=("last", "first", "merge"),
        default="last",
        help="同一 step 重复出现时的处理方式（默认 last，保留最新）",
    )
    parser.add_argument(
        "--synthesize-steps",
        action="store_true",
        help="日志不打印 step 时，用行序号充当横轴（默认关闭，只告警）",
    )
    parser.add_argument(
        "--max-lines",
        type=int,
        default=None,
        metavar="N",
        help="最多读取 N 行，给超大日志设上限",
    )
    parser.add_argument("--encoding", default="utf-8", help="日志文件编码（默认 utf-8）")
    parser.add_argument("--json", action="store_true", help="输出 JSON 而不是文本报告")
    parser.add_argument(
        "--omit-points",
        action="store_true",
        help="JSON 输出中省略逐点数据，只保留统计量",
    )
    parser.add_argument(
        "--json-non-finite",
        choices=("null", "literal"),
        default="null",
        help="JSON 中 NaN/inf 的写法：null（严格 JSON，默认）或 literal（NaN/Infinity）",
    )
    parser.add_argument("--output", "-o", type=Path, default=None, help="把报告写入文件而不是 stdout")
    parser.add_argument("--show-aliases", action="store_true", help="打印最终生效的别名表后退出")
    parser.add_argument("--quiet", action="store_true", help="不把告警打到 stderr")
    return parser


def _resolve_aliases(
    arguments: argparse.Namespace,
) -> Tuple[Dict[str, Tuple[str, ...]], List[str]]:
    """合成最终别名表并收集 profile 侧告警。"""
    profile_aliases: Dict[str, List[str]] = {}
    warnings: List[str] = []
    if arguments.profile is not None:
        profile_aliases, warnings = load_metric_aliases_from_profile(arguments.profile)

    override_aliases = parse_metric_key_overrides(arguments.metric_key)
    selected = None
    if arguments.metrics:
        selected = [item.strip() for item in arguments.metrics.split(",") if item.strip()]

    aliases = build_metric_aliases(
        profile_aliases=profile_aliases,
        override_aliases=override_aliases,
        selected_metrics=selected,
    )
    if selected:
        unknown = [name for name in selected if name not in aliases]
        if unknown:
            warnings.append(
                "以下指标既不在内置别名表也没有 --metric-key 定义，已忽略: " + ", ".join(unknown)
            )
    return aliases, warnings


def main(argv: Optional[Sequence[str]] = None) -> int:
    """命令行入口。

    Args:
        argv: 参数列表，``None`` 表示用 ``sys.argv[1:]``。

    Returns:
        退出码。
    """
    parser = build_argument_parser()
    arguments = parser.parse_args(argv)

    try:
        metric_aliases, setup_warnings = _resolve_aliases(arguments)
    except (ValueError, RuntimeError) as error:
        print(f"错误: {error}", file=sys.stderr)
        return EXIT_INPUT_ERROR

    if arguments.show_aliases:
        for name, aliases in metric_aliases.items():
            print(f"{name:<16} <- {', '.join(aliases)}")
        return EXIT_SUCCESS

    if not arguments.quiet:
        for warning in setup_warnings:
            print(f"[告警] {warning}", file=sys.stderr)

    try:
        main_log = parse_log_file(
            arguments.log,
            metric_aliases,
            max_lines=arguments.max_lines,
            on_duplicate_step=arguments.on_duplicate_step,
            synthesize_steps=arguments.synthesize_steps,
            encoding=arguments.encoding,
        )
        compare_log = None
        if arguments.compare is not None:
            compare_log = parse_log_file(
                arguments.compare,
                metric_aliases,
                max_lines=arguments.max_lines,
                on_duplicate_step=arguments.on_duplicate_step,
                synthesize_steps=arguments.synthesize_steps,
                encoding=arguments.encoding,
            )
    except (RuntimeError, ValueError) as error:
        print(f"错误: {error}", file=sys.stderr)
        return EXIT_INPUT_ERROR

    for parsed in (main_log, compare_log):
        if parsed is None:
            continue
        if not parsed.points:
            print(f"错误: {parsed.source} 没有解析到任何 step/指标。", file=sys.stderr)
            for warning in parsed.warnings:
                print(f"       - {warning}", file=sys.stderr)
            return EXIT_NO_METRIC_PARSED

    comparison: Optional[LogComparison] = None
    if compare_log is not None:
        baseline, candidate = (
            (compare_log, main_log) if arguments.baseline_is_compare else (main_log, compare_log)
        )
        try:
            comparison = compare_parsed_logs(baseline, candidate)
        except NoCommonStepError as error:
            print(f"错误: {error}", file=sys.stderr)
            return EXIT_NO_COMMON_STEP

    include_points = not arguments.omit_points
    if arguments.json:
        if comparison is not None:
            payload: Dict[str, object] = {
                "mode": "compare",
                "setup_warnings": setup_warnings,
                "comparison": comparison.to_dict(include_points=include_points),
            }
        else:
            payload = {
                "mode": "single",
                "setup_warnings": setup_warnings,
                "log": main_log.to_dict(include_points=include_points),
            }
        report = _dump_json(payload, arguments.json_non_finite)
    else:
        sections = [render_log_report(main_log)]
        if compare_log is not None:
            sections.append(render_log_report(compare_log))
        if comparison is not None:
            sections.append(render_comparison_report(comparison))
        report = "\n\n".join(sections)

    if arguments.output is not None:
        try:
            arguments.output.parent.mkdir(parents=True, exist_ok=True)
            arguments.output.write_text(report + "\n", encoding="utf-8")
        except OSError as error:
            print(f"错误: 无法写入 {arguments.output}: {error}", file=sys.stderr)
            return EXIT_INPUT_ERROR
        if not arguments.quiet:
            print(f"报告已写入 {arguments.output}", file=sys.stderr)
    else:
        print(report)

    return EXIT_SUCCESS


if __name__ == "__main__":
    sys.exit(main())
