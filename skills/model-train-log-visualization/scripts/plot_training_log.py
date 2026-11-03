#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""训练日志曲线绘制。

复用 ``parse_training_log`` 的解析能力，把指标画成以 **step 为统一横轴**的曲线图。

约定：

* 横轴恒为 step。任何指标都按 step 对齐，不用墙钟时间、不用行号（除非显式
  ``--synthesize-steps``）。
* ``loss`` 与 ``grad_norm`` **必绘**。即使日志里没有，也会画一个占位面板并告警，
  而不是悄悄少一张图——图少了一张比图上写着「没找到」更容易被忽略。
* 给了 ``--compare`` 时，每个指标额外画**有符号的绝对误差**与**相对误差**两幅图，
  排成「曲线叠加 | 绝对误差 | 相对误差」三列。
* 两份日志没有共同 step 时直接报错退出，不出图。

命令行示例::

    python plot_training_log.py train.log --no-show
    python plot_training_log.py train.log --compare baseline.log -o diff.png --no-show
    python plot_training_log.py train.log --metrics memory,throughput --all-metrics
"""

from __future__ import annotations

import argparse
import math
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from parse_training_log import (  # noqa: E402  （需要先补 sys.path 才能导入同目录模块）
    EXIT_GENERAL_ERROR,
    EXIT_INPUT_ERROR,
    EXIT_NO_COMMON_STEP,
    EXIT_NO_METRIC_PARSED,
    EXIT_SUCCESS,
    MANDATORY_METRICS,
    LogComparison,
    NoCommonStepError,
    ParsedLog,
    build_metric_aliases,
    compare_parsed_logs,
    load_metric_aliases_from_profile,
    parse_log_file,
)
from parse_training_log import _parse_metric_key_overrides  # noqa: E402

EXIT_PLOT_ERROR = 6

# 每个子面板的默认尺寸（英寸）。
DEFAULT_PANEL_WIDTH = 5.6
DEFAULT_PANEL_HEIGHT = 3.4

# 常见的中日韩字体名。matplotlib 默认字体不含中文字形，找不到就退回英文标签，
# 否则图上会出现一排豆腐块。
_CJK_FONT_CANDIDATES: Tuple[str, ...] = (
    "Noto Sans CJK SC",
    "Noto Sans CJK JP",
    "Noto Sans SC",
    "Source Han Sans SC",
    "Source Han Sans CN",
    "WenQuanYi Zen Hei",
    "WenQuanYi Micro Hei",
    "Microsoft YaHei",
    "SimHei",
    "PingFang SC",
    "Heiti SC",
    "Droid Sans Fallback",
    "Arial Unicode MS",
)

# (中文, 英文) 成对的图面文案。
_TEXT: Dict[str, Tuple[str, str]] = {
    "x_axis": ("训练 step", "training step"),
    "absolute_error": ("绝对误差（对比 − 基线）", "absolute error (candidate - baseline)"),
    "relative_error": ("相对误差 (%)", "relative error (%)"),
    "metric_missing": ("日志中未找到该指标", "metric not found in log"),
    "metric_all_nan": ("该指标全部为 NaN", "all values are NaN"),
    "not_comparable": ("该指标无法对比\n（至少一侧缺失）", "not comparable\n(missing on one side)"),
    "first_nan": ("首个 NaN", "first NaN"),
    "nan_note": ("{count} 个 NaN / {inf} 个 Inf", "{count} NaN / {inf} Inf"),
    "single_title": ("训练日志曲线", "training log curves"),
    "compare_title": ("训练日志对比", "training log comparison"),
}


class PlotLabels:
    """按字体可用性在中英文文案之间切换。"""

    def __init__(self, use_chinese: bool) -> None:
        """
        Args:
            use_chinese: 当前 matplotlib 字体是否能渲染中文。
        """
        self.use_chinese = use_chinese

    def __call__(self, key: str, **format_arguments: object) -> str:
        """取一条文案。

        Args:
            key: ``_TEXT`` 中的键。
            **format_arguments: 传给 ``str.format`` 的参数。

        Returns:
            渲染后的文案。
        """
        chinese, english = _TEXT[key]
        template = chinese if self.use_chinese else english
        return template.format(**format_arguments) if format_arguments else template


def _is_headless_environment() -> bool:
    """判断当前是否没有可用的图形显示。

    Returns:
        True 表示应该强制使用 Agg 后端。
    """
    if sys.platform.startswith("win") or sys.platform == "darwin":
        return False
    return not (os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))


def _import_matplotlib(force_headless: bool):
    """延迟导入 matplotlib，必要时先切到 Agg 后端。

    后端必须在 ``pyplot`` 导入之前设置，所以不能把 import 放在模块顶部。

    Args:
        force_headless: 是否强制 Agg 后端。

    Returns:
        ``(matplotlib 模块, pyplot 模块)``。

    Raises:
        RuntimeError: matplotlib 未安装。
    """
    try:
        import matplotlib
    except ImportError as error:
        raise RuntimeError("绘图需要 matplotlib，请先执行 `pip install matplotlib`") from error
    if force_headless:
        matplotlib.use("Agg")
    import matplotlib.pyplot as pyplot

    return matplotlib, pyplot


def configure_chinese_font(matplotlib_module) -> bool:
    """尝试启用一款能渲染中文的字体。

    Args:
        matplotlib_module: 已导入的 matplotlib 模块。

    Returns:
        True 表示找到了中文字体；False 表示图面文案需退回英文。
    """
    from matplotlib import font_manager

    available = {font.name for font in font_manager.fontManager.ttflist}
    for candidate in _CJK_FONT_CANDIDATES:
        if candidate in available:
            matplotlib_module.rcParams["font.sans-serif"] = [candidate, "DejaVu Sans"]
            matplotlib_module.rcParams["axes.unicode_minus"] = False
            return True
    return False


def _grid_shape(panel_count: int) -> Tuple[int, int]:
    """为单日志模式挑一个方正的子图网格。

    Args:
        panel_count: 面板数量。

    Returns:
        ``(行数, 列数)``。
    """
    if panel_count <= 1:
        return 1, 1
    if panel_count <= 4:
        columns = 2
    elif panel_count <= 9:
        columns = 3
    else:
        columns = 4
    rows = math.ceil(panel_count / columns)
    return rows, columns


def _axis_label(metric: str, parsed: ParsedLog) -> str:
    """生成带单位的纵轴标签。"""
    unit = parsed.units.get(metric)
    return f"{metric} ({unit})" if unit else metric


def _annotate_empty(axes, message: str) -> None:
    """在空面板中央写一行说明。"""
    axes.text(
        0.5,
        0.5,
        message,
        transform=axes.transAxes,
        horizontalalignment="center",
        verticalalignment="center",
        fontsize=11,
        color="#999999",
    )
    axes.set_xticks([])
    axes.set_yticks([])


def _mark_non_finite(axes, steps: Sequence[int], values: Sequence[float], labels: PlotLabels) -> None:
    """标注 NaN / Inf：首个 NaN 画竖线，右上角写总数。

    loss 从第几步开始变 NaN 是定位发散点的关键信息，不能只在终端里提一句。
    """
    nan_steps = [step for step, value in zip(steps, values) if math.isnan(value)]
    inf_count = sum(1 for value in values if math.isinf(value))
    if not nan_steps and not inf_count:
        return
    if nan_steps:
        axes.axvline(nan_steps[0], color="#d62728", linestyle=":", linewidth=1.2)
        axes.annotate(
            f"{labels('first_nan')} @ {nan_steps[0]}",
            xy=(nan_steps[0], 0.02),
            xycoords=("data", "axes fraction"),
            color="#d62728",
            fontsize=8,
            rotation=90,
            verticalalignment="bottom",
        )
    axes.text(
        0.98,
        0.95,
        labels("nan_note", count=len(nan_steps), inf=inf_count),
        transform=axes.transAxes,
        horizontalalignment="right",
        verticalalignment="top",
        fontsize=8,
        color="#d62728",
    )


def _mask_non_finite(values: Sequence[float]) -> List[float]:
    """把 ±Inf 也换成 NaN，供 matplotlib 绘图使用。

    不能把非有限点直接删掉：那样 matplotlib 会把缺口两端连成一条直线，
    视觉上「loss 一路平滑下降」，恰好把发散区间藏了起来。保留 NaN 则会断线，
    缺口一眼可见。
    """
    return [value if math.isfinite(value) else math.nan for value in values]


def _finite_values(values: Sequence[float]) -> List[float]:
    """只保留有限值，用于判空与对数轴下界检查。"""
    return [value for value in values if math.isfinite(value)]


def _marker_for(values: Sequence[float]) -> Optional[str]:
    """序列里有断点时给出点标记。

    被 NaN 夹在中间的孤立点在纯折线模式下画不出来（两侧都没得连），
    加上标记才看得见。
    """
    return "." if any(not math.isfinite(value) for value in values) else None


def _draw_single_metric(
    axes,
    parsed: ParsedLog,
    metric: str,
    labels: PlotLabels,
    *,
    log_scale: bool,
    color: str = "#1f77b4",
) -> None:
    """在一个面板上画单份日志的某个指标。"""
    steps, values = parsed.series(metric)
    if not steps:
        _annotate_empty(axes, f"{metric}\n{labels('metric_missing')}")
        return
    finite_values = _finite_values(values)
    if not finite_values:
        _annotate_empty(axes, f"{metric}\n{labels('metric_all_nan')}")
        return

    axes.plot(
        steps,
        _mask_non_finite(values),
        linewidth=1.3,
        color=color,
        marker=_marker_for(values),
        markersize=3,
    )
    axes.set_xlabel(labels("x_axis"))
    axes.set_ylabel(_axis_label(metric, parsed))
    axes.set_title(metric)
    axes.grid(True, alpha=0.3)
    if log_scale and min(finite_values) > 0:
        axes.set_yscale("log")
    _mark_non_finite(axes, steps, values, labels)


def _draw_overlay(
    axes,
    comparison: LogComparison,
    metric: str,
    labels: PlotLabels,
    baseline_label: str,
    candidate_label: str,
    *,
    log_scale: bool,
) -> None:
    """在一个面板上叠加两份日志的同一指标。"""
    drawn = False
    for parsed, label, color, style in (
        (comparison.baseline, baseline_label, "#1f77b4", "-"),
        (comparison.candidate, candidate_label, "#ff7f0e", "--"),
    ):
        steps, values = parsed.series(metric)
        if not _finite_values(values):
            continue
        axes.plot(
            steps,
            _mask_non_finite(values),
            style,
            linewidth=1.3,
            color=color,
            label=label,
            marker=_marker_for(values),
            markersize=3,
        )
        _mark_non_finite(axes, steps, values, labels)
        drawn = True

    if not drawn:
        _annotate_empty(axes, f"{metric}\n{labels('metric_missing')}")
        return
    axes.set_xlabel(labels("x_axis"))
    axes.set_ylabel(_axis_label(metric, comparison.baseline))
    axes.set_title(metric)
    axes.grid(True, alpha=0.3)
    axes.legend(fontsize=8)
    if log_scale:
        pooled = _finite_values(comparison.baseline.series(metric)[1]) + _finite_values(
            comparison.candidate.series(metric)[1]
        )
        if pooled and min(pooled) > 0:
            axes.set_yscale("log")


def _draw_error(
    axes,
    comparison: LogComparison,
    metric: str,
    labels: PlotLabels,
    *,
    relative: bool,
) -> None:
    """画误差面板。

    Args:
        axes: 目标坐标轴。
        comparison: 对比结果。
        metric: 指标名。
        labels: 文案表。
        relative: True 画相对误差（百分比），False 画绝对误差。
    """
    metric_comparison = comparison.metrics.get(metric)
    if metric_comparison is None:
        _annotate_empty(axes, labels("not_comparable"))
        return

    raw_errors = metric_comparison.relative_errors if relative else metric_comparison.absolute_errors
    scale = 100.0 if relative else 1.0
    if not _finite_values(raw_errors):
        _annotate_empty(axes, labels("metric_all_nan"))
        return

    steps = metric_comparison.steps
    scaled = [error * scale for error in _mask_non_finite(raw_errors)]
    color = "#2ca02c" if relative else "#9467bd"
    axes.plot(steps, scaled, linewidth=1.1, color=color)
    axes.axhline(0.0, color="#555555", linewidth=0.8)
    axes.fill_between(steps, 0.0, scaled, alpha=0.15, color=color)
    axes.set_xlabel(labels("x_axis"))
    axes.set_ylabel(labels("relative_error") if relative else labels("absolute_error"))
    axes.set_title(f"{metric} · {'rel. err' if relative else 'abs. err'}")
    axes.grid(True, alpha=0.3)


def _resolve_panel_metrics(
    parsed_logs: Sequence[ParsedLog],
    extra_metrics: Sequence[str],
    include_all: bool,
) -> List[str]:
    """决定要画哪些指标。

    ``loss`` 与 ``grad_norm`` 无条件在列（必绘约定），其余按用户选择或「出现过
    就画」追加。

    Args:
        parsed_logs: 已解析的日志（1 份或 2 份）。
        extra_metrics: ``--metrics`` 指定的追加指标。
        include_all: 是否把所有出现过的指标都画出来。

    Returns:
        去重且保序的指标名列表。
    """
    metrics: List[str] = list(MANDATORY_METRICS)
    for metric in extra_metrics:
        if metric not in metrics:
            metrics.append(metric)
    if include_all:
        for parsed in parsed_logs:
            for metric in parsed.present_metrics:
                if metric not in metrics:
                    metrics.append(metric)
    return metrics


def plot_single_log(
    parsed: ParsedLog,
    metrics: Sequence[str],
    output_path: Path,
    *,
    pyplot,
    labels: PlotLabels,
    panel_width: float,
    panel_height: float,
    dpi: int,
    title: Optional[str],
    log_scale_metrics: Sequence[str],
) -> Path:
    """画单份日志。

    Args:
        parsed: 解析结果。
        metrics: 要画的指标。
        output_path: PNG 输出路径。
        pyplot: 已导入的 ``matplotlib.pyplot``。
        labels: 文案表。
        panel_width: 单面板宽度（英寸）。
        panel_height: 单面板高度（英寸）。
        dpi: 输出分辨率。
        title: 总标题，None 则自动生成。
        log_scale_metrics: 纵轴取对数的指标名。

    Returns:
        实际写出的文件路径。
    """
    rows, columns = _grid_shape(len(metrics))
    figure, axes_grid = pyplot.subplots(
        rows,
        columns,
        figsize=(columns * panel_width, rows * panel_height),
        squeeze=False,
    )
    flat_axes = [axes for row in axes_grid for axes in row]

    for index, metric in enumerate(metrics):
        _draw_single_metric(
            flat_axes[index],
            parsed,
            metric,
            labels,
            log_scale=metric in log_scale_metrics,
        )
    for axes in flat_axes[len(metrics) :]:
        axes.axis("off")

    figure.suptitle(title or f"{labels('single_title')} — {Path(parsed.source).name}", fontsize=13)
    figure.tight_layout(rect=(0, 0, 1, 0.97))
    return _save_figure(figure, output_path, dpi, pyplot)


def plot_comparison(
    comparison: LogComparison,
    metrics: Sequence[str],
    output_path: Path,
    *,
    pyplot,
    labels: PlotLabels,
    panel_width: float,
    panel_height: float,
    dpi: int,
    title: Optional[str],
    baseline_label: str,
    candidate_label: str,
    log_scale_metrics: Sequence[str],
) -> Path:
    """画双日志对比：每个指标一行，三列分别是曲线叠加、绝对误差、相对误差。

    Args:
        comparison: 对比结果。
        metrics: 要画的指标。
        output_path: PNG 输出路径。
        pyplot: 已导入的 ``matplotlib.pyplot``。
        labels: 文案表。
        panel_width: 单面板宽度（英寸）。
        panel_height: 单面板高度（英寸）。
        dpi: 输出分辨率。
        title: 总标题，None 则自动生成。
        baseline_label: 基线曲线图例名。
        candidate_label: 对比曲线图例名。
        log_scale_metrics: 纵轴取对数的指标名。

    Returns:
        实际写出的文件路径。
    """
    rows = len(metrics)
    figure, axes_grid = pyplot.subplots(
        rows,
        3,
        figsize=(3 * panel_width, rows * panel_height),
        squeeze=False,
    )

    for row_index, metric in enumerate(metrics):
        _draw_overlay(
            axes_grid[row_index][0],
            comparison,
            metric,
            labels,
            baseline_label,
            candidate_label,
            log_scale=metric in log_scale_metrics,
        )
        _draw_error(axes_grid[row_index][1], comparison, metric, labels, relative=False)
        _draw_error(axes_grid[row_index][2], comparison, metric, labels, relative=True)

    common = comparison.common_steps
    default_title = (
        f"{labels('compare_title')} — "
        f"{baseline_label} vs {candidate_label} "
        f"(step {common[0]}..{common[-1]}, n={len(common)})"
    )
    figure.suptitle(title or default_title, fontsize=13)
    figure.tight_layout(rect=(0, 0, 1, 0.97))
    return _save_figure(figure, output_path, dpi, pyplot)


def _save_figure(figure, output_path: Path, dpi: int, pyplot) -> Path:
    """保存图像并校验产物非空。

    Raises:
        RuntimeError: 写出失败或得到 0 字节文件。
    """
    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(output_path, dpi=dpi, bbox_inches="tight")
    except (OSError, ValueError) as error:
        pyplot.close(figure)
        raise RuntimeError(f"写出图像失败 {output_path}: {error}") from error
    if not output_path.is_file() or output_path.stat().st_size == 0:
        pyplot.close(figure)
        raise RuntimeError(f"图像写出后为空文件: {output_path}")
    return output_path


def build_argument_parser() -> argparse.ArgumentParser:
    """构造命令行解析器。

    Returns:
        配置好的 ``ArgumentParser``。
    """
    parser = argparse.ArgumentParser(
        prog="plot_training_log.py",
        description="把训练日志指标画成以 step 为横轴的曲线；给 --compare 时额外画绝对/相对误差。",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "退出码:\n"
            f"  {EXIT_SUCCESS}  成功\n"
            f"  {EXIT_NO_METRIC_PARSED}  日志里没解析到任何指标\n"
            f"  {EXIT_NO_COMMON_STEP}  两份日志没有共同 step\n"
            f"  {EXIT_INPUT_ERROR}  输入文件或 profile 有问题\n"
            f"  {EXIT_PLOT_ERROR}  绘图或写文件失败\n"
        ),
    )
    parser.add_argument("log", type=Path, help="主训练日志路径")
    parser.add_argument("--compare", type=Path, default=None, metavar="LOG", help="对比日志路径")
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
        help="覆盖某指标在日志中的字段名，可重复",
    )
    parser.add_argument(
        "--metrics",
        default=None,
        metavar="A,B,C",
        help=f"在必绘指标（{'/'.join(MANDATORY_METRICS)}）之外追加要画的指标",
    )
    parser.add_argument("--all-metrics", action="store_true", help="把日志里出现过的所有指标都画出来")
    parser.add_argument(
        "--log-scale",
        default=None,
        metavar="A,B",
        help="这些指标的纵轴取对数（取值含非正数时自动忽略）",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=None,
        help="PNG 输出路径（默认 training_log.png / training_log_compare.png）",
    )
    parser.add_argument("--dpi", type=int, default=140, help="输出分辨率（默认 140）")
    parser.add_argument(
        "--panel-width", type=float, default=DEFAULT_PANEL_WIDTH, help="单个子图宽度（英寸）"
    )
    parser.add_argument(
        "--panel-height", type=float, default=DEFAULT_PANEL_HEIGHT, help="单个子图高度（英寸）"
    )
    parser.add_argument("--title", default=None, help="自定义总标题")
    parser.add_argument("--baseline-label", default=None, help="基线曲线图例名（默认取文件名）")
    parser.add_argument("--candidate-label", default=None, help="对比曲线图例名（默认取文件名）")
    parser.add_argument(
        "--baseline-is-compare",
        action="store_true",
        help="把 --compare 的日志当作基线（默认主日志是基线）",
    )
    parser.add_argument(
        "--on-duplicate-step",
        choices=("last", "first", "merge"),
        default="last",
        help="同一 step 重复出现时的处理方式（默认 last）",
    )
    parser.add_argument(
        "--synthesize-steps",
        action="store_true",
        help="日志不打印 step 时，用行序号充当横轴",
    )
    parser.add_argument("--max-lines", type=int, default=None, metavar="N", help="最多读取 N 行")
    parser.add_argument("--encoding", default="utf-8", help="日志文件编码（默认 utf-8）")
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="只保存文件、不弹窗。无图形显示的环境会自动按此处理",
    )
    parser.add_argument("--quiet", action="store_true", help="不把告警打到 stderr")
    return parser


def _default_output_path(is_comparison: bool) -> Path:
    """未指定 ``--output`` 时的默认文件名。"""
    return Path("training_log_compare.png" if is_comparison else "training_log.png")


def main(argv: Optional[Sequence[str]] = None) -> int:
    """命令行入口。

    Args:
        argv: 参数列表，``None`` 表示用 ``sys.argv[1:]``。

    Returns:
        退出码。
    """
    parser = build_argument_parser()
    arguments = parser.parse_args(argv)

    setup_warnings: List[str] = []
    try:
        profile_aliases: Dict[str, List[str]] = {}
        if arguments.profile is not None:
            profile_aliases, setup_warnings = load_metric_aliases_from_profile(arguments.profile)
        override_aliases = _parse_metric_key_overrides(arguments.metric_key)
    except (ValueError, RuntimeError) as error:
        print(f"错误: {error}", file=sys.stderr)
        return EXIT_INPUT_ERROR

    extra_metrics = (
        [item.strip() for item in arguments.metrics.split(",") if item.strip()]
        if arguments.metrics
        else []
    )
    log_scale_metrics = (
        [item.strip() for item in arguments.log_scale.split(",") if item.strip()]
        if arguments.log_scale
        else []
    )
    # 只有在既不追加指标也不要全量时，才把别名表裁剪到必绘指标，避免白解析。
    selected = None if arguments.all_metrics else list(MANDATORY_METRICS) + extra_metrics
    metric_aliases = build_metric_aliases(
        profile_aliases=profile_aliases,
        override_aliases=override_aliases,
        selected_metrics=selected,
    )

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

    parsed_logs = [log for log in (main_log, compare_log) if log is not None]
    for parsed in parsed_logs:
        if not parsed.points:
            print(f"错误: {parsed.source} 没有解析到任何 step/指标，无法绘图。", file=sys.stderr)
            for warning in parsed.warnings:
                print(f"       - {warning}", file=sys.stderr)
            return EXIT_NO_METRIC_PARSED

    if not arguments.quiet:
        for warning in setup_warnings:
            print(f"[告警] {warning}", file=sys.stderr)
        for parsed in parsed_logs:
            for warning in parsed.warnings:
                print(f"[告警] {Path(parsed.source).name}: {warning}", file=sys.stderr)

    metrics = _resolve_panel_metrics(parsed_logs, extra_metrics, arguments.all_metrics)
    for metric in MANDATORY_METRICS:
        for parsed in parsed_logs:
            if metric not in parsed.present_metrics and not arguments.quiet:
                print(
                    f"[告警] {Path(parsed.source).name}: 必绘指标 {metric!r} 缺失，"
                    "对应面板将留空占位",
                    file=sys.stderr,
                )

    comparison: Optional[LogComparison] = None
    if compare_log is not None:
        baseline, candidate = (
            (compare_log, main_log) if arguments.baseline_is_compare else (main_log, compare_log)
        )
        try:
            comparison = compare_parsed_logs(baseline, candidate, metrics=metrics)
        except NoCommonStepError as error:
            print(f"错误: {error}", file=sys.stderr)
            return EXIT_NO_COMMON_STEP
        if not arguments.quiet:
            for warning in comparison.warnings:
                print(f"[告警] 对比: {warning}", file=sys.stderr)

    headless = arguments.no_show or _is_headless_environment()
    try:
        matplotlib_module, pyplot = _import_matplotlib(headless)
    except RuntimeError as error:
        print(f"错误: {error}", file=sys.stderr)
        return EXIT_PLOT_ERROR

    labels = PlotLabels(configure_chinese_font(matplotlib_module))
    if not labels.use_chinese and not arguments.quiet:
        print("[告警] 未找到中文字体，图面文案改用英文；安装 Noto Sans CJK 可恢复中文", file=sys.stderr)

    output_path = arguments.output or _default_output_path(comparison is not None)
    try:
        if comparison is not None:
            written = plot_comparison(
                comparison,
                metrics,
                output_path,
                pyplot=pyplot,
                labels=labels,
                panel_width=arguments.panel_width,
                panel_height=arguments.panel_height,
                dpi=arguments.dpi,
                title=arguments.title,
                baseline_label=arguments.baseline_label or Path(comparison.baseline.source).name,
                candidate_label=arguments.candidate_label or Path(comparison.candidate.source).name,
                log_scale_metrics=log_scale_metrics,
            )
        else:
            written = plot_single_log(
                main_log,
                metrics,
                output_path,
                pyplot=pyplot,
                labels=labels,
                panel_width=arguments.panel_width,
                panel_height=arguments.panel_height,
                dpi=arguments.dpi,
                title=arguments.title,
                log_scale_metrics=log_scale_metrics,
            )
    except RuntimeError as error:
        print(f"错误: {error}", file=sys.stderr)
        return EXIT_PLOT_ERROR
    except Exception as error:  # noqa: BLE001  matplotlib 的异常类型很杂，统一兜底
        print(f"错误: 绘图失败: {type(error).__name__}: {error}", file=sys.stderr)
        return EXIT_GENERAL_ERROR

    print(f"已写出 {written}（{written.stat().st_size} 字节，{len(metrics)} 个指标）")
    if not headless:
        pyplot.show()
    pyplot.close("all")
    return EXIT_SUCCESS


if __name__ == "__main__":
    sys.exit(main())
