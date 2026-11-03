"""命令行入口：把 profiler 产物转成统一 kernel 事件，并校验数据够不够用。

用法见同目录 README.md。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

from .adapters import ADAPTERS, AdapterConfig, detect_format
from .schema import validate, write_jsonl


def _load_profile(profile_path: Path | None) -> dict[str, Any]:
    if profile_path is None:
        return {}
    if not profile_path.exists():
        raise SystemExit(f"profile 不存在: {profile_path}")
    try:
        import yaml
    except ImportError:
        raise SystemExit(
            "读取 profile 需要 PyYAML。安装: pip install pyyaml\n"
            "或者不传 --profile，改用 --column 手工指定列名映射"
        )
    with profile_path.open("r", encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle)
    if not isinstance(loaded, dict):
        raise SystemExit(f"{profile_path} 不是合法的 profile（顶层应为映射）")
    return loaded


def _apply_column_overrides(config: AdapterConfig, overrides: list[str]) -> None:
    for item in overrides:
        if "=" not in item:
            raise SystemExit(f"--column 的格式应为 统一字段名=你的列名，收到: {item!r}")
        unified, actual = item.split("=", 1)
        config.column_mapping[unified.strip()] = actual.strip()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="profiler_adapter.normalize",
        description="把各家 profiler 产物转成统一 kernel 事件 schema",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--input", required=True, type=Path, help="profiler 产物路径")
    parser.add_argument(
        "--format", default="auto", choices=["auto", *sorted(ADAPTERS)],
        help="输入格式。auto 按扩展名与内容猜测",
    )
    parser.add_argument("--output", type=Path, help="输出 jsonl 路径。省略则只打印校验报告")
    parser.add_argument("--profile", type=Path, help="硬件 profile YAML，用于取列名映射与计数器映射")
    parser.add_argument(
        "--column", action="append", default=[], metavar="统一字段=实际列名",
        help="手工覆盖列名映射，可重复。优先级高于 profile",
    )
    parser.add_argument(
        "--comm-pattern", action="append", default=[], metavar="正则",
        help="追加通信 kernel 的名称识别模式，可重复",
    )
    parser.add_argument(
        "--include-host-events", action="store_true",
        help="把主机侧算子记录也纳入。默认排除，因为它们是下发记录而非设备执行",
    )
    parser.add_argument("--validate-only", action="store_true", help="只校验不写出")
    parser.add_argument("--quiet", action="store_true", help="不打印校验报告")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if not args.input.exists():
        print(f"输入文件不存在: {args.input}", file=sys.stderr)
        return 2
    if not args.validate_only and args.output is None:
        print("需要 --output 指定输出路径，或加 --validate-only 只做校验", file=sys.stderr)
        return 2

    input_format = args.format
    if input_format == "auto":
        try:
            input_format = detect_format(args.input)
        except ValueError as error:
            print(str(error), file=sys.stderr)
            return 2
        if not args.quiet:
            print(f"识别到格式: {input_format}", file=sys.stderr)

    config = AdapterConfig.from_profile(_load_profile(args.profile))
    _apply_column_overrides(config, args.column)
    if args.comm_pattern:
        config.communication_patterns = tuple(
            [*config.communication_patterns, *args.comm_pattern]
        )
    config.include_host_events = args.include_host_events

    adapter = ADAPTERS[input_format]
    try:
        if args.validate_only:
            report = validate(adapter(args.input, config))
        else:
            # 先落盘再从盘上校验，避免把整份 trace 读进内存
            written = write_jsonl(adapter(args.input, config), args.output)
            if not args.quiet:
                print(f"已写出 {written} 条事件 -> {args.output}", file=sys.stderr)
            report = validate(ADAPTERS["jsonl"](args.output, config))
    except (ValueError, OSError) as error:
        print(f"转换失败: {error}", file=sys.stderr)
        return 1

    if not args.quiet:
        print(report.render())

    if report.total_events == 0:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
