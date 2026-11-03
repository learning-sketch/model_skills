#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""PyTorch 显存快照分析器。

解析 ``torch.cuda.memory._dump_snapshot()`` 一类接口产出的 memory snapshot，
给出七个维度的分析：

1. **总览** —— reserved / allocated / active / requested，以及碎片率
2. **语义分类** —— 按调用栈关键字把在用显存分到激活、梯度、优化器状态、参数、
   通信缓冲等桶里（关键字表可替换）
3. **峰值时刻** —— 回放 trace 重建 allocated/reserved 曲线，定位峰值与触发它的调用栈
4. **碎片化** —— 空闲块分布、最大空闲块、外部碎片率、内部碎片（取整浪费）
5. **Top-N 大块** —— 最大的若干在用块及其调用栈
6. **双快照对比** —— 总览差值、分类差值、按调用栈签名找增长最快的分配点
7. **多快照泄漏趋势** —— 三份及以上快照上做最小二乘拟合 + 单调性检查，筛泄漏嫌疑

.. warning::
   **反序列化安全。** snapshot 是 pickle 文件，而 pickle 在反序列化时可以执行
   任意代码。**只分析你自己采集的、或来源完全可信的文件。** 本脚本默认使用受限
   Unpickler：只允许还原普通容器类型，遇到任何需要 import 的对象直接拒绝。
   确有需要时可用 ``--allow-arbitrary-pickle`` 解除限制，但那意味着你为这个文件
   的来源背书。

命令行示例::

    python analyze_memory_snapshot.py snap.pickle
    python analyze_memory_snapshot.py before.pickle after.pickle            # 双快照对比
    python analyze_memory_snapshot.py step{10,20,30,40}.pickle --mode trend # 泄漏趋势
    python analyze_memory_snapshot.py snap.pickle --json -o report.json
"""

from __future__ import annotations

import argparse
import gzip
import io
import json
import math
import pickle
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

__all__ = [
    "DEFAULT_CATEGORY_RULES",
    "CategoryReport",
    "ComparisonReport",
    "FragmentationReport",
    "OverviewReport",
    "PeakReport",
    "Snapshot",
    "SnapshotAnalysis",
    "TrendReport",
    "analyze_snapshot",
    "compare_snapshots",
    "detect_leak_trend",
    "load_snapshot",
]


# ------------------------------------------------------------------------------
# 退出码
# ------------------------------------------------------------------------------
EXIT_SUCCESS = 0
EXIT_GENERAL_ERROR = 1
EXIT_USAGE_ERROR = 2  # argparse 自身使用
EXIT_INPUT_ERROR = 5
EXIT_FORMAT_ERROR = 7

PICKLE_SECURITY_WARNING = (
    "安全提示: memory snapshot 是 pickle 文件，反序列化可执行任意代码。"
    "请只分析你自己采集的或来源可信的文件。当前使用受限 Unpickler（只还原普通容器）。"
)

# 超过这个大小的快照会先提示一句，避免用户以为脚本卡死。
LARGE_FILE_WARNING_BYTES = 512 * 1024 * 1024

# 在用块的 state 取值。PyTorch 不同版本写法略有出入，统一按前缀判定。
_ACTIVE_STATE_PREFIX = "active"
_PENDING_FREE_STATES = frozenset({"active_awaiting_free", "active_pending_free"})

# trace 中代表「占用增加/减少」的动作名。
_ALLOC_ACTIONS = frozenset({"alloc"})
_FREE_COMPLETED_ACTIONS = frozenset({"free_completed"})
_FREE_REQUESTED_ACTIONS = frozenset({"free_requested"})
_SEGMENT_GROW_ACTIONS = frozenset({"segment_alloc", "segment_map"})
_SEGMENT_SHRINK_ACTIONS = frozenset({"segment_free", "segment_unmap"})
_OOM_ACTIONS = frozenset({"oom"})


# ------------------------------------------------------------------------------
# 语义分类规则
# ------------------------------------------------------------------------------
# 顺序即优先级：一个调用栈命中多条规则时，排在前面的胜出。
#
# 这张表是**默认值，不是契约**。它刻意只用 PyTorch 上游与通用生态里的公开命名，
# 不绑定任何特定厂商的运行时。换成别的通信库、别的优化器实现、别的训练框架时，
# 用 --category-rules 换一张表即可，不需要改代码。通信一类默认同时认 nccl / gloo /
# mpi / ucc 这些常见名字；你的集合通信库叫别的名字时，可以让脚本从硬件 profile 的
# collective.library / collective.torch_backend 自动补进来（--profile），
# 或者直接在规则文件里写。
DEFAULT_CATEGORY_RULES: Tuple[Tuple[str, Tuple[str, ...]], ...] = (
    (
        "optimizer_state",
        (
            "optim/",
            "optimizer",
            "adamw",
            "adam.py",
            "adafactor",
            "sgd.py",
            "lamb",
            "_init_group",
            "exp_avg",
            "_single_tensor_",
            "_multi_tensor_",
            "_fused_",
            "shard_optim",
            "zero_optim",
        ),
    ),
    (
        "communication",
        (
            "nccl",
            "gloo",
            "mpi",
            "ucc",
            "c10d",
            "process_group",
            "distributed/",
            "all_reduce",
            "allreduce",
            "all_gather",
            "allgather",
            "reduce_scatter",
            "reducescatter",
            "all_to_all",
            "alltoall",
            "_broadcast_",
            "_coalescing_manager",
            "grad_bucket",
            "reducer.cpp",
        ),
    ),
    (
        "dataloader",
        ("dataloader", "_worker", "collate", "pin_memory", "_fetcher", "data/dataset"),
    ),
    (
        "gradient",
        (
            "backward",
            "autograd",
            "accumulategrad",
            "accumulate_grad",
            "grad_fn",
            "_engine_run_backward",
            "torch/_functorch",
        ),
    ),
    # parameter 必须排在 activation 之前。权重的分配栈里几乎总会出现模块类名
    # （Linear.reset_parameters、Embedding.__init__ 之类），而 activation 的关键字里
    # 恰好有 linear / embedding / mlp 这些模块名，先匹配 activation 会把权重判成激活。
    # 反过来则不会误伤：parameter 的关键字都是权重专有的动作名，前向路径上不会出现。
    (
        "parameter",
        (
            "_apply",
            "load_state_dict",
            "_load_from_state_dict",
            "init_weights",
            "reset_parameters",
            "nn/init",
            "register_parameter",
            "materialize",
            "from_pretrained",
        ),
    ),
    (
        "activation",
        (
            "forward",
            "checkpoint",
            "recompute",
            "attention",
            "mlp",
            "layer_norm",
            "layernorm",
            "rmsnorm",
            "softmax",
            "matmul",
            "linear",
            "embedding",
            "gelu",
            "silu",
            "swiglu",
            "dropout",
            "transformer",
            "rotary",
            "moe",
        ),
    ),
    (
        "compiled_graph",
        ("inductor", "triton", "cudagraph", "graph_capture", "dynamo", "codecache", "_compile"),
    ),
)

UNCLASSIFIED_CATEGORY = "unclassified"


# ------------------------------------------------------------------------------
# 格式化
# ------------------------------------------------------------------------------
def format_bytes(byte_count: float) -> str:
    """把字节数格式化成人读的字符串。

    Args:
        byte_count: 字节数，可为负（差值场景）。

    Returns:
        例如 ``"12.34 GiB"`` / ``"-512.00 MiB"``。
    """
    if byte_count != byte_count:  # NaN
        return "nan"
    sign = "-" if byte_count < 0 else ""
    remaining = abs(float(byte_count))
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if remaining < 1024.0 or unit == "TiB":
            precision = 0 if unit == "B" else 2
            return f"{sign}{remaining:.{precision}f} {unit}"
        remaining /= 1024.0
    return f"{sign}{remaining:.2f} TiB"


def _format_ratio(numerator: float, denominator: float) -> str:
    """安全地把比值格式化成百分比。"""
    if denominator == 0:
        return "n/a"
    return f"{numerator / denominator:.1%}"


# ------------------------------------------------------------------------------
# 数据结构
# ------------------------------------------------------------------------------
@dataclass(frozen=True)
class StackFrame:
    """调用栈的一帧。"""

    filename: str
    line: int
    name: str

    def render(self) -> str:
        """渲染成 ``name (file:line)``。"""
        return f"{self.name} ({self.filename}:{self.line})"

    def haystack(self) -> str:
        """供关键字匹配使用的小写串。"""
        return f"{self.filename}\n{self.name}".lower()


@dataclass(frozen=True)
class MemoryBlock:
    """segment 内的一个块。"""

    device: int
    segment_address: int
    segment_type: str
    stream: int
    size: int
    requested_size: int
    state: str
    frames: Tuple[StackFrame, ...]

    @property
    def is_active(self) -> bool:
        """块是否仍被占用（含等待跨流释放的）。"""
        return self.state.startswith(_ACTIVE_STATE_PREFIX)

    @property
    def is_pending_free(self) -> bool:
        """块是否已被用户释放但还在等跨流事件。"""
        return self.state in _PENDING_FREE_STATES

    @property
    def is_free(self) -> bool:
        """块是否是 segment 内的空闲空间。"""
        return not self.is_active


@dataclass(frozen=True)
class MemorySegment:
    """一次底层显存申请拿到的连续区间。"""

    device: int
    address: int
    total_size: int
    allocated_size: int
    active_size: int
    requested_size: int
    stream: int
    segment_type: str
    is_expandable: bool
    blocks: Tuple[MemoryBlock, ...]


@dataclass(frozen=True)
class TraceEvent:
    """device_traces 中的一条记录。"""

    device: int
    index: int
    action: str
    address: int
    size: int
    stream: int
    time_us: Optional[int]
    device_free: Optional[int]
    frames: Tuple[StackFrame, ...]


@dataclass
class Snapshot:
    """一份已归一化的显存快照。"""

    path: str
    segments: Tuple[MemorySegment, ...]
    trace_events: Tuple[TraceEvent, ...]
    allocator_settings: Dict[str, Any] = field(default_factory=dict)
    trace_truncated: bool = False
    warnings: List[str] = field(default_factory=list)

    @property
    def label(self) -> str:
        """用于报告的短名字。"""
        return Path(self.path).name

    @property
    def blocks(self) -> List[MemoryBlock]:
        """所有 segment 的全部块。"""
        return [block for segment in self.segments for block in segment.blocks]

    @property
    def devices(self) -> List[int]:
        """快照里出现过的设备号。"""
        return sorted({segment.device for segment in self.segments})

    def filter_device(self, device: Optional[int]) -> "Snapshot":
        """只保留某个设备的数据。

        Args:
            device: 设备号；``None`` 表示不过滤。

        Returns:
            过滤后的新快照（原对象不变）。
        """
        if device is None:
            return self
        return Snapshot(
            path=self.path,
            segments=tuple(segment for segment in self.segments if segment.device == device),
            trace_events=tuple(event for event in self.trace_events if event.device == device),
            allocator_settings=self.allocator_settings,
            trace_truncated=self.trace_truncated,
            warnings=list(self.warnings),
        )


# ------------------------------------------------------------------------------
# 加载
# ------------------------------------------------------------------------------
class _RestrictedUnpickler(pickle.Unpickler):
    """只允许还原普通容器的 Unpickler。

    PyTorch 导出的 snapshot 全部由 dict / list / tuple / int / str 构成，这些类型
    走的是 pickle 原生操作码，根本不会调到 ``find_class``。因此把 ``find_class``
    一律拒掉，既不影响正常文件，又能挡住「反序列化时执行任意代码」这条路。
    """

    _ALLOWED: frozenset = frozenset({("collections", "OrderedDict")})

    def find_class(self, module: str, name: str) -> Any:
        """拒绝加载任何不在允许名单内的类型。

        Raises:
            pickle.UnpicklingError: 文件试图还原一个需要 import 的对象。
        """
        if (module, name) in self._ALLOWED:
            return super().find_class(module, name)
        raise pickle.UnpicklingError(
            f"受限模式拒绝加载 {module}.{name}。正常的 PyTorch snapshot 不需要它；"
            "若你确认文件来源可信，可加 --allow-arbitrary-pickle 解除限制"
        )


def _read_raw_document(path: Path, allow_arbitrary_pickle: bool) -> Any:
    """按文件实际格式读出原始文档。

    支持裸 pickle、gzip 压缩的 pickle，以及部分工具导出的 JSON。

    Args:
        path: 快照文件路径。
        allow_arbitrary_pickle: 是否解除受限 Unpickler。

    Returns:
        反序列化后的对象。

    Raises:
        RuntimeError: 文件不存在、为空或无法解析。
    """
    if not path.exists():
        raise RuntimeError(f"快照文件不存在: {path}")
    if not path.is_file():
        raise RuntimeError(f"快照路径不是文件: {path}")
    if path.stat().st_size == 0:
        raise RuntimeError(f"快照文件为空: {path}")

    try:
        payload = path.read_bytes()
    except OSError as error:
        raise RuntimeError(f"无法读取快照文件 {path}: {error}") from error

    if payload[:2] == b"\x1f\x8b":
        try:
            payload = gzip.decompress(payload)
        except OSError as error:
            raise RuntimeError(f"gzip 解压失败 {path}: {error}") from error

    stripped = payload.lstrip()
    if stripped[:1] in (b"{", b"["):
        try:
            return json.loads(payload.decode("utf-8", errors="replace"))
        except json.JSONDecodeError as error:
            raise RuntimeError(f"文件看起来像 JSON 但解析失败 {path}: {error}") from error

    handle = io.BytesIO(payload)
    unpickler = pickle.Unpickler(handle) if allow_arbitrary_pickle else _RestrictedUnpickler(handle)
    try:
        return unpickler.load()
    except pickle.UnpicklingError as error:
        raise RuntimeError(f"pickle 反序列化失败 {path}: {error}") from error
    except Exception as error:  # noqa: BLE001  pickle 可能抛出各种底层异常
        raise RuntimeError(
            f"pickle 反序列化失败 {path}: {type(error).__name__}: {error}。"
            "确认这是 memory snapshot 文件而不是别的 pickle"
        ) from error


def _coerce_frames(raw_frames: Any, max_frames: int) -> Tuple[StackFrame, ...]:
    """把原始 frames 列表归一化成 ``StackFrame`` 元组。"""
    if not isinstance(raw_frames, Sequence) or isinstance(raw_frames, (str, bytes)):
        return ()
    frames: List[StackFrame] = []
    for raw_frame in raw_frames[:max_frames]:
        if isinstance(raw_frame, Mapping):
            frames.append(
                StackFrame(
                    filename=str(raw_frame.get("filename", "?")),
                    line=int(raw_frame.get("line", 0) or 0),
                    name=str(raw_frame.get("name", "?")),
                )
            )
        elif isinstance(raw_frame, str):
            frames.append(StackFrame(filename="?", line=0, name=raw_frame))
    return tuple(frames)


def _coerce_int(value: Any, default: int = 0) -> int:
    """尽力把任意取值转成 int。"""
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def load_snapshot(
    path: Path,
    *,
    include_trace: bool = True,
    max_trace_events: Optional[int] = None,
    max_frames: int = 32,
    allow_arbitrary_pickle: bool = False,
) -> Snapshot:
    """加载并归一化一份显存快照。

    .. warning::
       pickle 反序列化有代码执行风险，只分析可信来源的文件。默认使用受限
       Unpickler。

    Args:
        path: 快照文件路径。
        include_trace: 是否解析 ``device_traces``。大快照里 trace 往往占九成体积，
            只看静态占用时可以关掉。
        max_trace_events: 每个设备最多保留的 trace 事件数（保留最新的那些，
            因为 OOM 发生在末尾）。``None`` 表示不限。
        max_frames: 每个调用栈最多保留的帧数。
        allow_arbitrary_pickle: 解除受限 Unpickler。

    Returns:
        归一化后的快照。

    Raises:
        RuntimeError: 文件无法读取或结构不符合 snapshot 约定。
    """
    document = _read_raw_document(path, allow_arbitrary_pickle)
    if not isinstance(document, Mapping):
        raise RuntimeError(
            f"快照顶层不是字典（实际是 {type(document).__name__}），"
            f"不像 memory snapshot: {path}"
        )

    warnings: List[str] = []
    raw_segments = document.get("segments")
    if raw_segments is None:
        raise RuntimeError(
            f"快照缺少 'segments' 字段，无法分析: {path}。"
            f"实际顶层键: {sorted(str(key) for key in document)}"
        )
    if not isinstance(raw_segments, Sequence):
        raise RuntimeError(f"快照的 'segments' 不是列表: {path}")

    segments: List[MemorySegment] = []
    aggregate_mismatches: List[str] = []
    for raw_segment in raw_segments:
        if not isinstance(raw_segment, Mapping):
            continue
        device = _coerce_int(raw_segment.get("device"))
        segment_address = _coerce_int(raw_segment.get("address"))
        segment_type = str(raw_segment.get("segment_type", "unknown"))
        stream = _coerce_int(raw_segment.get("stream"))

        blocks: List[MemoryBlock] = []
        raw_blocks = raw_segment.get("blocks")
        if isinstance(raw_blocks, Sequence):
            for raw_block in raw_blocks:
                if not isinstance(raw_block, Mapping):
                    continue
                size = _coerce_int(raw_block.get("size"))
                blocks.append(
                    MemoryBlock(
                        device=device,
                        segment_address=segment_address,
                        segment_type=segment_type,
                        stream=stream,
                        size=size,
                        requested_size=_coerce_int(raw_block.get("requested_size"), size),
                        state=str(raw_block.get("state", "unknown")),
                        frames=_coerce_frames(raw_block.get("frames"), max_frames),
                    )
                )

        total_size = _coerce_int(raw_segment.get("total_size"))
        if blocks:
            # 一律从块重新求和，而不是直接采信 segment 上的汇总字段。
            # 这几个字段在不同 PyTorch 版本里对「算不算 awaiting_free」的口径不一致，
            # 照抄会让总览与碎片分析互相打架（例如 allocated - requested 算出负数）。
            allocated_size = sum(block.size for block in blocks if block.state == "active_allocated")
            active_size = sum(block.size for block in blocks if block.is_active)
            requested_size = sum(
                block.requested_size for block in blocks if block.state == "active_allocated"
            )
            declared_allocated = raw_segment.get("allocated_size")
            if (
                isinstance(declared_allocated, int)
                and declared_allocated != allocated_size
                and len(aggregate_mismatches) < 5
            ):
                aggregate_mismatches.append(
                    f"segment 0x{segment_address:x} 声明 allocated={declared_allocated}，"
                    f"按块求和得 {allocated_size}"
                )
            if not total_size:
                total_size = sum(block.size for block in blocks)
        else:
            allocated_size = _coerce_int(raw_segment.get("allocated_size"))
            active_size = _coerce_int(raw_segment.get("active_size"), allocated_size)
            requested_size = _coerce_int(raw_segment.get("requested_size"), allocated_size)

        segments.append(
            MemorySegment(
                device=device,
                address=segment_address,
                total_size=total_size,
                allocated_size=allocated_size,
                active_size=active_size,
                requested_size=requested_size,
                stream=stream,
                segment_type=segment_type,
                is_expandable=bool(raw_segment.get("is_expandable", False)),
                blocks=tuple(blocks),
            )
        )

    trace_events: List[TraceEvent] = []
    trace_truncated = False
    raw_traces = document.get("device_traces")
    if include_trace and isinstance(raw_traces, Sequence):
        for device_index, raw_device_trace in enumerate(raw_traces):
            if not isinstance(raw_device_trace, Sequence):
                continue
            selected = list(raw_device_trace)
            if max_trace_events is not None and len(selected) > max_trace_events:
                # 保留末尾：OOM 和峰值几乎总是出现在 trace 的最后一段。
                selected = selected[-max_trace_events:]
                trace_truncated = True
            for event_index, raw_event in enumerate(selected):
                if not isinstance(raw_event, Mapping):
                    continue
                device_free = raw_event.get("device_free")
                time_us = raw_event.get("time_us")
                trace_events.append(
                    TraceEvent(
                        device=_coerce_int(raw_event.get("device"), device_index),
                        index=event_index,
                        action=str(raw_event.get("action", "unknown")),
                        address=_coerce_int(raw_event.get("addr")),
                        size=_coerce_int(raw_event.get("size")),
                        stream=_coerce_int(raw_event.get("stream")),
                        time_us=_coerce_int(time_us) if time_us is not None else None,
                        device_free=_coerce_int(device_free) if device_free is not None else None,
                        frames=_coerce_frames(raw_event.get("frames"), max_frames),
                    )
                )
    elif include_trace and raw_traces is None:
        warnings.append(
            "快照没有 device_traces：峰值与泄漏趋势分析不可用。"
            "采集时需要先开启内存历史记录（record_memory_history）"
        )

    if trace_truncated:
        warnings.append(f"trace 事件超过 --max-trace-events，已只保留每个设备最后 {max_trace_events} 条")
    if not segments:
        warnings.append("快照里一个 segment 都没有：可能是在分配任何显存之前就采集了")
    if aggregate_mismatches:
        warnings.append(
            "segment 的汇总字段与按块求和的结果不一致，已以按块求和为准。"
            "这通常意味着快照格式与上游 PyTorch 有出入，请核对采集侧的实现: "
            + "; ".join(aggregate_mismatches)
        )

    raw_settings = document.get("allocator_settings")
    allocator_settings = dict(raw_settings) if isinstance(raw_settings, Mapping) else {}

    return Snapshot(
        path=str(path),
        segments=tuple(segments),
        trace_events=tuple(trace_events),
        allocator_settings=allocator_settings,
        trace_truncated=trace_truncated,
        warnings=warnings,
    )


# ------------------------------------------------------------------------------
# 维度 1：总览
# ------------------------------------------------------------------------------
@dataclass
class DeviceOverview:
    """单设备的占用总览。"""

    device: int
    reserved_bytes: int
    allocated_bytes: int
    active_bytes: int
    requested_bytes: int
    segment_count: int
    large_segment_count: int
    small_segment_count: int
    expandable_segment_count: int
    device_capacity_bytes: Optional[int]

    @property
    def free_in_reserved_bytes(self) -> int:
        """已向底层申请但当前没被用的字节数。"""
        return self.reserved_bytes - self.allocated_bytes

    @property
    def fragmentation_ratio(self) -> float:
        """碎片率：reserved 里闲置的比例。"""
        if self.reserved_bytes == 0:
            return 0.0
        return self.free_in_reserved_bytes / self.reserved_bytes

    @property
    def internal_waste_bytes(self) -> int:
        """分配取整造成的浪费：allocated 与用户真正请求量之差。"""
        return max(self.allocated_bytes - self.requested_bytes, 0)

    @property
    def pending_free_bytes(self) -> int:
        """已释放但还在等跨流事件的字节数。"""
        return max(self.active_bytes - self.allocated_bytes, 0)

    def to_dict(self) -> Dict[str, Any]:
        """转为可序列化字典。"""
        return {
            "device": self.device,
            "reserved_bytes": self.reserved_bytes,
            "allocated_bytes": self.allocated_bytes,
            "active_bytes": self.active_bytes,
            "requested_bytes": self.requested_bytes,
            "free_in_reserved_bytes": self.free_in_reserved_bytes,
            "fragmentation_ratio": self.fragmentation_ratio,
            "internal_waste_bytes": self.internal_waste_bytes,
            "pending_free_bytes": self.pending_free_bytes,
            "segment_count": self.segment_count,
            "large_segment_count": self.large_segment_count,
            "small_segment_count": self.small_segment_count,
            "expandable_segment_count": self.expandable_segment_count,
            "device_capacity_bytes": self.device_capacity_bytes,
            "capacity_utilization": (
                self.reserved_bytes / self.device_capacity_bytes
                if self.device_capacity_bytes
                else None
            ),
        }


@dataclass
class OverviewReport:
    """所有设备的占用总览。"""

    devices: List[DeviceOverview]

    @property
    def total_reserved_bytes(self) -> int:
        """全部设备的 reserved 之和。"""
        return sum(device.reserved_bytes for device in self.devices)

    @property
    def total_allocated_bytes(self) -> int:
        """全部设备的 allocated 之和。"""
        return sum(device.allocated_bytes for device in self.devices)

    def to_dict(self) -> Dict[str, Any]:
        """转为可序列化字典。"""
        return {
            "total_reserved_bytes": self.total_reserved_bytes,
            "total_allocated_bytes": self.total_allocated_bytes,
            "devices": [device.to_dict() for device in self.devices],
        }


def summarize_overview(
    snapshot: Snapshot, device_capacity_bytes: Optional[int] = None
) -> OverviewReport:
    """汇总 reserved / allocated / active / 碎片率。

    Args:
        snapshot: 已加载的快照。
        device_capacity_bytes: 单卡可用显存字节数，用于算占用率；``None`` 则不算。

    Returns:
        总览报告。
    """
    by_device: Dict[int, List[MemorySegment]] = {}
    for segment in snapshot.segments:
        by_device.setdefault(segment.device, []).append(segment)

    devices: List[DeviceOverview] = []
    for device in sorted(by_device):
        segments = by_device[device]
        devices.append(
            DeviceOverview(
                device=device,
                reserved_bytes=sum(segment.total_size for segment in segments),
                allocated_bytes=sum(segment.allocated_size for segment in segments),
                active_bytes=sum(segment.active_size for segment in segments),
                requested_bytes=sum(segment.requested_size for segment in segments),
                segment_count=len(segments),
                large_segment_count=sum(1 for s in segments if s.segment_type == "large"),
                small_segment_count=sum(1 for s in segments if s.segment_type == "small"),
                expandable_segment_count=sum(1 for s in segments if s.is_expandable),
                device_capacity_bytes=device_capacity_bytes,
            )
        )
    return OverviewReport(devices=devices)


# ------------------------------------------------------------------------------
# 维度 2：语义分类
# ------------------------------------------------------------------------------
@dataclass
class CategoryBucket:
    """一个语义分类的汇总。"""

    category: str
    total_bytes: int
    block_count: int
    largest_block_bytes: int
    example_stack: List[str]

    def to_dict(self) -> Dict[str, Any]:
        """转为可序列化字典。"""
        return {
            "category": self.category,
            "total_bytes": self.total_bytes,
            "block_count": self.block_count,
            "largest_block_bytes": self.largest_block_bytes,
            "example_stack": list(self.example_stack),
        }


@dataclass
class CategoryReport:
    """语义分类结果。"""

    buckets: List[CategoryBucket]
    classified_bytes: int
    unclassified_bytes: int
    rule_order: List[str]
    blocks_without_stack: int

    def to_dict(self) -> Dict[str, Any]:
        """转为可序列化字典。"""
        total = self.classified_bytes + self.unclassified_bytes
        return {
            "rule_order": list(self.rule_order),
            "classified_bytes": self.classified_bytes,
            "unclassified_bytes": self.unclassified_bytes,
            "unclassified_ratio": self.unclassified_bytes / total if total else 0.0,
            "blocks_without_stack": self.blocks_without_stack,
            "buckets": [bucket.to_dict() for bucket in self.buckets],
        }


def classify_block(block: MemoryBlock, rules: Sequence[Tuple[str, Sequence[str]]]) -> str:
    """按调用栈关键字给一个块归类。

    规则顺序即优先级，第一条命中的胜出。匹配在「文件名 + 函数名」的小写串上做，
    任意一帧命中即算命中。

    Args:
        block: 待分类的块。
        rules: ``[(类别名, [关键字, ...]), ...]``，顺序敏感。

    Returns:
        类别名；没有任何规则命中时返回 ``"unclassified"``。
    """
    if not block.frames:
        return UNCLASSIFIED_CATEGORY
    haystack = "\n".join(frame.haystack() for frame in block.frames)
    for category, keywords in rules:
        for keyword in keywords:
            if keyword.lower() in haystack:
                return category
    return UNCLASSIFIED_CATEGORY


def classify_snapshot(
    snapshot: Snapshot, rules: Sequence[Tuple[str, Sequence[str]]]
) -> CategoryReport:
    """把在用显存按调用栈关键字分到语义桶里。

    Args:
        snapshot: 已加载的快照。
        rules: 分类规则，顺序即优先级。

    Returns:
        分类报告。
    """
    totals: Dict[str, int] = {}
    counts: Dict[str, int] = {}
    largest: Dict[str, int] = {}
    examples: Dict[str, List[str]] = {}
    blocks_without_stack = 0

    for block in snapshot.blocks:
        if not block.is_active:
            continue
        if not block.frames:
            blocks_without_stack += 1
        category = classify_block(block, rules)
        totals[category] = totals.get(category, 0) + block.size
        counts[category] = counts.get(category, 0) + 1
        if block.size > largest.get(category, -1):
            largest[category] = block.size
            examples[category] = [frame.render() for frame in block.frames[:6]]

    buckets = [
        CategoryBucket(
            category=category,
            total_bytes=totals[category],
            block_count=counts[category],
            largest_block_bytes=largest.get(category, 0),
            example_stack=examples.get(category, []),
        )
        for category in sorted(totals, key=lambda name: totals[name], reverse=True)
    ]
    unclassified = totals.get(UNCLASSIFIED_CATEGORY, 0)
    return CategoryReport(
        buckets=buckets,
        classified_bytes=sum(totals.values()) - unclassified,
        unclassified_bytes=unclassified,
        rule_order=[category for category, _ in rules],
        blocks_without_stack=blocks_without_stack,
    )


# ------------------------------------------------------------------------------
# 维度 3：峰值时刻
# ------------------------------------------------------------------------------
@dataclass
class OutOfMemoryEvent:
    """trace 里记录的一次分配失败。"""

    device: int
    requested_bytes: int
    device_free_bytes: Optional[int]
    time_us: Optional[int]
    stack: List[str]

    def to_dict(self) -> Dict[str, Any]:
        """转为可序列化字典。"""
        return {
            "device": self.device,
            "requested_bytes": self.requested_bytes,
            "device_free_bytes": self.device_free_bytes,
            "time_us": self.time_us,
            "stack": list(self.stack),
        }


@dataclass
class DevicePeak:
    """单设备的峰值信息。"""

    device: int
    peak_allocated_bytes: int
    peak_allocated_event_index: int
    peak_allocated_time_us: Optional[int]
    peak_allocated_stack: List[str]
    peak_reserved_bytes: int
    final_allocated_bytes: int
    event_count: int
    largest_single_allocation_bytes: int
    largest_single_allocation_stack: List[str]

    def to_dict(self) -> Dict[str, Any]:
        """转为可序列化字典。"""
        return {
            "device": self.device,
            "peak_allocated_bytes": self.peak_allocated_bytes,
            "peak_allocated_event_index": self.peak_allocated_event_index,
            "peak_allocated_time_us": self.peak_allocated_time_us,
            "peak_allocated_stack": list(self.peak_allocated_stack),
            "peak_reserved_bytes": self.peak_reserved_bytes,
            "final_allocated_bytes": self.final_allocated_bytes,
            "event_count": self.event_count,
            "largest_single_allocation_bytes": self.largest_single_allocation_bytes,
            "largest_single_allocation_stack": list(self.largest_single_allocation_stack),
        }


@dataclass
class PeakReport:
    """峰值与 OOM 事件。"""

    devices: List[DevicePeak]
    oom_events: List[OutOfMemoryEvent]
    trace_available: bool
    notes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """转为可序列化字典。"""
        return {
            "trace_available": self.trace_available,
            "devices": [device.to_dict() for device in self.devices],
            "oom_events": [event.to_dict() for event in self.oom_events],
            "notes": list(self.notes),
        }


def analyze_peak(snapshot: Snapshot) -> PeakReport:
    """回放 trace，重建占用曲线并定位峰值。

    ``alloc`` 增加 allocated，``free_completed`` 减少；没有 ``free_completed``
    的 trace 退回用 ``free_requested``。``segment_alloc`` / ``segment_map`` 增加
    reserved，对应的 free / unmap 减少。

    Args:
        snapshot: 已加载的快照。

    Returns:
        峰值报告；没有 trace 时 ``trace_available`` 为 False。
    """
    if not snapshot.trace_events:
        return PeakReport(
            devices=[],
            oom_events=[],
            trace_available=False,
            notes=["快照里没有 trace 事件，无法重建占用曲线。采集时需要先开启内存历史记录"],
        )

    notes: List[str] = []
    if snapshot.trace_truncated:
        notes.append("trace 被截断，峰值只反映保留下来的那段事件")

    by_device: Dict[int, List[TraceEvent]] = {}
    for event in snapshot.trace_events:
        by_device.setdefault(event.device, []).append(event)

    has_completed = any(event.action in _FREE_COMPLETED_ACTIONS for event in snapshot.trace_events)
    free_actions = _FREE_COMPLETED_ACTIONS if has_completed else _FREE_REQUESTED_ACTIONS
    if not has_completed:
        notes.append("trace 里没有 free_completed 事件，改用 free_requested 重建曲线")

    devices: List[DevicePeak] = []
    oom_events: List[OutOfMemoryEvent] = []

    for device in sorted(by_device):
        events = by_device[device]
        running_allocated = 0
        running_reserved = 0
        peak_allocated = 0
        peak_reserved = 0
        peak_index = -1
        peak_time: Optional[int] = None
        peak_stack: List[str] = []
        largest_allocation = 0
        largest_allocation_stack: List[str] = []

        for event in events:
            if event.action in _ALLOC_ACTIONS:
                running_allocated += event.size
                if event.size > largest_allocation:
                    largest_allocation = event.size
                    largest_allocation_stack = [frame.render() for frame in event.frames[:8]]
                if running_allocated > peak_allocated:
                    peak_allocated = running_allocated
                    peak_index = event.index
                    peak_time = event.time_us
                    peak_stack = [frame.render() for frame in event.frames[:8]]
            elif event.action in free_actions:
                running_allocated -= event.size
            elif event.action in _SEGMENT_GROW_ACTIONS:
                running_reserved += event.size
                peak_reserved = max(peak_reserved, running_reserved)
            elif event.action in _SEGMENT_SHRINK_ACTIONS:
                running_reserved -= event.size
            elif event.action in _OOM_ACTIONS:
                oom_events.append(
                    OutOfMemoryEvent(
                        device=device,
                        requested_bytes=event.size,
                        device_free_bytes=event.device_free,
                        time_us=event.time_us,
                        stack=[frame.render() for frame in event.frames[:12]],
                    )
                )

        devices.append(
            DevicePeak(
                device=device,
                peak_allocated_bytes=peak_allocated,
                peak_allocated_event_index=peak_index,
                peak_allocated_time_us=peak_time,
                peak_allocated_stack=peak_stack,
                peak_reserved_bytes=peak_reserved,
                final_allocated_bytes=running_allocated,
                event_count=len(events),
                largest_single_allocation_bytes=largest_allocation,
                largest_single_allocation_stack=largest_allocation_stack,
            )
        )

    return PeakReport(
        devices=devices, oom_events=oom_events, trace_available=True, notes=notes
    )


# ------------------------------------------------------------------------------
# 维度 4：碎片化
# ------------------------------------------------------------------------------
_FREE_BLOCK_BUCKET_EDGES: Tuple[int, ...] = (
    4 * 1024,
    64 * 1024,
    1024 * 1024,
    4 * 1024 * 1024,
    16 * 1024 * 1024,
    64 * 1024 * 1024,
    256 * 1024 * 1024,
)


@dataclass
class FragmentationReport:
    """碎片化分析。"""

    free_block_count: int
    free_block_total_bytes: int
    largest_free_block_bytes: int
    external_fragmentation_ratio: float
    internal_waste_bytes: int
    histogram: List[Tuple[str, int, int]]
    worst_segments: List[Tuple[int, int, int, int]]

    def to_dict(self) -> Dict[str, Any]:
        """转为可序列化字典。"""
        return {
            "free_block_count": self.free_block_count,
            "free_block_total_bytes": self.free_block_total_bytes,
            "largest_free_block_bytes": self.largest_free_block_bytes,
            "external_fragmentation_ratio": self.external_fragmentation_ratio,
            "internal_waste_bytes": self.internal_waste_bytes,
            "histogram": [
                {"bucket": label, "count": count, "bytes": total}
                for label, count, total in self.histogram
            ],
            "worst_segments": [
                {
                    "address": address,
                    "total_size": total,
                    "free_bytes": free,
                    "free_block_count": count,
                }
                for address, total, free, count in self.worst_segments
            ],
        }


def _bucket_label(index: int) -> str:
    """生成直方图分桶标签。"""
    if index == 0:
        return f"< {format_bytes(_FREE_BLOCK_BUCKET_EDGES[0])}"
    if index == len(_FREE_BLOCK_BUCKET_EDGES):
        return f">= {format_bytes(_FREE_BLOCK_BUCKET_EDGES[-1])}"
    lower = format_bytes(_FREE_BLOCK_BUCKET_EDGES[index - 1])
    upper = format_bytes(_FREE_BLOCK_BUCKET_EDGES[index])
    return f"{lower} ~ {upper}"


def analyze_fragmentation(snapshot: Snapshot, top_segments: int = 5) -> FragmentationReport:
    """分析空闲块分布与碎片程度。

    关键判据是**最大空闲块**，不是空闲总量。空闲总量 20 GiB 但最大空闲块只有
    200 MiB 时，一次 1 GiB 的分配照样 OOM——这就是典型的碎片型 OOM，和「真的不够用」
    是两个问题、两套解法。

    Args:
        snapshot: 已加载的快照。
        top_segments: 列出空闲最多的前几个 segment。

    Returns:
        碎片化报告。
    """
    free_sizes: List[int] = []
    histogram_counts = [0] * (len(_FREE_BLOCK_BUCKET_EDGES) + 1)
    histogram_bytes = [0] * (len(_FREE_BLOCK_BUCKET_EDGES) + 1)
    segment_rows: List[Tuple[int, int, int, int]] = []
    internal_waste = 0

    for segment in snapshot.segments:
        segment_free_bytes = 0
        segment_free_count = 0
        for block in segment.blocks:
            if block.is_active:
                internal_waste += max(block.size - block.requested_size, 0)
                continue
            free_sizes.append(block.size)
            segment_free_bytes += block.size
            segment_free_count += 1
            bucket_index = len(_FREE_BLOCK_BUCKET_EDGES)
            for index, edge in enumerate(_FREE_BLOCK_BUCKET_EDGES):
                if block.size < edge:
                    bucket_index = index
                    break
            histogram_counts[bucket_index] += 1
            histogram_bytes[bucket_index] += block.size
        if segment_free_bytes:
            segment_rows.append(
                (segment.address, segment.total_size, segment_free_bytes, segment_free_count)
            )

    total_free = sum(free_sizes)
    largest_free = max(free_sizes) if free_sizes else 0
    external_ratio = 1.0 - (largest_free / total_free) if total_free else 0.0
    segment_rows.sort(key=lambda row: row[2], reverse=True)

    histogram = [
        (_bucket_label(index), histogram_counts[index], histogram_bytes[index])
        for index in range(len(histogram_counts))
        if histogram_counts[index]
    ]
    return FragmentationReport(
        free_block_count=len(free_sizes),
        free_block_total_bytes=total_free,
        largest_free_block_bytes=largest_free,
        external_fragmentation_ratio=external_ratio,
        internal_waste_bytes=internal_waste,
        histogram=histogram,
        worst_segments=segment_rows[:top_segments],
    )


# ------------------------------------------------------------------------------
# 维度 5：Top-N 大块
# ------------------------------------------------------------------------------
@dataclass
class TopAllocation:
    """一个大块及其调用栈。"""

    size_bytes: int
    requested_bytes: int
    device: int
    state: str
    segment_type: str
    stream: int
    category: str
    stack: List[str]

    def to_dict(self) -> Dict[str, Any]:
        """转为可序列化字典。"""
        return {
            "size_bytes": self.size_bytes,
            "requested_bytes": self.requested_bytes,
            "device": self.device,
            "state": self.state,
            "segment_type": self.segment_type,
            "stream": self.stream,
            "category": self.category,
            "stack": list(self.stack),
        }


def top_allocations(
    snapshot: Snapshot,
    rules: Sequence[Tuple[str, Sequence[str]]],
    limit: int = 10,
    stack_depth: int = 8,
) -> List[TopAllocation]:
    """列出最大的若干在用块。

    Args:
        snapshot: 已加载的快照。
        rules: 语义分类规则。
        limit: 返回多少条。
        stack_depth: 每条保留几帧调用栈。

    Returns:
        按大小降序的大块列表。
    """
    active_blocks = [block for block in snapshot.blocks if block.is_active]
    active_blocks.sort(key=lambda block: block.size, reverse=True)
    return [
        TopAllocation(
            size_bytes=block.size,
            requested_bytes=block.requested_size,
            device=block.device,
            state=block.state,
            segment_type=block.segment_type,
            stream=block.stream,
            category=classify_block(block, rules),
            stack=[frame.render() for frame in block.frames[:stack_depth]],
        )
        for block in active_blocks[:limit]
    ]


# ------------------------------------------------------------------------------
# 单快照分析汇总
# ------------------------------------------------------------------------------
@dataclass
class SnapshotAnalysis:
    """一份快照的完整分析结果。"""

    label: str
    path: str
    overview: OverviewReport
    categories: CategoryReport
    peak: PeakReport
    fragmentation: FragmentationReport
    top_allocations: List[TopAllocation]
    warnings: List[str]

    def to_dict(self) -> Dict[str, Any]:
        """转为可序列化字典。"""
        return {
            "label": self.label,
            "path": self.path,
            "overview": self.overview.to_dict(),
            "categories": self.categories.to_dict(),
            "peak": self.peak.to_dict(),
            "fragmentation": self.fragmentation.to_dict(),
            "top_allocations": [item.to_dict() for item in self.top_allocations],
            "warnings": list(self.warnings),
        }


def analyze_snapshot(
    snapshot: Snapshot,
    rules: Sequence[Tuple[str, Sequence[str]]],
    *,
    top_n: int = 10,
    stack_depth: int = 8,
    device_capacity_bytes: Optional[int] = None,
) -> SnapshotAnalysis:
    """对一份快照跑完全部静态分析维度。

    Args:
        snapshot: 已加载的快照。
        rules: 语义分类规则。
        top_n: Top-N 大块的 N。
        stack_depth: 调用栈保留深度。
        device_capacity_bytes: 单卡可用显存，用于算占用率。

    Returns:
        完整分析结果。
    """
    return SnapshotAnalysis(
        label=snapshot.label,
        path=snapshot.path,
        overview=summarize_overview(snapshot, device_capacity_bytes),
        categories=classify_snapshot(snapshot, rules),
        peak=analyze_peak(snapshot),
        fragmentation=analyze_fragmentation(snapshot),
        top_allocations=top_allocations(snapshot, rules, top_n, stack_depth),
        warnings=list(snapshot.warnings),
    )


# ------------------------------------------------------------------------------
# 维度 6：双快照对比
# ------------------------------------------------------------------------------
def stack_signature(frames: Sequence[StackFrame], depth: int) -> str:
    """把调用栈压成一个可用于分组的签名。

    取最内层的若干帧——它们离分配点最近，最能区分「哪一处代码在涨」。

    Args:
        frames: 调用栈。
        depth: 取几帧。

    Returns:
        签名串；没有调用栈时返回 ``"<no stack>"``。
    """
    if not frames:
        return "<no stack>"
    return " <- ".join(frame.render() for frame in frames[:depth])


def _signature_totals(snapshot: Snapshot, depth: int) -> Dict[str, Tuple[int, int]]:
    """按调用栈签名汇总在用字节数与块数。"""
    totals: Dict[str, Tuple[int, int]] = {}
    for block in snapshot.blocks:
        if not block.is_active:
            continue
        signature = stack_signature(block.frames, depth)
        current_bytes, current_count = totals.get(signature, (0, 0))
        totals[signature] = (current_bytes + block.size, current_count + 1)
    return totals


@dataclass
class ComparisonReport:
    """两份快照的差异。"""

    baseline_label: str
    candidate_label: str
    reserved_delta_bytes: int
    allocated_delta_bytes: int
    category_deltas: List[Tuple[str, int, int, int]]
    growing_signatures: List[Tuple[str, int, int, int]]
    shrinking_signatures: List[Tuple[str, int, int, int]]
    new_signature_bytes: int
    notes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """转为可序列化字典。"""
        return {
            "baseline_label": self.baseline_label,
            "candidate_label": self.candidate_label,
            "reserved_delta_bytes": self.reserved_delta_bytes,
            "allocated_delta_bytes": self.allocated_delta_bytes,
            "new_signature_bytes": self.new_signature_bytes,
            "category_deltas": [
                {
                    "category": category,
                    "baseline_bytes": baseline,
                    "candidate_bytes": candidate,
                    "delta_bytes": delta,
                }
                for category, baseline, candidate, delta in self.category_deltas
            ],
            "growing_signatures": [
                {
                    "signature": signature,
                    "baseline_bytes": baseline,
                    "candidate_bytes": candidate,
                    "delta_bytes": delta,
                }
                for signature, baseline, candidate, delta in self.growing_signatures
            ],
            "shrinking_signatures": [
                {
                    "signature": signature,
                    "baseline_bytes": baseline,
                    "candidate_bytes": candidate,
                    "delta_bytes": delta,
                }
                for signature, baseline, candidate, delta in self.shrinking_signatures
            ],
            "notes": list(self.notes),
        }


def compare_snapshots(
    baseline: Snapshot,
    candidate: Snapshot,
    rules: Sequence[Tuple[str, Sequence[str]]],
    *,
    signature_depth: int = 8,
    top_n: int = 10,
) -> ComparisonReport:
    """对比两份快照，找出显存涨在哪里。

    Args:
        baseline: 基线快照（通常是较早采的那份）。
        candidate: 对比快照。
        rules: 语义分类规则。
        signature_depth: 调用栈签名深度。深度越大分组越细、越容易因为一点点
            栈差异而分裂；越小越容易把不同来源混成一组。
        top_n: 增长/收缩榜各列几条。

    Returns:
        对比报告。
    """
    baseline_overview = summarize_overview(baseline)
    candidate_overview = summarize_overview(candidate)
    baseline_categories = {
        bucket.category: bucket.total_bytes
        for bucket in classify_snapshot(baseline, rules).buckets
    }
    candidate_categories = {
        bucket.category: bucket.total_bytes
        for bucket in classify_snapshot(candidate, rules).buckets
    }

    category_deltas = [
        (
            category,
            baseline_categories.get(category, 0),
            candidate_categories.get(category, 0),
            candidate_categories.get(category, 0) - baseline_categories.get(category, 0),
        )
        for category in sorted(set(baseline_categories) | set(candidate_categories))
    ]
    category_deltas.sort(key=lambda row: row[3], reverse=True)

    baseline_signatures = _signature_totals(baseline, signature_depth)
    candidate_signatures = _signature_totals(candidate, signature_depth)
    signature_rows: List[Tuple[str, int, int, int]] = []
    new_signature_bytes = 0
    for signature in set(baseline_signatures) | set(candidate_signatures):
        baseline_bytes = baseline_signatures.get(signature, (0, 0))[0]
        candidate_bytes = candidate_signatures.get(signature, (0, 0))[0]
        if signature not in baseline_signatures:
            new_signature_bytes += candidate_bytes
        signature_rows.append(
            (signature, baseline_bytes, candidate_bytes, candidate_bytes - baseline_bytes)
        )
    signature_rows.sort(key=lambda row: row[3], reverse=True)

    notes: List[str] = []
    if baseline.devices != candidate.devices:
        notes.append(
            f"两份快照覆盖的设备不同（{baseline.devices} vs {candidate.devices}），差值可能没有可比性"
        )
    if not any(block.frames for block in baseline.blocks[:200]):
        notes.append("基线快照没有调用栈信息，按签名分组的结果会退化成一个桶")

    return ComparisonReport(
        baseline_label=baseline.label,
        candidate_label=candidate.label,
        reserved_delta_bytes=(
            candidate_overview.total_reserved_bytes - baseline_overview.total_reserved_bytes
        ),
        allocated_delta_bytes=(
            candidate_overview.total_allocated_bytes - baseline_overview.total_allocated_bytes
        ),
        category_deltas=category_deltas,
        growing_signatures=[row for row in signature_rows if row[3] > 0][:top_n],
        shrinking_signatures=[row for row in signature_rows if row[3] < 0][-top_n:][::-1],
        new_signature_bytes=new_signature_bytes,
        notes=notes,
    )


# ------------------------------------------------------------------------------
# 维度 7：多快照泄漏趋势
# ------------------------------------------------------------------------------
def _least_squares_slope(values: Sequence[float]) -> float:
    """对等间隔序列做最小二乘，返回每个采样点的平均增量。

    Args:
        values: 按采样顺序排列的取值。

    Returns:
        斜率；点数不足 2 时返回 0.0。
    """
    count = len(values)
    if count < 2:
        return 0.0
    mean_x = (count - 1) / 2.0
    mean_y = math.fsum(values) / count
    numerator = math.fsum((index - mean_x) * (value - mean_y) for index, value in enumerate(values))
    denominator = math.fsum((index - mean_x) ** 2 for index in range(count))
    return numerator / denominator if denominator else 0.0


def _monotonic_ratio(values: Sequence[float]) -> float:
    """相邻点中「没有下降」的比例。

    Args:
        values: 按采样顺序排列的取值。

    Returns:
        0.0~1.0；点数不足 2 时返回 0.0。
    """
    if len(values) < 2:
        return 0.0
    non_decreasing = sum(
        1 for previous, current in zip(values, values[1:]) if current >= previous
    )
    return non_decreasing / (len(values) - 1)


@dataclass
class LeakSuspect:
    """一条疑似泄漏的调用栈。"""

    signature: str
    series_bytes: List[int]
    slope_bytes_per_snapshot: float
    monotonic_ratio: float
    total_growth_bytes: int
    present_in: int

    def to_dict(self) -> Dict[str, Any]:
        """转为可序列化字典。"""
        return {
            "signature": self.signature,
            "series_bytes": list(self.series_bytes),
            "slope_bytes_per_snapshot": self.slope_bytes_per_snapshot,
            "monotonic_ratio": self.monotonic_ratio,
            "total_growth_bytes": self.total_growth_bytes,
            "present_in": self.present_in,
        }


@dataclass
class TrendReport:
    """多快照的泄漏趋势。"""

    labels: List[str]
    reserved_series: List[int]
    allocated_series: List[int]
    reserved_slope: float
    allocated_slope: float
    allocated_monotonic_ratio: float
    category_series: Dict[str, List[int]]
    category_slopes: Dict[str, float]
    suspects: List[LeakSuspect]
    verdict: str
    notes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """转为可序列化字典。"""
        return {
            "labels": list(self.labels),
            "reserved_series": list(self.reserved_series),
            "allocated_series": list(self.allocated_series),
            "reserved_slope_bytes_per_snapshot": self.reserved_slope,
            "allocated_slope_bytes_per_snapshot": self.allocated_slope,
            "allocated_monotonic_ratio": self.allocated_monotonic_ratio,
            "category_series": {name: list(series) for name, series in self.category_series.items()},
            "category_slopes": dict(self.category_slopes),
            "suspects": [suspect.to_dict() for suspect in self.suspects],
            "verdict": self.verdict,
            "notes": list(self.notes),
        }


def detect_leak_trend(
    snapshots: Sequence[Snapshot],
    rules: Sequence[Tuple[str, Sequence[str]]],
    *,
    signature_depth: int = 8,
    leak_threshold_bytes: int = 1024 * 1024,
    monotonic_threshold: float = 0.8,
    top_n: int = 10,
) -> TrendReport:
    """在多份按时间排列的快照上找泄漏。

    判据是「斜率为正 **且** 基本单调上升」。只看首末差值会把一次性的峰值当成
    泄漏；只看斜率会把剧烈抖动的曲线算进来。两条合用才能把「每个 step 稳定涨一点」
    这种真泄漏挑出来。

    Args:
        snapshots: 按采集时间升序排列的快照。
        rules: 语义分类规则。
        signature_depth: 调用栈签名深度。
        leak_threshold_bytes: 每份快照之间的平均增量超过它才算嫌疑。
        monotonic_threshold: 单调性阈值。
        top_n: 列出几条嫌疑。

    Returns:
        趋势报告。

    Raises:
        ValueError: 快照少于 2 份。
    """
    if len(snapshots) < 2:
        raise ValueError("泄漏趋势分析至少需要 2 份快照")

    notes: List[str] = []
    if len(snapshots) < 3:
        notes.append("只有 2 份快照，趋势判断很不可靠：一次性的峰值和持续泄漏长得一样。建议至少采 4 份")

    labels = [snapshot.label for snapshot in snapshots]
    overviews = [summarize_overview(snapshot) for snapshot in snapshots]
    reserved_series = [overview.total_reserved_bytes for overview in overviews]
    allocated_series = [overview.total_allocated_bytes for overview in overviews]

    category_reports = [classify_snapshot(snapshot, rules) for snapshot in snapshots]
    all_categories = sorted(
        {bucket.category for report in category_reports for bucket in report.buckets}
    )
    category_series: Dict[str, List[int]] = {}
    for category in all_categories:
        category_series[category] = [
            next(
                (bucket.total_bytes for bucket in report.buckets if bucket.category == category),
                0,
            )
            for report in category_reports
        ]
    category_slopes = {
        category: _least_squares_slope([float(value) for value in series])
        for category, series in category_series.items()
    }

    signature_maps = [_signature_totals(snapshot, signature_depth) for snapshot in snapshots]
    all_signatures = {signature for mapping in signature_maps for signature in mapping}
    suspects: List[LeakSuspect] = []
    for signature in all_signatures:
        series = [mapping.get(signature, (0, 0))[0] for mapping in signature_maps]
        slope = _least_squares_slope([float(value) for value in series])
        if slope < leak_threshold_bytes:
            continue
        ratio = _monotonic_ratio([float(value) for value in series])
        if ratio < monotonic_threshold:
            continue
        suspects.append(
            LeakSuspect(
                signature=signature,
                series_bytes=series,
                slope_bytes_per_snapshot=slope,
                monotonic_ratio=ratio,
                total_growth_bytes=series[-1] - series[0],
                present_in=sum(1 for value in series if value > 0),
            )
        )
    suspects.sort(key=lambda suspect: suspect.slope_bytes_per_snapshot, reverse=True)

    allocated_slope = _least_squares_slope([float(value) for value in allocated_series])
    allocated_ratio = _monotonic_ratio([float(value) for value in allocated_series])
    reserved_slope = _least_squares_slope([float(value) for value in reserved_series])

    if allocated_slope >= leak_threshold_bytes and allocated_ratio >= monotonic_threshold:
        verdict = "疑似泄漏：allocated 持续单调上升"
    elif allocated_slope >= leak_threshold_bytes:
        verdict = "allocated 整体在涨但不单调：更像周期性峰值或工作负载变化，先确认采样时机是否对齐同一个 step 阶段"
    elif reserved_slope >= leak_threshold_bytes:
        verdict = "allocated 稳定但 reserved 在涨：更像碎片累积，而不是对象泄漏"
    else:
        verdict = "未发现持续增长：显存占用在这几份快照之间是平的"

    if not any(snapshot.blocks for snapshot in snapshots):
        notes.append("所有快照都没有块数据，趋势结论无效")

    return TrendReport(
        labels=labels,
        reserved_series=reserved_series,
        allocated_series=allocated_series,
        reserved_slope=reserved_slope,
        allocated_slope=allocated_slope,
        allocated_monotonic_ratio=allocated_ratio,
        category_series=category_series,
        category_slopes=category_slopes,
        suspects=suspects[:top_n],
        verdict=verdict,
        notes=notes,
    )


# ------------------------------------------------------------------------------
# profile 读取
# ------------------------------------------------------------------------------
def load_profile_hints(profile_path: Path) -> Tuple[Optional[int], List[str], List[str]]:
    """从硬件 profile 读取分析所需的少量信息。

    读三处：

    * ``devices.models[0].memory_gb`` —— 单卡可用显存，用于算占用率
    * ``collective.library`` / ``collective.torch_backend`` —— 追加到通信类关键字，
      这样自研通信库的调用栈也能被归到通信缓冲
    * ``pytorch.memory_snapshot_format_compatible`` —— 快照格式是否与上游同构

    Args:
        profile_path: profile YAML 路径。

    Returns:
        ``(单卡显存字节数, 追加的通信关键字, 告警列表)``。

    Raises:
        RuntimeError: PyYAML 未安装、文件不存在或不是合法 YAML。
    """
    try:
        import yaml  # 可选依赖：只有用 --profile 时才需要
    except ImportError as error:
        raise RuntimeError(
            "读取 --profile 需要 PyYAML，请先 `pip install pyyaml`，"
            "或用 --device-memory-gb / --category-rules 在命令行直接给值"
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
    capacity_bytes: Optional[int] = None
    devices = document.get("devices")
    if isinstance(devices, Mapping):
        models = devices.get("models")
        if isinstance(models, Sequence) and models:
            first_model = models[0]
            if isinstance(first_model, Mapping):
                memory_gb = first_model.get("memory_gb")
                if isinstance(memory_gb, (int, float)):
                    capacity_bytes = int(float(memory_gb) * (1024**3))
                else:
                    warnings.append(
                        "profile 的 devices.models[0].memory_gb 为空，无法给出「占单卡显存多少」的判断"
                    )
    else:
        warnings.append("profile 缺少 devices 小节，无法算显存占用率")

    extra_keywords: List[str] = []
    collective = document.get("collective")
    if isinstance(collective, Mapping):
        for key in ("library", "torch_backend"):
            value = collective.get(key)
            if isinstance(value, str) and value.strip():
                normalized = value.strip().lower()
                if normalized not in extra_keywords:
                    extra_keywords.append(normalized)
        if not extra_keywords:
            warnings.append(
                "profile 的 collective.library / torch_backend 为空，"
                "通信缓冲只能靠默认关键字（nccl / gloo / c10d 等）识别"
            )
    else:
        warnings.append("profile 缺少 collective 小节，通信缓冲只能靠默认关键字识别")

    pytorch_section = document.get("pytorch")
    if isinstance(pytorch_section, Mapping):
        compatible = pytorch_section.get("memory_snapshot_format_compatible")
        if compatible is False:
            warnings.append(
                "profile 声明 pytorch.memory_snapshot_format_compatible = false："
                "本脚本按上游 PyTorch 的 snapshot 结构解析，结果可能不正确"
            )
        elif compatible is None:
            warnings.append(
                "profile 的 pytorch.memory_snapshot_format_compatible 未填，"
                "无法确认快照格式与上游同构，请先核对一份真实快照"
            )
        elif compatible == "partial":
            warnings.append(
                "profile 声明快照格式与上游只是部分兼容，缺失的字段会退回按块求和估算"
            )
    return capacity_bytes, extra_keywords, warnings


def load_category_rules(rules_path: Path) -> List[Tuple[str, Tuple[str, ...]]]:
    """从 JSON 文件读取自定义分类规则。

    支持两种写法，都保序（顺序即优先级）::

        [{"category": "activation", "keywords": ["forward", "attention"]}, ...]
        {"activation": ["forward", "attention"], "gradient": ["backward"]}

    Args:
        rules_path: 规则文件路径。

    Returns:
        ``[(类别名, (关键字, ...)), ...]``。

    Raises:
        RuntimeError: 文件不存在或格式不对。
    """
    if not rules_path.is_file():
        raise RuntimeError(f"分类规则文件不存在: {rules_path}")
    try:
        document = json.loads(rules_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as error:
        raise RuntimeError(f"分类规则文件无法解析 {rules_path}: {error}") from error

    rules: List[Tuple[str, Tuple[str, ...]]] = []
    if isinstance(document, Mapping):
        items: Iterable[Tuple[str, Any]] = document.items()
        for category, keywords in items:
            if not isinstance(keywords, Sequence) or isinstance(keywords, str):
                raise RuntimeError(f"分类 {category!r} 的关键字必须是列表")
            rules.append((str(category), tuple(str(keyword) for keyword in keywords)))
    elif isinstance(document, Sequence):
        for entry in document:
            if not isinstance(entry, Mapping) or "category" not in entry:
                raise RuntimeError("列表写法的每一项都需要有 category 与 keywords 两个键")
            keywords = entry.get("keywords", [])
            if not isinstance(keywords, Sequence) or isinstance(keywords, str):
                raise RuntimeError(f"分类 {entry['category']!r} 的 keywords 必须是列表")
            rules.append((str(entry["category"]), tuple(str(keyword) for keyword in keywords)))
    else:
        raise RuntimeError("分类规则文件顶层必须是对象或数组")

    if not rules:
        raise RuntimeError("分类规则文件里一条规则都没有")
    return rules


def extend_communication_keywords(
    rules: Sequence[Tuple[str, Sequence[str]]], extra_keywords: Sequence[str]
) -> List[Tuple[str, Tuple[str, ...]]]:
    """把 profile 里的通信库名追加进通信类关键字。

    Args:
        rules: 现有规则。
        extra_keywords: 追加的关键字。

    Returns:
        新的规则列表（原列表不变）。没有 ``communication`` 类时原样返回。
    """
    if not extra_keywords:
        return [(category, tuple(keywords)) for category, keywords in rules]
    extended: List[Tuple[str, Tuple[str, ...]]] = []
    for category, keywords in rules:
        if category == "communication":
            merged = list(keywords)
            for keyword in extra_keywords:
                if keyword not in merged:
                    merged.append(keyword)
            extended.append((category, tuple(merged)))
        else:
            extended.append((category, tuple(keywords)))
    return extended


# ------------------------------------------------------------------------------
# 报告渲染
# ------------------------------------------------------------------------------
def render_analysis(analysis: SnapshotAnalysis, *, stack_depth: int = 8) -> str:
    """渲染单快照的文本报告。

    Args:
        analysis: 分析结果。
        stack_depth: 调用栈最多打印几帧。

    Returns:
        多行文本。
    """
    lines: List[str] = []
    lines.append("=" * 96)
    lines.append(f"快照: {analysis.path}")

    lines.append("")
    lines.append("[1/5] 总览")
    for device in analysis.overview.devices:
        lines.append(f"  设备 {device.device}")
        lines.append(
            f"    reserved  {format_bytes(device.reserved_bytes):>12}"
            + (
                f"   占单卡 {_format_ratio(device.reserved_bytes, device.device_capacity_bytes)}"
                if device.device_capacity_bytes
                else ""
            )
        )
        lines.append(
            f"    allocated {format_bytes(device.allocated_bytes):>12}"
            f"   占 reserved {_format_ratio(device.allocated_bytes, device.reserved_bytes)}"
        )
        lines.append(
            f"    active    {format_bytes(device.active_bytes):>12}"
            f"   其中等待跨流释放 {format_bytes(device.pending_free_bytes)}"
        )
        lines.append(
            f"    requested {format_bytes(device.requested_bytes):>12}"
            f"   取整浪费 {format_bytes(device.internal_waste_bytes)}"
        )
        lines.append(
            f"    碎片率    {device.fragmentation_ratio:>11.1%}"
            f"   (reserved 里闲置 {format_bytes(device.free_in_reserved_bytes)})"
        )
        lines.append(
            f"    segment   {device.segment_count} 个"
            f"（large {device.large_segment_count} / small {device.small_segment_count}"
            f" / 可扩展 {device.expandable_segment_count}）"
        )

    lines.append("")
    lines.append("[2/5] 语义分类（按调用栈关键字，顺序即优先级: " + " > ".join(analysis.categories.rule_order) + "）")
    category_total = analysis.categories.classified_bytes + analysis.categories.unclassified_bytes
    if analysis.categories.buckets:
        lines.append(f"  {'类别':<18}{'占用':>14}{'占比':>9}{'块数':>8}{'最大块':>14}")
        for bucket in analysis.categories.buckets:
            lines.append(
                f"  {bucket.category:<18}{format_bytes(bucket.total_bytes):>14}"
                f"{_format_ratio(bucket.total_bytes, category_total):>9}"
                f"{bucket.block_count:>8}{format_bytes(bucket.largest_block_bytes):>14}"
            )
        unclassified_ratio = (
            analysis.categories.unclassified_bytes / category_total if category_total else 0.0
        )
        if unclassified_ratio > 0.3:
            lines.append(
                f"  注意: 未分类占 {unclassified_ratio:.1%}，分类规则对这份工作负载不够贴合。"
                "用 --category-rules 换一张关键字表会更准"
            )
        if analysis.categories.blocks_without_stack:
            lines.append(
                f"  注意: {analysis.categories.blocks_without_stack} 个块没有调用栈，"
                "采集时需要把栈记录打开才能分类"
            )
    else:
        lines.append("  没有在用的块")

    lines.append("")
    lines.append("[3/5] 峰值时刻")
    if not analysis.peak.trace_available:
        lines.append("  不可用：" + "；".join(analysis.peak.notes))
    else:
        for peak in analysis.peak.devices:
            lines.append(
                f"  设备 {peak.device}: 峰值 allocated {format_bytes(peak.peak_allocated_bytes)}"
                f"，峰值 reserved {format_bytes(peak.peak_reserved_bytes)}"
                f"，trace 末尾 allocated {format_bytes(peak.final_allocated_bytes)}"
                f"（{peak.event_count} 条事件）"
            )
            if peak.peak_allocated_stack:
                when = f"事件 #{peak.peak_allocated_event_index}" + (
                    f" @ {peak.peak_allocated_time_us} us" if peak.peak_allocated_time_us else ""
                )
                lines.append(f"    推到峰值的那次分配（{when}）:")
                for frame in peak.peak_allocated_stack[:stack_depth]:
                    lines.append(f"      {frame}")
            if peak.largest_single_allocation_bytes:
                lines.append(
                    f"    单次最大分配 {format_bytes(peak.largest_single_allocation_bytes)}"
                )
                for frame in peak.largest_single_allocation_stack[:4]:
                    lines.append(f"      {frame}")
        for note in analysis.peak.notes:
            lines.append(f"  注: {note}")
        if analysis.peak.oom_events:
            lines.append(f"  trace 里记录了 {len(analysis.peak.oom_events)} 次分配失败:")
            for event in analysis.peak.oom_events[:5]:
                free_text = (
                    format_bytes(event.device_free_bytes)
                    if event.device_free_bytes is not None
                    else "未知"
                )
                lines.append(
                    f"    设备 {event.device} 申请 {format_bytes(event.requested_bytes)}"
                    f"，当时底层空闲 {free_text}"
                )
                for frame in event.stack[:stack_depth]:
                    lines.append(f"      {frame}")

    lines.append("")
    lines.append("[4/5] 碎片化")
    fragmentation = analysis.fragmentation
    lines.append(
        f"  空闲块 {fragmentation.free_block_count} 个，合计 {format_bytes(fragmentation.free_block_total_bytes)}"
        f"，最大一块 {format_bytes(fragmentation.largest_free_block_bytes)}"
    )
    lines.append(
        f"  外部碎片率 {fragmentation.external_fragmentation_ratio:.1%}"
        "（空闲总量里「不在最大那一块中」的比例，越高说明空闲越零碎）"
    )
    lines.append(f"  取整浪费 {format_bytes(fragmentation.internal_waste_bytes)}（在用块 size 与 requested_size 之差）")
    if fragmentation.histogram:
        lines.append(f"  {'空闲块大小':<28}{'块数':>8}{'合计':>14}")
        for label, count, total in fragmentation.histogram:
            lines.append(f"  {label:<28}{count:>8}{format_bytes(total):>14}")
    if fragmentation.worst_segments:
        lines.append("  空闲最多的 segment:")
        for address, total, free, count in fragmentation.worst_segments:
            lines.append(
                f"    0x{address:x}  总 {format_bytes(total):>12}"
                f"  空闲 {format_bytes(free):>12}  空闲块 {count} 个"
            )

    lines.append("")
    lines.append(f"[5/5] Top-{len(analysis.top_allocations)} 在用大块")
    if analysis.top_allocations:
        for index, allocation in enumerate(analysis.top_allocations, start=1):
            lines.append(
                f"  #{index} {format_bytes(allocation.size_bytes)}"
                f"（请求 {format_bytes(allocation.requested_bytes)}）"
                f"  设备 {allocation.device}  流 {allocation.stream}"
                f"  {allocation.segment_type}/{allocation.state}  分类 {allocation.category}"
            )
            for frame in allocation.stack[:stack_depth]:
                lines.append(f"      {frame}")
    else:
        lines.append("  没有在用的块")

    if analysis.warnings:
        lines.append("")
        lines.append("告警:")
        for warning in analysis.warnings:
            lines.append(f"  - {warning}")
    return "\n".join(lines)


def render_comparison(report: ComparisonReport, *, signature_width: int = 110) -> str:
    """渲染双快照对比报告。

    Args:
        report: 对比结果。
        signature_width: 调用栈签名打印宽度上限。

    Returns:
        多行文本。
    """
    lines: List[str] = []
    lines.append("=" * 96)
    lines.append(f"[6] 双快照对比  基线 {report.baseline_label}  ->  对比 {report.candidate_label}")
    lines.append(f"  reserved  变化 {format_bytes(report.reserved_delta_bytes)}")
    lines.append(f"  allocated 变化 {format_bytes(report.allocated_delta_bytes)}")
    lines.append(f"  基线中不存在的新调用栈共占 {format_bytes(report.new_signature_bytes)}")

    lines.append("")
    lines.append(f"  {'类别':<18}{'基线':>14}{'对比':>14}{'变化':>14}")
    for category, baseline, candidate, delta in report.category_deltas:
        lines.append(
            f"  {category:<18}{format_bytes(baseline):>14}{format_bytes(candidate):>14}"
            f"{format_bytes(delta):>14}"
        )

    if report.growing_signatures:
        lines.append("")
        lines.append("  增长最多的调用栈:")
        for signature, baseline, candidate, delta in report.growing_signatures:
            lines.append(
                f"    +{format_bytes(delta)}  ({format_bytes(baseline)} -> {format_bytes(candidate)})"
            )
            lines.append(f"      {signature[:signature_width]}")
    if report.shrinking_signatures:
        lines.append("")
        lines.append("  下降最多的调用栈:")
        for signature, baseline, candidate, delta in report.shrinking_signatures:
            lines.append(
                f"    {format_bytes(delta)}  ({format_bytes(baseline)} -> {format_bytes(candidate)})"
            )
            lines.append(f"      {signature[:signature_width]}")

    if report.notes:
        lines.append("")
        lines.append("  注:")
        for note in report.notes:
            lines.append(f"    - {note}")
    return "\n".join(lines)


def render_trend(report: TrendReport, *, signature_width: int = 110) -> str:
    """渲染多快照趋势报告。

    Args:
        report: 趋势结果。
        signature_width: 调用栈签名打印宽度上限。

    Returns:
        多行文本。
    """
    lines: List[str] = []
    lines.append("=" * 96)
    lines.append(f"[7] 泄漏趋势（{len(report.labels)} 份快照）")
    lines.append(f"  结论: {report.verdict}")
    lines.append(
        f"  allocated 斜率 {format_bytes(report.allocated_slope)}/快照"
        f"，单调上升比例 {report.allocated_monotonic_ratio:.0%}"
    )
    lines.append(f"  reserved  斜率 {format_bytes(report.reserved_slope)}/快照")

    lines.append("")
    lines.append(f"  {'快照':<28}{'reserved':>14}{'allocated':>14}")
    for label, reserved, allocated in zip(
        report.labels, report.reserved_series, report.allocated_series
    ):
        lines.append(f"  {label[:27]:<28}{format_bytes(reserved):>14}{format_bytes(allocated):>14}")

    growing_categories = [
        (category, slope) for category, slope in report.category_slopes.items() if slope > 0
    ]
    if growing_categories:
        growing_categories.sort(key=lambda row: row[1], reverse=True)
        lines.append("")
        lines.append("  按类别的增长速度:")
        for category, slope in growing_categories:
            series = " -> ".join(format_bytes(value) for value in report.category_series[category])
            lines.append(f"    {category:<18}{format_bytes(slope):>12}/快照   {series}")

    lines.append("")
    if report.suspects:
        lines.append(f"  疑似泄漏点（斜率为正且基本单调，共 {len(report.suspects)} 条）:")
        for index, suspect in enumerate(report.suspects, start=1):
            series = " -> ".join(format_bytes(value) for value in suspect.series_bytes)
            lines.append(
                f"    #{index} {format_bytes(suspect.slope_bytes_per_snapshot)}/快照"
                f"  累计 +{format_bytes(suspect.total_growth_bytes)}"
                f"  单调 {suspect.monotonic_ratio:.0%}"
            )
            lines.append(f"       {series}")
            lines.append(f"       {suspect.signature[:signature_width]}")
    else:
        lines.append("  没有调用栈满足「斜率为正且基本单调」的泄漏判据")

    if report.notes:
        lines.append("")
        lines.append("  注:")
        for note in report.notes:
            lines.append(f"    - {note}")
    return "\n".join(lines)


# ------------------------------------------------------------------------------
# 命令行
# ------------------------------------------------------------------------------
def build_argument_parser() -> argparse.ArgumentParser:
    """构造命令行解析器。

    Returns:
        配置好的 ``ArgumentParser``。
    """
    parser = argparse.ArgumentParser(
        prog="analyze_memory_snapshot.py",
        description=(
            "分析 PyTorch 显存快照：总览 / 语义分类 / 峰值 / 碎片 / Top-N 大块 / "
            "双快照对比 / 多快照泄漏趋势。"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "安全提示:\n"
            "  snapshot 是 pickle 文件，反序列化可执行任意代码。只分析可信来源的文件。\n"
            "  默认使用受限 Unpickler，只还原普通容器类型。\n"
            "\n"
            "退出码:\n"
            f"  {EXIT_SUCCESS}  成功\n"
            f"  {EXIT_GENERAL_ERROR}  未预期错误\n"
            f"  {EXIT_USAGE_ERROR}  命令行用法错误\n"
            f"  {EXIT_INPUT_ERROR}  输入文件或 profile 有问题\n"
            f"  {EXIT_FORMAT_ERROR}  文件不是可识别的 memory snapshot\n"
        ),
    )
    parser.add_argument("snapshots", type=Path, nargs="+", help="一个或多个快照文件（按采集时间升序）")
    parser.add_argument(
        "--mode",
        choices=("auto", "single", "compare", "trend"),
        default="auto",
        help="auto（默认）：1 份走 single，2 份走 compare，3 份及以上走 trend",
    )
    parser.add_argument(
        "--profile",
        type=Path,
        default=None,
        help="硬件 profile YAML，用于读单卡显存与集合通信库名",
    )
    parser.add_argument(
        "--category-rules",
        type=Path,
        default=None,
        help="自定义语义分类关键字表（JSON，顺序即优先级）",
    )
    parser.add_argument(
        "--device-memory-gb",
        type=float,
        default=None,
        help="单卡可用显存（GB），覆盖 profile 里的值，用于算占用率",
    )
    parser.add_argument("--device", type=int, default=None, help="只分析指定设备号")
    parser.add_argument("--top-n", type=int, default=10, help="Top-N 大块与增长榜的条数（默认 10）")
    parser.add_argument("--stack-depth", type=int, default=8, help="调用栈打印深度（默认 8）")
    parser.add_argument(
        "--signature-depth",
        type=int,
        default=8,
        help="调用栈分组签名深度：越深分组越细，越浅越容易混淆来源（默认 8）",
    )
    parser.add_argument(
        "--leak-threshold-mb",
        type=float,
        default=1.0,
        help="泄漏判据：每份快照之间的平均增量下限（MiB，默认 1.0）",
    )
    parser.add_argument(
        "--monotonic-threshold",
        type=float,
        default=0.8,
        help="泄漏判据：相邻快照之间「没有下降」的比例下限（默认 0.8）",
    )
    parser.add_argument("--no-trace", action="store_true", help="跳过 device_traces 解析，只看静态占用")
    parser.add_argument(
        "--max-trace-events",
        type=int,
        default=None,
        metavar="N",
        help="每个设备最多保留 N 条 trace 事件（保留末尾），给超大快照设上限",
    )
    parser.add_argument(
        "--allow-arbitrary-pickle",
        action="store_true",
        help="解除受限 Unpickler。只有在你完全信任文件来源时才用",
    )
    parser.add_argument("--json", action="store_true", help="输出 JSON 而不是文本报告")
    parser.add_argument("--output", "-o", type=Path, default=None, help="把报告写入文件而不是 stdout")
    parser.add_argument("--quiet", action="store_true", help="不把安全提示与告警打到 stderr")
    return parser


def _resolve_mode(requested_mode: str, snapshot_count: int) -> str:
    """根据文件数量决定分析模式。"""
    if requested_mode != "auto":
        return requested_mode
    if snapshot_count == 1:
        return "single"
    if snapshot_count == 2:
        return "compare"
    return "trend"


def main(argv: Optional[Sequence[str]] = None) -> int:
    """命令行入口。

    Args:
        argv: 参数列表，``None`` 表示用 ``sys.argv[1:]``。

    Returns:
        退出码。
    """
    parser = build_argument_parser()
    arguments = parser.parse_args(argv)

    if not arguments.quiet:
        print(f"[安全] {PICKLE_SECURITY_WARNING}", file=sys.stderr)
        if arguments.allow_arbitrary_pickle:
            print(
                "[安全] 已通过 --allow-arbitrary-pickle 解除限制，确认文件来源可信后再继续",
                file=sys.stderr,
            )

    setup_warnings: List[str] = []
    capacity_bytes: Optional[int] = None
    extra_communication_keywords: List[str] = []
    try:
        if arguments.profile is not None:
            capacity_bytes, extra_communication_keywords, setup_warnings = load_profile_hints(
                arguments.profile
            )
        rules: Sequence[Tuple[str, Sequence[str]]] = (
            load_category_rules(arguments.category_rules)
            if arguments.category_rules is not None
            else DEFAULT_CATEGORY_RULES
        )
    except RuntimeError as error:
        print(f"错误: {error}", file=sys.stderr)
        return EXIT_INPUT_ERROR

    rules = extend_communication_keywords(rules, extra_communication_keywords)
    if arguments.device_memory_gb is not None:
        capacity_bytes = int(arguments.device_memory_gb * (1024**3))

    mode = _resolve_mode(arguments.mode, len(arguments.snapshots))
    if mode == "compare" and len(arguments.snapshots) != 2:
        print(f"错误: compare 模式需要正好 2 个快照，收到 {len(arguments.snapshots)} 个", file=sys.stderr)
        return EXIT_USAGE_ERROR
    if mode == "trend" and len(arguments.snapshots) < 2:
        print("错误: trend 模式至少需要 2 个快照", file=sys.stderr)
        return EXIT_USAGE_ERROR

    snapshots: List[Snapshot] = []
    for path in arguments.snapshots:
        if path.is_file() and path.stat().st_size > LARGE_FILE_WARNING_BYTES and not arguments.quiet:
            print(
                f"[告警] {path} 有 {format_bytes(path.stat().st_size)}，加载会比较慢。"
                "只看静态占用可加 --no-trace，或用 --max-trace-events 限量",
                file=sys.stderr,
            )
        try:
            snapshot = load_snapshot(
                path,
                include_trace=not arguments.no_trace,
                max_trace_events=arguments.max_trace_events,
                allow_arbitrary_pickle=arguments.allow_arbitrary_pickle,
            )
        except RuntimeError as error:
            print(f"错误: {error}", file=sys.stderr)
            return EXIT_FORMAT_ERROR
        available_devices = snapshot.devices
        filtered = snapshot.filter_device(arguments.device)
        if arguments.device is not None and not filtered.segments:
            print(
                f"错误: {path} 里没有设备 {arguments.device} 的数据。"
                f"可用设备: {available_devices}",
                file=sys.stderr,
            )
            return EXIT_INPUT_ERROR
        snapshots.append(filtered)

    if not arguments.quiet:
        for warning in setup_warnings:
            print(f"[告警] profile: {warning}", file=sys.stderr)
        for snapshot in snapshots:
            for warning in snapshot.warnings:
                print(f"[告警] {snapshot.label}: {warning}", file=sys.stderr)

    leak_threshold_bytes = int(arguments.leak_threshold_mb * 1024 * 1024)
    payload: Dict[str, Any] = {"mode": mode, "setup_warnings": setup_warnings}
    sections: List[str] = []

    try:
        analyses = [
            analyze_snapshot(
                snapshot,
                rules,
                top_n=arguments.top_n,
                stack_depth=arguments.stack_depth,
                device_capacity_bytes=capacity_bytes,
            )
            for snapshot in snapshots
        ]
        payload["analyses"] = [analysis.to_dict() for analysis in analyses]
        sections.extend(
            render_analysis(analysis, stack_depth=arguments.stack_depth) for analysis in analyses
        )

        if mode == "compare":
            comparison = compare_snapshots(
                snapshots[0],
                snapshots[1],
                rules,
                signature_depth=arguments.signature_depth,
                top_n=arguments.top_n,
            )
            payload["comparison"] = comparison.to_dict()
            sections.append(render_comparison(comparison))
        elif mode == "trend":
            trend = detect_leak_trend(
                snapshots,
                rules,
                signature_depth=arguments.signature_depth,
                leak_threshold_bytes=leak_threshold_bytes,
                monotonic_threshold=arguments.monotonic_threshold,
                top_n=arguments.top_n,
            )
            payload["trend"] = trend.to_dict()
            sections.append(render_trend(trend))
    except ValueError as error:
        print(f"错误: {error}", file=sys.stderr)
        return EXIT_USAGE_ERROR

    report = (
        json.dumps(payload, ensure_ascii=False, indent=2) if arguments.json else "\n\n".join(sections)
    )

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
