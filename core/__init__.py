"""Zoom Q8 Multitrack MOV Merger core package."""

from .merger import (
    STREAM_LABELS,
    ClassifiedFile,
    FFmpegController,
    MergeCancelledError,
    MergeExecutionError,
    StreamKind,
    ValidationReport,
    check_file_writable,
    classify_directory,
    format_size,
    get_ffmpeg_path,
    merge_multitrack,
    natural_sort_key,
    suggest_unique_output_path,
    validate_streams,
)

__all__ = [
    "STREAM_LABELS",
    "ClassifiedFile",
    "FFmpegController",
    "MergeCancelledError",
    "MergeExecutionError",
    "StreamKind",
    "ValidationReport",
    "check_file_writable",
    "classify_directory",
    "format_size",
    "get_ffmpeg_path",
    "merge_multitrack",
    "natural_sort_key",
    "suggest_unique_output_path",
    "validate_streams",
]
