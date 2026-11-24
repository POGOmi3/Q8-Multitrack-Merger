"""File classification, natural sorting, and lossless FFmpeg multitrack muxing logic."""

from __future__ import annotations

import codecs
import ctypes
import os
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Literal, Sequence

StreamKind = Literal["video", "mic", "line", "excluded"]

STREAM_LABELS: dict[StreamKind, str] = {
    "video": "映像 (MOV)",
    "mic": "A1: 内蔵マイク (WAV)",
    "line": "A2: ライン入力 (WAV)",
    "excluded": "対象外 / 未分類",
}

_DIGIT_RE = re.compile(r"(\d+)")
_DURATION_RE = re.compile(r"Duration:\s*(\d+):(\d+):(\d+(?:\.\d+)?)")

# --- ファイル分類の推定用正規表現ルール ---
# 1) 内蔵マイク (A1) と推定する強いキーワード:
#    LR, TrLR, MIC, INTMIC, BUILTIN 等。
#    "XLR" の末尾 "LR" に誤反応しないよう、直前が行頭・記号・数字の場合のみ一致させる。
_MIC_STRONG_RE = re.compile(
    r"(?:^|[_\-\s\.]|(?<=\d))(tr_?lr|lr|mic|int_?mic|builtin)(?:$|[_\-\s\.]|\d)",
    re.IGNORECASE,
)

# 2) ライン入力 (A2) と推定する強いキーワード:
#    Tr12, Tr1-2, IN12, INPUT12, CH12, LINE, XLR, RECOUT 等。
_LINE_STRONG_RE = re.compile(
    r"(?:^|[_\-\s\.]|(?<=\d))"
    r"(tr_?12|tr_?1[\-_]2|in_?12|in_?1[\-_]2|input_?12|ch_?12|line|xlr|rec_?out)"
    r"(?:$|[_\-\s\.]|\d)",
    re.IGNORECASE,
)

# 3) ライン入力の補助パターン:
#    ファイル名末尾付近の "_12" や "_1-2"（例: ZOOM0001_12.WAV, ZOOM0001_12_001.WAV）。
#    テイク番号 "ZOOM0012" に誤反応しないよう、直前に区切り記号を必須とする。
_LINE_WEAK_RE = re.compile(
    r"(?:[_\-\s])(12|1[\-_]2)(?:$|[_\-\s]\d+$)",
    re.IGNORECASE,
)


class MergeCancelledError(Exception):
    """Raised when the user cancels FFmpeg execution."""


class MergeExecutionError(Exception):
    """Raised when FFmpeg fails or files cannot be processed."""


@dataclass
class ClassifiedFile:
    """Represents a single media file and its assigned stream."""

    path: Path
    rel_path: str
    size_bytes: int
    stream: StreamKind
    reason: str


@dataclass
class ValidationReport:
    """Holds pre-execution errors (blocking) and warnings (confirmable)."""

    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


class FFmpegController:
    """Thread-safe controller for cancelling an active FFmpeg subprocess."""

    def __init__(self) -> None:
        self.cancel_event = threading.Event()
        self._lock = threading.Lock()
        self._proc: subprocess.Popen[bytes] | None = None

    @property
    def is_cancelled(self) -> bool:
        return self.cancel_event.is_set()

    def attach_process(self, proc: subprocess.Popen[bytes] | None) -> None:
        with self._lock:
            self._proc = proc
            if proc is not None and self.cancel_event.is_set():
                self._terminate_unlocked(proc)

    def cancel(self) -> None:
        self.cancel_event.set()
        with self._lock:
            if self._proc is not None:
                self._terminate_unlocked(self._proc)

    @staticmethod
    def _terminate_unlocked(proc: subprocess.Popen[bytes]) -> None:
        if proc.poll() is None:
            try:
                proc.terminate()
            except OSError:
                pass


def get_ffmpeg_path() -> Path | None:
    """Locate ffmpeg.exe strictly in the required priority order.

    Search order:
    1. Same folder as .exe (when frozen)
    2. _internal folder inside the .exe folder (PyInstaller onedir)
    3. Folder pointed to by sys._MEIPASS (PyInstaller runtime)
    4. Project root folder during development
    5. System PATH (shutil.which("ffmpeg"))
    """
    candidates: list[Path] = []

    # 1. .exe と同じフォルダ / 2. .exe と同じフォルダ内の _internal フォルダ
    if getattr(sys, "frozen", False):
        exe_dir = Path(sys.executable).resolve().parent
        candidates.append(exe_dir / "ffmpeg.exe")
        candidates.append(exe_dir / "_internal" / "ffmpeg.exe")

    # 3. sys._MEIPASS が指すフォルダ
    meipass = getattr(sys, "_MEIPASS", None)
    if meipass:
        candidates.append(Path(meipass) / "ffmpeg.exe")

    # 4. 開発時のプロジェクトフォルダ
    project_dir = Path(__file__).resolve().parent.parent
    candidates.append(project_dir / "ffmpeg.exe")
    if os.name != "nt":
        candidates.append(project_dir / "ffmpeg")

    for candidate in candidates:
        try:
            if candidate.is_file():
                return candidate.resolve()
        except OSError:
            continue

    # 5. システムの PATH
    which_ffmpeg = shutil.which("ffmpeg")
    if which_ffmpeg:
        return Path(which_ffmpeg).resolve()

    return None


def _natural_tokens(text: str) -> list[int | str]:
    parts = _DIGIT_RE.split(text.lower())
    return [int(part) if part.isdigit() else part for part in parts]


def natural_sort_key(path: Path, base_dir: Path | None = None) -> tuple:
    """Return a sort key using Zoom Q8's split-MOV rule for .MOV and natural order otherwise.

    Zoom Q8 MOV naming rule:
    - 'ZOOMxxxx.MOV' (first segment of recording xxxx) -> (rec_num=xxxx, part_num=0)
    - 'ZOyyxxxx.MOV' (split segment yy of recording xxxx) -> (rec_num=xxxx, part_num=yy)
    - Non-matching .MOV files and all .WAV files fall back to conventional natural sort.
    """
    try:
        rel = path.relative_to(base_dir) if base_dir else Path(path.name)
    except ValueError:
        rel = Path(path.name)

    parent_key = tuple(
        _natural_tokens(part) for part in rel.parent.parts if part not in ("", ".")
    )
    stem_key = _natural_tokens(rel.stem)
    suffix_key = rel.suffix.lower()

    priority_group = 1
    q8_mov_key = (0, 0, 0)

    if suffix_key == ".mov":
        m_first = re.fullmatch(r"zoom([0-9]{4})", rel.stem, re.IGNORECASE)
        if m_first:
            priority_group = 0
            q8_mov_key = (int(m_first.group(1)), 0, 0)
        else:
            m_split = re.fullmatch(r"zo([0-9]{2})([0-9]{4})", rel.stem, re.IGNORECASE)
            if m_split:
                priority_group = 0
                q8_mov_key = (int(m_split.group(2)), int(m_split.group(1)), 1)

    return (
        parent_key,
        priority_group,
        q8_mov_key,
        stem_key,
        suffix_key,
        rel.as_posix().lower(),
    )


def format_size(size_bytes: int) -> str:
    """Format byte count into human-readable units."""
    value = float(size_bytes)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if value < 1024.0 or unit == "TB":
            if unit == "B":
                return f"{int(value)} B"
            return f"{value:.2f} {unit}"
        value /= 1024.0
    return f"{size_bytes} B"


def _classify_single_wav(path: Path, input_dir: Path) -> tuple[StreamKind, str]:
    """Infer whether a WAV file belongs to Mic (A1), Line (A2), or Excluded/Unassigned."""
    stem = path.stem

    mic_match = _MIC_STRONG_RE.search(stem)
    line_match = _LINE_STRONG_RE.search(stem)

    if mic_match and not line_match:
        return "mic", f"ファイル名のパターン「{mic_match.group(1)}」から内蔵マイクと推定"
    if line_match and not mic_match:
        return "line", f"ファイル名のパターン「{line_match.group(1)}」からライン入力と推定"
    if mic_match and line_match:
        return (
            "excluded",
            "ファイル名にマイク系・ライン系両方のパターンが含まれるため未分類（手動で選択してください）",
        )

    weak_line_match = _LINE_WEAK_RE.search(stem)
    if weak_line_match:
        return "line", f"ファイル名末尾の「{weak_line_match.group(1)}」からライン入力と推定"

    # ファイル名で判別できない場合、サブフォルダ名も確認する
    try:
        rel_parent = path.parent.relative_to(input_dir).as_posix()
    except ValueError:
        rel_parent = ""

    if rel_parent and rel_parent != ".":
        folder_mic = _MIC_STRONG_RE.search(rel_parent)
        folder_line = _LINE_STRONG_RE.search(rel_parent)
        if folder_mic and not folder_line:
            return "mic", f"親フォルダ名「{rel_parent}」から内蔵マイクと推定"
        if folder_line and not folder_mic:
            return "line", f"親フォルダ名「{rel_parent}」からライン入力と推定"

    return (
        "excluded",
        "ファイル名からマイク/ラインを特定できないため未分類（手動で振り分けてください）",
    )


def classify_directory(input_dir: Path) -> list[ClassifiedFile]:
    """Scan a directory and classify .MOV / .WAV files into 3 streams + excluded.

    Classification rules:
    - Hidden/temporary files starting with '.' or '~$' are ignored.
    - Non-.mov/.wav files (e.g. .THM, .CTG) are ignored.
    - '*_multitrack.mov' files are placed in 'excluded' to avoid re-merging past outputs.
    - Other '.mov' files -> 'video'.
    - '.wav' files -> matched against Mic / Line patterns; ambiguous files -> 'excluded'.
    """
    resolved_dir = input_dir.resolve()
    if not resolved_dir.is_dir():
        raise MergeExecutionError(f"指定された録画フォルダが見つかりません: {resolved_dir}")

    collected: list[Path] = []
    for entry in resolved_dir.rglob("*"):
        if not entry.is_file():
            continue
        if any(
            part.startswith(".") or part.startswith("~$")
            for part in entry.relative_to(resolved_dir).parts
        ):
            continue
        ext = entry.suffix.lower()
        if ext in (".mov", ".wav"):
            collected.append(entry)

    collected.sort(key=lambda p: natural_sort_key(p, resolved_dir))

    results: list[ClassifiedFile] = []
    for file_path in collected:
        rel_str = file_path.relative_to(resolved_dir).as_posix()
        try:
            size_bytes = file_path.stat().st_size
        except OSError:
            size_bytes = 0

        ext = file_path.suffix.lower()
        if ext == ".mov":
            if file_path.stem.lower().endswith("_multitrack"):
                stream: StreamKind = "excluded"
                reason = "過去の出力ファイル（*_multitrack.mov）と判定し対象外に分類"
            else:
                stream = "video"
                reason = "拡張子 .MOV のため映像に分類"
        else:
            stream, reason = _classify_single_wav(file_path, resolved_dir)

        results.append(
            ClassifiedFile(
                path=file_path,
                rel_path=rel_str,
                size_bytes=size_bytes,
                stream=stream,
                reason=reason,
            )
        )

    return results


def _get_windows_filesystem_type(target_path: Path) -> str | None:
    """Return filesystem name (e.g. 'NTFS', 'FAT32', 'exFAT') on Windows, or None."""
    if sys.platform != "win32":
        return None
    try:
        anchor = target_path.resolve().anchor
        if not anchor:
            return None
        fs_name_buf = ctypes.create_unicode_buffer(256)
        vol_name_buf = ctypes.create_unicode_buffer(256)
        ok = ctypes.windll.kernel32.GetVolumeInformationW(
            ctypes.c_wchar_p(anchor),
            vol_name_buf,
            len(vol_name_buf),
            None,
            None,
            None,
            fs_name_buf,
            len(fs_name_buf),
        )
        if ok:
            return fs_name_buf.value.upper()
    except Exception:
        return None
    return None


def suggest_unique_output_path(output_path: Path) -> Path:
    """Return a non-existing path by appending _1, _2, ... to the stem."""
    if not output_path.exists():
        return output_path
    parent = output_path.parent
    stem = output_path.stem
    suffix = output_path.suffix or ".mov"
    counter = 1
    while True:
        candidate = parent / f"{stem}_{counter}{suffix}"
        if not candidate.exists():
            return candidate
        counter += 1


def check_file_writable(target_path: Path) -> str | None:
    """Check if an existing output file is locked by another process (e.g. DaVinci Resolve)."""
    if not target_path.exists():
        return None
    try:
        with target_path.open("r+b"):
            pass
        return None
    except PermissionError:
        return (
            f"出力先ファイルが他のアプリ（DaVinci Resolve 等）で開かれているため上書きできません:\n"
            f"{target_path}"
        )
    except OSError as exc:
        return f"出力先ファイルにアクセスできません: {exc}"


def validate_streams(
    video_files: Sequence[Path],
    mic_files: Sequence[Path],
    line_files: Sequence[Path],
    excluded_files: Sequence[Path],
    output_path: Path,
    input_dir: Path | None = None,
) -> ValidationReport:
    """Validate stream assignments and output destination before starting FFmpeg."""
    report = ValidationReport()

    if not video_files and not mic_files and not line_files:
        report.errors.append("結合対象のファイルが1つも割り当てられていません。")
        return report

    resolved_output = output_path.resolve()
    all_known_inputs = {
        p.resolve() for p in (*video_files, *mic_files, *line_files, *excluded_files)
    }
    if resolved_output in all_known_inputs:
        report.errors.append(
            "出力ファイル名が元の入力ファイルと一致しています。元ファイル保護のため別名を指定してください。"
        )
        return report

    # 各系統 0件チェック（確認のうえ続行可能）
    if len(video_files) == 0:
        report.warnings.append("「映像 (MOV)」が 0 件です（音声のみの出力になります）。")
    if len(mic_files) == 0:
        report.warnings.append("「A1: 内蔵マイク (WAV)」が 0 件です。")
    if len(line_files) == 0:
        report.warnings.append("「A2: ライン入力 (WAV)」が 0 件です。")

    # ファイル数・サイズバランスの不自然さチェック
    if len(mic_files) > 0 and len(line_files) > 0:
        if len(mic_files) != len(line_files):
            report.warnings.append(
                f"内蔵マイク WAV（{len(mic_files)}件）とライン入力 WAV（{len(line_files)}件）のファイル数が一致していません。"
            )
        mic_bytes = sum(_safe_size(p) for p in mic_files)
        line_bytes = sum(_safe_size(p) for p in line_files)
        max_audio = max(mic_bytes, line_bytes)
        if max_audio > 0 and abs(mic_bytes - line_bytes) / max_audio > 0.10:
            report.warnings.append(
                f"内蔵マイク WAV（{format_size(mic_bytes)}）とライン入力 WAV（{format_size(line_bytes)}）の合計サイズに 10% 以上の差があります。"
            )

    max_wav_count = max(len(mic_files), len(line_files))
    if len(video_files) > 0 and max_wav_count > len(video_files):
        report.warnings.append(
            f"映像 MOV（{len(video_files)}件）よりも音声 WAV（{max_wav_count}件）の方がファイル数が多く、分割ファイルの不足または別テイク混在の可能性があります。"
        )

    # 拡張子と系統の不一致チェック
    bad_video_ext = [p.name for p in video_files if p.suffix.lower() != ".mov"]
    bad_mic_ext = [p.name for p in mic_files if p.suffix.lower() != ".wav"]
    bad_line_ext = [p.name for p in line_files if p.suffix.lower() != ".wav"]
    if bad_video_ext or bad_mic_ext or bad_line_ext:
        report.warnings.append(
            "系統と拡張子の組み合わせが通常と異なるファイルが含まれています（映像枠の WAV または音声枠の MOV）。"
        )

    # 未分類の WAV / MOV が残っていないかチェック（過去の *_multitrack.mov は除く）
    unmerged_candidates = [
        p
        for p in excluded_files
        if not (p.suffix.lower() == ".mov" and p.stem.lower().endswith("_multitrack"))
    ]
    if unmerged_candidates:
        names_preview = ", ".join(p.name for p in unmerged_candidates[:3])
        extra = f" ほか{len(unmerged_candidates) - 3}件" if len(unmerged_candidates) > 3 else ""
        report.warnings.append(
            f"「対象外 / 未分類」にファイルが {len(unmerged_candidates)} 件残っています（{names_preview}{extra}）。これらは結合されません。"
        )

    # FAT32 出力先の 4GB 制限チェック
    total_input_bytes = sum(
        _safe_size(p) for p in (*video_files, *mic_files, *line_files)
    )
    fs_type = _get_windows_filesystem_type(output_path.parent)
    if fs_type in ("FAT", "FAT32") and total_input_bytes > 3_900_000_000:
        report.warnings.append(
            f"出力先ドライブが {fs_type} フォーマットのため、4GB を超える MOV ファイルの書き込みに失敗する可能性が高いです。PC 内蔵ドライブ（NTFS / exFAT）への出力を推奨します。"
        )

    # 空き容量チェック（出力先フォルダ内の中間ファイル＋最終出力＝入力合計の約2倍が必要）
    required_bytes = total_input_bytes * 2
    try:
        out_check_dir = resolved_output.parent
        while not out_check_dir.exists() and out_check_dir != out_check_dir.parent:
            out_check_dir = out_check_dir.parent
        free_out = shutil.disk_usage(out_check_dir).free
        if total_input_bytes > 0 and free_out < required_bytes:
            report.warnings.append(
                f"出力先ドライブの空き容量が不足する可能性があります（現在の空き: {format_size(free_out)} / "
                f"必要目安: 約 {format_size(required_bytes)}）。"
                f"出力先フォルダ内に中間ファイルと最終出力の両方を作成するため、"
                f"入力合計（{format_size(total_input_bytes)}）の約2倍の空き容量が必要です。"
            )
    except OSError:
        pass

    return report


def _safe_size(path: Path) -> int:
    try:
        return path.stat().st_size
    except OSError:
        return 0


def write_concat_list(files: Sequence[Path], list_path: Path) -> None:
    """Write FFmpeg concat demuxer list file in UTF-8 with single-quote escaping."""
    with list_path.open("w", encoding="utf-8", newline="\n") as fh:
        for file_path in files:
            posix_path = file_path.resolve().as_posix()
            escaped = posix_path.replace("'", r"'\''")
            fh.write(f"file '{escaped}'\n")


def _cleanup_temp_dir(temp_dir: Path) -> None:
    """Remove temporary folder reliably even if Windows briefly holds file handles."""
    for _ in range(5):
        if not temp_dir.exists():
            return
        try:
            shutil.rmtree(temp_dir)
            return
        except OSError:
            time.sleep(0.15)
    shutil.rmtree(temp_dir, ignore_errors=True)


def _run_ffmpeg_step(
    cmd: list[str],
    controller: FFmpegController,
    stage_label: str,
    stage_start_pct: int,
    stage_end_pct: int,
    expected_bytes: int,
    on_log: Callable[[str], None] | None,
    on_progress: Callable[[int, str], None] | None,
) -> None:
    """Execute a single FFmpeg command without console windows and stream realtime progress/logs."""
    if controller.is_cancelled:
        raise MergeCancelledError("ユーザーによって処理がキャンセルされました。")

    if on_log:
        on_log(f"\n=== {stage_label} ===")
        on_log("実行コマンド: " + " ".join(f'"{arg}"' if " " in arg else arg for arg in cmd))

    if on_progress:
        on_progress(stage_start_pct, f"{stage_label} (0%)")

    creationflags = subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0
    startupinfo = None
    if sys.platform == "win32":
        startupinfo = subprocess.STARTUPINFO()
        startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
        startupinfo.wShowWindow = subprocess.SW_HIDE

    try:
        proc = subprocess.Popen(
            cmd,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            creationflags=creationflags,
            startupinfo=startupinfo,
        )
    except OSError as exc:
        raise MergeExecutionError(f"FFmpeg の起動に失敗しました: {exc}") from exc

    controller.attach_process(proc)

    stderr_tail: list[str] = []
    detected_durations_us: list[int] = []
    stderr_lock = threading.Lock()

    def _read_stderr() -> None:
        assert proc.stderr is not None
        decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
        buf = ""
        last_cr_emit = 0.0

        def _handle_chunk(text_chunk: str, final: bool = False) -> None:
            nonlocal buf, last_cr_emit
            buf += text_chunk
            while True:
                idx_n = buf.find("\n")
                idx_r = buf.find("\r")
                if idx_n == -1 and idx_r == -1:
                    break

                if idx_r != -1 and (idx_n == -1 or idx_r < idx_n):
                    # \r\n の場合はまとめて改行として処理
                    if idx_r + 1 < len(buf) and buf[idx_r + 1] == "\n":
                        line = buf[:idx_r]
                        buf = buf[idx_r + 2 :]
                        is_cr_only = False
                    elif idx_r + 1 == len(buf) and not final:
                        # 次のバイトが \n かもしれないため次チャンクを待つ
                        break
                    else:
                        line = buf[:idx_r]
                        buf = buf[idx_r + 1 :]
                        is_cr_only = True
                else:
                    line = buf[:idx_n]
                    buf = buf[idx_n + 1 :]
                    is_cr_only = False

                clean_line = line.rstrip()
                if not clean_line:
                    continue

                dur_match = _DURATION_RE.search(clean_line)
                if dur_match:
                    hours = int(dur_match.group(1))
                    minutes = int(dur_match.group(2))
                    seconds = float(dur_match.group(3))
                    dur_us = int((hours * 3600 + minutes * 60 + seconds) * 1_000_000)
                    if dur_us > 0:
                        with stderr_lock:
                            detected_durations_us.append(dur_us)

                if is_cr_only:
                    now = time.monotonic()
                    if now - last_cr_emit >= 1.0:
                        last_cr_emit = now
                        if on_log:
                            on_log(clean_line)
                else:
                    with stderr_lock:
                        stderr_tail.append(clean_line)
                        if len(stderr_tail) > 30:
                            stderr_tail.pop(0)
                    if on_log:
                        on_log(clean_line)

            if final and buf.strip():
                clean_rem = buf.strip()
                with stderr_lock:
                    stderr_tail.append(clean_rem)
                    if len(stderr_tail) > 30:
                        stderr_tail.pop(0)
                if on_log:
                    on_log(clean_rem)
                buf = ""

        try:
            while True:
                raw = proc.stderr.read1(4096)  # type: ignore[attr-defined]
                if not raw:
                    rem = decoder.decode(b"", final=True)
                    _handle_chunk(rem, final=True)
                    break
                decoded = decoder.decode(raw, final=False)
                _handle_chunk(decoded, final=False)
        except Exception:
            pass

    stderr_thread = threading.Thread(target=_read_stderr, daemon=True)
    stderr_thread.start()

    out_time_us = 0
    total_size = 0
    try:
        assert proc.stdout is not None
        for raw_line in proc.stdout:
            if controller.is_cancelled:
                break
            line_str = raw_line.decode("utf-8", errors="replace").strip()
            if "=" not in line_str:
                continue
            key, val = line_str.split("=", 1)
            key = key.strip()
            val = val.strip()

            if key == "out_time_us":
                try:
                    out_time_us = max(0, int(val))
                except ValueError:
                    pass
            elif key == "total_size":
                try:
                    total_size = max(0, int(val))
                except ValueError:
                    pass
            elif key == "progress":
                if val == "end":
                    if on_progress:
                        on_progress(stage_end_pct, f"{stage_label} (100%)")
                    continue

                with stderr_lock:
                    max_dur_us = max(detected_durations_us) if detected_durations_us else 0

                ratio = 0.0
                if max_dur_us > 0 and out_time_us > 0:
                    ratio = min(0.99, out_time_us / max_dur_us)
                elif expected_bytes > 0 and total_size > 0:
                    ratio = min(0.99, total_size / expected_bytes)

                overall_pct = stage_start_pct + int(
                    (stage_end_pct - stage_start_pct) * ratio
                )
                if on_progress:
                    on_progress(overall_pct, f"{stage_label} ({int(ratio * 100)}%)")
    finally:
        stderr_thread.join(timeout=3.0)
        proc.wait()
        controller.attach_process(None)

    if controller.is_cancelled:
        raise MergeCancelledError("ユーザーによって処理がキャンセルされました。")

    if proc.returncode != 0:
        with stderr_lock:
            err_detail = "\n".join(stderr_tail[-10:]) if stderr_tail else f"終了コード: {proc.returncode}"
        raise MergeExecutionError(f"{stage_label} に失敗しました。\n{err_detail}")


def merge_multitrack(
    ffmpeg_path: Path,
    video_files: Sequence[Path],
    mic_files: Sequence[Path],
    line_files: Sequence[Path],
    output_path: Path,
    controller: FFmpegController,
    on_log: Callable[[str], None] | None = None,
    on_progress: Callable[[int, str], None] | None = None,
) -> Path:
    """Concatenate split Q8 files losslessly and mux into a single multitrack MOV.

    All intermediate files are created in a temporary folder inside the output directory
    (same drive) and deleted in `finally` regardless of success, error, or cancellation.
    """
    if not video_files and not mic_files and not line_files:
        raise MergeExecutionError("結合する入力ファイルがありません。")

    resolved_output = output_path.resolve()
    try:
        resolved_output.parent.mkdir(parents=True, exist_ok=True)
        temp_dir = Path(
            tempfile.mkdtemp(
                prefix=".q8_multitrack_tmp_",
                dir=str(resolved_output.parent),
            )
        )
    except OSError as exc:
        raise MergeExecutionError(
            f"出力先フォルダに一時作業フォルダを作成できませんでした: {exc}"
        ) from exc

    part_output_path = resolved_output.with_name(f".{resolved_output.stem}_writing.mov")

    try:
        active_streams: list[tuple[str, str, Sequence[Path], Path, list[str]]] = []

        if video_files:
            video_list = temp_dir / "concat_video.txt"
            write_concat_list(video_files, video_list)
            temp_video = temp_dir / "temp_video.mov"
            video_cmd = [
                str(ffmpeg_path),
                "-nostdin",
                "-y",
                "-f",
                "concat",
                "-safe",
                "0",
                "-i",
                str(video_list),
                "-map",
                "0:v:0",
                "-c",
                "copy",
                "-progress",
                "pipe:1",
                str(temp_video),
            ]
            active_streams.append(("video", "映像 MOV の無劣化結合", video_files, temp_video, video_cmd))

        if mic_files:
            mic_list = temp_dir / "concat_mic.txt"
            write_concat_list(mic_files, mic_list)
            temp_mic = temp_dir / "temp_mic.wav"
            mic_cmd = [
                str(ffmpeg_path),
                "-nostdin",
                "-y",
                "-f",
                "concat",
                "-safe",
                "0",
                "-i",
                str(mic_list),
                "-map",
                "0:a:0",
                "-c",
                "copy",
                "-rf64",
                "auto",
                "-progress",
                "pipe:1",
                str(temp_mic),
            ]
            active_streams.append(("mic", "A1: 内蔵マイク WAV の無劣化結合", mic_files, temp_mic, mic_cmd))

        if line_files:
            line_list = temp_dir / "concat_line.txt"
            write_concat_list(line_files, line_list)
            temp_line = temp_dir / "temp_line.wav"
            line_cmd = [
                str(ffmpeg_path),
                "-nostdin",
                "-y",
                "-f",
                "concat",
                "-safe",
                "0",
                "-i",
                str(line_list),
                "-map",
                "0:a:0",
                "-c",
                "copy",
                "-rf64",
                "auto",
                "-progress",
                "pipe:1",
                str(temp_line),
            ]
            active_streams.append(("line", "A2: ライン入力 WAV の無劣化結合", line_files, temp_line, line_cmd))

        total_steps = len(active_streams) + 1
        concat_pct_budget = 70
        per_concat_pct = max(1, concat_pct_budget // max(1, len(active_streams)))

        # Step 1..N: 系統ごとの concat 実行
        for idx, (_, title, src_files, _, cmd) in enumerate(active_streams):
            start_pct = idx * per_concat_pct
            end_pct = (idx + 1) * per_concat_pct if idx < len(active_streams) - 1 else concat_pct_budget
            expected_bytes = sum(_safe_size(p) for p in src_files)
            step_label = f"[{idx + 1}/{total_steps}] {title}"
            _run_ffmpeg_step(
                cmd=cmd,
                controller=controller,
                stage_label=step_label,
                stage_start_pct=start_pct,
                stage_end_pct=end_pct,
                expected_bytes=expected_bytes,
                on_log=on_log,
                on_progress=on_progress,
            )

        # Final Step: 中間ファイルを 1本のマルチトラック MOV に統合（Mux）
        # 通常時（3系統すべて存在）は -map 0:v:0 -map 1:a:0 -map 2:a:0 -c copy となる
        mux_cmd: list[str] = [str(ffmpeg_path), "-nostdin", "-y"]
        map_args: list[str] = []
        expected_mux_bytes = 0

        for input_idx, (kind, _, _, temp_media, _) in enumerate(active_streams):
            mux_cmd.extend(["-i", str(temp_media)])
            expected_mux_bytes += _safe_size(temp_media)
            if kind == "video":
                map_args.extend(["-map", f"{input_idx}:v:0"])
            else:
                map_args.extend(["-map", f"{input_idx}:a:0"])

        mux_cmd.extend(map_args)
        mux_cmd.extend(["-c", "copy", "-progress", "pipe:1", str(part_output_path)])

        mux_label = f"[{total_steps}/{total_steps}] マルチトラック MOV への統合 (Mux)"
        _run_ffmpeg_step(
            cmd=mux_cmd,
            controller=controller,
            stage_label=mux_label,
            stage_start_pct=concat_pct_budget,
            stage_end_pct=99,
            expected_bytes=expected_mux_bytes,
            on_log=on_log,
            on_progress=on_progress,
        )

        try:
            os.replace(part_output_path, resolved_output)
        except PermissionError as exc:
            raise MergeExecutionError(
                f"出力先ファイルが他のアプリ（DaVinci Resolve 等）で使用中のため書き込めませんでした:\n{resolved_output}"
            ) from exc
        except OSError as exc:
            raise MergeExecutionError(f"最終ファイルの保存に失敗しました: {exc}") from exc

        if on_progress:
            on_progress(100, "完了")
        if on_log:
            on_log(f"\n=== 処理完了 ===\n出力先: {resolved_output}")

        return resolved_output

    finally:
        if part_output_path.exists():
            try:
                part_output_path.unlink()
            except OSError:
                pass
        _cleanup_temp_dir(temp_dir)
