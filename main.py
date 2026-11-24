"""GUI entry point for Zoom Q8 Multitrack MOV Merger."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Sequence

from PySide6.QtCore import QPoint, Qt, QThread, Signal
from PySide6.QtGui import QCloseEvent, QDragEnterEvent, QDropEvent
from PySide6.QtWidgets import (
    QAbstractItemView,
    QApplication,
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMenu,
    QMessageBox,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from core.merger import (
    STREAM_LABELS,
    ClassifiedFile,
    FFmpegController,
    MergeCancelledError,
    MergeExecutionError,
    StreamKind,
    check_file_writable,
    classify_directory,
    format_size,
    get_ffmpeg_path,
    merge_multitrack,
    natural_sort_key,
    suggest_unique_output_path,
    validate_streams,
)

APP_STYLESHEET = """
QMainWindow, QWidget {
    background-color: #232323;
    color: #E6E6E6;
    font-family: "Yu Gothic UI", "Meiryo UI", sans-serif;
    font-size: 13px;
}
QGroupBox {
    border: 1px solid #3A3A3A;
    border-radius: 6px;
    margin-top: 10px;
    padding-top: 8px;
    font-weight: bold;
}
QGroupBox::title {
    subcontrol-origin: margin;
    left: 10px;
    padding: 0 4px;
    color: #EAEAEA;
}
QLineEdit {
    background-color: #1A1A1A;
    border: 1px solid #444444;
    border-radius: 4px;
    padding: 5px 8px;
    color: #F0F0F0;
}
QLineEdit:focus {
    border: 1px solid #1473E6;
}
QListWidget {
    background-color: #191919;
    border: 1px solid #383838;
    border-radius: 4px;
    padding: 4px;
}
QListWidget::item {
    padding: 4px 6px;
    border-radius: 3px;
}
QListWidget::item:selected {
    background-color: #1473E6;
    color: #FFFFFF;
}
QPlainTextEdit {
    background-color: #141414;
    border: 1px solid #383838;
    border-radius: 4px;
    color: #CCCCCC;
    font-family: "Consolas", "Yu Gothic UI", monospace;
    font-size: 12px;
}
QPushButton {
    background-color: #343434;
    border: 1px solid #4A4A4A;
    border-radius: 4px;
    padding: 6px 12px;
    color: #EAEAEA;
}
QPushButton:hover {
    background-color: #424242;
}
QPushButton:disabled {
    background-color: #2A2A2A;
    border-color: #333333;
    color: #6E6E6E;
}
QPushButton#primaryButton {
    background-color: #1473E6;
    border: 1px solid #1473E6;
    color: #FFFFFF;
    font-weight: bold;
    padding: 8px 20px;
}
QPushButton#primaryButton:hover {
    background-color: #1982FF;
}
QPushButton#primaryButton:disabled {
    background-color: #2A3F58;
    border-color: #2A3F58;
    color: #8899AA;
}
QPushButton#dangerButton {
    background-color: #5A2828;
    border: 1px solid #7A3838;
    color: #FFDADA;
}
QPushButton#dangerButton:hover {
    background-color: #743232;
}
QPushButton#dangerButton:disabled {
    background-color: #2A2A2A;
    border-color: #333333;
    color: #6E6E6E;
}
QProgressBar {
    background-color: #181818;
    border: 1px solid #3A3A3A;
    border-radius: 4px;
    text-align: center;
    height: 20px;
    color: #FFFFFF;
}
QProgressBar::chunk {
    background-color: #1473E6;
    border-radius: 3px;
}
"""


class MergeWorker(QThread):
    """Background worker thread running FFmpeg concat and multitrack mux."""

    progress_changed = Signal(int, str)
    log_received = Signal(str)
    finished_ok = Signal(str)
    failed = Signal(str)
    cancelled = Signal()

    def __init__(
        self,
        ffmpeg_path: Path,
        video_files: Sequence[Path],
        mic_files: Sequence[Path],
        line_files: Sequence[Path],
        output_path: Path,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.ffmpeg_path = ffmpeg_path
        self.video_files = list(video_files)
        self.mic_files = list(mic_files)
        self.line_files = list(line_files)
        self.output_path = output_path
        self.controller = FFmpegController()

    def request_cancel(self) -> None:
        self.controller.cancel()

    def run(self) -> None:
        try:
            result_path = merge_multitrack(
                ffmpeg_path=self.ffmpeg_path,
                video_files=self.video_files,
                mic_files=self.mic_files,
                line_files=self.line_files,
                output_path=self.output_path,
                controller=self.controller,
                on_log=self.log_received.emit,
                on_progress=self.progress_changed.emit,
            )
            self.finished_ok.emit(str(result_path))
        except MergeCancelledError:
            self.cancelled.emit()
        except MergeExecutionError as exc:
            self.failed.emit(str(exc))
        except Exception as exc:
            self.failed.emit(f"予期しないエラーが発生しました: {exc}")


class MainWindow(QMainWindow):
    """Single-window GUI for Zoom Q8 Multitrack MOV Merger."""

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Zoom Q8 Multitrack MOV Merger")
        self.resize(1120, 780)
        self.setMinimumSize(880, 640)
        self.setAcceptDrops(True)

        self._input_dir: Path | None = None
        self._worker: MergeWorker | None = None
        self._stream_lists: dict[StreamKind, QListWidget] = {}
        self._stream_count_labels: dict[StreamKind, QLabel] = {}
        self._selection_sync_guard = False

        self._build_ui()
        self._update_ffmpeg_indicator()

    def _build_ui(self) -> None:
        central = QWidget(self)
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setContentsMargins(14, 12, 14, 12)
        root.setSpacing(8)

        # --- 上部: 録画フォルダ選択エリア ---
        input_group = QGroupBox("1. 録画フォルダの選択（ウィンドウへのドラッグ＆ドロップ対応）")
        input_layout = QVBoxLayout(input_group)

        folder_row = QHBoxLayout()
        self.input_dir_edit = QLineEdit()
        self.input_dir_edit.setReadOnly(True)
        self.input_dir_edit.setPlaceholderText(
            "Q8 の録画フォルダを選択するか、ここへフォルダをドロップしてください"
        )
        self.browse_input_btn = QPushButton("フォルダを選択...")
        self.browse_input_btn.clicked.connect(self._on_browse_input)
        self.rescan_btn = QPushButton("再スキャン")
        self.rescan_btn.setEnabled(False)
        self.rescan_btn.clicked.connect(self._on_rescan_input)

        folder_row.addWidget(self.input_dir_edit, stretch=1)
        folder_row.addWidget(self.browse_input_btn)
        folder_row.addWidget(self.rescan_btn)

        self.ffmpeg_status_label = QLabel()
        input_layout.addLayout(folder_row)
        input_layout.addWidget(self.ffmpeg_status_label)
        root.addWidget(input_group)

        # --- 中段: 3系統＋対象外の振り分け確認・修正エリア ---
        streams_group = QGroupBox(
            "2. 3系統の振り分け確認・修正（ファイルを選択してボタンまたは右クリックで移動できます）"
        )
        streams_vbox = QVBoxLayout(streams_group)

        columns_row = QHBoxLayout()
        for kind in ("video", "mic", "line", "excluded"):
            col_box = QGroupBox(STREAM_LABELS[kind])
            col_layout = QVBoxLayout(col_box)
            col_layout.setContentsMargins(8, 8, 8, 8)

            count_lbl = QLabel("0 件 (0 B)")
            count_lbl.setStyleSheet("color: #AAAAAA; font-weight: normal;")

            lw = QListWidget()
            lw.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
            lw.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
            lw.itemSelectionChanged.connect(
                lambda k=kind: self._on_list_selection_changed(k)
            )
            lw.customContextMenuRequested.connect(
                lambda pos, k=kind: self._show_item_context_menu(k, pos)
            )

            col_layout.addWidget(count_lbl)
            col_layout.addWidget(lw, stretch=1)

            self._stream_lists[kind] = lw
            self._stream_count_labels[kind] = count_lbl
            columns_row.addWidget(col_box, stretch=1)

        streams_vbox.addLayout(columns_row, stretch=1)

        # 振り分け操作ボタン行
        move_row = QHBoxLayout()
        move_hint = QLabel("選択項目を移動:")
        move_hint.setStyleSheet("color: #BBBBBB;")

        self.to_video_btn = QPushButton("→ 映像 (MOV)")
        self.to_video_btn.clicked.connect(lambda: self._move_selected_to("video"))

        self.to_mic_btn = QPushButton("→ A1: マイク (WAV)")
        self.to_mic_btn.clicked.connect(lambda: self._move_selected_to("mic"))

        self.to_line_btn = QPushButton("→ A2: ライン (WAV)")
        self.to_line_btn.clicked.connect(lambda: self._move_selected_to("line"))

        self.to_excluded_btn = QPushButton("→ 対象外")
        self.to_excluded_btn.clicked.connect(lambda: self._move_selected_to("excluded"))

        self.swap_audio_btn = QPushButton("⇄ A1(マイク)とA2(ライン)を入れ替え")
        self.swap_audio_btn.clicked.connect(self._swap_mic_and_line)

        self.move_up_btn = QPushButton("▲ 上へ")
        self.move_up_btn.clicked.connect(lambda: self._shift_selected_order(-1))

        self.move_down_btn = QPushButton("▼ 下へ")
        self.move_down_btn.clicked.connect(lambda: self._shift_selected_order(1))

        self.sort_reset_btn = QPushButton("自然順に戻す")
        self.sort_reset_btn.clicked.connect(self._resort_all_lists)

        move_row.addWidget(move_hint)
        move_row.addWidget(self.to_video_btn)
        move_row.addWidget(self.to_mic_btn)
        move_row.addWidget(self.to_line_btn)
        move_row.addWidget(self.to_excluded_btn)
        move_row.addSpacing(12)
        move_row.addWidget(self.swap_audio_btn)
        move_row.addStretch(1)
        move_row.addWidget(self.move_up_btn)
        move_row.addWidget(self.move_down_btn)
        move_row.addWidget(self.sort_reset_btn)

        streams_vbox.addLayout(move_row)
        root.addWidget(streams_group, stretch=3)

        # --- 出力先指定エリア ---
        output_group = QGroupBox("3. 出力設定（無劣化マルチトラック MOV）")
        output_layout = QHBoxLayout(output_group)

        output_layout.addWidget(QLabel("出力先フォルダ:"))
        self.output_dir_edit = QLineEdit()
        self.output_dir_edit.setPlaceholderText("出力先フォルダパス")
        self.browse_output_btn = QPushButton("参照...")
        self.browse_output_btn.clicked.connect(self._on_browse_output_dir)

        output_layout.addWidget(self.output_dir_edit, stretch=2)
        output_layout.addWidget(self.browse_output_btn)
        output_layout.addSpacing(12)

        output_layout.addWidget(QLabel("ファイル名:"))
        self.output_name_edit = QLineEdit()
        self.output_name_edit.setPlaceholderText("output_multitrack.mov")
        output_layout.addWidget(self.output_name_edit, stretch=1)

        root.addWidget(output_group)

        # --- 実行・進捗・ログエリア ---
        exec_row = QHBoxLayout()
        self.run_btn = QPushButton("マルチトラック MOV を作成")
        self.run_btn.setObjectName("primaryButton")
        self.run_btn.clicked.connect(self._on_start_merge)

        self.cancel_btn = QPushButton("キャンセル")
        self.cancel_btn.setObjectName("dangerButton")
        self.cancel_btn.setEnabled(False)
        self.cancel_btn.clicked.connect(self._on_cancel_merge)

        self.open_out_btn = QPushButton("出力先を開く")
        self.open_out_btn.clicked.connect(self._on_open_output_dir)

        self.status_label = QLabel("待機中")
        self.status_label.setStyleSheet("color: #CCCCCC; font-weight: bold;")

        exec_row.addWidget(self.run_btn)
        exec_row.addWidget(self.cancel_btn)
        exec_row.addWidget(self.open_out_btn)
        exec_row.addSpacing(12)
        exec_row.addWidget(self.status_label, stretch=1)
        root.addLayout(exec_row)

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        root.addWidget(self.progress_bar)

        self.log_edit = QPlainTextEdit()
        self.log_edit.setReadOnly(True)
        self.log_edit.setMaximumBlockCount(5000)
        self.log_edit.setPlaceholderText("ここに FFmpeg の出力ログが表示されます...")
        root.addWidget(self.log_edit, stretch=2)

    def _update_ffmpeg_indicator(self) -> Path | None:
        ffmpeg_path = get_ffmpeg_path()
        if ffmpeg_path:
            self.ffmpeg_status_label.setText(f"FFmpeg 検出済み: {ffmpeg_path}")
            self.ffmpeg_status_label.setStyleSheet("color: #6BD47E;")
        else:
            self.ffmpeg_status_label.setText(
                "警告: ffmpeg.exe が見つかりません（実行前に配置または PATH 設定が必要です）"
            )
            self.ffmpeg_status_label.setStyleSheet("color: #FF7A7A; font-weight: bold;")
        return ffmpeg_path

    # --- ドラッグ＆ドロップ ---
    def dragEnterEvent(self, event: QDragEnterEvent) -> None:
        if self._is_running():
            event.ignore()
            return
        mime = event.mimeData()
        if mime.hasUrls():
            for url in mime.urls():
                if url.isLocalFile():
                    event.acceptProposedAction()
                    return
        event.ignore()

    def dropEvent(self, event: QDropEvent) -> None:
        if self._is_running():
            return
        for url in event.mimeData().urls():
            if url.isLocalFile():
                dropped = Path(url.toLocalFile())
                target_dir = dropped if dropped.is_dir() else dropped.parent
                self.load_input_directory(target_dir)
                break

    # --- フォルダ読み込み・分類表示 ---
    def _on_browse_input(self) -> None:
        start_dir = str(self._input_dir) if self._input_dir else ""
        chosen = QFileDialog.getExistingDirectory(
            self, "Zoom Q8 の録画フォルダを選択", start_dir
        )
        if chosen:
            self.load_input_directory(Path(chosen))

    def _on_rescan_input(self) -> None:
        if self._input_dir:
            self.load_input_directory(self._input_dir)

    def load_input_directory(self, folder: Path) -> None:
        try:
            classified = classify_directory(folder)
        except MergeExecutionError as exc:
            QMessageBox.critical(self, "読み込みエラー", str(exc))
            return

        self._input_dir = folder.resolve()
        self.input_dir_edit.setText(str(self._input_dir))
        self.rescan_btn.setEnabled(True)

        # 初期出力先とファイル名を自動セット
        folder_label = self._input_dir.name or "Q8"
        self.output_dir_edit.setText(str(self._input_dir))
        self.output_name_edit.setText(f"{folder_label}_multitrack.mov")

        for lw in self._stream_lists.values():
            lw.clear()

        for cf in classified:
            self._add_classified_item(self._stream_lists[cf.stream], cf)

        self._refresh_stream_summaries()
        self.progress_bar.setValue(0)

        self._append_log(f"フォルダを読み込みました: {self._input_dir}")
        for cf in classified:
            self._append_log(
                f"  [{STREAM_LABELS[cf.stream]}] {cf.rel_path} ({format_size(cf.size_bytes)}) - {cf.reason}"
            )

        unassigned_wavs = [
            cf for cf in classified if cf.stream == "excluded" and cf.path.suffix.lower() == ".wav"
        ]
        if unassigned_wavs:
            self.status_label.setText(
                f"未分類の WAV が {len(unassigned_wavs)} 件あります。一覧から振り分けてください。"
            )
            self.status_label.setStyleSheet("color: #FFB84D; font-weight: bold;")
        elif not classified:
            self.status_label.setText("対象となる .MOV / .WAV ファイルが見つかりませんでした。")
            self.status_label.setStyleSheet("color: #FF7A7A; font-weight: bold;")
        else:
            self.status_label.setText("振り分け結果を確認し、「マルチトラック MOV を作成」を押してください。")
            self.status_label.setStyleSheet("color: #CCCCCC; font-weight: bold;")

    def _add_classified_item(self, lw: QListWidget, cf: ClassifiedFile) -> None:
        item = QListWidgetItem()
        item.setData(Qt.ItemDataRole.UserRole, cf)
        item.setToolTip(
            f"ファイル: {cf.rel_path}\nサイズ: {format_size(cf.size_bytes)}\n推定理由: {cf.reason}"
        )
        lw.addItem(item)

    def _refresh_stream_summaries(self) -> None:
        for kind, lw in self._stream_lists.items():
            total_bytes = 0
            count = lw.count()
            for i in range(count):
                item = lw.item(i)
                cf: ClassifiedFile = item.data(Qt.ItemDataRole.UserRole)
                total_bytes += cf.size_bytes
                prefix = f"{i + 1:02d}. " if kind != "excluded" else ""
                item.setText(f"{prefix}{cf.rel_path}  ({format_size(cf.size_bytes)})")
            self._stream_count_labels[kind].setText(
                f"{count} 件 ({format_size(total_bytes)})"
            )

    def _on_list_selection_changed(self, active_kind: StreamKind) -> None:
        if self._selection_sync_guard:
            return
        active_lw = self._stream_lists[active_kind]
        if not active_lw.selectedItems():
            return
        self._selection_sync_guard = True
        try:
            for kind, lw in self._stream_lists.items():
                if kind != active_kind:
                    lw.clearSelection()
        finally:
            self._selection_sync_guard = False

    def _show_item_context_menu(self, source_kind: StreamKind, pos: QPoint) -> None:
        if self._is_running():
            return
        lw = self._stream_lists[source_kind]
        if not lw.selectedItems():
            return
        menu = QMenu(self)
        for target_kind, label in STREAM_LABELS.items():
            if target_kind == source_kind:
                continue
            action = menu.addAction(f"「{label}」へ移動")
            action.triggered.connect(
                lambda checked=False, tk=target_kind: self._move_selected_to(tk)
            )
        menu.exec(lw.mapToGlobal(pos))

    def _move_selected_to(self, target_kind: StreamKind) -> None:
        if self._is_running():
            return
        moved: list[ClassifiedFile] = []
        for kind, lw in self._stream_lists.items():
            if kind == target_kind:
                continue
            selected_rows = sorted((lw.row(it) for it in lw.selectedItems()), reverse=True)
            for row in selected_rows:
                item = lw.takeItem(row)
                cf: ClassifiedFile = item.data(Qt.ItemDataRole.UserRole)
                cf.stream = target_kind
                moved.append(cf)

        if not moved:
            return

        target_lw = self._stream_lists[target_kind]
        for cf in moved:
            self._add_classified_item(target_lw, cf)

        self._sort_single_list(target_kind)
        self._refresh_stream_summaries()

    def _swap_mic_and_line(self) -> None:
        if self._is_running():
            return
        mic_lw = self._stream_lists["mic"]
        line_lw = self._stream_lists["line"]

        mic_items = [
            mic_lw.item(i).data(Qt.ItemDataRole.UserRole) for i in range(mic_lw.count())
        ]
        line_items = [
            line_lw.item(i).data(Qt.ItemDataRole.UserRole) for i in range(line_lw.count())
        ]

        mic_lw.clear()
        line_lw.clear()

        for cf in line_items:
            cf.stream = "mic"
            self._add_classified_item(mic_lw, cf)
        for cf in mic_items:
            cf.stream = "line"
            self._add_classified_item(line_lw, cf)

        self._refresh_stream_summaries()

    def _shift_selected_order(self, delta: int) -> None:
        if self._is_running():
            return
        for lw in self._stream_lists.values():
            selected = lw.selectedItems()
            if not selected:
                continue
            rows = sorted(lw.row(it) for it in selected)
            if delta < 0:
                if rows[0] == 0:
                    return
                for r in rows:
                    item = lw.takeItem(r)
                    lw.insertItem(r - 1, item)
                    item.setSelected(True)
            else:
                if rows[-1] >= lw.count() - 1:
                    return
                for r in reversed(rows):
                    item = lw.takeItem(r)
                    lw.insertItem(r + 1, item)
                    item.setSelected(True)
            self._refresh_stream_summaries()
            return

    def _sort_single_list(self, kind: StreamKind) -> None:
        lw = self._stream_lists[kind]
        items_cf: list[ClassifiedFile] = [
            lw.item(i).data(Qt.ItemDataRole.UserRole) for i in range(lw.count())
        ]
        items_cf.sort(key=lambda cf: natural_sort_key(cf.path, self._input_dir))
        lw.clear()
        for cf in items_cf:
            self._add_classified_item(lw, cf)

    def _resort_all_lists(self) -> None:
        if self._is_running():
            return
        for kind in self._stream_lists:
            self._sort_single_list(kind)
        self._refresh_stream_summaries()

    def _get_stream_paths(self, kind: StreamKind) -> list[Path]:
        lw = self._stream_lists[kind]
        return [
            lw.item(i).data(Qt.ItemDataRole.UserRole).path for i in range(lw.count())
        ]

    # --- 出力設定 ---
    def _on_browse_output_dir(self) -> None:
        current = self.output_dir_edit.text().strip()
        chosen = QFileDialog.getExistingDirectory(self, "出力先フォルダを選択", current)
        if chosen:
            self.output_dir_edit.setText(chosen)

    def _on_open_output_dir(self) -> None:
        out_dir_text = self.output_dir_edit.text().strip()
        if not out_dir_text:
            return
        out_dir = Path(out_dir_text)
        if not out_dir.exists():
            QMessageBox.information(self, "情報", f"フォルダがまだ存在しません:\n{out_dir}")
            return
        if sys.platform == "win32":
            os.startfile(str(out_dir))  # type: ignore[attr-defined]


    # --- 実行・キャンセル制御 ---
    def _is_running(self) -> bool:
        return self._worker is not None and self._worker.isRunning()

    def _set_running_ui_state(self, running: bool) -> None:
        self.browse_input_btn.setEnabled(not running)
        self.rescan_btn.setEnabled(not running and self._input_dir is not None)
        self.browse_output_btn.setEnabled(not running)
        self.output_dir_edit.setEnabled(not running)
        self.output_name_edit.setEnabled(not running)

        self.to_video_btn.setEnabled(not running)
        self.to_mic_btn.setEnabled(not running)
        self.to_line_btn.setEnabled(not running)
        self.to_excluded_btn.setEnabled(not running)
        self.swap_audio_btn.setEnabled(not running)
        self.move_up_btn.setEnabled(not running)
        self.move_down_btn.setEnabled(not running)
        self.sort_reset_btn.setEnabled(not running)

        self.run_btn.setEnabled(not running)
        self.cancel_btn.setEnabled(running)
        self.cancel_btn.setText("キャンセル")

    def _on_start_merge(self) -> None:
        if self._is_running():
            return

        # 1. FFmpeg 存在チェック
        ffmpeg_path = self._update_ffmpeg_indicator()
        if ffmpeg_path is None:
            QMessageBox.critical(
                self,
                "FFmpeg エラー",
                "ffmpeg.exe が見つかりません。\n\n"
                "次のいずれかに ffmpeg.exe を配置してから再実行してください:\n"
                "・このアプリ（.exe）と同じフォルダ、または _internal フォルダ\n"
                "・プロジェクトフォルダ（開発時）\n"
                "・Windows の環境変数 PATH が通ったフォルダ",
            )
            return

        # 2. 出力先パスの検証
        out_dir_str = self.output_dir_edit.text().strip()
        out_name_str = self.output_name_edit.text().strip()
        if not out_dir_str or not out_name_str:
            QMessageBox.warning(self, "入力不足", "出力先フォルダと出力ファイル名を指定してください。")
            return

        invalid_chars = set('<>:"/\\|?*')
        if any(ch in invalid_chars for ch in out_name_str):
            QMessageBox.warning(
                self,
                "ファイル名エラー",
                '出力ファイル名に使用できない文字（<>:"/\\|?*）が含まれています。',
            )
            return

        if not out_name_str.lower().endswith(".mov"):
            out_name_str += ".mov"
            self.output_name_edit.setText(out_name_str)

        output_path = Path(out_dir_str) / out_name_str

        # 3. 3系統の事前検証（エラーと警告）
        video_files = self._get_stream_paths("video")
        mic_files = self._get_stream_paths("mic")
        line_files = self._get_stream_paths("line")
        excluded_files = self._get_stream_paths("excluded")

        report = validate_streams(
            video_files=video_files,
            mic_files=mic_files,
            line_files=line_files,
            excluded_files=excluded_files,
            output_path=output_path,
            input_dir=self._input_dir,
        )

        if report.errors:
            QMessageBox.critical(
                self, "実行前エラー", "\n\n".join(f"・{e}" for e in report.errors)
            )
            return

        if report.warnings:
            warn_text = (
                "以下の点を確認してください:\n\n"
                + "\n".join(f"・{w}" for w in report.warnings)
                + "\n\nこのまま結合処理を続行しますか？"
            )
            ans = QMessageBox.warning(
                self,
                "振り分け内容の確認",
                warn_text,
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No,
            )
            if ans != QMessageBox.StandardButton.Yes:
                return

        # 4. 同名ファイルが存在する場合の確認（無言で上書きしない）
        if output_path.exists():
            alt_path = suggest_unique_output_path(output_path)
            msg = QMessageBox(self)
            msg.setIcon(QMessageBox.Icon.Question)
            msg.setWindowTitle("同名ファイルの確認")
            msg.setText(f"出力先に同名のファイルが既に存在します:\n{output_path}")
            msg.setInformativeText("上書きするか、自動で別名にして保存するか選んでください。")
            overwrite_btn = msg.addButton("上書きする", QMessageBox.ButtonRole.AcceptRole)
            rename_btn = msg.addButton(
                f"別名で保存 ({alt_path.name})", QMessageBox.ButtonRole.ActionRole
            )
            cancel_btn = msg.addButton("キャンセル", QMessageBox.ButtonRole.RejectRole)
            msg.setDefaultButton(rename_btn)
            msg.exec()

            clicked = msg.clickedButton()
            if clicked == cancel_btn or clicked is None:
                return
            if clicked == rename_btn:
                output_path = alt_path
                self.output_name_edit.setText(output_path.name)
            elif clicked == overwrite_btn:
                lock_err = check_file_writable(output_path)
                if lock_err:
                    QMessageBox.critical(self, "ファイル使用中", lock_err)
                    return

        # 5. バックグラウンドスレッド開始
        self.progress_bar.setValue(0)
        self._append_log("\n----------------------------------------")
        self._append_log("マルチトラック MOV の作成を開始します...")
        self.status_label.setText("開始中...")
        self.status_label.setStyleSheet("color: #6BB5FF; font-weight: bold;")
        self._set_running_ui_state(True)

        self._worker = MergeWorker(
            ffmpeg_path=ffmpeg_path,
            video_files=video_files,
            mic_files=mic_files,
            line_files=line_files,
            output_path=output_path,
            parent=self,
        )
        self._worker.progress_changed.connect(self._on_worker_progress)
        self._worker.log_received.connect(self._append_log)
        self._worker.finished_ok.connect(self._on_worker_finished)
        self._worker.failed.connect(self._on_worker_failed)
        self._worker.cancelled.connect(self._on_worker_cancelled)
        self._worker.start()

    def _on_cancel_merge(self) -> None:
        if self._worker and self._worker.isRunning():
            self.cancel_btn.setEnabled(False)
            self.cancel_btn.setText("キャンセル中...")
            self.status_label.setText("FFmpeg を安全に停止しています...")
            self._worker.request_cancel()

    def _on_worker_progress(self, percent: int, message: str) -> None:
        self.progress_bar.setValue(percent)
        self.status_label.setText(message)

    def _on_worker_finished(self, output_path_str: str) -> None:
        self._set_running_ui_state(False)
        self.progress_bar.setValue(100)
        self.status_label.setText(f"完了: {output_path_str}")
        self.status_label.setStyleSheet("color: #6BD47E; font-weight: bold;")
        QMessageBox.information(
            self,
            "完了",
            f"マルチトラック MOV の作成が完了しました。\n\n出力先:\n{output_path_str}",
        )

    def _on_worker_failed(self, error_message: str) -> None:
        self._set_running_ui_state(False)
        self.status_label.setText("エラーが発生しました（ログ欄を確認してください）")
        self.status_label.setStyleSheet("color: #FF7A7A; font-weight: bold;")
        self._append_log(f"\n[エラー] {error_message}")
        QMessageBox.critical(
            self,
            "結合エラー",
            f"処理中にエラーが発生しました。\n一時ファイルは削除済みです。\n\n{error_message}",
        )

    def _on_worker_cancelled(self) -> None:
        self._set_running_ui_state(False)
        self.progress_bar.setValue(0)
        self.status_label.setText("キャンセルされました（一時ファイル削除済み）")
        self.status_label.setStyleSheet("color: #FFB84D; font-weight: bold;")
        self._append_log("\n[中断] 処理をキャンセルし、一時ファイルを削除しました。")

    def _append_log(self, text: str) -> None:
        self.log_edit.appendPlainText(text)
        scrollbar = self.log_edit.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())

    def closeEvent(self, event: QCloseEvent) -> None:
        if self._is_running() and self._worker is not None:
            ans = QMessageBox.question(
                self,
                "終了確認",
                "現在 FFmpeg の処理を実行中です。中断してアプリを終了しますか？",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No,
            )
            if ans != QMessageBox.StandardButton.Yes:
                event.ignore()
                return
            self._worker.request_cancel()
            self._worker.wait(5000)
        event.accept()


def main() -> None:
    app = QApplication(sys.argv)
    app.setStyleSheet(APP_STYLESHEET)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
