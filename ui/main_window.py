from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

from PyQt5.QtCore import QThread, QUrl, pyqtSignal
from PyQt5.QtGui import QDesktopServices
from PyQt5.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QPlainTextEdit,
    QProgressBar,
    QSpinBox,
    QTabWidget,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

import logging

from spvideo.gemini_analyzer import GENERATION_ROUTE_LABELS, SCENE_TYPE_LABELS
from spvideo.pipeline import run_segmentation


class _GuiLogHandler(logging.Handler):
    """将 Python logging 输出转发到 Qt 信号。"""
    def __init__(self, signal):
        super().__init__()
        self.signal = signal
        self.setFormatter(logging.Formatter("%(levelname).1s %(message)s"))

    def emit(self, record):
        try:
            self.signal.emit(self.format(record))
        except Exception:
            pass

TYPE_LABELS = {
    "with_human": "有人",
    "without_human": "无人",
}

SOURCE_LABELS = {
    "omnishotcut": "OmniShotCut",
    "pyscene": "PySceneDetect",
    "yolo": "YOLO",
    "yolo_transient_multi": "YOLO短暂多人",
    "face_id": "人脸身份",
    "gemini": "Gemini",
    "sam3": "SAM3",
}


class SplitWorker(QThread):
    finished_ok = pyqtSignal(dict)
    failed = pyqtSignal(str)
    log_msg = pyqtSignal(str)

    def __init__(
        self,
        video_path: str,
        project_dir: str,
        max_segment: int,
        gemini_api_key: str | None = None,
        gemini_model: str | None = None,
        export_video: bool = True,
        use_two_pass: bool = False,
        use_omnishotcut: bool = False,
        yolo_conf: float = 0.35,
        device: str | None = None,
        sample_interval: float = 0.5,
        gemini_concurrency: int = 10,
        use_scene_detect: bool = True,
        use_face_id: bool = True,
        use_visual_model: bool = True,
        extract_backgrounds: bool = False,
        use_sam3_finalize: bool = False,
        use_visual_merge: bool | None = None,
    ) -> None:
        super().__init__()
        self.video_path = video_path
        self.project_dir = project_dir
        self.max_segment = max_segment
        self.gemini_api_key = gemini_api_key
        self.gemini_model = gemini_model
        self.export_video = export_video
        self.use_two_pass = use_two_pass
        self.use_omnishotcut = use_omnishotcut
        self.yolo_conf = yolo_conf
        self.device = device
        self.sample_interval = sample_interval
        self.gemini_concurrency = gemini_concurrency
        self.use_scene_detect = use_scene_detect
        self.use_face_id = use_face_id
        self.use_visual_model = use_visual_model
        self.extract_backgrounds = extract_backgrounds
        self.use_sam3_finalize = use_sam3_finalize
        self.use_visual_merge = use_visual_merge

    def run(self) -> None:
        # 把 pipeline 内部的 logger 输出转发到 GUI + 终端
        import sys
        root_logger = logging.getLogger("spvideo")
        root_logger.setLevel(logging.INFO)

        gui_handler = _GuiLogHandler(self.log_msg)
        gui_handler.setLevel(logging.DEBUG)
        root_logger.addHandler(gui_handler)

        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(logging.Formatter("%(levelname)s %(message)s"))
        root_logger.addHandler(console_handler)

        try:
            result = run_segmentation(
                self.video_path,
                self.project_dir,
                sample_interval=self.sample_interval,
                max_segment_duration=float(self.max_segment),
                export_video=self.export_video,
                gemini_api_key=self.gemini_api_key or None,
                gemini_model=self.gemini_model,
                gemini_identity_concurrency=self.gemini_concurrency,
                use_two_pass=self.use_two_pass,
                use_omnishotcut=self.use_omnishotcut,
                yolo_conf_threshold=self.yolo_conf,
                device=self.device,
                use_scene_detect=self.use_scene_detect,
                use_face_id=self.use_face_id,
                use_visual_model=self.use_visual_model,
                extract_backgrounds=self.extract_backgrounds,
                use_sam3_finalize=self.use_sam3_finalize,
                use_visual_merge=self.use_visual_merge,
            )
            self.finished_ok.emit(result)
        except Exception as exc:  # noqa: BLE001
            self.failed.emit(str(exc))
        finally:
            root_logger.removeHandler(gui_handler)
            root_logger.removeHandler(console_handler)


class MainWindow(QMainWindow):
    def __init__(self, preload_project_dir: Path | None = None) -> None:
        super().__init__()
        self.setWindowTitle("SP AI Video Planner")
        self.resize(1220, 800)
        self.worker: SplitWorker | None = None
        self.last_clips_dir: Path | None = None
        self.last_report_path: Path | None = None
        self.last_raw_segments: list[dict] = []
        self.last_gemini_analysis: list[dict] | None = None

        # ── 输入控件 ──────────────────────────────────────────────
        self.video_input = QLineEdit()
        self.project_input = QLineEdit(str(preload_project_dir) if preload_project_dir else "")
        self.max_segment_input = QSpinBox()
        self.max_segment_input.setRange(3, 12)
        self.max_segment_input.setValue(6)
        self.gemini_key_input = QLineEdit()
        self.gemini_key_input.setPlaceholderText("填写则启用千问 Omni 视觉理解")
        self.gemini_key_input.setText("")
        self.export_video_cb = QCheckBox("裁切 MP4 片段")
        self.export_video_cb.setChecked(True)
        self.extract_backgrounds_cb = QCheckBox("提取背景图")
        self.extract_backgrounds_cb.setChecked(False)
        self.extract_backgrounds_cb.setToolTip("为后续换场景/转绘生成背景参考图；只看切分效果时建议关闭")
        self.cb_scene = QCheckBox("1. 硬切与人物状态 (PySceneDetect + YOLO)")
        self.cb_scene.setChecked(True)
        self.cb_scene.setToolTip("硬切候选 + 人物数量变化；为后续身份与轨迹层提供主体框")
        self.cb_visual_merge = QCheckBox("2. 弱边界清理 (画面相似合并)")
        self.cb_visual_merge.setChecked(True)
        self.cb_visual_merge.setToolTip("仅清理 OmniShotCut 产生的弱边界与转场碎片")
        self.cb_face = QCheckBox("3. 人脸身份复核 (InsightFace)")
        self.cb_face.setChecked(True)
        self.cb_face.setToolTip("先用本地人脸特征识别同镜头换人，稳定结果不再请求视觉模型")
        self.cb_omni = QCheckBox("1. 镜头候选 (OmniShotCut)")
        self.cb_omni.setChecked(True)
        self.cb_omni.setToolTip("语义镜头候选，与 PySceneDetect 融合，不是额外强制切点")
        self.cb_visual = QCheckBox("5. 视觉模型仲裁 (千问 Omni)")
        self.cb_visual.setChecked(True)
        self.cb_visual.setToolTip("只处理本地人脸层、YOLO 与 SAM3 仍无法确认的片段（需 API Key）")
        self.cb_sam3 = QCheckBox("4. 主体轨迹复核 (SAM3)")
        self.cb_sam3.setChecked(False)
        self.cb_sam3.setToolTip("仅对人物状态或身份存在风险的单人片段跟踪；主体持续丢失才补切")
        self.model_combo = QComboBox()
        self.model_combo.addItems([
            "qwen3.5-omni-plus（质量优先）",
            "qwen3.5-omni-flash（速度优先）",
        ])
        self.model_combo.setToolTip("只保留千问 Omni；Plus 质量优先，Flash 速度优先")
        self.gemini_concurrency_input = QSpinBox()
        self.gemini_concurrency_input.setRange(1, 32)
        self.gemini_concurrency_input.setValue(10)
        self.gemini_concurrency_input.setToolTip("同时请求视觉模型的片段数；中转站稳定时可调高")
        self.sample_interval_combo = QComboBox()
        self.sample_interval_combo.addItems(["0.1s（极致精度）", "0.5s（高精度）", "1.0s（推荐）"])
        self.sample_interval_combo.setCurrentIndex(0)
        self.sample_interval_combo.setToolTip("抽帧间隔：越小切点越精确，但计算量越大")
        self.yolo_conf_input = QSpinBox()
        self.yolo_conf_input.setRange(20, 90)
        self.yolo_conf_input.setValue(35)
        self.yolo_conf_input.setSuffix("%")
        self.yolo_conf_input.setToolTip("YOLO 人物检测置信度阈值（越低越敏感）")
        self.device_combo = QComboBox()
        self.device_combo.addItems(["auto (自动检测)", "cuda (GPU)", "cpu (CPU)"])
        self.device_combo.setToolTip("YOLO 推理设备：GPU 加速 vs CPU 兼容")

        # ── 进度 / 日志 / 按钮 ────────────────────────────────────
        self.progress = QProgressBar()
        self.progress.setRange(0, 0)
        self.progress.hide()
        self.log = QPlainTextEdit()
        self.log.setReadOnly(True)
        self.open_clips_button = QPushButton("打开 MP4 片段目录")
        self.open_clips_button.setEnabled(False)
        self.open_clips_button.clicked.connect(lambda: self._open_path(self.last_clips_dir))
        self.open_report_button = QPushButton("打开审片报告")
        self.open_report_button.setEnabled(False)
        self.open_report_button.clicked.connect(lambda: self._open_path(self.last_report_path))

        # ── 审片表格 ──────────────────────────────────────────────
        self.segment_table = QTableWidget(0, 12)
        self.segment_table.setHorizontalHeaderLabels([
            "ID", "开始", "结束", "时长", "人数",
            "场景类型(Gemini)", "生成路线(Gemini)", "描述", "需复审", "建议",
            "算法类别", "切割来源",
        ])

        tabs = QTabWidget()
        tabs.addTab(self._build_import_tab(), "1 导入与切分")
        tabs.addTab(self._build_review_tab(), "2 分镜审片")
        tabs.addTab(self._build_ai_tab(), "3 AI 输入包")
        tabs.addTab(self._build_export_tab(), "4 导出")
        self.setCentralWidget(tabs)

    def _build_import_tab(self) -> QWidget:
        widget = QWidget()
        layout = QVBoxLayout(widget)

        form = QFormLayout()
        form.addRow("原视频", self._with_button(self.video_input, "选择", self._choose_video))
        form.addRow("项目目录", self._with_button(self.project_input, "选择", self._choose_project))
        form.addRow("AI 友好最大分段秒数", self.max_segment_input)
        form.addRow("Gemini API Key", self.gemini_key_input)
        form.addRow("视觉模型", self.model_combo)
        form.addRow("视觉模型并发", self.gemini_concurrency_input)
        form.addRow("", self.cb_omni)
        form.addRow("", self.cb_scene)
        form.addRow("", self.cb_visual_merge)
        form.addRow("", self.cb_face)
        form.addRow("", self.cb_sam3)
        form.addRow("", self.cb_visual)
        form.addRow("YOLO 设备", self.device_combo)
        form.addRow("YOLO 置信度", self.yolo_conf_input)
        form.addRow("抽帧间隔", self.sample_interval_combo)
        form.addRow("", self.export_video_cb)
        form.addRow("", self.extract_backgrounds_cb)
        layout.addLayout(form)

        start_button = QPushButton("▶ 开始自动切分")
        start_button.setStyleSheet("font-size: 14px; font-weight: 700; padding: 8px;")
        start_button.clicked.connect(self._start_split)
        layout.addWidget(start_button)
        layout.addWidget(self.progress)

        output_buttons = QHBoxLayout()
        output_buttons.addWidget(self.open_clips_button)
        output_buttons.addWidget(self.open_report_button)
        output_buttons.addStretch(1)
        layout.addLayout(output_buttons)
        layout.addWidget(QLabel("运行日志"))
        layout.addWidget(self.log, 1)
        return widget

    def _build_review_tab(self) -> QWidget:
        widget = QWidget()
        layout = QVBoxLayout(widget)
        info = QLabel("自动分桶结果（算法基础分类 + Gemini 视觉理解）：")
        layout.addWidget(info)
        layout.addWidget(self.segment_table, 1)
        return widget

    def _build_ai_tab(self) -> QWidget:
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.addWidget(QLabel(
            "后续阶段：按 Gemini 生成路线分别处理<br><br>"
            "• human_driver → 导出 driving_video、pose、face、hand<br>"
            "• image_to_video → 准备产品图、prompt<br>"
            "• product_replication → 多角度素材、3D 参考<br>"
            "• graphic_animation → 图文素材、字幕文件"
        ))
        return widget

    def _build_export_tab(self) -> QWidget:
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.addWidget(QLabel("后续阶段：导出剪辑时间线、字幕、AI 生成任务清单和最终合成草稿。"))
        return widget

    # ── Helper ────────────────────────────────────────────────────

    def _with_button(self, line_edit: QLineEdit, text: str, callback) -> QWidget:
        wrapper = QWidget()
        layout = QHBoxLayout(wrapper)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(line_edit, 1)
        button = QPushButton(text)
        button.clicked.connect(callback)
        layout.addWidget(button)
        return wrapper

    def _choose_video(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "选择原视频", "", "Video Files (*.mp4 *.mov *.mkv);;All Files (*)")
        if path:
            self.video_input.setText(path)

    def _choose_project(self) -> None:
        path = QFileDialog.getExistingDirectory(self, "选择项目目录")
        if path:
            self.project_input.setText(path)

    # ── 核心动作 ──────────────────────────────────────────────────

    def _start_split(self) -> None:
        video_path = self.video_input.text().strip()
        project_dir = self.project_input.text().strip()
        if not video_path or not project_dir:
            QMessageBox.warning(self, "缺少信息", "请先选择原视频和项目目录。")
            return

        self.last_clips_dir = None
        self.last_report_path = None
        self.last_raw_segments = []
        self.last_gemini_analysis = None
        self.open_clips_button.setEnabled(False)
        self.open_report_button.setEnabled(False)

        gemini_key = self.gemini_key_input.text().strip()
        model_map = {0: "qwen3.5-omni-plus", 1: "qwen3.5-omni-flash"}
        gemini_model = model_map.get(self.model_combo.currentIndex(), "qwen3.5-omni-plus")
        gemini_concurrency = self.gemini_concurrency_input.value()
        use_omnishotcut = self.cb_omni.isChecked()
        use_scene_detect = self.cb_scene.isChecked()
        use_face_id = self.cb_face.isChecked()
        use_visual_model = self.cb_visual.isChecked()
        use_sam3_finalize = self.cb_sam3.isChecked()
        use_visual_merge = self.cb_visual_merge.isChecked()
        extract_backgrounds = self.extract_backgrounds_cb.isChecked()
        use_two_pass = use_scene_detect or use_omnishotcut or use_sam3_finalize
        yolo_conf = self.yolo_conf_input.value() / 100.0

        device_map = {0: None, 1: "cuda", 2: "cpu"}
        device_idx = self.device_combo.currentIndex()
        device = device_map.get(device_idx, None)
        sample_map = {0: 0.1, 1: 0.5, 2: 1.0}
        sample_interval = sample_map.get(self.sample_interval_combo.currentIndex(), 0.5)

        self.log.appendPlainText(f"> 开始切分: {video_path}  (抽帧={sample_interval}s)")
        # 显示启用的层
        layers = []
        if use_omnishotcut: layers.append("1. OmniShotCut")
        if use_scene_detect: layers.append("1. PySceneDetect+YOLO")
        if use_visual_merge: layers.append("2. 弱边界清理")
        if use_face_id: layers.append("3. 人脸身份")
        if use_sam3_finalize: layers.append("4. SAM3风险复核")
        if use_visual_model and gemini_key: layers.append("5. 视觉模型仲裁")
        if extract_backgrounds: layers.append("背景提取")
        active = " + ".join(layers) if layers else "无（仅按固定时长切分）"
        self.log.appendPlainText(f"   启用层: {active}")
        self.progress.show()

        self.worker = SplitWorker(
            video_path=video_path,
            project_dir=project_dir,
            max_segment=self.max_segment_input.value(),
            gemini_api_key=gemini_key or None,
            gemini_model=gemini_model,
            gemini_concurrency=gemini_concurrency,
            export_video=self.export_video_cb.isChecked(),
            use_two_pass=use_two_pass,
            use_omnishotcut=use_omnishotcut,
            yolo_conf=yolo_conf,
            device=device,
            sample_interval=sample_interval,
            use_scene_detect=use_scene_detect,
            use_face_id=use_face_id,
            use_visual_model=use_visual_model,
            extract_backgrounds=extract_backgrounds,
            use_sam3_finalize=use_sam3_finalize,
            use_visual_merge=use_visual_merge,
        )
        self.worker.finished_ok.connect(self._on_split_finished)
        self.worker.failed.connect(self._on_split_failed)
        self.worker.log_msg.connect(self.log.appendPlainText)
        self.worker.start()

    def _on_split_finished(self, result: dict) -> None:
        self.progress.hide()
        self.log.appendPlainText(f"✔ 完成。项目目录: {result.get('project_dir')}")

        clips_dir = result.get("clips_dir")
        report_path = result.get("report_path")
        self.last_clips_dir = Path(str(clips_dir)) if clips_dir else None
        self.last_report_path = Path(str(report_path)) if report_path else None
        self.last_raw_segments = result.get("segments", [])
        self.last_gemini_analysis = result.get("gemini_analysis")

        if self.last_clips_dir:
            self.log.appendPlainText(f"   MP4 片段: {self.last_clips_dir}")
            self.open_clips_button.setEnabled(True)
        if self.last_report_path:
            self.log.appendPlainText(f"   审片报告: {self.last_report_path}")
            self.open_report_button.setEnabled(True)
        if self.last_gemini_analysis:
            self.log.appendPlainText(f"   Gemini 分析了 {len(self.last_gemini_analysis)} 个片段")

        self._load_segments(self.last_raw_segments, self.last_gemini_analysis)

    def _on_split_failed(self, message: str) -> None:
        self.progress.hide()
        self.log.appendPlainText(f"✖ 失败: {message}")
        QMessageBox.critical(self, "切分失败", message)

    # ── 表格 ──────────────────────────────────────────────────────

    def _load_segments(self, segments: list[dict], gemini_analysis: list[dict] | None = None) -> None:
        gemini_map: dict[str, dict[str, Any]] = {}
        if gemini_analysis:
            for item in gemini_analysis:
                sid = item.get("segment_id", "")
                if sid:
                    gemini_map[sid] = item

        self.segment_table.setRowCount(len(segments))
        for row, seg in enumerate(segments):
            sid = seg.get("segment_id", "")
            g = gemini_map.get(sid)

            scene_type = ""
            route = ""
            desc = ""
            needs_review = "⚠ 是" if seg.get("needs_manual_check") else "✓ 否"
            if g:
                st = g.get("scene_type", "")
                scene_type = SCENE_TYPE_LABELS.get(st, st)
                rt = g.get("generation_route", "")
                route = GENERATION_ROUTE_LABELS.get(rt, rt)
                desc = g.get("description", "")[:60]
                needs_review = (
                    "⚠ 是"
                    if seg.get("needs_manual_check") or g.get("needs_manual_review", True)
                    else "✓ 否"
                )

            person_count = seg.get("person_count", -1)
            if person_count >= 0:
                if person_count == 0:
                    person_label = "BG(无人)"
                elif person_count == 1:
                    person_label = "1人"
                else:
                    person_label = f"{person_count}人"
                if seg.get("transient_multi_person"):
                    person_label += "（短暂入镜）"
            else:
                person_label = TYPE_LABELS.get(seg.get("segment_type", ""), seg.get("segment_type", ""))

            # 合并起止边界的所有来源，去重后显示
            all_sources: list[str] = []
            for src in seg.get("start_sources", []) + seg.get("end_sources", []):
                if src not in all_sources:
                    all_sources.append(src)
            if all_sources:
                source_label = " + ".join(SOURCE_LABELS.get(s, s) for s in all_sources)
            else:
                source_label = "-"

            values = [
                sid,
                f"{seg.get('start', 0):.2f}",
                f"{seg.get('end', 0):.2f}",
                f"{seg.get('duration', 0):.2f}",
                person_label,
                scene_type,
                route,
                desc,
                needs_review,
                seg.get("recommended_tech", ""),
                TYPE_LABELS.get(seg.get("segment_type", ""), seg.get("segment_type", "")),
                source_label,
            ]
            for col, value in enumerate(values):
                self.segment_table.setItem(row, col, QTableWidgetItem(str(value)))

        self.segment_table.resizeColumnsToContents()

    # ── 打开外部路径 ──────────────────────────────────────────────

    def _open_path(self, path: Path | None) -> None:
        if path is None:
            QMessageBox.information(self, "没有可打开的结果", "请先完成一次自动切分。")
            return
        if not path.exists():
            QMessageBox.warning(self, "路径不存在", str(path))
            return
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(path)))


def run_app(preload_project_dir: Path | None = None) -> int:
    app = QApplication(sys.argv)
    window = MainWindow(preload_project_dir)
    window.show()
    return app.exec_()
