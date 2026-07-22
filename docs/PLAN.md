# SP AI Video Planner - 总计划

## 目标

把一条参考短视频自动拆成可执行的 AI 复刻工程：

1. 自动读取视频参数、抽帧、切分片段。
2. 按技术路线分桶：人物动作驱动、人物合成、产品 B-roll、图文版式。
3. 为后续模型生成准备 driving video、pose、face、hand、mask、产品素材文件夹。
4. 在 PyQt5 桌面界面里完成导入、切分、审片、修正、导出任务清单。

## 阶段规划

### Phase 1 - 视频切分中台

不依赖重型 AI 模型，优先稳定：

- ffprobe 读取 meta。
- ffmpeg 抽 1fps 帧。
- PIL/numpy 计算画面差异、蓝色占比、肤色候选、白底/文字密度。
- 自动找切点。
- 超过 6 秒的片段继续拆成 AI 友好的短片段。
- 生成 `manifest.json`、分类文件夹、contact sheet、HTML 报告。

### Phase 2 - 轻模型增强

- OCR：PaddleOCR/EasyOCR，识别字幕、品牌、证书文字。
- ASR：Whisper/faster-whisper，转写旁白。
- Pose：DWPose/OpenPose/MediaPipe，提取 body/hand/face maps。

### Phase 3 - 视觉大模型理解

- 让 VLM 判断片段语义：口播、产品证明、证书背书、工艺演示、CTA。
- 自动写分镜说明和复刻建议。

### Phase 4 - 生成端对接

- 人物片段：Runway/Kling/ComfyUI driving video。
- 产品片段：实拍、图生视频、3D、静图推拉。
- 图文片段：AE/剪映/PR/Python 动效模板。

## PyQt5 界面规划

### Tab 1 导入与切分

- 选择原视频。
- 选择项目目录。
- 设置最大 AI 友好分段秒数。
- 启动自动切分。
- 显示日志和进度。

### Tab 2 分镜审片

- 表格显示每段：ID、起止时间、类型、置信度、技术路线。
- 后续加入缩略图预览、视频片段预览。
- 支持手动改分类。

### Tab 3 AI 输入包

- 展示 driving_video、pose_maps、face_maps、hand_maps、masks 等文件夹状态。
- 后续一键提取姿态/表情/手势。

### Tab 4 导出

- 导出 manifest。
- 导出给剪辑软件的时间线。
- 导出给 AI 生成工具的任务清单。

## 输出目录规范

```text
project/
  00_source/
  01_probe/
    frames_1fps/
    scene_scores.json
    contact_sheet_segments.jpg
    segmentation_report.html
  02_segments/
    01_human_driver/
    02_human_composite/
    03_product_broll/
    04_graphic_layout/
    05_reference_only/
  03_ai_inputs/
    driving_video/
    pose_maps/
    face_maps/
    hand_maps/
    masks/
    background_refs/
    character_refs/
    product_refs/
  04_ai_outputs/
  05_edit/
  06_export/
  manifest.json
```

