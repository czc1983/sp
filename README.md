# SP AI Video Planner

用于把参考视频拆成 AI 复刻工程的 Python/PyQt5 工具。

当前第一版完成：

- 读取视频元数据。
- 抽 1fps 样本帧。
- 根据画面变化自动粗切。
- 将长片段拆成 6 秒左右的 AI 友好片段。
- 规则分类：`human_driver`、`human_composite`、`product_broll`、`graphic_layout`、`unknown_need_check`。
- 输出 mp4 片段、manifest、contact sheet、HTML 审片报告。
- 提供 PyQt5 界面骨架。

## 命令行使用

```powershell
python main.py split --input "C:\path\input.mp4" --project-dir "E:\sp\project_001"
```

如果只想分析不切 mp4：

```powershell
python main.py split --input "C:\path\input.mp4" --project-dir "E:\sp\project_001" --no-export-video
```

## GUI

```powershell
python main.py gui
```

当前环境如果缺少 PyQt5，需要先安装：

```powershell
pip install PyQt5
```

## 说明

第一版分类是规则算法，不接大模型。它的目标不是一次性完全判断正确，而是先把项目文件夹、片段、manifest 和人工审片入口建起来。后续再接 OCR、ASR、Pose、视觉大模型和生成端。

