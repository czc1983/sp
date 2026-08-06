# Project Collaboration Rules

## Parallel Agent Preference

For this project, treat parallel agent usage as a standing user preference.

The main agent should default to acting as the coordinator/controller for non-trivial work: estimate the effort, judge whether the task can be split, assign concrete code reading, implementation, and verification slices to sub-agents when that will help, then integrate the results and make the final decision.

Use sub-agents when the work can be divided into independent reading, implementation, or verification slices with disjoint write scopes. Long-running tasks may be split across multiple rounds and multiple sub-agents, with each round narrowing the next step. Keep the work local when the task is small, tightly coupled, or would be slower to coordinate than to complete directly.

When a change is highly coupled in the same file or same small code path, do not let multiple agents edit it in parallel. Prefer one implementation agent, plus any number of read-only review or verification agents.

This preference does not override higher-priority safety, tool, or user instructions. It means: actively consider multi-agent execution by default, and use it whenever it materially speeds up the work without increasing risk.

## Mode 1 / Mode 2 Boundary

Mode 2 must remain independent from Mode 1. It may reuse ideas and small utility patterns from Mode 1, but should not import or depend on Mode 1's transfer/render workflow. When sharing concepts such as role anchors or timeline annotation, prefer Mode-2-specific functions, endpoints, and storage fields unless a small pure helper is clearly safe.

## Frontend Safety Rules

Large Mode 2 frontend files are high-risk change targets, especially `web_ui/story_generate_dashboard.html`.

- Before editing large frontend files, run `git status --short` and create a checkpoint with `scripts/safe_checkpoint.ps1`.
- Do not rewrite large Chinese HTML files with PowerShell `Set-Content`, shell redirection, or ad hoc full-file string generation.
- Prefer small `apply_patch` edits. If a large redesign is necessary, make a checkpoint commit first.
- After editing `story_generate_dashboard.html`, extract scripts and run `node --check`; also run `git diff --check`.
- If a UI change affects layout, routes, assistant guidance, or generation flow, keep a screenshot or clear visual verification note.
- Do not claim a page is fixed only because it loads. Confirm the expected UI state and workflow are still present.
- Do not let multiple agents edit the same large HTML file in parallel.

## Windows Path Safety

Chinese and other non-ASCII Windows paths are fragile when passed through PowerShell here-strings, terminal output, or copied logs.

- Do not hand-type non-ASCII paths inside inline Python launched from PowerShell.
- Prefer resolving files by ASCII filename, hash, or JSON key inside Python.
- Use `scripts/safe_path.py` before uploading or processing user files whose paths may contain non-ASCII characters.
- Read large project JSON in Python with `encoding="utf-8"` instead of PowerShell `ConvertFrom-Json` when encoding risk exists.
- Before calling ComfyUI, ffmpeg, or SCAIL2 with a user file, assert the resolved `Path(...).exists()` and log the ASCII-escaped JSON path.


## 翻译（外语对口型）模块

从 `E:\fy\foreign_lipsync` 合并进来的独立功能，与 Mode 1 / Mode 2 无关，不要让它依赖分镜/编排链路。

- 业务代码在顶层包 `dubvideo/`（ASR、OCR、翻译、MiniMax TTS、音色复刻、Comfy 对口型、混音）。
- 路由桥接在 `web_ui/dub_bridge.py`，由 `web_ui/server.py` 挂载：页面 `GET /foreign_lipsync`，接口 `/api/dub/...`，媒体复用现有 `GET /media?path=`。
- 页面 `web_ui/foreign_lipsync_dashboard.html` 通过 iframe 嵌入 `story_generate_dashboard.html` 的「翻译」导航页（`data-module="dub"`）。
- 运行数据在 `.dub_config/`（设置、密钥）、`.dub_projects/`、`.dub_uploads/`、`.dub_jobs/`、`.dub_exports/`，均按内容指纹无关的项目制存储；旧数据仍在 `E:\fy\foreign_lipsync`，不在 git 里。
