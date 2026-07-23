# Git Checkpoint Workflow

本项目以后按“先保命，再改功能”的节奏处理 Git 和 GitHub。

## 什么时候必须打 checkpoint

以下情况必须先打本地 checkpoint：

1. 修改 `web_ui/story_generate_dashboard.html`。
2. 修改 Mode2 助手、路线、资产库、生成流程。
3. 修改服务器生成链路，例如 Scail2、SAM3、白膜、分轨。
4. 一次改动预计超过 30 分钟。
5. UI 已经被用户确认“这个版本好看/能用”。

## 推荐命令

只检查并保存 patch：

```powershell
powershell -ExecutionPolicy Bypass -File scripts/safe_checkpoint.ps1 -Name "before-mode2-change"
```

检查后创建本地提交：

```powershell
powershell -ExecutionPolicy Bypass -File scripts/safe_checkpoint.ps1 -Name "mode2-stable-ui" -Commit
```

用户明确同意后再推 GitHub：

```powershell
powershell -ExecutionPolicy Bypass -File scripts/safe_checkpoint.ps1 -Name "mode2-stable-ui" -Commit -Push
```

## GitHub 策略

1. GitHub 只保存用户认可的稳定版本。
2. 半成品可以本地 commit，但不要自动 push。
3. 每次用户说“这版可以”“这个效果对了”“界面这样行”，马上建议提交。
4. 出事故时优先回滚到 commit，不依赖 `.bak` 文件。
5. `.bak` 只用于临时抢救，不能替代 Git。

## 智能守护脚本

项目现在提供两层守护：

1. `scripts/safe_checkpoint.ps1`：手动检查，生成 patch，必要时 commit/push。
2. `scripts/smart_git_guard.ps1`：智能守护，能自动过滤源码文件、跳过视频/图片/模型/日志，并在检查通过后自动 checkpoint。

只做本地 patch，不 commit：

```powershell
powershell -ExecutionPolicy Bypass -File scripts/smart_git_guard.ps1 -Name "manual-guard"
```

检查通过后自动本地 commit：

```powershell
powershell -ExecutionPolicy Bypass -File scripts/smart_git_guard.ps1 -Name "stable-work" -AutoCommit -IncludeUntracked
```

检查通过后自动 commit 并 push GitHub：

```powershell
powershell -ExecutionPolicy Bypass -File scripts/smart_git_guard.ps1 -Name "stable-work" -AutoCommit -Push -IncludeUntracked
```

### 安装定时守护

默认每 10 分钟自动检查一次，只保存 patch，不提交：

```powershell
powershell -ExecutionPolicy Bypass -File scripts/install_smart_git_guard.ps1
```

每 10 分钟自动本地 commit：

```powershell
powershell -ExecutionPolicy Bypass -File scripts/install_smart_git_guard.ps1 -AutoCommit
```

每 10 分钟自动本地 commit 并 push：

```powershell
powershell -ExecutionPolicy Bypass -File scripts/install_smart_git_guard.ps1 -AutoCommit -Push
```

自动 push 只适合稳定开发阶段。大改 UI 时建议先只开自动本地 commit，等用户确认版本好看、流程正确后再 push。

守护脚本默认只纳入源码/文档范围：

- `web_ui/`
- `spvideo/`
- `scripts/`
- `docs/`
- `tests/`
- `ui/`
- `AGENTS.md`
- `README.md`
- `WORK_RECORD.md`

会跳过：

- 视频、图片、音频
- 模型权重
- 日志
- `.bak`
- `.codex/`
- `.tmp*`
- 生成项目目录

## 改坏后的恢复顺序

1. 停止继续加功能。
2. 保存当前坏版本 patch。
3. 查看最近 commit：`git log --oneline -10`。
4. 找用户确认过的稳定点。
5. 从稳定点恢复或手工对比恢复。
6. 先验证页面能运行，再恢复 UI 视觉，再恢复新增逻辑。

## 检查标准

每次关键改动后至少检查：

```powershell
git status --short
powershell -ExecutionPolicy Bypass -File scripts/safe_checkpoint.ps1 -Name "after-change-check"
```

对于 `story_generate_dashboard.html`，必须通过：

1. 没有 Unicode replacement character 乱码。
2. `node --check` 通过。
3. `git diff --check` 通过。
4. 页面能打开。
5. 关键按钮和当前路线状态能看到。
