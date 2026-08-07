# H3 Turbo LoRA × Ref2VA 兼容性测试计划

> 整理日期：2026-08-06 ｜ 状态：**等另一对话完工后开测**（34GB ref2va 完整版下载 + turbo 接入未完成）
> 触发方式：用户说「开测」→ 本对话按此计划执行

## 背景结论（已查证）

- T8 仓库 `t8star/minimax-h3-4step-turbo-loras-comfyui-exp` 只有两个文件：
  `minimax_h3_turbo_4步加速_comfyui.safetensors`（普通版，优先用）和 `..._ema_comfyui.safetensors`（EMA 版，原作者标注欠成熟，预期偏软）。
- 原作者 larryvrh 的 LoRA **在 FL2VA bf16 上训练验证**，Ref2VA 兼容性无任何文档——本次测试就是回答这个问题。
- 硬性参数（T8 README）：采样器 **euler**、调度器 **beta**、**8~10 步**（4 步爆音；或用双时钟采样器）。
- 模型必须**非剪裁**：`minimax_h3_ref2va_int8_convrot.safetensors`（34.04GB）。
- 风险预警：5090 32G 显存装 34GB 模型 + TE + VAE 大概率 offload，可能吃掉加速收益——**显存占用是首要测量项**。

## 开工前核对清单（另一对话的交付物）

- [ ] `/root` 下模型文件完整：`minimax_h3_ref2va_int8_convrot.safetensors` = 34.04GB（`ls -la` 核对字节数，防止断流残缺）
- [ ] LoRA 两个文件已放入 ComfyUI `models/loras/`
- [ ] 双时钟采样器节点已注册（T8/MiniMax H3/Audio 菜单下可见）
- [ ] 生产客户端 turbo 模式**默认关闭**，降级回路可用
- [ ] turbo 覆写里调度器已改为 euler + beta（不是 res_multistep/simple）
- [ ] ComfyUI 8189 正常，自启问题有结论
- [ ] pod 磁盘余量 ≥ 40GB（`df -h`）

## 测试矩阵（按顺序，一步挂了不硬闯下一步）

### T0 加载冒烟
- 最简工作流：Load Diffusion Model(ref2va 完整版) + LoraLoader(普通版 LoRA, strength 1.0)，只加载不采样。
- 判定：不报错、不 OOM 即过；`nvidia-smi` 记录加载后显存占用。

### T1 显存与速度基线
- 同参考片、480×864、124 帧、euler + beta、**8 步**、固定种子。
- 记录：采样耗时、峰值显存、是否触发 offload（ComfyUI 日志）。
- 对照基线：备忘录现有数字——5090 急速 219s/条、超分 338s/条（20 步 S+H / S+C）。

### T2 质量判定（对口型场景核心）
- 能播、不爆音、脸部不乱码只是及格线；
- **口型对齐**：生成片与配音音频的唇形同步，与老链路 20 步产物肉眼对比 + 截图存档；
- 脸部特写稳定性（T8 自承低步数拟合差、小脸劣化）。

### T3 参数变体（T2 过了才做）
- 步数：4 / 8 / 10（4 步看爆音是否复现，双时钟采样器单独测一组）；
- LoRA：普通版 vs EMA 版；
- 叠加：LoRA8 + Sage + BlockCache（注意：不与 HyperStep 同开，备忘录已证冲突）。

### T4 生产准入判定
- 全部通过 → turbo 模式可对 R2V 镜头开放（仍默认关，逐项目开）；
- 口型明显劣化 → Ref2VA 留在老链路，turbo 只给 FL2VA 白膜单角色镜头用（需另下 fl2va 32GB，先确认磁盘）。

## 执行方式

- pod 访问：`bash scripts/cpod_ssh4.sh '<远程命令>'`（本地 E:\sp 下执行）
- 传文件：`bash scripts/cpod_ssh4.sh -scp <本地> <远程>`
- 测试产物存档：`E:\sp\abtest\turbo_ref2va_<日期>/`（含生成片、对照截图、耗时表）
- 结果追加到 `H3_加速与5090迁移备忘录.md`

## 回退预案

- LoRA 挂不上 / OOM / 口型崩 → turbo 保持关闭，生产照旧 20 步 S+C/S+H，无损失；
- 模型文件残缺 → aria2 断点续传或换 ModelScope 镜像。
