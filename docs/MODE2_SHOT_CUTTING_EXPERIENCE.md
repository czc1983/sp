# Mode2 分镜切段经验

## 背景

这次问题表现为：自动切出来的第二段，画面末尾带入了第三段开头。用户看到的是“第二段存在了第三段的开头”，直觉上像自动切镜不准。

实际排查后确认：硬切检测大方向是对的，主要问题在“短镜头被检测器吞掉”和“按秒裁切时边界帧泄漏”。

## 事故现象

测试片段：

`E:\sp\.storyboard_mode2_projects\146d1f489c97ccd9\clips\mode2_shots\S001_00000000_00007012.mp4`

真实硬切点约为：

- `0.400s`
- `2.133s`
- `3.433s`
- `4.533s`

如果 UI 或手工切点保存成 `0.39s`、`2.13s` 这种秒级小数，再用普通 ffmpeg `-ss start -t duration` 裁切，就可能出现：

- 第二段最后一帧匹配原片第 `64` 帧；
- 原片第 `64` 帧其实已经是第三段第一帧；
- 用户看到的结果就是“上一段夹了下一段开头”。

## 根因

1. PySceneDetect 默认最短镜头长度太长

`ContentDetector` 默认 `min_scene_len=15` 帧，30fps 下约 `0.5s`。如果开头真实镜头只有 `0.4s`，默认会把这个切点吞掉。

经验：短剧/AI 重绘片段里，0.3-0.5 秒的短镜头是有效镜头，不能因为短就合并。

2. 秒级切段不是帧精确切段

旧逻辑使用：

```text
ffmpeg -i input -ss start -t end-start ...
```

这个方式按秒和时长裁，遇到 `0.387s`、`0.393s`、`2.130s` 这种不贴帧边界的时间，会出现取整误差。检测出的 `end` 本质上是“下一段开始”，所以应该按半开区间 `[start_frame, end_frame)` 裁，不能把 `end_frame` 放进上一段。

3. 旧切段缓存会误导 UI

`_mode2_create_reference_mask_subclips()` 之前如果目标 mp4 已存在就复用。修完裁切逻辑后，如果还读旧文件，用户仍然会看到旧错误结果。

经验：切段算法一改，输出文件名必须加版本后缀，或强制覆盖重切。

## 修复原则

1. 硬切检测要保留短真实镜头

`ContentDetector` 必须显式传入较短的 `min_scene_len`，不能依赖默认值。

当前策略：

```python
min_scene_len = max(1, int(round(float(min_scene_duration) * 10)))
ContentDetector(threshold=threshold, min_scene_len=min_scene_len)
```

2. 分镜切段必须先对齐到帧

新增专用函数：

- `align_segment_to_frames(video_path, start, end)`
- `cut_segment_precise(video_path, start, end, output_path)`

规则：

- 根据源视频 fps 把秒转换为帧号；
- `start_frame = round(start * fps)`；
- `end_frame = round(end * fps)`；
- 输出区间按 `[start_frame, end_frame)` 理解；
- 再用对齐后的秒数交给 ffmpeg 重编码。

3. Mode2 子镜头和 SCAIL2 子片段必须走同一套切段

不能出现 UI 预览用一种切法，真正送 SCAIL2 又用另一种切法。

当前要统一使用：

```python
cut_segment_precise(...)
```

使用位置：

- Mode2 子镜头预览：`_mode2_create_reference_mask_subclips()`
- SCAIL2 子片段运行：`_mode2_make_scail2_subshot_run_clip()`

4. 子镜头文件名带版本

当前精确切段文件名加 `_p1.mp4`，避免复用旧错误缓存。

## 验证标准

以 `0.39s-2.13s` 这段为例，修复后应自动对齐为：

```text
start = 0.400s, frame 12
end   = 2.133s, frame 64
```

输出第二段应为 52 帧：

- 第二段最后一帧接近原片第 `63` 帧；
- 第二段最后一帧不能接近原片第 `64` 帧；
- 第三段第一帧应接近原片第 `64` 帧。

本次验证结果：

```text
sub2 frames: 52
sub2 last vs src63: 0.718
sub2 last vs src64: 48.106
sub3 first vs src64: 0.274
```

这说明第二段没有再带入第三段开头。

## 以后不要再踩的坑

- 不要用肉眼显示的小数秒直接判断切段是否准，必须看帧号。
- 不要把 `end` 当成“上一段最后一帧”，它是“下一段第一帧”。
- 不要让分镜预览复用旧缓存，尤其是裁切算法变更后。
- 不要只修 UI 预览，SCAIL2 实际运行片段也必须同步修。
- 不要迷信自动硬切完全正确，必须给用户保留复查入口，但自动结果不能先天漏掉 0.4 秒这种短镜头。

## 回归命令

```powershell
python -m py_compile E:\sp\web_ui\server.py E:\sp\spvideo\ffmpeg_tools.py E:\sp\spvideo\scene_detector.py E:\sp\spvideo\scail2_client.py
git -C E:\sp diff --check
```

抽帧验证时，重点比对边界前后一帧，不要只看视频时长。
