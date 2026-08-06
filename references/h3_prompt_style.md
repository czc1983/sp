# MiniMax H3 提示词标准写法（官方示例拆解）

来源：ComfyUI MiniMax H3 官方节点示例提示词（2026-08-03 用户确认可用作标准写法）。

## 结构公式（按顺序写）

1. **镜头总纲**：`Single continuous shot, {时长} seconds, one take, no cuts.` + 镜头风格（如 third-person chase camera, steadicam fluidity）+ 画质标签（photorealistic, ultra-high-definition, hyperdetailed, 35mm film quality, professional color grading, sharp focus, high detail texture, subtle film grain, depth of field mastery）+ `Aspect ratio {16:9|9:16}.`
2. **主体描述**：人物/角色的外观、服装、姿态、动作细节。
3. **环境描述**：场景、天气、光线、氛围（越具体越好）。
4. **逐秒时间轴**（H3 特色，必须写）：`Second 0–1: ... Second 1–2.5: ... Second 2.5–4: ... Second 4–5: ...` —— 每一秒发生什么动作、镜头怎么动，时长变了就重新分段。
5. **特效块**：`[VFX: 逗号分隔的特效关键词]`
6. **收尾约束**：一句话总结风格 + 负面清单（`No text, no logos, no dialogue. No ...`）。

## 官方示例原文（5 秒 16:9 极地飞行）

```
Single continuous shot, 5 seconds, one take, no cuts. Cinematic oner, third-person chase camera, steadicam fluidity, photorealistic, ultra-high-definition, hyperdetailed, 35mm film quality, professional color grading, sharp focus, high detail texture, subtle film grain, depth of field mastery. Aspect ratio 16:9.A lone rider cuts through a violent arctic storm above a frozen fjord canyon. The rider is a compact figure in layered polar flight gear, frost-lined goggles, fur-lined collar, and a torn white wind shell pressed low against the mount's neck, gloved hands locked on a leather harness.The mount is an original creature design, not a dragon: a sleek frost-glider with a long manta-like body, raptor shoulders, and translucent wing membranes edged with crystalline ice ridges. Pale gold vein-light pulses along its wing seams and throat, the only warmth in a frozen steel-blue world. No scales, no fire breath, no classic dragon silhouette.The storm is brutal and cold: slate-grey thunderheads, sideways sleet, black cloud canyons, lightning tearing across hanging glaciers and sheer ice walls. Snow streaks horizontally past the lens.Flight choreography for one unbroken 5-second arc:Second 0–1: The camera chases low and tight behind the rider as the frost-glider banks hard between two ice pillars, wing edges slicing spray from the storm.Second 1–2.5: It whips around to the side, tearing through a curtain of driving snow, lightning flashing behind the rider's silhouette, the pale gold wing-light cutting through blue murk.Second 2.5–4: Without warning, a colossal ice-ridge serpent erupts head-on from a wall of fog and shattered spindrift directly ahead. Time violently ramps into extreme slow motion. At near-standstill, the giant's frost-crusted head fills the frame, one enormous pale-gold eye reflecting the tiny rider, breath steaming, ice crystals and snowflakes hanging frozen in the air, horizontal anamorphic flares raking the lens.Second 4–5: Reality snaps back to full speed. The frost-glider rolls hard beneath the serpent's jaw and dives into a narrow ice ravine, the camera surging after it as fog and snow swallow the frame.[VFX: pale gold vein bioluminescence, colossal ice serpent, volumetric storm fog, lightning, suspended ice particles in slow motion]Epic, cinematic, photoreal, fluid, uninterrupted. No text, no logos, no dialogue. No traditional dragon, no dragon-versus-dragon confrontation, no castle, no fantasy army.
```

## 套用到白膜（R2V）的要点

- 第 1 段改成：`Keep the reference video's timeline, camera movement, framing and character actions exactly unchanged.`
- 主体段：把所有人物替换为哑光纯白低多边形 3D 人偶（matte pure-white low-poly 3D mannequins / white models），动作姿态位置完全复刻参考视频。
- 环境段：`Keep the background, props and lighting identical to the reference video.`
- 逐秒时间轴：照抄分镜的动作描述（如果有），否则省略让模型跟随参考视频。
- 收尾：`No text, no subtitles, no logos.`


## 已定稿：通用彩色白膜提示词 v5（2026-08-04 用户直接给定，已写入模板）

设计要点：不再指定具体颜色（废掉 v3 的性别定色和 v4 的位置定色），改为约束
「同一人偶色调恒定 + 人偶之间颜色明显区分」，并新增「口型与音频对齐」要求。

```
把视频里的人物全部转换成彩色树脂关节人偶素体：口型与音频对齐，同一人偶的颜色保持同一种色调不变，与其它人偶颜色要有明显区别。衣服完全消融进身体，皮肤和服装统一为同色的光滑哑光树脂材质，像BJD娃娃素体；光头没有头发、没有眉毛和睫毛、身体带球形关节、极简人偶面部，但眼神不变、有明确视线方向的造型；背景、镜子和物体变成纯白色哑光材质。严格按照参考视频的构图、人物数量、位置、姿态和遮挡关系演变，运镜和时间轴完全不变，仅保留柔和光影明暗。最关键的要求：必须准确保留原片人物的眼神和表情——眼睛，眼睑开合程度与参考视频一致，视线方向完全一致，情绪状态不变，绝不能改；绝对不要真实皮肤质感、不要头发丝、不要睫毛、不要布料质感。
```

实际生效位置：`comfy_workflows/h3_whitemask_api.json` 节点 136（API 链路，前端不传 prompt 时用模板原文）和 `comfy_workflows/h3_whitemask_r2v.json` 节点 138 PrimitiveStringMultiline（UI 链路）。
