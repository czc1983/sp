# H3 加速组合实测档案（5090 · cu132 · 2026-08-06）

测试环境：RTX 5090 32G、torch 2.13.0+cu132、ComfyUI 0.30.0、全局 `--use-sage-attention`。
测试片：S024 参考视频 73f → 生成 124 帧 480×864 20 步同种子；长片组为 277 帧。
判定维度：耗时 + 场景忠实度（对比原片：户外院子/中式门廊/地面土渍/无镜子）。

## 短片（124 帧）

| 组合 | 耗时 | 忠实 | 结论 |
|---|---|---|---|
| 纯 sage（基准） | 131.6s | ✅ | 基准 |
| **sage + BlockCache 0.12（生产定案）** | **72.3s** | ✅ 几乎无损 | **1.8x，首选** |
| sage + BlockCache 0.06 | 104.0s | ✅ | 太温和 |
| sage + HyperStep Turbo | 54.3s | ⚠️ 场景对但不听提示词（改人偶颜色/加五官/姿态偏离） | 快但不可用（用户实测确认） |
| SolAttn exact off | 90.6s | ❌ 丢参考 token，场景漂成室内+脑补镜子 | 禁用 |
| SolAttn exact_kv | 142.7s | ✅ | 比基准还慢，无价值 |
| SolAttn + BlockCache | — | — | **代码级冲突**：BlockCache 的 H3BlockCacheHit 异常被 SolAttn 包装层吞掉直接报错，不可叠加 |
| SolAttn exact_kv + HyperStep | 143.4s | — | 慢且 HS 本身不可用 |

## 长片（277 帧，验证"SolAttn 长片翻身"假设）

| 组合 | 耗时 | 结论 |
|---|---|---|
| 纯 sage（基准） | 249.1s | 基准 |
| 纯 HyperStep Turbo | 123.2s | 2.0x 但同样不听提示词 |
| SolAttn exact_kv 单独 | 270.4s | ✅ 忠实但比基准慢 8.5%，**未翻盘** |
| SolAttn exact_kv + HyperStep | 274.7s | 最差 |

SolAttn 在 5090 走 flex_attention（Python 路径），固定开销 > 稀疏收益；片长 ≤277 帧均为负资产。
备用模板 `comfy_workflows/h3_whitemask_solattn_api.json`（exact_kv 唯一可用形态）已留存，
若未来片长上千帧或该插件出 SM120 原生 kernel 版，可重新评估。

## 生产配置（已写进 spvideo/minimax_h3_client.py）

- sage：ComfyUI 启动参数 `--use-sage-attention` 全局生效，无需节点（两台服务器均有 cron 守护自动拉起/纠偏）
- BlockCache 0.12：代码自动注入（env `H3_BLOCKCACHE=off` 可关，`H3_BLOCKCACHE_THRESHOLD` 调阈值，
  远程无插件时自动降级不报错）
- HyperStep / SolAttn：不进生产
