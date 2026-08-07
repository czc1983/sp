# H3 5090 生产链路与加速实测

更新时间：2026-08-07。适用于 Mode 2 的白膜生成与白膜换人工作流。

## 当前生产定案

两套 H3 API 工作流固定使用同一条模型、采样和输出链路：

```text
UNETLoader 127
  minimax_h3_ref2va_pruned_int8_convrot.safetensors
    -> MiniMaxH3MemoryEfficientSageAttentionPatch 204
    -> MiniMaxH3BlockCacheT8 205
         residual_diff_threshold = 0.12
         start_percent = 0.08
         end_percent = 0.95
         max_consecutive_hits = 2
         cache_device = cpu
         metric_stride = 8
         verbose = false
    -> BasicScheduler 124 (simple, 固定 20 步)
    -> BasicGuider 126

KSamplerSelect 123 = res_multistep
SamplerCustomAdvanced 125 = 123 sampler + 124 sigmas + 126 guider

125 -> VAEDecode 122 -> FlashVSRNode 206 -> CreateVideo 130
125 -> VAEDecodeAudio 121 -----------------> CreateVideo 130
```

FlashVSR 固定为 `FlashVSR-v1.1`、`tiny`、2 倍放大，并启用 `tiled_vae` 与
`tiled_dit`。H3 基础生成尺寸按源视频宽高比自动推导到约 0.4MP，宽高均为 32
的倍数；显式传入正宽高时保持调用值。日志同时记录源尺寸、基础尺寸和 2 倍输出尺寸。

采样步数固定为 `20`，BlockCache 阈值固定为 `0.12`，均不接受环境变量或接口参数
覆盖。调用方传入非 20 步时会明确报错，不会静默生成另一条链路。固定链路不可关闭；
远程缺少 Sage patch、BlockCache 或 FlashVSR 任一节点时直接报错，不切换模型，也不
降级为其他加速或无加速链路。

## 5090 实测依据

测试环境：RTX 5090 32G、torch 2.13.0+cu132、ComfyUI 0.30.0。历史测试片为
S024，生成 124 帧、480x864、20 步、同种子。以下耗时是接入 FlashVSR 输出节点前的
生成阶段对比，用于确定 BlockCache 参数：

| 组合 | 耗时 | 画面结论 |
|---|---:|---|
| Sage 基准 | 131.6s | 忠实基准 |
| Sage + BlockCache 0.12 | 72.3s | 几乎无损，约 1.8x |
| Sage + BlockCache 0.06 | 104.0s | 忠实但收益偏低 |

因此生产阈值定为 `0.12`。FlashVSR 2 倍输出是当前工作流的一部分，其耗时应在后续
端到端基准中单独记录，不能与上表的旧生成阶段耗时直接比较。

## 已废弃路线

- HyperStep：提示词服从和人物细节会偏离基准，不用于生产。
- Turbo LoRA 与 DualClock：全量 UNet、低步数采样和对应测试脚本均已移除。
- SolAttn：会丢参考 token 或在保真模式下变慢，并与 BlockCache 存在异常处理冲突；
  模板和补丁脚本均已移除。
- 全量非裁剪 UNet、无缓存回退和多级 fallback：均不属于当前生产链路。

这些路线不作为应急降级选项。生产结果必须来自本页记录的固定链路，避免不同任务
在模型、采样步数或输出分辨率上悄然分叉。
