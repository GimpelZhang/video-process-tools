# 试点视频评估报告（M3）

- 试点视频：`CARLA_Cosmos_p2.mp4`（823.3s，2560×1440@30fps，Vorbis 48kHz）
- 评估日期：2026-09-27
- 转写配置：faster-whisper 1.1.0 + Whisper large-v3，Silero VAD，批处理 batch_size=16，`language="zh"`，词级时间戳
- 真值：人工逐句校对稿 `data/srt/CARLA_Cosmos_p2.reviewed.srt`（全片校对，前 3 分钟作为量化区间）

## 量化结果（前 3 分钟，0–180s，33 条字幕，有效字符 860）

| 指标 | 结果 | 达标线 | 结论 |
|---|---|---|---|
| 字错率 CER（不含标点） | **0.23%**（编辑距离 2） | ≤ 5% | ✅ |
| 英文术语保留率 | **100%**（30/30） | ≥ 95% | ✅ |
| 时间轴偏差 | 人工全片核对通过 | ≤ ±200ms | ✅ |
| 幻觉条目 | 0 | 0 | ✅ |

## 全片人工修正记录（共 3 处）

1. `韩墨格` → `寒墨阁`（UP 主频道名，前 3 分钟内，占全部 2 个字符错误）
2. `CPU内存配乐是96G` → `CPU内存配了是96G`
3. `CARLA的TOMO` → `CARLA的demo`（英文词误听）

其余 144 条机器稿与人工校对一致。

## 转写模式对比

| 模式 | 耗时 | 速度倍率 | 质量差异 |
|---|---|---|---|
| 逐段（VAD filter） | 57.6s | 14.3x | 出现 `Kala`、`Inveda` 等术语误听 |
| **批处理 batch_size=16** | **15.1s** | **54.4x** | CARLA / NVIDIA / NuRec / SimBody 等术语全部正确 |

结论：批处理模式在速度与质量上均优于逐段模式，设为默认。

## 结论

- 主方案（faster-whisper large-v3 批处理）全面达标，**无需触发备选方案**（SenseVoice / turbo / WhisperX）。
- 英文术语通过 `initial_prompt` 术语表引导效果显著：加入 NVIDIA / NuRec 后识别立即正确。
- 后续处理其他主题视频时，只需在 `configs/default.json` 中按主题替换术语表。
