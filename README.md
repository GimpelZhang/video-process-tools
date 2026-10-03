# vidsub — 本地视频字幕提取与合成工具

输入 mp4 → 提取音频 → faster-whisper 转写为带时间轴的简体中文字幕（英文术语保留原文）→ 人工校对 SRT → ffmpeg 将字幕烧录回 mp4。

- ASR：faster-whisper 1.1.0 + Whisper large-v3 + Silero VAD + 词级时间戳
- 字幕：SRT / UTF-8 无 BOM
- 合成：默认硬烧录（h264_nvenc，自动回退 libx264），可选软字幕（mov_text）

## 环境要求

- Ubuntu 22.04、ffmpeg（含 libass）、Python 3.10、NVIDIA GPU（CUDA 运行库由 pip 提供，无需系统 cuDNN）
- 中文字体：Noto Sans CJK SC（`fc-list :lang=zh` 查看）

## 安装

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt   # 国内可加 -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install -e .                  # 安装 vidsub 命令（可编辑模式）
```

模型默认从 HuggingFace 下载到 `~/.cache/huggingface`。国内网络请先设置镜像：

```bash
export HF_ENDPOINT=https://hf-mirror.com
```

GPU 所需的 cuBLAS/cuDNN 来自 pip 包，工具启动时自动预加载，**无需手动设置 `LD_LIBRARY_PATH`**。

## 使用流程（四步，校对为人工中断点）

```bash
# 1) 抽取 16kHz 单声道音频
vidsub extract input.mp4
#    → data/audio/input.wav

# 2) GPU 转写生成 SRT
vidsub transcribe data/audio/input.wav
#    → data/srt/input.srt
# 也可直接对视频执行 transcribe（自动抽音），或一键运行：
vidsub run input.mp4

# 3) 人工校对
cp data/srt/input.srt data/srt/input.reviewed.srt
#   用任意文本/字幕编辑器（推荐 Subtitle Edit）修改正文
vidsub validate data/srt/input.reviewed.srt --video input.mp4

# 4) 合成
vidsub mux input.mp4 data/srt/input.reviewed.srt
#    → data/output/input.subbed.mp4
vidsub mux input.mp4 data/srt/input.reviewed.srt --soft
#    → 软字幕版（不重编码视频）
```

## 批量处理（多个视频）

把每个视频的绝对路径写入清单文件（每行一个），用两个脚本完成批量流水线：

```bash
# 清单示例 videos.txt：
# /home/junchuan/Videos/Bench2Drive.mp4
# /home/junchuan/Videos/HUGSIM.mp4

# 1) 批量抽音 + 转写 + 校验（产物 data/audio/*.wav、data/srt/*.srt）
bash scripts/batch_transcribe.sh videos.txt configs/batch2.json

# 2) 人工校对（可直接就地修改 data/srt/*.srt），然后批量校验：
#    见 scripts/batch_transcribe.sh 的清单用法，或逐个 vidsub validate

# 3) 批量合成：每个视频同时产出硬烧录与软字幕
bash scripts/batch_mux.sh videos.txt
#    → data/output/<名称>.subbed.mp4 与 data/output/<名称>.soft.mp4
```

批量约定与注意事项：

- **严格串行**：不要同时跑多个 GPU 转写/合成任务，会显存 OOM
- 无语音的视频（转写为 0 段）自动跳过；无条目的空 SRT 在合成时按 SKIP 处理
- 换一批主题时，复制 `configs/default.json` 改 `initial_prompt` 术语表即可（`batch2.json` 即第二批的示例，模型可直接指向本地缓存 `.hf-cache/large-v3`，离线运行）

## 人工校对约定

- **只改正文，不要改序号与时间轴行**；确需调整时间请用字幕工具，改完重新 `validate`。
- 保持 UTF-8、LF 换行、`.srt` 扩展名；不留空条目。
- 校对稿命名 `*.reviewed.srt`，原始稿保留以便 diff。

## 准确率评估

对人工精校真值（如前 3 分钟）计算 CER 与英文术语保留率：

```bash
python scripts/cer.py data/srt/input.srt data/srt/input.reviewed.srt
bash scripts/spotcheck.sh data/audio/input.wav 62.4 65.8 /tmp/clip.wav  # 抽片段试听
```

达标线：CER ≤ 5%、英文保留率 ≥ 95%、时间轴偏差 ≤ ±200ms、幻觉条目 0。

## 故障排查

| 现象 | 处理 |
|---|---|
| `Library cudnn*.9 not found` | 确认已安装 pip 版 nvidia-cudnn-cu12；工具会自动预加载 |
| 模型下载停滞/失败 | 设 `HF_ENDPOINT=https://hf-mirror.com`，用 `hf download` 断点续传；备选 ModelScope |
| nvenc 报错（ffmpeg 4.4） | 工具自动回退 `medium` 预设，再回退 libx264 |
| 烧录后字幕是方块/缺字 | force_style 指定系统已安装的中文字体（Noto Sans CJK SC） |
| 成片音轨兼容性差 | 源音轨为 Vorbis 时，成品自动转 AAC 192k |
| `CUDA failed with error out of memory` | 有其他任务占用显存；批量脚本本身串行，勿并发跑第二个 GPU 任务 |
| 批量处理时清单整行丢失/路径被截断 | ffmpeg 从 stdin 吞字节所致；勿让清单走 stdin，用脚本中的 FD3 方式 |
| 抽音提示"时长差超过 100ms" | 仅警告（退出码 2），WAV 已正常生成，可继续转写 |

## 测试

```bash
pip install -r requirements-dev.txt
pytest tests/ -q
```

## 许可证

Apache License 2.0（见 LICENSE）。
