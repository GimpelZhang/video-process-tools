# 本地视频字幕提取与合成工具 实施 Plan（v1）

> 目标：输入 mp4 → 提取音频 → 转写为带时间轴的简体中文字幕（英文术语保留英文原文）→ 人工校对 SRT → 将字幕合成回 mp4，时间轴与音画严格对齐。
>
> 制定日期：2026-09-27
> 工作目录：`/home/junchuan/video_process`

---

## 0. 环境核实结果（已实际执行命令核实）

| 项目 | 实测结果 | 结论 |
|---|---|---|
| OS | Ubuntu 22.04，内核 6.8.0-138 | 满足 |
| GPU | RTX 3090 24GB，驱动 580.178.04（支持 CUDA 13.0），空闲显存约 23.9GB | 满足 |
| CUDA toolkit | 本机装有 `/usr/local/cuda-11.8` 和 `/usr/local/cuda-12.8`，`nvcc` 为 12.8 | 满足，驱动向后兼容 CUDA 12 运行时 |
| cuDNN | 系统目录与 cuda-12.8 目录下**均未发现 cuDNN** | 需用 pip 安装 `nvidia-cudnn-cu12`（见 M2） |
| ffmpeg | 4.4.2，编译参数含 **`--enable-libass`**；`subtitles`、`ass` filter 可用 | 字幕烧录可用 |
| 编码器 | `h264_nvenc`、`hevc_nvenc`、`libx264` 均可用 | GPU 编码可用，libx264 可回退 |
| Python | 3.10.12，pip 22.0.2，`python3 -m venv` 可用 | 满足。Ubuntu 22.04 自带 pip 22 **不触发 PEP 668**（23.04 才有），但仍统一使用 venv |
| 中文字体 | 有 **Noto Sans CJK SC**（`/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc`），另有 Noto Serif CJK SC、文鼎等共 89 个 zh 字体条目 | 烧录用 Noto Sans CJK SC |
| git | 2.34.1 | 满足 |
| 磁盘 | `/` 剩余 372GB | 满足 |
| HF 缓存 | `~/.cache/huggingface` 已存在 | 注意模型默认下载位置 |

### 试点视频实测信息（ffprobe）

- 路径：`/home/junchuan/Documents/bilibili/online/CARLA_Cosmos_p2.mp4`（存在，136,594,343 字节 ≈ 130MB）
- 时长：**823.322 秒（约 13 分 43 秒）**
- 视频：h264，**2560×1440，30fps**
- 音频：**Vorbis**，48kHz，立体声（B站抓流常见封装；mp4 里装 vorbis 不标准，但 ffmpeg 可正常解码）

---

## 1. 技术选型对比与决策

### 1.1 ASR 候选方案对比（证据见第 8 节引用）

| 维度 | faster-whisper (CTranslate2) | WhisperX | FunASR / Paraformer-zh | SenseVoice-Small |
|---|---|---|---|---|
| 模型 | Whisper large-v3 / turbo 转换权重 | 前端仍为 faster-whisper，后端加 wav2vec2 强制对齐 | Paraformer-zh 220M（large-vad-punc 等） | 234M，非自回归，仅 Small 版 |
| GitHub 活跃度 | 25.6k stars，持续维护，PyPI 最新 1.1.0，MIT | 24.3k stars，BSD-2；2026 年仍在更新（Context-Aware Batching），但 issue/PR 积压多 | 20.5k stars，5900+ commits，FunASR 1.4.16，MIT（**模型权重另有许可**） | 9.4k stars，**权重为 FunASR 模型许可（非 MIT，商用需署名/遵守条款）** |
| 中文准确率 | large-v3 是 Whisper 系列中文最好的版本；turbo 略低（介于 large-v2 与 large-v3 之间） | 同 faster-whisper，转写准确率不变 | 中文基准强，标点自然 | 中文很强，官方称 15x 快于 Whisper-Large |
| 中英混读保留英文 | **天然支持**：`language="zh"` 强制中文为主语言，Whisper 词表含拉丁字母，英文术语直接输出英文；可通过 `initial_prompt` 引导 | 转写阶段同 faster-whisper；**但中文对齐模型 `jonatasgrosman/wav2vec2-large-xlsr-53-chinese-zh-cn` 词表基本不含拉丁字母，混在中文里的英文词会对齐失败（拿不到时间戳）** | 模型标注 zh/en，但 code-switch 时英文常被音译/译成汉字（社区已知现象），需实测 | 仓库自带中英混读样例（`asr_example_cn_en`），code-switch 能力是设计目标之一，英文保留能力预期好于 Paraformer |
| 时间戳粒度 | 自带**词级时间戳**（DTW 跨语言/跨字符集统一，中英混合不影响），句段由 VAD 切分 | 词/音素级强制对齐，英文等对齐模型成熟；**中文对齐模型年代较早（XLSR-53 时代），数字/符号/英文无法对齐是官方明确列出的限制** | 句/段级时间戳（`sentence_info`），无成熟词级 | 支持词级 `[start_ms,end_ms]`，但**直推上限 30 秒**，长音频靠 FSMN-VAD 切段，边界处时间戳质量官方承认未经充分验证 |
| 幻觉/静音问题 | Silero VAD 内置（`vad_filter`），有 `hallucination_silence_threshold` 等开关 | pyannote VAD 预处理，官方称显著降幻觉，无 WER 损失 | FSMN-VAD 成熟 | FSMN-VAD；但输出带**情绪/事件标签（HAPPY、BGM、Laughter 等）需后处理剔除** |
| RTX 3090 速度/显存 | large-v3 批处理约 10–20x 实时，14 分钟视频预计 1–2 分钟；显存约 4–8GB；turbo 更快 | large-v2 beam=5 <8GB；额外加载对齐模型 | 3090 上极快（非/弱自回归），但需 PyTorch 全栈 | 极快（非自回归），需 PyTorch + FunASR 全栈 |
| 安装复杂度 | **最轻**：不需要 PyTorch；pip 装 cublas/cudnn 即可 | 重：需 CUDA 12.8 toolkit + PyTorch + 对齐模型 | 重：PyTorch + torchaudio + ModelScope/HF 多模型 | 重：同 FunASR |

### 1.2 决策结论

**推荐主方案：faster-whisper 1.1.0 + Whisper `large-v3` + Silero VAD + 批处理 + 词级时间戳，自写 SRT 生成器。**

理由：

1. **准确率**：large-v3 是开源多语 ASR 里中文第一梯队，且 Whisper 的转写（transcribe）模式对中英混读的处理就是"听到英文写英文"，与需求"保留英文原文、不翻译"完全吻合；`language="zh"` 锁定主语言，避免英文术语导致整段语言误判。
2. **时间戳**：faster-whisper 自带的词级时间戳对中英混合文本统一有效；而 WhisperX 的中文对齐模型对拉丁字母词对齐失败，恰好打在本需求的痛点上，因此 WhisperX 不作为主链路。
3. **工程成本**：CTranslate2 不依赖 PyTorch，venv 体积小、安装坑少；3090 上速度冗余很大。
4. 许可证干净（MIT）。

**关键转写参数（M2 落地为默认配置）：**

```python
model = WhisperModel("large-v3", device="cuda", compute_type="float16")
segments, info = model.transcribe(
    audio,
    language="zh",                    # 强制中文为主语言，不使用自动检测
    task="transcribe",                # 转写而非翻译（默认即 transcribe，显式写出）
    beam_size=5,
    vad_filter=True,                  # Silero VAD，过滤音乐/长静音引发的幻觉
    vad_parameters=dict(min_silence_duration_ms=500, speech_pad_ms=200),
    word_timestamps=True,
    condition_on_previous_text=False, # 关闭上文条件，防止错误滚动放大与重复幻觉
    hallucination_silence_threshold=2.0,
    no_speech_threshold=0.6,
    compression_ratio_threshold=2.4,
    log_prob_threshold=-1.0,
    temperature=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0],  # 温度回退，默认策略显式化
    initial_prompt=(                  # 引导简体字、中文标点、英文术语表
        "以下是普通话的技术演讲，夹杂英文术语时保留英文原文。"
        "CARLA, Cosmos, GPU, simulation, reinforcement learning, "
        "autonomous driving, neural network, token, benchmark。"
    ),
)
```

> 批处理使用 `BatchedInferencePipeline(model).transcribe(..., batch_size=16)`（VAD 在批处理模式默认开启）。批处理与逐段转写在 M2 各跑一次对比，若批处理出现断句劣化则回退逐段模式。

**备选方案与切换触发条件：**

| 备选 | 何时启用 |
|---|---|
| `turbo` 模型（同一套代码，改 `--model turbo`） | large-v3 速度不可接受时；14 分钟视频预计用不到，仅作为长视频（>1h）的提速手段，启用前必须重跑 M3 准确率评估 |
| **SenseVoice-Small**（经 FunASR，FSMN-VAD 切段） | M3 评估中 large-v3 中文 CER > 8% 且调参无改善，或中英混读英文保留率 < 90%。用 SenseVoice 重跑试点，达标则并行保留，后处理剔除情绪/事件标签，并用其词级时间戳重建 SRT |
| **Paraformer-large-vad-punc** | 仅当中文标点/断句成为主要问题且 SenseVoice 也不达标时作为对照；其英文 code-switch 弱点已知，不作为首选 |
| WhisperX 强制对齐 | 仅当 M3 发现 faster-whisper 自带词级时间戳偏移系统性超过 ±200ms 时启用；**需接受中文句子中英文词可能无时间戳**，届时对英文词回退使用 faster-whisper 原始词时间戳（混合策略） |

---

## 2. 字幕格式与人工编辑环节

### 2.1 格式决策：SRT

- **默认输出 SRT**，UTF-8 **无 BOM**（libass 对带 BOM 的 SRT 个别老版本有解析问题；现代编辑器均能识别 UTF-8）。
- 时间戳格式 `HH:MM:SS,mmm`，序号从 1 连续递增。
- 不用 VTT：VTT 的 `.` 毫秒分隔与 cue 设置对人工编辑无增益，且 ffmpeg 烧录同样走 libass。
- 不用 ASS：ASS 适合复杂排版，但文本不可直接被普通文本编辑器安全维护，人工误改样式头的风险高；本需求样式在烧录命令中统一控制（见第 3 节）。如未来需要双语字幕再评估 ASS。

### 2.2 断句与换行策略（在 SRT 生成器中实现）

输入为 faster-whisper 的词级时间戳（中文按字、英文按词），按以下规则聚合：

1. 单条字幕显示 **最多 2 行**；每行中文上限 **20 个汉字**（含英文术语按显示宽度折算，英文约 2 个字符折 1 个汉字宽）。
2. 单条时长目标 **1.5–7 秒**；阅读速度上限 **17 汉字/秒**，超出则在标点处再切分。
3. 切分优先级：句末标点（。！？）> 分句标点（，、；）> 词组边界（用词时间戳找停顿 ≥ 300ms 处）。
4. 相邻条目间留 **30–80ms 间隙**（避免播放器重复帧）；**不允许时间重叠**，时间严格单调递增。
5. 纯英文术语不独占一条，跟随所在中文句段。
6. 每条结束时间不晚于该段最后一个词/字的结束时间 + 80ms。

### 2.3 人工编辑约定（写入 README）

- 只改正文字，**不要改序号与时间轴行**；如需调整某条出现/消失时间，使用支持波形的字幕工具（推荐 Subtitle Edit / VS Code 扩展），改完必须重新过 `validate`。
- 保存时保持 UTF-8、LF 换行、SRT 扩展名不变；文件内不要留空条目（正文为空、只有时间轴的块）。
- 校对稿另存为 `*.reviewed.srt`，原始稿保留，便于 diff 和复盘。

### 2.4 SRT 校验脚本（`transcribe validate`）

检查项：

1. 编码可按 UTF-8 解码，无乱码/无 BOM 异常；
2. 序号连续且从 1 开始；
3. 时间格式合法、`start < end`；
4. 相邻条目不重叠、单调递增（容差 1ms）；
5. 单条时长 > 0 且 ≤ 10s（超长给 warning）；
6. 空文本块、超过 2 个换行、超长行（>22 汉字宽）给 warning；
7. 最后一条结束时间不超过视频时长（需 `--video` 时检查）。
退出码：error 非 0，纯 warning 为 0。

---

## 3. 字幕合成环节

### 3.1 硬烧录 vs 软字幕

- **默认硬烧录（burn-in）**：任何播放器都可见，时间轴由渲染保证，不存在播放器支持问题。使用 ffmpeg `subtitles` filter（底层 libass）。
- 同时提供**软字幕**可选：mp4 容器用 `mov_text`（MP4 不支持直接装 SRT，ffmpeg 会转 tx3g），命令见 3.4，供不想重编码视频时使用。
- 注意：试点视频音轨为 vorbis（mp4 中非标准），烧录重编码视频时**音频统一转 AAC 192k**，顺带修正兼容性；软字幕方式可 `-c:a copy`。

### 3.2 字幕样式（针对 1440p / Noto Sans CJK SC）

- 字体 `Noto Sans CJK SC`（已核实存在）；
- 白字、黑色描边（Outline 2）、轻微阴影（Shadow 0.5）、底部居中，底边距约画面高 4%。

libass 对 SRT 的默认脚本分辨率 PlayResY=288 并自动缩放，因此以下字号在各分辨率下比例一致（约为画面高的 4.5%）：

```
FontName=Noto Sans CJK SC,FontSize=18,PrimaryColour=&H00FFFFFF,
OutlineColour=&H00000000,BorderStyle=1,Outline=2,Shadow=0.5,MarginV=12
```

（颜色为 ASS 的 AABBGGRR；`&H00FFFFFF` = 不透明白色。）

### 3.3 默认烧录命令（GPU 编码，M4 验收用）

```bash
# 注意 subtitles filter 的文件名转义：路径用单引号包住，':' 需写成 '\:'
# 本路径无空格无冒号（除开头盘符式语法外 Linux 下没有），直接写绝对路径即可
ffmpeg -y -i /home/junchuan/Documents/bilibili/online/CARLA_Cosmos_p2.mp4 \
  -vf "subtitles=/home/junchuan/video_process/data/srt/CARLA_Cosmos_p2.reviewed.srt:force_style='FontName=Noto Sans CJK SC,FontSize=18,PrimaryColour=&H00FFFFFF&,OutlineColour=&H00000000&,BorderStyle=1,Outline=2,Shadow=0.5,MarginV=12'" \
  -c:v h264_nvenc -preset p4 -rc vbr -cq 23 -b:v 0 \
  -c:a aac -b:a 192k \
  /home/junchuan/video_process/data/output/CARLA_Cosmos_p2.subbed.mp4
```

说明：

- ffmpeg 4.4 的 h264_nvenc 已支持 `p1–p7` 新预设名（p4 约等于旧 medium）与 `-cq`；若个别参数报错，回退写法：`-preset medium -cq 23`。
- **CPU 回退**：`-c:v libx264 -preset medium -crf 20`（14 分钟 1440p 用 3090 主机的 CPU 预计 15–30 分钟）。
- force_style 中 `&` 在 shell 单引号内安全；若命令行报 filter 解析错误，优先检查 `&` 与引号。
- 可选先抽一帧验证样式：在 filter 链加 `,trim=start=60:duration=2` 或用 `-ss 60 -t 5` 只渲染 5 秒小样，确认字体/字号后再跑全片。

### 3.4 软字幕命令（可选）

```bash
ffmpeg -y -i /home/junchuan/Documents/bilibili/online/CARLA_Cosmos_p2.mp4 \
  -i /home/junchuan/video_process/data/srt/CARLA_Cosmos_p2.reviewed.srt \
  -map 0:v -map 0:a -map 1:0 -c copy -c:s mov_text \
  -metadata:s:s:0 language=chi \
  /home/junchuan/video_process/data/output/CARLA_Cosmos_p2.soft.mp4
```

---

## 4. 工程结构与依赖管理

### 4.1 目录结构

```
/home/junchuan/video_process/
├── .gitignore
├── README.md                    # 使用说明 + 人工校对约定
├── requirements.txt             #  pinned 版本
├── configs/
│   └── default.json             # 转写/断句/样式默认参数（dataclass 序列化）
├── src/vidsub/
│   ├── __init__.py
│   ├── cli.py                   # argparse 子命令入口
│   ├── audio.py                 # 抽音频（subprocess 调 ffmpeg）
│   ├── transcribe.py            # faster-whisper 封装 + 参数
│   ├── srt.py                   # 词级时间戳 -> 断句 -> SRT
│   ├── validate.py              # SRT 校验
│   └── mux.py                   # 烧录/软字幕命令构造与执行
├── scripts/
│   ├── spotcheck.sh             # 抽片段音频/短视频用于试听核对
│   └── cer.py                  # 字符级 CER / 英文保留率统计
├── tests/
│   ├── test_srt.py              # SRT 解析/生成/断句规则单测
│   └── test_validate.py
├── docs/
│   └── Plan_v1.md
├── .venv/                       # 不入库
└── data/                        # 全部不入库
    ├── audio/                   # 抽出的 16k wav
    ├── srt/                     # 生成稿 + *.reviewed.srt
    └── output/                  # 合成产物
```

模型缓存策略：**不建项目内 models 目录，默认用 `~/.cache/huggingface`**（已存在），多项目共享；如需隔离，运行时设 `HF_HOME=/home/junchuan/video_process/.hf-cache`（该目录加入 .gitignore）。模型权重（large-v3 约 3GB）一律不入库。

### 4.2 依赖管理：venv + pip + requirements.txt

选 venv + pip 而非 uv/poetry：本机 pip 22 / Python 3.10 环境标准、团队学习成本低、需求只有一个应用；requirements.txt 锁定主版本即可保证可复现。若后续依赖膨胀再迁移 uv。

`requirements.txt`：

```
faster-whisper==1.1.0
nvidia-cublas-cu12==12.8.*
nvidia-cudnn-cu12==9.*
```

（faster-whisper 会自动带 ctranslate2、av、tokenizers、onnxruntime；不安装 torch。）

### 4.3 CLI 设计（argparse，分步子命令 + 一键流水线）

```bash
python -m vidsub extract   <video> [-o data/audio/x.wav]
python -m vidsub transcribe <audio|video> [-o data/srt/x.srt] [--model large-v3|turbo] [--config configs/default.json]
python -m vidsub validate  <srt> [--video <mp4>]
python -m vidsub mux       <video> <srt> [-o out.mp4] [--soft] [--cpu]
python -m vidsub run       <video>          # extract+transcribe，人工校对后需手动 mux
```

人工校对是必须的中断点，因此**不做全自动 extract→mux**，避免未经校对的字幕直接烧录。

配置方式：`configs/default.json` 存放第 1.2 节转写参数、第 2.2 节断句阈值、第 3.2 节样式；CLI flag 覆盖配置文件。

### 4.4 .gitignore（M0 直接写入）

```gitignore
.venv/
__pycache__/
*.pyc
.hf-cache/
data/
*.wav
*.mp4
*.srt
!tests/fixtures/**
.omc/
.DS_Store
```

> `tests/fixtures/` 如需小样本字幕，用白名单保留（仅放极短合成文本，不放媒体）。

### 4.5 git 提交粒度规划

1. `chore: git init, gitignore and project skeleton`（M0）
2. `feat: extract 16k mono wav from video`（M1）
3. `feat: faster-whisper transcription with VAD and word timestamps`（M2 上半）
4. `feat: SRT builder with segmentation and line-wrapping rules`（M2 下半）
5. `feat: SRT validator`（可并入 M2 或单独提交）
6. `test: unit tests for srt builder and validator`
7. `feat: burn-in and soft-subtitle mux with nvenc/libx264`（M4）
8. `docs: README workflow, review conventions and evaluation notes`（M3/M5）

媒体、wav、SRT 产物、模型、venv 一律不入库；M3 评估结论以 `docs/eval_pilot.md` 形式入库。

---

## 5. 分阶段实施步骤（里程碑）

### M0：仓库初始化与骨架（约 30 分钟）

```bash
cd /home/junchuan/video_process
git init
git config user.name  "<your name>"      # 若全局已设可跳过
git config user.email "<your email>"
# 写入 .gitignore、requirements.txt、configs/default.json、src/vidsub/ 空包、tests/
git add . && git commit -m "chore: git init, gitignore and project skeleton"
```

验收：`git log` 有首个提交；`data/`、`.venv/` 被忽略（`git status --ignored` 可见）。

### M1：音频抽取（约 30 分钟）

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -U pip
# 国内网络可加 -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install -r requirements.txt
```

抽音命令（工具内部同一命令）：

```bash
ffmpeg -y -i /home/junchuan/Documents/bilibili/online/CARLA_Cosmos_p2.mp4 \
  -vn -ac 1 -ar 16000 -c:a pcm_s16le \
  /home/junchuan/video_process/data/audio/CARLA_Cosmos_p2.wav
```

验收：wav 为 16kHz/单声道/16-bit PCM；`ffprobe` 时长与源视频差 < 100ms；可正常播放。提交 `feat: extract`。

### M2：GPU 转写并生成 SRT（约 1.5–2 小时，含模型下载）

1. 配置库环境（cuDNN 用 pip 方案，**每个 shell 都要设 LD_LIBRARY_PATH**，写进 README 或封装进 cli 启动逻辑）：

```bash
source .venv/bin/activate
export LD_LIBRARY_PATH=$(python3 -c 'import os,nvidia.cublas.lib,nvidia.cudnn.lib;print(os.path.dirname(nvidia.cublas.lib.__file__)+":"+os.path.dirname(nvidia.cudnn.lib.__file__))')
# 模型下载走镜像（见第 7 节）
export HF_ENDPOINT=https://hf-mirror.com
```

2. 先用前 60 秒做冒烟：`ffmpeg -t 60 -i data/audio/CARLA_Cosmos_p2.wav -c copy /tmp/smoke.wav` → 跑 transcribe，确认 GPU 被调用（`nvidia-smi` 可见进程）、英文术语保留。
3. 全片转写，输出 `data/srt/CARLA_Cosmos_p2.srt`；批处理/逐段各跑一版到临时文件对比断句。
4. `validate` 通过。

验收：SRT 条目数与视频内容相称；抽查 10 条时间轴与音频对得上；`nvidia-smi` 确认走 GPU；全片转写耗时记录（预期 1–3 分钟）。提交 `feat: transcription` 与 `feat: SRT builder`。

### M3：人工校对与准确率评估（约 1.5–2 小时，人工为主）

1. 复制为 `CARLA_Cosmos_p2.reviewed.srt`，对**前 3 分钟（0–180s）逐句精校**并以精校稿作为真值；全片其余部分正常校对。
2. 用 `scripts/cer.py` 对原始稿 vs 精校真值算字错率（CER）、英文术语保留率（见第 6 节）。
3. 对 10 个抽样字幕条，用 `scripts/spotcheck.sh` 抽音频核对起止时间，统计偏移。
4. 结论写入 `docs/eval_pilot.md` 并提交；达标线见第 6 节，不达标按 1.2 节触发备选。

### M4：字幕合成（约 1 小时）

1. 先烧 5 秒小样（`-ss 60 -t 5`）目检字体、字号、描边、位置；
2. 全片 h264_nvenc 烧录；再出一版软字幕；
3. `ffprobe` 核对产物时长/分辨率/音轨；播放核对口型与字幕。

验收：成品时长与源片差 < ±100ms；抽 5 处字幕出现时刻与语音偏差 ≤ 200ms；无花屏/音画不同步。提交 `feat: mux`。

### M5：文档与收尾（约 45 分钟）

- README：环境准备（含 LD_LIBRARY_PATH、HF_ENDPOINT）、四步命令、人工校对约定、故障排查（NVENC 参数报错、字体找不到、模型下载失败）。
- 跑全部单测；`git status` 干净；标签 `v0.1.0`（可选）。

---

## 6. 准确率保障与验证方案

### 6.1 量化指标（试点前 3 分钟，真值为人工逐句精校稿）

| 指标 | 定义 | 达标线 |
|---|---|---|
| 字错率 CER | 字符级编辑距离（插入+删除+替换）/ 真值总字数，`scripts/cer.py` 实现，标点不计入 | **≤ 5%**（目标），> 8% 触发备选方案 |
| 英文术语保留率 | 真值中的英文术语片段，转写稿中同位置仍为英文原文（允许大小写差异）的比例 | **≥ 95%** |
| 时间轴偏移 | 抽样 10 条，字幕起止 vs 语音实际起止 | 起/止偏差均 **≤ 200ms**，无系统性提前/滞后 |
| 幻觉条目 | 无语音处出现字幕、或同句重复输出的条数 | 0（音乐段落允许 warning 但不得出字幕） |

### 6.2 试听/抽片段核对命令（spotcheck.sh 封装）

```bash
# 抽某字幕条对应音频（起止时间精确，先 -ss 后 -i 为快速 seek；精确核对用 -i 在前）
ffmpeg -y -ss 62.40 -to 65.80 -i data/audio/CARLA_Cosmos_p2.wav /tmp/clip.wav
# 带原视频画面的 5 秒小样（核对口型）
ffmpeg -y -ss 60 -t 5 -i <源mp4> -c:v libx264 -c:a aac /tmp/clip.mp4
```

### 6.3 Whisper 幻觉缓解清单（已内置为默认参数）

- Silero VAD（`vad_filter`，`min_silence_duration_ms=500`，`speech_pad_ms=200`）；
- `condition_on_previous_text=False`，避免错误滚动与逐段重复；
- `hallucination_silence_threshold=2.0`（faster-whisper 专用，静音中吐字幕时自动截断）；
- `no_speech_threshold=0.6`、`compression_ratio_threshold=2.4`（压缩率过高通常是重复唠叨）、`log_prob_threshold=-1.0`；
- 温度回退阶梯（0→1.0），困惑度高时自动重解码；
- `language="zh"` 强制锁定，防止英文片段触发语言漂移；
- `beam_size=5`；批处理与逐段结果对照，防止批处理边界切词。

---

## 7. 风险与对策

| 风险 | 影响 | 对策 |
|---|---|---|
| HuggingFace 国内下载失败/慢 | 模型拉不下来，M2 阻塞 | `export HF_ENDPOINT=https://hf-mirror.com`（已核实可用，支持 `huggingface-cli download` 断点续传；gated 模型镜像不支持登录，需官方 token）。备选 ModelScope（`https://modelscope.cn`，FunASR/Paraformer/SenseVoice 官方都在上面）。pip 走清华源 `-i https://pypi.tuna.tsinghua.edu.cn/simple` |
| 本机无系统 cuDNN | CTranslate2 GPU 初始化失败（常见报错 `Library cudnn_ops_infer64_9.dll not found`） | 用 pip 的 `nvidia-cublas-cu12` + `nvidia-cudnn-cu12==9.*`，并按 M2 设置 `LD_LIBRARY_PATH`；把该 export 写入 README。**不要**为省事去 apt 装 cudnn 8（版本不匹配） |
| CUDA 版本错配 | 黑盒报错 | 当前 CTranslate2 要 CUDA12 + cuDNN9（pip 方案自洽，不依赖 toolkit）。若未来换到 CUDA11 环境，须钉 `ctranslate2==3.24.0`；CUDA12+cuDNN 钉 `4.4.0` |
| ffmpeg 4.4 偏老 | 个别 nvenc 新参数名不支持 | 用 `-preset p4 -cq 23`（4.4 支持）；报错即回退 `-preset medium`；烧录与 libass 在 4.4 无问题。不建议为这个项目升级 ffmpeg |
| WhisperX 中文对齐对英文词失效 | 英文字幕无时间戳 | 主方案不用 WhisperX；若启用走"英文词回退 faster-whisper 时间戳"混合策略 |
| 长视频显存/时间 | 本试点 14 分钟无压力；未来长视频 | VAD + 批处理（batch_size 可调到 8）；>1 小时可换 turbo；CTranslate2 内部按 VAD 段流式处理，不会整片载入 |
| 英文术语表 | Whisper 无真正热词机制 | 用 `initial_prompt` 携带术语（如上）；术语表在 `configs/default.json` 维护，按视频主题更换。SenseVoice/Paraformer 路线另查热词接口 |
| 人工误改 SRT 时间轴/编码 | 烧录失败或错位 | `validate` 把关；README 约定只改文本、UTF-8 保存、校对稿更名 |
| vorbis 音轨 | 部分播放器/流程对 mp4+vorbis 兼容差 | 成品统一转 AAC；抽音用 PCM；不保留 vorbis 到新文件 |
| 背景音乐导致唱歌转写 | 技术视频头尾可能有 BGM | VAD + `hallucination_silence_threshold`；M3 重点抽查片头片尾 |

---

## 8. 调研引用（关键 URL）

1. faster-whisper 仓库（安装、GPU 库、VAD、词级时间戳）：https://github.com/SYSTRAN/faster-whisper
2. large-v3 转换权重模型卡：https://huggingface.co/Systran/faster-whisper-large-v3
3. CTranslate2 硬件/ CUDA 要求：https://opennmt.net/CTranslate2/installation.html
4. WhisperX 仓库（VAD、批处理、对齐机制、对齐局限）：https://github.com/m-bain/whisperX
5. WhisperX 中文对齐模型映射（zh → jonatasgrosman/wav2vec2-large-xlsr-53-chinese-zh-cn）：https://github.com/m-bain/whisperX/blob/main/whisperx/alignment.py
6. FunASR 仓库（Paraformer、FSMN-VAD、ct-punc、SenseVoice 集成）：https://github.com/modelscope/FunASR
7. SenseVoice 仓库（多语/混读、词级时间戳、30 秒限制、权重许可证）：https://github.com/FunAudioLLM/SenseVoice
8. OpenAI Whisper（参数与 transcribe/translate 语义）：https://github.com/openai/whisper
9. HF 镜像站（HF_ENDPOINT 用法）：https://hf-mirror.com/
10. ModelScope（国内模型备选源）：https://modelscope.cn
11. ffmpeg subtitles filter 文档：https://ffmpeg.org/ffmpeg-filters.html#subtitles-1
12. ffmpeg h264_nvenc 文档：https://ffmpeg.org/ffmpeg-codecs.html#Video-Encoders
