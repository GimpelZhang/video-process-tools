# CLAUDE.md — 给进入本项目的 AI Agent

## 项目是什么

`vidsub`：本地视频字幕工具，一条命令行流水线：

```
mp4 → ffmpeg 抽 16kHz WAV → faster-whisper(Whisper large-v3) 转写
    → 断句排版生成 SRT →【人工校对，强制中断点】→ ffmpeg 合成回 mp4（硬烧录 / 软字幕）
```

核心需求（不可妥协）：

1. **简体中文准确**，但音频中的英文必须**保留英文原文**（code-switch 转写，不是翻译）
2. 合成后**时间轴与音画对齐**
3. 人工校对环节不可跳过

完整设计依据见 `docs/Plan_v1.md`；试点验证结果见 `docs/eval_pilot.md`（CER 0.23%、英文保留 100%）。

## 环境与日常命令

- Python 3.10 venv 在 `.venv/`；包已做可编辑安装，**直接用 `.venv/bin/vidsub`，不要再加 `PYTHONPATH=src`**
- 国内网络：pip 用清华源，HuggingFace 用 `HF_ENDPOINT=https://hf-mirror.com`
- 模型（2.9GB）在本地 `.hf-cache/large-v3`，不入库、不重复下载

```bash
.venv/bin/pytest -q                          # 跑测试，改动前后都要跑（当前 12 passed）
.venv/bin/vidsub run input.mp4              # 抽音+转写
.venv/bin/vidsub validate x.reviewed.srt --video input.mp4
.venv/bin/vidsub mux input.mp4 x.reviewed.srt            # 硬烧录
.venv/bin/vidsub mux input.mp4 x.reviewed.srt --soft     # 软字幕
.venv/bin/vidsub mux input.mp4 x.srt --sample --ss 60 --duration 5  # 5 秒样式小样
python scripts/cer.py 机器稿.srt 校对稿.srt             # CER + 英文保留率
```

## 代码结构（`src/vidsub/`）

| 文件 | 职责 |
|---|---|
| `cli.py` | argparse 子命令：extract / transcribe / validate / mux / run |
| `audio.py` | ffmpeg 抽音、探测时长 |
| `gpu.py` | 启动时用 ctypes RTLD_GLOBAL 预加载 pip 版 cuBLAS/cuDNN（本机无系统 cuDNN，**勿删此步骤**） |
| `transcribe.py` | faster-whisper 封装；返回 Segment/Word 数据类 |
| `srt.py` | 词级时间戳 → 断句单元 → 打包排版 → SRT；项目逻辑最复杂的模块，改前先通读 |
| `validate.py` | SRT 完整性/单调性/重叠/超长校验 |
| `mux.py` | ffmpeg 命令构造；硬烧录三级回退 nvenc p4 → nvenc medium → libx264 |
| `config.py` | 读 `configs/default.json` |

## 关键设计决策（改代码前必读，勿回退）

- **转写参数**：`language="zh"`、`task="transcribe"`、`condition_on_previous_text=False`、`hallucination_silence_threshold=2.0`、Silero VAD、批处理 `BatchedInferencePipeline`（batch_size=16，质量与速度均优于逐段模式，`--no-batch` 仅用于对比）
- **英文术语靠 `initial_prompt` 术语表引导**（`configs/default.json`）；处理新主题视频时替换术语表，不要改主链路代码
- **不用 WhisperX 对齐**：其中文对齐模型词表不含拉丁字母，中英混读时英文词对齐失败
- **SRT 排版规则**：单条 ≤2 行、每行 ≤20 汉字宽（列宽 40，CJK 字符显示宽度=2）、1.5–7 秒、≤17 汉字/秒；按句末标点 → 分句标点 → ≥300ms 停顿切分，超长无停顿句强切；条目间强制 30–80ms 间隙
- 时间戳毫秒用 `int(seconds*1000 + 0.5)`，**不要用 `round()`**（银行家舍入会出错）
- CJK 相邻半角标点转全角（`srt.py` 的 `normalize_cjk_punct`），切分行后行首标点要移回上行
- **硬烧录**：字幕渲染进画面像素，必可见，视频需重编码；**软字幕**：mov_text 独立字幕轨，视频 `-c copy` 零损失、可开关，但依赖播放器支持。两者同源一份校对 SRT
- 源片音轨可能是 Vorbis，硬烧录时顺带转 AAC 192k

## 约定

- **代码与注释用中文**（docstring、commit message、面向用户的输出），风格与现有模块保持一致
- 配置走 `configs/default.json`，不要硬编码参数
- `data/`、`.hf-cache/`、`.venv/`、`*.egg-info/` 均不入库；测试用 SRT 放 `tests/fixtures/`（.gitignore 中唯一的 `*.srt` 例外）
- **机密**：`user_access_methods.txt` 含 token/密码，已被 .gitignore 排除，**永不读取入库内容提交、永不 echo 到输出、永不写入 git config**；需要 push 凭据时构造 `/tmp` 临时 askpass 脚本，用完即删
- Git commit author 必须是 **GimpelZhang**（仓库已配置）；commit message 结尾保留 `Co-Authored-By: Claude Code <noreply@anthropic.com>`
- 远程仓库：`github.com/GimpelZhang/video-process-tools`，main 分支；已打 tag v0.1.0

## 完成一项改动的标准

1. 相关测试通过（新功能要补测试，不允许 stub/skip 占位）
2. 若动了断句/排版逻辑：用真实素材转写并人工核对，必要时 `--sample` 烧录目检
3. 若动了转写参数或术语表：跑 `scripts/cer.py` 确认 CER 与英文保留率不退化
4. 汇报时给证据（测试输出、命令结果），不要只说"应该可以"
