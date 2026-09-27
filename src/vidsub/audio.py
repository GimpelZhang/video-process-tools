"""音频抽取：用 ffmpeg 把任意视频/音频转为 16kHz 单声道 16-bit PCM WAV。"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path


class FFmpegError(RuntimeError):
    """ffmpeg/ffprobe 执行失败。"""


def ensure_ffmpeg() -> None:
    if shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None:
        raise FFmpegError("未找到 ffmpeg/ffprobe，请先安装：sudo apt install ffmpeg")


def probe_duration(path: str | Path) -> float:
    """返回媒体时长（秒）。"""
    ensure_ffmpeg()
    proc = subprocess.run(
        [
            "ffprobe", "-v", "error",
            "-show_entries", "format=duration",
            "-of", "json",
            str(path),
        ],
        capture_output=True, text=True, check=False,
    )
    if proc.returncode != 0:
        raise FFmpegError(f"ffprobe 失败：{proc.stderr.strip()}")
    data = json.loads(proc.stdout)
    try:
        return float(data["format"]["duration"])
    except (KeyError, ValueError) as exc:
        raise FFmpegError(f"无法解析时长：{data}") from exc


def extract_wav(
    video: str | Path,
    out_wav: str | Path,
    sample_rate: int = 16000,
    mono: bool = True,
) -> Path:
    """抽取音频为 PCM s16le WAV，默认 16kHz 单声道（Whisper 要求）。"""
    ensure_ffmpeg()
    video = Path(video)
    out_wav = Path(out_wav)
    if not video.exists():
        raise FileNotFoundError(f"输入文件不存在：{video}")
    out_wav.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        "ffmpeg", "-y",
        "-i", str(video),
        "-vn",
        "-ac", "1" if mono else "2",
        "-ar", str(sample_rate),
        "-c:a", "pcm_s16le",
        str(out_wav),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if proc.returncode != 0:
        raise FFmpegError(f"ffmpeg 抽音失败：{proc.stderr.strip()[-2000:]}")
    if not out_wav.exists() or out_wav.stat().st_size == 0:
        raise FFmpegError("抽音完成但输出文件为空")
    return out_wav
