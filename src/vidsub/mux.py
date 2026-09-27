"""字幕合成：ffmpeg 硬烧录（默认 h264_nvenc，回退 libx264）与软字幕（mov_text）。"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from typing import Any


class MuxError(RuntimeError):
    pass


def escape_filter_path(path: str | Path) -> str:
    """subtitles filter 文件名转义：\\ : ' 需转义。"""
    p = str(path).replace("\\", "\\\\").replace(":", "\\:").replace("'", "\\'")
    return p


def force_style_string(style: dict[str, Any]) -> str:
    order = [
        "font_name", "font_size", "primary_colour", "secondary_colour",
        "outline_colour", "back_colour", "border_style", "outline", "shadow",
        "margin_l", "margin_r", "margin_v",
    ]
    keymap = {
        "font_name": "FontName",
        "font_size": "FontSize",
        "primary_colour": "PrimaryColour",
        "secondary_colour": "SecondaryColour",
        "outline_colour": "OutlineColour",
        "back_colour": "BackColour",
        "border_style": "BorderStyle",
        "outline": "Outline",
        "shadow": "Shadow",
        "margin_l": "MarginL",
        "margin_r": "MarginR",
        "margin_v": "MarginV",
    }
    parts = [f"{keymap[k]}={style[k]}" for k in order if k in style]
    return ",".join(parts)


def build_burn_command(
    video: str | Path,
    srt: str | Path,
    out: str | Path,
    cfg: dict[str, Any],
    style: dict[str, Any],
    video_encoder: str | None = None,
    nvenc_preset: str | None = None,
    audio_extra_in: str | None = None,
    ss: float | None = None,
    duration: float | None = None,
) -> list[str]:
    mcfg = cfg["mux"]
    encoder = video_encoder or mcfg.get("video_encoder", "h264_nvenc")
    vf = f"subtitles={escape_filter_path(srt)}:force_style='{force_style_string(style)}'"
    cmd = ["ffmpeg", "-y"]
    if ss is not None:
        cmd += ["-ss", f"{ss:.3f}"]
    if duration is not None:
        cmd += ["-t", f"{duration:.3f}"]
    cmd += ["-i", str(video), "-vf", vf]
    if encoder == "h264_nvenc":
        cmd += ["-c:v", "h264_nvenc", "-preset", nvenc_preset or mcfg.get("nvenc_preset", "p4"),
                "-rc", "vbr", "-cq", str(mcfg.get("cq", 23)), "-b:v", "0"]
    elif encoder == "hevc_nvenc":
        cmd += ["-c:v", "hevc_nvenc", "-preset", nvenc_preset or mcfg.get("nvenc_preset", "p4"),
                "-rc", "vbr", "-cq", str(mcfg.get("cq", 23)), "-b:v", "0"]
    else:
        cmd += ["-c:v", "libx264", "-preset", mcfg.get("cpu_preset", "medium"),
                "-crf", str(mcfg.get("crf", 20))]
    cmd += ["-c:a", mcfg.get("audio_codec", "aac"), "-b:a", mcfg.get("audio_bitrate", "192k")]
    cmd.append(str(out))
    return cmd


def build_soft_command(
    video: str | Path, srt: str | Path, out: str | Path
) -> list[str]:
    return [
        "ffmpeg", "-y", "-i", str(video), "-i", str(srt),
        "-map", "0:v", "-map", "0:a", "-map", "1:0",
        "-c", "copy", "-c:s", "mov_text",
        "-metadata:s:s:0", "language=chi",
        str(out),
    ]


def _run(cmd: list[str]) -> tuple[int, str]:
    if shutil.which("ffmpeg") is None:
        raise MuxError("未找到 ffmpeg")
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    tail = (proc.stderr or "")[-3000:]
    return proc.returncode, tail


def burn_in(
    video: str | Path,
    srt: str | Path,
    out: str | Path,
    cfg: dict[str, Any],
    style: dict[str, Any],
    ss: float | None = None,
    duration: float | None = None,
) -> Path:
    video, srt, out = Path(video), Path(srt), Path(out)
    for p in (video, srt):
        if not p.exists():
            raise FileNotFoundError(p)
    out.parent.mkdir(parents=True, exist_ok=True)

    attempts: list[tuple[str, str | None]] = [
        ("h264_nvenc", None),      # -preset p4 -cq
        ("h264_nvenc", "medium"),  # 旧预设名回退
        ("libx264", None),         # CPU 最终回退
    ]
    last_err = ""
    for encoder, preset in attempts:
        cmd = build_burn_command(video, srt, out, cfg, style,
                                 video_encoder=encoder, nvenc_preset=preset,
                                 ss=ss, duration=duration)
        rc, stderr = _run(cmd)
        if rc == 0 and out.exists() and out.stat().st_size > 0:
            return out
        last_err = stderr
        # nvenc 不支持类错误才继续回退；其余错误也继续尝试，最后统一报错
    raise MuxError(f"所有编码方案均失败：\n{last_err}")


def soft_mux(video: str | Path, srt: str | Path, out: str | Path) -> Path:
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    cmd = build_soft_command(video, srt, out)
    rc, stderr = _run(cmd)
    if rc != 0 or not out.exists():
        raise MuxError(f"软字幕合成失败：{stderr}")
    return out
