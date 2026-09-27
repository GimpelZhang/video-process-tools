"""vidsub 命令行入口。

子命令：
  extract    从视频抽取 16kHz 单声道 WAV
  transcribe GPU 语音转写生成 SRT（M2）
  validate   校验 SRT（M2）
  mux        字幕合成回视频（M4）
  run        extract + transcribe（人工校对后再手动 mux）
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from . import __version__
from .audio import FFmpegError, extract_wav, probe_duration


def _default_out(video: str, kind: str, ext: str) -> Path:
    stem = Path(video).stem
    root = Path(__file__).resolve().parents[2] / "data"
    return root / kind / f"{stem}.{ext}"


def cmd_extract(args: argparse.Namespace) -> int:
    out = Path(args.output) if args.output else _default_out(args.video, "audio", "wav")
    try:
        wav = extract_wav(args.video, out)
        src_dur = probe_duration(args.video)
        wav_dur = probe_duration(wav)
    except (FFmpegError, FileNotFoundError) as exc:
        print(f"[extract] 失败：{exc}", file=sys.stderr)
        return 1
    delta = abs(src_dur - wav_dur)
    print(f"[extract] {wav}")
    print(f"[extract] 源时长 {src_dur:.3f}s，WAV 时长 {wav_dur:.3f}s，差 {delta * 1000:.0f}ms")
    if delta > 0.1:
        print("[extract] 警告：时长差超过 100ms", file=sys.stderr)
        return 2
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="vidsub", description="本地视频字幕提取与合成工具"
    )
    parser.add_argument("--version", action="version", version=f"vidsub {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    p_extract = sub.add_parser("extract", help="抽取 16kHz 单声道 WAV")
    p_extract.add_argument("video", help="输入视频文件")
    p_extract.add_argument("-o", "--output", help="输出 WAV 路径")
    p_extract.set_defaults(func=cmd_extract)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
