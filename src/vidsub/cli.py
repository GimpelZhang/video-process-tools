"""vidsub 命令行入口。

子命令：
  extract    从视频抽取 16kHz 单声道 WAV
  transcribe GPU 语音转写生成 SRT
  validate   校验 SRT
  mux        字幕合成回视频（M4）
  run        extract + transcribe（人工校对后再手动 mux）
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from . import __version__
from .audio import FFmpegError, extract_wav, probe_duration
from .config import load_config, merge_overrides
from .srt import segments_to_srt
from .validate import validate_file


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _default_out(video: str, kind: str, ext: str) -> Path:
    stem = Path(video).stem
    return _project_root() / "data" / kind / f"{stem}.{ext}"


# ---------------- extract ----------------

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


# ---------------- transcribe ----------------

def _ensure_wav(source: str) -> tuple[Path, bool]:
    """输入是视频则先抽 WAV。返回 (wav路径, 是否临时生成)。"""
    p = Path(source)
    if p.suffix.lower() in (".wav", ".flac", ".mp3", ".m4a", ".aac", ".ogg"):
        return p, False
    out = _default_out(source, "audio", "wav")
    extract_wav(source, out)
    return out, False


def cmd_transcribe(args: argparse.Namespace) -> int:
    # 延迟导入，避免 extract-only 场景强依赖 CUDA
    from .transcribe import transcribe_audio

    cfg = load_config(args.config)
    overrides = {"model": args.model}
    cfg = merge_overrides(cfg, "transcribe", overrides)

    try:
        wav, _ = _ensure_wav(args.source)
    except (FFmpegError, FileNotFoundError) as exc:
        print(f"[transcribe] 失败：{exc}", file=sys.stderr)
        return 1

    batch_size = None if args.no_batch else (args.batch_size or cfg["transcribe"].get("batch_size"))
    print(f"[transcribe] 模型 {cfg['transcribe']['model']} | 输入 {wav} | "
          f"{'逐段模式' if batch_size is None else f'批处理 batch_size={batch_size}'}")
    try:
        result = transcribe_audio(wav, cfg["transcribe"], batch_size=batch_size)
    except Exception as exc:
        print(f"[transcribe] 转写失败：{exc}", file=sys.stderr)
        return 1

    print(f"[transcribe] 语言 {result.language}（p={result.language_probability:.2f}），"
          f"音频 {result.duration:.1f}s，耗时 {result.elapsed:.1f}s "
          f"（{result.duration / max(result.elapsed, 0.01):.1f}x 实时），"
          f"{len(result.segments)} 段")

    out = Path(args.output) if args.output else _default_out(str(wav), "srt", "srt")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(segments_to_srt(result.segments, cfg["segmentation"]), encoding="utf-8")
    print(f"[transcribe] SRT 已写入 {out}")

    findings = validate_file(out)
    errors = [f for f in findings if f.level == "error"]
    for f in findings:
        print(f"[validate:{f.level}] {f.message}")
    if errors:
        return 2
    return 0


# ---------------- validate ----------------

def cmd_validate(args: argparse.Namespace) -> int:
    try:
        findings = validate_file(args.srt, video=args.video)
    except (FileNotFoundError, ValueError) as exc:
        print(f"[validate] 失败：{exc}", file=sys.stderr)
        return 1
    for f in findings:
        print(f"[{f.level}] {f.message}")
    errors = [f for f in findings if f.level == "error"]
    print(f"[validate] {len(findings)} 项问题，{len(errors)} 个 error")
    return 1 if errors else 0


# ---------------- mux ----------------

def cmd_mux(args: argparse.Namespace) -> int:
    from .mux import MuxError, burn_in, soft_mux

    cfg = load_config(args.config)
    if getattr(args, "cpu", False):
        cfg["mux"]["video_encoder"] = "libx264"
    video = Path(args.video)
    srt = Path(args.srt)
    if args.output:
        out = Path(args.output)
    else:
        suffix = "soft" if args.soft else "subbed"
        out = _default_out(str(video), "output", "mp4").with_name(
            f"{video.stem}.{suffix}.mp4"
        )

    try:
        if args.soft:
            result = soft_mux(video, srt, out)
        else:
            ss = args.ss if args.sample else None
            duration = args.duration if args.sample else None
            result = burn_in(video, srt, out, cfg, cfg["style"], ss=ss, duration=duration)
    except (MuxError, FileNotFoundError) as exc:
        print(f"[mux] 失败：{exc}", file=sys.stderr)
        return 1

    print(f"[mux] 已写出 {result}")
    try:
        src_dur = probe_duration(video)
        out_dur = probe_duration(result)
        delta = abs(src_dur - out_dur)
        print(f"[mux] 源时长 {src_dur:.3f}s，产物 {out_dur:.3f}s，差 {delta * 1000:.0f}ms")
        if not args.sample and delta > 0.1:
            print("[mux] 警告：成片时长差超过 100ms", file=sys.stderr)
            return 2
    except FFmpegError:
        pass
    return 0


# ---------------- run ----------------

def cmd_run(args: argparse.Namespace) -> int:
    nested = argparse.Namespace(
        source=args.video, output=None, model=None, config=args.config,
        no_batch=False, batch_size=None,
    )
    rc = cmd_transcribe(nested)
    if rc != 0:
        return rc
    srt = _default_out(args.video, "srt", "srt")
    print("\n[run] 请人工校对：")
    reviewed = srt.with_name(srt.name.replace(".srt", ".reviewed.srt"))
    print(f"  cp {srt} {reviewed}")
    print("  # 校对正文（勿改时间轴），随后执行：")
    print(f"  python -m vidsub validate {reviewed}")
    print(f"  python -m vidsub mux {args.video} {reviewed}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="vidsub", description="本地视频字幕提取与合成工具")
    parser.add_argument("--version", action="version", version=f"vidsub {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    p_extract = sub.add_parser("extract", help="抽取 16kHz 单声道 WAV")
    p_extract.add_argument("video")
    p_extract.add_argument("-o", "--output")
    p_extract.set_defaults(func=cmd_extract)

    p_tr = sub.add_parser("transcribe", help="GPU 转写生成 SRT")
    p_tr.add_argument("source", help="音频或视频文件")
    p_tr.add_argument("-o", "--output")
    p_tr.add_argument("--model", help="large-v3 / turbo 或 HF 模型路径")
    p_tr.add_argument("--config")
    p_tr.add_argument("--no-batch", action="store_true", help="使用逐段模式而非批处理")
    p_tr.add_argument("--batch-size", type=int)
    p_tr.set_defaults(func=cmd_transcribe)

    p_val = sub.add_parser("validate", help="校验 SRT")
    p_val.add_argument("srt")
    p_val.add_argument("--video", help="同时校验末条不超过视频时长")
    p_val.set_defaults(func=cmd_validate)

    p_mux = sub.add_parser("mux", help="将 SRT 合成回视频")
    p_mux.add_argument("video")
    p_mux.add_argument("srt")
    p_mux.add_argument("-o", "--output")
    p_mux.add_argument("--soft", action="store_true", help="软字幕（mov_text，不重编码视频）")
    p_mux.add_argument("--cpu", action="store_true", help="强制 libx264 CPU 编码")
    p_mux.add_argument("--sample", action="store_true", help="只渲染 5 秒小样（用于样式目检）")
    p_mux.add_argument("--ss", type=float, default=60.0, help="小样起始秒，默认 60")
    p_mux.add_argument("--duration", type=float, default=5.0, help="小样时长，默认 5")
    p_mux.add_argument("--config")
    p_mux.set_defaults(func=cmd_mux)

    p_run = sub.add_parser("run", help="extract + transcribe，随后人工校对")
    p_run.add_argument("video")
    p_run.add_argument("--config")
    p_run.set_defaults(func=cmd_run)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
