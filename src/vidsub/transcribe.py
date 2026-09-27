"""faster-whisper 封装：GPU 转写，输出带词级时间戳的 Segment 列表。"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .gpu import preload_cuda_libs


@dataclass
class Word:
    start: float
    end: float
    text: str
    probability: float | None = None


@dataclass
class Segment:
    start: float
    end: float
    text: str
    words: list[Word] = field(default_factory=list)


@dataclass
class TranscribeResult:
    language: str
    language_probability: float
    duration: float
    elapsed: float
    segments: list[Segment]


def _fill_missing_times(seg: Any, words: list[Word]) -> list[Word]:
    """极少数词缺时间戳时，用段边界/相邻词插值补齐。"""
    n = len(words)
    for i, w in enumerate(words):
        if w.start is not None and w.end is not None:
            continue
        prev_end = words[i - 1].end if i > 0 and words[i - 1].end is not None else seg.start
        next_start = (
            words[i + 1].start if i + 1 < n and words[i + 1].start is not None else seg.end
        )
        w.start = prev_end
        w.end = max(next_start, prev_end + 0.05)
    return words


def transcribe_audio(
    audio_path: str | Path,
    tcfg: dict[str, Any],
    batch_size: int | None = None,
) -> TranscribeResult:
    """执行转写。batch_size 非 None 时走 BatchedInferencePipeline。"""
    preload_cuda_libs()
    # 仅在预加载后引入
    from faster_whisper import WhisperModel

    audio_path = Path(audio_path)
    if not audio_path.exists():
        raise FileNotFoundError(f"音频不存在：{audio_path}")

    model = WhisperModel(
        tcfg.get("model", "large-v3"),
        device=tcfg.get("device", "cuda"),
        compute_type=tcfg.get("compute_type", "float16"),
    )

    kwargs: dict[str, Any] = {
        "language": tcfg.get("language", "zh"),
        "task": tcfg.get("task", "transcribe"),
        "beam_size": tcfg.get("beam_size", 5),
        "best_of": tcfg.get("best_of", 5),
        "word_timestamps": tcfg.get("word_timestamps", True),
        "condition_on_previous_text": tcfg.get("condition_on_previous_text", False),
        "hallucination_silence_threshold": tcfg.get("hallucination_silence_threshold"),
        "no_speech_threshold": tcfg.get("no_speech_threshold", 0.6),
        "compression_ratio_threshold": tcfg.get("compression_ratio_threshold", 2.4),
        "log_prob_threshold": tcfg.get("log_prob_threshold", -1.0),
        "temperature": tcfg.get("temperatures", [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]),
        "initial_prompt": tcfg.get("initial_prompt") or None,
        "vad_filter": tcfg.get("vad_filter", True),
        "vad_parameters": {
            "min_silence_duration_ms": tcfg.get("vad_min_silence_duration_ms", 500),
            "speech_pad_ms": tcfg.get("vad_speech_pad_ms", 200),
        },
    }

    started = time.time()
    if batch_size is not None and batch_size > 1:
        from faster_whisper import BatchedInferencePipeline

        pipe = BatchedInferencePipeline(model=model)
        seg_iter, info = pipe.transcribe(**kwargs, batch_size=batch_size)
    else:
        seg_iter, info = model.transcribe(**kwargs)

    segments: list[Segment] = []
    for seg in seg_iter:
        words = [
            Word(
                start=getattr(w, "start", None),
                end=getattr(w, "end", None),
                text=w.word,
                probability=getattr(w, "probability", None),
            )
            for w in (seg.words or [])
        ]
        # 用轻量 shim 供插值使用
        class _SegShim:
            start = seg.start
            end = seg.end

        words = _fill_missing_times(_SegShim, words)
        segments.append(Segment(seg.start, seg.end, seg.text.strip(), words))

    elapsed = time.time() - started
    return TranscribeResult(
        language=info.language,
        language_probability=getattr(info, "language_probability", 0.0),
        duration=getattr(info, "duration", 0.0),
        elapsed=elapsed,
        segments=segments,
    )
