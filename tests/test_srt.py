"""SRT 时间戳、解析/生成与断句排版测试。"""

from __future__ import annotations

from vidsub.srt import (
    build_cues,
    display_width,
    format_timestamp,
    parse_srt,
    parse_timestamp,
    render_srt,
)
from vidsub.transcribe import Segment, Word

SCFG = {
    "max_lines": 2,
    "hanzi_per_line": 20,
    "min_duration_s": 1.5,
    "max_duration_s": 7.0,
    "max_chars_per_second": 17,
    "gap_ms": 60,
    "min_gap_ms": 30,
}


def words_from_text(text: str, start: float = 0.0, per: float = 0.1) -> list[Word]:
    words: list[Word] = []
    t = start
    for ch in text:
        if ch == " ":
            words[-1].text += ch
            continue
        words.append(Word(t, t + per, ch))
        t += per
    return words


def seg(text: str, start: float) -> Segment:
    ws = words_from_text(text, start)
    return Segment(start, ws[-1].end, text, ws)


def test_timestamps():
    assert format_timestamp(0) == "00:00:00,000"
    assert format_timestamp(3661.2345) == "01:01:01,235"
    assert parse_timestamp("01:01:01,235") == 3661.235


def test_display_width():
    assert display_width("中") == 2
    assert display_width("ab") == 2
    assert display_width("中文ab") == 6


def test_parse_render_roundtrip():
    srt = (
        "1\n00:00:00,000 --> 00:00:02,000\n你好世界\n\n"
        "2\n00:00:02,000 --> 00:00:04,000\n第二行\n"
    )
    cues = parse_srt(srt)
    assert len(cues) == 2
    assert cues[0].text == "你好世界"
    out = render_srt(cues)
    assert "00:00:00,000 --> 00:00:02,000" in out


def test_build_cues_basic():
    segments = [seg("今天我们来介绍自动驾驶技术。", 0.0), seg("这里会用到GPU加速。", 2.0)]
    cues = build_cues(segments, SCFG)
    assert len(cues) >= 1
    assert cues[0].start >= 0
    assert cues[-1].end <= 10
    # 时间单调、不重叠、有序号
    for i, cue in enumerate(cues, 1):
        assert cue.index == i
        assert cue.end > cue.start
        assert len(cue.lines) <= 2
    for a, b in zip(cues, cues[1:]):
        assert b.start >= a.end - 0.001


def test_build_cues_gaps():
    # 两个紧挨的句子，条目间应留出至少 30ms 间隙
    s1 = seg("第一句话内容。", 0.0)
    s2_start = s1.end
    s2 = seg("第二句话内容。", s2_start)
    cues = build_cues([s1, s2], SCFG)
    if len(cues) == 2:
        assert cues[1].start - cues[0].end >= 0.029


def test_long_segment_split():
    text = "这是一个非常非常非常非常非常非常非常非常非常非常长的句子内容需要被切分。"
    cues = build_cues([seg(text, 0.0)], SCFG)
    assert len(cues) >= 1
    for cue in cues:
        assert display_width(cue.text) <= 80
