"""SRT 校验器测试。"""

from __future__ import annotations

from vidsub.srt import Cue
from vidsub.validate import validate_cues


def cue(i: int, start: float, end: float, text: str = "内容") -> Cue:
    return Cue(i, start, end, [text])


def test_clean():
    findings = validate_cues([cue(1, 0, 2), cue(2, 2, 4)])
    assert [f for f in findings if f.level == "error"] == []


def test_overlap_detected():
    findings = validate_cues([cue(1, 0, 3), cue(2, 2, 4)])
    assert any("重叠" in f.message for f in findings if f.level == "error")


def test_bad_time():
    findings = validate_cues([cue(1, 3, 2)])
    assert any("起始不小于结束" in f.message for f in findings)


def test_empty_text():
    findings = validate_cues([Cue(1, 0, 2, [""])])
    assert any("正文为空" in f.message for f in findings)


def test_non_monotonic():
    findings = validate_cues([cue(1, 0, 5), cue(2, 4, 6)])
    assert any("单调" in f.message or "重叠" in f.message for f in findings)


def test_empty_list():
    findings = validate_cues([])
    assert findings and findings[0].level == "error"
