"""SRT 解析、词级时间戳 -> 断句排版 -> SRT 生成。

排版规则（见 Plan 2.2）：
- 单条最多 2 行，每行 ≤ 20 汉字宽（显示列）
- 单条 1.5–7 秒，阅读速度 ≤ 17 汉字/秒
- 切分优先级：句末标点 > 分句标点 > 词组边界
- 条目间留 30–80ms 间隙，不重叠，时间单调递增
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path

from .transcribe import Segment, Word

# ---------------- 基础工具 ----------------

_SENTENCE_END = "。！？!?…"
_CLAUSE_END = "，、；,;"
_PUNCT = _SENTENCE_END + _CLAUSE_END


def display_width(text: str) -> int:
    """显示列宽：CJK 全角=2，其余=1。"""
    width = 0
    for ch in text:
        if unicodedata.east_asian_width(ch) in ("W", "F"):
            width += 2
        else:
            width += 1
    return width


def format_timestamp(seconds: float) -> str:
    if seconds < 0:
        seconds = 0.0
    total_ms = int(seconds * 1000 + 0.5)
    ms = total_ms % 1000
    total_s = total_ms // 1000
    s = total_s % 60
    m = (total_s // 60) % 60
    h = total_s // 3600
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def parse_timestamp(text: str) -> float:
    m = re.match(r"^(\d{2}):(\d{2}):(\d{2})[,.](\d{3})$", text.strip())
    if not m:
        raise ValueError(f"非法时间戳：{text!r}")
    h, mi, s, ms = (int(x) for x in m.groups())
    return h * 3600 + mi * 60 + s + ms / 1000.0


@dataclass
class Cue:
    index: int
    start: float
    end: float
    lines: list[str] = field(default_factory=list)

    @property
    def text(self) -> str:
        return "\n".join(self.lines)


# ---------------- SRT 解析 ----------------

_TIME_LINE = re.compile(
    r"^(\d{2}:\d{2}:\d{2}[,.]\d{3})\s*-->\s*(\d{2}:\d{2}:\d{2}[,.]\d{3})"
)


def parse_srt(content: str) -> list[Cue]:
    """解析 SRT 文本为 Cue 列表（序号保持文件中的值）。"""
    content = content.lstrip("﻿").replace("\r\n", "\n").replace("\r", "\n")
    blocks = re.split(r"\n[ \t]*\n", content.strip())
    cues: list[Cue] = []
    for block in blocks:
        lines = block.split("\n")
        if not any(ln.strip() for ln in lines):
            continue
        declared_index = len(cues) + 1
        idx_line = 0
        if not _TIME_LINE.match(lines[0].strip()):
            declared_index = int(lines[0].strip())
            idx_line = 1
        time_match = _TIME_LINE.match(lines[idx_line].strip())
        if not time_match:
            raise ValueError(f"无法解析 SRT 块：\n{block}")
        start = parse_timestamp(time_match.group(1))
        end = parse_timestamp(time_match.group(2))
        cues.append(Cue(declared_index, start, end, lines[idx_line + 1:]))
    return cues


def parse_srt_file(path: str | Path) -> list[Cue]:
    raw = Path(path).read_text(encoding="utf-8-sig")
    return parse_srt(raw)


def render_srt(cues: list[Cue]) -> str:
    blocks: list[str] = []
    for i, cue in enumerate(cues, start=1):
        body = "\n".join(line.strip() for line in cue.lines)
        blocks.append(
            f"{i}\n{format_timestamp(cue.start)} --> {format_timestamp(cue.end)}\n{body}"
        )
    return "\n\n".join(blocks) + "\n"


# ---------------- 断句单元 ----------------

@dataclass
class Unit:
    words: list[Word]
    text: str
    start: float
    end: float


def _build_units(segments: list[Segment]) -> list[Unit]:
    """按标点把词流切成最小分句单元，标点跟随前一单元。"""
    units: list[Unit] = []
    cur: list[Word] = []

    def flush() -> None:
        nonlocal cur
        if not cur:
            return
        text = "".join(w.text for w in cur).strip()
        if text:
            units.append(Unit(cur, text, cur[0].start, cur[-1].end))
        cur = []

    for seg in segments:
        words = seg.words
        if not words:
            continue
        for w in words:
            cur.append(w)
            if w.text and w.text[-1] in _PUNCT:
                flush()
        # 段结束但无标点收尾：也断开（faster-whisper 段通常以标点结尾）
        if cur:
            flush()
    return units


# ---------------- 行折叠 ----------------

def _wrap_lines(text: str, max_lines: int, col_per_line: int) -> list[str]:
    """把文本折成 ≤ max_lines 行。优先在 CJK 边界/空格/标点处断行。"""
    max_col = max_lines * col_per_line
    text = text.strip()
    if display_width(text) <= col_per_line or max_lines <= 1:
        return [text] if display_width(text) <= max_col else _hard_wrap(text, max_lines, col_per_line)

    lines: list[str] = []
    rest = text
    while rest and len(lines) < max_lines:
        limit = col_per_line if len(lines) < max_lines - 1 else max_col - sum(
            display_width(x) for x in lines
        )
        cut = _find_wrap_point(rest, limit)
        piece, rest = rest[:cut].strip(), rest[cut:].strip()
        if piece:
            lines.append(piece)
        if not rest:
            break
    if rest:
        # 理论上不会到这里（单元打包已限宽），兜底硬切
        lines[-1] = (lines[-1] + rest)[: col_per_line * 2]
    return lines


def _find_wrap_point(text: str, limit_cols: int) -> int:
    width = 0
    best = 0
    i = 0
    while i < len(text):
        ch = text[i]
        w = 2 if unicodedata.east_asian_width(ch) in ("W", "F") else 1
        if width + w > limit_cols:
            break
        width += w
        i += 1
        # 可断点：空格、CJK 字符之后、标点之后
        if ch == " ":
            best = i
        elif unicodedata.east_asian_width(ch) in ("W", "F") or ch in _PUNCT:
            best = i
    return best if best > 0 else i


def _hard_wrap(text: str, max_lines: int, col_per_line: int) -> list[str]:
    lines: list[str] = []
    width = 0
    cur: list[str] = []
    for ch in text:
        w = 2 if unicodedata.east_asian_width(ch) in ("W", "F") else 1
        if width + w > col_per_line and len(lines) < max_lines - 1:
            lines.append("".join(cur).strip())
            cur, width = [], 0
        cur.append(ch)
        width += w
    if cur:
        lines.append("".join(cur).strip())
    return [ln for ln in lines if ln]


# ---------------- 条目打包 ----------------

def build_cues(
    segments: list[Segment],
    scfg: dict,
) -> list[Cue]:
    max_lines = int(scfg.get("max_lines", 2))
    col_per_line = int(scfg.get("hanzi_per_line", 20)) * 2
    max_col = max_lines * col_per_line
    min_dur = float(scfg.get("min_duration_s", 1.5))
    max_dur = float(scfg.get("max_duration_s", 7.0))
    max_cps = float(scfg.get("max_chars_per_second", 17))
    min_gap = float(scfg.get("min_gap_ms", 30)) / 1000.0

    units = _build_units(segments)
    if not units:
        return []

    groups: list[list[Unit]] = []
    cur: list[Unit] = []

    def fits(group: list[Unit], nxt: Unit) -> bool:
        start = group[0].start
        end = max(nxt.end, group[-1].end)
        dur = max(end - start, 0.1)
        text = "".join(u.text for u in group) + nxt.text
        cols = display_width(text)
        cps = (cols / 2) / dur
        if cols > max_col or end - start > max_dur or cps > max_cps:
            return False
        return True

    for unit in units:
        if cur and not fits(cur, unit):
            groups.append(cur)
            cur = []
        cur.append(unit)
    if cur:
        groups.append(cur)

    # 过短条目尝试并入前一条
    merged: list[list[Unit]] = []
    for group in groups:
        if (
            merged
            and group[-1].end - group[0].start < min_dur
            and _group_fits(merged[-1] + group, max_col, max_dur, max_cps)
        ):
            merged[-1].extend(group)
        else:
            merged.append(group)

    cues: list[Cue] = []
    for group in merged:
        text = "".join(u.text for u in group).strip()
        lines = _wrap_lines(text, max_lines, col_per_line)
        cues.append(Cue(0, group[0].start, group[-1].end, lines))

    _enforce_gaps(cues, min_gap)
    for i, cue in enumerate(cues, start=1):
        cue.index = i
    return cues


def _group_fits(group: list[Unit], max_col: int, max_dur: float, max_cps: float) -> bool:
    start = group[0].start
    end = group[-1].end
    dur = max(end - start, 0.1)
    cols = display_width("".join(u.text for u in group))
    return cols <= max_col and dur <= max_dur and (cols / 2) / dur <= max_cps


def _enforce_gaps(cues: list[Cue], min_gap: float) -> None:
    for i in range(len(cues) - 1):
        cur, nxt = cues[i], cues[i + 1]
        if nxt.start <= cur.end or nxt.start - cur.end < min_gap:
            new_end = nxt.start - min_gap
            if new_end <= cur.start:
                new_end = cur.start + max((nxt.start - cur.start) / 2, 0.01)
            cur.end = new_end
        if cur.end <= cur.start:
            cur.end = cur.start + 0.05


def segments_to_srt(segments: list[Segment], scfg: dict) -> str:
    return render_srt(build_cues(segments, scfg))
