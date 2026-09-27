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


def _build_units(segments: list[Segment], pause_split_s: float = 0.3) -> list[Unit]:
    """按标点把词流切成最小分句单元，标点跟随前一单元。

    另外，相邻词停顿 ≥ pause_split_s 时也断开（Plan 2.2 规则3），
    避免无标点长句无法切分。
    """
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
        for i, w in enumerate(words):
            if cur and pause_split_s > 0:
                gap = w.start - cur[-1].end
                if gap >= pause_split_s:
                    flush()
            cur.append(w)
            if w.text and w.text[-1] in _PUNCT:
                flush()
        # 段结束但无标点收尾：也断开（faster-whisper 段通常以标点结尾）
        if cur:
            flush()
    return units


# ---------------- 行折叠 ----------------

def _wrap_lines(text: str, max_lines: int, col_per_line: int) -> list[str]:
    """把文本折成 ≤ max_lines 行。

    在可断点（空格、CJK 字符/标点之后）中选行宽最均衡的切法，
    避免行末把词劈开只剩一个字。
    """
    max_col = max_lines * col_per_line
    text = text.strip()
    width = display_width(text)
    if width <= col_per_line:
        return [text]
    if max_lines <= 1 or width > max_col:
        return _hard_wrap(text, max_lines, col_per_line)

    lines: list[str] = []
    rest = text
    remaining_slots = max_lines
    while rest and remaining_slots > 0:
        w = display_width(rest)
        if w <= col_per_line or remaining_slots == 1:
            lines.append(rest)
            break
        # 需要的行数与理想切分位置
        lines_needed = max(2, -(-w // col_per_line))  # ceil
        if lines_needed > remaining_slots:
            lines_needed = remaining_slots
        target = w / lines_needed
        cut = _find_wrap_point(rest, col_per_line, target)
        piece, rest = rest[:cut].rstrip(), rest[cut:]
        # 标点不允许出现在行首：把前导标点移到上一行末尾
        m = re.match(rf"^[{re.escape(_PUNCT)}]+", rest)
        if m:
            piece += m.group(0)
            rest = rest[m.end():]
        piece, rest = piece.strip(), rest.strip()
        if piece:
            lines.append(piece)
            remaining_slots -= 1
        if not rest:
            break
    return lines


def _break_candidates(text: str, limit_cols: int) -> list[tuple[int, int]]:
    """返回截至 limit_cols 的可断点 [(字符位置, 累计列宽)]。"""
    candidates: list[tuple[int, int]] = []
    width = 0
    for i, ch in enumerate(text):
        cw = 2 if unicodedata.east_asian_width(ch) in ("W", "F") else 1
        if width + cw > limit_cols:
            break
        width += cw
        if ch == " " or unicodedata.east_asian_width(ch) in ("W", "F") or ch in _PUNCT:
            candidates.append((i + 1, width))
    return candidates


def _find_wrap_point(text: str, limit_cols: int, target_cols: float) -> int:
    candidates = _break_candidates(text, limit_cols)
    if not candidates:
        # 英文长词等：在列限处硬切
        width = 0
        for i, ch in enumerate(text):
            cw = 2 if unicodedata.east_asian_width(ch) in ("W", "F") else 1
            if width + cw > limit_cols:
                return i
            width += cw
        return len(text)
    # 选离理想切分位置最近的断点（行宽均衡）
    return min(candidates, key=lambda c: abs(c[1] - target_cols))[0]


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


# ---------------- 标点规范化 ----------------

_CJK_CHAR = r"[㐀-䶿一-鿿豈-﫿]"
_HALF_PUNCT = {",": "，", ";": "；", ":": "：", "?": "？", "！": "！", "!": "！"}


def normalize_cjk_punct(text: str) -> str:
    """CJK 语境中的半角标点转全角。

    标点的前或后紧邻 CJK 字符即转换；数字小数点等不受影响。
    """
    def repl(m: re.Match) -> str:
        ch = m.group(0)
        return _HALF_PUNCT.get(ch, ch)

    # 标点前为 CJK：lookbehind 在标点位置检查其前字符
    # 标点后为 CJK：lookahead 必须放在标点匹配之后
    punct = r"[,;?!:！]"
    pattern = re.compile(
        rf"(?:(?<={_CJK_CHAR}){punct}|{punct}(?={_CJK_CHAR}))"
    )
    return pattern.sub(repl, text)


def _normalize_unit_punct(units: list[Unit]) -> None:
    """逐 unit 规范化标点；unit 末尾标点借下一 unit 首字符做 lookahead。"""
    for i, unit in enumerate(units):
        nxt = units[i + 1].text[:1] if i + 1 < len(units) else ""
        text = normalize_cjk_punct(unit.text + nxt)
        unit.text = text[: len(text) - len(nxt)] if nxt else text


def _make_unit(words: list[Word]) -> Unit:
    return Unit(words, "".join(w.text for w in words).strip(),
                words[0].start, words[-1].end)


def _split_oversized_units(units: list[Unit], max_dur: float) -> list[Unit]:
    """把超过 max_dur 的 unit 在最接近 start+max_dur 的词边界切开，循环处理。"""
    result: list[Unit] = []
    for unit in units:
        pieces = [unit]
        while pieces[-1].end - pieces[-1].start > max_dur and len(pieces[-1].words) > 1:
            cur = pieces[-1]
            target = cur.start + max_dur
            # 在目标时刻 ±2.5s 范围内，选停顿最大的词边界（更像分句点）；
            # 并列时取离目标最近
            best_i = 1
            best_key: tuple[float, float] = (float("inf"), float("inf"))
            for i in range(1, len(cur.words)):
                gap = cur.words[i].start - cur.words[i - 1].end
                delta = abs(cur.words[i].start - target)
                key = (-gap if delta <= 2.5 else 0.0, delta)
                if key < best_key:
                    best_key, best_i = key, i
            left = _make_unit(cur.words[:best_i])
            right = _make_unit(cur.words[best_i:])
            pieces[-1:] = [left, right]
        result.extend(pieces)
    return result


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

    pause_split_s = float(scfg.get("pause_split_ms", 300)) / 1000.0
    units = _build_units(segments, pause_split_s=pause_split_s)
    units = _split_oversized_units(units, max_dur)
    _normalize_unit_punct(units)
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
        text = normalize_cjk_punct("".join(u.text for u in group).strip())
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
