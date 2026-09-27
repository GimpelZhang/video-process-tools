"""SRT 校验：编码、序号、时间合法性、重叠、空块、超长等。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .audio import probe_duration
from .srt import Cue, display_width, parse_srt_file


@dataclass
class Finding:
    level: str  # "error" | "warning"
    message: str


def validate_cues(
    cues: list[Cue],
    video: str | None = None,
    expected_index: bool = True,
) -> list[Finding]:
    findings: list[Finding] = []

    if not cues:
        return [Finding("error", "SRT 中没有任何字幕条目")]

    for i, cue in enumerate(cues, start=1):
        if expected_index and cue.index != i:
            findings.append(Finding("error", f"第 {i} 块序号异常：读到 {cue.index}，期望 {i}"))
        if cue.start >= cue.end:
            findings.append(Finding("error", f"第 {i} 条起始不小于结束：{cue.start:.3f}≥{cue.end:.3f}"))
        dur = cue.end - cue.start
        if dur > 10.0:
            findings.append(Finding("warning", f"第 {i} 条时长 {dur:.1f}s 超过 10s"))
        text = cue.text
        if not text.strip():
            findings.append(Finding("error", f"第 {i} 条正文为空"))
        if cue.lines and len(cue.lines) > 2:
            findings.append(Finding("warning", f"第 {i} 条超过 2 行（{len(cue.lines)} 行）"))
        for line in cue.lines:
            if display_width(line) > 44:  # 22 汉字宽
                findings.append(Finding("warning", f"第 {i} 条单行超长（{display_width(line) // 2} 汉字宽）"))

    # 单调性 / 重叠
    for prev, cur in zip(cues, cues[1:]):
        if cur.start < prev.start or cur.end < prev.end:
            findings.append(Finding("error",
                f"第 {prev.index}、{cur.index} 条时间未单调递增"))
        if cur.start < prev.end - 0.001:
            findings.append(Finding("error",
                f"第 {prev.index}、{cur.index} 条时间重叠：{prev.end:.3f} > {cur.start:.3f}"))

    if video:
        try:
            total = probe_duration(video)
            last_end = cues[-1].end
            if last_end > total + 0.1:
                findings.append(Finding("error",
                    f"最后一条结束 {last_end:.3f}s 超过视频时长 {total:.3f}s"))
        except Exception as exc:  # 探测失败不阻塞其余检查
            findings.append(Finding("warning", f"无法探测视频时长：{exc}"))

    return findings


def validate_file(path: str | Path, video: str | None = None) -> list[Finding]:
    raw = Path(path).read_bytes()
    if raw.startswith(b"\xef\xbb\xbf"):
        bw: list[Finding] = []  # BOM 给 warning（个别 libass 版本有问题）
    else:
        bw = []
    try:
        text = raw.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        return [Finding("error", f"文件不是合法 UTF-8：{exc}")]
    if raw.startswith(b"\xef\xbb\xbf"):
        bw = [Finding("warning", "文件带 UTF-8 BOM，建议另存为无 BOM")]
    cues = parse_srt_file(path)
    return bw + validate_cues(cues, video=video)
