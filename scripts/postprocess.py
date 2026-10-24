#!/usr/bin/env python3
"""SRT 后处理：繁转简 + 幻觉重复过滤 + 时间重叠修复。

用于港台腔等场景：Whisper 输出繁体字，且可能在非语音段陷入
"我…我…"式的 token 重复循环幻觉。

用法：python scripts/postprocess.py <input.srt> [-o output.srt]
默认原地覆盖输入文件（-o 可指定输出）。
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from vidsub.srt import format_timestamp, parse_srt_file  # noqa: E402

_PUNCT_WS = re.compile(r"[\s…，。！？、；：,.!?;:\-—…·「」『』【】《》()（）]+")


def _core(text: str) -> str:
    return _PUNCT_WS.sub("", text)


def is_spam_cue(text: str) -> bool:
    """判定重复循环幻觉：去标点后字符种类极少且重复次数多。

    例：「我…我…我…」（19 个"我"）、「好。好。好。好。」等。
    """
    core = _core(text)
    if len(core) < 6:
        return False
    return len(set(core)) <= 2 and len(core) / max(len(set(core)), 1) >= 5


def drop_spam_runs(cues, run_min: int = 3, max_core_len: int = 3) -> list:
    """删除连续重复短句组成的幻觉串。

    单条「我…」可能是真实台词，但连续 ≥3 条 core 完全相同的短句
    （如几十条「我…」相邻）是 Whisper 循环幻觉，整组删除。
    """
    kept = []
    i = 0
    n = len(cues)
    dropped = 0
    while i < n:
        core = _core(cues[i].text)
        j = i + 1
        if core and len(core) <= max_core_len:
            while j < n and _core(cues[j].text) == core:
                j += 1
        if j - i >= run_min:
            dropped += j - i
            i = j
            continue
        kept.extend(cues[i:j])
        i = j
    return kept, dropped


def fix_overlaps(cues, min_gap: float = 0.03) -> list:
    """消除重叠：优先裁剪前条 end；前条被压扁时推后后条 start；
    仍无法容纳（时长被压没）的后条删除。返回处理后的列表。"""
    out = [cues[0]] if cues else []
    for nxt in cues[1:]:
        cur = out[-1]
        if nxt.start > cur.end:
            out.append(nxt)
            continue
        new_end = nxt.start - min_gap
        if new_end > cur.start:
            cur.end = new_end
            out.append(nxt)
        elif nxt.start + min_gap < nxt.end:
            nxt.start = cur.end + min_gap
            out.append(nxt)
        # 两条 start 几乎重合且后条极短：丢弃后条
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("srt")
    ap.add_argument("-o", "--output", help="输出路径，默认原地覆盖")
    args = ap.parse_args()

    try:
        from opencc import OpenCC
    except ImportError:
        print("缺少依赖：pip install opencc-python-reimplemented", file=sys.stderr)
        return 1

    cc = OpenCC("t2s")
    cues = parse_srt_file(args.srt)

    for cue in cues:
        cue.lines = cc.convert(cue.text).split("\n")

    kept, dropped = [], 0
    for cue in cues:
        if is_spam_cue(cue.text):
            dropped += 1
            continue
        kept.append(cue)
    kept, run_dropped = drop_spam_runs(kept)
    dropped += run_dropped
    kept = fix_overlaps(kept)
    for i, cue in enumerate(kept, start=1):
        cue.index = i

    out = Path(args.output or args.srt)
    blocks = [
        f"{c.index}\n{format_timestamp(c.start)} --> {format_timestamp(c.end)}\n{c.text}"
        for c in kept
    ]
    out.write_text("\n\n".join(blocks) + "\n", encoding="utf-8")
    print(f"[postprocess] {out}：{len(cues)} → {len(kept)} 条"
          f"（删除幻觉 {dropped} 条，已繁转简）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
