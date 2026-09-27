#!/usr/bin/env python3
"""对比两份 SRT：字错率 CER 与英文术语保留率。

用法：python scripts/cer.py <hypothesis.srt> <reference.srt>
标点/空白不计入 CER；拉丁字母按小写比较。
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from vidsub.srt import parse_srt_file  # noqa: E402

_KEEP = re.compile(r"[一-鿿 a-zA-Z0-9]")
_EN_RUN = re.compile(r"[A-Za-z][A-Za-Z0-9+#.\-]*")


def srt_text(path: str) -> str:
    return "\n".join(c.text for c in parse_srt_file(path))


def normalize(text: str) -> str:
    return "".join(_KEEP.findall(text)).lower().replace(" ", "")


def edit_distance(a: str, b: str) -> int:
    """Levenshtein，空间 O(len(b))。"""
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        cur = [i]
        for j, cb in enumerate(b, start=1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def english_runs(text: str) -> list[str]:
    return [m.group(0).lower() for m in _EN_RUN.finditer(text)]


def english_retention(hyp_text: str, ref_text: str) -> tuple[int, int]:
    """参考稿英文片段在转写稿中按序命中数（子序列匹配）。"""
    ref = english_runs(ref_text)
    hyp = english_runs(hyp_text)
    if not ref:
        return 0, 0
    j = 0
    hit = 0
    for term in ref:
        while j < len(hyp) and hyp[j] != term:
            j += 1
        if j < len(hyp):
            hit += 1
            j += 1
    return hit, len(ref)


def main(argv: list[str]) -> int:
    if len(argv) != 3:
        print(__doc__)
        return 2
    hyp_text = srt_text(argv[1])
    ref_text = srt_text(argv[2])
    hyp_n = normalize(hyp_text)
    ref_n = normalize(ref_text)
    dist = edit_distance(hyp_n, ref_n)
    cer = dist / max(len(ref_n), 1)
    hit, total = english_retention(hyp_text, ref_text)
    print(f"参考稿有效字符数：{len(ref_n)}")
    print(f"编辑距离：{dist}")
    print(f"CER：{cer * 100:.2f}%")
    print(f"英文术语：{hit}/{total} 保留，保留率 {(hit / total * 100 if total else 100):.1f}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
