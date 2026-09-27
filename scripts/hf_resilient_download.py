#!/usr/bin/env python3
"""弹性下载 HuggingFace 仓库全部文件（经 hf-mirror）。

- requests 流式下载，Range 断点续传
- 断流自动重试（默认无限次），适合"每 N MB 被掐"的网络
- 下载为本地目录，可直接作为 faster-whisper 的模型路径

用法：python scripts/hf_resilient_download.py <repo_id> <out_dir>
"""

from __future__ import annotations

import os
import sys
import time

import requests

ENDPOINT = os.environ.get("HF_ENDPOINT", "https://hf-mirror.com").rstrip("/")
CHUNK = 1024 * 1024  # 1 MiB


def list_files(repo: str) -> list[str]:
    url = f"{ENDPOINT}/api/models/{repo}"
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    return [s["rfilename"] for s in r.json()["siblings"]]


def remote_size(url: str) -> int | None:
    try:
        r = requests.head(url, timeout=30, allow_redirects=True)
        if r.status_code == 200 and "Content-Length" in r.headers:
            return int(r.headers["Content-Length"])
    except requests.RequestException:
        pass
    return None


def download_one(repo: str, filename: str, dest: str) -> None:
    url = f"{ENDPOINT}/{repo}/resolve/main/{filename}"
    os.makedirs(os.path.dirname(dest) or ".", exist_ok=True)
    total = remote_size(url)
    attempt = 0
    while True:
        have = os.path.getsize(dest) if os.path.exists(dest) else 0
        if total is not None and have >= total:
            print(f"  完成：{filename}（{have // 1024 // 1024}MB）", flush=True)
            return
        headers = {"Range": f"bytes={have}-"} if have else {}
        mode = "ab" if have else "wb"
        attempt += 1
        try:
            with requests.get(url, headers=headers, stream=True, timeout=60) as r:
                if r.status_code not in (200, 206):
                    raise requests.RequestException(f"HTTP {r.status_code}")
                if total is None:
                    cl = r.headers.get("Content-Length")
                    if cl:
                        total = (have if r.status_code == 206 else 0) + int(cl)
                with open(dest, mode) as f:
                    last_print = time.time()
                    for chunk in r.iter_content(chunk_size=CHUNK):
                        if chunk:
                            f.write(chunk)
                            have += len(chunk)
                            if time.time() - last_print > 5:
                                pct = f"{have * 100 / total:.1f}%" if total else f"{have // 1048576}MB"
                                print(f"  {filename}: {pct}", flush=True)
                                last_print = time.time()
        except requests.RequestException as exc:
            wait = min(2 * attempt, 30)
            print(f"  {filename} 第 {attempt} 次中断（{exc}），{wait}s 后续传 "
                  f"（已 {os.path.getsize(dest) // 1048576 if os.path.exists(dest) else 0}MB）",
                  flush=True)
            time.sleep(wait)


def main(argv: list[str]) -> int:
    if len(argv) != 3:
        print(__doc__)
        return 2
    repo, out_dir = argv[1], argv[2]
    files = list_files(repo)
    print(f"仓库 {repo} 含 {len(files)} 个文件：{files}", flush=True)
    for filename in files:
        download_one(repo, filename, os.path.join(out_dir, filename))
    print(f"全部完成：{out_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
