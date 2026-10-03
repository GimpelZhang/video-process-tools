#!/bin/bash
# 批量合成：读取视频清单（每行一个 mp4），用 data/srt/<stem>.srt 生成
# 硬烧录（data/output/<stem>.subbed.mp4）与软字幕（data/output/<stem>.soft.mp4）。
# 用法：bash scripts/batch_mux.sh <清单文件>
# 与 batch_transcribe.sh 相同：清单走 FD3，所有子命令 stdin 接 /dev/null。
set -u

LIST="${1:?用法：batch_mux.sh <清单文件>}"
[ -f "$LIST" ] || { echo "清单不存在：$LIST" >&2; exit 2; }

ok=0
fail=0
skip=0
while IFS= read -r video <&3 || [ -n "$video" ]; do
  video="${video%%#*}"
  [ -z "${video// /}" ] && continue
  stem=$(basename "${video%.*}")
  srt="data/srt/${stem}.srt"
  if [ ! -f "$video" ]; then
    echo "[FAIL] 视频不存在：$video"
    fail=$((fail + 1))
    continue
  fi
  if [ ! -f "$srt" ]; then
    echo "[SKIP] 无字幕文件：$stem"
    skip=$((skip + 1))
    continue
  fi
  if ! grep -q -- "-->" "$srt"; then
    echo "[SKIP] 字幕文件为空（无条目）：$stem"
    skip=$((skip + 1))
    continue
  fi

  hard="data/output/${stem}.subbed.mp4"
  soft="data/output/${stem}.soft.mp4"
  echo "=================================================================="
  echo "[MUX] 处理：$stem"
  mkdir -p data/output
  if .venv/bin/vidsub mux "$video" "$srt" -o "$hard" </dev/null \
      && .venv/bin/vidsub mux "$video" "$srt" --soft -o "$soft" </dev/null; then
    echo "[OK] $stem"
    ok=$((ok + 1))
  else
    echo "[FAIL] 合成失败：$stem"
    fail=$((fail + 1))
  fi
done 3< "$LIST"

echo "=================================================================="
echo "[MUX] 完成：成功 $ok，跳过 $skip，失败 $fail"
exit "$fail"
