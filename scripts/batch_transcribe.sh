#!/bin/bash
# 批量抽音 + 转写：读取视频清单（每行一个 mp4 路径），逐个处理并校验。
# 用法：bash scripts/batch_transcribe.sh <清单文件> <配置文件> [输出名前缀映射...]
# 产物：data/audio/<stem>.wav、data/srt/<stem>.srt
# 注意：转写完成后人工校对为强制中断点，本脚本不做 mux。
set -u

LIST="${1:?用法：batch_transcribe.sh <清单> <配置>}"
CONFIG="${2:?缺少配置文件}"
[ -f "$LIST" ] || { echo "清单不存在：$LIST" >&2; exit 2; }
[ -f "$CONFIG" ] || { echo "配置不存在：$CONFIG" >&2; exit 2; }

ok=0
fail=0
nospeech=0
# 清单用 FD 3 读取：循环内的 ffmpeg 默认会读 stdin，
# 若清单走 stdin 会被 ffmpeg 吞掉若干字节导致整行丢失，故必须分离。
while IFS= read -r video <&3 || [ -n "$video" ]; do
  video="${video%%#*}"                      # 去掉行内注释
  [ -z "${video// /}" ] && continue
  if [ ! -f "$video" ]; then
    echo "[FAIL] 文件不存在：$video"
    fail=$((fail + 1))
    continue
  fi
  stem=$(basename "${video%.*}")
  wav="data/audio/${stem}.wav"
  srt="data/srt/${stem}.srt"

  echo "=================================================================="
  echo "[BATCH] 处理：$video"
  .venv/bin/vidsub extract "$video" -o "$wav" </dev/null
  rc_extract=$?
  # extract rc：0=正常，2=时长差>100ms 警告（WAV 已写出，可继续），其余=失败
  if [ "$rc_extract" -eq 0 ] || [ "$rc_extract" -eq 2 ]; then
    .venv/bin/vidsub transcribe "$wav" -o "$srt" --config "$CONFIG" </dev/null > /tmp/.batch_tr.log 2>&1
  fi
  if [ "$rc_extract" -eq 0 ] || [ "$rc_extract" -eq 2 ]; then
    cat /tmp/.batch_tr.log
    if grep -q "0 段" /tmp/.batch_tr.log; then
      echo "[NOSPEECH] $stem（无语音轨，跳过校验）"
      rm -f "$srt"
      nospeech=$((nospeech + 1))
      continue
    fi
    if .venv/bin/vidsub validate "$srt" --video "$video"; then
      echo "[OK] $stem"
      ok=$((ok + 1))
    else
      echo "[FAIL] 校验未通过：$stem"
      fail=$((fail + 1))
    fi
  else
    echo "[FAIL] 处理失败：$video"
    fail=$((fail + 1))
  fi
done 3< "$LIST"

echo "=================================================================="
echo "[BATCH] 完成：成功 $ok，无语音 $nospeech，失败 $fail"
exit "$fail"
