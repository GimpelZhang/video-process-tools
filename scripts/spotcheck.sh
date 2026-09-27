#!/bin/bash
# 抽取指定时间段的音频片段用于试听核对。
# 用法：spotcheck.sh <input.wav> <start_seconds> <end_seconds> <output.wav>
set -euo pipefail

if [ "$#" -ne 4 ]; then
  echo "用法：$0 <input.wav> <start_seconds> <end_seconds> <output.wav>" >&2
  exit 2
fi

ffmpeg -y -ss "$2" -to "$3" -i "$1" -c:a pcm_s16le "$4" -loglevel error
echo "已写出 $4（$2 -> $3）"
