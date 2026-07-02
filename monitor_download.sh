#!/usr/bin/env zsh
# Monitor WikiArt download; restart if stalled. Run via loop every 25m.

set -euo pipefail
PROJECT=~/Projects/painting-search
STATE_FILE=$PROJECT/wikiart_data/.monitor_state
LOG=$PROJECT/download.log
CACHE=~/.cache/huggingface/hub/datasets--huggan--wikiart

cache_bytes=$(du -sb "$CACHE" 2>/dev/null | cut -f1 || echo 0)
image_count=$(ls "$PROJECT/wikiart_data/images/" 2>/dev/null | wc -l)
running=$(pgrep -fc "python download_wikiart.py" 2>/dev/null || echo 0)
now=$(date +%s)

prev_bytes=0
prev_time=0
if [[ -f $STATE_FILE ]]; then
  source "$STATE_FILE"
fi

echo "[$(date)] cache=${cache_bytes}B images=$image_count running=$running"

stalled=0
if [[ $running -gt 0 && $cache_bytes -eq $prev_bytes && $image_count -eq ${prev_images:-0} ]]; then
  stalled=1
  echo "STALLED: no progress since last check"
fi

if [[ $running -eq 0 && $image_count -lt 81444 ]]; then
  echo "DOWNLOAD NOT RUNNING — restarting"
  stalled=1
fi

if [[ $stalled -eq 1 ]]; then
  pkill -f "python download_wikiart.py" 2>/dev/null || true
  find "$CACHE" -name '*.incomplete' -size 0 -delete 2>/dev/null || true
  sleep 2
  cd "$PROJECT" && source .venv/bin/activate
  export HF_HUB_DISABLE_XET=1
  export HF_HUB_DOWNLOAD_TIMEOUT=300
  nohup python download_wikiart.py -o wikiart_data --resume >> "$LOG" 2>&1 &
  echo "Restarted download (PID $!)"
fi

cat > "$STATE_FILE" <<EOF
prev_bytes=$cache_bytes
prev_images=$image_count
prev_time=$now
EOF
