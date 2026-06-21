#!/usr/bin/env bash
# Unpack WaterScenes on the training box for A4 fusion training. Two-level: pull the
# needed component zips out of WaterScenes-Published.zip, then extract each into
# datasets/WaterScenes/ (image/ radar/ detection/ calib/ + train/val/test.txt at root).
set -u
cd /root/autodl-tmp || exit 1
OUT=/root/autodl-tmp/SoarVision/datasets/WaterScenes
TMP=/root/autodl-tmp/ws_extract
say(){ echo "[$(date +%T)] $*"; }

say "extract component zips from outer (small first, image last)"
mkdir -p "$TMP" "$OUT"
unzip -o -j WaterScenes-Published.zip \
  "WaterScenes-Published/radar.zip" "WaterScenes-Published/detection.zip" \
  "WaterScenes-Published/calib.zip" \
  "WaterScenes-Published/train.txt" "WaterScenes-Published/val.txt" "WaterScenes-Published/test.txt" \
  -d "$TMP" || say "outer(small) extract FAILED"
for z in radar detection calib; do
  say "unzip $z.zip"
  unzip -q -o "$TMP/$z.zip" -d "$OUT" || say "$z FAILED"
done
cp "$TMP"/train.txt "$TMP"/val.txt "$TMP"/test.txt "$OUT"/ 2>/dev/null

say "extract image.zip from outer (11G) ..."
unzip -o -j WaterScenes-Published.zip "WaterScenes-Published/image.zip" -d "$TMP" || say "image outer FAILED"
say "unzip image.zip (54k jpgs, minutes) ..."
unzip -q -o "$TMP/image.zip" -d "$OUT" || say "image FAILED"

say "cleanup intermediate zips"
rm -rf "$TMP"
say "counts: image=$(ls "$OUT"/image 2>/dev/null | wc -l) radar=$(ls "$OUT"/radar 2>/dev/null | wc -l) det=$(ls "$OUT"/detection/yolo 2>/dev/null | wc -l)"
df -h /root/autodl-tmp | tail -1
say "DONE WaterScenes -> $OUT"
