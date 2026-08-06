#!/bin/bash
cd /scratch/gpfs/GRIFFITHS/aw2418/llm-polysemy
TARGET=3000
while true; do
  sft=$(find data/outputs_image/*/simplear_sft -name '*.png' 2>/dev/null | wc -l)
  rl=$(find data/outputs_image/*/simplear_rl -name '*.png' 2>/dev/null | wc -l)
  run=$(squeue -u aw2418 -h -t RUNNING -o "%T" 2>/dev/null | wc -l)
  pend=$(squeue -u aw2418 -h -t PENDING -r 2>/dev/null | wc -l)
  echo "$(date '+%H:%M:%S')  sft=$sft/$TARGET  rl=$rl/$TARGET  | running=$run pending=$pend"
  if [ "$sft" -ge "$TARGET" ] && [ "$rl" -ge "$TARGET" ]; then echo "BOTH_COMPLETE"; break; fi
  sleep 120
done
