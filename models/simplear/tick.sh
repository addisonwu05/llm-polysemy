#!/bin/bash
cd /scratch/gpfs/GRIFFITHS/aw2418/llm-polysemy
sleep 600
sft=$(find data/outputs_image/*/simplear_sft -name '*.png' 2>/dev/null | wc -l)
rl=$(find data/outputs_image/*/simplear_rl -name '*.png' 2>/dev/null | wc -l)
run=$(squeue -u aw2418 -h -t RUNNING -r 2>/dev/null | wc -l)
pend=$(squeue -u aw2418 -h -t PENDING -r 2>/dev/null | wc -l)
sh_left=$(squeue -j 9988050 -h -r 2>/dev/null | wc -l)
rl_left=$(squeue -j 9988051 -h -r 2>/dev/null | wc -l)
echo "TICK $(date '+%H:%M:%S')  sft=$sft/3000  rl=$rl/3000  | running=$run pending=$pend | sft_tasks_left=$sh_left rl_tasks_left=$rl_left"
if [ "$sft" -ge 3000 ] && [ "$rl" -ge 3000 ]; then echo "BOTH_COMPLETE"; fi
