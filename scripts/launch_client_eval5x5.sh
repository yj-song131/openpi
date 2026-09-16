#!/bin/bash
# Launch rollout collection client — 5x5 eval style.
# 10 pretrain tasks, 5 fixed layouts x 5 episodes = 25 ep/task = 250 total.

ROLLOUT_ROOT=/mnt/aix24702/robocasa-rollouts/phase1_eval5x5
LOG=$ROLLOUT_ROOT/client.log

TASKS="TurnOnStove,CloseToasterOvenDoor,GatherCuttingTools,DrainVeggies,CutBuffetPizza,AfterwashSorting,TurnOffStove,StockingBreakfastFoods,PickPlaceToasterToCounter,PrepareSausageCheese"

export CUDA_VISIBLE_DEVICES=4
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl

cd /mnt/aix24702/robocasa-openpi

echo "[client] Waiting for server at 127.0.0.1:8000 ..." | tee -a $LOG
until .venv/bin/python -c "
import socket, sys
s = socket.socket()
s.settimeout(2)
try:
    s.connect(('127.0.0.1', 8000))
    s.close()
    sys.exit(0)
except:
    sys.exit(1)
" 2>/dev/null; do
    sleep 5
done
echo "[client] Server ready. Starting 5x5 eval rollout collection." | tee -a $LOG

.venv/bin/python examples/robocasa/main.py \
    --args.tasks "$TASKS" \
    --args.num-trials 25 \
    --args.eval-fixed-layouts True \
    --args.episodes-per-layout 5 \
    --args.rollout-root "$ROLLOUT_ROOT" \
    --args.host 127.0.0.1 \
    --args.port 8000 \
    --args.replan-steps 5 \
    --args.seed 42 \
    2>&1 | tee -a $LOG

echo "[client] Phase 1 eval5x5 collection complete." | tee -a $LOG
