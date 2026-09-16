#!/bin/bash
# Launch rollout collection client for Phase 1 (seen tasks, pretrain split).
# 10 tasks sampled from pretrain300 (seed=42), 50 episodes each.
# Waits for server to be ready, then runs on GPU 4 (for rendering).

ROLLOUT_ROOT=/mnt/aix24702/robocasa-rollouts/phase1_seen
LOG=$ROLLOUT_ROOT/client.log

# 10 tasks sampled from pretrain300 with seed=42
TASKS="TurnOnStove,CloseToasterOvenDoor,GatherCuttingTools,DrainVeggies,CutBuffetPizza,AfterwashSorting,TurnOffStove,StockingBreakfastFoods,PickPlaceToasterToCounter,PrepareSausageCheese"

export CUDA_VISIBLE_DEVICES=4
# robocasa uses EGL for headless rendering
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl

cd /mnt/aix24702/robocasa-openpi

# Wait for server to come up (model loading can take 2-5 min)
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
echo "[client] Server ready. Starting rollout collection." | tee -a $LOG

.venv/bin/python examples/robocasa/main.py \
    --args.split pretrain \
    --args.tasks "$TASKS" \
    --args.num-trials 50 \
    --args.rollout-root "$ROLLOUT_ROOT" \
    --args.host 127.0.0.1 \
    --args.port 8000 \
    --args.replan-steps 5 \
    --args.seed 42 \
    2>&1 | tee -a $LOG

echo "[client] Phase 1 collection complete." | tee -a $LOG
