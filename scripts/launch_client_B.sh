#!/bin/bash
# Pair B: seen tasks 5-9 (task_id 5-9 via offset), GPU 5, port 8011
ROLLOUT_ROOT=/mnt/aix24702/robocasa-rollouts/phase1_eval5x5
TASKS="AfterwashSorting,TurnOffStove,StockingBreakfastFoods,PickPlaceToasterToCounter,PrepareSausageCheese"
export CUDA_VISIBLE_DEVICES=7 MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
cd /mnt/aix24702/robocasa-openpi
echo "[client-B] Waiting for server at port 8011..." | tee -a $ROLLOUT_ROOT/client_B.log
until .venv/bin/python -c "import socket,sys; s=socket.socket(); s.settimeout(2); s.connect(('127.0.0.1',8011)); s.close()" 2>/dev/null; do sleep 5; done
echo "[client-B] Server ready. Starting tasks 5-9 (offset=5)." | tee -a $ROLLOUT_ROOT/client_B.log
.venv/bin/python examples/robocasa/main.py \
    --args.tasks "$TASKS" \
    --args.task-id-offset 5 \
    --args.num-trials 25 \
    --args.eval-fixed-layouts \
    --args.episodes-per-layout 5 \
    --args.rollout-root "$ROLLOUT_ROOT" \
    --args.host 127.0.0.1 --args.port 8011 \
    --args.replan-steps 5 --args.seed 42 \
    2>&1 | tee -a $ROLLOUT_ROOT/client_B.log
echo "[client-B] Done." | tee -a $ROLLOUT_ROOT/client_B.log
