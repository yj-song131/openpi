#!/bin/bash
# Pair A: seen tasks 0-4 (task_id 0-4), GPU 2, port 8010
ROLLOUT_ROOT=/mnt/aix24702/robocasa-rollouts/phase1_eval5x5
TASKS="TurnOnStove,CloseToasterOvenDoor,GatherCuttingTools,DrainVeggies,CutBuffetPizza"
export CUDA_VISIBLE_DEVICES=2 MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
cd /mnt/aix24702/robocasa-openpi
echo "[client-A] Waiting for server at port 8010..." | tee -a $ROLLOUT_ROOT/client_A.log
until .venv/bin/python -c "import socket,sys; s=socket.socket(); s.settimeout(2); s.connect(('127.0.0.1',8010)); s.close()" 2>/dev/null; do sleep 5; done
echo "[client-A] Server ready. Starting tasks 0-4." | tee -a $ROLLOUT_ROOT/client_A.log
.venv/bin/python examples/robocasa/main.py \
    --args.tasks "$TASKS" \
    --args.task-id-offset 0 \
    --args.num-trials 25 \
    --args.eval-fixed-layouts \
    --args.episodes-per-layout 5 \
    --args.rollout-root "$ROLLOUT_ROOT" \
    --args.host 127.0.0.1 --args.port 8010 \
    --args.replan-steps 5 --args.seed 42 \
    2>&1 | tee -a $ROLLOUT_ROOT/client_A.log
echo "[client-A] Done." | tee -a $ROLLOUT_ROOT/client_A.log
