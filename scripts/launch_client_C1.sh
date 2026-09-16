#!/bin/bash
# Client C1: unseen tasks 0-7 (ArrangeBreadBasket~HeatKebabSandwich), server C port 8012, GPU 2
ROLLOUT_ROOT=/mnt/aix24702/robocasa-rollouts/phase2_unseen
TASKS="ArrangeBreadBasket,ArrangeTea,BreadSelection,CategorizeCondiments,CuttingToolSelection,GarnishPancake,GatherTableware,HeatKebabSandwich"
export CUDA_VISIBLE_DEVICES=2 MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
cd /mnt/aix24702/robocasa-openpi
echo "[client-C1] Waiting for server at port 8012..." | tee -a $ROLLOUT_ROOT/client_C1.log
until .venv/bin/python -c "import socket; s=socket.socket(); s.settimeout(2); s.connect(('127.0.0.1',8012)); s.close()" 2>/dev/null; do sleep 5; done
echo "[client-C1] Server ready. Starting unseen tasks 0-7." | tee -a $ROLLOUT_ROOT/client_C1.log
.venv/bin/python examples/robocasa/main.py \
    --args.tasks "$TASKS" \
    --args.task-id-offset 0 \
    --args.num-trials 25 \
    --args.eval-fixed-layouts \
    --args.episodes-per-layout 5 \
    --args.rollout-root "$ROLLOUT_ROOT" \
    --args.host 127.0.0.1 --args.port 8012 \
    --args.replan-steps 5 --args.seed 42 \
    2>&1 | tee -a $ROLLOUT_ROOT/client_C1.log
echo "[client-C1] Done." | tee -a $ROLLOUT_ROOT/client_C1.log
