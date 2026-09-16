#!/bin/bash
# Launch rollout collection client for composite_unseen tasks.
# 16 tasks × 25 ep (5 layouts × 5 ep) = 400 total episodes.
# Uses port 8001 (parallel server on GPU 2).

ROLLOUT_ROOT=/mnt/aix24702/robocasa-rollouts/phase2_unseen
LOG=$ROLLOUT_ROOT/client.log

TASKS="ArrangeBreadBasket,ArrangeTea,BreadSelection,CategorizeCondiments,CuttingToolSelection,GarnishPancake,GatherTableware,HeatKebabSandwich,MakeIceLemonade,PanTransfer,PortionHotDogs,RecycleBottlesByType,SeparateFreezerRack,WaffleReheat,WashFruitColander,WeighIngredients"

export CUDA_VISIBLE_DEVICES=5
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl

mkdir -p "$ROLLOUT_ROOT/env_records"

cd /mnt/aix24702/robocasa-openpi

echo "[client-unseen] Waiting for server at 127.0.0.1:8001 ..." | tee -a $LOG
until .venv/bin/python -c "
import socket, sys
s = socket.socket()
s.settimeout(2)
try:
    s.connect(('127.0.0.1', 8001))
    s.close()
    sys.exit(0)
except:
    sys.exit(1)
" 2>/dev/null; do
    sleep 5
done
echo "[client-unseen] Server ready. Starting composite_unseen rollout collection." | tee -a $LOG

.venv/bin/python examples/robocasa/main.py \
    --args.tasks "$TASKS" \
    --args.num-trials 25 \
    --args.eval-fixed-layouts \
    --args.episodes-per-layout 5 \
    --args.rollout-root "$ROLLOUT_ROOT" \
    --args.host 127.0.0.1 \
    --args.port 8001 \
    --args.replan-steps 5 \
    --args.seed 42 \
    2>&1 | tee -a $LOG

echo "[client-unseen] Phase 2 (composite_unseen) collection complete." | tee -a $LOG
