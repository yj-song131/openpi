#!/bin/bash
# Pair C: composite_unseen 16 tasks (task_id 0-15), GPU 7, port 8012
ROLLOUT_ROOT=/mnt/aix24702/robocasa-rollouts/phase2_unseen
TASKS="ArrangeBreadBasket,ArrangeTea,BreadSelection,CategorizeCondiments,CuttingToolSelection,GarnishPancake,GatherTableware,HeatKebabSandwich,MakeIceLemonade,PanTransfer,PortionHotDogs,RecycleBottlesByType,SeparateFreezerRack,WaffleReheat,WashFruitColander,WeighIngredients"
export CUDA_VISIBLE_DEVICES=2 MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
mkdir -p "$ROLLOUT_ROOT/env_records"
cd /mnt/aix24702/robocasa-openpi
echo "[client-C] Waiting for server at port 8012..." | tee -a $ROLLOUT_ROOT/client_C.log
until .venv/bin/python -c "import socket,sys; s=socket.socket(); s.settimeout(2); s.connect(('127.0.0.1',8012)); s.close()" 2>/dev/null; do sleep 5; done
echo "[client-C] Server ready. Starting composite_unseen 16 tasks." | tee -a $ROLLOUT_ROOT/client_C.log
.venv/bin/python examples/robocasa/main.py \
    --args.tasks "$TASKS" \
    --args.task-id-offset 0 \
    --args.num-trials 25 \
    --args.eval-fixed-layouts \
    --args.episodes-per-layout 5 \
    --args.rollout-root "$ROLLOUT_ROOT" \
    --args.host 127.0.0.1 --args.port 8012 \
    --args.replan-steps 5 --args.seed 42 \
    2>&1 | tee -a $ROLLOUT_ROOT/client_C.log
echo "[client-C] Done." | tee -a $ROLLOUT_ROOT/client_C.log
