#!/bin/bash
# Client C2: unseen tasks 8-15 (MakeIceLemonade~WeighIngredients), server A port 8010, GPU 2
ROLLOUT_ROOT=/mnt/aix24702/robocasa-rollouts/phase2_unseen
TASKS="MakeIceLemonade,PanTransfer,PortionHotDogs,RecycleBottlesByType,SeparateFreezerRack,WaffleReheat,WashFruitColander,WeighIngredients"
export CUDA_VISIBLE_DEVICES=2 MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
cd /mnt/aix24702/robocasa-openpi
echo "[client-C2] Waiting for server at port 8010..." | tee -a $ROLLOUT_ROOT/client_C2.log
until .venv/bin/python -c "import socket; s=socket.socket(); s.settimeout(2); s.connect(('127.0.0.1',8010)); s.close()" 2>/dev/null; do sleep 5; done
echo "[client-C2] Server ready. Starting unseen tasks 8-15 (offset=8)." | tee -a $ROLLOUT_ROOT/client_C2.log
.venv/bin/python examples/robocasa/main.py \
    --args.tasks "$TASKS" \
    --args.task-id-offset 8 \
    --args.num-trials 25 \
    --args.eval-fixed-layouts \
    --args.episodes-per-layout 5 \
    --args.rollout-root "$ROLLOUT_ROOT" \
    --args.host 127.0.0.1 --args.port 8010 \
    --args.replan-steps 5 --args.seed 42 \
    2>&1 | tee -a $ROLLOUT_ROOT/client_C2.log
echo "[client-C2] Done." | tee -a $ROLLOUT_ROOT/client_C2.log
