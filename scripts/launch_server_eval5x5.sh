#!/bin/bash
# Launch pi0.5 policy server for 5x5 eval rollout collection.
# 5 fixed layouts x 5 episodes = 25 ep/task (original-paper eval style).

ROLLOUT_ROOT=/mnt/aix24702/robocasa-rollouts/phase1_eval5x5
CKPT_DIR=/mnt/aix24702/robocasa-openpi/checkpoints/pi05_pretrain_human300/multitask_learning/75000
CONFIG=pi05_pretrain_human300
LOG=$ROLLOUT_ROOT/server.log

mkdir -p "$ROLLOUT_ROOT/policy_records"
export CUDA_VISIBLE_DEVICES=0

cd /mnt/aix24702/robocasa-openpi
echo "[server] Starting pi0.5 server on GPU 0 at port 8000..." | tee -a $LOG
.venv/bin/python scripts/serve_policy.py \
    --record \
    --record-dir "$ROLLOUT_ROOT/policy_records" \
    --port 8000 \
    policy:checkpoint \
    --policy.config "$CONFIG" \
    --policy.dir "$CKPT_DIR" \
    2>&1 | tee -a $LOG
