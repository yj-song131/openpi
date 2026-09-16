#!/bin/bash
# Launch pi0.5 policy server for composite_unseen rollout collection.
# GPU 2, port 8001 (parallel to seen collection on GPU 0, port 8000).

ROLLOUT_ROOT=/mnt/aix24702/robocasa-rollouts/phase2_unseen
CKPT_DIR=/mnt/aix24702/robocasa-openpi/checkpoints/pi05_pretrain_human300/multitask_learning/75000
CONFIG=pi05_pretrain_human300
LOG=$ROLLOUT_ROOT/server.log

mkdir -p "$ROLLOUT_ROOT/policy_records"
export CUDA_VISIBLE_DEVICES=2

cd /mnt/aix24702/robocasa-openpi
echo "[server-unseen] Starting pi0.5 server on GPU 2 at port 8001..." | tee -a $LOG
.venv/bin/python scripts/serve_policy.py \
    --record \
    --record-dir "$ROLLOUT_ROOT/policy_records" \
    --port 8001 \
    policy:checkpoint \
    --policy.config "$CONFIG" \
    --policy.dir "$CKPT_DIR" \
    2>&1 | tee -a $LOG
