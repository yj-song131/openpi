#!/bin/bash
# Launch pi0.5 policy server for Phase 1 rollout collection.
# Uses GPU 0 (free). Policy records saved to phase1_seen/policy_records/.

ROLLOUT_ROOT=/mnt/aix24702/robocasa-rollouts/phase1_seen
CKPT_DIR=/mnt/aix24702/robocasa-openpi/checkpoints/pi05_pretrain_human300/multitask_learning/75000
CONFIG=pi05_pretrain_human300
LOG=/mnt/aix24702/robocasa-rollouts/phase1_seen/server.log

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
