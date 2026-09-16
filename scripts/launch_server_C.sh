#!/bin/bash
# Pair C: unseen (composite_unseen 16 tasks), GPU 6, port 8012
ROLLOUT_ROOT=/mnt/aix24702/robocasa-rollouts/phase2_unseen
CKPT_DIR=/mnt/aix24702/robocasa-openpi/checkpoints/pi05_pretrain_human300/multitask_learning/75000
CONFIG=pi05_pretrain_human300
mkdir -p "$ROLLOUT_ROOT/policy_records"
export CUDA_VISIBLE_DEVICES=6
cd /mnt/aix24702/robocasa-openpi
echo "[server-C] GPU 6, port 8012" | tee -a $ROLLOUT_ROOT/server_C.log
.venv/bin/python scripts/serve_policy.py \
    --record --record-dir "$ROLLOUT_ROOT/policy_records" --port 8012 \
    policy:checkpoint --policy.config "$CONFIG" --policy.dir "$CKPT_DIR" \
    2>&1 | tee -a $ROLLOUT_ROOT/server_C.log
