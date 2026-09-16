#!/bin/bash
# Pair B: seen tasks 5-9, GPU 4, port 8011
ROLLOUT_ROOT=/mnt/aix24702/robocasa-rollouts/phase1_eval5x5
CKPT_DIR=/mnt/aix24702/robocasa-openpi/checkpoints/pi05_pretrain_human300/multitask_learning/75000
CONFIG=pi05_pretrain_human300
mkdir -p "$ROLLOUT_ROOT/policy_records"
export CUDA_VISIBLE_DEVICES=5
cd /mnt/aix24702/robocasa-openpi
echo "[server-B] GPU 4, port 8011" | tee -a $ROLLOUT_ROOT/server_B.log
.venv/bin/python scripts/serve_policy.py \
    --record --record-dir "$ROLLOUT_ROOT/policy_records" --port 8011 \
    policy:checkpoint --policy.config "$CONFIG" --policy.dir "$CKPT_DIR" \
    2>&1 | tee -a $ROLLOUT_ROOT/server_B.log
