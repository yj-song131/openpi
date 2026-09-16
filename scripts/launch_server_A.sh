#!/bin/bash
# Pair A: seen tasks 0-4, GPU 0, port 8010
ROLLOUT_ROOT=/mnt/aix24702/robocasa-rollouts/phase1_eval5x5
CKPT_DIR=/mnt/aix24702/robocasa-openpi/checkpoints/pi05_pretrain_human300/multitask_learning/75000
CONFIG=pi05_pretrain_human300
mkdir -p "$ROLLOUT_ROOT/policy_records"
export CUDA_VISIBLE_DEVICES=0
cd /mnt/aix24702/robocasa-openpi
echo "[server-A] GPU 0, port 8010" | tee -a $ROLLOUT_ROOT/server_A.log
.venv/bin/python scripts/serve_policy.py \
    --record --record-dir "$ROLLOUT_ROOT/policy_records" --port 8010 \
    policy:checkpoint --policy.config "$CONFIG" --policy.dir "$CKPT_DIR" \
    2>&1 | tee -a $ROLLOUT_ROOT/server_A.log
