#!/bin/bash
# Server A restarted for unseen tasks: GPU 0, port 8010, writes to phase2_unseen
ROLLOUT_ROOT=/mnt/aix24702/robocasa-rollouts/phase2_unseen
CKPT_DIR=/mnt/aix24702/robocasa-openpi/checkpoints/pi05_pretrain_human300/multitask_learning/75000
CONFIG=pi05_pretrain_human300
mkdir -p "$ROLLOUT_ROOT/policy_records"
export CUDA_VISIBLE_DEVICES=0
cd /mnt/aix24702/robocasa-openpi
echo "[server-A-unseen] GPU 0, port 8010" | tee -a $ROLLOUT_ROOT/server_A_unseen.log
.venv/bin/python scripts/serve_policy.py \
    --record --record-dir "$ROLLOUT_ROOT/policy_records" --port 8010 \
    policy:checkpoint --policy.config "$CONFIG" --policy.dir "$CKPT_DIR" \
    2>&1 | tee -a $ROLLOUT_ROOT/server_A_unseen.log
