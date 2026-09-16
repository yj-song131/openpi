#!/usr/bin/env bash
# Wait for TurnOnStove (task 0) to have 25 episodes, then run SAFE training test.
set -euo pipefail

SAFE_ROOT=/mnt/aix24702/SAFE
DATA_PATH=/mnt/aix24702/robocasa-rollouts/phase1_eval5x5
ENV_RECORDS="${DATA_PATH}/env_records"
TARGET_TASK_ID=0
TARGET_EP_COUNT=25
GPU_ID=7

echo "[$(date)] Waiting for task ${TARGET_TASK_ID} to reach ${TARGET_EP_COUNT} episodes..."

while true; do
    count=$(.venv/bin/python3 -c "
import pickle, glob
files = glob.glob('${ENV_RECORDS}/*.pkl')
n = sum(1 for f in files if pickle.load(open(f,'rb'))['task_id'] == ${TARGET_TASK_ID})
print(n)
" 2>/dev/null || echo 0)

    echo "[$(date)] task ${TARGET_TASK_ID}: ${count}/${TARGET_EP_COUNT} episodes"

    if [ "${count}" -ge "${TARGET_EP_COUNT}" ]; then
        echo "[$(date)] Data ready! Starting SAFE training..."
        break
    fi
    sleep 60
done

cd "${SAFE_ROOT}"
source setup_envs.bash

CUDA_VISIBLE_DEVICES=${GPU_ID} .venv/bin/python -m failure_prob.train \
    dataset=pizero_robocasa365 \
    model=lstm \
    train.seed=0 \
    train.eval_save_ckpt=False \
    train.wandb_group_name=pi0_robocasa365_safe \
    train.exp_suffix=seen_only_test_run

echo "[$(date)] SAFE training done."
