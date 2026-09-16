#!/bin/bash
# Waits for Client A (seen tasks 0-4) to finish, then:
#   1. Kills Server A and restarts it pointing at phase2_unseen
#   2. Stops current Client C (single), restarts as C1 + C2 in parallel
set -euo pipefail

ROLLOUT_ROOT_SEEN=/mnt/aix24702/robocasa-rollouts/phase1_eval5x5
SCRIPTS=/mnt/aix24702/robocasa-openpi/scripts

echo "[$(date)] Waiting for Client A (pi0-client-A) to finish..."
while tmux has-session -t pi0-client-A 2>/dev/null; do
    count=$(ls "$ROLLOUT_ROOT_SEEN/env_records/" 2>/dev/null | grep "^task[01234]--" | wc -l)
    echo "[$(date)] seen tasks 0-4: ${count}/125 ep"
    sleep 120
done

echo "[$(date)] Client A finished. Starting switchover..."

# 1. Kill current Client C (it'll be replaced by C1 + C2)
echo "[$(date)] Stopping pi0-client-C..."
tmux kill-session -t pi0-client-C 2>/dev/null || true
sleep 5

# 2. Kill Server A and restart for unseen
echo "[$(date)] Restarting Server A for unseen tasks..."
tmux kill-session -t pi0-server-A 2>/dev/null || true
sleep 10  # wait for GPU 0 to free up

tmux new-session -d -s pi0-server-A-unseen -x 220 -y 50
tmux send-keys -t pi0-server-A-unseen "bash ${SCRIPTS}/launch_server_A_unseen.sh" Enter

echo "[$(date)] Waiting 60s for Server A (unseen) to load model..."
sleep 60

# 3. Launch Client C1 (tasks 0-7, port 8012) — continues from where old C left off
echo "[$(date)] Launching Client C1 (unseen tasks 0-7, port 8012)..."
tmux new-session -d -s pi0-client-C1 -x 220 -y 50
tmux send-keys -t pi0-client-C1 "bash ${SCRIPTS}/launch_client_C1.sh" Enter

# 4. Launch Client C2 (tasks 8-15, port 8010)
echo "[$(date)] Launching Client C2 (unseen tasks 8-15, port 8010)..."
tmux new-session -d -s pi0-client-C2 -x 220 -y 50
tmux send-keys -t pi0-client-C2 "bash ${SCRIPTS}/launch_client_C2.sh" Enter

echo "[$(date)] Switchover complete. Running: pi0-server-A-unseen, pi0-client-C1, pi0-client-C2"
