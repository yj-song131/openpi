#!/bin/bash
# Wait for checkpoint download to finish, then launch server + client in parallel.

CKPT_DIR=/mnt/aix24702/robocasa-openpi/checkpoints/pi05_pretrain_human300/multitask_learning/75000
SCRIPTS_DIR=/mnt/aix24702/robocasa-openpi/scripts

echo "[phase1] Waiting for checkpoint download to complete (need 44 files, 0 incomplete)..."
while true; do
    n_files=$(find "$CKPT_DIR" -type f 2>/dev/null | grep -v ".lock\|.metadata\|incomplete" | wc -l)
    n_incomplete=$(find "$CKPT_DIR" -name "*.incomplete" 2>/dev/null | wc -l)
    dl_proc=$(pgrep -f "snapshot_download" | wc -l)
    echo "  [$(date +%H:%M:%S)] files=$n_files/44, incomplete=$n_incomplete, dl_proc=$dl_proc"
    if [ "$dl_proc" -eq 0 ] && [ "$n_files" -ge 40 ] && [ "$n_incomplete" -eq 0 ]; then
        echo "[phase1] Download complete! Starting server + client."
        break
    fi
    sleep 30
done

# Launch server in background
echo "[phase1] Launching serve_policy (GPU 0)..."
bash "$SCRIPTS_DIR/launch_server_phase1.sh" &
SERVER_PID=$!
echo "[phase1] Server PID: $SERVER_PID"

# Launch client (blocks until all 10 tasks × 50 episodes done)
echo "[phase1] Launching rollout client (GPU 4)..."
bash "$SCRIPTS_DIR/launch_client_phase1.sh"

echo "[phase1] All done. Killing server (PID $SERVER_PID)."
kill $SERVER_PID 2>/dev/null
