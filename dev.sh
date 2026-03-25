#!/bin/bash
# Starts a tmux session with 4 windows: frontend, backend, rq worker, and a general shell.

SESSION="earningsiq"
PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"

# Kill existing session if running
tmux kill-session -t "$SESSION" 2>/dev/null

tmux new-session -d -s "$SESSION" -n "frontend" -x 220 -y 50

# Window 1: Frontend
tmux send-keys -t "$SESSION:frontend" "cd \"$PROJECT_DIR/frontend\" && npm run dev" Enter

# Window 2: Backend
tmux new-window -t "$SESSION" -n "backend"
tmux send-keys -t "$SESSION:backend" "cd \"$PROJECT_DIR\" && uv run uvicorn api.main:app --reload" Enter

# Window 3: RQ Worker
tmux new-window -t "$SESSION" -n "rq"
tmux send-keys -t "$SESSION:rq" "cd \"$PROJECT_DIR\" && PYTHONPATH=. uv run rq worker --worker-class rq.SimpleWorker" Enter

# Window 4: Shell
tmux new-window -t "$SESSION" -n "shell"
tmux send-keys -t "$SESSION:shell" "cd \"$PROJECT_DIR\"" Enter

# Attach to session at the shell window
tmux select-window -t "$SESSION:shell"
tmux attach-session -t "$SESSION"
