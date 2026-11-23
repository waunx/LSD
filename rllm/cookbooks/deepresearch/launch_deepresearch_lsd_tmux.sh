#!/usr/bin/env bash
set -euo pipefail

SESSION="${SESSION:-deepresearch_lsd_train}"
SCRIPT=/opt/tiger/rllm/cookbooks/math/train_verl_lsd_online_ema_sample_pg_rkl.sh
PYTHON_BIN="${PYTHON_BIN:-/tmp/rllm_fully_async_venv_seqmean/bin/python}"

if tmux list-sessions -F '#S' 2>/dev/null | awk -v s="$SESSION" '$0==s{found=1} END{exit found?0:1}'; then
    echo "Session already exists; refusing to overwrite: $SESSION" >&2
    exit 1
fi

if ! "$PYTHON_BIN" -m wandb login --verify >/dev/null 2>&1; then
    echo "W&B is not logged in for $PYTHON_BIN. Run: tmux attach -t deepresearch_lsd_wandb_login" >&2
    exit 1
fi

busy=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null | sed '/^$/d' | wc -l)
if [[ "$busy" != "0" ]]; then
    echo "GPUs are busy ($busy compute process entries). Refusing to start LSD." >&2
    nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits >&2 || true
    exit 1
fi

tmux new-session -d -s "$SESSION" \
  "cd /opt/tiger/rllm/cookbooks/math; \
   export RLLM_TASK=deepresearch; \
   export PYTHON_BIN='$PYTHON_BIN'; \
   export RLLM_HOME='/tmp/deepresearch_seqmean_data/rllm_deepresearch_home'; \
   export MODEL_PATH='/tmp/deepresearch_seqmean_models/Qwen3-8B'; \
   export WANDB_DIR='/tmp/deepresearch_lsd_wandb'; \
   exec bash '$SCRIPT'"

tmux capture-pane -t "$SESSION" -p -S -100
