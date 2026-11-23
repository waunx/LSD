#!/usr/bin/env bash
# Two-node Merlin entrypoint for EMA-LSD. Worker 0 starts the Ray head and
# launches the driver; worker 1 joins the cluster and provides its GPUs.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/../.." && pwd)"
export DEEPRESEARCH_REFINE_URLS="http://[2605:340:cd51:4900:8b45:1059:2a1:e33f]:40003/v1,http://[2605:340:cd51:4900:8b45:1059:2a1:e33f]:40004/v1,http://[2605:340:cd51:4900:8b45:1059:2a1:e33f]:40005/v1,http://[2605:340:cd51:4900:8b45:1059:2a1:e33f]:40006/v1,http://[2605:340:cd51:4900:8b45:1059:2a1:e33f]:40007/v1,http://[2605:340:cd51:4900:8b45:1059:2a1:e33f]:40008/v1,http://[2605:340:cd51:4900:8b45:1059:2a1:e33f]:40009/v1,http://[2605:340:cd51:4900:8b45:1059:2a1:e33f]:40011/v1"
export PYTHONPATH="$REPO_ROOT:$REPO_ROOT/rllm-model-gateway/src:$REPO_ROOT/vendor/verl:${PYTHONPATH:-}"
export PYTHON_BIN="${PYTHON_BIN:-/opt/tiger/rllm/.runtime/rllm_fully_async_venv_seqmean/bin/python}"
EXPECTED_NODES="${EXPECTED_NODES:-2}"
GPUS_PER_NODE="${ARNOLD_WORKER_GPU:-8}"
WORKER_COUNT="${ARNOLD_WORKER_NUM:-1}"

if [[ -n "${ARNOLD_ID:-}" ]]; then
    WORKER_INDEX="$ARNOLD_ID"
elif [[ "${MY_POD_NAME:-}" =~ worker-([0-9]+)$ ]]; then
    WORKER_INDEX="${BASH_REMATCH[1]}"
else
    WORKER_INDEX=0
fi

if [[ "$WORKER_COUNT" -ne "$EXPECTED_NODES" ]]; then
    echo "Expected ARNOLD_WORKER_NUM=$EXPECTED_NODES, got $WORKER_COUNT" >&2
    exit 1
fi

export MODEL_PATH="${MODEL_PATH:-/mnt/hdfs/__MERLIN_USER_DIR__/lsd/model/Qwen3-8B-Base}"
export RLLM_HOME="${RLLM_HOME:-/mnt/hdfs/__MERLIN_USER_DIR__/lsd/runtime/rllm_deepresearch_home}"
ray_cli() { "$PYTHON_BIN" -m ray.scripts.scripts "$@"; }
HEAD_HOST="${ARNOLD_WORKER_0_HOST:-${BYTED_RAY_POD_IP:-}}"
HEAD_PORT="${METIS_WORKER_0_RESERVED_0_PORT:-6379}"
NODE_HOST="${BYTED_HOST_IPV6:-${ARNOLD_WORKER_0_HOST:-}}"
if [[ "$HEAD_HOST" == *:* ]]; then
    HEAD_ADDRESS="[$HEAD_HOST]:$HEAD_PORT"
else
    HEAD_ADDRESS="$HEAD_HOST:$HEAD_PORT"
fi
export RAY_ADDRESS="$HEAD_ADDRESS"
export DEEPRESEARCH_TRAIN_RAG_URLS="${DEEPRESEARCH_TRAIN_RAG_URLS:-http://[2605:340:cd51:7700:5d78:7385:9158:1a46]:19002}"
export DEEPRESEARCH_VAL_RAG_URLS="${DEEPRESEARCH_VAL_RAG_URLS:-http://[2605:340:cd51:7700:5d78:7385:9158:1a46]:19003}"
export DEEPRESEARCH_USE_REFINE="${DEEPRESEARCH_USE_REFINE:-true}"
CONTROL_ROOT="${CONTROL_ROOT:-/mnt/hdfs/__MERLIN_USER_DIR__/rllm/job-control}"
CONTROL_ID="${MERLIN_JOB_ID:-local-two-node-lsd}"
CONTROL_DIR="$CONTROL_ROOT/$CONTROL_ID"
DONE_FILE="$CONTROL_DIR/done"

if [[ "${MULTINODE_DRY_RUN:-0}" == "1" ]]; then
    printf 'worker_index=%s worker_count=%s expected_gpus=%s ray_address=%s node_host=%s\n' \
        "$WORKER_INDEX" "$WORKER_COUNT" "$((EXPECTED_NODES * GPUS_PER_NODE))" "$RAY_ADDRESS" "$NODE_HOST"
    exit 0
fi

if [[ -z "$HEAD_HOST" || -z "$NODE_HOST" ]]; then
    echo "Missing Ray host environment: HEAD_HOST=$HEAD_HOST NODE_HOST=$NODE_HOST" >&2
    exit 1
fi

mkdir -p "$CONTROL_DIR"
ray_cli stop --force >/dev/null 2>&1 || true

if [[ "$WORKER_INDEX" -ne 0 ]]; then
    echo "Worker $WORKER_INDEX joining Ray head at $HEAD_ADDRESS."
    until ray_cli start \
        --address="$HEAD_ADDRESS" \
        --node-ip-address="$NODE_HOST" \
        --num-gpus="$GPUS_PER_NODE" \
        --min-worker-port=20000 \
        --max-worker-port=29999 \
        --disable-usage-stats; do
        sleep 5
    done
    while [[ ! -f "$DONE_FILE" ]]; do
        sleep 30
    done
    status="$(cat "$DONE_FILE")"
    ray_cli stop --force >/dev/null 2>&1 || true
    exit "$status"
fi

rm -f "$DONE_FILE"
ray_cli start \
    --head \
    --node-ip-address="$NODE_HOST" \
    --port="$HEAD_PORT" \
    --num-gpus="$GPUS_PER_NODE" \
    --min-worker-port=20000 \
    --max-worker-port=29999 \
    --include-dashboard=false \
    --disable-usage-stats

"$PYTHON_BIN" - <<PY
import time
import ray

deadline = time.time() + 900
expected_gpus = ${EXPECTED_NODES} * ${GPUS_PER_NODE}
while True:
    try:
        ray.init(address="auto", ignore_reinit_error=True)
        gpus = int(ray.cluster_resources().get("GPU", 0))
        nodes = sum(1 for node in ray.nodes() if node.get("Alive"))
        if gpus >= expected_gpus and nodes >= ${EXPECTED_NODES}:
            print(f"Ray cluster ready: nodes={nodes}, GPUs={gpus}")
            ray.shutdown()
            break
        ray.shutdown()
    except Exception as exc:
        print(f"Waiting for Ray cluster: {exc}")
    if time.time() >= deadline:
        raise RuntimeError(
            f"Timed out waiting for {expected_gpus} GPUs across ${EXPECTED_NODES} nodes"
        )
    time.sleep(10)
PY

export EXPERIMENT_NAME="${EXPERIMENT_NAME:-qwen3_8b_base_cutthebill_lsd_tau1_hl4_token_mean_2node_hybrid_$(date +%Y%m%d_%H%M%S)}"
export RESUME_MODE="${RESUME_MODE:-disable}"
export TEST_FREQ="${TEST_FREQ:-20}"
export SAVE_FREQ=50
export SKIP_WANDB_LOGIN="${SKIP_WANDB_LOGIN:-1}"
if [[ -z "${WANDB_API_KEY:-}" ]]; then
    export WANDB_MODE="${WANDB_MODE:-offline}"
    export WANDB_DIR="${WANDB_DIR:-/tmp/rllm-wandb}"
    WANDB_ARCHIVE_DIR="${WANDB_ARCHIVE_DIR:-/mnt/hdfs/__MERLIN_USER_DIR__/rllm/wandb-offline}"
    mkdir -p "$WANDB_DIR" "$WANDB_ARCHIVE_DIR"
fi

set +e
bash "$SCRIPT_DIR/train_verl_lsd_online_ema_direct_fkl_deepresearch.sh" \
    trainer.nnodes="$EXPECTED_NODES" \
    trainer.n_gpus_per_node="$GPUS_PER_NODE" \
    "$@"
status=$?
set -e
if [[ "${WANDB_MODE:-}" == "offline" && -d "${WANDB_DIR:-}" ]]; then
    tar -czf "$WANDB_ARCHIVE_DIR/$EXPERIMENT_NAME.tar.gz" -C "$WANDB_DIR" . || true
fi
printf '%s\n' "$status" > "$DONE_FILE.tmp"
mv "$DONE_FILE.tmp" "$DONE_FILE"
ray_cli stop --force >/dev/null 2>&1 || true
exit "$status"
