#!/usr/bin/env bash
# Two-node Fully Async Base-RL entrypoint:
# one 8-GPU Ray pool for Actor training and one 8-GPU pool for rollout.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/../.." && pwd)"
SOURCE_REPO="${SOURCE_REPO:-/opt/tiger/deepresearch_seqmean_deploy/rllm}"
export PYTHON_BIN="${PYTHON_BIN:-$REPO_ROOT/.runtime/rllm_fully_async_venv_seqmean/bin/python}"
export PYTHON="$PYTHON_BIN"
export PYTHONPATH="$SOURCE_REPO:$REPO_ROOT/vendor/verl:${PYTHONPATH:-}"

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
export DEEPRESEARCH_TRAIN_RAG_URLS="${DEEPRESEARCH_TRAIN_RAG_URLS:-http://[2605:340:cd51:7700:5d78:7385:9158:1a46]:19002}"
export DEEPRESEARCH_VAL_RAG_URLS="${DEEPRESEARCH_VAL_RAG_URLS:-http://[2605:340:cd51:7700:5d78:7385:9158:1a46]:19003}"
export DEEPRESEARCH_REFINE_URLS="http://[2605:340:cd51:4900:2d65:9d6b:b6e4:6020]:40100/v1,http://[2605:340:cd51:4900:2d65:9d6b:b6e4:6020]:40101/v1,http://[2605:340:cd51:4900:2d65:9d6b:b6e4:6020]:40102/v1,http://[2605:340:cd51:4900:2d65:9d6b:b6e4:6020]:40103/v1,http://[2605:340:cd51:4900:2d65:9d6b:b6e4:6020]:40104/v1,http://[2605:340:cd51:4900:2d65:9d6b:b6e4:6020]:40105/v1,http://[2605:340:cd51:4900:2d65:9d6b:b6e4:6020]:40106/v1,http://[2605:340:cd51:4900:2d65:9d6b:b6e4:6020]:40107/v1"
export DEEPRESEARCH_USE_REFINE="${DEEPRESEARCH_USE_REFINE:-true}"
export DEEPRESEARCH_MAX_TURNS="${DEEPRESEARCH_MAX_TURNS:-48}"
export EXPERIMENT_NAME="${EXPERIMENT_NAME:-qwen3_8b_base_cutthebill_stale05_rs_h20_8train_8rollout_token_mean_refine_$(date +%Y%m%d_%H%M%S)}"
export RLLM_VALIDATION_EPISODE_DIR="${RLLM_VALIDATION_EPISODE_DIR:-/mnt/hdfs/__MERLIN_USER_DIR__/rllm/episodes/deepresearch/$EXPERIMENT_NAME}"

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

CONTROL_ROOT="${CONTROL_ROOT:-/mnt/hdfs/__MERLIN_USER_DIR__/rllm/job-control}"
CONTROL_ID="${MERLIN_JOB_ID:-local-two-node-base-rl}"
CONTROL_DIR="$CONTROL_ROOT/$CONTROL_ID"
DONE_FILE="$CONTROL_DIR/done"

if [[ "${MULTINODE_DRY_RUN:-0}" == 1 ]]; then
    printf 'worker=%s nodes=%s GPUs=%s Ray=%s train_pool=8 rollout_pool=8\n' \
        "$WORKER_INDEX" "$WORKER_COUNT" "$((EXPECTED_NODES * GPUS_PER_NODE))" "$HEAD_ADDRESS"
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
while True:
    try:
        ray.init(address="auto", ignore_reinit_error=True)
        gpus = int(ray.cluster_resources().get("GPU", 0))
        nodes = sum(1 for node in ray.nodes() if node.get("Alive"))
        if gpus >= 16 and nodes >= 2:
            print(f"Ray cluster ready: nodes={nodes}, GPUs={gpus}")
            ray.shutdown()
            break
        ray.shutdown()
    except Exception as exc:
        print(f"Waiting for Ray cluster: {exc}")
    if time.time() >= deadline:
        raise RuntimeError("Timed out waiting for the two-node 16-GPU Ray cluster")
    time.sleep(10)
PY

export NGPUS_TRAINING="${NGPUS_TRAINING:-8}"
export NGPUS_ROLLOUT="${NGPUS_ROLLOUT:-8}"
export FSDP_SIZE="${FSDP_SIZE:-4}"
export LOSS_AGG_MODE="${LOSS_AGG_MODE:-token-mean}"
export MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-700}"
export MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-20000}"
export TEST_FREQ="${TEST_FREQ:-20}"
export SAVE_FREQ=50
export ACTOR_PPO_MAX_TOKEN_LEN_PER_GPU="${ACTOR_PPO_MAX_TOKEN_LEN_PER_GPU:-24576}"
export INFER_MAX_TOKEN_LEN="${LOG_PROB_MAX_TOKEN_LEN_PER_GPU:-24576}"
export RESUME_MODE="${RESUME_MODE:-disable}"
export RUN_PREFLIGHT=0
export SKIP_ENDPOINT_CHECK=1
export SKIP_GPU_MEMORY_CHECK=1
export CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-/tmp/deepresearch_base_rl_ckpts}"
export DEFAULT_HDFS_DIR="${DEFAULT_HDFS_DIR:-hdfs://haruna/home/byte_data_seed/ssd_hldy/user/wanxu.001/rllm/ckpts/deepresearch/$EXPERIMENT_NAME}"

set +e
bash "$SCRIPT_DIR/8b_stale05_rs_h20_remote_rag_4x4_seq_mean.sh"
status=$?
set -e
if [[ -n "${WANDB_ARCHIVE_DIR:-}" && -d "${WANDB_DIR:-}" ]]; then
    mkdir -p "$WANDB_ARCHIVE_DIR"
    tar -czf "$WANDB_ARCHIVE_DIR/$EXPERIMENT_NAME.tar.gz" -C "$WANDB_DIR" . || true
fi
printf '%s\n' "$status" > "$DONE_FILE.tmp"
mv "$DONE_FILE.tmp" "$DONE_FILE"
ray_cli stop --force >/dev/null 2>&1 || true
exit "$status"
