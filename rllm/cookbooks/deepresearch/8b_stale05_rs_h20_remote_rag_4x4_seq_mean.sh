#!/usr/bin/env bash
set -euo pipefail

# The HDFS-delivered H20 recipe and its preflight/runtime dependencies live in
# the isolated deployment tree. Keep this cookbook entry as the machine-local
# launcher so the original deployment remains untouched.
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
SOURCE_REPO=${SOURCE_REPO:-/opt/tiger/deepresearch_seqmean_deploy/rllm}
BASE_RECIPE=${SCRIPT_DIR}/8b_stale05_rs_h20_4x4.sh

export PYTHON=${PYTHON:-/tmp/rllm_fully_async_venv_seqmean/bin/python}
export MODEL_PATH=${MODEL_PATH:-/tmp/deepresearch_seqmean_models/Qwen3-8B-Base}
export RLLM_HOME=${RLLM_HOME:-/tmp/deepresearch_seqmean_data/rllm_deepresearch_home}

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}
export NGPUS_TRAINING=${NGPUS_TRAINING:-4}
export NGPUS_ROLLOUT=${NGPUS_ROLLOUT:-4}
export FSDP_SIZE=${FSDP_SIZE:-4}

export MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH:-700}
export MAX_RESPONSE_LENGTH=${MAX_RESPONSE_LENGTH:-20000}
export LOSS_AGG_MODE=${LOSS_AGG_MODE:-token-mean}
export VAL_BEFORE_TRAIN=${VAL_BEFORE_TRAIN:-True}
export TEST_FREQ=${TEST_FREQ:-20}
export RESUME_MODE=${RESUME_MODE:-disable}
export STALENESS_THRESHOLD=${STALENESS_THRESHOLD:-0.5}
export PARTIAL_ROLLOUT=${PARTIAL_ROLLOUT:-True}
export EXPERIMENT_NAME=${EXPERIMENT_NAME:-qwen3_8b_base_cutthebill_stale05_rs_h20_remote_rag_4x4_token_mean_refine_$(hostname -s)}
export CHECKPOINT_ROOT=${CHECKPOINT_ROOT:-/tmp/deepresearch_base_rl_ckpts}
export WANDB_DIR=${WANDB_DIR:-/tmp/deepresearch_base_rl_wandb}
export HF_HOME=${HF_HOME:-/tmp/deepresearch_seqmean_hf}
export RAY_TMPDIR=${RAY_TMPDIR:-/tmp/ray_deepresearch_base}
export TMPDIR=${TMPDIR:-/tmp/rllm_deepresearch_base_tmp}

export DEEPRESEARCH_TRAIN_RAG_URLS=${DEEPRESEARCH_TRAIN_RAG_URLS:-http://[2605:340:cd51:7700:6b39:c6a8:fb27:657c]:19002}
export DEEPRESEARCH_VAL_RAG_URLS=${DEEPRESEARCH_VAL_RAG_URLS:-http://[2605:340:cd51:7700:6b39:c6a8:fb27:657c]:19003}
source "$SCRIPT_DIR/refine_pool.sh"
export DEEPRESEARCH_REFINE_MODEL=${DEEPRESEARCH_REFINE_MODEL:-Qwen/Qwen3-8B}
export DEEPRESEARCH_REFINE_MAX_TOKENS=${DEEPRESEARCH_REFINE_MAX_TOKENS:-8092}
export DEEPRESEARCH_REFINE_MAX_CONCURRENCY=${DEEPRESEARCH_REFINE_MAX_CONCURRENCY:-128}
export DEEPRESEARCH_MAX_TURNS=${DEEPRESEARCH_MAX_TURNS:-48}
export DEEPRESEARCH_USE_REFINE=${DEEPRESEARCH_USE_REFINE:-true}

# The endpoints are checked explicitly by the launch procedure because Python
# proxy bypass parsing is unreliable for bracketed IPv6 literals.
export SKIP_ENDPOINT_CHECK=${SKIP_ENDPOINT_CHECK:-1}

exec bash "${BASE_RECIPE}"
