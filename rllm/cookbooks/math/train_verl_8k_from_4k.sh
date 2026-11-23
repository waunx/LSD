#!/usr/bin/env bash

set -euo pipefail

export PYTHONFAULTHANDLER=1
export PYTHON_BIN=/opt/tiger/rllm_verl_cu124_venv/bin/python
export PROJECT_NAME=math
export EXPERIMENT_NAME=qwen3-4b-base-dapo17k-aime25-26-amc23-pass32-8k-from-4k-aligned-1
export CHECKPOINT_ROOT=/tmp/rllm_checkpoints
export LOG_ROOT=/opt/tiger/rllm/cookbooks/math/logs
export HYDRA_RUN_DIR=/tmp/rllm_hydra/math/qwen3-4b-base-dapo17k-aime25-26-amc23-pass32-8k-from-4k-aligned-1
export WANDB_DIR=/tmp/rllm_wandb
export TOTAL_EPOCHS=10

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

SOURCE_4K_EXPERIMENT="${SOURCE_4K_EXPERIMENT:-qwen3-4b-base-dapo17k-aime25-26-amc23-pass32-4k-aligned-1}"
SOURCE_4K_DIR="${SOURCE_4K_DIR:-/tmp/rllm_checkpoints/math/${SOURCE_4K_EXPERIMENT}}"
SOURCE_4K_STEP_FILE="${SOURCE_4K_STEP_FILE:-${SOURCE_4K_DIR}/latest_checkpointed_iteration.txt}"

if [[ ! -f "$SOURCE_4K_STEP_FILE" ]]; then
  echo "Missing 4k latest checkpoint tracker: $SOURCE_4K_STEP_FILE" >&2
  exit 1
fi

SOURCE_4K_STEP="$(tr -d '[:space:]' < "$SOURCE_4K_STEP_FILE")"
if [[ -z "$SOURCE_4K_STEP" || ! "$SOURCE_4K_STEP" =~ ^[0-9]+$ ]]; then
  echo "Invalid 4k checkpoint step in $SOURCE_4K_STEP_FILE: $SOURCE_4K_STEP" >&2
  exit 1
fi

SOURCE_4K_CKPT="${SOURCE_4K_CKPT:-${SOURCE_4K_DIR}/global_step_${SOURCE_4K_STEP}}"
if [[ ! -f "$SOURCE_4K_CKPT/actor/model_world_size_8_rank_0.pt" ]]; then
  echo "4k checkpoint is missing actor weights: $SOURCE_4K_CKPT" >&2
  exit 1
fi

echo "Resuming 8k training from 4k checkpoint: $SOURCE_4K_CKPT"

bash "$SCRIPT_DIR/train_verl.sh" \
  data.max_response_length=8192 \
  rllm.data.max_response_length=8192 \
  actor_rollout_ref.rollout.max_model_len=12288 \
  actor_rollout_ref.rollout.response_length=8192 \
  actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=16384 \
  actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=16384 \
  rllm.rollout.train.max_tokens=8192 \
  rllm.rollout.val.max_tokens=8192 \
  rllm.rollout.n=8 \
  rllm.rollout.n_val=32 \
  rllm.workflow.n_parallel_tasks=256 \
  rllm.compact_filtering.enable=false \
  rllm.compact_filtering.mask_error=true \
  rllm.compact_filtering.mask_timeout=true \
  rllm.compact_filtering.mask_max_response_length_exceeded=true \
  actor_rollout_ref.actor.use_kl_loss=false \
  actor_rollout_ref.actor.kl_loss_coef=0.0 \
  actor_rollout_ref.model.lora_rank=0 \
  actor_rollout_ref.model.lora_alpha=32 \
  actor_rollout_ref.model.lora.rank=0 \
  actor_rollout_ref.model.lora.alpha=32 \
  actor_rollout_ref.model.lora.merge=true \
  actor_rollout_ref.model.lora_adapter_path=null \
  trainer.default_hdfs_dir=hdfs://haruna/home/byte_data_seed/ssd_hldy/user/wanxu.001/rllm/ckpts/math/qwen3-4b-base-dapo17k-aime25-26-amc23-pass32-8k-from-4k-aligned-1 \
  trainer.save_freq=100 \
  trainer.max_actor_ckpt_to_keep=1 \
  trainer.max_critic_ckpt_to_keep=1 \
  trainer.resume_mode=resume_path \
  trainer.resume_from_path="$SOURCE_4K_CKPT" \
  2>&1 | tee -a /tmp/qwen3-4b-base-dapo17k-aime25-26-amc23-pass32-8k-from-4k-aligned-1_tmux0.log
