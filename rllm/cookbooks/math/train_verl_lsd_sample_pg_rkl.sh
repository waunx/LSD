#!/usr/bin/env bash
# LSD ablation: sampled-action policy-gradient reverse KL.
# This script is intentionally self-contained.

set -euo pipefail

unset ROCR_VISIBLE_DEVICES 2>/dev/null || true
unset PYTORCH_CUDA_ALLOC_CONF 2>/dev/null || true

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
RLLM_REPO_DIR="$(cd -- "$SCRIPT_DIR/../.." && pwd)"
RLLM_GATEWAY_DIR="$RLLM_REPO_DIR/rllm-model-gateway/src"
export RLLM_HOME="${RLLM_HOME:-/opt/tiger/rllm_home}"
export PYTHONPATH="$RLLM_REPO_DIR:$RLLM_GATEWAY_DIR:${PYTHONPATH:-}"

LOCAL_ENV_FILE="$SCRIPT_DIR/.env.local"
if [[ -f "$LOCAL_ENV_FILE" ]]; then
    set -a
    # shellcheck disable=SC1090
    source "$LOCAL_ENV_FILE"
    set +a
fi

SAFE_NO_PROXY="${RLLM_NO_PROXY:-localhost,127.0.0.1,.byted.org,byted.org,.bytedance.net,bytedance.net}"
export no_proxy="$SAFE_NO_PROXY"
export NO_PROXY="$SAFE_NO_PROXY"

DEFAULT_PYTHON=/opt/tiger/rllm_verl_cu124_venv/bin/python
if [[ -x "$DEFAULT_PYTHON" ]]; then
    PYTHON_BIN="${PYTHON_BIN:-$DEFAULT_PYTHON}"
elif command -v python3 >/dev/null 2>&1; then
    PYTHON_BIN="${PYTHON_BIN:-python3}"
else
    PYTHON_BIN="${PYTHON_BIN:-python}"
fi

if [[ -n "${WANDB_API_KEY:-}" ]]; then
    "$PYTHON_BIN" -m wandb login "$WANDB_API_KEY"
fi


"$PYTHON_BIN" - <<'PY'
import sys

if sys.version_info[0] < 3:
    raise SystemExit("train_verl_lsd_sample_pg_rkl.sh requires Python 3. Set PYTHON_BIN to a Python 3 interpreter.")
PY

NVIDIA_LIBS="$("$PYTHON_BIN" - <<'PY'
import os
import site

libs = []
roots = []
if hasattr(site, "getsitepackages"):
    roots.extend(site.getsitepackages())
if hasattr(site, "getusersitepackages"):
    roots.append(site.getusersitepackages())

for root in roots:
    nvidia_root = os.path.join(root, "nvidia")
    if not os.path.isdir(nvidia_root):
        continue
    for name in os.listdir(nvidia_root):
        lib_path = os.path.join(nvidia_root, name, "lib")
        if os.path.isdir(lib_path):
            libs.append(lib_path)
print(":".join(libs))
PY
)"
if [[ -n "$NVIDIA_LIBS" ]]; then
    export LD_LIBRARY_PATH="$NVIDIA_LIBS:${LD_LIBRARY_PATH:-}"
fi

MODEL_PATH="${MODEL_PATH:-/opt/tiger/models/Qwen3-4B-Base}"
PROJECT_NAME="${PROJECT_NAME:-math}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-qwen3-4b-base-dapo17k-aime25-26-amc23-pass32-lsd-sample-pg-rkl-thr0.875-coef0.1}"
CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-/tmp/rllm_checkpoints}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-$CHECKPOINT_ROOT/$PROJECT_NAME/$EXPERIMENT_NAME}"
LOG_ROOT="${LOG_ROOT:-/tmp/rllm_logs}"
LOG_DIR="${LOG_DIR:-$LOG_ROOT/$PROJECT_NAME/$EXPERIMENT_NAME}"
HYDRA_RUN_DIR="${HYDRA_RUN_DIR:-/tmp/rllm_hydra/$PROJECT_NAME/$EXPERIMENT_NAME}"
export WANDB_DIR="${WANDB_DIR:-/tmp/rllm_wandb}"
export WANDB_MODE="${WANDB_MODE:-online}"
mkdir -p "$CHECKPOINT_DIR" "$LOG_DIR" "$HYDRA_RUN_DIR" "$WANDB_DIR"

export RLLM_MATH_TRAIN_DATASET="${RLLM_MATH_TRAIN_DATASET:-dapo_math_17k}"
export RLLM_MATH_VAL_DATASET="${RLLM_MATH_VAL_DATASET:-aime25_26_amc23}"

LSD_MAPPING_PATH="${LSD_MAPPING_PATH:-/opt/tiger/LSD/outputs/0714-qwen3_4b-base-dapo17k-dedup-train-eval-16k/postprocess/dapo-math-17k-dedup-base-eval-8x-acc-map.jsonl}"
LSD_OPD_PG_ADV_CLIP="${LSD_OPD_PG_ADV_CLIP:-10.0}"
TOTAL_EPOCHS="${TOTAL_EPOCHS:-10}"

cd "$SCRIPT_DIR"

"$PYTHON_BIN" -u train.py \
    rllm/backend=verl \
    algorithm.adv_estimator=grpo \
    algorithm.norm_adv_by_std_in_grpo=true \
    rllm.algorithm.use_rllm=true \
    rllm.lsd.enable=true \
    rllm.lsd.routing_mode=fixed_base \
    rllm.lsd.teacher_mode=fixed_base \
    rllm.lsd.mapping_path="$LSD_MAPPING_PATH" \
    rllm.lsd.easy_threshold=0.875 \
    rllm.lsd.lsd_coef=0.1 \
    rllm.lsd.easy_rl_coef=0.0 \
    rllm.lsd.loss_agg_mode=seq_mean_token_mean \
    rllm.lsd.opd_loss_mode=sample_token \
    rllm.lsd.opd_optimization_mode=policy_gradient \
    rllm.lsd.opd_pg_adv_clip="$LSD_OPD_PG_ADV_CLIP" \
    rllm.lsd.fail_on_missing=true \
    data.train_batch_size=32 \
    data.val_batch_size=500 \
    data.max_prompt_length=4096 \
    data.max_response_length=4096 \
    +model.name="$MODEL_PATH" \
    actor_rollout_ref.model.path="$MODEL_PATH" \
    actor_rollout_ref.model.lora_rank=0 \
    actor_rollout_ref.model.lora_alpha=16 \
    actor_rollout_ref.model.lora.rank=32 \
    actor_rollout_ref.model.lora.alpha=32 \
    actor_rollout_ref.model.lora.merge=true \
    actor_rollout_ref.hybrid_engine=True \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.ppo_mini_batch_size=64 \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=16384 \
    actor_rollout_ref.actor.fsdp_config.param_offload=true \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=true \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.loss_agg_mode=seq-mean-token-mean \
    actor_rollout_ref.actor.clip_ratio_low=0.2 \
    actor_rollout_ref.actor.clip_ratio_high=0.28 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.mode=async \
    actor_rollout_ref.rollout.enforce_eager=False \
    actor_rollout_ref.rollout.max_model_len=8192 \
    actor_rollout_ref.rollout.response_length=4096 \
    actor_rollout_ref.rollout.temperature=0.6 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.8 \
    actor_rollout_ref.rollout.n=8 \
    actor_rollout_ref.rollout.val_kwargs.n=32 \
    actor_rollout_ref.rollout.val_kwargs.temperature=0.6 \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
    trainer.logger="['console','tracking']" \
    trainer.project_name="$PROJECT_NAME" \
    trainer.experiment_name="$EXPERIMENT_NAME" \
    trainer.val_before_train=true \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.save_freq=100 \
    trainer.test_freq=5 \
    trainer.total_epochs="$TOTAL_EPOCHS" \
    trainer.default_local_dir="$CHECKPOINT_DIR" \
    trainer.default_hdfs_dir=null \
    trainer.resume_mode=disable \
    rllm.episode_logging.log_episodes=true \
    rllm.episode_logging.episode_log_dir="$LOG_DIR" \
    rllm.episode_logging.modes="['val']" \
    rllm.episode_logging.summary_only=true \
    rllm.episode_logging.include_text=false \
    rllm.episode_logging.file_format=jsonl \
    hydra.run.dir="$HYDRA_RUN_DIR" \
    "$@"
