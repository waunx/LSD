#!/usr/bin/env bash
# DeepResearch regular-VERL LSD run using online-group EMA direct top-k forward KL.
# This intentionally does not embed a W&B key. If WANDB_API_KEY is present in
# the environment it is used for login and then unset.

set -euo pipefail

unset ROCR_VISIBLE_DEVICES 2>/dev/null || true

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
RLLM_REPO_DIR="$(cd -- "$SCRIPT_DIR/../.." && pwd)"
RLLM_GATEWAY_DIR="$RLLM_REPO_DIR/rllm-model-gateway/src"
source "$SCRIPT_DIR/refine_pool.sh"

export PATH="/home/tiger/.local/bin:${PATH}"
if command -v rg >/dev/null 2>&1; then
    eval "$(rg -n '^[[:space:]]*export[[:space:]]+(http_proxy|https_proxy|HTTP_PROXY|HTTPS_PROXY|all_proxy|ALL_PROXY|no_proxy|NO_PROXY)=' ~/.bashrc | sed 's/^[0-9]*://')"
fi

export PYTHONPATH="$RLLM_REPO_DIR:$RLLM_GATEWAY_DIR:${PYTHONPATH:-}"
export RLLM_HOME="${RLLM_HOME:-/tmp/deepresearch_seqmean_data/rllm_deepresearch_home}"
export DEEPRESEARCH_TRAIN_RAG_URLS="${DEEPRESEARCH_TRAIN_RAG_URLS:-http://[2605:340:cd51:7700:7739:4273:3005:67c1]:19002}"
export DEEPRESEARCH_VAL_RAG_URLS="${DEEPRESEARCH_VAL_RAG_URLS:-http://[2605:340:cd51:7700:7739:4273:3005:67c1]:19003}"
export DEEPRESEARCH_MAX_TURNS="${DEEPRESEARCH_MAX_TURNS:-48}"
export DEEPRESEARCH_USE_REFINE="${DEEPRESEARCH_USE_REFINE:-true}"
export RLLM_DEFER_VERL_WORKER_PATCHES="${RLLM_DEFER_VERL_WORKER_PATCHES:-1}"
export RLLM_FULLY_ASYNC_METRIC_COMPAT="${RLLM_FULLY_ASYNC_METRIC_COMPAT:-1}"
export RLLM_WANDB_FULLY_ASYNC_KEYSET="${RLLM_WANDB_FULLY_ASYNC_KEYSET:-1}"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:False"
export PYTORCH_ALLOC_CONF="expandable_segments:False"
export WANDB_API_KEY="${WANDB_API_KEY:-}"

PYTHON_BIN="${PYTHON_BIN:-/tmp/rllm_fully_async_venv_seqmean/bin/python}"

PY_SITE_PACKAGES="$($PYTHON_BIN - <<'PY'
import site
print(site.getsitepackages()[0])
PY
)"
TORCH_LIB_DIR="$PY_SITE_PACKAGES/torch/lib"
NVIDIA_LIB_DIRS="$(find "$PY_SITE_PACKAGES/nvidia" -maxdepth 2 -type d -name lib 2>/dev/null | paste -sd: -)"
export LD_LIBRARY_PATH="$TORCH_LIB_DIR:${NVIDIA_LIB_DIRS}:${LD_LIBRARY_PATH:-}"

MODEL_PATH="${MODEL_PATH:-/tmp/deepresearch_seqmean_models/Qwen3-8B-Base}"
PROJECT_NAME="${PROJECT_NAME:-deepresearch}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-qwen3_8b_base_cutthebill_lsd_online_ema_direct_fkl_coef1_tau1_hl4_token_mean_base_numerics_$(hostname -s)}"
CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-/tmp/deepresearch_lsd_ckpts}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-$CHECKPOINT_ROOT/$PROJECT_NAME/$EXPERIMENT_NAME}"
HDFS_CHECKPOINT_ROOT="${HDFS_CHECKPOINT_ROOT:-hdfs://haruna/home/byte_data_seed/ssd_hldy/user/wanxu.001/rllm/ckpts}"
HDFS_CHECKPOINT_DIR="${HDFS_CHECKPOINT_DIR:-$HDFS_CHECKPOINT_ROOT/$PROJECT_NAME/$EXPERIMENT_NAME}"
MAX_ACTOR_CKPT_TO_KEEP="${MAX_ACTOR_CKPT_TO_KEEP:-1}"
MAX_CRITIC_CKPT_TO_KEEP="${MAX_CRITIC_CKPT_TO_KEEP:-1}"
LOG_ROOT="${LOG_ROOT:-/tmp/deepresearch_lsd_logs}"
LOG_DIR="${LOG_DIR:-$LOG_ROOT/$PROJECT_NAME/$EXPERIMENT_NAME}"
HYDRA_RUN_DIR="${HYDRA_RUN_DIR:-/tmp/deepresearch_lsd_hydra/$PROJECT_NAME/$EXPERIMENT_NAME}"
export WANDB_DIR="${WANDB_DIR:-/tmp/deepresearch_lsd_wandb}"
export WANDB_MODE="${WANDB_MODE:-online}"
mkdir -p "$CHECKPOINT_DIR" "$LOG_DIR" "$HYDRA_RUN_DIR" "$WANDB_DIR"

if [[ "${REQUIRE_BASE_MODEL:-1}" == "1" ]]; then
    if [[ ! -f "$MODEL_PATH/README.md" ]] || ! rg -q '^# Qwen3-8B-Base$' "$MODEL_PATH/README.md"; then
        echo "Expected official Qwen3-8B-Base at MODEL_PATH=$MODEL_PATH" >&2
        exit 1
    fi
    shopt -s nullglob
    model_shards=("$MODEL_PATH"/model-*.safetensors)
    shopt -u nullglob
    if [[ "${#model_shards[@]}" -ne 5 ]]; then
        echo "Qwen3-8B-Base is incomplete: expected 5 model shards at $MODEL_PATH" >&2
        exit 1
    fi
fi

# #region debug-point A:launch-env
source /opt/tiger/rllm/.dbg/deepresearch-lsd-run.env 2>/dev/null || { DEBUG_SERVER_URL="http://127.0.0.1:7777/event"; DEBUG_SESSION_ID="deepresearch-lsd-run"; }; curl -sS -m 2 -X POST "$DEBUG_SERVER_URL" -H 'Content-Type: application/json' -d "{\"sessionId\":\"$DEBUG_SESSION_ID\",\"runId\":\"pre-fix\",\"hypothesisId\":\"A\",\"location\":\"train_verl_lsd_online_ema_direct_fkl_deepresearch.sh:launch\",\"msg\":\"[DEBUG] launch environment before wandb/login\",\"data\":{\"cuda_visible_devices\":\"${CUDA_VISIBLE_DEVICES-}\",\"nvidia_visible_devices\":\"${NVIDIA_VISIBLE_DEVICES-}\",\"ray_no_set_cuda\":\"${RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES-}\",\"rllm_exclude\":\"${RLLM_EXCLUDE-}\",\"python_bin\":\"$PYTHON_BIN\",\"model_path\":\"$MODEL_PATH\",\"dry_run\":\"${DRY_RUN:-0}\"}}" >/dev/null 2>&1 || true
# #endregion

if [[ "${SKIP_WANDB_LOGIN:-0}" != "1" && -n "${WANDB_API_KEY:-}" ]]; then
    "$PYTHON_BIN" -m wandb login "$WANDB_API_KEY"
    unset WANDB_API_KEY
fi

if [[ ! -x "$PYTHON_BIN" ]]; then
    echo "Python not found or not executable: $PYTHON_BIN" >&2
    exit 1
fi

if [[ "${SKIP_WANDB_LOGIN:-0}" != "1" && "${DRY_RUN:-0}" != "1" && "${WANDB_MODE:-online}" != "offline" ]]; then
    if ! "$PYTHON_BIN" -m wandb login --verify >/dev/null 2>&1; then
        echo "WANDB login is not configured. Export WANDB_API_KEY for this command, or run wandb login in $PYTHON_BIN environment, or set WANDB_MODE=offline." >&2
        exit 1
    fi
fi

# DeepResearch official registry names/splits.
export RLLM_DEEPRESEARCH_TRAIN_DATASET="${RLLM_DEEPRESEARCH_TRAIN_DATASET:-cutthebill}"
export RLLM_DEEPRESEARCH_TRAIN_SPLIT="${RLLM_DEEPRESEARCH_TRAIN_SPLIT:-train}"
export RLLM_DEEPRESEARCH_VAL_DATASET="${RLLM_DEEPRESEARCH_VAL_DATASET:-browsecomp-plus}"
export RLLM_DEEPRESEARCH_VAL_SPLIT="${RLLM_DEEPRESEARCH_VAL_SPLIT:-test}"

LSD_ONLINE_GROUP_THRESHOLD="${LSD_ONLINE_GROUP_THRESHOLD:-1.0}"
LSD_EXPECTED_GROUP_SIZE="${LSD_EXPECTED_GROUP_SIZE:-8}"
LSD_EMA_HALF_LIFE_UPDATES="${LSD_EMA_HALF_LIFE_UPDATES:-4}"
LSD_EMA_UPDATE_INTERVAL="${LSD_EMA_UPDATE_INTERVAL:-1}"
LSD_EMA_ACCUM_DTYPE="${LSD_EMA_ACCUM_DTYPE:-fp32}"
LSD_OPD_TOPK="${LSD_OPD_TOPK:-128}"
LSD_OPD_DIRECT_LOSS_CLIP="${LSD_OPD_DIRECT_LOSS_CLIP:-null}"
LSD_OPD_PG_ADV_CLIP="${LSD_OPD_PG_ADV_CLIP:-10.0}"
RESUME_MODE="${RESUME_MODE:-disable}"
TOTAL_EPOCHS="${TOTAL_EPOCHS:-10}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-256}"
ACTOR_ULYSSES_SEQUENCE_PARALLEL_SIZE="${ACTOR_ULYSSES_SEQUENCE_PARALLEL_SIZE:-1}"
VAL_BEFORE_TRAIN="${VAL_BEFORE_TRAIN:-True}"
TEST_FREQ="${TEST_FREQ:-20}"
SAVE_FREQ="${SAVE_FREQ:-50}"
MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-700}"
MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-20000}"
MAX_TOTAL_LENGTH="${MAX_TOTAL_LENGTH:-$((MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH))}"
ACTOR_PPO_MAX_TOKEN_LEN_PER_GPU="${ACTOR_PPO_MAX_TOKEN_LEN_PER_GPU:-$MAX_TOTAL_LENGTH}"
ROLLOUT_MAX_NUM_BATCHED_TOKENS="${ROLLOUT_MAX_NUM_BATCHED_TOKENS:-$MAX_TOTAL_LENGTH}"
ROLLOUT_MAX_MODEL_LEN="${ROLLOUT_MAX_MODEL_LEN:-$MAX_TOTAL_LENGTH}"
ROLLOUT_MAX_NUM_SEQS=128
LOG_PROB_MAX_TOKEN_LEN_PER_GPU="${LOG_PROB_MAX_TOKEN_LEN_PER_GPU:-$MAX_TOTAL_LENGTH}"

CMD=("$PYTHON_BIN" -u train.py)
if [[ "${DRY_RUN:-0}" == "1" ]]; then
    CMD+=(--cfg job --resolve)
fi

cd "$SCRIPT_DIR"
"${CMD[@]}" \
    rllm/backend=verl \
    rllm.algorithm.use_rllm=true \
    rllm.algorithm.loss_agg_mode=token-mean \
    rllm.rejection_sample.enable=true \
    rllm.rejection_sample.min_partial_solve_tasks=1 \
    rllm.rejection_sample.filter_uniform_groups=true \
    rllm.rejection_sample.keep_all_correct_groups=true \
    rllm.workflow.n_parallel_tasks=128 \
    rllm.workflow.retry_limit=3 \
    rllm.workflow.raise_on_error=false \
    rllm.gateway.health_failure_threshold=1 \
    rllm.trainer.save_freq="$SAVE_FREQ" \
    rllm.rollout.n=8 \
    rllm.rollout.n_val=1 \
    rllm.rollout.train.temperature=1.0 \
    rllm.rollout.train.top_p=1.0 \
    +rllm.rollout.train.top_k=-1 \
    rllm.rollout.train.max_tokens="$MAX_RESPONSE_LENGTH" \
    rllm.rollout.val.temperature=1.0 \
    rllm.rollout.val.top_p=0.7 \
    +rllm.rollout.val.top_k=-1 \
    rllm.rollout.val.max_tokens="$MAX_RESPONSE_LENGTH" \
    rllm.lsd.enable=true \
    rllm.lsd.routing_mode=online_group \
    rllm.lsd.online_group_threshold="$LSD_ONLINE_GROUP_THRESHOLD" \
    rllm.lsd.expected_group_size="$LSD_EXPECTED_GROUP_SIZE" \
    rllm.lsd.teacher_mode=ema \
    rllm.lsd.ema_half_life_updates="$LSD_EMA_HALF_LIFE_UPDATES" \
    rllm.lsd.ema_update_interval="$LSD_EMA_UPDATE_INTERVAL" \
    rllm.lsd.ema_accum_dtype="$LSD_EMA_ACCUM_DTYPE" \
    rllm.lsd.easy_threshold="$LSD_ONLINE_GROUP_THRESHOLD" \
    rllm.lsd.lsd_coef=1.0 \
    rllm.lsd.easy_rl_coef=0.0 \
    rllm.lsd.loss_agg_mode=token_mean \
    rllm.lsd.opd_loss_mode=topk_forward_kl \
    rllm.lsd.opd_optimization_mode=supervised_gradient \
    rllm.lsd.opd_topk="$LSD_OPD_TOPK" \
    rllm.lsd.opd_direct_loss_clip="$LSD_OPD_DIRECT_LOSS_CLIP" \
    rllm.lsd.opd_pg_adv_clip="$LSD_OPD_PG_ADV_CLIP" \
    rllm.lsd.fail_on_missing=false \
    algorithm.adv_estimator=grpo \
    algorithm.norm_adv_by_std_in_grpo=False \
    algorithm.use_kl_in_reward=False \
    algorithm.kl_ctrl.kl_coef=0.0 \
    algorithm.rollout_correction.rollout_is=sequence \
    algorithm.rollout_correction.rollout_is_threshold=2.0 \
    data.train_batch_size="$TRAIN_BATCH_SIZE" \
    data.val_batch_size=830 \
    +data.train_dataset_name="$RLLM_DEEPRESEARCH_TRAIN_DATASET" \
    +data.train_split="$RLLM_DEEPRESEARCH_TRAIN_SPLIT" \
    +data.val_dataset_name="$RLLM_DEEPRESEARCH_VAL_DATASET" \
    +data.val_split="$RLLM_DEEPRESEARCH_VAL_SPLIT" \
    +data.gen_batch_size=1 \
    data.val_files=null \
    data.max_prompt_length="$MAX_PROMPT_LENGTH" \
    data.max_response_length="$MAX_RESPONSE_LENGTH" \
    data.prompt_key=prompt \
    data.truncation=left \
    +model.name="$MODEL_PATH" \
    actor_rollout_ref.model.path="$MODEL_PATH" \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.model.enable_activation_offload=True \
    actor_rollout_ref.model.lora_rank=0 \
    actor_rollout_ref.hybrid_engine=True \
    actor_rollout_ref.actor.strategy=fsdp2 \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.optim.lr_warmup_steps=10 \
    actor_rollout_ref.actor.ppo_mini_batch_size=2048 \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu="$ACTOR_PPO_MAX_TOKEN_LEN_PER_GPU" \
    actor_rollout_ref.actor.fsdp_config.param_offload=true \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=true \
    actor_rollout_ref.actor.fsdp_config.fsdp_size=4 \
    actor_rollout_ref.actor.fsdp_config.dtype=float16 \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.grad_clip=1.0 \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.loss_agg_mode=token-mean \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size="$ACTOR_ULYSSES_SEQUENCE_PARALLEL_SIZE" \
    actor_rollout_ref.actor.clip_ratio_low=0.2 \
    actor_rollout_ref.actor.clip_ratio_high=0.28 \
    actor_rollout_ref.actor.clip_ratio_c=10.0 \
    actor_rollout_ref.rollout.name=sglang \
    actor_rollout_ref.rollout.mode=async \
    actor_rollout_ref.rollout.n=8 \
    actor_rollout_ref.rollout.dtype=float16 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.65 \
    actor_rollout_ref.rollout.enforce_eager=True \
    actor_rollout_ref.rollout.max_num_seqs="$ROLLOUT_MAX_NUM_SEQS" \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.skip_tokenizer_init=False \
    actor_rollout_ref.rollout.enable_chunked_prefill=True \
    actor_rollout_ref.rollout.max_num_batched_tokens="$ROLLOUT_MAX_NUM_BATCHED_TOKENS" \
    actor_rollout_ref.rollout.max_model_len="$ROLLOUT_MAX_MODEL_LEN" \
    actor_rollout_ref.rollout.response_length="$MAX_RESPONSE_LENGTH" \
    actor_rollout_ref.rollout.temperature=1.0 \
    actor_rollout_ref.rollout.top_p=1.0 \
    actor_rollout_ref.rollout.top_k=-1 \
    actor_rollout_ref.rollout.calculate_log_probs=True \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu="$LOG_PROB_MAX_TOKEN_LEN_PER_GPU" \
    actor_rollout_ref.rollout.val_kwargs.temperature=1.0 \
    actor_rollout_ref.rollout.val_kwargs.top_p=0.7 \
    actor_rollout_ref.rollout.val_kwargs.top_k=-1 \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.rollout.val_kwargs.n=1 \
    +actor_rollout_ref.rollout.engine_kwargs.sglang.disable_overlap_schedule=true \
    +actor_rollout_ref.rollout.engine_kwargs.sglang.tool_call_parser=qwen \
    +actor_rollout_ref.rollout.engine_kwargs.sglang.reasoning_parser=qwen3 \
    actor_rollout_ref.ref.strategy=fsdp2 \
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=True \
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu="$LOG_PROB_MAX_TOKEN_LEN_PER_GPU" \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.ref.fsdp_config.fsdp_size=4 \
    actor_rollout_ref.ref.ulysses_sequence_parallel_size="$ACTOR_ULYSSES_SEQUENCE_PARALLEL_SIZE" \
    trainer.logger="['console','wandb']" \
    trainer.project_name="$PROJECT_NAME" \
    trainer.experiment_name="$EXPERIMENT_NAME" \
    trainer.val_before_train="$VAL_BEFORE_TRAIN" \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.save_freq="$SAVE_FREQ" \
    trainer.test_freq="$TEST_FREQ" \
    trainer.total_epochs="$TOTAL_EPOCHS" \
    +rollout.n=8 \
    +rollout.rejection_sample=true \
    +rollout.test_freq="$TEST_FREQ" \
    +rollout.total_epochs="$TOTAL_EPOCHS" \
    trainer.default_local_dir="$CHECKPOINT_DIR" \
    trainer.default_hdfs_dir="$HDFS_CHECKPOINT_DIR" \
    trainer.max_actor_ckpt_to_keep="$MAX_ACTOR_CKPT_TO_KEEP" \
    trainer.max_critic_ckpt_to_keep="$MAX_CRITIC_CKPT_TO_KEEP" \
    trainer.resume_mode="$RESUME_MODE" \
    rllm.episode_logging.log_episodes=true \
    rllm.episode_logging.episode_log_dir="$LOG_DIR" \
    rllm.episode_logging.modes="['val']" \
    rllm.episode_logging.summary_only=false \
    rllm.episode_logging.include_text=true \
    rllm.episode_logging.file_format=jsonl \
    hydra.run.dir="$HYDRA_RUN_DIR" \
    "$@"
