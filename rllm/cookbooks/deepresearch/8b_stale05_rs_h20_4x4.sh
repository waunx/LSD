#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
SOURCE_REPO=${SOURCE_REPO:-/opt/tiger/deepresearch_seqmean_deploy/rllm}
RECIPE_DIR=${SOURCE_REPO}/examples/fully_async/deepresearch
REPO_ROOT=${SOURCE_REPO}
SCRIPT_NAME=${EXPERIMENT_NAME:-$(basename "$0" .sh)}

# Runtime and storage. Keep large artifacts off the nearly-full overlay filesystem.
PYTHON=${PYTHON:-/tmp/rllm_fully_async_venv/bin/python}
MODEL_PATH=${MODEL_PATH:-Qwen/Qwen3-8B}
RLLM_HOME=${RLLM_HOME:-/tmp/rllm_deepresearch_home}
HF_HOME=${HF_HOME:-/tmp/huggingface}
CHECKPOINT_ROOT=${CHECKPOINT_ROOT:-/tmp/rllm_deepresearch_ckpts}
CKPTS_DIR=${CKPTS_DIR:-${CHECKPOINT_ROOT}/deepresearch/${SCRIPT_NAME}}
WANDB_DIR=${WANDB_DIR:-/tmp/rllm_deepresearch_wandb}
DEFAULT_HDFS_DIR=${DEFAULT_HDFS_DIR:-null}
RAY_TMPDIR=${RAY_TMPDIR:-/tmp/ray_deepresearch}
TMPDIR=${TMPDIR:-/tmp/rllm_deepresearch_tmp}

if [[ "${SKIP_WANDB_LOGIN:-0}" != 1 && -n "${WANDB_API_KEY:-}" ]]; then
    "$PYTHON" -m wandb login "$WANDB_API_KEY"
fi

# Registered DatasetRegistry names.
DATASET_NAME=${DATASET_NAME:-cutthebill}
VAL_DATASET_NAME=${VAL_DATASET_NAME:-browsecomp-plus}

# External services. Comma-separated URL env vars take precedence over URL files.
source "$SCRIPT_DIR/refine_pool.sh"
DEEPRESEARCH_TRAIN_RAG_URL_FILE=${DEEPRESEARCH_TRAIN_RAG_URL_FILE:-${RECIPE_DIR}/.url/rag_url}
DEEPRESEARCH_VAL_RAG_URL_FILE=${DEEPRESEARCH_VAL_RAG_URL_FILE:-${RECIPE_DIR}/.url/val_url}
DEEPRESEARCH_REFINE_URL_FILE=${DEEPRESEARCH_REFINE_URL_FILE:-${RECIPE_DIR}/.url/refine_url}
DEEPRESEARCH_USE_REFINE=${DEEPRESEARCH_USE_REFINE:-true}
DEEPRESEARCH_MAX_TURNS=${DEEPRESEARCH_MAX_TURNS:-48}

# One H20 node: four GPUs train and four GPUs serve independent SGLang replicas.
NNODES=1
NGPUS_TRAINING=${NGPUS_TRAINING:-4}
NGPUS_ROLLOUT=${NGPUS_ROLLOUT:-4}
FSDP_SIZE=${FSDP_SIZE:-4}
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}

if ((FSDP_SIZE > 0 && (NGPUS_TRAINING % FSDP_SIZE != 0 || NGPUS_ROLLOUT % FSDP_SIZE != 0))); then
    echo "FSDP_SIZE=${FSDP_SIZE} must divide both trainer GPUs (${NGPUS_TRAINING}) and rollout GPUs (${NGPUS_ROLLOUT})" >&2
    exit 1
fi

# Optimization and rollout scale.
MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH:-700}
MAX_RESPONSE_LENGTH=${MAX_RESPONSE_LENGTH:-20000}
N_RESP_PER_PROMPT=${N_RESP_PER_PROMPT:-8}
REQUIRED_SAMPLES=${REQUIRED_SAMPLES:-256}
TRAIN_STEPS=${TRAIN_STEPS:-500}
TOTAL_ROLLOUT_STEPS=$((REQUIRED_SAMPLES * TRAIN_STEPS))
PPO_MINI_BATCH_SIZE=${PPO_MINI_BATCH_SIZE:-2048}
LOSS_AGG_MODE=${LOSS_AGG_MODE:-token-mean}
ACTOR_ULYSSES_SEQUENCE_PARALLEL_SIZE=${ACTOR_ULYSSES_SEQUENCE_PARALLEL_SIZE:-1}
TEST_FREQ=${TEST_FREQ:-20}
SAVE_FREQ=${SAVE_FREQ:-50}
TRAINER_LOGGERS=${TRAINER_LOGGERS:-"['console','wandb']"}
VAL_BEFORE_TRAIN=${VAL_BEFORE_TRAIN:-True}
RESUME_MODE=${RESUME_MODE:-disable}
STALENESS_THRESHOLD=${STALENESS_THRESHOLD:-0.5}
PARTIAL_ROLLOUT=${PARTIAL_ROLLOUT:-True}
ROLLOUT_GPU_MEMORY_UTILIZATION=${ROLLOUT_GPU_MEMORY_UTILIZATION:-0.65}
ROLLOUT_MAX_NUM_SEQS=128

# Controls.
RUN_PREFLIGHT=${RUN_PREFLIGHT:-1}
PREFLIGHT_ONLY=${PREFLIGHT_ONLY:-0}
DRY_RUN=${DRY_RUN:-0}
MIN_FREE_GPU_GB=${MIN_FREE_GPU_GB:-85}
MIN_FREE_DISK_GB=${MIN_FREE_DISK_GB:-80}

if [[ ! -x "${PYTHON}" ]]; then
    echo "Python environment not found: ${PYTHON}" >&2
    echo "Run: bash ${RECIPE_DIR}/scripts/setup_h20_env.sh" >&2
    exit 1
fi

mkdir -p "${RLLM_HOME}" "${HF_HOME}" "${CKPTS_DIR}" \
    "${WANDB_DIR}" "${RAY_TMPDIR}" "${TMPDIR}"

PYTHON_SITE=$(${PYTHON} -c 'import site; print(site.getsitepackages()[0])')
CUDA_RUNTIME_LIB=${PYTHON_SITE}/nvidia/cuda_runtime/lib
if [[ -d "${CUDA_RUNTIME_LIB}" ]]; then
    export LD_LIBRARY_PATH=${CUDA_RUNTIME_LIB}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
fi

export PYTHONPATH=${REPO_ROOT}:${PYTHONPATH:-}
export RLLM_HOME HF_HOME WANDB_DIR RAY_TMPDIR TMPDIR CUDA_VISIBLE_DEVICES
export DEEPRESEARCH_MODEL_PATH=${MODEL_PATH}
export DEEPRESEARCH_TRAIN_RAG_URL_FILE DEEPRESEARCH_VAL_RAG_URL_FILE
export DEEPRESEARCH_REFINE_URL_FILE DEEPRESEARCH_USE_REFINE DEEPRESEARCH_MAX_TURNS
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:False"
export PYTORCH_ALLOC_CONF="expandable_segments:False"
export NCCL_DEBUG=${NCCL_DEBUG:-WARN}

# #region debug-point B:allocator-config
if [[ -f .dbg/training-stopped-resume200.env ]]; then source .dbg/training-stopped-resume200.env; curl -sS -X POST "$DEBUG_SERVER_URL" -H "Content-Type: application/json" -d "{\"sessionId\":\"$DEBUG_SESSION_ID\",\"runId\":\"post-fix\",\"hypothesisId\":\"B\",\"location\":\"8b_stale05_rs_h20_4x4.sh:allocator\",\"msg\":\"[DEBUG] trainer allocator configuration\",\"data\":{\"pytorch_alloc_conf\":\"$PYTORCH_ALLOC_CONF\",\"max_prompt_length\":${MAX_PROMPT_LENGTH},\"max_response_length\":${MAX_RESPONSE_LENGTH},\"infer_token_cap\":$((MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH))}}" >/dev/null || true; fi
# #endregion

PREFLIGHT_ARGS=(
    --train-dataset "${DATASET_NAME}"
    --train-split train
    --val-dataset "${VAL_DATASET_NAME}"
    --val-split test
    --model-path "${MODEL_PATH}"
    --checkpoint-dir "${CKPTS_DIR}"
    --train-rag-url-file "${DEEPRESEARCH_TRAIN_RAG_URL_FILE}"
    --val-rag-url-file "${DEEPRESEARCH_VAL_RAG_URL_FILE}"
    --refine-url-file "${DEEPRESEARCH_REFINE_URL_FILE}"
    --trainer-gpus "${NGPUS_TRAINING}"
    --rollout-gpus "${NGPUS_ROLLOUT}"
    --min-free-gpu-gb "${MIN_FREE_GPU_GB}"
    --min-free-disk-gb "${MIN_FREE_DISK_GB}"
)

case "${DEEPRESEARCH_USE_REFINE,,}" in
    1|true|yes|on) PREFLIGHT_ARGS+=(--use-refine) ;;
esac
[[ "${SKIP_ENDPOINT_CHECK:-0}" == 1 ]] && PREFLIGHT_ARGS+=(--skip-endpoint-check)
[[ "${SKIP_GPU_MEMORY_CHECK:-0}" == 1 ]] && PREFLIGHT_ARGS+=(--skip-gpu-memory-check)

if [[ "${RUN_PREFLIGHT}" == 1 ]]; then
    "${PYTHON}" "${RECIPE_DIR}/scripts/preflight.py" "${PREFLIGHT_ARGS[@]}"
fi
if [[ "${PREFLIGHT_ONLY}" == 1 ]]; then
    exit 0
fi

ACTOR_MAX_TOKEN_LEN=$((MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH))
ACTOR_PPO_MAX_TOKEN_LEN_PER_GPU=${ACTOR_PPO_MAX_TOKEN_LEN_PER_GPU:-${ACTOR_MAX_TOKEN_LEN}}
INFER_MAX_TOKEN_LEN=${INFER_MAX_TOKEN_LEN:-${ACTOR_MAX_TOKEN_LEN}}

DATA_ARGS=(
    data.train_dataset_name="${DATASET_NAME}"
    data.train_split=train
    data.val_dataset_name="${VAL_DATASET_NAME}"
    data.val_split=test
    data.val_files=null
    data.prompt_key=prompt
    data.truncation=left
    data.max_prompt_length="${MAX_PROMPT_LENGTH}"
    data.max_response_length="${MAX_RESPONSE_LENGTH}"
    data.train_batch_size=0
    data.gen_batch_size=1
)

ALGORITHM_ARGS=(
    algorithm.adv_estimator=grpo
    algorithm.norm_adv_by_std_in_grpo=False
    algorithm.use_kl_in_reward=False
    algorithm.kl_ctrl.kl_coef=0.0
    algorithm.rollout_correction.rollout_is=sequence
    algorithm.rollout_correction.rollout_is_threshold=2.0
)

MODEL_ARGS=(
    actor_rollout_ref.model.path="${MODEL_PATH}"
    actor_rollout_ref.model.use_remove_padding=True
    actor_rollout_ref.model.enable_gradient_checkpointing=True
    actor_rollout_ref.model.enable_activation_offload=True
    actor_rollout_ref.hybrid_engine=False
)

ACTOR_ARGS=(
    actor_rollout_ref.actor.strategy=fsdp2
    actor_rollout_ref.actor.use_kl_loss=False
    actor_rollout_ref.actor.kl_loss_coef=0.0
    actor_rollout_ref.actor.clip_ratio_low=0.2
    actor_rollout_ref.actor.clip_ratio_high=0.28
    actor_rollout_ref.actor.clip_ratio_c=10.0
    actor_rollout_ref.actor.use_dynamic_bsz=True
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu="${ACTOR_PPO_MAX_TOKEN_LEN_PER_GPU}"
    actor_rollout_ref.actor.ppo_mini_batch_size="${PPO_MINI_BATCH_SIZE}"
    actor_rollout_ref.actor.optim.lr=1e-6
    actor_rollout_ref.actor.optim.lr_warmup_steps=10
    actor_rollout_ref.actor.fsdp_config.param_offload=True
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True
    actor_rollout_ref.actor.fsdp_config.fsdp_size="${FSDP_SIZE}"
    actor_rollout_ref.actor.fsdp_config.dtype=float16
    actor_rollout_ref.actor.entropy_coeff=0
    actor_rollout_ref.actor.grad_clip=1.0
    actor_rollout_ref.actor.loss_agg_mode="${LOSS_AGG_MODE}"
    actor_rollout_ref.actor.ulysses_sequence_parallel_size="${ACTOR_ULYSSES_SEQUENCE_PARALLEL_SIZE}"
)

ROLLOUT_ARGS=(
    actor_rollout_ref.rollout.name=sglang
    actor_rollout_ref.rollout.mode=async
    actor_rollout_ref.rollout.n="${N_RESP_PER_PROMPT}"
    actor_rollout_ref.rollout.dtype=float16
    actor_rollout_ref.rollout.gpu_memory_utilization="${ROLLOUT_GPU_MEMORY_UTILIZATION}"
    actor_rollout_ref.rollout.enforce_eager=True
    actor_rollout_ref.rollout.max_num_seqs="${ROLLOUT_MAX_NUM_SEQS}"
    actor_rollout_ref.rollout.tensor_model_parallel_size=1
    actor_rollout_ref.rollout.enable_chunked_prefill=True
    actor_rollout_ref.rollout.max_num_batched_tokens="${ACTOR_MAX_TOKEN_LEN}"
    actor_rollout_ref.rollout.temperature=1.0
    actor_rollout_ref.rollout.top_p=1.0
    actor_rollout_ref.rollout.top_k=-1
    actor_rollout_ref.rollout.calculate_log_probs=True
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu="${INFER_MAX_TOKEN_LEN}"
    actor_rollout_ref.rollout.val_kwargs.temperature=1.0
    actor_rollout_ref.rollout.val_kwargs.top_p=0.7
    actor_rollout_ref.rollout.val_kwargs.top_k=-1
    actor_rollout_ref.rollout.val_kwargs.do_sample=True
    actor_rollout_ref.rollout.val_kwargs.n=1
    +actor_rollout_ref.rollout.engine_kwargs.sglang.disable_overlap_schedule=true
    +actor_rollout_ref.rollout.engine_kwargs.sglang.tool_call_parser=qwen
    +actor_rollout_ref.rollout.engine_kwargs.sglang.reasoning_parser=qwen3
)

REFERENCE_ARGS=(
    critic.strategy=fsdp2
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=True
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu="${INFER_MAX_TOKEN_LEN}"
    actor_rollout_ref.ref.fsdp_config.param_offload=True
    actor_rollout_ref.ref.ulysses_sequence_parallel_size=1
)

TRAINER_ARGS=(
    trainer.logger="${TRAINER_LOGGERS}"
    trainer.project_name=deepresearch
    trainer.experiment_name="${SCRIPT_NAME}"
    trainer.val_before_train="${VAL_BEFORE_TRAIN}"
    trainer.save_freq="${SAVE_FREQ}"
    trainer.default_local_dir="${CKPTS_DIR}"
    trainer.default_hdfs_dir="${DEFAULT_HDFS_DIR}"
    trainer.max_actor_ckpt_to_keep=1
    trainer.max_critic_ckpt_to_keep=1
    trainer.resume_mode="${RESUME_MODE}"
    trainer.nnodes="${NNODES}"
    trainer.n_gpus_per_node="${NGPUS_TRAINING}"
    trainer.balance_batch=True
)

ASYNC_ARGS=(
    rollout.nnodes="${NNODES}"
    rollout.n_gpus_per_node="${NGPUS_ROLLOUT}"
    rollout.n="${N_RESP_PER_PROMPT}"
    rollout.total_rollout_steps="${TOTAL_ROLLOUT_STEPS}"
    rollout.total_epochs=10
    rollout.test_freq="${TEST_FREQ}"
    rollout.rejection_sample=True
    async_training.staleness_threshold="${STALENESS_THRESHOLD}"
    async_training.trigger_parameter_sync_step=1
    async_training.require_batches=1
    async_training.required_samples="${REQUIRED_SAMPLES}"
    async_training.partial_rollout="${PARTIAL_ROLLOUT}"
    async_training.use_rollout_log_probs=True
    async_training.compute_prox_log_prob=True
)

ALL_ARGS=(
    "${DATA_ARGS[@]}"
    "${ALGORITHM_ARGS[@]}"
    "${MODEL_ARGS[@]}"
    "${ACTOR_ARGS[@]}"
    "${ROLLOUT_ARGS[@]}"
    "${REFERENCE_ARGS[@]}"
    "${TRAINER_ARGS[@]}"
    "${ASYNC_ARGS[@]}"
)

cd "${REPO_ROOT}"
if [[ "${DRY_RUN}" == 1 ]]; then
    exec "${PYTHON}" -m examples.fully_async.deepresearch.train \
        --cfg job --resolve "${ALL_ARGS[@]}"
fi

exec "${PYTHON}" -m examples.fully_async.deepresearch.train "${ALL_ARGS[@]}"
