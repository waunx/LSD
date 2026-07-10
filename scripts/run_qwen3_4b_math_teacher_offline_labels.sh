#!/usr/bin/env bash
# Infer train-set easy/hard labels with the frozen 4K RL teacher.
#
# Output:
#   ${LABEL_OUTPUT_DIR}/0.jsonl
#
# Each prompt is sampled VAL_N times. The training script's offline loader will
# aggregate these repeated rows by prompt and mark a prompt easy when mean acc >
# EASY_THRESHOLD.

set -xeuo pipefail

# ---- user-adjustable ----
DEVICE=gpu
INFER_BACKEND=vllm
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export VLLM_USE_V1=1
export HYDRA_FULL_ERROR=1
export RAY_DEDUP_LOGS=0
FORCE_LOCAL_RAY=1
if [[ "${FORCE_LOCAL_RAY}" == "1" ]]; then
    unset RAY_ADDRESS
fi

TEACHER_CKPT_DIR="/data/wx_data/verl-lsd/checkpoints/length_scaling/0609-qwen3_4b_base-math-4k-3/global_step_200/actor"
TEACHER_MODEL_PATH="/data/wx_data/verl-lsd/checkpoints/length_scaling/0609-qwen3_4b_base-math-4k-3/global_step_200/actor_hf"

TRAIN_FILE="['/data/wx_data/datasets/OPD/DAPO-Math-17k/data/dapo-math-17k.parquet']"

NNODES=1
NGPUS_PER_NODE=8
TRAIN_BATCH_SIZE=1
VAL_BATCH_SIZE=1024
DATA_SEED=42
MAX_PROMPT_LENGTH=2048
MAX_RESPONSE_LENGTH=4096
MAX_TOKEN_LEN_PER_GPU=$((MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH + 1))
MAX_MODEL_LEN=$((MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH + 1))
MAX_NUM_BATCHED_TOKENS=8192
MAX_NUM_SEQS=1024

ROLLOUT_TP=4
ROLLOUT_GPU_MEM_UTIL=0.6
VAL_N=8
VAL_TEMPERATURE=1.0
VAL_TOP_P=1.0
VAL_TOP_K=-1

PROJECT_NAME=length_scaling_efficiency
EXPERIMENT_NAME=qwen3_4b_teacher_4k_train_offline_labels-5
LABEL_OUTPUT_DIR="/data/wx_data/verl-lsd/outputs/easy_hard_offline_labels/${EXPERIMENT_NAME}/validation_data"
RAY_TMPDIR="/tmp/ray_teacher_labels"
# For 200 training steps with TRAIN_BATCH_SIZE=256, only 51,200 unique
# prompts are consumed. Keep this seed and count aligned with the training
# script when using EASY_HARD_MODE=offline.
VAL_MAX_SAMPLES=51200
# ---- end user-adjustable ----

mkdir -p "${RAY_TMPDIR}" "${LABEL_OUTPUT_DIR}" logs

case "${DEVICE}" in
    gpu)
        ;;
    npu)
        export VLLM_USE_V1=1
        export TASK_QUEUE_ENABLE=2
        export CPU_AFFINITY_CONF=1
        export LD_PRELOAD="/usr/lib/aarch64-linux-gnu/libjemalloc.so.2${LD_PRELOAD:+:$LD_PRELOAD}"
        NGPUS_PER_NODE=16
        ROLLOUT_GPU_MEM_UTIL=0.85
        ;;
    *)
        echo "Unsupported DEVICE=${DEVICE}. Expected 'gpu' or 'npu'." >&2
        exit 1
        ;;
esac

if [[ ! -d "${TEACHER_MODEL_PATH}" ]]; then
    echo "TEACHER_MODEL_PATH=${TEACHER_MODEL_PATH} does not exist." >&2
    echo "Export the 4K RL checkpoint first, for example:" >&2
    echo "python3 -m verl.model_merger merge --backend fsdp --local_dir ${TEACHER_CKPT_DIR} --target_dir ${TEACHER_MODEL_PATH}" >&2
    exit 1
fi

DATA=(
    algorithm.adv_estimator=grpo
    algorithm.use_kl_in_reward=False
    critic.enable=False
    "data.train_files=${TRAIN_FILE}"
    "data.val_files=${TRAIN_FILE}"
    data.train_batch_size=${TRAIN_BATCH_SIZE}
    data.val_batch_size=${VAL_BATCH_SIZE}
    data.val_max_samples=${VAL_MAX_SAMPLES}
    data.shuffle=True
    data.seed=${DATA_SEED}
    data.max_prompt_length=${MAX_PROMPT_LENGTH}
    data.max_response_length=${MAX_RESPONSE_LENGTH}
    data.filter_overlong_prompts=False
    data.truncation='left'
)

MODEL=(
    actor_rollout_ref.model.path=${TEACHER_MODEL_PATH}
    actor_rollout_ref.model.use_remove_padding=True
    actor_rollout_ref.model.enable_gradient_checkpointing=False
)

ACTOR=(
    actor_rollout_ref.actor.use_kl_loss=False
    actor_rollout_ref.actor.fsdp_config.param_offload=True
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True
    actor_rollout_ref.actor.fsdp_config.model_dtype=bfloat16
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${MAX_TOKEN_LEN_PER_GPU}
    actor_rollout_ref.actor.use_dynamic_bsz=True
)

ROLLOUT=(
    actor_rollout_ref.rollout.tensor_model_parallel_size=${ROLLOUT_TP}
    actor_rollout_ref.rollout.name=${INFER_BACKEND}
    actor_rollout_ref.rollout.gpu_memory_utilization=${ROLLOUT_GPU_MEM_UTIL}
    actor_rollout_ref.rollout.enable_chunked_prefill=False
    actor_rollout_ref.rollout.enforce_eager=False
    actor_rollout_ref.rollout.free_cache_engine=True
    actor_rollout_ref.rollout.max_model_len=${MAX_MODEL_LEN}
    actor_rollout_ref.rollout.max_num_seqs=${MAX_NUM_SEQS}
    actor_rollout_ref.rollout.max_num_batched_tokens=${MAX_NUM_BATCHED_TOKENS}
    actor_rollout_ref.rollout.n=1
    actor_rollout_ref.rollout.val_kwargs.n=${VAL_N}
    actor_rollout_ref.rollout.val_kwargs.do_sample=True
    actor_rollout_ref.rollout.val_kwargs.temperature=${VAL_TEMPERATURE}
    actor_rollout_ref.rollout.val_kwargs.top_p=${VAL_TOP_P}
    actor_rollout_ref.rollout.val_kwargs.top_k=${VAL_TOP_K}
)

TRAINER=(
    trainer.logger='["console"]'
    trainer.project_name=${PROJECT_NAME}
    trainer.experiment_name=${EXPERIMENT_NAME}
    trainer.validation_data_dir=${LABEL_OUTPUT_DIR}
    trainer.n_gpus_per_node=${NGPUS_PER_NODE}
    trainer.nnodes=${NNODES}
    trainer.resume_mode=disable
    trainer.val_before_train=True
    trainer.val_only=True
    trainer.test_freq=-1
    trainer.save_freq=-1
    trainer.total_epochs=1
    +ray_kwargs.ray_init._temp_dir=${RAY_TMPDIR}
)

python3 -m verl.trainer.main_ppo \
    "${DATA[@]}" \
    "${MODEL[@]}" \
    "${ACTOR[@]}" \
    "${ROLLOUT[@]}" \
    "${TRAINER[@]}" \
    "$@"

echo "Offline label rollout file: ${LABEL_OUTPUT_DIR}/0.jsonl"
