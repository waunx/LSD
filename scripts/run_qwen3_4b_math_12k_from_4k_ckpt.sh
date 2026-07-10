#!/usr/bin/env bash
# GRPO | Qwen3-4B | continue 12K RL post-training from the 4K checkpoint

set -xeuo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)

# ---- user-adjustable ----
DEVICE=${DEVICE:-gpu}
INFER_BACKEND=${INFER_BACKEND:-vllm}
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=4,5,6,7
# 0,1,2,3
export VLLM_USE_V1=1

FOUR_K_OUTPUT_DIR=${FOUR_K_OUTPUT_DIR:-/data/wx_data/verl-lsd/outputs/0609-qwen3_4b_base-math-4k-train_data_deepseek_rl-1}
# FOUR_K_OUTPUT_DIR stores validation artifacts such as validation_data/234.jsonl.
# The matching FSDP checkpoint for that step lives under the checkpoint tree and
# should be exported to an HF directory before use here.
MODEL_CKPT_DIR=${MODEL_CKPT_DIR:-/data/wx_data/verl-lsd/checkpoints/length_scaling/0609-qwen3_4b_base-math-4k-train_data_deepseek_rl-1/global_step_234/actor}
MODEL_PATH=${MODEL_PATH:-/data/wx_data/verl-lsd/checkpoints/length_scaling/0609-qwen3_4b_base-math-4k-train_data_deepseek_rl-1/global_step_234/actor_hf}
TRAIN_FILE="['/data/wx_data/datasets/DeepScaleR/deepscaler_train_math.parquet']"
TEST_FILE="['/data/wx_data/datasets/LSD/aime.parquet','/data/wx_data/datasets/LSD/amc.parquet','/data/wx_data/datasets/LSD/math.parquet','/data/wx_data/datasets/LSD/minerva.parquet','/data/wx_data/datasets/LSD/olympiad_bench.parquet']"

NNODES=1
NGPUS_PER_NODE=4

TRAIN_BATCH_SIZE=512
PPO_MINI_BATCH_SIZE=256
PPO_MICRO_BATCH_SIZE_PER_GPU=2
LOG_PROB_MICRO_BATCH_SIZE_PER_GPU=2
MAX_PROMPT_LENGTH=2048
MAX_RESPONSE_LENGTH=$((12 * 1024))
MAX_TOKEN_LEN_PER_GPU=${MAX_TOKEN_LEN_PER_GPU:-$((MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH + 1))}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-${MAX_TOKEN_LEN_PER_GPU}}
MAX_NUM_BATCHED_TOKENS=${MAX_NUM_BATCHED_TOKENS:-16384}

ACTOR_LR=1e-6
KL_LOSS_COEF=1e-5
ENTROPY_COEFF=0.0

ROLLOUT_TP=4
ROLLOUT_GPU_MEM_UTIL=0.6
ROLLOUT_N=8
VAL_N=8
VAL_TEMPERATURE=0.0
VAL_TOP_P=1.0
VAL_TOP_K=-1

PROJECT_NAME=length_scaling
EXPERIMENT_NAME=0709-qwen3_4b_math_12k-Base_4krl_step234-traindata_deepscaler-1
VALIDATION_DATA_DIR=${VALIDATION_DATA_DIR:-"/data/wx_data/verl-lsd/outputs/${EXPERIMENT_NAME}/validation_data"}
SWANLAB_MODE=${SWANLAB_MODE:-cloud}
SWANLAB_LOG_DIR=${SWANLAB_LOG_DIR:-swanlog}
SWANLAB_NO_PROXY_HOSTS=${SWANLAB_NO_PROXY_HOSTS:-api.swanlab.cn,swanlab.cn}
if [[ -n "${NO_PROXY:-}" ]]; then
    export NO_PROXY="${SWANLAB_NO_PROXY_HOSTS},${NO_PROXY}"
else
    export NO_PROXY="${SWANLAB_NO_PROXY_HOSTS}"
fi
export no_proxy="${NO_PROXY}"
export SWANLAB_MODE
export SWANLAB_LOG_DIR
SAVE_FREQ=50
TEST_FREQ=5
TOTAL_EPOCHS=3

# ---- end user-adjustable ----

if [[ ! -d "${FOUR_K_OUTPUT_DIR}" ]]; then
    echo "FOUR_K_OUTPUT_DIR=${FOUR_K_OUTPUT_DIR} does not exist." >&2
    exit 1
fi

if [[ ! -d "${MODEL_CKPT_DIR}" ]]; then
    echo "MODEL_CKPT_DIR=${MODEL_CKPT_DIR} does not exist." >&2
    exit 1
fi

if [[ ! -d "${MODEL_PATH}" ]]; then
    echo "MODEL_PATH=${MODEL_PATH} does not exist." >&2
    echo "Export the 4K RL checkpoint first, for example:" >&2
    echo "python3 -m verl.model_merger merge --backend fsdp --local_dir ${MODEL_CKPT_DIR} --target_dir ${MODEL_PATH}" >&2
    exit 1
fi

if [[ "${MODEL_PATH}" == "${MODEL_CKPT_DIR}/huggingface" ]]; then
    echo "MODEL_PATH=${MODEL_PATH} only contains config/tokenizer files, not model weights." >&2
    echo "Use the exported HF model directory instead, for example:" >&2
    echo "MODEL_PATH=${MODEL_CKPT_DIR%/actor}/actor_hf bash ${BASH_SOURCE[0]}" >&2
    exit 1
fi

if [[ ! -f "${MODEL_PATH}/model.safetensors" && ! -f "${MODEL_PATH}/pytorch_model.bin" && ! -f "${MODEL_PATH}/model.safetensors.index.json" && ! -f "${MODEL_PATH}/pytorch_model.bin.index.json" ]]; then
    echo "MODEL_PATH=${MODEL_PATH} is missing HF weight files." >&2
    echo "Export the 4K RL checkpoint first, for example:" >&2
    echo "python3 -m verl.model_merger merge --backend fsdp --local_dir ${MODEL_CKPT_DIR} --target_dir ${MODEL_PATH}" >&2
    exit 1
fi

python3 - "${TRAIN_FILE}" <<'PY'
import ast
import sys

import pandas as pd

train_files = ast.literal_eval(sys.argv[1])
bad_files = []
for path in train_files:
    data_source = pd.read_parquet(path, columns=["data_source"])["data_source"]
    unique_values = sorted(str(value) for value in data_source.dropna().unique())
    if not unique_values or unique_values == [""]:
        bad_files.append(path)

if bad_files:
    joined = ", ".join(bad_files)
    raise SystemExit(
        "Training parquet has empty data_source, so verl cannot select a reward function: "
        f"{joined}"
    )
PY

LOG_DIR="${SCRIPT_DIR}/../logs"
mkdir -p "${LOG_DIR}"
LOG_FILE=${LOG_FILE:-"${LOG_DIR}/${EXPERIMENT_NAME}_$(date +%Y%m%d_%H%M%S).log"}
echo "Saving train log to ${LOG_FILE}"
exec > >(tee -a "${LOG_FILE}") 2>&1

case "${DEVICE}" in
    gpu)
        ;;
    npu)
        export VLLM_USE_V1=1
        export TASK_QUEUE_ENABLE=2
        export CPU_AFFINITY_CONF=1
        export LD_PRELOAD="/usr/lib/aarch64-linux-gnu/libjemalloc.so.2${LD_PRELOAD:+:$LD_PRELOAD}"
        NGPUS_PER_NODE=16
        ROLLOUT_GPU_MEM_UTIL=0.9
        ;;
    *)
        echo "Unsupported DEVICE=${DEVICE}. Expected 'gpu' or 'npu'." >&2
        exit 1
        ;;
esac

########################### parameter arrays ###########################

DATA=(
    algorithm.adv_estimator=grpo
    "data.train_files=${TRAIN_FILE}"
    "data.val_files=${TEST_FILE}"
    data.train_batch_size=${TRAIN_BATCH_SIZE}
    data.max_prompt_length=${MAX_PROMPT_LENGTH}
    data.max_response_length=${MAX_RESPONSE_LENGTH}
    data.filter_overlong_prompts=False
    data.truncation='left'
    algorithm.use_kl_in_reward=False
)

MODEL=(
    actor_rollout_ref.model.path=${MODEL_PATH}
    actor_rollout_ref.model.use_remove_padding=True
    actor_rollout_ref.model.enable_gradient_checkpointing=True
)

ACTOR=(
    actor_rollout_ref.actor.optim.lr=${ACTOR_LR}
    actor_rollout_ref.actor.ppo_mini_batch_size=${PPO_MINI_BATCH_SIZE}
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=${PPO_MICRO_BATCH_SIZE_PER_GPU}
    actor_rollout_ref.actor.use_kl_loss=True
    actor_rollout_ref.actor.kl_loss_coef=${KL_LOSS_COEF}
    actor_rollout_ref.actor.kl_loss_type=low_var_kl
    actor_rollout_ref.actor.entropy_coeff=${ENTROPY_COEFF}
    actor_rollout_ref.actor.fsdp_config.param_offload=False
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False
    actor_rollout_ref.actor.fsdp_config.model_dtype=bfloat16
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${MAX_TOKEN_LEN_PER_GPU}
    actor_rollout_ref.actor.use_dynamic_bsz=True
)

ROLLOUT=(
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=${LOG_PROB_MICRO_BATCH_SIZE_PER_GPU}
    actor_rollout_ref.rollout.tensor_model_parallel_size=${ROLLOUT_TP}
    actor_rollout_ref.rollout.name=${INFER_BACKEND}
    actor_rollout_ref.rollout.gpu_memory_utilization=${ROLLOUT_GPU_MEM_UTIL}
    actor_rollout_ref.rollout.enable_chunked_prefill=False
    actor_rollout_ref.rollout.enforce_eager=False
    actor_rollout_ref.rollout.free_cache_engine=True
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=${MAX_TOKEN_LEN_PER_GPU}
    actor_rollout_ref.rollout.max_model_len=${MAX_MODEL_LEN}
    actor_rollout_ref.rollout.max_num_batched_tokens=${MAX_NUM_BATCHED_TOKENS}
    actor_rollout_ref.rollout.checkpoint_engine.update_weights_bucket_megabytes=4096
    actor_rollout_ref.rollout.n=${ROLLOUT_N}
    actor_rollout_ref.rollout.val_kwargs.n=${VAL_N}
    actor_rollout_ref.rollout.val_kwargs.do_sample=True
    actor_rollout_ref.rollout.val_kwargs.temperature=${VAL_TEMPERATURE}
    actor_rollout_ref.rollout.val_kwargs.top_p=${VAL_TOP_P}
    actor_rollout_ref.rollout.val_kwargs.top_k=${VAL_TOP_K}
)

REF=(
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=${LOG_PROB_MICRO_BATCH_SIZE_PER_GPU}
    actor_rollout_ref.ref.fsdp_config.param_offload=True
    actor_rollout_ref.ref.fsdp_config.model_dtype=bfloat16
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=True
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=${MAX_TOKEN_LEN_PER_GPU}
)

TRAINER=(
    trainer.critic_warmup=0
    trainer.logger='["console","swanlab"]'
    trainer.project_name=${PROJECT_NAME}
    trainer.experiment_name=${EXPERIMENT_NAME}
    trainer.validation_data_dir=${VALIDATION_DATA_DIR}
    trainer.n_gpus_per_node=${NGPUS_PER_NODE}
    trainer.nnodes=${NNODES}
    trainer.save_freq=${SAVE_FREQ}
    trainer.test_freq=${TEST_FREQ}
    trainer.total_epochs=${TOTAL_EPOCHS}
)

EXTRA=(
    +ray_kwargs.ray_init.runtime_env.env_vars.SWANLAB_MODE=${SWANLAB_MODE}
    +ray_kwargs.ray_init.runtime_env.env_vars.SWANLAB_LOG_DIR=${SWANLAB_LOG_DIR}
    "+ray_kwargs.ray_init.runtime_env.env_vars.NO_PROXY='${NO_PROXY}'"
    "+ray_kwargs.ray_init.runtime_env.env_vars.no_proxy='${NO_PROXY}'"
)

########################### launch ###########################
python3 -m verl.trainer.main_ppo \
    "${DATA[@]}" \
    "${MODEL[@]}" \
    "${ACTOR[@]}" \
    "${ROLLOUT[@]}" \
    "${REF[@]}" \
    "${TRAINER[@]}" \
    "${EXTRA[@]}" \
    "$@"
