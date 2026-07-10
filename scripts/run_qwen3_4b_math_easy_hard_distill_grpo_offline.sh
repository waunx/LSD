#!/usr/bin/env bash
# Offline easy/hard routing: read teacher rollout labels from OFFLINE_LABEL_FILE.

set -xeuo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)

# OPD_MODE=supervised:
#   easy prompts -> supervised OPD, default forward_kl_topk.
# OPD_MODE=rl:
#   easy prompts -> task reward + teacher distillation reward before advantage;
#   hard prompts -> task reward only.

OPD_MODE=rl

if [[ -z "${DISTILLATION_LOSS_MODE:-}" ]]; then
    case "${OPD_MODE}" in
        supervised)
            DISTILLATION_LOSS_MODE=forward_kl_topk
            ;;
        rl)
            DISTILLATION_LOSS_MODE=k3
            ;;
        *)
            echo "Unsupported OPD_MODE=${OPD_MODE}. Expected 'supervised' or 'rl'." >&2
            exit 1
            ;;
    esac
fi


RL_DISTILLATION_REWARD_LOSS_MODE=k3

# 4-card layout on physical GPUs 4,5,6,7:
#   actor/rollout -> 2 visible GPUs
#   frozen teacher -> 2 visible GPUs
CUDA_VISIBLE_DEVICES=4,5,6,7
NGPUS_PER_NODE=2
TEACHER_WORLD_SIZE=2
ROLLOUT_TP=2
TEACHER_TP=2

TRAIN_MAX_SAMPLES=51200

EASY_THRESHOLD=0.75

OFFLINE_LABEL_FILE=/data/wx_data/verl-lsd/outputs/easy_hard_offline_labels/qwen3_4b_teacher_4k_train_offline_labels-5/validation_data/0.jsonl

EXPERIMENT_NAME=0616-qwen3_4b_math_12k-offline_pg_opd_k3-thr0p75-train51200_teacher_4k-8

VALIDATION_DATA_DIR=/data/wx_data/verl-lsd/outputs/${EXPERIMENT_NAME}/validation_data

PROJECT_NAME=length_scaling
SWANLAB_MODE=${SWANLAB_MODE:-cloud}
SWANLAB_LOG_DIR=${SWANLAB_LOG_DIR:-swanlog}
SWANLAB_NO_PROXY_HOSTS=${SWANLAB_NO_PROXY_HOSTS:-api.swanlab.cn,swanlab.cn}
if [[ -n "${NO_PROXY:-}" ]]; then
    NO_PROXY="${SWANLAB_NO_PROXY_HOSTS},${NO_PROXY}"
else
    NO_PROXY="${SWANLAB_NO_PROXY_HOSTS}"
fi

LOG_DIR="${SCRIPT_DIR}/../logs"

mkdir -p "${LOG_DIR}"
LOG_FILE=${LOG_FILE:-"${LOG_DIR}/${EXPERIMENT_NAME}_$(date +%Y%m%d_%H%M%S).log"}
echo "Saving train log to ${LOG_FILE}"
exec > >(tee -a "${LOG_FILE}") 2>&1

EASY_HARD_MODE=offline \
OPD_MODE="${OPD_MODE}" \
DISTILLATION_LOSS_MODE="${DISTILLATION_LOSS_MODE}" \
RL_DISTILLATION_REWARD_LOSS_MODE="${RL_DISTILLATION_REWARD_LOSS_MODE}" \
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" \
NGPUS_PER_NODE="${NGPUS_PER_NODE}" \
TEACHER_WORLD_SIZE="${TEACHER_WORLD_SIZE}" \
ROLLOUT_TP="${ROLLOUT_TP}" \
TEACHER_TP="${TEACHER_TP}" \
SWANLAB_MODE="${SWANLAB_MODE}" \
SWANLAB_LOG_DIR="${SWANLAB_LOG_DIR}" \
NO_PROXY="${NO_PROXY}" \
no_proxy="${NO_PROXY}" \
bash "${SCRIPT_DIR}/run_qwen3_4b_math_easy_hard_distill_grpo.sh" \
    data.train_max_samples="${TRAIN_MAX_SAMPLES}" \
    distillation.easy_hard_split.accuracy_threshold="${EASY_THRESHOLD}" \
    distillation.easy_hard_split.offline_label_path="${OFFLINE_LABEL_FILE}" \
    trainer.project_name="${PROJECT_NAME}" \
    trainer.experiment_name="${EXPERIMENT_NAME}" \
    trainer.validation_data_dir="${VALIDATION_DATA_DIR}" \
    "$@"
