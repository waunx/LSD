#!/usr/bin/env bash
# Online easy/hard routing: label each prompt from the current policy rollout group.

set -xeuo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)

EASY_HARD_MODE=online exec bash "${SCRIPT_DIR}/run_qwen3_4b_math_easy_hard_distill_grpo.sh" "$@"
