#!/usr/bin/env bash
# Source before Ray starts so both Base and LSD workers inherit the same pool.
# Explicit URL or URL-file settings retain precedence.
if [[ -z "${DEEPRESEARCH_REFINE_URLS:-}" && -z "${DEEPRESEARCH_REFINE_URL_FILE:-}" ]]; then
    _refine_pool_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
    DEEPRESEARCH_REFINE_URLS="$(
        "${REFINE_CONTROL_PYTHON:-python3}" "$_refine_pool_dir/serve_refine_local.py" resolve \
            --state "${REFINE_POOL_STATE:-$_refine_pool_dir/../../logs/refine_local/servers.json}"
    )" || return 1
    export DEEPRESEARCH_REFINE_URLS
    unset _refine_pool_dir
fi
export DEEPRESEARCH_REFINE_MODEL="${DEEPRESEARCH_REFINE_MODEL:-Qwen/Qwen3-8B}"
