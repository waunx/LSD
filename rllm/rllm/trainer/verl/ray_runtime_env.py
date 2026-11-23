import json
import os
import urllib.request

from ray._private.runtime_env.constants import RAY_JOB_CONFIG_JSON_ENV_VAR

PPO_RAY_RUNTIME_ENV = {
    "env_vars": {
        "TOKENIZERS_PARALLELISM": "true",
        "NCCL_DEBUG": "WARN",
        "VLLM_LOGGING_LEVEL": "WARN",
        "VLLM_ALLOW_RUNTIME_LORA_UPDATING": "true",
        "CUDA_DEVICE_MAX_CONNECTIONS": "1",
        # TODO: disable compile cache due to cache corruption issue
        # https://github.com/vllm-project/vllm/issues/31199
        "VLLM_DISABLE_COMPILE_CACHE": "1",
        # To prevent hanging or crash during synchronization of weights between actor and rollout
        # in disaggregated mode. See:
        # https://docs.vllm.ai/en/latest/usage/troubleshooting.html?h=nccl_cumem_enable#known-issues
        # https://github.com/vllm-project/vllm/blob/c6b0a7d3ba03ca414be1174e9bd86a97191b7090/vllm/worker/worker_base.py#L445
        "NCCL_CUMEM_ENABLE": "0",
    },
}

FORWARD_PREFIXES = [
    "VLLM_",
    "SGL_",
    "SGLANG_",
    "HF_",
    "TOKENIZERS_",
    "DATASETS_",
    "TORCH_",
    "PYTORCH_",
    "DEEPSPEED_",
    "MEGATRON_",
    "NCCL_",
    "CUDA_",
    "CUBLAS_",
    "CUDNN_",
    "NV_",
    "NVIDIA_",
    "WANDB_",
    "RLLM_",
    "DEEPRESEARCH_",
]


def _get_forwarded_env_vars():
    """
    Get the forwarded environment variables. The `RLLM_EXCLUDE` environment variable can be used to
    exclude specific environment variables or all variables with a specific prefix.

    Example:
    ```
    RLLM_EXCLUDE=VLLM*,CUDA*,NCCL_IB_DISABLE
    ```
    will exclude all variables with prefix `VLLM_`, `CUDA_`, and `NCCL_IB_DISABLE`.

    By default, all environment variables with prefix in `FORWARD_PREFIXES` are forwarded.
    """
    if os.environ.get("RLLM_EXCLUDE", None) is not None:
        rllm_exclude = str(os.environ.get("RLLM_EXCLUDE")).split(",")
    else:
        rllm_exclude = []

    forward_prefix = FORWARD_PREFIXES.copy()

    # RLLM_EXCLUDE is a control var read on the launching node; it matches the
    # RLLM_ prefix but must never be forwarded into workers.
    exclude_vars = {"RLLM_EXCLUDE"}
    for name in rllm_exclude:
        if "*" in name:  # denote a prefix match, e.g. "VLLM*"
            prefix = name.replace("*", "_")
            try:
                forward_prefix.remove(prefix)
            except ValueError:
                pass
        else:
            exclude_vars.add(name)

    forwarded = {k: v for k, v in os.environ.items() if any(k.startswith(p) for p in forward_prefix) and k not in exclude_vars}
    if "PYTHONPATH" in os.environ:
        forwarded["PYTHONPATH"] = os.environ["PYTHONPATH"]
    return forwarded


def get_ppo_ray_runtime_env():
    """Build the runtime_env to pass to ray.init().

    Priority (low → high):
      1. PPO_RAY_RUNTIME_ENV — rllm defaults
      2. forwarded host env vars (VLLM_*, NCCL_*, CUDA_*, etc.)
      3. RAY_JOB_CONFIG_JSON_ENV_VAR — runtime_env from `ray job submit --runtime-env-json=...`

    Ray's ray.init() will merge the runtime_env we return here with the job config's
    runtime_env, and raises on any key conflict unless RAY_OVERRIDE_JOB_RUNTIME_ENV=1.
    We avoid that by popping any key the job config sets from our returned dict, so
    the job config's value wins.
    """
    env = PPO_RAY_RUNTIME_ENV.get("env_vars", {}).copy()
    env.update(_get_forwarded_env_vars())

    # Parse the job-submission runtime_env (if launched via `ray job submit`)
    try:
        job_runtime_env = json.loads(os.environ.get(RAY_JOB_CONFIG_JSON_ENV_VAR, "{}")).get("runtime_env", {}) or {}
    except (json.JSONDecodeError, TypeError):
        job_runtime_env = {}

    # Pop keys that the job config sets — let the job config's values win during ray.init merge
    for key in job_runtime_env.get("env_vars", {}) or {}:
        env.pop(key, None)

    runtime_env = {"env_vars": env}
    if (
        job_runtime_env.get("worker_process_setup_hook") is None
        and os.environ.get("RLLM_DEFER_VERL_WORKER_PATCHES") != "1"
    ):
        runtime_env["worker_process_setup_hook"] = "rllm.trainer.verl.patch.apply_all_verl_patches"
    # Only set working_dir=None when the job config doesn't specify one (avoid merge conflict)
    if job_runtime_env.get("working_dir") is None:
        runtime_env["working_dir"] = None
    # #region debug-point D:runtime-env
    try:
        url = "http://127.0.0.1:7777/event"
        session_id = "deepresearch-lsd-run"
        try:
            with open("/opt/tiger/rllm/.dbg/deepresearch-lsd-run.env", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line.startswith("DEBUG_SERVER_URL="):
                        url = line.split("=", 1)[1]
                    elif line.startswith("DEBUG_SESSION_ID="):
                        session_id = line.split("=", 1)[1]
        except OSError:
            pass
        gpu_env = {
            key: env.get(key)
            for key in (
                "CUDA_VISIBLE_DEVICES",
                "NVIDIA_VISIBLE_DEVICES",
                "RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES",
                "RLLM_EXCLUDE",
                "NCCL_DEBUG",
            )
        }
        urllib.request.urlopen(
            urllib.request.Request(
                url,
                data=json.dumps(
                    {
                        "sessionId": session_id,
                        "runId": os.environ.get("TRAE_DEBUG_RUN_ID", "pre-fix"),
                        "hypothesisId": "D",
                        "location": "ray_runtime_env.py:get_ppo_ray_runtime_env",
                        "msg": "[DEBUG] ray runtime env gpu vars",
                        "data": {
                            "gpu_env": gpu_env,
                            "has_worker_hook": runtime_env.get("worker_process_setup_hook") is not None,
                            "forwarded_key_count": len(env),
                        },
                    },
                    default=str,
                ).encode(),
                headers={"Content-Type": "application/json"},
            ),
            timeout=2,
        ).read()
    except Exception:
        pass
    # #endregion
    return runtime_env
