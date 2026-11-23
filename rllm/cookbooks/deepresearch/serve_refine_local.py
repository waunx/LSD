#!/usr/bin/env python3
"""Launch independent Qwen3-8B vLLM replicas and publish only ready endpoints."""

from __future__ import annotations

import argparse
import csv
import ipaddress
import json
import os
import shlex
import signal
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from contextlib import suppress
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_STATE = REPO_ROOT / "logs/refine_local/servers.json"
MODEL_NAME = "Qwen/Qwen3-8B"
REMOTE_HOST = "2605:340:cd51:4900:8b45:1059:2a1:e33f"
REMOTE_URLS = [f"http://[{REMOTE_HOST}]:{port}/v1" for port in (40003, 40004, 40005, 40006, 40007, 40008, 40009, 40011)]


def request_json(url: str, payload: dict | None = None, timeout: float = 5) -> dict:
    # Internal IPv6 services must not go through the user's outbound proxy.
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    data = json.dumps(payload).encode() if payload is not None else None
    request = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    with opener.open(request, timeout=timeout) as response:
        content = response.read()
    return json.loads(content) if content else {}


def check_endpoint(url: str, min_context: int) -> dict:
    request_json(url.removesuffix("/v1") + "/health", timeout=2)
    models = request_json(url + "/models", timeout=2)
    for model in models.get("data", []):
        if model.get("id") == MODEL_NAME and int(model.get("max_model_len", 0)) >= min_context:
            return model
    raise ValueError(f"{url} does not serve {MODEL_NAME} with context >= {min_context}")


def smoke_endpoint(url: str) -> None:
    response = request_json(
        url + "/chat/completions",
        {
            "model": MODEL_NAME,
            "messages": [
                {
                    "role": "user",
                    "content": ("Summarize this fact: Paris is the capital of France. Put the summary inside <information> and </information> tags."),
                }
            ],
            "temperature": 0.6,
            "top_p": 0.95,
            "top_k": 20,
            "max_tokens": 1024,
        },
        timeout=120,
    )
    content = response["choices"][0]["message"].get("content") or ""
    if not all(tag in content for tag in ("<think>", "</think>", "<information>", "</information>")):
        raise ValueError(f"{url} returned output incompatible with the Refine parser")


def resolve_urls(state_file: Path) -> list[str]:
    urls = list(REMOTE_URLS)
    if not state_file.exists():
        return urls
    state = json.loads(state_file.read_text())
    for server in state["servers"]:
        url = server["url"]
        try:
            check_endpoint(url, int(state["context_length"]))
        except (OSError, ValueError) as exc:
            print(f"Skipping unavailable local Refine {url}: {exc}", file=sys.stderr)
            continue
        urls.append(url)
    return list(dict.fromkeys(urls))


def validate_model(model_path: Path) -> None:
    config = json.loads((model_path / "config.json").read_text())
    tokenizer = json.loads((model_path / "tokenizer_config.json").read_text())
    if config.get("model_type") != "qwen3" or config.get("hidden_size") != 4096:
        raise ValueError("Refine requires Qwen3-8B, not a different model size")
    if not tokenizer.get("chat_template") or "base" in model_path.name.lower():
        raise ValueError("Refine requires the chat Qwen3-8B model, not Qwen3-8B-Base")
    index = json.loads((model_path / "model.safetensors.index.json").read_text())
    for filename in set(index["weight_map"].values()):
        if not (model_path / filename).is_file():
            raise FileNotFoundError(f"Missing model shard: {filename}")


def parse_gpu_ids(value: str) -> list[int]:
    ids = [int(item) for item in value.split(",")]
    if not ids or len(ids) != len(set(ids)) or any(gpu < 0 for gpu in ids):
        raise ValueError("GPU IDs must be unique nonnegative integers")
    return ids


def free_gpu_ids(requested: str) -> list[int]:
    def query(fields: str, kind: str) -> list[list[str]]:
        result = subprocess.run(
            ["nvidia-smi", f"--query-{kind}={fields}", "--format=csv,noheader,nounits"],
            check=True,
            capture_output=True,
            text=True,
            timeout=15,
        )
        return list(csv.reader(result.stdout.splitlines(), skipinitialspace=True))

    gpus = query("index,uuid,memory.total,memory.used", "gpu")
    busy = {row[0].strip() for row in query("gpu_uuid,pid", "compute-apps")}
    available = [int(index) for index, uuid, total, used in gpus if uuid.strip() not in busy and float(used) < 1024 and float(total) >= 80000]
    selected = available if requested == "auto" else parse_gpu_ids(requested)
    if not selected or any(gpu not in available for gpu in selected):
        raise RuntimeError(f"No requested idle H20-class GPUs available; idle GPU IDs: {available}. Existing jobs are not stopped automatically.")
    return selected


def model_overrides(args: argparse.Namespace) -> dict:
    if not 32768 <= args.context_length <= 131072:
        raise ValueError("Context length must be between 32768 and 131072")
    if args.context_length <= 40960:
        return {}
    config = json.loads((args.model_path / "config.json").read_text())
    rope = {
        "rope_type": "yarn",
        "factor": args.context_length / 32768,
        "original_max_position_embeddings": 32768,
        "rope_theta": config["rope_theta"],
    }
    # vLLM 0.15 applies dict overrides after normalizing legacy rope_scaling.
    # Its model and length validator both consume rope_parameters.
    return {"rope_parameters": rope, "rope_scaling": dict(rope)}


def build_command(args: argparse.Namespace, gpu: int) -> list[str]:
    overrides = model_overrides(args)
    command = [
        args.python,
        "-m",
        "vllm.entrypoints.openai.api_server",
        "--model",
        str(args.model_path),
        "--served-model-name",
        MODEL_NAME,
        "--host",
        "::",
        "--port",
        str(args.base_port + gpu),
        "--tensor-parallel-size",
        "1",
        "--dtype",
        "bfloat16",
        "--max-model-len",
        str(args.context_length),
        "--gpu-memory-utilization",
        "0.8",
        "--max-num-seqs",
        "32",
        "--max-num-batched-tokens",
        "8192",
        "--enable-chunked-prefill",
        "--enforce-eager",
        "--disable-log-requests",
    ]
    if overrides:
        command += ["--hf-overrides", json.dumps(overrides)]
    # Do not enable a reasoning parser: refine_agent expects raw <think> tags.
    return command


def server_environment(python: str, gpu: int | None) -> dict[str, str]:
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    env.pop("RAY_ADDRESS", None)
    env["CUDA_VISIBLE_DEVICES"] = "" if gpu is None else str(gpu)
    env["PYTORCH_ALLOC_CONF"] = "expandable_segments:False"
    env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:False"
    env["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
    env["VLLM_ALLOW_LONG_MAX_MODEL_LEN"] = "0"
    env["TOKENIZERS_PARALLELISM"] = "false"
    env["PYTHONUNBUFFERED"] = "1"
    site = subprocess.check_output(
        [python, "-c", "import site; print(site.getsitepackages()[0])"],
        env=env,
        text=True,
        timeout=15,
    ).strip()
    libs = [Path(site) / "torch/lib", *sorted((Path(site) / "nvidia").glob("*/lib"))]
    env["LD_LIBRARY_PATH"] = ":".join(map(str, libs)) + ":" + env.get("LD_LIBRARY_PATH", "")
    # Preserve existing preloads, including any sandbox instrumentation.
    return env


def start(args: argparse.Namespace) -> None:
    validate_model(args.model_path)
    if not os.access(args.python, os.X_OK):
        raise FileNotFoundError(args.python)
    host = ipaddress.ip_address(args.advertise_host.strip("[]"))
    if host.is_loopback or host.is_unspecified:
        raise ValueError("Use a routable host IP, not localhost, for remote training clients")
    ids = parse_gpu_ids(args.gpu_ids) if args.dry_run and args.gpu_ids != "auto" else None
    if args.dry_run and ids is None:
        raise ValueError("Specify --gpu-ids for a GPU-free --dry-run")
    if ids is None:
        ids = free_gpu_ids(args.gpu_ids)
    commands = {gpu: build_command(args, gpu) for gpu in ids}
    if args.dry_run:
        for gpu, command in commands.items():
            print(f"CUDA_VISIBLE_DEVICES={gpu} {shlex.join(command)}")
        print("DRY RUN: no GPU access, process launch or pool publication")
        return
    if args.state.exists():
        raise RuntimeError(f"State already exists: {args.state}; inspect existing replicas first")

    for gpu in ids:
        with socket.socket(socket.AF_INET6, socket.SOCK_STREAM) as probe:
            probe.bind(("::", args.base_port + gpu))

    args.state.parent.mkdir(parents=True, exist_ok=True)
    processes = []
    servers = []
    try:
        for gpu, command in commands.items():
            port = args.base_port + gpu
            log = args.state.parent / f"gpu{gpu}_port{port}.log"
            with log.open("ab") as handle:
                process = subprocess.Popen(
                    command,
                    env=server_environment(args.python, gpu),
                    stdin=subprocess.DEVNULL,
                    stdout=handle,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
            processes.append(process)
            authority = f"[{host}]" if host.version == 6 else str(host)
            servers.append({"gpu": gpu, "pid": process.pid, "url": f"http://{authority}:{port}/v1", "log": str(log)})
            print(f"Loading Refine on GPU {gpu}, pid={process.pid}, log={log}", flush=True)

        deadline = time.monotonic() + args.startup_timeout
        pending = list(servers)
        while pending:
            if any(process.poll() is not None for process in processes):
                raise RuntimeError("A Refine server exited during startup; inspect the per-GPU logs")
            for server in pending[:]:
                try:
                    check_endpoint(server["url"], args.context_length)
                except (OSError, ValueError):
                    continue
                smoke_endpoint(server["url"])
                pending.remove(server)
                print(f"Ready: {server['url']}", flush=True)
            if pending:
                if time.monotonic() >= deadline:
                    raise TimeoutError(f"Refine startup timed out: {pending}")
                time.sleep(5)

        state = {"model": MODEL_NAME, "context_length": args.context_length, "servers": servers}
        temporary = args.state.with_suffix(".tmp")
        temporary.write_text(json.dumps(state, indent=2) + "\n")
        temporary.replace(args.state)
        print(f"Published {len(servers)} ready replicas: {args.state}")
    except BaseException:
        # Only terminate children started by this invocation, never old jobs.
        for process in processes:
            with suppress(ProcessLookupError):
                os.killpg(process.pid, signal.SIGTERM)
        for process in processes:
            try:
                process.wait(timeout=20)
            except subprocess.TimeoutExpired:
                with suppress(ProcessLookupError):
                    os.killpg(process.pid, signal.SIGKILL)
                process.wait()
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("start", "resolve", "status"))
    parser.add_argument("--state", type=Path, default=DEFAULT_STATE)
    parser.add_argument("--python", default=str(REPO_ROOT / ".runtime/rllm_fully_async_venv_seqmean/bin/python"))
    parser.add_argument("--model-path", type=Path, default=Path("/tmp/deepresearch_seqmean_models/Qwen3-8B"))
    parser.add_argument("--gpu-ids", default="auto")
    parser.add_argument("--base-port", type=int, default=40100)
    parser.add_argument("--context-length", type=int, default=65536)
    parser.add_argument("--advertise-host", default=os.environ.get("BYTED_HOST_IPV6", ""))
    parser.add_argument("--startup-timeout", type=float, default=900)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    try:
        if args.action == "resolve":
            print(",".join(resolve_urls(args.state)))
        elif args.action == "status":
            for url in resolve_urls(args.state):
                try:
                    model = check_endpoint(url, 32768)
                    print(f"{url} ready context={model['max_model_len']}")
                except (OSError, ValueError) as exc:
                    print(f"{url} unavailable: {exc}")
        else:
            start(args)
    except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as exc:
        parser.exit(1, f"Refine setup failed: {exc}\n")


if __name__ == "__main__":
    main()
