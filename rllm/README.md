# LSD: rLLM implementation

Source snapshot for **Mitigating the Length-Scaling Tax with Online Distillation**.

This directory contains the rLLM-based LSD implementation, its vendored VERL
backend, the model gateway, and historical math and deep-research recipes.
The repository root preserves the earlier VERL checkout; run this implementation
from this directory and use `vendor/verl` on `PYTHONPATH`.

## Code map

| Component | Location |
| --- | --- |
| Routing and LSD configuration | `rllm/trainer/lsd.py` |
| Distillation losses and training integration | `rllm/trainer/verl/verl_backend.py` |
| EMA teacher, checkpoint and tensor-layout patches | `rllm/trainer/verl/patch.py` |
| Default configuration | `rllm/trainer/config/rllm/base.yaml` |
| Math training and launch scripts | `cookbooks/math/` |
| Multi-turn deep-research recipes | `cookbooks/deepresearch/` |
| Deep-research tools | `examples/fully_async/deepresearch/` |
| Vendored backend used by this implementation | `vendor/verl/` |
| LLM trace gateway | `rllm-model-gateway/` |

## Environment and inputs

The historical experiment runtime used Python 3.12.13, PyTorch 2.9.1,
vLLM 0.15.0, Ray 2.57, and Transformers 4.57.1. The inherited package metadata
also contains newer optional dependency ranges; it is not an experiment lockfile.
A fresh installation and end-to-end GPU reproduction have not been validated
for this source release.

From this directory, configure the import paths before running a recipe:

```bash
export PYTHONPATH="$PWD:$PWD/vendor/verl:$PWD/rllm-model-gateway/src:${PYTHONPATH:-}"
```

Model weights, datasets, dataset registration, checkpoints and external retrieval
services must be supplied separately. Historical launch scripts retain their
experiment-specific paths and defaults. Set `PYTHON_BIN`, `RLLM_HOME`,
`MODEL_PATH`, output paths, and service endpoints for your environment.
W&B credentials, when needed, must be supplied through `WANDB_API_KEY`.

This is an initial source snapshot, not a claim that every historical launcher
reproduces the final paper configuration. See [SOURCE.md](SOURCE.md) for scope
and provenance, and [README.upstream.md](README.upstream.md) for upstream rLLM
documentation. Upstream licenses and copyright notices are retained.

## Focused tests

In an environment with the training dependencies installed:

```bash
CUDA_VISIBLE_DEVICES="" python -m pytest --noconftest -q \
  tests/trainer/verl/test_lsd_topk_patch.py \
  tests/unified_trainer/test_lsd_loss_aggregation.py
```

These tests cover LSD loss aggregation, gradients, and teacher tensor handling;
they do not run model training.
