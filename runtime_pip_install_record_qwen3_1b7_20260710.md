# Qwen3-1.7B DeepScaleR Runtime Pip Install Record

Date: 2026-07-10

Purpose: make the current Python runtime compatible with this LSD checkout for
`scripts/run_qwen3_1b7_math_4k.sh`.

## Current Final Runtime

After upgrading `vllm` to match `scripts/install_vllm_sglang_mcore.sh`, the
runtime was aligned to the `vllm==0.11.0` stack. A mixed install was observed
where `vllm/torchvision/torchaudio` were on the 2.8 stack but `torch` still
imported from `/usr/local/lib/python3.11/dist-packages` as `2.4.1`, causing:

```text
ImportError: cannot import name 'DiagnosticOptions' from 'torch.onnx._internal.exporter'
```

This was fixed by explicitly installing `torch==2.8.0` and its CUDA 12.8
runtime dependencies into the user site so they take precedence over the old
global install.

Additional commands applied:

```bash
python3 -m pip install --user --upgrade 'vllm==0.11.0'
python3 -m pip install --user --upgrade 'flash-attn==2.8.3.post1'
python3 -m pip install --user --upgrade --force-reinstall --no-deps 'torch==2.8.0'
python3 -m pip install --user --upgrade --force-reinstall --no-deps \
  'nvidia-cuda-nvrtc-cu12==12.8.93' \
  'nvidia-cuda-runtime-cu12==12.8.90' \
  'nvidia-cuda-cupti-cu12==12.8.90' \
  'nvidia-cudnn-cu12==9.10.2.21' \
  'nvidia-cublas-cu12==12.8.4.1' \
  'nvidia-cufft-cu12==11.3.3.83' \
  'nvidia-cufile-cu12==1.13.1.3' \
  'nvidia-curand-cu12==10.3.9.90' \
  'nvidia-cusolver-cu12==11.7.3.90' \
  'nvidia-cusparse-cu12==12.5.8.93' \
  'nvidia-cusparselt-cu12==0.7.1' \
  'nvidia-nccl-cu12==2.27.3' \
  'nvidia-nvtx-cu12==12.8.90' \
  'nvidia-nvjitlink-cu12==12.8.93' \
  'triton==3.4.0'
```

Current key versions:

```text
torch==2.8.0
torchvision==0.23.0
torchaudio==2.8.0
vllm==0.11.0
ray==2.56.0
transformers==4.55.4
tensordict==0.10.0
torchdata==0.11.0
flash-attn==2.8.3.post1
sympy==1.14.0
xformers==0.0.32.post1
xgrammar==0.1.25
```

Validation after the fix:

```bash
python3 - <<'PY'
import torch
import torch.onnx
import tensordict
import vllm
import flash_attn
from torch.distributed.tensor import DTensor
print(torch.__version__, torch.version.cuda, torch.cuda.is_available())
print("torch.onnx ok")
print("DTensor ok")
print(tensordict.__version__, vllm.__version__, flash_attn.__version__)
PY
```

## Commands Applied On Top Of The Existing Environment

The existing environment had `torch==2.4.1`, `ray==2.10.0`, no `vllm`, and no
`math_verify`. `torch==2.4.1` could not import `DTensor`, which this LSD code
requires.

```bash
python3 -m pip install --upgrade --no-deps \
  'torch==2.6.0' 'torchvision==0.21.0' 'torchaudio==2.6.0'
```

```bash
python3 -m pip install --upgrade --no-deps \
  'nvidia-cuda-nvrtc-cu12==12.4.127' \
  'nvidia-cuda-runtime-cu12==12.4.127' \
  'nvidia-cuda-cupti-cu12==12.4.127' \
  'nvidia-cudnn-cu12==9.1.0.70' \
  'nvidia-cublas-cu12==12.4.5.8' \
  'nvidia-cufft-cu12==11.2.1.3' \
  'nvidia-curand-cu12==10.3.5.147' \
  'nvidia-cusolver-cu12==11.6.1.9' \
  'nvidia-cusparse-cu12==12.3.1.170' \
  'nvidia-cusparselt-cu12==0.6.2' \
  'nvidia-nccl-cu12==2.21.5' \
  'nvidia-nvtx-cu12==12.4.127' \
  'nvidia-nvjitlink-cu12==12.4.127'
```

```bash
python3 -m pip install --user --upgrade 'sympy==1.13.1'
python3 -m pip install --user --upgrade pylatexenc
```

```bash
python3 -m pip install --user --upgrade \
  'tensordict==0.10.0' 'torchdata==0.11.0' 'ray==2.43.0' \
  'transformers[hf_xet]==4.55.4' 'optree>=0.13.0' \
  'nvidia-ml-py>=12.560.30' 'grpcio>=1.62.1' \
  'pydantic>=2.9' 'fastapi>=0.115.0' \
  hf-transfer mathruler math-verify 'xgrammar==0.2.2' \
  qwen-vl-utils liger-kernel
```

```bash
python3 -m pip install --user --upgrade 'vllm==0.8.4'
```

Note: `vllm==0.8.4` resolves `xgrammar` to `0.1.18`, overriding the earlier
explicit `xgrammar==0.2.2`.

## Final Key Versions

```text
torch==2.6.0
torchvision==0.21.0
torchaudio==2.6.0
vllm==0.8.4
ray==2.43.0
transformers==4.55.4
tensordict==0.10.0
torchdata==0.11.0
sympy==1.13.1
pylatexenc==2.10
math-verify==0.9.0
mathruler==0.1.0
xformers==0.0.29.post2
xgrammar==0.1.18
nvidia-ml-py==13.610.43
fastapi==0.139.0
pydantic==2.13.4
grpcio==1.82.1
optree==0.19.1
liger-kernel==0.8.0
```

## Verification

These checks passed:

```bash
python3 - <<'PY'
import torch
from torch.distributed.tensor import DTensor
import vllm, ray, transformers, tensordict, math_verify
print(torch.__version__, torch.version.cuda, torch.cuda.is_available())
print(vllm.__version__, ray.__version__, transformers.__version__, tensordict.__version__)
print("DTensor ok")
PY
```

```bash
HYDRA_FULL_ERROR=1 bash /opt/tiger/LSD/scripts/run_qwen3_1b7_math_4k.sh --cfg job
```

The second command completed successfully and printed the composed Hydra config.
