# Source provenance

Imported on 2026-11-23 from the original experiment checkout. The source directory
had no Git metadata, so no upstream commit ID is asserted for this snapshot.
The four core files below are copied byte-for-byte from that checkout.

Included: rLLM source, model gateway, vendored VERL, math/deep-research/tool-agent
recipes, supporting examples, scripts, and tests.

Excluded: credentials, local environment files, runtime environments, Git
metadata, caches, logs, W&B exports, datasets, model weights, checkpoints,
plotting artifacts and backup files. This is a selected source export, not a
filesystem backup. Source files larger than 2 MB were excluded.

Release-only changes remove embedded W&B credentials from scripts, make the
math launchers' W&B login conditional on an environment-provided key, and add
these documentation files. Training algorithms and historical hyperparameters
are unchanged. Additional rebuttal code from other machines is not included.

## Core source SHA-256

- `rllm/trainer/lsd.py`: `ba47348c4e039401c63256f89119513932512c87a891d3edf1ddd24372656b04`
- `rllm/trainer/verl/verl_backend.py`: `84de0549ab2c3192fa0227fdb6985be35d45ff56ffa3e8200081af60722f9ab3`
- `rllm/trainer/verl/patch.py`: `646f07bef0d351a81ada80d5674d1d868b2fe4800c356c95e1d021ef03eea64b`
- `rllm/trainer/config/rllm/base.yaml`: `bef10fb08a61634c255735f543bd5c953824e9289ef57fd2dddbd7969aa04978`
