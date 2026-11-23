"""Monkey-patches for Verl and vLLM within the rLLM unified trainer.

All patches are applied lazily (on first call) and are idempotent — calling
them multiple times is safe.
"""

from __future__ import annotations

import logging
import json
import os
import urllib.request

logger = logging.getLogger(__name__)

_VERL_DYNAMIC_BATCH_PATCHED = False
_VERL_QWEN3_VL_DUMMY_INPLACE_PATCHED = False
_VERL_TENSORDICT_JAGGED_PATCHED = False
_VERL_FSDP_TOPK_OUTPUTS_PATCHED = False
_VERL_TRAINING_WORKER_ENV_DEBUG_PATCHED = False
_VERL_SGLANG_SLEEP_DRAIN_PATCHED = False
_VERL_FSDP_REDUCE_SCATTER_CACHE_PATCHED = False
_VERL_FSDP_TRAIN_OUTPUT_RELEASE_PATCHED = False
_SGLANG_OPENAI_TOKEN_IDS_PATCHED = False


# #region debug-point E:patch-report
def _debug_report_patch(hypothesis_id: str, location: str, msg: str, data: dict | None = None) -> None:
    try:
        session_id = os.environ.get("RLLM_DEBUG_SESSION_ID")
        if not session_id:
            return
        url = "http://127.0.0.1:7777/event"
        env_path = f"/opt/tiger/rllm/.dbg/{session_id}.env"
        try:
            with open(env_path, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line.startswith("DEBUG_SERVER_URL="):
                        url = line.split("=", 1)[1]
                    elif line.startswith("DEBUG_SESSION_ID="):
                        session_id = line.split("=", 1)[1]
        except OSError:
            pass
        urllib.request.urlopen(
            urllib.request.Request(
                url,
                data=json.dumps(
                    {
                        "sessionId": session_id,
                        "runId": os.environ.get("RLLM_DEBUG_RUN_ID", os.environ.get("TRAE_DEBUG_RUN_ID", "pre-fix")),
                        "hypothesisId": hypothesis_id,
                        "location": location,
                        "msg": f"[DEBUG] {msg}",
                        "data": data or {},
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


# #region debug-point C:training-worker-env-wrapper
def patch_verl_training_worker_env_debug() -> None:
    global _VERL_TRAINING_WORKER_ENV_DEBUG_PATCHED
    if _VERL_TRAINING_WORKER_ENV_DEBUG_PATCHED:
        return
    try:
        from verl.workers.engine_workers import TrainingWorker
    except Exception:
        return

    original_init = TrainingWorker.__init__

    def wrapped_init(self, *args, **kwargs):
        _debug_report_patch(
            "C",
            "patch.py:TrainingWorker.__init__",
            "training worker before init",
            {
                "rank": os.environ.get("RANK"),
                "world_size": os.environ.get("WORLD_SIZE"),
                "local_rank": os.environ.get("LOCAL_RANK"),
                "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
                "nvidia_visible_devices": os.environ.get("NVIDIA_VISIBLE_DEVICES"),
            },
        )
        original_init(self, *args, **kwargs)
        _debug_report_patch(
            "C",
            "patch.py:TrainingWorker.__init__",
            "training worker after init",
            {
                "rank": os.environ.get("RANK"),
                "world_size": os.environ.get("WORLD_SIZE"),
                "local_rank": os.environ.get("LOCAL_RANK"),
                "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
                "nvidia_visible_devices": os.environ.get("NVIDIA_VISIBLE_DEVICES"),
            },
        )

    TrainingWorker.__init__ = wrapped_init
    _VERL_TRAINING_WORKER_ENV_DEBUG_PATCHED = True
# #endregion


def patch_sglang_openai_token_ids() -> None:
    """Preserve SGLang's exact output IDs in OpenAI response metadata."""
    global _SGLANG_OPENAI_TOKEN_IDS_PATCHED
    if _SGLANG_OPENAI_TOKEN_IDS_PATCHED:
        return

    import functools

    from sglang.srt.entrypoints.openai.serving_chat import OpenAIServingChat

    original_build_chat_response = OpenAIServingChat._build_chat_response

    @functools.wraps(original_build_chat_response)
    def wrapped_build_chat_response(self, request, ret, created):
        response = original_build_chat_response(self, request, ret, created)
        if hasattr(response, "metadata"):
            metadata = dict(response.metadata or {})
            metadata["completion_token_ids"] = [
                list(item.get("output_ids") or []) for item in ret
            ]
            response.metadata = metadata
        return response

    OpenAIServingChat._build_chat_response = wrapped_build_chat_response
    _SGLANG_OPENAI_TOKEN_IDS_PATCHED = True
    logger.info("Patched SGLang OpenAI responses to preserve exact completion token IDs")


def patch_verl_sglang_sleep_drain() -> None:
    """Install SGLang HTTP-server lifecycle fixes before Ray wraps the class."""
    global _VERL_SGLANG_SLEEP_DRAIN_PATCHED
    if _VERL_SGLANG_SLEEP_DRAIN_PATCHED:
        return

    import asyncio

    from verl.workers.rollout.sglang_rollout.async_sglang_server import SGLangHttpServer

    original_launch_server = getattr(SGLangHttpServer, "launch_server", None)
    original_sleep = SGLangHttpServer.sleep

    async def wrapped_launch_server(self, *args, **kwargs):
        # Run this inside the dedicated SGLang actor. The regular VERL recipe
        # defers the global worker hook to preserve Ray's GPU assignment.
        patch_sglang_openai_token_ids()
        assert original_launch_server is not None
        return await original_launch_server(self, *args, **kwargs)

    async def wrapped_sleep(self):
        mode_name = getattr(self.rollout_mode, "name", str(self.rollout_mode)).upper()
        should_drain = (
            self.node_rank == 0
            and self.config.free_cache_engine
            and mode_name in {"HYBRID", "COLOCATED"}
        )
        if should_drain:
            timeout_s = float(os.environ.get("RLLM_SGLANG_DRAIN_TIMEOUT_S", "30"))
            deadline = asyncio.get_running_loop().time() + timeout_s
            attempt = 0
            while True:
                attempt += 1
                self.tokenizer_manager.abort_request(abort_all=True)
                requests_active = await self.tokenizer_manager.model_update_lock.is_locked()
                # #region debug-point C:sglang-drain
                _debug_report_patch(
                    "C",
                    "patch.py:SGLangHttpServer.sleep",
                    "SGLang request drain before sleep",
                    {
                        "attempt": attempt,
                        "requests_active": bool(requests_active),
                        "replica_rank": self.replica_rank,
                        "rollout_mode": mode_name,
                    },
                )
                # #endregion
                if not requests_active:
                    break
                if asyncio.get_running_loop().time() >= deadline:
                    raise TimeoutError(
                        f"SGLang requests did not drain within {timeout_s:.1f}s before sleep "
                        f"(replica_rank={self.replica_rank})"
                    )
                await asyncio.sleep(0.1)

        await original_sleep(self)

    if original_launch_server is not None:
        SGLangHttpServer.launch_server = wrapped_launch_server
    SGLangHttpServer.sleep = wrapped_sleep
    _VERL_SGLANG_SLEEP_DRAIN_PATCHED = True
    logger.info(
        "Patched SGLangHttpServer lifecycle for exact output IDs and request draining"
    )


# ---------------------------------------------------------------------------
# Verl dynamic batch: sync micro-batch counts across DP ranks
# ---------------------------------------------------------------------------


def patch_verl_dynamic_batch_sync() -> None:
    """Patch ``prepare_dynamic_batch`` to sync micro-batch counts across DP ranks.

    Fixes `verl#5750 <https://github.com/verl-project/verl/issues/5750>`_:
    when ``use_dynamic_bsz=True``, each DP rank independently calculates
    ``num_micro_batches`` based on its local sequence lengths.  Different
    ranks can end up with different counts, causing NCCL collective
    operations (AllGather/ReduceScatter in FSDP) to deadlock.

    The fix defaults ``dp_group`` to ``torch.distributed.group.WORLD`` so
    that ``prepare_dynamic_batch`` performs an ``all_reduce(MAX)`` across
    ranks, forcing every rank to iterate through the same number of
    micro-batches.  This is the same approach as verl PR #5591.
    """
    global _VERL_DYNAMIC_BATCH_PATCHED
    if _VERL_DYNAMIC_BATCH_PATCHED:
        return

    import verl.utils.seqlen_balancing as sbl

    _original_prepare = sbl.prepare_dynamic_batch

    def _patched_prepare(data, max_token_len, dp_group=None, **kwargs):
        if dp_group is None:
            import torch.distributed

            if torch.distributed.is_initialized():
                dp_group = torch.distributed.group.WORLD
        return _original_prepare(data, max_token_len, dp_group=dp_group, **kwargs)

    sbl.prepare_dynamic_batch = _patched_prepare

    # Also patch the already-imported reference in dp_actor so both
    # compute_log_prob and update_policy use the patched version.
    try:
        from verl.workers.actor import dp_actor

        dp_actor.prepare_dynamic_batch = _patched_prepare
    except (ImportError, AttributeError):
        pass  # dp_actor may not be importable outside GPU workers

    _VERL_DYNAMIC_BATCH_PATCHED = True
    logger.info("Patched prepare_dynamic_batch to sync micro-batch counts across DP ranks (verl#5750)")


# ---------------------------------------------------------------------------
# Verl qwen3_vl: out-of-place add in dummy visual forward (backport PR #5881)
# ---------------------------------------------------------------------------


def patch_verl_qwen3_vl_dummy_inplace() -> None:
    """Backport `volcengine/verl#5881` for ``verl.models.transformers.qwen3_vl``.

    The dummy/no-image branch of ``_get_input_embeds`` mutates ``inputs_embeds``
    inplace::

        inputs_embeds += 0.0 * image_embeds.mean()
        for emb in dummy_deepstack_image_embeds or []:
            inputs_embeds += 0.0 * emb.mean()

    ``inputs_embeds`` is produced by ``model.get_input_embeddings()(input_ids)``
    and is a leaf with ``requires_grad=True`` after FSDP wrapping; the inplace
    ``+=`` raises a RuntimeError on the autograd backward and, in our 0.7.1
    pin, also surfaces as ``CUDNN_STATUS_NOT_INITIALIZED`` on the preceding
    ``model.visual(...)`` call due to the way the failure propagates inside
    Conv3d's cuDNN workspace setup. PR #5881 (merged main 2026-04-07; not in
    0.7.1) replaces both lines with out-of-place addition.

    We patch by re-exec'ing a fixed source of ``_get_input_embeds`` in the
    module's global namespace; ``qwen3_vl_base_forward`` looks up
    ``_get_input_embeds`` from module globals at call time, so the rebound
    function is picked up automatically.
    """
    global _VERL_QWEN3_VL_DUMMY_INPLACE_PATCHED
    if _VERL_QWEN3_VL_DUMMY_INPLACE_PATCHED:
        return

    import inspect

    from verl.models.transformers import qwen3_vl as mod

    src = inspect.getsource(mod._get_input_embeds)
    new_src = src.replace(
        "inputs_embeds += 0.0 * image_embeds.mean()",
        "inputs_embeds = inputs_embeds + 0.0 * image_embeds.mean()",
    ).replace(
        "inputs_embeds += 0.0 * emb.mean()",
        "inputs_embeds = inputs_embeds + 0.0 * emb.mean()",
    )

    if new_src == src:
        # Upstream already fixed (e.g. user bumped verl past 0.7.1).
        _VERL_QWEN3_VL_DUMMY_INPLACE_PATCHED = True
        logger.info("qwen3_vl PR #5881 patch: source already uses out-of-place add; nothing to patch.")
        return

    # Compile and exec into the module's globals so the rebound function shares
    # the module's namespace (torch, Optional, etc.) and so callers in the
    # same module (qwen3_vl_base_forward) pick up the patched version on the
    # next attribute lookup.
    exec(compile(new_src, mod.__file__, "exec"), mod.__dict__)

    _VERL_QWEN3_VL_DUMMY_INPLACE_PATCHED = True
    logger.info("Patched verl qwen3_vl._get_input_embeds: dummy visual path now uses out-of-place addition (backport of volcengine/verl#5881)")


# ---------------------------------------------------------------------------
# Verl tensordict utils: preserve _ragged_idx when rebuilding 3D NestedTensors
# (backport PR #6127)
# ---------------------------------------------------------------------------


def patch_verl_tensordict_jagged_layout() -> None:
    """Backport `volcengine/verl#6127` for ``verl.utils.tensordict_utils``.

    Fixes a bug in the rebuilding of 3D jagged NestedTensors after
    selection / chunking. ``torch.nested.as_nested_tensor(tensors,
    layout=torch.jagged)`` is ambiguous when all input tensors share the
    same last-dimension length — torch picks the wrong dimension as the
    jagged axis. For mRoPE ``position_ids`` with per-sample shape
    ``(num_heads, seq_len)``, this produces a rebuilt nested tensor whose
    ``_ragged_idx`` is 1 (heads dim) instead of 2 (seq dim), causing two
    downstream crashes:

    - ``index_select_tensor_dict`` → ``unbind()`` →
      ``torch.split(values, [batch_size], dim=ragged_idx-1)`` fails because
      ``values`` has total length = sum-of-row-lengths, not batch_size
      (hits the ``use_dynamic_bsz=True`` micro-batch partitioning path).
    - rmpad path's rotary cos/sin gets shape ``(B,)`` instead of ``(B, S)``
      and ``apply_rotary_pos_emb`` crashes with
      ``RuntimeError: The size of tensor a (<S>) must match the size of
      tensor b (<B>) at non-singleton dimension 2``.

    The fix introduces a ``nested_tensor_from_tensor_list(tensors,
    ragged_idx)`` helper that explicitly preserves the intended ragged
    dimension via ``torch.nested.nested_tensor_from_jagged`` plus an
    explicit ``_ragged_idx`` set, and uses it in ``concat_nested_tensors``,
    ``chunk_tensordict``, and ``index_select_tensor_dict``.

    Merged into verl main 2026-04-24; not in 0.7.1.
    """
    global _VERL_TENSORDICT_JAGGED_PATCHED
    if _VERL_TENSORDICT_JAGGED_PATCHED:
        return

    import torch
    from tensordict import TensorDict
    from verl.utils import tensordict_utils as tu

    if hasattr(tu, "nested_tensor_from_tensor_list"):
        # Upstream already provides the helper (e.g. user bumped verl past 0.7.1).
        _VERL_TENSORDICT_JAGGED_PATCHED = True
        logger.info("verl tensordict jagged-layout patch: upstream already provides nested_tensor_from_tensor_list; nothing to patch.")
        return

    def nested_tensor_from_tensor_list(tensors, ragged_idx=None):
        assert len(tensors) > 0, "Must provide at least one tensor"
        sample_dim = tensors[0].dim()
        if ragged_idx is None:
            ragged_idx = sample_dim
        # #region debug-point E:jagged-concat-layout
        _debug_report_patch(
            "E",
            "patch.py:nested_tensor_from_tensor_list",
            "[DEBUG] rebuilding nested tensor from dense samples",
            {
                "ragged_idx": ragged_idx,
                "sample_dim": sample_dim,
                "sample_shapes": [list(tensor.shape) for tensor in tensors[:8]],
            },
        )
        # #endregion
        assert 1 <= ragged_idx <= sample_dim, (
            f"ragged_idx must address a sample dimension. Got {ragged_idx=} "
            f"and {sample_dim=}"
        )

        if sample_dim == 1:
            return torch.nested.as_nested_tensor(tensors, layout=torch.jagged)

        sample_ragged_idx = ragged_idx - 1
        values = torch.cat(tensors, dim=sample_ragged_idx)
        lengths = torch.tensor(
            [t.shape[sample_ragged_idx] for t in tensors],
            dtype=torch.long,
            device=values.device,
        )
        offsets = torch.zeros(len(tensors) + 1, dtype=torch.long, device=values.device)
        torch.cumsum(lengths, dim=0, out=offsets[1:])

        return torch.nested.nested_tensor_from_jagged(
            values=values,
            offsets=offsets,
            jagged_dim=ragged_idx,
        )

    def concat_nested_tensors(tensors):
        for tensor in tensors:
            assert tensor.is_nested and tensor.is_contiguous()
        unbind_tensors = []
        for tensor in tensors:
            assert len(tensor.shape) >= 2, f"nested tensor must have 2 or more dimensions. Got {tensor.shape}"
            unbind_tensors.extend(list(tensor.unbind(0)))
        ragged_idx = getattr(tensors[0], "_ragged_idx", tensors[0].dim() - 1)
        return nested_tensor_from_tensor_list(unbind_tensors, ragged_idx=ragged_idx)

    def chunk_tensordict(td, chunks):
        assert isinstance(td, TensorDict) and len(td) % chunks == 0, f"expecting td with length divisible by chunks, but got {len(td)} and {chunks}"
        chunk_size = len(td) // chunks
        nested_keys = {key for key, val in td.items() if isinstance(val, torch.Tensor) and val.is_nested}
        new_td = TensorDict({k: v for k, v in td.items() if k not in nested_keys}, batch_size=td.batch_size, device=td.device)
        tds = new_td.chunk(chunks=chunks)
        for key in nested_keys:
            nt = td[key]
            try:
                tensors = nt.unbind(dim=0)
            except RuntimeError:
                padded = nt.to_padded_tensor(0)
                padded_chunks = padded.chunk(chunks, dim=0)
                offsets = nt.offsets()
                lengths = offsets.diff().tolist()
                for i, chunk_td in enumerate(tds):
                    chunk_lengths = lengths[i * chunk_size : (i + 1) * chunk_size]
                    sample_ragged_idx = getattr(nt, "_ragged_idx", nt.dim() - 1) - 1
                    chunk_tensors = []
                    for j, seq_len in enumerate(chunk_lengths):
                        sample = padded_chunks[i][j]
                        slices = [slice(None)] * sample.dim()
                        slices[sample_ragged_idx] = slice(0, seq_len)
                        chunk_tensors.append(sample[tuple(slices)])
                    chunk_td[key] = nested_tensor_from_tensor_list(
                        chunk_tensors,
                        ragged_idx=getattr(nt, "_ragged_idx", nt.dim() - 1),
                    )
                continue
            for i, chunk_td in enumerate(tds):
                chunk_td[key] = nested_tensor_from_tensor_list(
                    list(tensors[i * chunk_size : (i + 1) * chunk_size]),
                    ragged_idx=getattr(nt, "_ragged_idx", nt.dim() - 1),
                )
        return tds

    def index_select_tensor_dict(batch, indices):
        if isinstance(indices, list):
            indices = torch.tensor(indices)
        assert indices.dim() == 1, "indices must be a 1D tensor"
        data_dict = {}
        batch_size = indices.shape[0]
        if batch is not None:
            for key, tensor in batch.items():
                if isinstance(tensor, torch.Tensor) and not tensor.is_nested:
                    data_dict[key] = tensor[indices]
                elif isinstance(tensor, torch.Tensor) and tensor.is_nested:
                    tensor_lst = tensor.unbind()
                    selected_tensors = [tensor_lst[idx] for idx in indices]
                    data_dict[key] = nested_tensor_from_tensor_list(
                        selected_tensors,
                        ragged_idx=getattr(tensor, "_ragged_idx", tensor.dim() - 1),
                    )
                else:
                    if tensor.shape:
                        data_dict[key] = tensor[indices]
                    else:
                        data_dict[key] = tensor
            selected_batch = TensorDict(source=data_dict, batch_size=batch_size)
        else:
            selected_batch = None
        return selected_batch

    tu.nested_tensor_from_tensor_list = nested_tensor_from_tensor_list
    tu.concat_nested_tensors = concat_nested_tensors
    tu.chunk_tensordict = chunk_tensordict
    tu.index_select_tensor_dict = index_select_tensor_dict

    _VERL_TENSORDICT_JAGGED_PATCHED = True
    logger.info("Patched verl.utils.tensordict_utils: rebuild jagged NestedTensors with explicit _ragged_idx (backport of volcengine/verl#6127)")


def patch_verl_fsdp_topk_processor_outputs() -> None:
    """Allow FSDP logits-processor outputs to carry a top-K dimension.

    Verl's FSDP eager logits-processor path assumes every extra output is a
    scalar per token and asserts ``v.shape == log_probs.shape``. LSD top-K OPD
    uses the same hook on the reference worker to return teacher top-K tensors
    with shape ``[num_tokens, topk]``. Those tensors are valid jagged values:
    the first dimension is still the flattened token dimension used by
    ``cu_seqlens`` and the extra dimension is fixed-width top-K data.
    """
    global _VERL_FSDP_TOPK_OUTPUTS_PATCHED
    if _VERL_FSDP_TOPK_OUTPUTS_PATCHED:
        return

    import functools
    import inspect

    from verl.workers.engine.fsdp import transformer_impl as mod

    engine_cls = mod.FSDPEngineWithLMHead
    if getattr(engine_cls, "_rllm_accepts_topk_processor_outputs", False):
        _VERL_FSDP_TOPK_OUTPUTS_PATCHED = True
        logger.info("verl FSDP top-K processor-output patch: already applied; nothing to patch.")
        return

    src = inspect.getsource(engine_cls.prepare_model_outputs)
    src = src.replace(
        "    def prepare_model_outputs(",
        "def prepare_model_outputs(",
        1,
    )
    old_remove_padding = '''                    for k, v in outputs.items():
                        v = v.squeeze(0)
                        assert v.shape == log_probs.shape, f"log_probs shape: {log_probs.shape}, {k} shape: {v.shape}"
                        if self.use_ulysses_sp:
                            pad_size = output_args["pad_size"]
                            v = gather_outputs_and_unpad(v, gather_dim=0, unpad_dim=0, padding_size=pad_size)
                        model_output[k] = torch.nested.nested_tensor_from_jagged(v, cu_seqlens)
'''
    new_remove_padding = '''                    for k, v in outputs.items():
                        v = v.squeeze(0)
                        if not (v.shape == log_probs.shape or (v.dim() >= 2 and v.shape[0] == log_probs.shape[0])):
                            raise AssertionError(f"log_probs shape: {log_probs.shape}, {k} shape: {v.shape}")
                        if self.use_ulysses_sp:
                            pad_size = output_args["pad_size"]
                            v = gather_outputs_and_unpad(v, gather_dim=0, unpad_dim=0, padding_size=pad_size)
                        model_output[k] = torch.nested.nested_tensor_from_jagged(v, cu_seqlens)
'''
    old_no_remove_padding = '''                        for k, v in outputs.items():
                            v = v.squeeze(0)
                            assert v.shape == log_probs.shape, (
                                f"log_probs shape: {log_probs.shape}, {k} shape: {v.shape}"
                            )
                            model_output[k] = torch.nested.nested_tensor_from_jagged(v, cu_seqlens)
'''
    new_no_remove_padding = '''                        for k, v in outputs.items():
                            v = v.squeeze(0)
                            if not (v.shape == log_probs.shape or (v.dim() >= 2 and v.shape[0] == log_probs.shape[0])):
                                raise AssertionError(f"log_probs shape: {log_probs.shape}, {k} shape: {v.shape}")
                            model_output[k] = torch.nested.nested_tensor_from_jagged(v, cu_seqlens)
'''
    patched = src.replace(old_remove_padding, new_remove_padding).replace(
        old_no_remove_padding,
        new_no_remove_padding,
    )
    if patched == src:
        original_prepare_model_outputs = engine_cls.prepare_model_outputs
        original_forward_step = engine_cls.forward_step

        @functools.wraps(original_prepare_model_outputs)
        def _prepare_model_outputs(self, output, output_args, micro_batch):
            model_output = original_prepare_model_outputs(self, output, output_args, micro_batch)
            processor = getattr(self, "_rllm_logits_processor_func", None)
            use_topk = mod.tu.get_non_tensor_data(
                data=micro_batch, key="distillation_use_topk", default=False
            )
            if not use_topk or processor is None:
                return model_output

            use_remove_padding = mod.tu.get_non_tensor_data(
                data=micro_batch, key="use_remove_padding", default=True
            )
            use_fused_kernels = mod.tu.get_non_tensor_data(
                data=micro_batch, key="use_fused_kernels", default=False
            )
            if use_fused_kernels:
                raise NotImplementedError(
                    "LSD top-K processor outputs require eager logits on this VERL version"
                )

            input_ids = micro_batch["input_ids"]
            cu_seqlens = input_ids.offsets()
            if use_remove_padding:
                logits_rmpad = output.logits.squeeze(0)
            else:
                pad_mode = mod.tu.get_non_tensor_data(
                    data=micro_batch, key="pad_mode", default=mod.DatasetPadMode.NO_PADDING
                )
                if pad_mode != mod.DatasetPadMode.NO_PADDING:
                    raise NotImplementedError(f"pad_mode {pad_mode} not implemented")
                seq_lengths = cu_seqlens.diff()
                starts = mod.torch.zeros_like(seq_lengths, dtype=mod.torch.int64)
                logits = mod.torch.nested.narrow(
                    output.logits, 1, starts, seq_lengths, layout=mod.torch.jagged
                )
                logits_rmpad = mod.torch.cat([tensor for tensor in logits.unbind()])

            outputs = processor(student_logits=logits_rmpad.unsqueeze(0), data=micro_batch)
            expected_tokens = model_output["log_probs"].values().shape[0]
            for key, value in outputs.items():
                value = value.squeeze(0)
                if self.use_ulysses_sp:
                    value = mod.gather_outputs_and_unpad(
                        value,
                        gather_dim=0,
                        unpad_dim=0,
                        padding_size=output_args["pad_size"],
                    )
                if value.dim() < 1 or value.shape[0] != expected_tokens:
                    raise AssertionError(
                        f"log_probs tokens: {expected_tokens}, {key} shape: {value.shape}"
                    )
                model_output[key] = mod.torch.nested.nested_tensor_from_jagged(
                    value, cu_seqlens
                )
            return model_output

        @functools.wraps(original_forward_step)
        def _forward_step(self, micro_batch, loss_function, forward_only):
            self._rllm_logits_processor_func = loss_function
            try:
                return original_forward_step(self, micro_batch, loss_function, forward_only)
            finally:
                self._rllm_logits_processor_func = None

        engine_cls.prepare_model_outputs = _prepare_model_outputs
        engine_cls.forward_step = _forward_step
        engine_cls._rllm_accepts_topk_processor_outputs = True
        _VERL_FSDP_TOPK_OUTPUTS_PATCHED = True
        logger.info("Patched legacy verl FSDP engine to expose top-K logits-processor outputs")
        return

    namespace = dict(mod.__dict__)
    exec(compile(patched, mod.__file__, "exec"), namespace)
    engine_cls.prepare_model_outputs = namespace["prepare_model_outputs"]
    engine_cls._rllm_accepts_topk_processor_outputs = True

    _VERL_FSDP_TOPK_OUTPUTS_PATCHED = True
    logger.info("Patched verl FSDP logits-processor outputs to accept [num_tokens, topk] tensors")



# ---------------------------------------------------------------------------
# LSD worker hooks: global hard/easy denominators + EMA teacher RPC
# ---------------------------------------------------------------------------

_VERL_LSD_WORKER_HOOKS_PATCHED = False


def patch_verl_lsd_worker_hooks() -> None:
    """Patch standard Verl workers with LSD runtime hooks.

    The denominator hook runs on each actual optimizer mini-batch, before Verl
    splits it into dynamic micro-batches. The resulting scalar NonTensorData
    fields are therefore visible to every DP rank and every dynamic microbatch.
    """
    global _VERL_LSD_WORKER_HOOKS_PATCHED
    if _VERL_LSD_WORKER_HOOKS_PATCHED:
        return

    import functools
    import inspect

    import numpy as np
    import torch
    from verl.single_controller.base.decorator import Dispatch, register
    from verl.utils import tensordict_utils as tu
    from verl.utils.metric.utils import Metric
    from verl.workers import engine_workers as ew

    def _assign_lsd_global_counts(self, data):
        if "lsd_is_easy" not in data or "response_mask" not in data:
            return
        padded = data.select("response_mask", "lsd_is_easy").to_padded_tensor()
        response_mask = padded["response_mask"].bool()
        easy_seq = padded["lsd_is_easy"].bool()
        if easy_seq.dim() > 1:
            easy_seq = easy_seq.view(easy_seq.shape[0], -1)[:, 0]
        token_count = response_mask.sum(dim=-1)
        seq_mask = token_count > 0
        easy_mask = response_mask & easy_seq.unsqueeze(-1)
        hard_mask = response_mask & (~easy_seq).unsqueeze(-1)
        local_counts = [
            float(hard_mask.sum().item()),
            float(easy_mask.sum().item()),
            float((seq_mask & (~easy_seq)).sum().item()),
            float((seq_mask & easy_seq).sum().item()),
        ]
        dp_group = self.engine.get_data_parallel_group()
        if dp_group is not None and torch.distributed.get_world_size(dp_group) > 1:
            gathered_counts = [None] * torch.distributed.get_world_size(dp_group)
            torch.distributed.all_gather_object(gathered_counts, local_counts, dp_group)
            global_counts = [sum(float(x[i]) for x in gathered_counts) for i in range(4)]
        else:
            global_counts = local_counts
        tu.assign_non_tensor(
            data,
            lsd_global_hard_token_count=global_counts[0],
            lsd_global_easy_token_count=global_counts[1],
            lsd_global_hard_seq_count=global_counts[2],
            lsd_global_easy_seq_count=global_counts[3],
        )

    if not hasattr(ew.TrainingWorker, "_assign_lsd_global_counts"):
        ew.TrainingWorker._assign_lsd_global_counts = _assign_lsd_global_counts

    train_batch_src = ""
    try:
        train_batch_src = inspect.getsource(ew.TrainingWorker.train_batch)
    except Exception:
        pass
    if "_assign_lsd_global_counts(data)" not in train_batch_src:
        original_train_batch = ew.TrainingWorker.train_batch

        @functools.wraps(original_train_batch)
        def _patched_train_batch(self, data, *args, **kwargs):
            self._assign_lsd_global_counts(data)
            output = original_train_batch(self, data, *args, **kwargs)
            if output is not None:
                metrics = tu.get(output, "metrics", default=None)
                if isinstance(metrics, dict) and "optimizer_step_applied_min" not in metrics:
                    grad_norm = metrics.get("grad_norm")
                    try:
                        applied = bool(np.isfinite(float(grad_norm)))
                    except Exception:
                        applied = bool(getattr(self.engine, "_last_optimizer_step_applied", False))
                    metrics["optimizer_step_applied_min"] = Metric(
                        value=float(getattr(self.engine, "_last_optimizer_step_applied", applied)),
                        aggregation="min",
                    )
            return output

        _patched_train_batch.__dict__.update(getattr(original_train_batch, "__dict__", {}))
        ew.TrainingWorker.train_batch = _patched_train_batch

    if not hasattr(ew.ActorRolloutRefWorker, "set_ref_loss_fn"):

        @register(dispatch_mode=Dispatch.ONE_TO_ALL)
        def set_ref_loss_fn(self, loss_fn):
            assert self.ref is not None, "set_ref_loss_fn requires a ref worker"
            self.ref.set_loss_fn(loss_fn=loss_fn)

        ew.ActorRolloutRefWorker.set_ref_loss_fn = set_ref_loss_fn

    if not hasattr(ew.ActorRolloutRefWorker, "lsd_update_ema_teacher"):

        @register(dispatch_mode=Dispatch.ONE_TO_ALL)
        def lsd_update_ema_teacher(self, beta: float, accum_dtype: str = "fp32"):
            assert "actor" in self.role and "ref" in self.role, "EMA teacher update requires actor_rollout_ref role"
            assert self.actor is not None and self.ref is not None, "EMA teacher update requires both actor and ref workers"
            dtype_map = {"fp32": torch.float32, "bf16": torch.bfloat16, "fp16": torch.float16}
            if accum_dtype not in dtype_map:
                raise ValueError(f"Unsupported LSD EMA accum_dtype={accum_dtype!r}")
            accum_torch_dtype = dtype_map[accum_dtype]
            beta = float(beta)
            if not 0.0 <= beta <= 1.0:
                raise ValueError(f"EMA beta must be in [0, 1], got {beta}")

            actor_module = getattr(self.actor.engine.module, "_fsdp_wrapped_module", self.actor.engine.module)
            ref_module = getattr(self.ref.engine.module, "_fsdp_wrapped_module", self.ref.engine.module)
            actor_params = dict(self.actor.engine.module.named_parameters())
            ref_params = dict(self.ref.engine.module.named_parameters())
            if actor_params.keys() != ref_params.keys():
                missing = sorted(set(actor_params) ^ set(ref_params))[:8]
                raise RuntimeError(f"Actor/ref parameter sets differ; cannot update LSD EMA teacher, diff={missing}")

            actor_has_lora = hasattr(actor_module, "peft_config")
            ref_has_lora = hasattr(ref_module, "peft_config")
            if actor_has_lora != ref_has_lora:
                raise RuntimeError("Actor/ref LoRA structure mismatch; cannot update LSD EMA teacher")
            trainable_names = {name for name, param in actor_params.items() if getattr(param, "requires_grad", False)}
            if actor_has_lora:
                target_names = trainable_names or {
                    name for name in actor_params if "lora_" in name or "modules_to_save" in name
                }
                if not target_names:
                    raise RuntimeError("LoRA EMA teacher update found no trainable/adapter parameters")
            else:
                target_names = {name for name, param in actor_params.items() if torch.is_floating_point(param)}

            pre_max_diff = 0.0
            post_max_diff = 0.0
            with torch.no_grad():
                for name in sorted(target_names):
                    actor_param = actor_params[name]
                    ref_param = ref_params[name]
                    if actor_param.shape != ref_param.shape:
                        raise RuntimeError(
                            f"Actor/ref parameter shape mismatch for {name}: actor={tuple(actor_param.shape)} ref={tuple(ref_param.shape)}"
                        )
                    try:
                        shared_storage = actor_param.untyped_storage().data_ptr() == ref_param.untyped_storage().data_ptr()
                    except Exception:
                        shared_storage = False
                    if shared_storage:
                        raise RuntimeError(f"Actor/ref unexpectedly share storage for {name}; refusing EMA update")
                    if not torch.is_floating_point(ref_param):
                        continue
                    actor_value = actor_param.detach().to(device=ref_param.device, dtype=accum_torch_dtype)
                    ref_value = ref_param.detach().to(dtype=accum_torch_dtype)
                    if actor_value.numel() > 0:
                        pre_max_diff = max(pre_max_diff, float((ref_value - actor_value).abs().max().item()))
                    updated = ref_value.mul(beta).add(actor_value, alpha=1.0 - beta)
                    ref_param.copy_(updated.to(dtype=ref_param.dtype))
                    post_value = ref_param.detach().to(dtype=accum_torch_dtype)
                    if actor_value.numel() > 0:
                        post_max_diff = max(post_max_diff, float((post_value - actor_value).abs().max().item()))
            return {
                "ema_pre_update_max_param_diff": pre_max_diff,
                "ema_post_update_max_param_diff": post_max_diff,
                "ema_param_count": float(len(target_names)),
            }

        ew.ActorRolloutRefWorker.lsd_update_ema_teacher = lsd_update_ema_teacher

    if not hasattr(ew.ActorRolloutRefWorker, "save_lsd_teacher_checkpoint"):

        @register(dispatch_mode=Dispatch.ONE_TO_ALL)
        def save_lsd_teacher_checkpoint(self, local_path, hdfs_path=None, global_step=0, max_ckpt_to_keep=None):
            assert "ref" in self.role, "save_lsd_teacher_checkpoint requires a ref model"
            checkpoint_manager = getattr(self.ref.engine, "checkpoint_manager", None)
            if checkpoint_manager is None:
                raise RuntimeError("save_lsd_teacher_checkpoint requires ref.engine.checkpoint_manager")
            original_save_contents = list(getattr(checkpoint_manager, "checkpoint_save_contents", []))
            checkpoint_manager.checkpoint_save_contents = ["model"]
            try:
                self.ref.save_checkpoint(local_path, hdfs_path, global_step, max_ckpt_to_keep)
            finally:
                checkpoint_manager.checkpoint_save_contents = original_save_contents

        ew.ActorRolloutRefWorker.save_lsd_teacher_checkpoint = save_lsd_teacher_checkpoint

    if not hasattr(ew.ActorRolloutRefWorker, "load_lsd_teacher_checkpoint"):

        @register(dispatch_mode=Dispatch.ONE_TO_ALL)
        def load_lsd_teacher_checkpoint(self, local_path, hdfs_path=None, del_local_after_load=False):
            assert "ref" in self.role, "load_lsd_teacher_checkpoint requires a ref model"
            checkpoint_manager = getattr(self.ref.engine, "checkpoint_manager", None)
            if checkpoint_manager is None:
                raise RuntimeError("load_lsd_teacher_checkpoint requires ref.engine.checkpoint_manager")
            original_load_contents = list(getattr(checkpoint_manager, "checkpoint_load_contents", []))
            checkpoint_manager.checkpoint_load_contents = ["model"]
            try:
                self.ref.load_checkpoint(local_path, hdfs_path, del_local_after_load=False)
            finally:
                checkpoint_manager.checkpoint_load_contents = original_load_contents

        ew.ActorRolloutRefWorker.load_lsd_teacher_checkpoint = load_lsd_teacher_checkpoint

    _VERL_LSD_WORKER_HOOKS_PATCHED = True
    logger.info("Patched Verl workers with LSD denominator and EMA teacher hooks")


def patch_verl_fsdp_reduce_scatter_cache() -> None:
    """Release inactive allocator blocks immediately before FSDP grad sharding."""

    global _VERL_FSDP_REDUCE_SCATTER_CACHE_PATCHED
    if _VERL_FSDP_REDUCE_SCATTER_CACHE_PATCHED:
        return

    import torch
    from torch.distributed.fsdp import _runtime_utils

    original = _runtime_utils._get_reduce_scatter_tensors

    def _with_empty_cache(*args, **kwargs):
        torch.cuda.empty_cache()
        return original(*args, **kwargs)

    _runtime_utils._get_reduce_scatter_tensors = _with_empty_cache
    _VERL_FSDP_REDUCE_SCATTER_CACHE_PATCHED = True


def patch_verl_fsdp_training_output_release() -> None:
    """Do not retain unused per-micro-batch model outputs during training."""

    global _VERL_FSDP_TRAIN_OUTPUT_RELEASE_PATCHED
    if _VERL_FSDP_TRAIN_OUTPUT_RELEASE_PATCHED:
        return

    import functools
    import torch
    from verl.workers.engine.fsdp.transformer_impl import FSDPEngineWithLMHead

    original = FSDPEngineWithLMHead.forward_step

    @functools.wraps(original)
    def _without_training_model_output(self, micro_batch, loss_function, forward_only):
        loss, output = original(self, micro_batch, loss_function, forward_only)
        if not forward_only and isinstance(output, dict):
            model_output = output.pop("model_output", None)
            # #region debug-point J:training-output-release
            _debug_report_patch(
                "J",
                "patch.py:FSDPEngineWithLMHead.forward_step",
                "released unused training model output",
                {
                    "keys": list(model_output.keys()) if isinstance(model_output, dict) else [],
                    "allocated_gib": torch.cuda.memory_allocated() / 2**30,
                    "reserved_gib": torch.cuda.memory_reserved() / 2**30,
                },
            )
            # #endregion
        return loss, output

    FSDPEngineWithLMHead.forward_step = _without_training_model_output
    _VERL_FSDP_TRAIN_OUTPUT_RELEASE_PATCHED = True


# ---------------------------------------------------------------------------
# Worker-side entry point (used as Ray runtime_env worker_process_setup_hook)
# ---------------------------------------------------------------------------

_ALL_VERL_PATCHES = {
    "patch_sglang_openai_token_ids": patch_sglang_openai_token_ids,
    "patch_verl_dynamic_batch_sync": patch_verl_dynamic_batch_sync,
    "patch_verl_qwen3_vl_dummy_inplace": patch_verl_qwen3_vl_dummy_inplace,
    "patch_verl_tensordict_jagged_layout": patch_verl_tensordict_jagged_layout,
    "patch_verl_fsdp_topk_processor_outputs": patch_verl_fsdp_topk_processor_outputs,
    "patch_verl_fsdp_reduce_scatter_cache": patch_verl_fsdp_reduce_scatter_cache,
    "patch_verl_fsdp_training_output_release": patch_verl_fsdp_training_output_release,
    "patch_verl_training_worker_env_debug": patch_verl_training_worker_env_debug,
    "patch_verl_lsd_worker_hooks": patch_verl_lsd_worker_hooks,
}


def apply_all_verl_patches() -> None:
    """Apply every Verl patch that is safe to run unconditionally on workers.

    Designed to be wired in as ``runtime_env.worker_process_setup_hook =
    "rllm.trainer.verl.patch.apply_all_verl_patches"`` so that each Ray
    worker process applies the patches in its own interpreter (driver-side
    monkey-patches do not propagate to worker processes).

    Each patch below is lazy and idempotent, so it is safe to call this
    repeatedly and from any process.

    Optional extension hook: if the ``RLLM_EXTRA_WORKER_SETUP_HOOK``
    environment variable is set, this function will additionally invoke the
    callable it names. The value is ``"<absolute-file-path.py>:<func>"``;
    the function is loaded directly from the file via ``importlib.util`` so
    it does not need to live in a package on ``sys.path``. This is intended
    for environment-specific workarounds (e.g. disabling cuDNN on a host
    with a broken cuDNN install) that should not be baked into rLLM itself.
    """
    # #region debug-point C:worker-setup-env
    accelerator_ids = None
    accelerator_error = None
    try:
        import ray

        accelerator_ids = ray.get_runtime_context().get_accelerator_ids()
    except Exception as exc:  # pragma: no cover - diagnostic only
        accelerator_error = repr(exc)
    _debug_report_patch(
        "C",
        "patch.py:apply_all_verl_patches",
        "worker setup environment",
        {
            "rank": os.environ.get("RANK"),
            "world_size": os.environ.get("WORLD_SIZE"),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "nvidia_visible_devices": os.environ.get("NVIDIA_VISIBLE_DEVICES"),
            "ray_no_set_cuda": os.environ.get("RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES"),
            "accelerator_ids": accelerator_ids,
            "accelerator_error": accelerator_error,
        },
    )
    # #endregion
    for patch_name, patch_func in _ALL_VERL_PATCHES.items():
        try:
            patch_func()
            # #region debug-point E:patch-success
            _debug_report_patch(
                "E",
                "patch.py:apply_all_verl_patches",
                "worker setup patch success",
                {"patch_name": patch_name},
            )
            # #endregion
        except Exception:  # pragma: no cover — patch is best-effort
            # #region debug-point E:patch-failure
            _debug_report_patch(
                "E",
                "patch.py:apply_all_verl_patches",
                "worker setup patch failure",
                {"patch_name": patch_name},
            )
            # #endregion
            logger.exception(f"{patch_name} failed in worker setup hook")

    _run_extra_worker_setup_hook()


def _run_extra_worker_setup_hook() -> None:
    """Optionally invoke a user-supplied setup hook from RLLM_EXTRA_WORKER_SETUP_HOOK.

    Format: ``"<absolute path to .py file>:<function name>"``. The function
    is loaded via ``importlib.util.spec_from_file_location`` so it works
    even when the file is not on ``sys.path`` (e.g. lives under a
    gitignored ``tmp/`` directory). Failures are logged but never raised —
    this is a best-effort extension point and must not crash worker init.
    """
    import os

    spec_str = os.environ.get("RLLM_EXTRA_WORKER_SETUP_HOOK")
    if not spec_str:
        return

    path, _, func_name = spec_str.rpartition(":")
    if not path or not func_name:
        logger.warning(
            "RLLM_EXTRA_WORKER_SETUP_HOOK=%r is malformed; expected '<file.py>:<func>'",
            spec_str,
        )
        return

    try:
        import importlib.util

        mod_name = f"_rllm_extra_setup_hook_{os.getpid()}"
        spec = importlib.util.spec_from_file_location(mod_name, path)
        if spec is None or spec.loader is None:
            logger.warning("RLLM_EXTRA_WORKER_SETUP_HOOK: could not load spec for %s", path)
            return
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        getattr(mod, func_name)()
    except Exception:
        logger.exception("RLLM_EXTRA_WORKER_SETUP_HOOK=%r failed", spec_str)
