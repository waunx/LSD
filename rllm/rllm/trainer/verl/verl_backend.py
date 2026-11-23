"""Verl backend implementation for the UnifiedTrainer."""

from __future__ import annotations

import inspect
import math
from collections import defaultdict
from collections.abc import Iterable
from functools import reduce
from typing import TYPE_CHECKING, Any

import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf, open_dict
from verl import DataProto
from verl.checkpoint_engine import CheckpointEngineManager
from verl.experimental.agent_loop.agent_loop import AgentLoopManager
from verl.single_controller.ray import RayClassWithInitArgs, RayWorkerGroup, ResourcePoolManager
from verl.single_controller.ray.base import create_colocated_worker_cls
from verl.trainer.ppo.core_algos import agg_loss
from verl.trainer.ppo.metric_utils import (
    compute_data_metrics,
    compute_throughout_metrics,
    compute_timing_metrics,
)
from verl.trainer.ppo.utils import Role, WorkerType, need_reference_policy
from verl.utils import tensordict_utils as tu
from verl.utils.config import omega_conf_to_dataclass
from verl.utils.metric import reduce_metrics
try:
    from verl.workers.rollout.llm_server import LLMServerManager
except ImportError:  # verl versions before the llm_server module
    LLMServerManager = None
from verl.workers.utils.padding import left_right_2_no_padding, no_padding_2_padding

from rllm.data.utils import interleave_tasks
from rllm.engine.rollout import RolloutEngine, VerlEngine
from rllm.trainer.algorithms import (
    AlgorithmConfig,
    collect_reward_and_advantage_from_trajectory_groups,
    simple_timer,
)
from rllm.trainer.backend_protocol import BackendProtocol
from rllm.trainer.lsd import LSDConfig, LSDRouter
from rllm.trainer.verl import transform_episodes_to_dataproto, transform_trajectory_groups_to_dataproto, update_dataproto_with_advantages
from rllm.trainer.verl.metrics import calculate_debug_metrics_compat
from rllm.trainer.verl.utils import (
    balance_batch,
    build_wg_kwargs,
    load_checkpoint,
    save_checkpoint,
    start_profiling,
    stop_profiling,
)
from rllm.types import Episode

if TYPE_CHECKING:
    from rllm.engine.unified_workflow_engine import UnifiedWorkflowEngine
    from rllm.trainer.unified_trainer import TrainerState

import logging

logger = logging.getLogger(__name__)

_DEFAULT_VERL_LOSS = "vanilla"
_VERL_KNOWN_LOSSES: set[str] | None = None


async def _checkpoint_update_weights_compat(checkpoint_manager: Any, step: int | None = None) -> None:
    """Call VERL checkpoint weight sync across old/new manager signatures."""

    update_weights = checkpoint_manager.update_weights
    try:
        signature = inspect.signature(update_weights)
        accepts_step = any(
            parameter.kind in (parameter.POSITIONAL_ONLY, parameter.POSITIONAL_OR_KEYWORD, parameter.VAR_POSITIONAL)
            for parameter in signature.parameters.values()
        )
    except (TypeError, ValueError):
        accepts_step = True

    result = update_weights(step) if accepts_step else update_weights()
    if inspect.isawaitable(result):
        await result


def _create_rllm_colocated_worker_cls(class_dict: dict[str, RayClassWithInitArgs]):
    worker_with_init = create_colocated_worker_cls(class_dict=class_dict)
    worker_dict_cls = worker_with_init.cls.__ray_actor_class__

    class RLLMPatchedWorkerDict(worker_dict_cls):
        def __init__(self):
            # Ray has assigned CUDA_VISIBLE_DEVICES by actor __init__; patch before
            # the inner ActorRolloutRefWorker objects are constructed.
            from rllm.trainer.verl.patch import apply_all_verl_patches

            apply_all_verl_patches()
            super().__init__()

    import ray

    return RayClassWithInitArgs(cls=ray.remote(RLLMPatchedWorkerDict))


class TeacherTopKExtractor:
    """Logits processor used on the teacher/ref worker to return teacher top-k support."""

    def __init__(self, topk: int, stop_token_ids: list[int] | None = None):
        self.topk = int(topk)
        self.stop_token_ids = sorted({int(x) for x in (stop_token_ids or []) if x is not None})

    @staticmethod
    def _zero_like(model_output):
        log_probs = model_output.get("log_probs")
        values = log_probs.values() if getattr(log_probs, "is_nested", False) else log_probs
        return values.sum() * 0.0

    def __call__(self, model_output=None, data=None, dp_group=None, student_logits=None, data_format="thd"):
        import torch
        import torch.nn.functional as F

        if student_logits is None:
            return self._zero_like(model_output), {}

        topk = min(self.topk, student_logits.shape[-1])
        teacher_log_probs = F.log_softmax(student_logits.float(), dim=-1)
        topk_log_probs, topk_ids = torch.topk(teacher_log_probs, k=topk, dim=-1)
        if self.stop_token_ids:
            stop_ids = torch.tensor(self.stop_token_ids, dtype=torch.long, device=student_logits.device)
            stop_ids = stop_ids[(stop_ids >= 0) & (stop_ids < student_logits.shape[-1])]
            if stop_ids.numel() > 0:
                stop_ids = torch.unique(stop_ids, sorted=True)
                stop_ids_expanded = stop_ids.view(1, 1, -1).expand(*topk_ids.shape[:2], -1)
                stop_log_probs = torch.gather(teacher_log_probs, dim=-1, index=stop_ids_expanded)
                duplicate_stop = (topk_ids.unsqueeze(-1) == stop_ids_expanded.unsqueeze(-2)).any(dim=-2)
                stop_log_probs = torch.where(
                    duplicate_stop,
                    torch.full_like(stop_log_probs, -1.0e9),
                    stop_log_probs,
                )
                topk_ids = torch.cat([topk_ids, stop_ids_expanded], dim=-1)
                topk_log_probs = torch.cat([topk_log_probs, stop_log_probs], dim=-1)
        return {
            "teacher_logprobs": topk_log_probs.to(student_logits.dtype),
            "teacher_ids": topk_ids.to(torch.long),
        }


def _no_padding_to_attention_padded(tensor: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """Restore a no-padding full-sequence tensor to the original attention-mask layout."""

    values = tensor.values() if getattr(tensor, "is_nested", False) else tensor
    mask = attention_mask.bool()
    output_shape = (*mask.shape, *values.shape[1:])
    output = torch.zeros(output_shape, dtype=values.dtype, device=values.device)
    offset = 0
    for row_idx in range(mask.shape[0]):
        row_count = int(mask[row_idx].sum().item())
        output[row_idx, mask[row_idx]] = values[offset : offset + row_count]
        offset += row_count
    if offset != values.shape[0]:
        raise ValueError(f"no-padding tensor length mismatch: consumed={offset}, values={values.shape[0]}")
    return output


def _attention_padded_to_no_padding(
    tensor: torch.Tensor,
    attention_mask: torch.Tensor,
    offsets: torch.Tensor,
) -> torch.Tensor:
    """Convert a padded per-token tensor to a jagged tensor using shared offsets."""

    mask = attention_mask.bool()
    values = tensor[mask]
    if int(offsets[-1].item()) != values.shape[0]:
        raise ValueError(
            "attention-padded tensor length mismatch: "
            f"offsets={int(offsets[-1].item())}, values={values.shape[0]}"
        )
    return torch.nested.nested_tensor_from_jagged(
        values=values,
        offsets=offsets,
        jagged_dim=1,
    )


class CustomPPOLoss:
    """Wraps Verl's ``ppo_loss`` to support per-call loss mode override.

    When the data TensorDict contains ``policy_loss_mode_override``,
    the loss mode is temporarily overridden for that call.  Instances
    are serialised via cloudpickle and sent to remote workers through
    Verl's ``set_loss_fn`` RPC.
    """

    def __init__(self, config, lsd_config=None):
        # Convert OmegaConf DictConfig → ActorConfig dataclass
        from verl.utils.config import omega_conf_to_dataclass

        self.config = omega_conf_to_dataclass(config)
        self.lsd_config = lsd_config

    def __call__(self, model_output=None, data=None, dp_group=None, student_logits=None, data_format="thd"):
        from verl.utils import tensordict_utils as _tu
        from verl.workers.utils.losses import ppo_loss

        if self.lsd_config is not None and getattr(self.lsd_config, "enable", False):
            try:
                if student_logits is not None:
                    # #region debug-point I:lsd-logits-shape
                    try:
                        from rllm.trainer.verl.patch import _debug_report_patch

                        import torch

                        _debug_report_patch(
                            "I",
                            "verl_backend.py:CustomPPOLoss.__call__",
                            "direct FKL logits before loss",
                            {
                                "shape": list(student_logits.shape),
                                "dtype": str(student_logits.dtype),
                                "device": str(student_logits.device),
                                "allocated_gib": torch.cuda.memory_allocated(student_logits.device) / 2**30,
                                "reserved_gib": torch.cuda.memory_reserved(student_logits.device) / 2**30,
                                "free_total": list(torch.cuda.mem_get_info(student_logits.device)),
                                "teacher_values_shape": list(data["teacher_ids"].values().shape)
                                if data is not None and "teacher_ids" in data
                                else None,
                            },
                        )
                    except Exception:
                        pass
                    # #endregion
                    return self._lsd_topk_logits_processor(
                        student_logits=student_logits,
                        data=data,
                        data_format=data_format,
                    )
                if "lsd_is_easy" in data and "ref_log_prob" in data and "old_log_probs" in data:
                    return self._lsd_ppo_loss(model_output, data, dp_group)
            except Exception:
                logger.exception("LSD policy loss failed; re-raising to avoid silently changing training objective")
                raise

        override = _tu.get(data, "policy_loss_mode_override", default=None)
        if override is not None:
            original = self.config.policy_loss.get("loss_mode", "vanilla")
            self.config.policy_loss["loss_mode"] = override
            try:
                return ppo_loss(self.config, model_output, data, dp_group)
            finally:
                self.config.policy_loss["loss_mode"] = original
        return ppo_loss(self.config, model_output, data, dp_group)

    @staticmethod
    def _normalize_lsd_agg_mode(mode: str | None) -> str:
        mode = (mode or "seq_mean_token_mean").replace("-", "_")
        allowed = {"token_mean", "seq_mean_token_mean", "seq_mean_token_sum", "base_length_normalized"}
        if mode not in allowed:
            raise ValueError(f"Invalid rllm.lsd.loss_agg_mode={mode!r}; expected one of {sorted(allowed)}")
        return mode

    def _lsd_uses_topk_opd(self) -> bool:
        return self._lsd_opd_loss_mode() == "topk_forward_kl"

    def _lsd_opd_loss_mode(self) -> str:
        if hasattr(self.lsd_config, "normalized_opd_loss_mode"):
            return self.lsd_config.normalized_opd_loss_mode()
        mode = str(getattr(self.lsd_config, "opd_loss_mode", "sample_token")).replace("-", "_")
        return {"topk": "topk_forward_kl", "forward_kl_topk": "topk_forward_kl"}.get(mode, mode)

    def _lsd_opd_optimization_mode(self) -> str:
        if hasattr(self.lsd_config, "normalized_opd_optimization_mode"):
            return self.lsd_config.normalized_opd_optimization_mode()
        mode = str(getattr(self.lsd_config, "opd_optimization_mode", "supervised_gradient")).replace("-", "_")
        return {"supervised": "supervised_gradient", "direct": "supervised_gradient", "pg": "policy_gradient"}.get(mode, mode)

    def _validate_lsd_combo(self) -> None:
        combo = (self._lsd_opd_loss_mode(), self._lsd_opd_optimization_mode())
        if combo not in {
            ("topk_forward_kl", "supervised_gradient"),
            ("sample_token", "policy_gradient"),
        }:
            raise ValueError(
                "Invalid LSD OPD config. Only topk_forward_kl+supervised_gradient "
                "and sample_token+policy_gradient are supported."
            )

    @staticmethod
    def _zero_like(log_prob):
        return log_prob.sum() * 0.0

    def _lsd_topk_logits_processor(self, student_logits, data, data_format="thd"):
        import torch
        import torch.nn.functional as F

        if not self._lsd_uses_topk_opd():
            return {}
        if data is None or "teacher_logprobs" not in data or "teacher_ids" not in data:
            raise KeyError("LSD top-k OPD requires teacher_logprobs and teacher_ids in the actor batch")

        teacher_topk_log_probs = data["teacher_logprobs"]
        teacher_topk_ids = data["teacher_ids"]
        if not teacher_topk_log_probs.is_nested or not teacher_topk_ids.is_nested:
            raise ValueError("teacher_logprobs and teacher_ids must be nested tensors after left_right_2_no_padding")

        teacher_topk_log_probs = teacher_topk_log_probs.values().unsqueeze(0).float()
        teacher_topk_ids = teacher_topk_ids.values().unsqueeze(0).long()

        try:
            from verl.utils.ulysses import get_ulysses_sequence_parallel_world_size, slice_input_tensor

            if get_ulysses_sequence_parallel_world_size() > 1:
                teacher_topk_log_probs = slice_input_tensor(teacher_topk_log_probs, dim=1)
                teacher_topk_ids = slice_input_tensor(teacher_topk_ids, dim=1)
        except Exception:
            pass

        assert teacher_topk_log_probs.shape[:2] == teacher_topk_ids.shape[:2] == student_logits.shape[:2], (
            f"teacher_topk_log_probs={teacher_topk_log_probs.shape}, "
            f"teacher_topk_ids={teacher_topk_ids.shape}, student_logits={student_logits.shape}"
        )

        student_log_probs = F.log_softmax(student_logits.float(), dim=-1)
        topk = teacher_topk_ids.shape[-1]
        student_topk_log_probs, student_topk_ids = torch.topk(student_log_probs, k=topk, dim=-1)
        student_on_teacher_topk_log_probs = torch.gather(student_log_probs, dim=-1, index=teacher_topk_ids)

        raw_teacher_topk_log_probs = teacher_topk_log_probs
        raw_teacher_topk_probs = raw_teacher_topk_log_probs.exp()
        teacher_mass = raw_teacher_topk_probs.sum(dim=-1)
        teacher_log_z = torch.logsumexp(raw_teacher_topk_log_probs, dim=-1, keepdim=True)
        teacher_topk_log_probs = raw_teacher_topk_log_probs - teacher_log_z
        teacher_topk_probs = teacher_topk_log_probs.exp()

        token_kl = teacher_topk_probs * (teacher_topk_log_probs - student_on_teacher_topk_log_probs)
        distillation_losses = token_kl.sum(dim=-1)
        student_mass = student_on_teacher_topk_log_probs.exp().sum(dim=-1)
        student_topk_mass = student_topk_log_probs.exp().sum(dim=-1)

        overlap_mask = (teacher_topk_ids.unsqueeze(-1) == student_topk_ids.unsqueeze(-2)).any(dim=-1)
        overlap_count = overlap_mask.sum(dim=-1)
        overlap_token_advantage_sum = (-token_kl * overlap_mask).sum(dim=-1)
        overlap_token_advantage = overlap_token_advantage_sum / overlap_count.clamp_min(1)
        overlap_token_advantage = torch.where(
            overlap_count > 0, overlap_token_advantage, torch.zeros_like(overlap_token_advantage)
        )
        # Release inactive allocator blocks before FSDP backward reduce-scatter.
        torch.cuda.empty_cache()

        return {
            "lsd_topk_opd_losses": distillation_losses,
            "lsd_topk_student_mass": student_mass,
            "lsd_topk_teacher_mass": teacher_mass,
            "lsd_topk_student_topk_mass": student_topk_mass,
            "lsd_topk_overlap_count": overlap_count.float(),
            "lsd_topk_overlap_token_advantage": overlap_token_advantage,
        }

    def _aggregate_lsd_loss(
        self,
        loss_mat,
        loss_mask,
        mode,
        log_prob,
        base_lengths=None,
        global_denominators: dict[str, float] | None = None,
        denominator_key: str | None = None,
        dp_size: int = 1,
    ):
        import torch

        mask = loss_mask.float()
        mode = self._normalize_lsd_agg_mode(mode)
        dp_size = int(dp_size or 1)

        def _global_denominator(local_denominator):
            if global_denominators is not None and denominator_key is not None:
                value = float(global_denominators.get(denominator_key, 0.0) or 0.0)
                return torch.tensor(value, dtype=loss_mat.dtype, device=loss_mat.device)
            return local_denominator.to(dtype=loss_mat.dtype, device=loss_mat.device)

        if mode == "token_mean":
            local_num = (loss_mat * mask).sum()
            global_den = _global_denominator(mask.sum())
            if global_den <= 0:
                return self._zero_like(log_prob)
            return local_num / global_den.clamp_min(1.0) * dp_size

        token_count = mask.sum(dim=-1)
        seq_mask = (token_count > 0).float()
        if mode == "seq_mean_token_sum":
            seq_losses = (loss_mat * mask).sum(dim=-1)
        elif mode == "base_length_normalized":
            if base_lengths is None:
                raise ValueError("rllm.lsd.loss_agg_mode=base_length_normalized requires lsd_base_length")
            invalid = (base_lengths.float() <= 0) & (seq_mask > 0)
            if invalid.any():
                raise ValueError("rllm.lsd.loss_agg_mode=base_length_normalized requires positive lsd_base_length")
            denom = base_lengths.float()
            seq_losses = (loss_mat * mask).sum(dim=-1) / denom.clamp_min(1.0)
        else:
            seq_losses = (loss_mat * mask).sum(dim=-1) / token_count.clamp_min(1.0)

        local_num = (seq_losses * seq_mask).sum()
        global_den = _global_denominator(seq_mask.sum())
        if global_den <= 0:
            return self._zero_like(log_prob)
        return local_num / global_den.clamp_min(1.0) * dp_size

    @staticmethod
    def _lsd_global_denominators_from_data(data) -> dict[str, float] | None:
        keys = [
            "lsd_global_hard_token_count",
            "lsd_global_easy_token_count",
            "lsd_global_hard_seq_count",
            "lsd_global_easy_seq_count",
        ]
        out = {}
        found = False
        for key in keys:
            value = tu.get(data, key, default=None) if hasattr(data, "keys") else data.get(key)
            if value is None:
                continue
            found = True
            if hasattr(value, "item"):
                value = value.item()
            out[key] = float(value or 0.0)
        return out if found else None

    @staticmethod
    def _lsd_token_mass_scale(easy_tokens, hard_tokens, log_prob, global_denominators: dict[str, float] | None = None):
        if global_denominators is not None:
            easy_token_count = float(global_denominators.get("lsd_global_easy_token_count", 0.0) or 0.0)
            hard_token_count = float(global_denominators.get("lsd_global_hard_token_count", 0.0) or 0.0)
        else:
            easy_token_count = float(easy_tokens.detach().item())
            hard_token_count = float(hard_tokens.detach().item())

        if easy_token_count <= 0.0:
            scale = 0.0
        elif hard_token_count <= 0.0:
            # No hard reference mass exists in this optimizer batch; avoid
            # exploding the OPD coefficient while still allowing easy-only LSD.
            scale = 1.0
        else:
            scale = easy_token_count / hard_token_count
        return torch.tensor(scale, dtype=log_prob.dtype, device=log_prob.device)

    def _lsd_denominator_key(self, prefix: str, mode: str) -> str:
        mode = self._normalize_lsd_agg_mode(mode)
        suffix = "token_count" if mode == "token_mean" else "seq_count"
        return f"lsd_global_{prefix}_{suffix}"

    def _clipped_pg_loss(
        self,
        old_log_prob,
        log_prob,
        advantages,
        response_mask,
        agg_mode,
        base_lengths=None,
        global_denominators: dict[str, float] | None = None,
        denominator_key: str | None = None,
        dp_size: int = 1,
    ):
        import torch

        clip_ratio = self.config.clip_ratio
        clip_ratio_low = self.config.clip_ratio_low if self.config.clip_ratio_low is not None else clip_ratio
        clip_ratio_high = self.config.clip_ratio_high if self.config.clip_ratio_high is not None else clip_ratio
        clip_ratio_c = self.config.get("clip_ratio_c", 3.0)

        negative_approx_kl = torch.clamp(log_prob - old_log_prob, min=-20.0, max=20.0)
        ratio = torch.exp(negative_approx_kl)

        pg_losses1 = -advantages * ratio
        pg_losses2 = -advantages * torch.clamp(ratio, 1 - clip_ratio_low, 1 + clip_ratio_high)
        clip_pg_losses1 = torch.maximum(pg_losses1, pg_losses2)
        pg_losses3 = -advantages * clip_ratio_c
        clip_pg_losses2 = torch.min(pg_losses3, clip_pg_losses1)
        pg_losses = torch.where(advantages < 0, clip_pg_losses2, clip_pg_losses1)

        loss = self._aggregate_lsd_loss(
            pg_losses,
            response_mask,
            agg_mode,
            log_prob,
            base_lengths=base_lengths,
            global_denominators=global_denominators,
            denominator_key=denominator_key,
            dp_size=dp_size,
        )
        mask = response_mask.bool()
        if mask.sum() > 0:
            pg_clipfrac = torch.masked_select((pg_losses2 > pg_losses1).float(), mask).mean()
            pg_clipfrac_lower = torch.masked_select(((clip_pg_losses1 > pg_losses3) & (advantages < 0)).float(), mask).mean()
            ppo_kl = torch.masked_select(-negative_approx_kl, mask).mean()
        else:
            pg_clipfrac = self._zero_like(log_prob)
            pg_clipfrac_lower = self._zero_like(log_prob)
            ppo_kl = self._zero_like(log_prob)
        return loss, {
            "pg_clipfrac": pg_clipfrac.detach(),
            "pg_clipfrac_lower": pg_clipfrac_lower.detach(),
            "ppo_kl": ppo_kl.detach(),
        }

    def _lsd_ppo_loss(self, model_output, data, dp_group=None):
        import torch
        from verl.trainer.ppo.core_algos import agg_loss, kl_penalty
        from verl.utils.metric import AggregationType, Metric
        from verl.workers.utils.padding import no_padding_2_padding

        self._validate_lsd_combo()

        log_prob = no_padding_2_padding(model_output["log_probs"], data)
        entropy = model_output.get("entropy", None)
        if entropy is not None:
            entropy = no_padding_2_padding(entropy, data)

        self.config.global_batch_info["dp_size"] = data["dp_size"]
        self.config.global_batch_info["batch_num_tokens"] = data["batch_num_tokens"]
        self.config.global_batch_info["global_batch_size"] = data["global_batch_size"]
        self.config.global_batch_info["loss_scale_factor"] = self.config.loss_scale_factor

        metric_aggregation = (
            AggregationType.SUM
            if (
                data["dp_size"] > 1
                or data["batch_num_tokens"] is not None
                or data["global_batch_size"] is not None
                or self.config.loss_scale_factor is not None
            )
            else AggregationType.MEAN
        )

        fields = [
            "response_mask",
            "responses",
            "old_log_probs",
            "advantages",
            "ref_log_prob",
            "lsd_is_easy",
            "lsd_base_acc",
            "lsd_base_length",
        ]
        if "rollout_is_weights" in data:
            fields.append("rollout_is_weights")
        padded = data.select(*fields).to_padded_tensor()

        response_mask = padded["response_mask"].to(bool)
        old_log_prob = padded["old_log_probs"]
        ref_log_prob = padded["ref_log_prob"]
        rl_advantages = padded["advantages"].detach()
        easy_seq = padded["lsd_is_easy"].to(bool)
        if easy_seq.dim() > 1:
            easy_seq = easy_seq.view(easy_seq.shape[0], -1)[:, 0]
        easy_mask = response_mask & easy_seq.unsqueeze(-1)
        hard_mask = response_mask & (~easy_seq).unsqueeze(-1)
        base_lengths = padded.get("lsd_base_length", None)
        dp_size = int(tu.get(data, "dp_size", default=1) or 1)
        global_denominators = self._lsd_global_denominators_from_data(data)
        if global_denominators is None and dp_size > 1:
            raise RuntimeError("LSD loss requires global hard/easy denominators when dp_size > 1")
        actor_agg_mode = self.config.loss_agg_mode
        lsd_agg_mode = getattr(self.lsd_config, "loss_agg_mode", "seq_mean_token_mean")

        hard_loss, hard_metrics = self._clipped_pg_loss(
            old_log_prob=old_log_prob,
            log_prob=log_prob,
            advantages=rl_advantages,
            response_mask=hard_mask,
            agg_mode=actor_agg_mode,
            global_denominators=global_denominators,
            denominator_key=self._lsd_denominator_key("hard", actor_agg_mode),
            dp_size=dp_size,
        )

        easy_rl_coef = float(getattr(self.lsd_config, "easy_rl_coef", 0.0))
        if easy_rl_coef != 0.0:
            easy_rl_loss, easy_rl_metrics = self._clipped_pg_loss(
                old_log_prob=old_log_prob,
                log_prob=log_prob,
                advantages=rl_advantages,
                response_mask=easy_mask,
                agg_mode=actor_agg_mode,
                global_denominators=global_denominators,
                denominator_key=self._lsd_denominator_key("easy", actor_agg_mode),
                dp_size=dp_size,
            )
        else:
            easy_rl_loss = self._zero_like(log_prob)
            easy_rl_metrics = {"pg_clipfrac": easy_rl_loss.detach(), "pg_clipfrac_lower": easy_rl_loss.detach(), "ppo_kl": easy_rl_loss.detach()}

        topk_metrics: dict[str, float] = {}
        if self._lsd_uses_topk_opd():
            if "lsd_topk_opd_losses" not in model_output:
                raise KeyError("LSD top-k OPD enabled but actor model_output has no lsd_topk_opd_losses")
            opd_optimization_mode = self._lsd_opd_optimization_mode()
            if opd_optimization_mode != "supervised_gradient":
                raise ValueError("topk_forward_kl only supports supervised_gradient")
            opd_token_losses = no_padding_2_padding(model_output["lsd_topk_opd_losses"], data)
            direct_clip = getattr(self.lsd_config, "opd_direct_loss_clip", None)
            if direct_clip is not None:
                opd_token_losses = opd_token_losses.clamp(max=float(direct_clip))
            opd_loss = self._aggregate_lsd_loss(
                opd_token_losses,
                easy_mask,
                lsd_agg_mode,
                log_prob,
                base_lengths=base_lengths,
                global_denominators=global_denominators,
                denominator_key=self._lsd_denominator_key("easy", lsd_agg_mode),
                dp_size=dp_size,
            )
            opd_advantages = -opd_token_losses.detach()
            opd_metrics = {
                "pg_clipfrac": self._zero_like(log_prob).detach(),
                "pg_clipfrac_lower": self._zero_like(log_prob).detach(),
                "ppo_kl": self._zero_like(log_prob).detach(),
            }

            for src_key, dst_key in [
                ("lsd_topk_student_mass", "lsd/opd_topk_student_mass"),
                ("lsd_topk_student_topk_mass", "lsd/opd_topk_student_topk_mass"),
                ("lsd_topk_teacher_mass", "lsd/opd_topk_teacher_mass"),
                ("lsd_topk_overlap_count", "lsd/opd_topk_overlap_count"),
                ("lsd_topk_overlap_token_advantage", "lsd/opd_topk_overlap_token_advantage"),
            ]:
                if src_key in model_output:
                    values = no_padding_2_padding(model_output[src_key], data)
                    selected = torch.masked_select(values, easy_mask)
                    topk_metrics[dst_key] = selected.mean().detach().item() if selected.numel() > 0 else 0.0
            topk = max(int(getattr(self.lsd_config, "opd_topk", 1)), 1)
            if "lsd/opd_topk_overlap_count" in topk_metrics:
                topk_metrics["lsd/opd_topk_overlap_ratio"] = topk_metrics["lsd/opd_topk_overlap_count"] / float(topk)
            topk_metrics["lsd/opd_is_policy_gradient"] = 1.0 if opd_optimization_mode == "policy_gradient" else 0.0
        else:
            opd_advantages = (ref_log_prob - old_log_prob).detach()
            pg_adv_clip = getattr(self.lsd_config, "opd_pg_adv_clip", 10.0)
            if pg_adv_clip is not None:
                opd_advantages = opd_advantages.clamp(min=-float(pg_adv_clip), max=float(pg_adv_clip))
            opd_loss, opd_metrics = self._clipped_pg_loss(
                old_log_prob=old_log_prob,
                log_prob=log_prob,
                advantages=opd_advantages,
                response_mask=easy_mask,
                agg_mode=lsd_agg_mode,
                base_lengths=base_lengths,
                global_denominators=global_denominators,
                denominator_key=self._lsd_denominator_key("easy", lsd_agg_mode),
                dp_size=dp_size,
            )

        valid_tokens = response_mask.float().sum().clamp_min(1.0)
        easy_tokens = easy_mask.float().sum()
        hard_tokens = hard_mask.float().sum()
        seq_token_count = response_mask.sum(dim=-1)
        seq_mask = seq_token_count > 0
        easy_seqs = (seq_mask & easy_seq).float().sum()
        hard_seqs = (seq_mask & (~easy_seq)).float().sum()

        # #region debug-point B:lsd-loss-mass
        try:
            import json
            import urllib.request

            urllib.request.urlopen(
                urllib.request.Request(
                    "http://127.0.0.1:7777/event",
                    data=json.dumps(
                        {
                            "sessionId": "lsd-routing-relaunch",
                            "runId": "post-fix",
                            "hypothesisId": "B",
                            "location": "rllm/trainer/verl/verl_backend.py:_lsd_ppo_loss",
                            "msg": "[DEBUG] LSD loss branch mass",
                            "data": {
                                "actor_agg_mode": actor_agg_mode,
                                "lsd_agg_mode": lsd_agg_mode,
                                "easy_tokens": float(easy_tokens.detach().item()),
                                "hard_tokens": float(hard_tokens.detach().item()),
                                "easy_seqs": float(easy_seqs.detach().item()),
                                "hard_seqs": float(hard_seqs.detach().item()),
                            },
                        }
                    ).encode(),
                    headers={"Content-Type": "application/json"},
                ),
                timeout=2.0,
            ).read()
        except Exception:
            pass
        # #endregion
        lsd_coef = float(getattr(self.lsd_config, "lsd_coef", 1.0))
        lsd_reweigh_scale = self._lsd_token_mass_scale(
            easy_tokens=easy_tokens,
            hard_tokens=hard_tokens,
            log_prob=log_prob,
            global_denominators=global_denominators,
        )
        effective_lsd_coef = lsd_coef * lsd_reweigh_scale
        effective_opd_loss = effective_lsd_coef * opd_loss
        pg_loss = hard_loss + easy_rl_coef * easy_rl_loss + effective_opd_loss
        policy_loss = pg_loss

        if entropy is not None:
            # Preserve entropy regularization on all valid response tokens.
            entropy_loss = agg_loss(
                loss_mat=entropy,
                loss_mask=response_mask,
                loss_agg_mode=self.config.loss_agg_mode,
                **self.config.global_batch_info,
            )
            policy_loss -= self.config.entropy_coeff * entropy_loss
        else:
            entropy_loss = self._zero_like(log_prob)

        if self.config.use_kl_loss:
            # Keep global KL behavior for hard examples only.  Easy examples
            # use the LSD OPD objective instead of receiving both KL loss and OPD.
            kld = kl_penalty(logprob=log_prob, ref_logprob=ref_log_prob, kl_penalty=self.config.kl_loss_type)
            kl_loss = self._aggregate_lsd_loss(
                kld,
                hard_mask,
                actor_agg_mode,
                log_prob,
                global_denominators=global_denominators,
                denominator_key=self._lsd_denominator_key("hard", actor_agg_mode),
                dp_size=dp_size,
            )
            policy_loss += kl_loss * self.config.kl_loss_coef
        else:
            kl_loss = self._zero_like(log_prob)

        easy_rows = easy_seq.float().sum()
        row_count = torch.tensor(float(easy_seq.numel()), device=log_prob.device)
        valid_opd = torch.masked_select(opd_advantages, easy_mask)
        ref_old_delta = torch.masked_select(ref_log_prob - old_log_prob, response_mask)

        metrics = {
            "actor/pg_loss": Metric(value=pg_loss, aggregation=metric_aggregation),
            "lsd/hard_pg_loss": Metric(value=hard_loss, aggregation=metric_aggregation),
            "lsd/easy_rl_pg_loss": Metric(value=easy_rl_loss, aggregation=metric_aggregation),
            "lsd/opd_pg_loss": Metric(value=opd_loss, aggregation=metric_aggregation),
            "lsd/easy_row_fraction": Metric(value=easy_rows / row_count.clamp_min(1.0), aggregation=AggregationType.MEAN),
            "lsd/easy_token_fraction": Metric(value=easy_tokens / valid_tokens, aggregation=AggregationType.MEAN),
            "lsd/hard_token_fraction": Metric(value=hard_tokens / valid_tokens, aggregation=AggregationType.MEAN),
            "lsd/coef": lsd_coef,
            "lsd/reweigh_scale": lsd_reweigh_scale.detach().item(),
            "lsd/token_mass_scale": lsd_reweigh_scale.detach().item(),
            "lsd/effective_coef": effective_lsd_coef.detach().item(),
            "lsd/effective_opd_pg_loss": Metric(value=effective_opd_loss, aggregation=metric_aggregation),
            "lsd/easy_rl_coef": easy_rl_coef,
            "lsd/opd_adv_mean": (valid_opd.mean().detach().item() if valid_opd.numel() > 0 else 0.0),
            "lsd/opd_adv_std": (valid_opd.std(unbiased=False).detach().item() if valid_opd.numel() > 0 else 0.0),
            "lsd/ref_old_logratio_mean": (ref_old_delta.mean().detach().item() if ref_old_delta.numel() > 0 else 0.0),
            "lsd/ref_old_logratio_abs_mean": (ref_old_delta.abs().mean().detach().item() if ref_old_delta.numel() > 0 else 0.0),
            "lsd/hard/pg_clipfrac": hard_metrics["pg_clipfrac"].detach().item(),
            "lsd/hard/pg_clipfrac_lower": hard_metrics[
                "pg_clipfrac_lower"
            ].detach().item(),
            "lsd/hard/ppo_kl": hard_metrics["ppo_kl"].detach().item(),
            "lsd/opd/pg_clipfrac": opd_metrics["pg_clipfrac"].detach().item(),
            "lsd/opd/ppo_kl": opd_metrics["ppo_kl"].detach().item(),
        }
        metrics.update(topk_metrics)
        if entropy is not None:
            metrics["actor/entropy_loss"] = Metric(value=entropy_loss, aggregation=metric_aggregation)
        if self.config.use_kl_loss:
            metrics["kl_loss"] = Metric(value=kl_loss, aggregation=metric_aggregation)
            metrics["kl_coef"] = self.config.kl_loss_coef

        return policy_loss, metrics


def _get_verl_known_losses() -> set[str]:
    """Lazily load the set of registered Verl policy loss function names."""
    global _VERL_KNOWN_LOSSES
    if _VERL_KNOWN_LOSSES is None:
        from verl.trainer.ppo.core_algos import POLICY_LOSS_REGISTRY

        _VERL_KNOWN_LOSSES = set(POLICY_LOSS_REGISTRY.keys())
    return _VERL_KNOWN_LOSSES


class VerlBackend(BackendProtocol[Iterable, DataProto]):
    """Verl backend for the unified trainer."""

    name: str = "verl"

    def __init__(
        self,
        config: DictConfig,
        tokenizer,
        role_worker_mapping: dict[Role, WorkerType],
        resource_pool_manager: ResourcePoolManager,
        ray_worker_group_cls: type[RayWorkerGroup] = RayWorkerGroup,
        processor=None,
        **kwargs,
    ):
        BackendProtocol.__init__(self, config, **kwargs)

        self.tokenizer = tokenizer
        self.processor = processor
        self.full_config = config
        self.is_separated = config.rllm.get("async_training", {}).get("enable", False)
        self.hybrid_engine = config.actor_rollout_ref.hybrid_engine
        self.lsd_config = LSDConfig.from_config(config.rllm.get("lsd", {}))
        self.lsd_router = LSDRouter(self.lsd_config) if self.lsd_config.enable else None
        self.use_reference_policy = need_reference_policy(config) or self.lsd_config.enable
        self.use_prefix_grouper = config.actor_rollout_ref.actor.get("use_prefix_grouper", False)
        self.device_name = config.trainer.get("device", "cuda")

        lora_rank = config.actor_rollout_ref.model.get("lora", {}).get("rank", 0)
        if lora_rank <= 0:
            lora_rank = config.actor_rollout_ref.model.get("lora_rank", 0)
        self.ref_in_actor = (lora_rank > 0 or config.actor_rollout_ref.model.get("lora_adapter_path") is not None) and not self.lsd_config.enable

        self.role_worker_mapping = role_worker_mapping
        self.resource_pool_manager = resource_pool_manager
        self.ray_worker_group_cls = ray_worker_group_cls
        self.actor_rollout_wg = None
        self.ref_policy_wg = None

        self.async_rollout_manager = None
        self.checkpoint_manager: CheckpointEngineManager | None = None
        self.rollout_engine: VerlEngine | None = None
        self.algorithm_config: AlgorithmConfig | None = None
        self._last_actor_update_success = False
        self._pending_ema_init_metrics: dict[str, float] = {}

    def _init_colocated_workers(self) -> None:
        """Create worker groups for colocated (hybrid engine) mode."""
        config = self.config
        self.resource_pool_manager.create_resource_pool()
        resource_pool_to_cls = {pool: {} for pool in self.resource_pool_manager.resource_pool_dict.values()}

        actor_role = Role.ActorRolloutRef if Role.ActorRolloutRef in self.role_worker_mapping else Role.ActorRollout
        actor_rollout_resource_pool = self.resource_pool_manager.get_resource_pool(actor_role)
        resource_pool_to_cls[actor_rollout_resource_pool][str(actor_role)] = RayClassWithInitArgs(
            cls=self.role_worker_mapping[actor_role],
            config=config.actor_rollout_ref,
            role=str(actor_role),
        )

        if self.use_reference_policy and Role.RefPolicy in self.role_worker_mapping:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.RefPolicy)
            resource_pool_to_cls[resource_pool][str(Role.RefPolicy)] = RayClassWithInitArgs(
                self.role_worker_mapping[Role.RefPolicy],
                config=config.actor_rollout_ref,
                role=str(Role.RefPolicy),
            )

        wg_kwargs = build_wg_kwargs(config, self.device_name)
        all_wg = {}
        for resource_pool, class_dict in resource_pool_to_cls.items():
            if not class_dict:
                continue
            worker_dict_cls = _create_rllm_colocated_worker_cls(class_dict=class_dict)
            wg_dict = self.ray_worker_group_cls(resource_pool=resource_pool, ray_cls_with_init=worker_dict_cls, **wg_kwargs)
            all_wg.update(wg_dict.spawn(prefix_set=class_dict.keys()))

        if self.use_reference_policy and not self.ref_in_actor:
            if str(Role.RefPolicy) in all_wg:
                self.ref_policy_wg = all_wg[str(Role.RefPolicy)]
                self.ref_policy_wg.init_model()
            else:
                assert str(Role.ActorRolloutRef) in all_wg, f"{all_wg.keys()=}"
                self.ref_policy_wg = all_wg[str(Role.ActorRolloutRef)]

        self.actor_rollout_wg = all_wg[str(actor_role)]
        self.actor_rollout_wg.init_model()

        if self.ref_in_actor:
            self.ref_policy_wg = self.actor_rollout_wg

        if LLMServerManager is not None and hasattr(LLMServerManager, "create") and hasattr(AgentLoopManager, "create"):
            self.llm_server_manager = LLMServerManager.create(
                config=config,
                worker_group=self.actor_rollout_wg,
                rollout_resource_pool=actor_rollout_resource_pool,
            )
            self.async_rollout_manager = AgentLoopManager.create(
                config=config,
                llm_client=self.llm_server_manager.get_client(),
            )
        else:
            self.async_rollout_manager = AgentLoopManager(
                config=config,
                worker_group=self.actor_rollout_wg,
                rollout_resource_pool=actor_rollout_resource_pool,
            )
            self.llm_server_manager = self.async_rollout_manager

        self.checkpoint_manager = CheckpointEngineManager(
            backend=config.actor_rollout_ref.rollout.checkpoint_engine.backend,
            trainer=self.actor_rollout_wg,
            replicas=self._get_rollout_replicas(),
        )
        self.checkpoint_manager.sleep_replicas()

    def _init_separated_workers(self) -> None:
        """Create training-side workers and standalone rollout servers for separated (async) mode.

        Training workers are created via RayWorkerGroup on trainer GPUs.
        Rollout servers are launched by FullyAsyncAgentLoopManager in standalone
        mode (worker_group=None), using config.actor_rollout_ref.rollout.nnodes/n_gpus_per_node.
        """
        from rllm.trainer.verl.async_agent_loop import FullyAsyncAgentLoopManager, FullyAsyncLLMServerClient, FullyAsyncLLMServerManager

        config = self.config
        wg_kwargs = build_wg_kwargs(config, self.device_name)

        self.resource_pool_manager.create_resource_pool()
        resource_pool_to_cls = {pool: {} for pool in self.resource_pool_manager.resource_pool_dict.values()}

        actor_role = Role.Actor
        if Role.ActorRollout in self.role_worker_mapping:
            actor_role = Role.ActorRollout
        elif Role.Actor not in self.role_worker_mapping:
            raise ValueError(f"Separated mode requires Role.Actor or Role.ActorRollout, got {self.role_worker_mapping.keys()}")

        resource_pool = self.resource_pool_manager.get_resource_pool(actor_role)
        resource_pool_to_cls[resource_pool][str(actor_role)] = RayClassWithInitArgs(
            cls=self.role_worker_mapping[actor_role],
            config=config.actor_rollout_ref,
            role=str(actor_role),
        )

        if self.use_reference_policy and Role.RefPolicy in self.role_worker_mapping:
            ref_pool = self.resource_pool_manager.get_resource_pool(Role.RefPolicy)
            resource_pool_to_cls[ref_pool][str(Role.RefPolicy)] = RayClassWithInitArgs(
                self.role_worker_mapping[Role.RefPolicy],
                config=config.actor_rollout_ref,
                role=str(Role.RefPolicy),
            )

        all_wg = {}
        for rp, class_dict in resource_pool_to_cls.items():
            if not class_dict:
                continue
            worker_dict_cls = _create_rllm_colocated_worker_cls(class_dict=class_dict)
            wg_dict = self.ray_worker_group_cls(resource_pool=rp, ray_cls_with_init=worker_dict_cls, **wg_kwargs)
            all_wg.update(wg_dict.spawn(prefix_set=class_dict.keys()))

        if self.use_reference_policy and not self.ref_in_actor:
            if str(Role.RefPolicy) in all_wg:
                self.ref_policy_wg = all_wg[str(Role.RefPolicy)]
                self.ref_policy_wg.init_model()

        self.actor_rollout_wg = all_wg[str(actor_role)]
        self.actor_rollout_wg.init_model()

        if self.ref_in_actor:
            self.ref_policy_wg = self.actor_rollout_wg

        # verl 0.8.0: the LLMServerManager (not the AgentLoopManager) owns/launches the
        # rollout replicas. Standalone servers => worker_group=None (resources come from
        # rollout.nnodes/n_gpus_per_node). The AgentLoopManager then drives them via a
        # partial-rollout client minted from the manager.
        self.llm_server_manager = FullyAsyncLLMServerManager.create(
            config=config,
            worker_group=None,
            rollout_resource_pool=None,
        )
        self.async_rollout_manager = FullyAsyncAgentLoopManager.create(
            config=config,
            llm_client=self.llm_server_manager.get_client(client_cls=FullyAsyncLLMServerClient),
        )

        self.checkpoint_manager = CheckpointEngineManager(
            backend=config.actor_rollout_ref.rollout.checkpoint_engine.backend,
            trainer=self.actor_rollout_wg,
            replicas=self._get_rollout_replicas(),
        )

    def _get_rollout_replicas(self):
        if hasattr(self.llm_server_manager, "get_replicas"):
            return self.llm_server_manager.get_replicas()
        return self.llm_server_manager.rollout_replicas

    def _get_rollout_server_addresses(self):
        if hasattr(self.llm_server_manager, "get_addresses"):
            return self.llm_server_manager.get_addresses()
        return self.llm_server_manager.server_addresses

    # =========================================================================
    # BackendProtocol interface methods
    # =========================================================================
    def init_rollout_engine(self, **kwargs) -> RolloutEngine:
        """Initialize the VerlEngine rollout engine.

        Note: This should be called after init_workers() to ensure
        async_rollout_manager is available.

        Returns:
            VerlEngine: The initialized rollout engine.
        """
        # Apply driver-side patches. Most verl monkey-patches only affect
        # worker code paths (FSDP / vLLM), so they live in the worker hook
        # at rllm.trainer.verl.patch:apply_all_verl_patches (wired via
        # runtime_env.worker_process_setup_hook in ray_runtime_env.py).
        from rllm.trainer.verl.patch import patch_verl_lsd_worker_hooks, patch_verl_tensordict_jagged_layout

        patch_verl_tensordict_jagged_layout()
        if self.lsd_config.enable:
            # Patch before create_colocated_worker_cls(...) so Ray exposes LSD RPCs
            # such as lsd_update_ema_teacher on the worker group proxy.
            patch_verl_lsd_worker_hooks()

        # Set actor.optim.total_training_steps before workers build the LR scheduler;
        # a verl-native trainer.total_training_steps override wins over the computed count.
        total_training_steps = kwargs.get("total_training_steps")
        if total_training_steps is not None:
            if self.config.trainer.get("total_training_steps") is not None:
                total_training_steps = self.config.trainer.total_training_steps
            try:
                OmegaConf.set_struct(self.config, True)
                with open_dict(self.config):
                    if OmegaConf.select(self.config, "actor_rollout_ref.actor.optim"):
                        self.config.actor_rollout_ref.actor.optim.total_training_steps = total_training_steps
            except Exception as e:
                logger.warning(f"Could not set total_training_steps on actor.optim: {e}")

        if self.is_separated:
            self._init_separated_workers()
        else:
            self._init_colocated_workers()

        assert self.async_rollout_manager is not None

        if self.lsd_config.enable:
            logger.info(
                "LSD enabled: routing_mode=%s teacher_mode=%s model_path=%s "
                "ref_in_actor=%s, lsd_coef=%s, easy_rl_coef=%s, easy_threshold=%s, online_group_threshold=%s",
                self.lsd_config.routing_mode,
                self.lsd_config.teacher_mode,
                self.config.actor_rollout_ref.model.path,
                self.ref_in_actor,
                self.lsd_config.lsd_coef,
                self.lsd_config.easy_rl_coef,
                self.lsd_config.easy_threshold,
                self.lsd_config.online_group_threshold,
            )

        if hasattr(self.actor_rollout_wg, "set_loss_fn"):
            self.actor_rollout_wg.set_loss_fn(CustomPPOLoss(self.config.actor_rollout_ref.actor, lsd_config=self.lsd_config))
        else:
            logger.warning("RayWorkerGroup.set_loss_fn not available — skipping custom loss injection")

        if (
            self.lsd_config.enable
            and str(getattr(self.lsd_config, "opd_loss_mode", "")).replace("-", "_")
            in {"topk", "topk_forward_kl", "forward_kl_topk"}
        ):
            if not hasattr(self.ref_policy_wg, "set_ref_loss_fn"):
                raise RuntimeError("LSD top-k OPD requires ref_policy_wg.set_ref_loss_fn")
            self.ref_policy_wg.set_ref_loss_fn(
                TeacherTopKExtractor(
                    topk=self.lsd_config.opd_topk,
                    stop_token_ids=self._resolve_lsd_stop_token_ids(),
                )
            )

        self._initialize_ema_teacher()

        # Both paths obtain the rollout client from the LLMServerManager; the separated
        # path uses the partial-rollout-aware client so preempted rollouts can resume.
        if self.is_separated and hasattr(self.llm_server_manager, "get_client"):
            from rllm.trainer.verl.async_agent_loop import FullyAsyncLLMServerClient

            server_client = self.llm_server_manager.get_client(client_cls=FullyAsyncLLMServerClient)
        elif hasattr(self.llm_server_manager, "get_client"):
            server_client = self.llm_server_manager.get_client()
        else:
            from verl.experimental.agent_loop.agent_loop import AsyncLLMServerManager

            server_client = AsyncLLMServerManager(self.config, self.llm_server_manager.server_handles)

        self.rollout_engine = VerlEngine(
            config=self.config,
            server_manager=server_client,
            tokenizer=self.tokenizer,
            processor=self.processor,
        )
        self.rollout_engine.server_addresses = self._get_rollout_server_addresses()

        self.algorithm_config = kwargs.get("algorithm_config")

        return self.rollout_engine

    def _resolve_lsd_stop_token_ids(self) -> list[int]:
        configured = getattr(self.lsd_config, "opd_stop_token_ids", None)
        if configured is not None:
            stop_ids = sorted({int(x) for x in configured})
            self._log_lsd_stop_tokens(stop_ids)
            return stop_ids
        tokenizer = self.tokenizer
        stop_ids: set[int] = set()
        eos_token_id = getattr(tokenizer, "eos_token_id", None)
        if eos_token_id is not None:
            if isinstance(eos_token_id, list):
                stop_ids.update(int(x) for x in eos_token_id if x is not None)
            else:
                stop_ids.add(int(eos_token_id))
        for token in getattr(tokenizer, "additional_special_tokens", []) or []:
            token_l = str(token).lower()
            if any(key in token_l for key in ("im_end", "eot", "endofturn", "end_of_turn")):
                token_id = tokenizer.convert_tokens_to_ids(token)
                if token_id is not None and token_id != getattr(tokenizer, "unk_token_id", None):
                    stop_ids.add(int(token_id))
        resolved = sorted(stop_ids)
        self._log_lsd_stop_tokens(resolved)
        return resolved

    def _log_lsd_stop_tokens(self, stop_ids: list[int]) -> None:
        token_strings = []
        for token_id in stop_ids:
            try:
                token_strings.append(self.tokenizer.convert_ids_to_tokens(int(token_id)))
            except Exception:
                token_strings.append("<decode_failed>")
        logger.info("LSD OPD stop-token support resolved: %s", list(zip(stop_ids, token_strings, strict=False)))

    def _lsd_teacher_is_ema(self) -> bool:
        return self.lsd_config.enable and getattr(self.lsd_config, "teacher_mode", "fixed_base") == "ema"

    def _lsd_ema_beta(self) -> float:
        half_life = float(getattr(self.lsd_config, "ema_half_life_updates", 50.0))
        interval = int(getattr(self.lsd_config, "ema_update_interval", 1) or 1)
        return 2.0 ** (-float(interval) / half_life)

    @staticmethod
    def _summarize_ema_update_result(result: Any) -> dict[str, float]:
        records = []

        def _collect(obj):
            if obj is None:
                return
            if isinstance(obj, dict):
                records.append(obj)
                return
            if isinstance(obj, (list, tuple)):
                for item in obj:
                    _collect(item)
                return
            try:
                from verl.utils import tensordict_utils as _tu

                metrics = _tu.get(obj, "metrics", default=None)
                if isinstance(metrics, dict):
                    records.append(metrics)
            except Exception:
                return

        _collect(result)
        out = {
            "ema_pre_update_max_param_diff": 0.0,
            "ema_post_update_max_param_diff": 0.0,
            "ema_param_count": 0.0,
        }
        for record in records:
            for key in out:
                value = record.get(key)
                if value is None:
                    continue
                if hasattr(value, "item"):
                    value = value.item()
                if key.endswith("max_param_diff"):
                    out[key] = max(out[key], float(value))
                else:
                    out[key] = max(out[key], float(value))
        return out

    def _initialize_ema_teacher(self) -> None:
        if not self._lsd_teacher_is_ema():
            return
        if not hasattr(self.ref_policy_wg, "lsd_update_ema_teacher"):
            raise RuntimeError("rllm.lsd.teacher_mode=ema requires ref_policy_wg.lsd_update_ema_teacher")
        result = self.ref_policy_wg.lsd_update_ema_teacher(beta=0.0, accum_dtype=getattr(self.lsd_config, "ema_accum_dtype", "fp32"))
        metrics = self._summarize_ema_update_result(result)
        self._pending_ema_init_metrics = {
            "lsd/ema_init_pre_max_param_diff": metrics["ema_pre_update_max_param_diff"],
            "lsd/ema_init_max_param_diff": metrics["ema_post_update_max_param_diff"],
            "lsd/ema_param_count": metrics["ema_param_count"],
        }

    @staticmethod
    def _actor_update_succeeded(metrics: dict[str, Any]) -> bool:
        if "optimizer_step_applied_min" in metrics:
            try:
                return bool(float(metrics["optimizer_step_applied_min"]) > 0.5)
            except Exception:
                return False
        if "optimizer_step_applied" in metrics:
            try:
                return bool(float(metrics["optimizer_step_applied"]) > 0.5)
            except Exception:
                return False
        if "found_inf" in metrics:
            try:
                return bool(float(metrics["found_inf"]) == 0.0)
            except Exception:
                return False
        grad_norm = metrics.get("grad_norm")
        if grad_norm is None:
            return False
        try:
            return bool(np.isfinite(float(grad_norm)))
        except Exception:
            return False

    def _maybe_update_ema_teacher(self, trainer_state: TrainerState) -> None:
        if not self._lsd_teacher_is_ema():
            return
        if not self._last_actor_update_success:
            trainer_state.metrics["lsd/ema_update_skipped"] = 1.0
            return
        interval = int(getattr(self.lsd_config, "ema_update_interval", 1) or 1)
        if trainer_state.global_step % interval != 0:
            trainer_state.metrics["lsd/ema_update_skipped"] = 1.0
            return
        if not hasattr(self.ref_policy_wg, "lsd_update_ema_teacher"):
            raise RuntimeError("rllm.lsd.teacher_mode=ema requires ref_policy_wg.lsd_update_ema_teacher")

        beta = self._lsd_ema_beta()
        accum_dtype = getattr(self.lsd_config, "ema_accum_dtype", "fp32")
        result = self.ref_policy_wg.lsd_update_ema_teacher(beta=beta, accum_dtype=accum_dtype)
        metrics = self._summarize_ema_update_result(result)
        trainer_state.metrics["lsd/ema_beta"] = beta
        trainer_state.metrics["lsd/ema_update"] = 1.0
        trainer_state.metrics["lsd/ema_pre_update_max_param_diff"] = metrics["ema_pre_update_max_param_diff"]
        trainer_state.metrics["lsd/ema_post_update_max_param_diff"] = metrics["ema_post_update_max_param_diff"]
        trainer_state.metrics["lsd/ema_param_count"] = metrics["ema_param_count"]

    def validate_config(self) -> None:
        """Validate verl-specific configuration settings."""
        assert self.config.actor_rollout_ref.rollout.mode == "async", "Only async rollout mode is supported for VerlBackend"
        if self.is_separated:
            async_cfg = self.config.rllm.async_training
            fwd_bwd = async_cfg.fwd_bwd_group_size if async_cfg.fwd_bwd_group_size is not None else async_cfg.mini_batch_size
            assert fwd_bwd == async_cfg.mini_batch_size, (
                f"VerlBackend requires async_training.fwd_bwd_group_size == mini_batch_size (got {async_cfg.fwd_bwd_group_size} vs {async_cfg.mini_batch_size})"
            )
            if async_cfg.get("partial_rollout", False) and self.config.rllm.get("remote_runtime", {}).get("enabled", False):
                raise ValueError("VerlBackend: async_training.partial_rollout is not supported with remote_runtime; set it to false.")
            # Separated mode does trainer->rollout weight sync via verl's 'nccl' checkpoint
            # engine, which imports cupy. verl's checkpoint_engine/__init__ swallows a missing
            # cupy in a silent try/except, so the failure otherwise surfaces only as a confusing
            # "ValueError: Checkpoint engine nccl not registered" deep in actor_init_model.
            # Fail fast here with an actionable message instead.
            try:
                import cupy  # noqa: F401
            except ImportError as e:
                raise RuntimeError(
                    "Separated/async training (rllm.async_training.enable=true) requires cupy for "
                    "verl's 'nccl' checkpoint engine (trainer->rollout weight sync). Install the wheel "
                    "matching your CUDA toolkit: `pip install cupy-cuda12x` (CUDA 12.x) or "
                    "`pip install cupy-cuda13x` (CUDA 13.x)."
                ) from e
        if self.config.rllm.stepwise_advantage.mode != "broadcast":
            # automatically set the stepwise_advantage_mode to "broadcast", the warning is already shown in AlgorithmConfig.from_config
            self.config.rllm.stepwise_advantage.mode = "broadcast"

        rc = self.config.rllm.algorithm.get("rollout_correction", {})
        if rc.get("bypass_mode", False) and rc.get("tis_mode") is not None:
            raise ValueError("bypass_mode=True and tis_mode!=None is invalid: IS correction is meaningless when π_old = π_rollout")

        assert not self.config.algorithm.get("use_kl_in_reward", False), "only KL-in-loss is supported"

        reward_model_cfg = self.config.get("reward", {}).get("reward_model", {})
        assert not reward_model_cfg.get("enable", False), (
            "Reward models are not supported on the rLLM-native verl path; compute rewards in the workflow via a RewardFunction. Remove `reward.reward_model.enable=True` from your config."
        )

        router_replay_mode = self.config.rllm.algorithm.get("router_replay", "disabled")
        if router_replay_mode != "disabled":
            strategy = self.config.actor_rollout_ref.actor.strategy
            if strategy != "megatron":
                raise ValueError(f"router_replay={router_replay_mode!r} requires actor.strategy='megatron', got {strategy!r}")

        if self.lsd_config.enable:
            self.lsd_config.validate()
            if self._lsd_teacher_is_ema():
                hard_kl_enabled = bool(self.config.actor_rollout_ref.actor.get("use_kl_loss", False))
                hard_kl_enabled = hard_kl_enabled or bool(self.config.algorithm.get("use_kl_in_reward", False))
                hard_kl_enabled = hard_kl_enabled or float(self.config.rllm.algorithm.get("kl_beta", 0.0) or 0.0) > 0.0
                if hard_kl_enabled:
                    raise ValueError(
                        "rllm.lsd.teacher_mode=ema cannot reuse the fixed reference policy while hard RL KL is enabled. "
                        "Disable hard KL or add a separate EMA teacher worker."
                    )
                if self.ref_in_actor:
                    raise ValueError("rllm.lsd.teacher_mode=ema requires an isolated ref worker; ref_in_actor is not allowed")
                if self.is_separated:
                    raise ValueError("rllm.lsd.teacher_mode=ema is currently supported only in colocated ActorRolloutRefWorker mode")

    async def generate_episodes(self, batch: Any, agent_workflow_engine: UnifiedWorkflowEngine, is_validation: bool = False, **kwargs) -> list[Episode]:
        """Generate episodes using the workflow engine.

        For Verl backend, this function handles the following procedures:

        1. Build an "interleaved" batch, where each task is repeated `rollout.n` times.
        2. Extract the tasks and task IDs from the batch.
        3. Execute the tasks using the agent workflow engine.
        4. Return the episodes.

        Args:
            batch: Input batch (list of rllm task dicts from the dataloader).
            agent_workflow_engine: The workflow engine to use.
            **kwargs: Additional arguments.

        Returns:
            List of generated episodes.
        """
        repeat_times = self.full_config.rllm.rollout.n_val if is_validation else self.full_config.rllm.rollout.n
        tasks, task_ids = interleave_tasks(batch, repeat_times)
        episodes = await self._execute_tasks_async(tasks, task_ids, agent_workflow_engine, is_validation=is_validation, **kwargs)
        # Sleep the replicas to free kv_cache before weight sync (colocated only). Skip during
        # validation (no weight sync follows) and in separated mode (rollout on own resources).
        if not is_validation and not self.is_separated:
            await self.checkpoint_manager.sleep_replicas()
        return episodes

    async def _execute_tasks_async(self, tasks: list, task_ids: list[str], agent_workflow_engine: UnifiedWorkflowEngine, **kwargs) -> list[Episode]:
        """A Verl-specific helper function to execute tasks asynchronously."""
        assert self.rollout_engine is not None, "rollout_engine is not initialized."
        episodes = await agent_workflow_engine.execute_tasks(tasks, task_ids, **kwargs)
        for episode, task in zip(episodes, tasks, strict=True):
            data_source = task.get("data_source") if isinstance(task, dict) else None
            if data_source is not None:
                episode.info["data_source"] = data_source
        return episodes

    def transform_to_backend_batch(self, trainer_state: TrainerState, **kwargs) -> DataProto:
        """Transform rllm-native data structures to verl DataProto format.

        Sync path receives ``episodes``; fully-async path (post-buffer) receives
        ``trajectory_groups``.
        """
        assert self.rollout_engine is not None, "rollout_engine is not initialized."
        max_prompt_length = self.config.data.max_prompt_length
        # data.max_response_length is the per-turn generation cap at rollout
        # time, but merged multi-turn responses concatenate [A0, obs1, A1, ...]
        # and can grow up to the full context window - so use max_total_length to
        # bound the sequence.
        max_total_length = max_prompt_length + self.config.data.max_response_length

        if trainer_state.episodes is not None:
            batch = transform_episodes_to_dataproto(
                trainer_state.episodes,
                self.rollout_engine,
                max_prompt_length,
                max_total_length,
                lsd_router=self.lsd_router,
                lsd_config=self.lsd_config,
            )
            # Lift per-batch merge metrics (batch/steps_per_traj,
            # batch/step_response_length) out of meta_info so they show up in
            # the standard trainer_state.metrics path. Same metric names the
            # tinker backend logs, so dashboards work across both.
            merge_metrics = batch.meta_info.pop("merge_metrics", None)
            if merge_metrics:
                trainer_state.metrics.update(merge_metrics)
            return batch

        assert trainer_state.trajectory_groups is not None, "Either episodes or trajectory_groups must be set"
        if self.lsd_config.enable and self.lsd_config.routing_mode == "online_group":
            raise ValueError("rllm.lsd.routing_mode=online_group requires Episode routing; TrajectoryGroup batches are not supported")
        batch = transform_trajectory_groups_to_dataproto(trainer_state.trajectory_groups, self.rollout_engine, max_prompt_length, max_total_length)
        mode = self.algorithm_config.stepwise_advantage_mode if self.algorithm_config is not None else "broadcast"
        return update_dataproto_with_advantages(batch, trainer_state.trajectory_groups, mode=mode)

    def _remove_padding(self, batch: DataProto) -> DataProto:
        """Removes padded steps from the batch"""
        is_pad_step = batch.non_tensor_batch["is_pad_step"]
        non_pad_step_indices = np.where(is_pad_step == False)[0]
        batch = batch.select_idxs(non_pad_step_indices)  # This batch only has non_pad steps
        return batch

    def _get_dp_size(self, worker_group, mesh_name: str) -> int:
        """Query actual DP size for a worker group mesh via dispatch info.

        Mirrors ``RayPPOTrainer._get_dp_size``: the dispatch-info mapping
        assigns a dp_rank to each global worker rank, so dp_size is
        ``max(dp_rank_mapping) + 1``.
        """
        if mesh_name not in worker_group._dispatch_info:
            worker_group._dispatch_info[mesh_name] = worker_group._query_dispatch_info(mesh_name)
        return max(worker_group._dispatch_info[mesh_name]) + 1

    def _get_aggregate_dp_size(self) -> int | None:
        """Compute the LCM of DP sizes across all active worker-group meshes.

        Mesh names target the EngineWorker path:
        - actor_rollout_wg -> ``engine_workers.ActorRolloutRefWorker``
            registers ``"actor"`` and ``"ref"``.
        - ref_policy_wg   -> same ``ActorRolloutRefWorker`` as actor_rollout_wg,
            so the registered mesh is ``"ref"``.
        """
        dp_sizes: list[int] = []
        # ref_in_actor (LoRA): ref log-probs run on the actor mesh with
        # no_lora_adapter=True, and the "ref" mesh is not registered.
        if self.use_reference_policy and not self.ref_in_actor and self.ref_policy_wg.world_size != 0:
            dp_sizes.append(self._get_dp_size(self.ref_policy_wg, "ref"))
        if self.actor_rollout_wg.world_size != 0:
            dp_sizes.append(self._get_dp_size(self.actor_rollout_wg, "actor"))
        if not dp_sizes:
            return None
        return reduce(math.lcm, dp_sizes)

    def _pad_dataproto_to_world_size(self, batch: DataProto) -> DataProto:
        from verl.protocol import pad_dataproto_to_divisor

        dp_size = self._get_aggregate_dp_size()
        if dp_size is None:
            return batch

        # Multi-turn rollouts are prefix-merged into training rows on a best-effort
        # basis; the merge frequently breaks (chat-template re-render, context
        # management), so the real row count N is much larger than the rollout
        # count and varies step-to-step. We make the number of optimizer steps a
        # deterministic quantity r derived from config, and decouple the two batch
        # sizes the verl worker consumes (which it treats independently):
        #   - mini_batch_size = m  (ROWS per optimizer step) controls how
        #     train_mini_batch chunks the batch, hence the update count. Verl
        #     requires m % dp_size == 0 and total_rows % m == 0.
        #   - global_batch_size = gbs (ROLLOUTS per update) is the seq-mean loss
        #     denominator. Keeping it at the constant rollout count R_total // r
        #     (not the drifting row count) gives the conventional per-example loss
        #     scale and keeps it stable across steps regardless of merge ratio.
        rollout_n = self.config.actor_rollout_ref.rollout.n
        ppo_mbs = self.config.actor_rollout_ref.actor.ppo_mini_batch_size
        train_batch_size = self.config.data.train_batch_size
        r = max(1, train_batch_size // ppo_mbs)  # desired optimizer steps per generation batch
        r_total = train_batch_size * rollout_n  # total rollouts in this generation batch

        batch = self._remove_padding(batch)  # Remove any padded steps from the batch (just in case)
        original_batch_size = batch.batch["prompts"].shape[0]
        n_rows = original_batch_size

        # m = smallest multiple of dp_size that is >= ceil(N / r). Padding N up to
        # the next multiple of m then yields exactly r chunks (one optimizer step
        # each). Rounding m up to dp_size can rarely overshoot so that N spans only
        # r-1 multiples; in that case accept the smaller achievable count.
        #
        # Infeasibility (actual_r < r): each mini-batch must split across the
        # dp_size ranks, so m >= dp_size always. The most chunks N can be cut into
        # is therefore ceil(N / dp_size); requesting r > ceil(N / dp_size) is
        # physically impossible (e.g. N=10, dp_size=8, r=4 -> m=8 -> only 2 chunks).
        # This requires a tiny (heavily filtered) batch or an extreme
        # train_batch_size // ppo_mini_batch_size ratio; it does not occur for
        # normal configs where N (hundreds-thousands) >> dp_size * r. actual_r
        # is always <= r, so this only ever lowers the update count, never raises it.
        per_update = math.ceil(n_rows / r)
        m = math.ceil(per_update / dp_size) * dp_size
        actual_r = math.ceil(n_rows / m)
        if actual_r != r:
            logger.warning(f"[update-count] requested r={r} infeasible after dp_size rounding (N={n_rows}, m={m}); using r={actual_r}")
            r = actual_r
        gbs = max(1, r_total // r)
        divisor = m

        batch, pad_size = pad_dataproto_to_divisor(batch, divisor)

        # Stash the decoupled sizes; read back in _update_actor_with_loss_routing.
        # They are computed here (pad time) because the pad divisor and the
        # update-time sizes must agree, and these run in different trainer phases.
        batch.meta_info["rllm_actor_mini_batch_size"] = m
        batch.meta_info["rllm_actor_global_batch_size"] = gbs
        batch.meta_info["rllm_actor_num_updates"] = r
        batch.meta_info["rllm_actor_rows_pre_pad"] = n_rows

        # Neutralise the padded rows. `advantages=0` (set by
        # update_dataproto_with_advantages via is_pad_step) zeros the loss
        # numerator; zeroing `response_mask` keeps pad tokens out of the loss.
        # The seq-mean denominator (global_batch_size = gbs) is a fixed rollout
        # count, not a live row/token count, so pad rows do not dilute it at r=1.
        # At r>1 the single chunk holding the pad rows is under-scaled by ~pad/m
        # (bounded by r*dp_size rows total), but the aggregate over all r updates
        # is exact — an accepted trade-off vs. a variable per-step denominator.
        pad_start, pad_end = original_batch_size, original_batch_size + pad_size
        batch.batch["response_mask"][pad_start:pad_end] = 0
        if "lsd_is_easy" in batch.batch:
            batch.batch["lsd_is_easy"][pad_start:pad_end] = False
        if "lsd_base_acc" in batch.batch:
            batch.batch["lsd_base_acc"][pad_start:pad_end] = 0.0
        if "lsd_base_length" in batch.batch:
            batch.batch["lsd_base_length"][pad_start:pad_end] = 0.0
        batch.non_tensor_batch["is_last_step"][pad_start:pad_end] = False
        batch.non_tensor_batch["is_pad_step"][pad_start:pad_end] = True
        batch.non_tensor_batch["is_valid"][pad_start:pad_end] = False
        return batch

    async def process_backend_batch(self, trainer_state: TrainerState, **kwargs) -> None:
        """Compute step-level values: old_log_probs, ref_log_probs, critic values.

        Uses the new EngineWorker path: converts DataProto to TensorDict in
        no-padding format, calls workers, converts results back to padded
        DataProto.  The no-padding TensorDict (batch_td) is created once and
        reused across all inference worker calls.
        """
        metrics = trainer_state.metrics
        timing_dict = trainer_state.timing_dict
        batch: DataProto = trainer_state.backend_batch  # type: ignore[assignment]
        from rllm.trainer.verl.patch import _debug_report_patch

        # Balance the number of valid tokens across DP ranks.
        # NOTE: This usually changes the order of data in the `batch`,
        # which won't affect the advantage calculation (since it's based on uid),
        # but might affect the loss calculation (due to the change of mini-batching).
        if self.config.trainer.balance_batch:
            # pad batch size to world size for batch balancing
            batch = self._pad_dataproto_to_world_size(batch=batch)
            balance_batch(batch, self.actor_rollout_wg, metrics=metrics, use_prefix_grouper=self.use_prefix_grouper)
            # Surface the now-deterministic update count and the decoupled batch
            # sizes computed in _pad_dataproto_to_world_size.
            if "rllm_actor_num_updates" in batch.meta_info:
                metrics["actor/num_updates_per_epoch"] = batch.meta_info["rllm_actor_num_updates"]
                metrics["actor/mini_batch_rows"] = batch.meta_info["rllm_actor_mini_batch_size"]
                metrics["actor/global_batch_rollouts"] = batch.meta_info["rllm_actor_global_batch_size"]
                metrics["batch/rows_pre_pad"] = batch.meta_info["rllm_actor_rows_pre_pad"]
                metrics["batch/rows_post_pad"] = batch.batch["prompts"].shape[0]
            # #region debug-point E:balanced-batch
            _debug_report_patch(
                "E",
                "verl_backend.py:process_backend_batch",
                "[DEBUG] batch padding and balancing succeeded",
                {"tensor_shapes": {key: list(value.shape) for key, value in batch.batch.items()}},
            )
            # #endregion

        # Set meta_info needed by workers
        batch.meta_info["global_token_num"] = torch.sum(batch.batch["attention_mask"], dim=-1).tolist()
        batch.meta_info["temperature"] = self.config.actor_rollout_ref.rollout.temperature
        if "multi_modal_inputs" in batch.non_tensor_batch:
            images_seqlens_all = []
            for multi_modal_input in batch.non_tensor_batch["multi_modal_inputs"]:
                if "image_grid_thw" not in multi_modal_input:
                    continue
                images_seqlens_all.extend(multi_modal_input["images_seqlens"].tolist())
            batch.meta_info["images_seqlens"] = images_seqlens_all

        # Convert to TensorDict + no-padding ONCE — reused for all inference calls.
        # to_tensordict() does NOT mutate the original DataProto.
        # left_right_2_no_padding mutates batch_td in-place.
        batch_td = batch.to_tensordict()
        # #region debug-point F:padded-tensordict
        _debug_report_patch(
            "F",
            "verl_backend.py:process_backend_batch",
            "[DEBUG] padded TensorDict constructed",
            {"tensor_shapes": {key: list(value.shape) for key, value in batch_td.items() if hasattr(value, "shape")}},
        )
        # #endregion
        batch_td = left_right_2_no_padding(batch_td)
        # #region debug-point F:no-padding-tensordict
        _debug_report_patch(
            "F",
            "verl_backend.py:process_backend_batch",
            "[DEBUG] no-padding TensorDict constructed",
            {
                "tensor_shapes": {
                    key: {
                        "shape": list(value.shape),
                        "nested": bool(getattr(value, "is_nested", False)),
                        "values_shape": list(value.values().shape) if getattr(value, "is_nested", False) else None,
                    }
                    for key, value in batch_td.items()
                    if hasattr(value, "shape")
                }
            },
        )
        # #endregion

        # --- Compute old_log_probs ---
        rc = self.algorithm_config.rollout_correction if self.algorithm_config is not None else None
        bypass_mode = rc is not None and rc.bypass_mode
        has_rollout_log_probs = "rollout_log_probs" in batch.batch

        if bypass_mode:
            assert has_rollout_log_probs, "bypass_mode requires rollout_log_probs in batch"
            with simple_timer("old_log_probs", timing_dict):
                batch.batch["old_log_probs"] = batch.batch["rollout_log_probs"]
        else:
            with simple_timer("old_log_probs", timing_dict):
                tu.assign_non_tensor(batch_td, calculate_entropy=True, compute_loss=False)
                output = self.actor_rollout_wg.compute_log_prob(batch_td)
                # #region debug-point G:old-logprob
                _debug_report_patch(
                    "G",
                    "verl_backend.py:process_backend_batch",
                    "[DEBUG] actor old-logprob forward succeeded",
                    {
                        "output_shapes": {
                            key: list(value.shape)
                            for key, value in output.items()
                            if hasattr(value, "shape")
                        }
                    },
                )
                # #endregion
                log_probs = no_padding_2_padding(tu.get(output, "log_probs"), batch_td)
                entropy = no_padding_2_padding(tu.get(output, "entropy"), batch_td)
                routed_experts = tu.get(output, "routed_experts", default=None)

                # Build the old_log_prob DataProto. Include routed_experts when verl's
                # R2 router-replay path populated it during the proximal forward pass —
                # the actor update reads it from the batch in megatron_actor.py.
                old_log_prob_tensors = {"old_log_probs": log_probs.float(), "entropys": entropy.float()}
                if routed_experts is not None:
                    old_log_prob_tensors["routed_experts"] = routed_experts
                old_log_prob = DataProto.from_tensordict(tu.get_tensordict(old_log_prob_tensors))

                # Entropy metric (for logging only); pop before union so it doesn't
                # leak into the loss path.
                entropys = old_log_prob.batch["entropys"]
                response_masks = batch.batch["response_mask"]
                loss_agg_mode = self.config.actor_rollout_ref.actor.loss_agg_mode
                entropy_agg = agg_loss(loss_mat=entropys, loss_mask=response_masks, loss_agg_mode=loss_agg_mode)
                metrics["actor/entropy"] = entropy_agg.detach().item()
                old_log_prob.batch.pop("entropys")

                batch = batch.union(old_log_prob)

            tis_mode = rc.tis_mode if rc is not None else None
            if tis_mode is not None and has_rollout_log_probs:
                with simple_timer("rollout_correction", timing_dict):
                    from verl.trainer.ppo.rollout_corr_helper import compute_rollout_correction_weights

                    log_ratio = batch.batch["old_log_probs"] - batch.batch["rollout_log_probs"]
                    rollout_is_weights, is_metrics = compute_rollout_correction_weights(
                        log_ratio=log_ratio,
                        response_mask=batch.batch["response_mask"],
                        rollout_is=tis_mode,
                        rollout_is_threshold=rc.tis_cap,
                    )
                    batch.batch["rollout_is_weights"] = rollout_is_weights
                    metrics.update({f"rollout_correction/{k}": v for k, v in is_metrics.items()})

        # Off-policy diagnostics: KL, log-PPL difference, ppl ratio, pearson correlation,
        # etc. between rollout and proximal log-probs. Runs for both bypass and non-bypass
        # paths whenever rollout_log_probs is available, so the same metrics show up
        # regardless of which π_old source is being used.
        if has_rollout_log_probs:
            from verl.trainer.ppo.rollout_corr_helper import compute_offpolicy_metrics

            offpolicy_metrics = compute_offpolicy_metrics(
                old_log_prob=batch.batch["old_log_probs"],
                rollout_log_prob=batch.batch["rollout_log_probs"],
                response_mask=batch.batch["response_mask"],
            )
            metrics.update({f"offpolicy/{k}": v for k, v in offpolicy_metrics.items()})
            metrics.update(calculate_debug_metrics_compat(batch))

            # #region debug-point B:actor-rollout-version-and-route-diff
            try:
                import json
                import urllib.request

                response_mask = batch.batch["response_mask"].bool()
                probability_diff = (
                    batch.batch["old_log_probs"].exp() - batch.batch["rollout_log_probs"].exp()
                ).abs()
                route_metrics = {}
                if "lsd_is_easy" in batch.batch:
                    easy_rows = batch.batch["lsd_is_easy"].bool()
                    if easy_rows.dim() > 1:
                        easy_rows = easy_rows.reshape(easy_rows.shape[0], -1)[:, 0]
                    for route_name, row_mask in (("easy", easy_rows), ("hard", ~easy_rows)):
                        token_mask = response_mask & row_mask.unsqueeze(-1)
                        if token_mask.any():
                            route_metrics[f"{route_name}_prob_diff_mean"] = float(
                                probability_diff[token_mask].mean().detach().item()
                            )
                            route_metrics[f"{route_name}_actor_prob_mean"] = float(
                                batch.batch["old_log_probs"].exp()[token_mask].mean().detach().item()
                            )
                            route_metrics[f"{route_name}_rollout_prob_mean"] = float(
                                batch.batch["rollout_log_probs"].exp()[token_mask].mean().detach().item()
                            )
                urllib.request.urlopen(
                    urllib.request.Request(
                        "http://127.0.0.1:7779/event",
                        data=json.dumps(
                            {
                                "sessionId": "rollout-prob-diff",
                                "runId": "post-fix",
                                "hypothesisId": "B",
                                "location": "rllm/trainer/verl/verl_backend.py:process_backend_batch",
                                "msg": "[DEBUG] actor-rollout probability evidence",
                                "data": {
                                    "prob_diff_mean": metrics.get("training/rollout_probs_diff_mean"),
                                    "pearson_corr": metrics.get("training/rollout_actor_probs_pearson_corr"),
                                    "offpolicy_log_ppl_abs_diff": metrics.get("offpolicy/log_ppl_abs_diff"),
                                    "actor_entropy": metrics.get("actor/entropy"),
                                    "current_param_version": getattr(self, "current_param_version", None),
                                    "batch_version_meta": {
                                        key: value
                                        for key, value in batch.meta_info.items()
                                        if "version" in str(key)
                                    },
                                    "actor_strategy": str(self.config.actor_rollout_ref.actor.strategy),
                                    "actor_dtype": str(self.config.actor_rollout_ref.actor.fsdp_config.dtype),
                                    "rollout_dtype": str(self.config.actor_rollout_ref.rollout.dtype),
                                    "sequence_parallel_size": int(
                                        self.config.actor_rollout_ref.actor.ulysses_sequence_parallel_size
                                    ),
                                    "hybrid_engine": bool(self.config.actor_rollout_ref.hybrid_engine),
                                    **route_metrics,
                                },
                            }
                        ).encode(),
                        headers={"Content-Type": "application/json"},
                    ),
                    timeout=2.0,
                ).read()
            except Exception:
                pass
            # #endregion

        # --- Compute reference log_probs (reuse batch_td) ---
        if self.use_reference_policy:
            with simple_timer("ref", timing_dict):
                lsd_topk_opd = self.lsd_config.enable and str(
                    getattr(self.lsd_config, "opd_loss_mode", "")
                ).replace("-", "_") in {"topk", "topk_forward_kl", "forward_kl_topk"}
                tu.assign_non_tensor(
                    batch_td,
                    calculate_entropy=False,
                    compute_loss=lsd_topk_opd and not self.ref_in_actor,
                    distillation_use_topk=lsd_topk_opd and not self.ref_in_actor,
                )
                if not self.ref_in_actor:
                    ref_output = self.ref_policy_wg.compute_ref_log_prob(batch_td)
                else:
                    tu.assign_non_tensor(batch_td, no_lora_adapter=True)
                    ref_output = self.actor_rollout_wg.compute_log_prob(batch_td)
                ref_lp = no_padding_2_padding(tu.get(ref_output, "log_probs"), batch_td)
                ref_tensors = {"ref_log_prob": ref_lp.float()}
                if lsd_topk_opd and not self.ref_in_actor:
                    teacher_ids = tu.get(ref_output, "teacher_ids", default=None)
                    teacher_logprobs = tu.get(ref_output, "teacher_logprobs", default=None)
                    if teacher_ids is None or teacher_logprobs is None:
                        raise RuntimeError("LSD top-k OPD requested but ref worker did not return teacher top-k tensors")
                    ref_tensors["teacher_ids"] = _no_padding_to_attention_padded(teacher_ids, batch.batch["attention_mask"]).long()
                    ref_tensors["teacher_logprobs"] = _no_padding_to_attention_padded(teacher_logprobs, batch.batch["attention_mask"]).float()
                ref_log_prob = DataProto.from_tensordict(tu.get_tensordict(ref_tensors))
                batch = batch.union(ref_log_prob)

                if self.lsd_config.enable and "old_log_probs" in batch.batch:
                    response_mask = batch.batch["response_mask"].bool()
                    easy = batch.batch.get("lsd_is_easy")
                    metrics["lsd/use_reference_policy"] = 1.0
                    metrics["lsd/ref_in_actor"] = 1.0 if self.ref_in_actor else 0.0
                    metrics["lsd/teacher_mode_ema"] = 1.0 if self._lsd_teacher_is_ema() else 0.0
                    metrics["lsd/teacher_frozen"] = 0.0 if self._lsd_teacher_is_ema() else 1.0
                    if easy is not None:
                        metrics["lsd/easy_row_fraction_pre_update"] = float(easy.float().mean().item())
                    if response_mask.any():
                        delta = torch.masked_select(batch.batch["ref_log_prob"] - batch.batch["old_log_probs"], response_mask)
                        metrics["lsd/ref_old_logratio_pre_update/mean"] = float(delta.mean().item())
                        metrics["lsd/ref_old_logratio_pre_update/abs_mean"] = float(delta.abs().mean().item())

        # Mask truncated samples if configured
        if self.config.rllm.get("mask_truncated_samples", False):
            mask = batch.batch["attention_mask"][:, -1] == 1
            batch = batch[~mask]

        trainer_state.backend_batch = batch

    async def compute_advantages(self, trainer_state: TrainerState, algorithm_config: AlgorithmConfig, **kwargs) -> None:
        """Compute advantages from trajectory groups.

        Note: This is async for protocol compatibility but operations are sync.
        """
        assert trainer_state.episodes is not None, "Episodes are not set"
        assert trainer_state.trajectory_groups is not None, "Trajectory groups are not set"
        episodes, trajectory_groups = trainer_state.episodes, trainer_state.trajectory_groups
        batch: DataProto = trainer_state.backend_batch  # type: ignore[assignment]

        with simple_timer("adv", trainer_state.timing_dict):
            adv_metrics = collect_reward_and_advantage_from_trajectory_groups(trajectory_groups, algorithm_config)
            updated_batch = update_dataproto_with_advantages(batch, episodes, mode=algorithm_config.stepwise_advantage_mode)

        trainer_state.metrics.update(adv_metrics)
        trainer_state.backend_batch = updated_batch

    async def update_policy(self, trainer_state: TrainerState, **kwargs) -> None:
        """Update actor and critic policies.

        Uses the new EngineWorker path: converts DataProto to TensorDict in
        no-padding format with training metadata, then calls workers.  The new
        workers handle micro-batching internally, so no manual re-padding is
        needed before the update.
        """
        global_steps = trainer_state.global_step
        batch: DataProto = trainer_state.backend_batch  # type: ignore[assignment]

        if self.config.trainer.get("critic_warmup", 0) <= global_steps:
            with simple_timer("update_actor", trainer_state.timing_dict):
                self._update_actor_with_loss_routing(batch, trainer_state)
                self._maybe_update_ema_teacher(trainer_state)

    def _update_actor_with_loss_routing(self, batch: DataProto, trainer_state: TrainerState) -> None:
        """Update actor with per-loss-group splitting when ``loss_fn_map`` is set.

        Roles that share the same policy loss function are grouped together
        into a single ``update_actor`` call, minimising the number of
        optimiser steps.  Each (sub-)batch is converted to TensorDict +
        no-padding format with training metadata before being sent to the
        worker.
        """
        loss_fn_map = self.algorithm_config.loss_fn_map if self.algorithm_config is not None else {}
        group_roles = batch.non_tensor_batch.get("group_roles") if hasattr(batch, "non_tensor_batch") and batch.non_tensor_batch is not None else None

        # Common training metadata. mini_batch_size (ROWS per optimizer step) and
        # global_batch_size (ROLLOUTS, the seq-mean loss denominator) are decoupled
        # and computed in _pad_dataproto_to_world_size; read them back here. The
        # fallback (both = ppo_mini_batch_size * rollout.n) preserves the legacy
        # behavior when balance_batch is off and no meta was stashed.
        rollout_n = self.config.actor_rollout_ref.rollout.n
        actor_cfg = self.config.actor_rollout_ref.actor
        default_mbs = actor_cfg.ppo_mini_batch_size * rollout_n
        full_mini_batch_size = batch.meta_info.get("rllm_actor_mini_batch_size", default_mbs)
        full_global_batch_size = batch.meta_info.get("rllm_actor_global_batch_size", default_mbs)

        def _send_actor_update(
            sub_batch: DataProto,
            loss_override: str | None = None,
            mini_batch_size: int | None = None,
            global_batch_size: int | None = None,
        ) -> None:
            """Convert DataProto to TensorDict, inject metadata, send to worker."""
            batch_td = sub_batch.to_tensordict()
            batch_td = left_right_2_no_padding(batch_td)
            for key in ("teacher_ids", "teacher_logprobs"):
                if key in batch_td and not batch_td[key].is_nested:
                    batch_td[key] = _attention_padded_to_no_padding(
                        batch_td[key],
                        batch_td["attention_mask"],
                        batch_td["input_ids"].offsets(),
                    )
            # #region debug-point H:teacher-no-padding
            if "teacher_ids" in batch_td and "teacher_logprobs" in batch_td:
                from rllm.trainer.verl.patch import _debug_report_patch

                _debug_report_patch(
                    "H",
                    "verl_backend.py:_send_actor_update",
                    "[DEBUG] teacher tensors converted to no-padding layout",
                    {
                        key: {
                            "shape": list(batch_td[key].shape),
                            "nested": bool(batch_td[key].is_nested),
                            "values_shape": list(batch_td[key].values().shape),
                            "ragged_idx": getattr(batch_td[key], "_ragged_idx", None),
                        }
                        for key in ("teacher_ids", "teacher_logprobs")
                    },
                )
            # #endregion
            metadata: dict[str, Any] = dict(
                calculate_entropy=(actor_cfg.entropy_coeff != 0.0),
                global_batch_size=global_batch_size if global_batch_size is not None else full_global_batch_size,
                mini_batch_size=mini_batch_size if mini_batch_size is not None else full_mini_batch_size,
                epochs=actor_cfg.ppo_epochs,
                seed=actor_cfg.data_loader_seed,
                dataloader_kwargs={"shuffle": actor_cfg.shuffle},
            )
            if self.lsd_config.enable and str(getattr(self.lsd_config, "opd_loss_mode", "")).replace("-", "_") in {
                "topk",
                "topk_forward_kl",
                "forward_kl_topk",
            }:
                metadata["distillation_use_topk"] = True
            if loss_override is not None:
                metadata["policy_loss_mode_override"] = loss_override
            tu.assign_non_tensor(batch_td, **metadata)
            actor_output = self.actor_rollout_wg.update_actor(batch_td)
            actor_metrics = tu.get(actor_output, "metrics")
            reduced_metrics = reduce_metrics(actor_metrics)
            trainer_state.metrics.update(reduced_metrics)
            self._last_actor_update_success = self._last_actor_update_success and self._actor_update_succeeded(
                reduced_metrics
            )

        # Fast path: no per-role loss overrides or no role annotations.
        self._last_actor_update_success = True
        if not loss_fn_map or group_roles is None:
            _send_actor_update(batch)
            return

        # Resolve each role to a Verl loss name with validation + fallback.
        known = _get_verl_known_losses()
        role_to_loss: dict[str, str] = {}
        for role in set(group_roles.tolist()):
            loss_name = loss_fn_map.get(role, _DEFAULT_VERL_LOSS)
            if loss_name not in known:
                logger.warning(f"Unknown Verl loss '{loss_name}' for role '{role}', falling back to '{_DEFAULT_VERL_LOSS}'")
                loss_name = _DEFAULT_VERL_LOSS
            role_to_loss[role] = loss_name

        # Regroup: collect roles by their loss function.
        loss_to_roles: dict[str, list[str]] = defaultdict(list)
        for role, loss in role_to_loss.items():
            loss_to_roles[loss].append(role)

        if len(loss_to_roles) <= 1:
            # All roles share the same loss — single update.
            _send_actor_update(batch, next(iter(loss_to_roles)))
            return

        # Multiple distinct losses: split batch by loss group, update each.
        # TODO: multi-loss path still uses flat ppo_mbs*rollout_n semantics;
        # needs per-group padding when exercised with the new const-steps logic.
        for loss_name, roles in loss_to_roles.items():
            role_set = set(roles)
            mask = np.array([role in role_set for role in group_roles])
            indices = np.where(mask)[0]
            sub_batch = batch[indices]
            _send_actor_update(sub_batch, loss_name, mini_batch_size=default_mbs, global_batch_size=default_mbs)

    def shutdown(self) -> None:
        """Free GPU memory held by the rollout replicas.

        Without this, a crash mid-training (or anywhere ``on_train_end``
        does not get to run) leaves the vLLM replicas awake on every GPU,
        holding tens of GB each until the Ray actors are torn down.
        """
        try:
            if self.checkpoint_manager is not None:
                self.checkpoint_manager.sleep_replicas()
        except Exception:
            logger.exception("VerlBackend.shutdown: sleep_replicas failed")

    # =========================================================================
    # Async hook methods - leverage RayPPOTrainer utilities where possible
    # =========================================================================

    async def on_train_start(self, trainer_state: TrainerState) -> None:
        """Called at the start of training."""
        self.global_steps = trainer_state.global_step
        self.global_steps = load_checkpoint(
            self.config,
            self.actor_rollout_wg,
            train_dataloader=trainer_state.train_dataloader,
            lsd_teacher_wg=self.ref_policy_wg,
            load_lsd_teacher=self._lsd_teacher_is_ema(),
        )
        await _checkpoint_update_weights_compat(self.checkpoint_manager, self.global_steps)
        # we need to set trainer's global_steps to sync with the loaded checkpoint
        trainer_state.global_step = self.global_steps
        trainer_state.epoch = trainer_state.train_dataloader.epoch if trainer_state.train_dataloader is not None else 0
        if self.global_steps > 0:
            self._pending_ema_init_metrics = {}

    async def on_batch_start(self, trainer_state: TrainerState) -> None:
        """Called at the start of each batch."""
        self.global_steps = trainer_state.global_step
        if self._pending_ema_init_metrics:
            trainer_state.metrics.update(self._pending_ema_init_metrics)
            self._pending_ema_init_metrics = {}
        # Start profiling if configured
        do_profile = trainer_state.is_training and trainer_state.global_step in self.config.trainer.profile_steps if self.config.trainer.get("profile_steps") is not None else False
        if do_profile:
            with simple_timer("start_profile", trainer_state.timing_dict):
                start_profiling(self.global_steps, self.actor_rollout_wg, self.ref_policy_wg, self.use_reference_policy)

    async def on_batch_end(self, trainer_state: TrainerState) -> None:
        """Called at the end of each batch."""
        # Stop profiling
        do_profile = trainer_state.is_training and trainer_state.global_step in self.config.trainer.profile_steps if self.config.trainer.get("profile_steps") is not None else False
        if do_profile:
            with simple_timer("stop_profile", trainer_state.timing_dict):
                stop_profiling(self.actor_rollout_wg, self.ref_policy_wg, self.use_reference_policy)

        # Save checkpoint if configured
        if self.config.trainer.save_freq > 0 and trainer_state.global_step % self.config.trainer.save_freq == 0:
            with simple_timer("save_checkpoint", trainer_state.timing_dict):
                save_checkpoint(
                    self.config,
                    self.global_steps,
                    self.actor_rollout_wg,
                    train_dataloader=trainer_state.train_dataloader,
                    lsd_teacher_wg=self.ref_policy_wg,
                    save_lsd_teacher=self._lsd_teacher_is_ema(),
                )

        # Weight synchronization (colocated only — separated mode syncs in on_policy_updated)
        if not self.is_separated:
            with simple_timer("update_weights", trainer_state.timing_dict):
                await _checkpoint_update_weights_compat(self.checkpoint_manager, trainer_state.global_step)

        # Update metrics
        if trainer_state.has_backend_batch:
            batch: DataProto = trainer_state.backend_batch  # type: ignore[attr-defined]
            metrics = trainer_state.metrics
            metrics.update({"training/global_step": trainer_state.global_step, "training/epoch": trainer_state.epoch})
            metrics.update(compute_data_metrics(batch=batch, use_critic=False))
            metrics.update(compute_timing_metrics(batch=batch, timing_raw=trainer_state.timing_dict))

            n_gpus = self.resource_pool_manager.get_n_gpus()
            metrics.update(compute_throughout_metrics(batch=batch, timing_raw=trainer_state.timing_dict, n_gpus=n_gpus))

    async def on_policy_updated(self, trainer_state: TrainerState) -> None:
        """Weight-sync hook for separated mode (called by the async pipeline)."""
        if self.is_separated and self.checkpoint_manager is not None:
            with simple_timer("weight_sync", trainer_state.timing_dict):
                await _checkpoint_update_weights_compat(self.checkpoint_manager, trainer_state.weight_version)

    async def on_validation_start(self, trainer_state: TrainerState) -> bool:
        """Called at the start of validation."""
        trainer_state.is_training = False
        self.rollout_engine.is_validation = True
        return True

    async def on_validation_end(self, trainer_state: TrainerState) -> None:
        """Called at the end of validation."""
        trainer_state.is_training = True
        self.rollout_engine.is_validation = False
