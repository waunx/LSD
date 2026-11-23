"""LSD routing helpers."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from omegaconf import DictConfig, OmegaConf

logger = logging.getLogger(__name__)


@dataclass
class LSDConfig:
    enable: bool = False
    routing_mode: str = "fixed_base"
    mapping_path: str | None = None
    easy_threshold: float = 0.875
    online_group_threshold: float = 0.875
    expected_group_size: int = 8
    teacher_mode: str = "fixed_base"
    ema_half_life_updates: float = 50.0
    ema_update_interval: int = 1
    ema_accum_dtype: str = "fp32"
    lsd_coef: float = 1.0
    easy_rl_coef: float = 0.0
    loss_agg_mode: str = "seq_mean_token_mean"
    opd_loss_mode: str = "topk_forward_kl"
    opd_optimization_mode: str = "supervised_gradient"
    opd_topk: int = 128
    opd_stop_token_ids: list[int] | None = None
    opd_direct_loss_clip: float | None = None
    opd_pg_adv_clip: float | None = 10.0
    default_base_length: float = 0.0
    fail_on_missing: bool = False

    @classmethod
    def from_config(cls, config: DictConfig | dict | None) -> "LSDConfig":
        if config is None:
            return cls()
        data = OmegaConf.to_container(config, resolve=True) if OmegaConf.is_config(config) else dict(config)
        lsd_config = cls(
            enable=bool(data.get("enable", False)),
            routing_mode=str(data.get("routing_mode", "fixed_base")).replace("-", "_"),
            mapping_path=data.get("mapping_path"),
            easy_threshold=float(data.get("easy_threshold", 0.875)),
            online_group_threshold=float(data.get("online_group_threshold", data.get("easy_threshold", 0.875))),
            expected_group_size=int(data.get("expected_group_size", 8)),
            teacher_mode=str(data.get("teacher_mode", "fixed_base")).replace("-", "_"),
            ema_half_life_updates=float(data.get("ema_half_life_updates", 50.0)),
            ema_update_interval=int(data.get("ema_update_interval", 1)),
            ema_accum_dtype=str(data.get("ema_accum_dtype", "fp32")).lower(),
            lsd_coef=float(data.get("lsd_coef", 1.0)),
            easy_rl_coef=float(data.get("easy_rl_coef", 0.0)),
            loss_agg_mode=str(data.get("loss_agg_mode", "seq_mean_token_mean")),
            opd_loss_mode=str(data.get("opd_loss_mode", "topk_forward_kl")),
            opd_optimization_mode=str(data.get("opd_optimization_mode", "supervised_gradient")),
            opd_topk=int(data.get("opd_topk", 128)),
            opd_stop_token_ids=(
                None
                if data.get("opd_stop_token_ids") is None
                else [int(x) for x in data.get("opd_stop_token_ids", [])]
            ),
            opd_direct_loss_clip=(
                None
                if data.get("opd_direct_loss_clip", None) is None
                else float(data.get("opd_direct_loss_clip"))
            ),
            opd_pg_adv_clip=(None if data.get("opd_pg_adv_clip", 10.0) is None else float(data.get("opd_pg_adv_clip", 10.0))),
            default_base_length=float(data.get("default_base_length", 0.0) or 0.0),
            fail_on_missing=bool(data.get("fail_on_missing", False)),
        )
        lsd_config.validate()
        return lsd_config

    def validate(self) -> None:
        if not self.enable:
            return

        if self.routing_mode not in {"fixed_base", "online_group"}:
            raise ValueError("rllm.lsd.routing_mode must be fixed_base or online_group")
        if self.routing_mode == "fixed_base" and not self.mapping_path:
            raise ValueError("rllm.lsd.routing_mode=fixed_base requires rllm.lsd.mapping_path")
        if self.routing_mode == "online_group" and self.mapping_path:
            logger.warning("rllm.lsd.routing_mode=online_group ignores rllm.lsd.mapping_path")
        if self.expected_group_size <= 0:
            raise ValueError("rllm.lsd.expected_group_size must be positive")

        if self.teacher_mode not in {"fixed_base", "ema"}:
            raise ValueError("rllm.lsd.teacher_mode must be fixed_base or ema")
        if self.ema_update_interval <= 0:
            raise ValueError("rllm.lsd.ema_update_interval must be positive")
        if self.ema_half_life_updates <= 0:
            raise ValueError("rllm.lsd.ema_half_life_updates must be positive")
        if self.ema_accum_dtype not in {"fp32", "bf16", "fp16"}:
            raise ValueError("rllm.lsd.ema_accum_dtype must be fp32, bf16, or fp16")

        loss_mode = self.normalized_opd_loss_mode()
        opt_mode = self.normalized_opd_optimization_mode()
        if (loss_mode, opt_mode) not in {
            ("topk_forward_kl", "supervised_gradient"),
            ("sample_token", "policy_gradient"),
        }:
            raise ValueError(
                "Invalid LSD OPD config. Only topk_forward_kl+supervised_gradient "
                "and sample_token+policy_gradient are supported."
            )

    def normalized_opd_loss_mode(self) -> str:
        mode = str(self.opd_loss_mode).replace("-", "_")
        aliases = {
            "topk": "topk_forward_kl",
            "forward_kl_topk": "topk_forward_kl",
            "sample": "sample_token",
            "sampled_token": "sample_token",
        }
        mode = aliases.get(mode, mode)
        if mode not in {"topk_forward_kl", "sample_token"}:
            raise ValueError("rllm.lsd.opd_loss_mode must be topk_forward_kl or sample_token")
        return mode

    def normalized_opd_optimization_mode(self) -> str:
        mode = str(self.opd_optimization_mode).replace("-", "_")
        aliases = {
            "supervised": "supervised_gradient",
            "direct": "supervised_gradient",
            "sg": "supervised_gradient",
            "pg": "policy_gradient",
            "policy": "policy_gradient",
        }
        mode = aliases.get(mode, mode)
        if mode not in {"supervised_gradient", "policy_gradient"}:
            raise ValueError("rllm.lsd.opd_optimization_mode must be supervised_gradient or policy_gradient")
        return mode


@dataclass(frozen=True)
class LSDRoute:
    is_easy: bool = False
    base_acc: float = 0.0
    base_length: float = 0.0
    found: bool = False


class LSDRouter:
    """Lookup base-model difficulty for a dataset row."""

    def __init__(self, config: LSDConfig):
        self.config = config
        self.by_source_index: dict[int, LSDRoute] = {}
        self.by_id: dict[str, LSDRoute] = {}
        if config.enable and config.routing_mode == "fixed_base":
            self._load_mapping()

    @property
    def enabled(self) -> bool:
        return self.config.enable

    def _load_mapping(self) -> None:
        path = self.config.mapping_path
        if not path:
            raise ValueError("rllm.lsd.enable=true requires rllm.lsd.mapping_path")

        mapping_path = Path(path)
        if not mapping_path.exists():
            raise FileNotFoundError(f"LSD mapping file not found: {mapping_path}")

        n_rows = 0
        with mapping_path.open(encoding="utf-8") as f:
            for line_no, line in enumerate(f, start=1):
                if not line.strip():
                    continue
                record = json.loads(line)
                route = self._route_from_record(record)

                source_index = record.get("source_index")
                if source_index is not None:
                    self.by_source_index[int(source_index)] = route

                extra_info_index = record.get("extra_info_index")
                if extra_info_index is not None:
                    self.by_id[str(extra_info_index)] = route

                n_rows += 1

        logger.info(
            "Loaded LSD mapping rows=%d source_index_keys=%d id_keys=%d easy_threshold=%.4f from %s",
            n_rows,
            len(self.by_source_index),
            len(self.by_id),
            self.config.easy_threshold,
            mapping_path,
        )

    def _route_from_record(self, record: dict[str, Any]) -> LSDRoute:
        sample_count = record.get("sample_count") or record.get("expected_repeat_times") or 0
        acc_count = record.get("acc_count")
        if acc_count is None and isinstance(record.get("scores"), list):
            acc_count = sum(1 for v in record["scores"] if float(v) > 0)
        base_acc = float(acc_count or 0.0) / float(sample_count or 1.0)

        # The current DAPO base-eval mapping does not contain a reliable base
        # mean length.  Keep the field for future files that do include it.
        base_length = (
            record.get("base_mean_length")
            or record.get("base_length")
            or record.get("mean_length")
            or self.config.default_base_length
            or 0.0
        )
        return LSDRoute(
            is_easy=base_acc >= self.config.easy_threshold,
            base_acc=base_acc,
            base_length=float(base_length or 0.0),
            found=True,
        )

    def lookup(self, row: Any, fallback_source_index: int | None = None) -> LSDRoute:
        if not self.enabled:
            return LSDRoute()

        metadata = row.metadata if hasattr(row, "metadata") else row
        if not isinstance(metadata, dict):
            metadata = {}

        source_index = metadata.get("source_index")
        if source_index is None:
            source_index = metadata.get("lsd_source_index")
        if source_index is None:
            source_index = fallback_source_index
        if source_index is not None:
            route = self.by_source_index.get(int(source_index))
            if route is not None:
                return route

        row_id = metadata.get("id") or metadata.get("extra_info_index")
        if row_id is not None:
            route = self.by_id.get(str(row_id))
            if route is not None:
                return route

        if self.config.fail_on_missing:
            raise KeyError(f"LSD mapping missing for row id={row_id!r}, source_index={source_index!r}")
        return LSDRoute()
