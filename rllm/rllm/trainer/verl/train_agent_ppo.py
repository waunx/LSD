"""
Copyright under Agentica Project.

Note that we don't combine the main with ray_trainer as ray_trainer is used by other main.
"""

import os
import socket
import json
import urllib.request

import hydra
import ray
from omegaconf import OmegaConf
from verl.trainer.ppo.ray_trainer import Role
from verl.trainer.ppo.reward import load_reward_manager
from verl.trainer.ppo.utils import need_critic, need_reference_policy
from verl.single_controller.ray import RayWorkerGroup


# #region debug-point A:python-report
def _debug_report(hypothesis_id: str, location: str, msg: str, data: dict | None = None) -> None:
    try:
        env_path = "/opt/tiger/rllm/.dbg/deepresearch-lsd-run.env"
        url = "http://127.0.0.1:7777/event"
        session_id = "deepresearch-lsd-run"
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
        payload = {
            "sessionId": session_id,
            "runId": os.environ.get("TRAE_DEBUG_RUN_ID", "pre-fix"),
            "hypothesisId": hypothesis_id,
            "location": location,
            "msg": f"[DEBUG] {msg}",
            "data": data or {},
        }
        urllib.request.urlopen(
            urllib.request.Request(
                url,
                data=json.dumps(payload, default=str).encode(),
                headers={"Content-Type": "application/json"},
            ),
            timeout=2,
        ).read()
    except Exception:
        pass
# #endregion


class LocalRankCudaVisibleRayWorkerGroup(RayWorkerGroup):
    def _create_worker(self, rank, pg_idx, pg, local_rank, resource_pool, ray_cls_with_init, worker_env, detached):
        # #region debug-point C:worker-create-entry
        _debug_report(
            "C",
            "train_agent_ppo.py:LocalRankCudaVisibleRayWorkerGroup._create_worker",
            "worker create entry",
            {
                "rank": rank,
                "pg_idx": pg_idx,
                "local_rank": local_rank,
                "max_colocate_count": getattr(resource_pool, "max_colocate_count", None),
                "resource_store": getattr(resource_pool, "store", None),
                "incoming_worker_env_cuda": (worker_env or {}).get("CUDA_VISIBLE_DEVICES") if isinstance(worker_env, dict) else None,
                "incoming_worker_env_nvidia": (worker_env or {}).get("NVIDIA_VISIBLE_DEVICES") if isinstance(worker_env, dict) else None,
                "host_cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
                "host_nvidia_visible_devices": os.environ.get("NVIDIA_VISIBLE_DEVICES"),
                "ray_no_set_cuda": os.environ.get("RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES"),
            },
        )
        # #endregion
        # #region debug-point C:worker-create-final-env
        _debug_report(
            "C",
            "train_agent_ppo.py:LocalRankCudaVisibleRayWorkerGroup._create_worker",
            "worker create final env",
            {
                "rank": rank,
                "local_rank": local_rank,
                "final_worker_env_cuda": (worker_env or {}).get("CUDA_VISIBLE_DEVICES") if isinstance(worker_env, dict) else None,
                "final_worker_env_nvidia": (worker_env or {}).get("NVIDIA_VISIBLE_DEVICES") if isinstance(worker_env, dict) else None,
                "device_name": self.device_name,
                "manual_cuda_visible_devices_injection": False,
            },
        )
        # #endregion
        return super()._create_worker(rank, pg_idx, pg, local_rank, resource_pool, ray_cls_with_init, worker_env, detached)

from verl.utils.config import validate_config
from verl.utils.device import is_cuda_available

from rllm.trainer.verl.agent_workflow_trainer import AgentWorkflowPPOTrainer
from rllm.trainer.verl.ray_runtime_env import get_ppo_ray_runtime_env


@hydra.main(config_path="../config", config_name="agent_ppo_trainer", version_base=None)
def main(config):
    run_ppo_agent(config)


def run_ppo_agent(config):
    # Check if Ray is not initialized
    if not ray.is_initialized():
        from rllm.trainer.ray_init_utils import get_ray_init_settings

        ray_init_settings = get_ray_init_settings(config)
        ray.init(runtime_env=get_ppo_ray_runtime_env(), **ray_init_settings)

    runner_cls = ray.remote(num_cpus=1)(TaskRunner)

    # Create a remote instance of the TaskRunner class, and
    # Execute the `run` method of the TaskRunner instance remotely and wait for it to complete
    if is_cuda_available and config.trainer.get("profile_steps") is not None and len(config.trainer.get("profile_steps", [])) > 0:
        nsight_options = OmegaConf.to_container(config.trainer.controller_nsight_options)
        runner = runner_cls.options(runtime_env={"nsight": nsight_options}).remote()
    else:
        runner = runner_cls.remote()
    ray.get(runner.run.remote(config))

    # [Optional] get the path of the timeline trace file from the configuration, default to None
    # This file is used for performance analysis
    timeline_json_file = config.ray_init.get("timeline_json_file", None)
    if timeline_json_file:
        ray.timeline(filename=timeline_json_file)


class TaskRunner:
    """Ray remote class for executing distributed PPO training tasks.

    This class encapsulates the main training logic and runs as a Ray remote actor
    to enable distributed execution across multiple nodes and GPUs.

    Attributes:
        role_worker_mapping: Dictionary mapping Role enums to Ray remote worker classes
        mapping: Dictionary mapping Role enums to resource pool IDs for GPU allocation
    """

    def __init__(self):
        self.role_worker_mapping = {}
        self.mapping = {}

    def add_actor_rollout_worker(self, config):
        """Add actor rollout worker based on the actor strategy."""
        from verl.workers.engine_workers import ActorRolloutRefWorker

        from rllm.trainer.verl.patch import patch_verl_lsd_worker_hooks

        # Patch before ray.remote(...) so the Ray actor proxy exposes LSD RPCs
        # such as set_ref_loss_fn and lsd_update_ema_teacher. The worker setup
        # hook applies the same patch again inside each worker interpreter.
        patch_verl_lsd_worker_hooks()

        actor_rollout_cls = ActorRolloutRefWorker
        ray_worker_group_cls = LocalRankCudaVisibleRayWorkerGroup

        lora_rank = config.actor_rollout_ref.model.get("lora", {}).get("rank", 0)
        if lora_rank <= 0:
            lora_rank = config.actor_rollout_ref.model.get("lora_rank", 0)
        needs_lsd_ref = bool(config.rllm.get("lsd", {}).get("enable", False))
        ref_in_actor = (lora_rank > 0 or config.actor_rollout_ref.model.get("lora_adapter_path") is not None) and not needs_lsd_ref
        if (need_reference_policy(config) or needs_lsd_ref) and not ref_in_actor:
            role = Role.ActorRolloutRef
        else:
            role = Role.ActorRollout
        self.role_worker_mapping[role] = ray.remote(actor_rollout_cls)
        self.mapping[role] = "global_pool"
        return actor_rollout_cls, ray_worker_group_cls

    def add_critic_worker(self, config):
        """Add critic worker to role mapping."""
        from verl.workers.engine_workers import TrainingWorker

        self.role_worker_mapping[Role.Critic] = ray.remote(TrainingWorker)
        self.mapping[Role.Critic] = "global_pool"

    def add_ref_policy_worker(self, config, ref_policy_cls):
        """Add reference policy worker if KL loss or KL reward is used.

        Note: Ref policy is fused into ActorRolloutRefWorker in the unified engine,
        so this is a no-op.
        """
        return

    def init_resource_pool_mgr(self, config):
        """Initialize resource pool manager."""

        global_pool_id = "global_pool"
        resource_pool_spec = {
            global_pool_id: [config.trainer.n_gpus_per_node] * config.trainer.nnodes,
        }
        # TODO Here you can use the new registration method to support dynamic registration of roles
        reward_model_cfg = OmegaConf.select(config, "reward_model")
        if reward_model_cfg is not None and reward_model_cfg.get("enable_resource_pool", False):
            if reward_model_cfg.n_gpus_per_node <= 0:
                raise ValueError("config.reward_model.n_gpus_per_node must be greater than 0")
            if reward_model_cfg.nnodes <= 0:
                raise ValueError("config.reward_model.nnodes must be greater than 0")

            reward_pool = [reward_model_cfg.n_gpus_per_node] * reward_model_cfg.nnodes
            resource_pool_spec["reward_pool"] = reward_pool

        from verl.trainer.ppo.ray_trainer import ResourcePoolManager

        resource_pool_manager = ResourcePoolManager(resource_pool_spec=resource_pool_spec, mapping=self.mapping)
        create_resource_pool = resource_pool_manager.create_resource_pool

        def create_resource_pool_without_colocation():
            # #region debug-point B:resource-pool-before
            _debug_report(
                "B",
                "train_agent_ppo.py:init_resource_pool_mgr.create_resource_pool",
                "resource pool before create",
                {
                    "resource_pool_spec": resource_pool_spec,
                    "mapping": {str(k): v for k, v in self.mapping.items()},
                    "trainer_n_gpus_per_node": config.trainer.n_gpus_per_node,
                    "trainer_nnodes": config.trainer.nnodes,
                },
            )
            # #endregion
            create_resource_pool()
            # #region debug-point B:resource-pool-created
            _debug_report(
                "B",
                "train_agent_ppo.py:init_resource_pool_mgr.create_resource_pool",
                "resource pool created",
                {
                    name: {
                        "store": getattr(pool, "store", None),
                        "world_size": getattr(pool, "world_size", None),
                        "max_colocate_count": getattr(pool, "max_colocate_count", None),
                        "use_gpu": getattr(pool, "use_gpu", None),
                    }
                    for name, pool in resource_pool_manager.resource_pool_dict.items()
                },
            )
            # #endregion
            for resource_pool in resource_pool_manager.resource_pool_dict.values():
                resource_pool.max_colocate_count = 1
            # #region debug-point B:resource-pool-after
            _debug_report(
                "B",
                "train_agent_ppo.py:init_resource_pool_mgr.create_resource_pool",
                "resource pool after no-colocation override",
                {
                    name: {
                        "store": getattr(pool, "store", None),
                        "world_size": getattr(pool, "world_size", None),
                        "max_colocate_count": getattr(pool, "max_colocate_count", None),
                    }
                    for name, pool in resource_pool_manager.resource_pool_dict.items()
                },
            )
            # #endregion

        resource_pool_manager.create_resource_pool = create_resource_pool_without_colocation
        return resource_pool_manager

    def run(
        self,
        config,
        workflow_class=None,
        workflow_args=None,
    ):
        """Execute the main PPO training workflow.

        This method sets up the distributed training environment, initializes
        workers, datasets, and reward functions, then starts the training process.

        Args:
            config: Training configuration object containing all parameters needed
                   for setting up and running the PPO training process.
        """
        # Print the initial configuration. `resolve=True` will evaluate symbolic values.
        from pprint import pprint

        from omegaconf import OmegaConf
        from verl.utils.fs import copy_to_local

        print(f"TaskRunner hostname: {socket.gethostname()}, PID: {os.getpid()}")
        pprint(OmegaConf.to_container(config))
        OmegaConf.register_new_resolver("mul", lambda x, y: int(x) * int(y))
        OmegaConf.resolve(config)

        actor_rollout_cls, ray_worker_group_cls = self.add_actor_rollout_worker(config)
        self.add_critic_worker(config)
        # TODO(listar2000): in rLLM, we skip the reward worker setup and assume a simple verified reward.
        # Add a reference policy worker if KL loss or KL reward is used.
        self.add_ref_policy_worker(config, actor_rollout_cls)

        validate_config(
            config=config,
            use_reference_policy=need_reference_policy(config) or bool(config.rllm.get("lsd", {}).get("enable", False)),
            use_critic=need_critic(config),
        )

        # Download the checkpoint from HDFS to the local machine.
        # `use_shm` determines whether to use shared memory, which could lead to faster model loading if turned on
        local_path = copy_to_local(
            config.actor_rollout_ref.model.path,
            use_shm=config.actor_rollout_ref.model.get("use_shm", False),
        )

        # Instantiate the tokenizer and processor.
        from verl.utils import hf_processor, hf_tokenizer

        trust_remote_code = config.data.get("trust_remote_code", False)
        tokenizer = hf_tokenizer(local_path, trust_remote_code=trust_remote_code)
        # Used for multimodal LLM, could be None
        processor = hf_processor(local_path, trust_remote_code=trust_remote_code, use_fast=True)

        # Load the reward manager for training and validation.
        reward_model_cfg = OmegaConf.select(config, "reward_model") or {}
        reward_fn = load_reward_manager(
            config,
            tokenizer,
            **reward_model_cfg.get("reward_kwargs", {}),
        )
        val_reward_fn = load_reward_manager(
            config,
            tokenizer,
            **reward_model_cfg.get("reward_kwargs", {}),
        )

        resource_pool_manager = self.init_resource_pool_mgr(config)

        if workflow_class is not None:
            workflow_args = workflow_args or {}
            if config.rllm.workflow.get("workflow_args") is not None:
                for key, value in config.rllm.workflow.get("workflow_args").items():
                    if value is not None:
                        if key in workflow_args and isinstance(workflow_args[key], dict):
                            workflow_args[key].update(value)
                        else:
                            workflow_args[key] = value

            trainer = AgentWorkflowPPOTrainer(
                config=config,
                tokenizer=tokenizer,
                processor=processor,
                role_worker_mapping=self.role_worker_mapping,
                resource_pool_manager=resource_pool_manager,
                ray_worker_group_cls=ray_worker_group_cls,
                reward_fn=reward_fn,
                val_reward_fn=val_reward_fn,
                workflow_class=workflow_class,
                workflow_args=workflow_args,
            )

        else:
            raise ValueError("TaskRunner.run requires workflow_class. The legacy agent_class + env_class and agent_run_func paths have been removed; port your agent to a Workflow or AgentFlow.")

        trainer.init_workers()
        try:
            trainer.fit_agent()
        finally:
            trainer.shutdown()


if __name__ == "__main__":
    main()
