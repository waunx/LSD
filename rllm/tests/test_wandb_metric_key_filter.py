from rllm.utils.tracking import filter_fully_async_compatible_wandb_metrics


def test_fully_async_wandb_filter_keeps_target_and_lsd_keys_only():
    metrics = {
        "actor/lr": 1e-6,
        "custom/raw_reward/avg": 0.5,
        "val/raw_reward": 0.5,
        "val/extracted_answer": 2024.0,
        "val/browsecomp-plus/pass@1": 0.5,
        "rollout_corr/kl": 0.1,
        "offpolicy/kl": 0.1,
        "batch/accuracy": 0.5,
        "training/global_step": 1,
        "lsd/opd_pg_loss": 0.2,
    }

    filtered = filter_fully_async_compatible_wandb_metrics(metrics)

    assert filtered == {
        "actor/lr": 1e-6,
        "custom/raw_reward/avg": 0.5,
        "val/raw_reward": 0.5,
        "rollout_corr/kl": 0.1,
        "lsd/opd_pg_loss": 0.2,
    }
