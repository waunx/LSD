import torch

from rllm.trainer.verl.verl_backend import CustomPPOLoss


def _loss_helper():
    helper = object.__new__(CustomPPOLoss)
    return helper


def _global_denominators(response_mask, easy_seq):
    hard_mask = response_mask & (~easy_seq).unsqueeze(-1)
    easy_mask = response_mask & easy_seq.unsqueeze(-1)
    token_count = response_mask.sum(dim=-1)
    seq_mask = token_count > 0
    return {
        "lsd_global_hard_token_count": float(hard_mask.sum().item()),
        "lsd_global_easy_token_count": float(easy_mask.sum().item()),
        "lsd_global_hard_seq_count": float((seq_mask & (~easy_seq)).sum().item()),
        "lsd_global_easy_seq_count": float((seq_mask & easy_seq).sum().item()),
    }


def _branch_losses(w, features, response_mask, easy_seq, partitions):
    helper = _loss_helper()
    global_denoms = _global_denominators(response_mask, easy_seq)
    dp_size = len(partitions)
    hard_parts = []
    easy_parts = []
    for idxs in partitions:
        loss_mat = w * features[idxs]
        local_response_mask = response_mask[idxs]
        local_easy_seq = easy_seq[idxs]
        hard_mask = local_response_mask & (~local_easy_seq).unsqueeze(-1)
        easy_mask = local_response_mask & local_easy_seq.unsqueeze(-1)
        hard_parts.append(
            helper._aggregate_lsd_loss(
                loss_mat=loss_mat,
                loss_mask=hard_mask,
                mode="seq_mean_token_mean",
                log_prob=loss_mat,
                global_denominators=global_denoms,
                denominator_key="lsd_global_hard_seq_count",
                dp_size=dp_size,
            )
        )
        easy_parts.append(
            helper._aggregate_lsd_loss(
                loss_mat=loss_mat,
                loss_mask=easy_mask,
                mode="seq_mean_token_mean",
                log_prob=loss_mat,
                global_denominators=global_denoms,
                denominator_key="lsd_global_easy_seq_count",
                dp_size=dp_size,
            )
        )
    return sum(hard_parts) / dp_size, sum(easy_parts) / dp_size


def test_lsd_masked_mean_is_invariant_to_rank_partition_and_microbatching():
    features = torch.tensor(
        [
            [1.0, 2.0, 0.0],
            [3.0, 0.0, 0.0],
            [4.0, 5.0, 6.0],
            [7.0, 8.0, 0.0],
        ]
    )
    response_mask = torch.tensor(
        [
            [True, True, False],
            [True, False, False],
            [True, True, True],
            [True, True, False],
        ]
    )
    easy_seq = torch.tensor([False, True, False, True])

    full_w = torch.tensor(0.5, requires_grad=True)
    full_hard, full_easy = _branch_losses(full_w, features, response_mask, easy_seq, [torch.arange(4)])
    full_total = full_hard + 0.7 * full_easy
    full_total.backward()

    for partitions in (
        [torch.tensor([0, 1]), torch.tensor([2, 3])],
        [torch.tensor([0, 3]), torch.tensor([1, 2])],
        [torch.tensor([0]), torch.tensor([1]), torch.tensor([2]), torch.tensor([3])],
    ):
        split_w = torch.tensor(0.5, requires_grad=True)
        split_hard, split_easy = _branch_losses(split_w, features, response_mask, easy_seq, partitions)
        split_total = split_hard + 0.7 * split_easy
        split_total.backward()

        assert torch.allclose(split_hard.detach(), full_hard.detach(), atol=1e-6)
        assert torch.allclose(split_easy.detach(), full_easy.detach(), atol=1e-6)
        assert torch.allclose(split_w.grad, full_w.grad, atol=1e-6)


def test_lsd_masked_mean_all_easy_or_all_hard_has_no_nan():
    helper = _loss_helper()
    loss_mat = torch.ones(2, 3, requires_grad=True)
    response_mask = torch.tensor([[True, True, False], [True, False, False]])
    empty_mask = torch.zeros_like(response_mask, dtype=torch.bool)
    denoms = {
        "lsd_global_hard_seq_count": 0.0,
        "lsd_global_easy_seq_count": 2.0,
        "lsd_global_hard_token_count": 0.0,
        "lsd_global_easy_token_count": 3.0,
    }

    zero_hard = helper._aggregate_lsd_loss(
        loss_mat,
        empty_mask,
        "seq_mean_token_mean",
        loss_mat,
        global_denominators=denoms,
        denominator_key="lsd_global_hard_seq_count",
        dp_size=1,
    )
    easy = helper._aggregate_lsd_loss(
        loss_mat,
        response_mask,
        "seq_mean_token_mean",
        loss_mat,
        global_denominators=denoms,
        denominator_key="lsd_global_easy_seq_count",
        dp_size=1,
    )

    total = zero_hard + easy
    total.backward()
    assert torch.isfinite(total)
    assert torch.isfinite(loss_mat.grad).all()


def test_lsd_token_mass_scale_uses_global_easy_over_hard_tokens():
    helper = _loss_helper()
    log_prob = torch.ones(())
    easy_tokens = torch.tensor(999.0)
    hard_tokens = torch.tensor(999.0)
    denoms = {
        "lsd_global_hard_token_count": 8.0,
        "lsd_global_easy_token_count": 2.0,
    }

    scale = helper._lsd_token_mass_scale(easy_tokens, hard_tokens, log_prob, denoms)

    assert torch.allclose(scale, torch.tensor(0.25))


def test_lsd_token_mass_scale_handles_empty_partitions():
    helper = _loss_helper()
    log_prob = torch.ones(())

    no_easy = helper._lsd_token_mass_scale(
        easy_tokens=torch.tensor(0.0),
        hard_tokens=torch.tensor(8.0),
        log_prob=log_prob,
        global_denominators=None,
    )
    no_hard = helper._lsd_token_mass_scale(
        easy_tokens=torch.tensor(8.0),
        hard_tokens=torch.tensor(0.0),
        log_prob=log_prob,
        global_denominators=None,
    )

    assert torch.allclose(no_easy, torch.tensor(0.0))
    assert torch.allclose(no_hard, torch.tensor(1.0))


def test_lsd_worker_hook_injects_optimizer_minibatch_denominators():
    from tensordict import TensorDict
    from verl.utils import tensordict_utils as tu
    from verl.workers.engine_workers import TrainingWorker

    from rllm.trainer.verl.patch import patch_verl_lsd_worker_hooks

    patch_verl_lsd_worker_hooks()

    class FakeEngine:
        def get_data_parallel_group(self):
            return None

    worker = object.__new__(TrainingWorker)
    worker.engine = FakeEngine()
    batch = TensorDict(
        {
            "response_mask": torch.tensor([[1, 1, 0], [1, 0, 0], [1, 1, 1]], dtype=torch.bool),
            "lsd_is_easy": torch.tensor([False, True, True], dtype=torch.bool),
        },
        batch_size=[3],
    )

    worker._assign_lsd_global_counts(batch)

    assert tu.get(batch, "lsd_global_hard_token_count") == 2.0
    assert tu.get(batch, "lsd_global_easy_token_count") == 4.0
    assert tu.get(batch, "lsd_global_hard_seq_count") == 1.0
    assert tu.get(batch, "lsd_global_easy_seq_count") == 2.0


