from pathlib import Path

import torch


def test_lsd_topk_patch_registered_for_workers():
    source = Path("rllm/trainer/verl/patch.py").read_text(encoding="utf-8")

    assert "def patch_verl_fsdp_topk_processor_outputs" in source
    assert '"patch_verl_fsdp_topk_processor_outputs": patch_verl_fsdp_topk_processor_outputs' in source
    assert 'engine_cls = mod.FSDPEngineWithLMHead' in source


def test_lsd_topk_patch_accepts_token_by_topk_outputs():
    source = Path("rllm/trainer/verl/patch.py").read_text(encoding="utf-8")

    relaxed_shape_check = "v.dim() >= 2 and v.shape[0] == log_probs.shape[0]"
    assert source.count(relaxed_shape_check) == 2
    assert "torch.nested.nested_tensor_from_jagged(v, cu_seqlens)" in source


def test_lsd_ema_teacher_checkpoint_is_model_only():
    source = Path("rllm/trainer/verl/patch.py").read_text(encoding="utf-8")

    assert 'checkpoint_manager.checkpoint_save_contents = ["model"]' in source
    assert 'checkpoint_manager.checkpoint_load_contents = ["model"]' in source
    assert "self.ref.load_checkpoint(local_path, hdfs_path, del_local_after_load=False)" in source



def test_jagged_concat_preserves_teacher_and_position_ragged_axes():
    from verl.utils import tensordict_utils as tu

    from rllm.trainer.verl import patch

    patch.patch_verl_tensordict_jagged_layout()

    teacher_parts = [
        torch.nested.nested_tensor_from_jagged(
            values=torch.zeros(length, 128),
            offsets=torch.tensor([0, length]),
            jagged_dim=1,
        )
        for length in (761, 1358)
    ]
    teacher = tu.concat_nested_tensors(teacher_parts)
    assert teacher._ragged_idx == 1
    assert [tuple(item.shape) for item in teacher.unbind()] == [
        (761, 128),
        (1358, 128),
    ]

    position_parts = [
        torch.nested.nested_tensor_from_jagged(
            values=torch.zeros(4, length),
            offsets=torch.tensor([0, length]),
            jagged_dim=2,
        )
        for length in (761, 1358)
    ]
    positions = tu.concat_nested_tensors(position_parts)
    assert positions._ragged_idx == 2
    assert [tuple(item.shape) for item in positions.unbind()] == [
        (4, 761),
        (4, 1358),
    ]


def test_attention_padded_teacher_tensor_uses_input_offsets():
    from rllm.trainer.verl.verl_backend import _attention_padded_to_no_padding

    attention_mask = torch.tensor(
        [
            [0, 1, 1, 0],
            [1, 1, 1, 0],
        ],
        dtype=torch.bool,
    )
    input_ids = torch.nested.nested_tensor_from_jagged(
        values=torch.arange(5),
        offsets=torch.tensor([0, 2, 5]),
    )
    teacher = torch.arange(2 * 4 * 3).reshape(2, 4, 3)

    nested = _attention_padded_to_no_padding(
        teacher,
        attention_mask,
        input_ids.offsets(),
    )

    assert nested.is_nested
    assert nested._ragged_idx == 1
    assert nested.offsets() is input_ids.offsets()
    assert [tuple(item.shape) for item in nested.unbind()] == [(2, 3), (3, 3)]
    assert torch.equal(nested.values(), teacher[attention_mask])
