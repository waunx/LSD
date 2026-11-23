import pytest
from omegaconf import OmegaConf

from rllm.trainer.verl.utils import load_checkpoint


class _ActorRolloutWorker:
    def load_checkpoint(self, path, del_local_after_load=False):
        raise AssertionError(f"unexpected checkpoint load: {path}")


def _config(tmp_path, resume_mode="auto", resume_from_path=None):
    return OmegaConf.create(
        {
            "trainer": {
                "resume_mode": resume_mode,
                "resume_from_path": resume_from_path,
                "default_local_dir": str(tmp_path),
                "default_hdfs_dir": "hdfs://cluster/checkpoints/experiment",
                "del_local_ckpt_after_load": False,
            }
        }
    )


def test_auto_resume_ignores_hdfs_backup_path(tmp_path):
    config = _config(tmp_path)

    assert load_checkpoint(config, _ActorRolloutWorker()) == 0


def test_explicit_hdfs_resume_path_is_rejected(tmp_path):
    config = _config(
        tmp_path,
        resume_mode="resume_path",
        resume_from_path="hdfs://cluster/checkpoints/experiment/global_step_100",
    )

    with pytest.raises(NotImplementedError, match="load from hdfs"):
        load_checkpoint(config, _ActorRolloutWorker())
