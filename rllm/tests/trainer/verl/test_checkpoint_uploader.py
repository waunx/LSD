import json
import subprocess

from omegaconf import OmegaConf

from rllm.trainer.verl import checkpoint_uploader
from rllm.trainer.verl import utils as verl_utils


def test_checkpoint_upload_launches_detached_process(monkeypatch, tmp_path):
    local_path = tmp_path / "global_step_25"
    local_path.mkdir()
    tracker_path = tmp_path / "latest_checkpointed_iteration.txt"
    tracker_path.write_text("25", encoding="utf-8")
    remote_path = "hdfs://cluster/checkpoints/run/global_step_25"
    tracker_remote_path = "hdfs://cluster/checkpoints/run/latest_checkpointed_iteration.txt"
    captured = {}

    class FakePopen:
        def __init__(self, command, **kwargs):
            captured["command"] = command
            captured["kwargs"] = kwargs
            self.pid = 12345

    monkeypatch.setattr(verl_utils.shutil, "which", lambda name: "/usr/bin/hdfs")
    monkeypatch.setattr(verl_utils.subprocess, "Popen", FakePopen)

    pid = verl_utils._launch_checkpoint_upload_to_hdfs(
        str(local_path),
        remote_path,
        str(tracker_path),
        tracker_remote_path,
        max_local_ckpt_to_keep=1,
    )

    assert pid == 12345
    command = captured["command"]
    assert command[0] == verl_utils.sys.executable
    assert command[1].endswith("checkpoint_uploader.py")
    assert command[command.index("--local-path") + 1] == str(local_path)
    assert command[command.index("--remote-path") + 1] == remote_path
    assert command[command.index("--max-local-ckpt-to-keep") + 1] == "1"
    kwargs = captured["kwargs"]
    assert kwargs["start_new_session"] is True
    assert kwargs["stdin"] is subprocess.DEVNULL
    assert kwargs["stderr"] is subprocess.STDOUT
    assert (tmp_path / ".hdfs_upload_global_step_25.log").exists()


def test_checkpoint_uploader_updates_tracker_after_verified_directory(monkeypatch, tmp_path):
    local_path = tmp_path / "global_step_25"
    local_path.mkdir()
    (local_path / "data.pt").write_bytes(b"checkpoint")
    tracker_path = tmp_path / "latest_checkpointed_iteration.txt"
    tracker_path.write_text("25", encoding="utf-8")
    status_path = tmp_path / "upload.json"
    calls = []

    def fake_run(command, *, check, **kwargs):
        calls.append((command, check))
        stdout = "24" if "-cat" in command else ""
        return subprocess.CompletedProcess(command, 0, stdout=stdout)

    monkeypatch.setattr(checkpoint_uploader.subprocess, "run", fake_run)
    checkpoint_uploader.upload_checkpoint(
        hdfs_bin="/usr/bin/hdfs",
        local_path=str(local_path),
        remote_path="hdfs://cluster/checkpoints/run/global_step_25",
        tracker_local_path=str(tracker_path),
        tracker_remote_path="hdfs://cluster/checkpoints/run/latest_checkpointed_iteration.txt",
        status_path=str(status_path),
    )

    assert json.loads(status_path.read_text(encoding="utf-8"))["status"] == "complete"
    assert calls == [
        (["/usr/bin/hdfs", "dfs", "-mkdir", "-p", "hdfs://cluster/checkpoints/run"], True),
        (["/usr/bin/hdfs", "dfs", "-rm", "-r", "-f", "hdfs://cluster/checkpoints/run/global_step_25"], False),
        (["/usr/bin/hdfs", "dfs", "-put", "-f", str(local_path), "hdfs://cluster/checkpoints/run"], True),
        (["/usr/bin/hdfs", "dfs", "-test", "-e", "hdfs://cluster/checkpoints/run/global_step_25"], True),
        (["/usr/bin/hdfs", "dfs", "-mkdir", "-p", "hdfs://cluster/checkpoints/run"], True),
        (["/usr/bin/hdfs", "dfs", "-cat", "hdfs://cluster/checkpoints/run/latest_checkpointed_iteration.txt"], False),
        (["/usr/bin/hdfs", "dfs", "-put", "-f", f"{status_path}.tracker", "hdfs://cluster/checkpoints/run/latest_checkpointed_iteration.txt"], True),
    ]


def test_prune_only_removes_uploaded_old_checkpoints(tmp_path):
    old_complete = tmp_path / "global_step_25"
    old_pending = tmp_path / "global_step_50"
    latest = tmp_path / "global_step_75"
    for path in (old_complete, old_pending, latest):
        path.mkdir()
    (tmp_path / ".hdfs_upload_global_step_25.json").write_text('{"status":"complete"}', encoding="utf-8")
    (tmp_path / ".hdfs_upload_global_step_50.json").write_text('{"status":"running"}', encoding="utf-8")

    checkpoint_uploader._prune_local_checkpoints(str(tmp_path), 1)
    assert not old_complete.exists()
    assert old_pending.exists()
    assert latest.exists()


def test_save_checkpoint_defers_rotation_until_hdfs_upload(monkeypatch, tmp_path):
    actor_calls = []
    teacher_calls = []
    upload_calls = []

    class ActorWorker:
        def save_checkpoint(self, *args, **kwargs):
            actor_calls.append((args, kwargs))

    class TeacherWorker:
        def save_lsd_teacher_checkpoint(self, *args, **kwargs):
            teacher_calls.append((args, kwargs))

    config = OmegaConf.create(
        {
            "trainer": {
                "default_local_dir": str(tmp_path),
                "default_hdfs_dir": "hdfs://cluster/checkpoints/run",
                "max_actor_ckpt_to_keep": 1,
            },
            "actor_rollout_ref": {
                "actor": {"checkpoint": {"async_save": False}},
            },
        }
    )

    def fake_upload(*args, **kwargs):
        upload_calls.append((args, kwargs))
        return 12345

    monkeypatch.setattr(
        verl_utils,
        "_launch_checkpoint_upload_to_hdfs",
        fake_upload,
    )
    verl_utils.save_checkpoint(
        config,
        25,
        ActorWorker(),
        lsd_teacher_wg=TeacherWorker(),
        save_lsd_teacher=True,
    )

    assert actor_calls[0][1]["max_ckpt_to_keep"] is None
    assert teacher_calls[0][1]["max_ckpt_to_keep"] is None
    assert upload_calls[0][1]["max_local_ckpt_to_keep"] == 1
