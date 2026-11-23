"""Detached HDFS checkpoint uploader used by the Verl trainer."""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import re
import shutil
import subprocess
import time


def _write_status(path: str, status: str, **data) -> None:
    payload = {"status": status, "updated_at": time.time(), **data}
    tmp_path = f"{path}.tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, sort_keys=True)
        f.write("\n")
    os.replace(tmp_path, path)


def _prune_local_checkpoints(checkpoint_dir: str, max_to_keep: int | None) -> None:
    if max_to_keep is None or max_to_keep <= 0:
        return
    checkpoints = []
    for name in os.listdir(checkpoint_dir):
        match = re.fullmatch(r"global_step_(\d+)", name)
        path = os.path.join(checkpoint_dir, name)
        if match and os.path.isdir(path):
            checkpoints.append((int(match.group(1)), path))
    checkpoints.sort(reverse=True)
    for _, path in checkpoints[max_to_keep:]:
        status_path = os.path.join(
            checkpoint_dir,
            f".hdfs_upload_{os.path.basename(path)}.json",
        )
        try:
            with open(status_path, encoding="utf-8") as f:
                status = json.load(f).get("status")
        except (OSError, ValueError):
            continue
        if status == "complete":
            shutil.rmtree(path)


def upload_checkpoint(
    *,
    hdfs_bin: str,
    local_path: str,
    remote_path: str,
    tracker_local_path: str,
    tracker_remote_path: str,
    status_path: str,
    max_local_ckpt_to_keep: int | None = None,
) -> None:
    if not os.path.isdir(local_path):
        raise FileNotFoundError(f"Local checkpoint directory does not exist: {local_path}")
    if not os.path.isfile(tracker_local_path):
        raise FileNotFoundError(f"Local checkpoint tracker does not exist: {tracker_local_path}")
    with open(tracker_local_path, encoding="utf-8") as f:
        tracker_value = f.read().strip()
    local_step = int(tracker_value)
    tracker_snapshot_path = f"{status_path}.tracker"
    tracker_snapshot_tmp = f"{tracker_snapshot_path}.tmp"
    with open(tracker_snapshot_tmp, "w", encoding="utf-8") as f:
        f.write(tracker_value)
    os.replace(tracker_snapshot_tmp, tracker_snapshot_path)

    remote_parent = os.path.dirname(remote_path.rstrip("/"))
    tracker_remote_parent = os.path.dirname(tracker_remote_path.rstrip("/"))
    lock_path = os.path.join(os.path.dirname(local_path), ".hdfs_upload.lock")
    _write_status(
        status_path,
        "waiting",
        local_path=local_path,
        remote_path=remote_path,
        pid=os.getpid(),
    )
    lock_file = open(lock_path, "a+", encoding="utf-8")
    fcntl.flock(lock_file, fcntl.LOCK_EX)
    _write_status(
        status_path,
        "running",
        local_path=local_path,
        remote_path=remote_path,
        pid=os.getpid(),
    )

    subprocess.run([hdfs_bin, "dfs", "-mkdir", "-p", remote_parent], check=True)
    subprocess.run(
        [hdfs_bin, "dfs", "-rm", "-r", "-f", remote_path],
        check=False,
    )
    subprocess.run(
        [hdfs_bin, "dfs", "-put", "-f", local_path, remote_parent],
        check=True,
    )
    subprocess.run(
        [hdfs_bin, "dfs", "-test", "-e", remote_path],
        check=True,
    )
    subprocess.run(
        [hdfs_bin, "dfs", "-mkdir", "-p", tracker_remote_parent],
        check=True,
    )
    remote_tracker = subprocess.run(
        [hdfs_bin, "dfs", "-cat", tracker_remote_path],
        check=False,
        capture_output=True,
        text=True,
    )
    try:
        remote_step = int(remote_tracker.stdout.strip()) if remote_tracker.returncode == 0 else -1
    except ValueError:
        remote_step = -1
    if local_step >= remote_step:
        subprocess.run(
            [hdfs_bin, "dfs", "-put", "-f", tracker_snapshot_path, tracker_remote_path],
            check=True,
        )
    _write_status(
        status_path,
        "complete",
        local_path=local_path,
        remote_path=remote_path,
        pid=os.getpid(),
    )

    _prune_local_checkpoints(os.path.dirname(local_path), max_local_ckpt_to_keep)
    os.remove(tracker_snapshot_path)
    lock_file.close()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hdfs-bin", required=True)
    parser.add_argument("--local-path", required=True)
    parser.add_argument("--remote-path", required=True)
    parser.add_argument("--tracker-local-path", required=True)
    parser.add_argument("--tracker-remote-path", required=True)
    parser.add_argument("--status-path", required=True)
    parser.add_argument("--max-local-ckpt-to-keep", type=int, default=None)
    args = parser.parse_args()

    try:
        upload_checkpoint(
            hdfs_bin=args.hdfs_bin,
            local_path=args.local_path,
            remote_path=args.remote_path,
            tracker_local_path=args.tracker_local_path,
            tracker_remote_path=args.tracker_remote_path,
            status_path=args.status_path,
            max_local_ckpt_to_keep=args.max_local_ckpt_to_keep,
        )
    except Exception as exc:
        _write_status(
            args.status_path,
            "failed",
            local_path=args.local_path,
            remote_path=args.remote_path,
            pid=os.getpid(),
            error=repr(exc),
        )
        raise
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
