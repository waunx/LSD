import contextlib
import importlib.util
import io
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

SCRIPT_DIR = Path(__file__).resolve().parents[1] / "cookbooks/deepresearch"
SPEC = importlib.util.spec_from_file_location("refine_local", SCRIPT_DIR / "serve_refine_local.py")
refine = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(refine)


class RefineLocalServiceTest(unittest.TestCase):
    def args(self, root, **overrides):
        values = dict(
            python=sys.executable,
            model_path=Path("/tmp/deepresearch_seqmean_models/Qwen3-8B"),
            gpu_ids="0",
            base_port=40100,
            context_length=65536,
            advertise_host="2001:db8::1",
            state=Path(root) / "servers.json",
            startup_timeout=10,
            dry_run=False,
        )
        values.update(overrides)
        return SimpleNamespace(**values)

    def test_long_context_uses_yarn_without_editing_weights(self):
        cmd = refine.build_command(self.args("."), 3)
        self.assertEqual(cmd[cmd.index("--port") + 1], "40103")
        self.assertEqual(cmd[cmd.index("--max-model-len") + 1], "65536")
        overrides = json.loads(cmd[cmd.index("--hf-overrides") + 1])
        self.assertEqual(overrides["rope_scaling"]["factor"], 2.0)
        self.assertEqual(overrides["rope_scaling"]["original_max_position_embeddings"], 32768)
        self.assertIn("--enforce-eager", cmd)
        self.assertNotIn("--reasoning-parser", cmd)

    def test_native_context_does_not_add_yarn(self):
        cmd = refine.build_command(self.args(".", context_length=40960), 0)
        self.assertNotIn("--hf-overrides", cmd)
        with self.assertRaises(ValueError):
            refine.build_command(self.args(".", context_length=262144), 0)

    def test_gpu_probe_excludes_any_existing_cuda_process(self):
        responses = [
            SimpleNamespace(stdout="0, GPU-a, 97871, 0\n1, GPU-b, 97871, 0\n"),
            SimpleNamespace(stdout="GPU-a, 1234\n"),
        ]
        with patch.object(refine.subprocess, "run", side_effect=responses):
            self.assertEqual(refine.free_gpu_ids("auto"), [1])
        with patch.object(refine.subprocess, "run", side_effect=responses):
            with self.assertRaises(RuntimeError):
                refine.free_gpu_ids("0")

    def test_gpu_permission_failure_is_not_ignored(self):
        with patch.object(refine.subprocess, "run", side_effect=subprocess.CalledProcessError(1, "nvidia-smi")):
            with self.assertRaises(subprocess.CalledProcessError):
                refine.free_gpu_ids("auto")

    def test_missing_state_keeps_remote_pool(self):
        with tempfile.TemporaryDirectory() as root:
            self.assertEqual(refine.resolve_urls(Path(root) / "missing.json"), refine.REMOTE_URLS)

    def test_pool_only_adds_healthy_long_context_replicas(self):
        with tempfile.TemporaryDirectory() as root:
            state = Path(root) / "servers.json"
            state.write_text(
                json.dumps(
                    {
                        "context_length": 65536,
                        "servers": [{"url": "http://ready/v1"}, {"url": "http://dead/v1"}],
                    }
                )
            )
            with patch.object(refine, "check_endpoint", side_effect=[{}, OSError("down")]):
                self.assertEqual(refine.resolve_urls(state), [*refine.REMOTE_URLS, "http://ready/v1"])

    def test_context_and_model_validation(self):
        models = {"data": [{"id": refine.MODEL_NAME, "max_model_len": 40960}]}
        with patch.object(refine, "request_json", side_effect=[{}, models]):
            with self.assertRaises(ValueError):
                refine.check_endpoint("http://host/v1", 65536)
        models["data"][0]["max_model_len"] = 65536
        with patch.object(refine, "request_json", side_effect=[{}, models]):
            self.assertEqual(refine.check_endpoint("http://host/v1", 65536)["id"], refine.MODEL_NAME)

    def test_smoke_requires_refine_parser_compatible_output(self):
        response = {"choices": [{"message": {"content": "no required tags"}}]}
        with patch.object(refine, "request_json", return_value=response):
            with self.assertRaises(ValueError):
                refine.smoke_endpoint("http://host/v1")

    def test_dry_run_does_not_touch_gpus_or_publish(self):
        with tempfile.TemporaryDirectory() as root:
            args = self.args(root, dry_run=True)
            with (
                patch.object(refine, "validate_model"),
                patch.object(refine, "free_gpu_ids") as gpu_probe,
                patch.object(refine.subprocess, "Popen") as process,
                contextlib.redirect_stdout(io.StringIO()),
            ):
                refine.start(args)
            gpu_probe.assert_not_called()
            process.assert_not_called()
            self.assertFalse(args.state.exists())

    def test_startup_failure_does_not_publish(self):
        with tempfile.TemporaryDirectory() as root, contextlib.ExitStack() as stack:
            args = self.args(root)
            process = MagicMock(pid=98765)
            process.poll.return_value = None
            stack.enter_context(patch.object(refine, "validate_model"))
            stack.enter_context(patch.object(refine, "free_gpu_ids", return_value=[0]))
            stack.enter_context(patch.object(refine.socket, "socket"))
            stack.enter_context(patch.object(refine, "server_environment", return_value={}))
            stack.enter_context(patch.object(refine.subprocess, "Popen", return_value=process))
            stack.enter_context(patch.object(refine, "check_endpoint", return_value={}))
            stack.enter_context(patch.object(refine, "smoke_endpoint", side_effect=ValueError("bad response")))
            kill = stack.enter_context(patch.object(refine.os, "killpg"))
            stack.enter_context(contextlib.redirect_stdout(io.StringIO()))
            with self.assertRaises(ValueError):
                refine.start(args)
            self.assertFalse(args.state.exists())
            kill.assert_called_once_with(98765, refine.signal.SIGTERM)

    def test_environment_preserves_sandbox_preloads(self):
        with (
            patch.dict(os.environ, {"LD_PRELOAD": "/existing/instrumentation.so", "PYTHONPATH": "/training"}),
            patch.object(refine.subprocess, "check_output", return_value="/site\n"),
        ):
            env = refine.server_environment(sys.executable, 2)
        self.assertEqual(env["LD_PRELOAD"], "/existing/instrumentation.so")
        self.assertEqual(env["CUDA_VISIBLE_DEVICES"], "2")
        self.assertNotIn("PYTHONPATH", env)

    def test_pool_shell_preserves_explicit_urls(self):
        env = {**os.environ, "DEEPRESEARCH_REFINE_URLS": "http://custom/v1"}
        result = subprocess.run(
            ["bash", "-c", 'source "$1"\nprintf "%s" "$DEEPRESEARCH_REFINE_URLS"', "_", str(SCRIPT_DIR / "refine_pool.sh")],
            env=env,
            check=True,
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.stdout, "http://custom/v1")

    def test_two_node_entries_pin_separate_refine_pools_before_ray(self):
        base = (SCRIPT_DIR / "launch_base_rl_two_node.sh").read_text()
        lsd = (SCRIPT_DIR / "launch_lsd_two_node_hybrid.sh").read_text()
        marker = 'export DEEPRESEARCH_REFINE_URLS='
        self.assertLess(base.index(marker), base.index("ray_cli start"))
        self.assertLess(lsd.index(marker), lsd.index("ray_cli start"))
        self.assertIn(":40100/v1", base)
        self.assertNotIn(":40003/v1", base)
        self.assertIn(":40003/v1", lsd)
        self.assertNotIn(":40100/v1", lsd)


if __name__ == "__main__":
    unittest.main()
