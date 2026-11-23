import asyncio
import sys
import types
from types import SimpleNamespace


def _install_fake_sglang_server_module(monkeypatch, server_cls):
    module_name = "verl.workers.rollout.sglang_rollout.async_sglang_server"
    module = types.ModuleType(module_name)
    module.SGLangHttpServer = server_cls
    monkeypatch.setitem(sys.modules, module_name, module)


def test_sglang_sleep_aborts_and_drains_before_release(monkeypatch):
    from rllm.trainer.verl import patch as patch_module

    events = []

    class FakeModelUpdateLock:
        def __init__(self):
            self.results = iter([True, False])

        async def is_locked(self):
            result = next(self.results)
            events.append(("locked", result))
            return result

    class FakeTokenizerManager:
        def __init__(self):
            self.model_update_lock = FakeModelUpdateLock()

        def abort_request(self, *, abort_all):
            events.append(("abort", abort_all))

    class FakeSGLangHttpServer:
        async def sleep(self):
            events.append(("release", True))

    _install_fake_sglang_server_module(monkeypatch, FakeSGLangHttpServer)
    monkeypatch.setattr(patch_module, "_VERL_SGLANG_SLEEP_DRAIN_PATCHED", False)
    monkeypatch.setattr(patch_module, "_debug_report_patch", lambda *args, **kwargs: None)
    monkeypatch.setenv("RLLM_SGLANG_DRAIN_TIMEOUT_S", "1")

    patch_module.patch_verl_sglang_sleep_drain()

    server = FakeSGLangHttpServer()
    server.node_rank = 0
    server.replica_rank = 3
    server.config = SimpleNamespace(free_cache_engine=True)
    server.rollout_mode = SimpleNamespace(name="COLOCATED")
    server.tokenizer_manager = FakeTokenizerManager()

    asyncio.run(server.sleep())

    assert events == [
        ("abort", True),
        ("locked", True),
        ("abort", True),
        ("locked", False),
        ("release", True),
    ]


def test_sglang_sleep_does_not_drain_standalone_server(monkeypatch):
    from rllm.trainer.verl import patch as patch_module

    events = []

    class FakeTokenizerManager:
        def abort_request(self, *, abort_all):
            events.append(("abort", abort_all))

        async def flush_cache(self):
            events.append(("flush", True))
            return SimpleNamespace(success=True)

    class FakeSGLangHttpServer:
        async def sleep(self):
            events.append(("release", True))

    _install_fake_sglang_server_module(monkeypatch, FakeSGLangHttpServer)
    monkeypatch.setattr(patch_module, "_VERL_SGLANG_SLEEP_DRAIN_PATCHED", False)
    monkeypatch.setattr(patch_module, "_debug_report_patch", lambda *args, **kwargs: None)

    patch_module.patch_verl_sglang_sleep_drain()

    server = FakeSGLangHttpServer()
    server.node_rank = 0
    server.replica_rank = 0
    server.config = SimpleNamespace(free_cache_engine=True)
    server.rollout_mode = SimpleNamespace(name="STANDALONE")
    server.tokenizer_manager = FakeTokenizerManager()

    asyncio.run(server.sleep())

    assert events == [("release", True)]
