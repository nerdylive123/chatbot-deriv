import sys

import pytest

from support_pipeline import runtime
from support_pipeline.runtime import LocalModelPool, detect_device


class Model:
    def __init__(self, name, device):
        self.name = name
        self.device = device


def _pool():
    created = []

    def factory(name):
        def build(device):
            model = Model(name, device)
            created.append(model)
            return model

        return build

    pool = LocalModelPool({"enc": factory("enc"), "rr": factory("rr")})
    return pool, created


def test_nothing_loaded_until_acquire():
    pool, created = _pool()
    assert created == []
    assert pool.active is None


def test_acquire_loads_lazily_and_reuses_resident_model():
    pool, created = _pool()
    first = pool.acquire("enc")
    again = pool.acquire("enc")
    assert first is again
    assert len(created) == 1
    assert pool.active == "enc"
    assert first.device == pool.device


def test_one_model_resident_at_a_time():
    pool, created = _pool()
    enc = pool.acquire("enc")
    rr = pool.acquire("rr")
    assert pool.active == "rr"
    assert pool.resident is rr
    assert pool.resident is not enc
    enc2 = pool.acquire("enc")
    assert enc2 is not enc  # reloaded after being released
    assert [m.name for m in created] == ["enc", "rr", "enc"]


def test_release_drops_model():
    pool, _ = _pool()
    pool.acquire("enc")
    pool.release()
    assert pool.active is None
    assert pool.resident is None
    pool.release()  # idempotent


def test_unknown_model_raises():
    pool, _ = _pool()
    with pytest.raises(KeyError):
        pool.acquire("missing")


def test_device_is_cpu_without_torch(monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", None)  # makes `import torch` fail
    assert detect_device() == "cpu"
    assert LocalModelPool({}).device == "cpu"


def test_device_cuda_when_available(monkeypatch):
    class FakeCuda:
        emptied = 0

        @staticmethod
        def is_available():
            return True

        @classmethod
        def empty_cache(cls):
            cls.emptied += 1

    class FakeTorch:
        cuda = FakeCuda

    monkeypatch.setitem(sys.modules, "torch", FakeTorch)
    assert detect_device() == "cuda"
    pool = LocalModelPool({"enc": lambda device: Model("enc", device)})
    assert pool.device == "cuda"
    pool.acquire("enc")
    pool.release()
    assert FakeCuda.emptied >= 1


def test_device_cpu_when_torch_check_raises(monkeypatch):
    class BrokenCuda:
        @staticmethod
        def is_available():
            raise RuntimeError("driver error")

        @staticmethod
        def empty_cache():
            raise RuntimeError("driver error")

    class FakeTorch:
        cuda = BrokenCuda

    monkeypatch.setitem(sys.modules, "torch", FakeTorch)
    assert detect_device() == "cpu"
    pool = LocalModelPool({"enc": lambda device: Model("enc", device)})
    pool.acquire("enc")
    pool.release()  # best effort: must not raise
    assert pool.active is None


def test_runtime_import_is_light():
    assert "torch" not in vars(runtime)
