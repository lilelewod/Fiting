import pytest

from core.collector import Collector


class _Pipe:
    def __init__(self, payload):
        self.payload = payload

    def recv(self):
        return self.payload


def _collector_with(payloads):
    collector = Collector.__new__(Collector)
    collector.num_envs = len(payloads)
    collector._pipe_parents = {
        env_id: _Pipe(payload) for env_id, payload in enumerate(payloads)
    }
    return collector


def test_receive_reraises_worker_exception():
    collector = _collector_with([AttributeError("worker traceback")])

    with pytest.raises(AttributeError, match="worker traceback"):
        collector.receive(0)


def test_receive_returns_normal_worker_payload():
    payload = ([1.0, 2.0], object())
    collector = _collector_with([payload])

    assert collector.receive(0) is payload


def test_receive_all_reraises_worker_exception():
    collector = _collector_with([None, RuntimeError("worker failed")])

    with pytest.raises(RuntimeError, match="worker failed"):
        collector.receive_all()
