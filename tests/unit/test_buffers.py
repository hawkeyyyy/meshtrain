import torch

from meshtrain.runtime.buffers import BufferPool


def test_reuse_by_shape_and_dtype():
    pool = BufferPool()
    a = pool.acquire((4, 8), torch.float32)
    pool.release(a)
    b = pool.acquire((4, 8), torch.float32)
    assert b is a
    c = pool.acquire((4, 8), torch.float16)  # different dtype: new buffer
    assert c is not a
    s = pool.stats()
    assert s["allocation_count"] == 2 and s["reuse_count"] == 1


def test_disabled_pool_always_allocates():
    pool = BufferPool(enabled=False)
    a = pool.acquire((2,), torch.float32)
    pool.release(a)
    assert pool.acquire((2,), torch.float32) is not a and pool.stats()["reuse_count"] == 0


def test_bounded_idle_buffers():
    pool = BufferPool(max_free_per_key=2)
    bufs = [pool.acquire((16,), torch.float32) for _ in range(5)]
    for b in bufs:
        pool.release(b)
    assert pool.stats()["bytes_reserved"] == 2 * 16 * 4
    pool2 = BufferPool(max_bytes=100)
    pool2.release(torch.empty(1000))
    assert pool2.stats()["bytes_reserved"] == 0


def test_pinned_request_falls_back_cleanly():
    pool = BufferPool(pinned=True)
    t = pool.acquire((8,), torch.float32)
    assert t.shape == (8,)
    if not torch.cuda.is_available():
        # without a CUDA driver pinning may be unavailable; the pool must still work
        assert pool.stats()["pinned_fallbacks"] in (0, 1)


class FakeEvent:
    def __init__(self):
        self.done = False

    def query(self):
        return self.done


def test_deferred_release_waits_for_event():
    pool = BufferPool()
    a = pool.acquire((3,), torch.float32)
    ev = FakeEvent()
    pool.release(a, ev)
    assert pool.acquire((3,), torch.float32) is not a  # copy still "in flight"
    ev.done = True
    assert pool.acquire((3,), torch.float32) is a
