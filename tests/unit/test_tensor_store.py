import pytest
import torch

from meshtrain.models import TinyTransformerSpec
from meshtrain.runtime.distributed_autograd import MicrobatchContext
from meshtrain.runtime.stage import Stage
from meshtrain.runtime.tensor_store import LocalTensorStore, MemoryTier, TensorRole, parameter_id


def _stage(offset=2):
    spec = TinyTransformerSpec(layers=4, hidden_size=32, heads=4, vocab_size=32, seq_len=8)
    return Stage(spec.build_stage(offset, offset + 2), stage_index=1, num_stages=3, optimizer="adamw",
                 layer_offset=offset), spec


def test_stable_global_parameter_ids():
    st, spec = _stage(offset=2)
    ids = st.tensor_store.ids(TensorRole.PARAMETER)
    assert "model.layers.2.qkv.weight" in ids and "model.layers.3.fc2.bias" in ids
    full = {parameter_id(int(n.split(".", 1)[0]), n.split(".", 1)[1]) for n, _ in spec.build_full().named_parameters()}
    assert set(ids) <= full  # same identity whichever worker owns the layer


def test_roles_bytes_and_static_tier():
    st, _ = _stage()
    store = st.tensor_store
    meta = store.locate("model.layers.2.qkv.weight")
    assert meta.role == TensorRole.PARAMETER and meta.tier == MemoryTier.LOCAL_RAM and meta.owner == st.name
    x = torch.randn(2, 8, 32)
    st.forward(x, MicrobatchContext(0, 0))
    assert store.ids(TensorRole.ACTIVATION) == ["act.s0.mb0.stage1.input"]
    out = st.contexts._contexts[(0, 0)].output
    st.backward(torch.ones_like(out), (0, 0))
    assert store.ids(TensorRole.ACTIVATION) == []  # released with the microbatch
    st.optimizer_step()
    st.refresh_tensor_store()
    by = store.bytes_by_role()
    rep = st.memory_report()
    assert by["parameter"] == rep["parameters"] and by["gradient"] == rep["gradients"]
    assert by["optimizer_state"] == pytest.approx(rep["optimizer_state"], rel=0.01)  # minus scalar step counters
    assert "model.layers.2.qkv.weight.optim.exp_avg" in store.ids(TensorRole.OPTIMIZER_STATE)


def test_v2_operations_are_refused_not_faked():
    store = LocalTensorStore("w", "cuda")
    store.put("t", torch.zeros(2), TensorRole.PARAMETER)
    assert store.locate("t").tier == MemoryTier.LOCAL_ACCELERATOR
    store.prefetch("t")  # resident: no-op
    with pytest.raises(NotImplementedError):
        store.evict("t", MemoryTier.LOCAL_NVME)
    with pytest.raises(NotImplementedError):
        store.put("u", torch.zeros(2), TensorRole.PARAMETER, MemoryTier.REMOTE_RAM)
    with pytest.raises(KeyError):
        store.get("missing")
