import importlib

import pytest
import torch
from torch.nn import functional as F

from cs781.contracts import ModelConfig


def model_class():
    try:
        return importlib.import_module("cs781.models.a").LegalGPT
    except ImportError as exc:
        pytest.fail(f"A graph missing: {exc}")


def make_model(config):
    torch.set_num_threads(2)
    torch.manual_seed(781)
    return model_class()(config)


def test_default_unique_parameter_count():
    cls = model_class()
    with torch.device("meta"):
        model = cls(ModelConfig())
    assert model.count_parameters() == 110_119_680


def test_future_tokens_do_not_change_past(tiny_config):
    model = make_model(tiny_config).eval()
    x = torch.randint(4,100,(1,16))
    changed = x.clone()
    changed[:,8:] = 200
    with torch.no_grad():
        torch.testing.assert_close(model(x)[0][:,:8],model(changed)[0][:,:8],atol=1e-6,rtol=1e-5)


def test_cached_decode_matches_full_prefix(tiny_config):
    model = make_model(tiny_config).eval()
    x = torch.randint(4,100,(1,16))
    with torch.no_grad():
        _,_,cache = model(x[:,:7],use_cache=True)
        for i in range(7,16):
            logits,_,cache = model(x[:,i:i+1],past_kvs=cache,use_cache=True)
            torch.testing.assert_close(logits[:,-1],model(x[:,:i+1])[0][:,-1],atol=1e-5,rtol=1e-5)


def test_suffix_only_loss_and_pad_invariance(tiny_config):
    model = make_model(tiny_config).eval()
    x = torch.tensor([[1,4,5,6]])
    targets = torch.tensor([[-100,-100,6,2]])
    logits,loss,_ = model(x,targets)
    assert loss.item() == pytest.approx(F.cross_entropy(logits[:,2:].reshape(-1,512),targets[:,2:].reshape(-1)).item())
    padded = torch.tensor([[1,4,5,6,0,0]])
    torch.testing.assert_close(model(padded)[0][:,:4],logits,atol=1e-6,rtol=1e-5)


def test_gradient_checkpointing_and_tiny_overfit(tiny_config):
    model = make_model(tiny_config).train()
    x = torch.tensor([[1,4,5,6,7,8,9]])
    y = torch.tensor([[-100,-100,6,7,8,9,2]])
    opt = torch.optim.AdamW(model.parameters(),lr=.02)
    initial = model(x,y)[1].item()
    for _ in range(80):
        opt.zero_grad()
        loss = model(x,y)[1]
        loss.backward()
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
        opt.step()
    assert model(x,y)[1].item() < .2*initial


def test_invalid_dimensions_overlength_and_cache(tiny_config):
    cls = model_class()
    with pytest.raises(ValueError):
        cls(ModelConfig(n_heads=11))
    model = make_model(tiny_config)
    with pytest.raises(ValueError):
        model(torch.ones((1,129),dtype=torch.long))
    with pytest.raises(ValueError):
        model(torch.ones((1,1),dtype=torch.long),past_kvs=[(torch.zeros(1),torch.zeros(1))],use_cache=True)
