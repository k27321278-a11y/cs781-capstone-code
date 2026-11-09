import importlib
import json
import os
from pathlib import Path

import pytest
import torch

from cs781.contracts import ModelConfig


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(),reason="CUDA required")
def test_flash_training_prefill_and_cached_decode(tmp_path):
    try:
        cls = importlib.import_module("cs781.models.a").LegalGPT
    except ImportError as exc:
        pytest.fail(f"A graph missing: {exc}")
    from torch.nn.attention import sdpa_kernel, SDPBackend
    torch.manual_seed(781)
    model = cls(ModelConfig()).to(device="cuda",dtype=torch.bfloat16)
    tokens = torch.randint(4,1000,(1,2048),device="cuda")
    results = {}
    for phase in ("training","prefill","decode"):
        model.train(phase == "training")
        with sdpa_kernel(SDPBackend.FLASH_ATTENTION), torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA]) as profile:
            if phase == "training":
                _,loss,_ = model(tokens,tokens)
                loss.backward()
                assert torch.isfinite(loss)
                assert all(torch.isfinite(p.grad).all() for p in model.parameters())
                model.zero_grad(set_to_none=True)
            else:
                with torch.no_grad():
                    if phase == "prefill":
                        logits,_,_ = model(tokens,use_cache=True)
                    else:
                        _,_,cache = model(tokens[:,:64],use_cache=True)
                        logits,_,_ = model(tokens[:,64:65],past_kvs=cache,use_cache=True)
                    assert torch.isfinite(logits).all()
            torch.cuda.synchronize()
        names = sorted({event.key for event in profile.key_averages() if "flash" in event.key})
        assert any("flash_attention" in name for name in names)
        results[phase] = names
    output = Path(os.environ.get("CS781_EVIDENCE_DIR",str(tmp_path)))
    output.mkdir(parents=True,exist_ok=True)
    (output/"flash_a.json").write_text(json.dumps({"torch":torch.__version__,"gpu":torch.cuda.get_device_name(),"dtype":"bfloat16","training_shape":[1,12,2048,64],"operators":results},indent=2))
