import importlib
from types import SimpleNamespace

import pytest
import torch


def api():
    try:
        return importlib.import_module("cs781.generate")
    except ImportError as exc:
        pytest.fail(f"Generation missing: {exc}")


class Pieces:
    def token_to_id(self,token):
        return {"<pad>":0,"<bos>":1,"<eos>":2,"<unk>":3}[token]
    def encode(self,text):
        return SimpleNamespace(ids=[4]*len(text.split()))
    def decode(self,ids,**kwargs):
        return "".join({4:"con",5:"tin",6:"uation "}.get(i,"") for i in ids)


class ScriptedModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(1),requires_grad=False)
        self.cfg = SimpleNamespace(max_seq_len=2048)
        self.step = 0
    def forward(self,ids,**kwargs):
        logits = torch.full((1,1,8),-100.)
        logits[:,:,2] = 1000  # Real generation must suppress early EOS.
        logits[:,:,4+self.step%3] = 100
        self.step += 1
        return logits,None,[]


def test_word_count_is_not_bpe_count_and_eos_is_suppressed():
    g = api()
    from cs781.contracts import GenerationConfig
    result = g.generate_continuation(ScriptedModel(),Pieces(),"prompt "*512,GenerationConfig(max_new_tokens=400))
    assert result.split() == ["continuation"]*128


def test_token_cap_and_long_prompt_are_errors_not_filler():
    g = api()
    from cs781.contracts import GenerationConfig,GenerationIncomplete,PromptTooLong
    with pytest.raises(GenerationIncomplete):
        g.generate_continuation(ScriptedModel(),Pieces(),"prompt "*512,GenerationConfig(max_new_tokens=10))
    model = ScriptedModel()
    model.cfg.max_seq_len = 128
    with pytest.raises(PromptTooLong):
        g.generate_continuation(model,Pieces(),"prompt "*512,GenerationConfig())
