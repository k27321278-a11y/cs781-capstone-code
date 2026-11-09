"""Catch malformed prompt acceptance, hidden filler and premature word completion."""
import importlib.util
import torch
import pytest
from legalspan.bpe import OwnBPE
from test_legalspan_model import tiny
from cs781.contracts import GenerationConfig, GenerationIncomplete


def api():
    assert importlib.util.find_spec('legalspan.generate') is not None, 'Legal-span generation not implemented'
    from legalspan.generate import generate
    return generate


def test_invalid_prompt_and_insufficient_context_raise_without_output():
    generate = api(); tok = OwnBPE.train(['abc '*100], 290); model = tiny()
    with pytest.raises(ValueError, match='512'):
        generate(model, tok, 'abc', GenerationConfig())
    with pytest.raises(Exception, match='context'):
        generate(model, tok, 'abc '*512, GenerationConfig())


def test_random_model_never_fills_token_cap_or_returns_partial_words():
    generate = api(); tok = OwnBPE.train(['abc '*100], 290)
    from legalspan.model import SpanConfig, LegalSpanGPT
    model = LegalSpanGPT(SpanConfig(vocab_size=tok.get_vocab_size(), max_seq_len=2048,
                   n_layers=2, d_model=16, n_heads=2, d_head=8, d_ff=32,
                   route_layers=1, route_dim=8)).eval()
    with pytest.raises(GenerationIncomplete):
        generate(model, tok, 'abc '*512, GenerationConfig(max_new_tokens=2))
