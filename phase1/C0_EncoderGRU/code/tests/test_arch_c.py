import importlib.util

import pytest
import torch
from torch.nn import functional as F


def classes():
    assert importlib.util.find_spec('cs781.models.c') is not None, 'C recurrent model is not implemented'
    from cs781.models.c import RecurrentConfig, RecurrentLegalModel
    return RecurrentConfig, RecurrentLegalModel


def tiny_model(**updates):
    cfg_class, model_class = classes()
    values = dict(vocab_size=41, max_seq_len=16, max_prompt_len=32,
                  n_encoder_layers=2, n_gru_layers=2, d_model=16,
                  n_heads=2, d_head=8, d_ff=32, gradient_checkpointing=False)
    values.update(updates)
    torch.manual_seed(781)
    return model_class(cfg_class(**values))


def test_c_exists_and_has_recurrent_generator():
    model = tiny_model()
    assert isinstance(model.gru, torch.nn.GRU)
    assert not hasattr(model, 'decoder')
    assert model.lm_head.weight is model.tok_embeddings.weight


def test_suffix_future_cannot_change_previous_logits():
    model = tiny_model().eval()
    prompt = torch.tensor([[1, 4, 5]])
    decoder = torch.tensor([[1, 6, 7, 8]])
    original = model(prompt, decoder)[0]
    decoder[0, 2:] = torch.tensor([13, 14])
    changed = model(prompt, decoder)[0]
    torch.testing.assert_close(original[:, :2], changed[:, :2], atol=1e-6, rtol=1e-5)


def test_cached_recurrence_matches_full_sequence_and_prompt_is_encoded_once():
    model = tiny_model().eval()
    prompt = torch.tensor([[1, 4, 5], [1, 8, 9]])
    decoder = torch.tensor([[1, 6, 7, 8], [1, 4, 3, 9]])
    full = model(prompt, decoder)[0]
    cache, pieces = None, []
    calls = []
    handle = model.encoder[0].register_forward_hook(lambda *args: calls.append(1))
    for i in range(decoder.size(1)):
        logits, _, cache = model(prompt if cache is None else None, decoder[:, i:i+1],
                                  past_kvs=cache, use_cache=True)
        pieces.append(logits)
    handle.remove()
    assert len(calls) == 1
    assert cache['hidden'].shape == (2, 2, 16)
    assert cache['steps'] == 4
    assert set(cache) == {'hidden', 'cross', 'prompt_mask', 'steps'}
    torch.testing.assert_close(torch.cat(pieces, dim=1), full, atol=2e-6, rtol=2e-5)


def test_padding_is_excluded_from_summary_and_cross_attention():
    model = tiny_model().eval()
    decoder = torch.tensor([[1, 8, 9]])
    prompt = torch.tensor([[1, 3, 4]])
    padded = torch.tensor([[1, 3, 4, 20, 21]])
    mask = torch.tensor([[True, True, True, False, False]])
    torch.testing.assert_close(model(prompt, decoder)[0], model(padded, decoder, prompt_mask=mask)[0],
                               atol=2e-6, rtol=2e-5)
    with pytest.raises(ValueError, match='valid token'):
        model(padded, decoder, prompt_mask=torch.zeros_like(padded, dtype=torch.bool))


def test_extended_prompt_capacity_does_not_extend_training_decoder():
    model = tiny_model().eval()
    assert model(torch.ones(1, 24, dtype=torch.long), torch.ones(1, 3, dtype=torch.long))[0].shape == (1, 3, 41)
    with pytest.raises(ValueError, match='encoder context'):
        model(torch.ones(1, 33, dtype=torch.long), torch.ones(1, 1, dtype=torch.long))
    with pytest.raises(ValueError, match='decoder context'):
        model(torch.ones(1, 3, dtype=torch.long), torch.ones(1, 17, dtype=torch.long))


def test_masked_loss_and_gradient_flow_through_recurrent_initialisation():
    model = tiny_model(gradient_checkpointing=True).train()
    prompt, decoder = torch.tensor([[1, 4, 5]]), torch.tensor([[1, 6, 7]])
    targets = torch.tensor([[6, 7, -100]])
    logits, loss, _ = model(prompt, decoder, targets=targets)
    expected = F.cross_entropy(logits.float().reshape(-1, 41), targets.flatten(), ignore_index=-100)
    torch.testing.assert_close(loss, expected)
    loss.backward()
    for name in ('initial_projection.weight', 'gru.weight_ih_l0', 'encoder.0.attn.wqkv.weight'):
        grad = dict(model.named_parameters())[name].grad
        assert grad is not None and torch.isfinite(grad).all() and grad.abs().sum() > 0


def test_invalid_cache_rejected_and_context_counted():
    model = tiny_model().eval()
    prompt, token = torch.tensor([[1, 4]]), torch.tensor([[1]])
    cache = model(prompt, token, use_cache=True)[2]
    with pytest.raises(ValueError, match='cache'):
        model(None, token, past_kvs={'hidden': cache['hidden']}, use_cache=True)
    with pytest.raises(ValueError, match='one token'):
        model(None, torch.tensor([[1, 2]]), past_kvs=cache, use_cache=True)
    cache['steps'] = 16
    with pytest.raises(ValueError, match='decoder context'):
        model(None, token, past_kvs=cache, use_cache=True)


def test_default_unique_parameters_fit_required_foundation_range():
    cfg_class, model_class = classes()
    with torch.device('meta'):
        model = model_class(cfg_class())
    assert len(model.encoder) == 10 and model.gru.num_layers == 4
    assert 50_000_000 <= model.count_parameters() <= 300_000_000


def test_tiny_recurrent_model_can_overfit_shifted_suffix():
    model = tiny_model(n_encoder_layers=1, n_gru_layers=1).train()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.025)
    prompt = torch.tensor([[1, 3, 4], [1, 5, 6]])
    decoder = torch.tensor([[1, 7, 8], [1, 9, 10]])
    target = torch.tensor([[7, 8, 2], [9, 10, 2]])
    for _ in range(80):
        optimizer.zero_grad()
        _, loss, _ = model(prompt, decoder, target)
        loss.backward()
        optimizer.step()
    assert float(loss) < 0.02


@pytest.mark.gpu
def test_cuda_bf16_surroundings_keep_gru_backward_finite():
    if not torch.cuda.is_available():
        pytest.skip('CUDA required')
    model = tiny_model(gradient_checkpointing=True).cuda().train()
    prompt, decoder = torch.tensor([[1, 4, 5]], device='cuda'), torch.tensor([[1, 6, 7]], device='cuda')
    with torch.autocast('cuda', dtype=torch.bfloat16):
        _, loss, _ = model(prompt, decoder, targets=torch.tensor([[6, 7, 2]], device='cuda'))
    loss.backward()
    assert torch.isfinite(loss)
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())


@pytest.mark.gpu
def test_cuda_bf16_cached_recurrence_matches_full_sequence():
    if not torch.cuda.is_available():
        pytest.skip('CUDA required')
    model = tiny_model(d_model=64, n_heads=4, d_head=16, d_ff=128).cuda().eval()
    prompt = torch.tensor([[1, 4, 5], [1, 6, 7]], device='cuda')
    decoder = torch.tensor([[1, 6, 7, 8], [1, 9, 10, 11]], device='cuda')
    with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
        full = model(prompt, decoder)[0]
        cache, pieces = None, []
        for i in range(decoder.size(1)):
            logits, _, cache = model(prompt if cache is None else None,
                                      decoder[:, i:i+1], past_kvs=cache, use_cache=True)
            pieces.append(logits)
        assert cache['hidden'].dtype == torch.float32
        torch.testing.assert_close(torch.cat(pieces, dim=1), full, atol=0.015, rtol=0.02)
