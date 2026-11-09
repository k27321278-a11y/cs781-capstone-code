"""Catch future-target leakage, wrong prefix boundary, lost copy mass and cache drift."""
import importlib.util
import torch
import pytest


def api():
    assert importlib.util.find_spec('legalspan.model') is not None, 'Legal-span route not implemented'
    from legalspan.model import SpanConfig, LegalSpanGPT, continuity, copy_distribution
    return SpanConfig, LegalSpanGPT, continuity, copy_distribution


def tiny(mode='full'):
    C, M, _, _ = api()
    torch.manual_seed(781)
    return M(C(vocab_size=300, max_seq_len=64, n_layers=3, d_model=24,
               n_heads=3, d_head=8, d_ff=48, route_layers=2, route_dim=8,
               route_mode=mode, gradient_checkpointing=False)).eval()


def test_observed_history_continuity_and_repeated_copy_ids():
    _, _, continuity, scatter = api()
    prompt = torch.tensor([[1, 8, 9, 8, 9, 12]])
    history = torch.tensor([[1, 8, 9, 8, 9, 12, 8, 9]])
    c = continuity(history, prompt, query_start=5)
    assert c.shape == (1, 3, 6)
    assert c[0, 2].tolist() == [0, 0, 0, 2, 0, 2]
    four = continuity(torch.tensor([[8, 9, 8, 9]]), prompt, query_start=3)
    assert four[0, 0, 5] == 4
    alpha = torch.tensor([[[.0, .1, .2, .3, .4, .0]]])
    p = scatter(alpha, prompt, 300)
    assert torch.allclose(p[0, 0, torch.tensor([8, 9])], torch.tensor([.4, .6]))
    assert torch.allclose(p.sum(-1), torch.ones(1, 1))


def test_future_perturbation_cannot_change_previous_suffix_probabilities():
    model = tiny()
    ids = torch.tensor([[1, 8, 9, 12, 8, 9, 14, 16]])
    roles = torch.tensor([[0, 1, 2, 3]])
    p, _, _ = model(ids, prompt_length=4, prompt_roles=roles)
    changed = ids.clone(); changed[0, 6:] = torch.tensor([19, 20])
    q, _, _ = model(changed, prompt_length=4, prompt_roles=roles)
    torch.testing.assert_close(p[:, :3], q[:, :3])
    assert not torch.allclose(p[:, 3:], q[:, 3:])


def test_first_suffix_and_cached_probs_equal_teacher_forcing():
    model = tiny()
    ids = torch.tensor([[1, 8, 9, 12, 8, 9, 14]])
    roles = torch.tensor([[0, 1, 2, 3]])
    full, _, _ = model(ids, prompt_length=4, prompt_roles=roles)
    first, _, cache = model(ids[:, :4], prompt_length=4, prompt_roles=roles, use_cache=True)
    torch.testing.assert_close(first[:, -1], full[:, 0], atol=2e-6, rtol=2e-6)
    for i in range(4, ids.shape[1]):
        p, _, cache = model(ids[:, i:i+1], cache=cache, use_cache=True)
        torch.testing.assert_close(p[:, -1], full[:, i-3], atol=2e-6, rtol=2e-6)


def test_normalized_mixture_no_eligible_copy_and_finite_gradients():
    model = tiny().train()
    x = torch.tensor([[1, 8, 9, 12, 8, 9]])
    y = torch.tensor([[-100, -100, -100, 8, 9, 2]])
    p, loss, _ = model(x, targets=y, prompt_length=4, prompt_roles=torch.zeros(1, 4, dtype=torch.long))
    torch.testing.assert_close(p.exp().sum(-1), torch.ones(1, 3), atol=2e-6, rtol=2e-6)
    loss.backward()
    assert torch.isfinite(loss) and all(torch.isfinite(v.grad).all() for v in model.parameters() if v.grad is not None)
    p, _, _ = model(torch.tensor([[1, 2, 3]]), prompt_length=3,
                    prompt_roles=torch.zeros(1, 3, dtype=torch.long))
    assert torch.isfinite(p).all()
    torch.testing.assert_close(p.exp().sum(-1), torch.ones(1, 1), atol=2e-6, rtol=2e-6)


def test_control_matches_existing_dense_graph_exactly():
    from cs781.models.a import LegalGPT
    from cs781.contracts import ModelConfig
    model = tiny('control')
    cfg = ModelConfig(**{k: getattr(model.cfg, k) for k in ModelConfig.__dataclass_fields__})
    dense = LegalGPT(cfg).eval()
    dense.load_state_dict(model.state_dict(), strict=True)
    ids = torch.tensor([[1, 8, 9, 12, 8, 9]])
    actual, _, _ = model(ids, prompt_length=4, prompt_roles=torch.zeros(1, 4, dtype=torch.long))
    logits, _, _ = dense(ids)
    torch.testing.assert_close(actual, logits[:, 3:].float().log_softmax(-1))


def test_memory_efficient_gold_loss_equals_full_distribution_loss():
    model = tiny().train()
    x = torch.tensor([[1, 8, 9, 12, 8, 9]])
    y = torch.tensor([[-100, -100, -100, 8, 9, 2]])
    roles = torch.zeros(1, 4, dtype=torch.long)
    _, expected, _ = model(x, targets=y, prompt_length=4, prompt_roles=roles)
    expected.backward(); gradients = {k: v.grad.clone() for k, v in model.named_parameters() if v.grad is not None}
    model.zero_grad(set_to_none=True)
    assert 'return_probs' in __import__('inspect').signature(model.forward).parameters, 'Gold-only normalized loss path missing'
    probs, actual, _ = model(x, targets=y, prompt_length=4, prompt_roles=roles, return_probs=False)
    assert probs is None
    torch.testing.assert_close(actual, expected)
    actual.backward()
    for k, v in model.named_parameters():
        if k in gradients: torch.testing.assert_close(v.grad, gradients[k], atol=1e-6, rtol=2e-5)


def test_generic_route_removes_continuity_from_fusion_and_copy_gate_features():
    model = tiny('generic')
    route = model.blocks[-1].route
    with torch.no_grad(): route.gate.weight[:, -1] = 3.
    x = torch.randn(1, 6, model.cfg.d_model)
    roles = torch.tensor([[0, 1, 2, 3]])
    eligible = torch.tensor([[False, True, True, True]])
    zero = torch.zeros(1, 3, 4); nonzero = torch.full_like(zero, 4)
    a = route(x, 3, 4, roles, eligible, zero)
    b = route(x, 3, 4, roles.flip(1), eligible, nonzero)
    torch.testing.assert_close(a[0], b[0])
    torch.testing.assert_close(b[-1], torch.zeros_like(b[-1]))


@pytest.mark.skipif(not torch.cuda.is_available(), reason='Actual CUDA required')
@pytest.mark.parametrize('mode', ['full', 'generic'])
def test_cuda_bfloat16_normalization_and_cache_parity(mode):
    model = tiny(mode).cuda()
    ids = torch.tensor([[1, 8, 9, 12, 8, 9, 14]], device='cuda')
    roles = torch.tensor([[0, 1, 2, 3]], device='cuda')
    with torch.autocast('cuda', dtype=torch.bfloat16):
        full, _, _ = model(ids, prompt_length=4, prompt_roles=roles)
        first, _, cache = model(ids[:, :4], prompt_length=4, prompt_roles=roles, use_cache=True)
        torch.testing.assert_close(first[:, -1], full[:, 0], atol=.015, rtol=.002)
        torch.testing.assert_close(first.exp().sum(-1), torch.ones(1, 1, device='cuda'), atol=2e-5, rtol=2e-5)
        for i in range(4, ids.shape[1]):
            probs, _, cache = model(ids[:, i:i+1], cache=cache, use_cache=True)
            torch.testing.assert_close(probs[:, -1], full[:, i-3], atol=.015, rtol=.002)
            torch.testing.assert_close(probs.exp().sum(-1), torch.ones(1, 1, device='cuda'), atol=2e-5, rtol=2e-5)
