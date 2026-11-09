"""Inference-only coverage ablation for the original LegalSpan prompt route.

Coverage is source-position attention mass, not a factuality guarantee.
Reference: See et al. (2017), https://arxiv.org/abs/1704.04368 .
This pilot applies a fixed decoding penalty; it does not implement their trained
coverage feature/loss and must not be reported as that full method.
"""
import math
import types
import torch
from torch.nn import functional as F


def coverage_attention(scores, eligible, strength, previous=None, active=None):
    if strength < 0 or not math.isfinite(strength):
        raise ValueError('Coverage strength must be finite and nonnegative')
    if scores.ndim != 3 or eligible.shape != scores.shape[::2] or eligible.dtype != torch.bool:
        raise ValueError('Expected scores [B,T,S] and boolean eligibility [B,S]')
    coverage = scores.new_zeros(scores.shape[0], scores.shape[2], dtype=torch.float32) if previous is None else previous.clone()
    if coverage.shape != eligible.shape or not torch.isfinite(coverage).all() or (coverage < 0).any():
        raise ValueError('Invalid prior coverage')
    if active is None:
        active = torch.ones(scores.shape[1], dtype=torch.bool, device=scores.device)
    if active.shape != (scores.shape[1],):
        raise ValueError('Active query mask must match time dimension')
    has_source = eligible.any(-1)
    safe = eligible | ~has_source[:,None]
    rows=[]
    for t in range(scores.shape[1]):
        adjusted = scores[:,t].float() - strength * coverage if active[t] else scores[:,t].float()
        alpha = adjusted.masked_fill(~safe,-torch.inf).softmax(-1) * eligible
        rows.append(alpha)
        if active[t]:
            coverage = coverage + alpha
    return torch.stack(rows,1),coverage


def _forward(route, x, start, prompt_length, roles, eligible, counts, memory=None):
    if route.training:
        raise RuntimeError('Coverage pilot is inference-only')
    if route.cfg.route_mode != 'full' or not route.cfg.use_continuity:
        counts=torch.zeros_like(counts)
    normalized=route.norm(x)
    fresh=memory is None
    if fresh:
        prompt=normalized[:,:prompt_length]
        memory=(route.k(prompt),route.v(prompt))
        route._coverage_state=None
    key,value=memory
    query=normalized[:,start:]
    scores=torch.bmm(route.q(query).float(),key.float().transpose(1,2))/math.sqrt(route.cfg.route_dim)
    if route.cfg.route_mode=='full' and route.cfg.use_roles:
        scores=scores+torch.bmm(route.role_q(query).float(),route.role_embeddings(roles).float().transpose(1,2))/math.sqrt(route.cfg.route_dim)
    if route.cfg.route_mode=='full' and route.cfg.use_continuity:
        scores=scores+F.softplus(route.gamma.float())*torch.log1p(counts)
    active=torch.ones(query.shape[1],device=x.device,dtype=torch.bool)
    if fresh:
        active=torch.arange(start,x.shape[1],device=x.device)>=prompt_length
    alpha,route._coverage_state=coverage_attention(scores,eligible,route._coverage_strength,route._coverage_state,active)
    context=torch.bmm(alpha.to(value.dtype),value)
    continuity_strength=(alpha*counts).sum(-1,keepdim=True)
    gate=route.gate(torch.cat((query,context.to(query.dtype),continuity_strength.to(query.dtype)),-1)).sigmoid()
    if route.cfg.use_fusion:
        x=torch.cat((x[:,:start],x[:,start:]+route.out(context)*gate),1)
    return x,memory,alpha,context,continuity_strength


def install_coverage(model, strength):
    """Install without changing parameters or state_dict keys. Reset on new prompt.

    One model instance handles one document stream at a time. Native LegalSpan
    already restricts batches to one document. Zero strength preserves its route.
    """
    if strength < 0 or not math.isfinite(strength):
        raise ValueError('Invalid coverage strength')
    found=0
    for block in model.blocks:
        if hasattr(block,'route'):
            route=block.route
            route._coverage_strength=float(strength)
            route._coverage_state=None
            route.forward=types.MethodType(_forward,route)
            found+=1
    if not found: raise ValueError('No LegalSpan source routes found')
    return model
