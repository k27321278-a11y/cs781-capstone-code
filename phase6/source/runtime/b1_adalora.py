"""B1 passage fusion with a gradually allocated additive low-rank budget."""
import math

import torch
from torch import nn
from torch.nn import functional as F

import b1_fusion


class AdaLoRALinear(nn.Module):
    def __init__(self, base, max_rank=16, target_rank=8, alpha=16):
        super().__init__()
        if not isinstance(base, nn.Linear) or min(max_rank, target_rank) < 1:
            raise ValueError('AdaLoRA requires a linear base and positive ranks')
        if target_rank > max_rank:
            raise ValueError('Target rank cannot exceed maximum rank')
        self.base = base
        self.base.requires_grad_(False)
        self.max_rank, self.target_rank = max_rank, target_rank
        self.scale = alpha / target_rank
        self.a = nn.Parameter(base.weight.new_empty(max_rank, base.in_features))
        self.b = nn.Parameter(base.weight.new_zeros(base.out_features, max_rank))
        nn.init.kaiming_uniform_(self.a, a=math.sqrt(5))
        # These values are part of the adapter checkpoint. They are adjusted
        # by the rank allocator, never by optimizer gradients.
        self.rank_mask = nn.Parameter(base.weight.new_ones(max_rank))
        self.importance = nn.Parameter(base.weight.new_zeros(max_rank))

    def forward(self, x):
        hidden = F.linear(x, self.a) * self.rank_mask.detach()
        return self.base(x) + self.scale * F.linear(hidden, self.b)

    @torch.no_grad()
    def update_importance(self):
        if self.a.grad is None or self.b.grad is None:
            return
        sensitivity = (self.a.norm(dim=1) * self.b.grad.norm(dim=0)
                       + self.b.norm(dim=0) * self.a.grad.norm(dim=1))
        self.importance.mul_(0.9).add_(sensitivity, alpha=0.1)


@torch.no_grad()
def allocate_ranks(layers, budget, min_rank=2):
    layers = list(layers)
    if not layers or min_rank < 1 or budget < len(layers) * min_rank:
        raise ValueError('Infeasible AdaLoRA rank budget')
    available = sum(layer.max_rank for layer in layers)
    if budget > available:
        raise ValueError('Rank budget exceeds capacity')
    chosen = []
    remaining = []
    for layer_index, layer in enumerate(layers):
        order = sorted(range(layer.max_rank),
                       key=lambda rank: (-float(layer.importance[rank]), rank))
        chosen.extend((layer_index, rank) for rank in order[:min_rank])
        remaining.extend((float(layer.importance[rank]), layer_index, rank)
                         for rank in order[min_rank:])
    remaining.sort(key=lambda item: (-item[0], item[1], item[2]))
    chosen.extend((i, r) for _, i, r in remaining[:budget-len(chosen)])
    for layer in layers:
        layer.rank_mask.zero_()
    for layer_index, rank in chosen:
        layers[layer_index].rank_mask[rank] = 1


def build(input_dir, device):
    model, tokenizer, metadata = b1_fusion.build(input_dir, device)
    layers = []
    for name, module in list(model.backbone.named_modules()):
        if not isinstance(module, b1_fusion.LoRMAOutputLinear):
            continue
        parent_name, child_name = name.rsplit('.', 1)
        new_layer = AdaLoRALinear(module.base)
        setattr(model.backbone.get_submodule(parent_name), child_name, new_layer)
        layers.append(new_layer)
    if len(layers) != len(model.adapter_targets):
        raise RuntimeError('B1 adapter target count changed')
    model.rank = 16
    metadata['method'] = 'B1 independent passage fusion with AdaLoRA'
    metadata['adaptation'] = {
        'type': 'AdaLoRA', 'formula': 'W x + alpha/target_rank B diag(mask) A x',
        'max_rank': 16, 'target_average_rank': 8, 'min_rank': 2,
        'importance': 'EMA of weight-norm times gradient-norm per rank',
        'allocation_start_update': 100, 'allocation_interval': 50,
        'targets': model.adapter_targets,
        'additional_parameters': 'passage position embeddings',
    }
    metadata['trainable_parameters'] = sum(p.numel() for p in model.parameters() if p.requires_grad)
    metadata['total_parameters'] = sum(p.numel() for p in model.parameters())
    metadata['frozen_parameters'] = metadata['total_parameters'] - metadata['trainable_parameters']
    return model, tokenizer, metadata


def on_optimizer_step(model, updates):
    layers = [m for m in model.modules() if isinstance(m, AdaLoRALinear)]
    for layer in layers:
        layer.update_importance()
    if updates >= 100 and updates % 50 == 0:
        allocate_ranks(layers, budget=len(layers)*8, min_rank=2)


training_loss = b1_fusion.training_loss
generate = b1_fusion.generate
