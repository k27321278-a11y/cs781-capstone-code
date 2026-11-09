"""LoRMA+ output-side multiplicative adaptation from Bihany et al. (ACL 2025).

For a frozen linear weight W0, computes (I + alpha/rank * B A) W0 x.
The zero B initialization preserves the exact Phase 1 function at step zero.
"""
import torch
from torch import nn
from torch.nn import functional as F


class LoRMAPlusLinear(nn.Module):
    def __init__(self, base: nn.Linear, rank: int = 8, alpha: float = 16):
        super().__init__()
        if rank < 1 or base.bias is not None:
            raise ValueError("Expected positive rank and bias-free base linear")
        self.base = base
        self.rank = rank
        self.scale = alpha / rank
        self.a = nn.Parameter(torch.empty(rank, base.out_features))
        self.b = nn.Parameter(torch.zeros(base.out_features, rank))
        nn.init.kaiming_uniform_(self.a, a=5**.5)

    def forward(self, x):
        frozen_output = self.base(x)
        transformed = F.linear(F.linear(frozen_output, self.a), self.b)
        return frozen_output + transformed * self.scale
