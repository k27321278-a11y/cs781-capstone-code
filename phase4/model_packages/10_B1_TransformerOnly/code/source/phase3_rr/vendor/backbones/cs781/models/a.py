"""Dense causal LegalGPT; graph preserved from the verified A0 implementation."""
import math

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from ..contracts import ModelConfig


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        values = x.float()
        return (values * torch.rsqrt(values.square().mean(-1,keepdim=True) + self.eps)).to(x.dtype) * self.weight


class RotaryEmbedding(nn.Module):
    def __init__(self, head_dim, max_seq_len, theta=10000.0):
        super().__init__()
        inv = 1.0 / theta ** (torch.arange(0,head_dim,2,dtype=torch.float32) / head_dim)
        frequencies = torch.outer(torch.arange(max_seq_len,dtype=torch.float32),inv)
        self.register_buffer("cos",frequencies.cos(),persistent=False)
        self.register_buffer("sin",frequencies.sin(),persistent=False)

    def apply_rotary(self, x, offset=0):
        t = x.size(-2)
        if offset+t > self.cos.size(0):
            raise ValueError("RoPE context exhausted")
        cos = self.cos[offset:offset+t].to(dtype=x.dtype)[None,None]
        sin = self.sin[offset:offset+t].to(dtype=x.dtype)[None,None]
        even, odd = x[...,0::2], x[...,1::2]
        return torch.stack((even*cos-odd*sin,even*sin+odd*cos),dim=-1).flatten(-2)


class CausalMHA(nn.Module):
    def __init__(self,cfg,rope):
        super().__init__()
        self.cfg,self.rope = cfg,rope
        self.wqkv = nn.Linear(cfg.d_model,3*cfg.d_model,bias=False)
        self.wo = nn.Linear(cfg.d_model,cfg.d_model,bias=False)

    def forward(self,x,past_kv=None,use_cache=False):
        b,t,d = x.shape
        q,k,v = [a.view(b,t,self.cfg.n_heads,self.cfg.d_head).transpose(1,2) for a in self.wqkv(x).chunk(3,dim=-1)]
        offset = 0 if past_kv is None else past_kv[0].size(-2)
        q,k = self.rope.apply_rotary(q,offset),self.rope.apply_rotary(k,offset)
        if past_kv is not None:
            if t != 1:
                raise ValueError("Cached decoding accepts one token at a time")
            k,v = torch.cat((past_kv[0],k),dim=-2),torch.cat((past_kv[1],v),dim=-2)
        # A newest single query sees every cached key; a square causal mask would be wrong.
        output = F.scaled_dot_product_attention(q,k,v,is_causal=past_kv is None,
                                                dropout_p=self.cfg.dropout if self.training else 0.0)
        output = self.wo(output.transpose(1,2).contiguous().view(b,t,d))
        return output, (k,v) if use_cache else None


class SwiGLU(nn.Module):
    def __init__(self,cfg):
        super().__init__()
        self.w1 = nn.Linear(cfg.d_model,cfg.d_ff,bias=False)
        self.w3 = nn.Linear(cfg.d_model,cfg.d_ff,bias=False)
        self.w2 = nn.Linear(cfg.d_ff,cfg.d_model,bias=False)

    def forward(self,x):
        return self.w2(F.silu(self.w1(x))*self.w3(x))


class TransformerBlock(nn.Module):
    def __init__(self,cfg,rope):
        super().__init__()
        self.attn_norm,self.ffn_norm = RMSNorm(cfg.d_model),RMSNorm(cfg.d_model)
        self.attn,self.ffn = CausalMHA(cfg,rope),SwiGLU(cfg)

    def forward(self,x,past_kv=None,use_cache=False):
        output,cache = self.attn(self.attn_norm(x),past_kv,use_cache)
        x = x + output
        return x+self.ffn(self.ffn_norm(x)),cache


class LegalGPT(nn.Module):
    def __init__(self,cfg: ModelConfig):
        super().__init__()
        cfg.validate()
        self.cfg = cfg
        self.tok_embeddings = nn.Embedding(cfg.vocab_size,cfg.d_model)
        self.rope = RotaryEmbedding(cfg.d_head,cfg.max_seq_len,cfg.rope_theta)
        self.blocks = nn.ModuleList([TransformerBlock(cfg,self.rope) for _ in range(cfg.n_layers)])
        self.norm = RMSNorm(cfg.d_model)
        self.lm_head = nn.Linear(cfg.d_model,cfg.vocab_size,bias=False)
        self.lm_head.weight = self.tok_embeddings.weight
        self.gradient_checkpointing = cfg.gradient_checkpointing
        self.apply(self._init_weights)
        std = .02/math.sqrt(2*cfg.n_layers)
        for block in self.blocks:
            nn.init.normal_(block.attn.wo.weight,std=std)
            nn.init.normal_(block.ffn.w2.weight,std=std)

    @staticmethod
    def _init_weights(module):
        if isinstance(module,(nn.Linear,nn.Embedding)):
            nn.init.normal_(module.weight,std=.02)

    def forward(self,input_ids,targets=None,past_kvs=None,use_cache=False):
        if input_ids.ndim != 2 or not input_ids.numel():
            raise ValueError("Expected nonempty [batch,time] token IDs")
        b,t = input_ids.shape
        if past_kvs is not None:
            if len(past_kvs) != self.cfg.n_layers or t != 1:
                raise ValueError("Invalid cache length or query length")
            lengths = set()
            for pair in past_kvs:
                if len(pair)!=2 or any(not isinstance(a,torch.Tensor) or a.ndim!=4 for a in pair):
                    raise ValueError("Invalid cache tensors")
                k,v = pair
                if k.shape!=v.shape or k.shape[:2]!=(b,self.cfg.n_heads) or k.shape[-1]!=self.cfg.d_head or k.shape[-2]<1:
                    raise ValueError("Invalid cache shape")
                if k.device!=input_ids.device or v.device!=input_ids.device or k.dtype!=v.dtype:
                    raise ValueError("Cache device/dtype mismatch")
                lengths.add(k.shape[-2])
            if len(lengths)!=1:
                raise ValueError("Inconsistent layer cache lengths")
        past_len = 0 if past_kvs is None else past_kvs[0][0].size(-2)
        if past_len+t > self.cfg.max_seq_len:
            raise ValueError("Sequence exceeds configured context")
        x,caches = self.tok_embeddings(input_ids),[]
        for i,block in enumerate(self.blocks):
            kv = None if past_kvs is None else past_kvs[i]
            if self.training and self.gradient_checkpointing and not use_cache and kv is None:
                x = checkpoint(lambda inp,blk=block: blk(inp)[0],x,use_reentrant=False)
                cache = None
            else:
                x,cache = block(x,kv,use_cache)
            if use_cache:
                caches.append(cache)
        logits = self.lm_head(self.norm(x))
        loss = None
        if targets is not None:
            if targets.shape!=input_ids.shape or not (targets != -100).any():
                raise ValueError("Invalid or entirely ignored targets")
            loss = F.cross_entropy(logits.float().reshape(-1,self.cfg.vocab_size),targets.reshape(-1),ignore_index=-100)
        return logits,loss,caches if use_cache else None

    def count_parameters(self):
        return sum(p.numel() for p in {id(p):p for p in self.parameters()}.values())
