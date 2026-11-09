"""222M grouped-query decoder with actual local-window computation."""
from dataclasses import dataclass,asdict
import math
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint
try:
 from flash_attn import flash_attn_func
except ImportError:
 flash_attn_func=None

@dataclass
class DConfig:
 vocab_size:int=32768
 layers:int=24
 width:int=896
 query_heads:int=14
 kv_heads:int=2
 head_dim:int=64
 ff_width:int=2304
 context:int=2048
 local_window:int=512
 global_every:int=4
 query_tile:int=128
 dropout:float=0.0
 rope_theta:float=10000.0
 checkpointing:bool=True
 def validate(self):
  if self.width!=self.query_heads*self.head_dim or self.query_heads%self.kv_heads:
   raise ValueError('Invalid GQA dimensions')
  if self.head_dim%2 or min(self.layers,self.context,self.local_window,self.query_tile,self.global_every)<1:
   raise ValueError('Invalid context/attention configuration')
  if not 0<=self.dropout<1: raise ValueError('Invalid dropout')
 def parameter_formula(self):
  d=self.width
  return self.vocab_size*d+self.layers*(2*d*d+2*d*self.kv_heads*self.head_dim+3*d*self.ff_width+2*d)+d

class RMSNorm(nn.Module):
 def __init__(self,d):
  super().__init__(); self.weight=nn.Parameter(torch.ones(d))
 def forward(self,x):
  return (x.float()*torch.rsqrt(x.float().square().mean(-1,keepdim=True)+1e-6)).to(x.dtype)*self.weight

def rotary(x,positions,theta):
 inv=1./theta**(torch.arange(0,x.shape[-1],2,device=x.device,dtype=torch.float32)/x.shape[-1])
 angles=positions.float()[:,None]*inv[None,:]
 c,s=angles.cos().to(x.dtype)[None,None],angles.sin().to(x.dtype)[None,None]
 a,b=x[...,::2],x[...,1::2]
 return torch.stack((a*c-b*s,a*s+b*c),-1).flatten(-2)

class GQAAttention(nn.Module):
 def __init__(self,cfg,global_layer):
  super().__init__(); self.cfg=cfg; self.global_layer=global_layer
  self.wq=nn.Linear(cfg.width,cfg.query_heads*cfg.head_dim,bias=False)
  self.wk=nn.Linear(cfg.width,cfg.kv_heads*cfg.head_dim,bias=False)
  self.wv=nn.Linear(cfg.width,cfg.kv_heads*cfg.head_dim,bias=False)
  self.wo=nn.Linear(cfg.width,cfg.width,bias=False)
 def attend(self,q,k,v,qp,kp):
  cfg=self.cfg; drop=cfg.dropout if self.training else 0.
  # A real window kernel where installed; package includes no model weights.
  if (flash_attn_func is not None and q.is_cuda and q.dtype in (torch.float16,torch.bfloat16)
      and len(qp)==len(kp) and torch.equal(qp,kp)):
   return flash_attn_func(q.transpose(1,2),k.transpose(1,2),v.transpose(1,2),dropout_p=drop,
    causal=True,window_size=(-1,-1) if self.global_layer else (cfg.local_window-1,0)).transpose(1,2)
  if self.global_layer:
   if len(qp)==len(kp) and torch.equal(qp,kp):
    return F.scaled_dot_product_attention(q,k,v,is_causal=True,dropout_p=drop,enable_gqa=True)
   mask=kp[None,:]<=qp[:,None]
   return F.scaled_dot_product_attention(q,k,v,attn_mask=mask,dropout_p=drop,enable_gqa=True)
  # Bounded query tiles and KV halos: allocate at most tile*(window+tile), not T*T.
  pieces=[]
  for a in range(0,len(qp),cfg.query_tile):
   b=min(len(qp),a+cfg.query_tile); positions=qp[a:b]
   left=max(0,a-cfg.local_window+1) if len(qp)==len(kp) else 0
   right=b if len(qp)==len(kp) else len(kp)
   keys=kp[left:right]
   mask=(keys[None,:]<=positions[:,None]) & (keys[None,:]>positions[:,None]-cfg.local_window)
   # Materialize grouped heads only for a bounded training tile. This also permits
   # PyTorch's efficient masked-attention backend; the stored cache stays compact.
   repeats=cfg.query_heads//cfg.kv_heads
   tile_k=k[:,:,left:right].repeat_interleave(repeats,dim=1)
   tile_v=v[:,:,left:right].repeat_interleave(repeats,dim=1)
   pieces.append(F.scaled_dot_product_attention(q[:,:,a:b],tile_k,tile_v,
    attn_mask=mask,dropout_p=drop))
  return torch.cat(pieces,2)
 def forward(self,x,cache=None,use_cache=False):
  b,t,_=x.shape; cfg=self.cfg
  start=0 if cache is None else cache['next_position']
  if start+t>cfg.context: raise ValueError('Configured absolute context exceeded')
  if cache is not None and t!=1: raise ValueError('Cached decoding requires one query token')
  qp=torch.arange(start,start+t,device=x.device)
  q=self.wq(x).view(b,t,cfg.query_heads,cfg.head_dim).transpose(1,2)
  k=self.wk(x).view(b,t,cfg.kv_heads,cfg.head_dim).transpose(1,2)
  v=self.wv(x).view(b,t,cfg.kv_heads,cfg.head_dim).transpose(1,2)
  q,k=rotary(q,qp,cfg.rope_theta),rotary(k,qp,cfg.rope_theta); kp=qp
  if cache is not None:
   k=torch.cat((cache['k'],k),2); v=torch.cat((cache['v'],v),2)
   kp=torch.cat((cache['positions'],qp))
  out=self.attend(q,k,v,qp,kp)
  saved=None
  if use_cache:
   keep=len(kp) if self.global_layer else min(cfg.local_window,len(kp))
   saved={'k':k[:,:,-keep:],'v':v[:,:,-keep:],'positions':kp[-keep:],'next_position':start+t}
  return self.wo(out.transpose(1,2).contiguous().view(b,t,cfg.width)),saved

class Block(nn.Module):
 def __init__(self,cfg,index):
  super().__init__(); self.attn_norm=RMSNorm(cfg.width); self.ffn_norm=RMSNorm(cfg.width)
  self.attn=GQAAttention(cfg,(index+1)%cfg.global_every==0)
  self.w1=nn.Linear(cfg.width,cfg.ff_width,bias=False); self.w3=nn.Linear(cfg.width,cfg.ff_width,bias=False)
  self.w2=nn.Linear(cfg.ff_width,cfg.width,bias=False)
 def forward(self,x,cache=None,use_cache=False):
  a,saved=self.attn(self.attn_norm(x),cache,use_cache); x=x+a; h=self.ffn_norm(x)
  return x+self.w2(F.silu(self.w1(h))*self.w3(h)),saved

class EfficientDecoder(nn.Module):
 def __init__(self,cfg):
  super().__init__(); cfg.validate(); self.cfg=cfg
  self.embedding=nn.Embedding(cfg.vocab_size,cfg.width)
  self.blocks=nn.ModuleList(Block(cfg,i) for i in range(cfg.layers)); self.norm=RMSNorm(cfg.width)
  self.lm_head=nn.Linear(cfg.width,cfg.vocab_size,bias=False); self.lm_head.weight=self.embedding.weight
  self.apply(self.initialize)
  for block in self.blocks:
   nn.init.normal_(block.attn.wo.weight,std=.02/math.sqrt(2*cfg.layers))
   nn.init.normal_(block.w2.weight,std=.02/math.sqrt(2*cfg.layers))
 def initialize(self,m):
  if isinstance(m,(nn.Linear,nn.Embedding)): nn.init.normal_(m.weight,std=.02)
 def features(self,ids,caches=None,use_cache=False):
  if ids.ndim!=2 or ids.numel()==0 or ids.shape[1]>self.cfg.context: raise ValueError('Invalid input IDs')
  if caches is not None:
   if len(caches)!=self.cfg.layers or len({c['next_position'] for c in caches})!=1:
    raise ValueError('Inconsistent layer cache positions')
  h=self.embedding(ids); out=[]
  for i,block in enumerate(self.blocks):
   saved=None if caches is None else caches[i]
   if self.training and self.cfg.checkpointing and caches is None and not use_cache:
    h=checkpoint(lambda v,blk=block:blk(v)[0],h,use_reentrant=False); new=None
   else: h,new=block(h,saved,use_cache)
   if use_cache: out.append(new)
  return self.norm(h),out if use_cache else None
 def forward(self,ids,targets=None,caches=None,use_cache=False):
  h,cache=self.features(ids,caches,use_cache); logits=self.lm_head(h); loss=None
  if targets is not None:
   if targets.shape!=ids.shape or not (targets!=-100).any(): raise ValueError('Invalid targets')
   loss=F.cross_entropy(logits.float().reshape(-1,self.cfg.vocab_size),targets.reshape(-1),ignore_index=-100)
  return logits,loss,cache
 def count_parameters(self): return sum(p.numel() for p in self.parameters())
