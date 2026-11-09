"""Own-model sentence classifiers and from-scratch LoRA, LoRMA and IA3.

References: Hu et al.2021 (LoRA); Bihany/Patel/Modi2025 (LoRMA);
Liu et al.2022, Few-Shot Parameter-Efficient Fine-Tuning (IA3).
"""
import json,math,sys
from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint
from .common import sha
ROOT=Path(__file__).parent
for p in (ROOT/'vendor',ROOT/'vendor/backbones'):
 if str(p) not in sys.path:sys.path.insert(0,str(p))

class LowRank(nn.Module):
 def __init__(self,base,method,rank=8,alpha=16):
  super().__init__();self.base=base;self.method=method;self.scale=alpha/rank
  self.down=nn.Linear(base.in_features,rank,bias=False)
  self.up=nn.Linear(rank,base.in_features if method=='lorma' else base.out_features,bias=False)
  self.dropout=nn.Dropout(.05)
  if method=='lorma':nn.init.zeros_(self.down.weight);nn.init.kaiming_uniform_(self.up.weight,a=math.sqrt(5))
  else:nn.init.kaiming_uniform_(self.down.weight,a=math.sqrt(5));nn.init.zeros_(self.up.weight)
 def forward(self,x):
  update=self.up(self.down(self.dropout(x)))*self.scale
  return self.base(x+update) if self.method=='lorma' else self.base(x)+update

class IA3(nn.Module):
 def __init__(self,base,name):
  super().__init__();self.base=base;self.input_scale=name=='w2';self.fused=name=='wqkv'
  n=base.in_features if self.input_scale else base.out_features
  if self.fused:
   if n%3:raise ValueError('Fused qkv shape mismatch')
   self.scale=nn.Parameter(torch.ones(2*n//3))
  else:self.scale=nn.Parameter(torch.ones(n))
 def forward(self,x):
  if self.input_scale:return self.base(x*self.scale.to(x.dtype))
  y=self.base(x)
  scale=torch.cat((torch.ones(y.shape[-1]//3,device=y.device),self.scale)) if self.fused else self.scale
  return y*scale.to(y.dtype)

def adapt(model,method,rank,alpha):
 if method=='full':return []
 model.requires_grad_(False);paths=[]
 low={'wqkv','wo','wq','wk','wv','w1','w2','w3'}
 ia={'wqkv','wk','wv','w2'}
 def walk(module,prefix=''):
  for name,child in list(module.named_children()):
   if isinstance(child,nn.Linear) and name in (ia if method=='ia3' else low):
    if method not in ('lora','lorma','ia3'):raise ValueError('Unknown adaptation')
    setattr(module,name,IA3(child,name) if method=='ia3' else LowRank(child,method,rank,alpha));paths.append(prefix+name)
   else:walk(child,prefix+name+'.')
 walk(model)
 if not paths:raise ValueError('No actual modules adapted')
 return paths

def load_backbone(bundle,family):
 payload=torch.load(Path(bundle)/'checkpoint.pt',map_location='cpu',weights_only=False)
 if family=='d222':
  from legal_d.model import DConfig,EfficientDecoder
  cfg=dict(payload['evidence']['model_config']);m=EfficientDecoder(DConfig(**cfg));width=cfg['width']
  evidence=payload['evidence'];digest=evidence['data'].get('tokenizer_sha256')
 else:
  from cs781.contracts import ModelConfig
  from cs781.models.a import LegalGPT
  from cs781.models.b import EncoderDecoderConfig,EncoderDecoderLegalModel
  from cs781.models.c import RecurrentConfig,RecurrentLegalModel
  evidence=payload['metadata'];cfg=dict(evidence['model_config']);digest=evidence.get('tokenizer_sha256');width=cfg['d_model']
  if family=='a0':m=LegalGPT(ModelConfig(**cfg))
  elif family=='b1':m=EncoderDecoderLegalModel(EncoderDecoderConfig(**cfg))
  elif family=='c1':m=RecurrentLegalModel(RecurrentConfig(**cfg))
  else:raise ValueError('Unknown family')
 if digest and digest!=sha(Path(bundle)/'tokenizer.json'):raise ValueError('Tokenizer differs from pretrained checkpoint')
 m.load_state_dict(payload['model'],strict=True)
 state=payload.get('state',{});del payload
 return m,width,{'model_config':cfg,'parent_state':state,'tokenizer_sha256':sha(Path(bundle)/'tokenizer.json'),'parent_sha256':sha(Path(bundle)/'checkpoint.pt')}

class RRClassifier(nn.Module):
 def __init__(self,backbone,family,width,classes=13):
  super().__init__();self.backbone=backbone;self.family=family
  self.head=nn.Sequential(nn.LayerNorm(width+2),nn.Dropout(.15),nn.Linear(width+2,classes))
 def representations(self,ids):
  m=self.backbone
  if self.family=='d222':return m.features(ids)[0]
  if self.family=='a0':
   x=m.tok_embeddings(ids)
   for block in m.blocks:
    x=checkpoint(lambda v,blk=block:blk(v)[0],x,use_reentrant=False) if self.training and m.cfg.gradient_checkpointing else block(x)[0]
   return m.norm(x)
  memory,valid=m.encode(ids,torch.ones_like(ids,dtype=torch.bool))
  if self.family=='b1':
   x=m.tok_embeddings(ids)
   for block in m.decoder:
    x=checkpoint(lambda v,blk=block:blk(v,memory,valid)[0],x,use_reentrant=False) if self.training and m.cfg.gradient_checkpointing else block(x,memory,valid)[0]
   return m.decoder_norm(x)
  initial=torch.tanh(m.initial_projection(memory.mean(1))).float()
  initial=initial[None].expand(m.cfg.n_gru_layers,-1,-1).contiguous()
  with torch.autocast(device_type=ids.device.type,enabled=False):values,_=m.gru(m.tok_embeddings(ids).float(),initial)
  # PyTorch GRU output is FP32 for T4 stability; attention remains autocast.
  update,_=m.cross_attn(m.cross_norm(values),valid,memory=memory)
  x=values+update
  return m.output_norm(x+m.fusion(m.fusion_norm(x)))
 def forward(self,ids,mask,position):
  features=self.representations(ids)
  w=mask.to(features.dtype);pooled=(features*w[:,:,None]).sum(1)/w.sum(1,keepdim=True).clamp_min(1)
  pos=position.to(pooled.dtype).reshape(-1,1)
  return self.head(torch.cat((pooled,pos,1-pos),1))

def make_model(bundle,family,method,rank=8,alpha=16):
 backbone,width,evidence=load_backbone(bundle,family)
 evidence['adapted_modules']=adapt(backbone,method,rank,alpha)
 model=RRClassifier(backbone,family,width)
 evidence.update({'family':family,'method':method,'rank':rank,'alpha':alpha,
  'parameters':sum(p.numel() for p in model.parameters()),'trainable_parameters':sum(p.numel() for p in model.parameters() if p.requires_grad)})
 return model,evidence

def restore_selected(path,device='cuda'):
 """Restore the complete selected RR state without needing the original parent file."""
 payload=torch.load(path,map_location='cpu',weights_only=False);evidence=payload['evidence']
 family=evidence['family'];cfg=evidence['model_config']
 if family=='d222':
  from legal_d.model import DConfig,EfficientDecoder
  backbone=EfficientDecoder(DConfig(**cfg));width=cfg['width']
 elif family=='a0':
  from cs781.models.a import LegalGPT
  from cs781.contracts import ModelConfig
  backbone=LegalGPT(ModelConfig(**cfg));width=cfg['d_model']
 elif family=='b1':
  from cs781.models.b import EncoderDecoderConfig,EncoderDecoderLegalModel
  backbone=EncoderDecoderLegalModel(EncoderDecoderConfig(**cfg));width=cfg['d_model']
 elif family=='c1':
  from cs781.models.c import RecurrentConfig,RecurrentLegalModel
  backbone=RecurrentLegalModel(RecurrentConfig(**cfg));width=cfg['d_model']
 else:raise ValueError('Unknown selected RR model family')
 adapt(backbone,evidence['method'],evidence['rank'],evidence['alpha'])
 model=RRClassifier(backbone,family,width);model.load_state_dict(payload['model'],strict=True)
 del payload
 return model.to(device).eval(),evidence

def tensors(row,device):
 return (torch.tensor([row['ids']],device=device),torch.tensor([row['mask']],device=device),torch.tensor([row['position']],device=device))
