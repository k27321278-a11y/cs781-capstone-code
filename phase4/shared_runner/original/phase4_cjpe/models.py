"""Four original backbones and six task adaptation strategies.

LoRMA uses an input-side multiplicative low-rank update. Adapters add a
bottleneck residual to FFN output projections. No external pretrained weights.
"""
from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint
from phase3_rr.models import load_backbone,RRClassifier,adapt

class OutputAdapter(nn.Module):
 def __init__(self,base,bottleneck=32):
  super().__init__();self.base=base
  self.down=nn.Linear(base.out_features,bottleneck)
  self.up=nn.Linear(bottleneck,base.out_features)
  nn.init.zeros_(self.up.weight);nn.init.zeros_(self.up.bias)
 def forward(self,x):
  y=self.base(x);return y+self.up(F.gelu(self.down(y)))

def configure(backbone,family,method,rank=8,alpha=16):
 if method=='full':backbone.requires_grad_(True)
 elif method=='lorma':adapt(backbone,'lorma',rank,alpha)
 elif method=='bitfit':
  backbone.requires_grad_(False)
  for name,p in backbone.named_parameters():
   if name.rsplit('.',1)[-1].startswith('bias'):p.requires_grad_(True)
 elif method=='partial':
  if family!='b1':raise ValueError('This partial experiment is defined for B1')
  backbone.requires_grad_(False)
  for block in list(backbone.decoder)[-2:]+list(backbone.encoder)[-1:]:block.requires_grad_(True)
  backbone.decoder_norm.requires_grad_(True)
 elif method=='adapters':
  backbone.requires_grad_(False)
  def walk(module):
   for name,child in list(module.named_children()):
    if name=='w2' and isinstance(child,nn.Linear):setattr(module,name,OutputAdapter(child))
    else:walk(child)
  walk(backbone)
 else:raise ValueError('Unknown adaptation method '+method)
 names=[n for n,p in backbone.named_parameters() if p.requires_grad]
 if not names:raise ValueError('No original-backbone/adaptation parameters trainable')
 return names

class CaseModel(nn.Module):
 def __init__(self,backbone,family,width):
  super().__init__();self.backbone=backbone;self.family=family;self.width=width
  self.pool=nn.Sequential(nn.Linear(width+2,64),nn.Tanh(),nn.Linear(64,1))
  self.head=nn.Sequential(nn.LayerNorm(width),nn.Dropout(.15),nn.Linear(width,2))
 def token_features(self,ids):return RRClassifier.representations(self,ids)
 def chunk_feature(self,ids):
  features=self.token_features(ids)
  # Chunks are unpadded, each encoded separately; all original tokens participate.
  return features.float().mean(1)
 def forward(self,chunks,positions,device):
  values=[]
  for ids in chunks:
   x=torch.as_tensor([ids],dtype=torch.long,device=device)
   if self.training:
    # Non-reentrant recomputation also preserves gradients with frozen embeddings.
    value=checkpoint(self.chunk_feature,x,use_reentrant=False)
   else:value=self.chunk_feature(x)
   values.append(value)
  h=torch.cat(values,0)
  pos=torch.as_tensor(positions,dtype=h.dtype,device=device).reshape(-1,1)
  scores=self.pool(torch.cat((h,pos,1-pos),-1)).float().squeeze(-1)
  weights=scores.softmax(0)
  return self.head((h*weights[:,None]).sum(0,keepdim=True))

def make_model(bundle,family,method,rank=8,alpha=16):
 backbone,width,evidence=load_backbone(bundle,family)
 names=configure(backbone,family,method,rank,alpha)
 model=CaseModel(backbone,family,width)
 evidence.update(family=family,method=method,rank=rank,alpha=alpha,width=width,
  trainable_backbone_names=names,parameters=sum(p.numel() for p in model.parameters()),
  trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad))
 return model,evidence

def restore(path,device='cuda'):
 obj=torch.load(path,map_location='cpu',weights_only=False);e=obj['evidence'];cfg=e['model_config'];f=e['family']
 if f=='d222':
  from legal_d.model import DConfig,EfficientDecoder
  backbone=EfficientDecoder(DConfig(**cfg))
 elif f=='a0':
  from cs781.contracts import ModelConfig
  from cs781.models.a import LegalGPT
  backbone=LegalGPT(ModelConfig(**cfg))
 elif f=='b1':
  from cs781.models.b import EncoderDecoderConfig,EncoderDecoderLegalModel
  backbone=EncoderDecoderLegalModel(EncoderDecoderConfig(**cfg))
 elif f=='c1':
  from cs781.models.c import RecurrentConfig,RecurrentLegalModel
  backbone=RecurrentLegalModel(RecurrentConfig(**cfg))
 else:raise ValueError('Unknown saved family')
 configure(backbone,f,e['method'],e['rank'],e['alpha'])
 model=CaseModel(backbone,f,e['width']);model.load_state_dict(obj['model'],strict=True)
 del obj
 return model.to(device).eval(),e
