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
  for block in list(backbone.decoder)[-2:]+list(backbone.encoder)[-2:]:block.requires_grad_(True)
  backbone.decoder_norm.requires_grad_(True)
  if hasattr(backbone,'encoder_norm'):backbone.encoder_norm.requires_grad_(True)
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

class EvidenceCaseModel(CaseModel):
 """Learned token evidence and ordered chunk context; own backbone stays intact."""
 def __init__(self,backbone,family,width):
  super().__init__(backbone,family,width)
  hidden=min(128,max(4,width//2))
  self.token_norm=nn.LayerNorm(width)
  self.token_score=nn.Sequential(nn.Linear(width,min(64,width)),nn.Tanh(),nn.Linear(min(64,width),1))
  nn.init.zeros_(self.token_score[-1].weight);nn.init.zeros_(self.token_score[-1].bias)
  self.token_mix=nn.Parameter(torch.zeros(3))
  self.position=nn.Linear(2,width,bias=False);nn.init.zeros_(self.position.weight)
  self.chunk_context=nn.GRU(width,hidden,bidirectional=True,batch_first=True)
  self.context_projection=nn.Linear(2*hidden,width)
  self.context_scale=nn.Parameter(torch.tensor(.1))
  self.context_norm=nn.LayerNorm(width)
  self.document_mix=nn.Parameter(torch.zeros(2))
 def chunk_feature(self,ids):
  values=self.token_features(ids).float()
  values=values[:,1:-1] if values.shape[1]>2 else values
  values=self.token_norm(values)
  attention=self.token_score(values).float().squeeze(-1).softmax(-1)
  summaries=torch.stack((values.mean(1),(values*attention[:,:,None]).sum(1),values[:,-1]),0)
  return (summaries*self.token_mix.softmax(0)[:,None,None]).sum(0)
 def forward(self,chunks,positions,device):
  if not chunks or len(chunks)!=len(positions):raise ValueError('Chunk/position mismatch')
  values=[]
  for ids in chunks:
   x=torch.as_tensor([ids],dtype=torch.long,device=device)
   values.append(checkpoint(self.chunk_feature,x,use_reentrant=False) if self.training else self.chunk_feature(x))
  h=torch.cat(values,0).float()
  pos=torch.as_tensor(positions,dtype=h.dtype,device=device).reshape(-1,1)
  h=h+self.position(torch.cat((pos,1-pos),-1))
  with torch.autocast(device_type=device.type,enabled=False):
   context,_=self.chunk_context(h[None].float())
   h=self.context_norm(h+self.context_scale*self.context_projection(context[0]))
  attention=self.pool(torch.cat((h,pos,1-pos),-1)).float().squeeze(-1).softmax(0)
  summaries=torch.stack(((h*attention[:,None]).sum(0),h.mean(0)),0)
  document=(summaries*self.document_mix.softmax(0)[:,None]).sum(0,keepdim=True)
  return self.head(document)

def make_model(bundle,family,method,rank=8,alpha=16):
 backbone,width,evidence=load_backbone(bundle,family)
 names=configure(backbone,family,method,rank,alpha)
 model=EvidenceCaseModel(backbone,family,width)
 evidence.update(family=family,method=method,rank=rank,alpha=alpha,width=width,
  task_head='token_evidence_chunk_bigru_v2',
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
 model_type=EvidenceCaseModel if e.get('task_head')=='token_evidence_chunk_bigru_v2' else CaseModel
 model=model_type(backbone,f,e['width']);model.load_state_dict(obj['model'],strict=True)
 del obj
 return model.to(device).eval(),e
