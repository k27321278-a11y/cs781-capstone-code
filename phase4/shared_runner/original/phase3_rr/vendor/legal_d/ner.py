"""Learned BIO CRF and original-character NER with auxiliary span boundaries."""
import torch
from torch import nn
from torch.nn import functional as F
from .model import DConfig,EfficientDecoder
from .common import autocast
from lner.models import inject_adapters

class LinearCRF(nn.Module):
 def __init__(self,classes):
  super().__init__(); self.transitions=nn.Parameter(torch.zeros(classes,classes))
  self.start=nn.Parameter(torch.zeros(classes)); self.end=nn.Parameter(torch.zeros(classes))
  allowed=torch.ones(classes,classes,dtype=torch.bool)
  for tag in range(2,classes,2):
   allowed[:,tag]=False; allowed[tag-1,tag]=allowed[tag,tag]=True
  starts=torch.ones(classes,dtype=torch.bool); starts[2::2]=False
  self.register_buffer('allowed',allowed); self.register_buffer('start_allowed',starts)
 def scores(self):
  return self.transitions.masked_fill(~self.allowed,-1e4),self.start.masked_fill(~self.start_allowed,-1e4)
 def nll(self,emissions,tags):
  # One variable-length character window. Repair only the left-edge truncated I tag.
  original_device=emissions.device
  # The sequential 25-state DP is small. CPU avoids thousands of tiny CUDA
  # launches; differentiable copies still send gradients to the GPU model.
  emissions=emissions.float().cpu(); tags=tags.clone().cpu()
  if int(tags[0])>0 and int(tags[0])%2==0: tags[0]-=1
  transition,start=self.scores(); transition,start=transition.cpu(),start.cpu(); end=self.end.cpu()
  score=start[tags[0]]+emissions[0,tags[0]]
  score=score+emissions[1:].gather(1,tags[1:,None]).sum()+transition[tags[:-1],tags[1:]].sum()+end[tags[-1]]
  alpha=start+emissions[0]
  for row in emissions[1:]: alpha=torch.logsumexp(alpha[:,None]+transition,0)+row
  partition=torch.logsumexp(alpha+end,0)
  return ((partition-score)/len(tags)).to(original_device)

class DNER(nn.Module):
 def __init__(self,backbone,char_vocab,classes,dropout=.1):
  super().__init__(); self.backbone=backbone; self.char_embedding=nn.Embedding(char_vocab,32)
  self.char_context=nn.GRU(backbone.cfg.width+32,128,batch_first=True,bidirectional=True)
  self.dropout=nn.Dropout(dropout); self.classifier=nn.Linear(256,classes)
  self.boundaries=nn.Linear(256,2); self.crf=LinearCRF(classes)
 def emissions(self,ids,mapping,chars,roles=None):
  hidden,_=self.backbone.features(ids)
  values=hidden.gather(1,mapping[:,:,None].expand(-1,-1,hidden.shape[-1]))
  values=torch.cat((values,self.char_embedding(chars)),dim=-1)
  with torch.autocast(device_type=ids.device.type,enabled=False): values,_=self.char_context(values.float())
  values=self.dropout(values)
  return self.classifier(values),self.boundaries(values)
 def forward(self,ids,mapping,chars,roles=None):
  return self.emissions(ids,mapping,chars,roles)[0]
 def loss(self,args,tags,class_weights,boundary_weight=.15):
  emissions,boundaries=self.emissions(*args); emissions=emissions[0].float(); boundaries=boundaries[0].float()
  ce=F.cross_entropy(emissions,tags,weight=class_weights,label_smoothing=.01)
  sequence=self.crf.nll(emissions,tags)
  starts=(tags>0)&(tags%2==1)
  after=torch.cat((tags[1:],torch.zeros(1,dtype=tags.dtype,device=tags.device)))
  kind=(tags+1)//2; next_kind=(after+1)//2
  ends=(tags>0)&((kind!=next_kind)|((after>0)&(after%2==1)))
  truth=torch.stack((starts,ends),1).float()
  boundary=F.binary_cross_entropy_with_logits(boundaries,truth,pos_weight=torch.full((2,),8.,device=tags.device))
  return .5*ce+.5*sequence+boundary_weight*boundary

def load_ner(checkpoint,char_vocab,classes,candidate,rank=8,alpha=32):
 payload=torch.load(checkpoint,map_location='cpu',weights_only=False)
 if payload['state']['target_tokens']<=0 or payload['evidence']['initialization']!='random_from_scratch':
  raise ValueError('A real own-pretrained D checkpoint is required')
 backbone=EfficientDecoder(DConfig(**payload['evidence']['model_config']))
 backbone.load_state_dict(payload['model'],strict=True)
 if candidate['method']=='lorma': inject_adapters(backbone,'lorma',rank,alpha,.05)
 model=DNER(backbone,char_vocab,classes,candidate['dropout'])
 evidence={'pretrained_state':payload['state'],'model_config':payload['evidence']['model_config'],
           'method':candidate['method'],'parameters':sum(p.numel() for p in model.parameters()),
           'trainable_parameters':sum(p.numel() for p in model.parameters() if p.requires_grad)}
 return model,evidence
