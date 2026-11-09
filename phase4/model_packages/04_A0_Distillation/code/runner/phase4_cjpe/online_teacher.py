"""Frozen, lazy teacher targets for training IDs only; cached across resumes."""
from pathlib import Path
import json
import numpy as np
import torch
from .common import atomic_json,sha


class CachedTeacher:
 def __init__(self,model,store,allowed_ids,path,fingerprint,device):
  self.model=model.eval().requires_grad_(False);self.store=store;self.allowed=set(allowed_ids)
  self.path=Path(path);self.fingerprint=fingerprint;self.device=device;self.values={}
  if self.path.exists():
   with np.load(self.path,allow_pickle=False) as f:
    if 'fingerprint' in f.files:metadata=json.loads(str(f['fingerprint'].item()))
    else:
     # Older derived caches used a separate manifest; incomplete ones can be rebuilt.
     record=self.path.with_suffix('.json')
     if not record.exists():return
     metadata=json.loads(record.read_text())
     if metadata.get('sha256')!=sha(self.path):return
     metadata=metadata['fingerprint']
    if metadata!=fingerprint:raise ValueError('Teacher cache fingerprint changed')
    self.values={k:v for k,v in zip(f['ids'].tolist(),f['logits'])}
   if not set(self.values)<=self.allowed:raise ValueError('Non-training ID in teacher cache')
 @torch.no_grad()
 def __call__(self,key):
  if key not in self.allowed:raise ValueError('Teacher targets are restricted to training IDs')
  if key not in self.values:
   row=self.store.row('train',key)
   with torch.autocast(device_type=self.device.type,dtype=torch.float16,enabled=self.device.type=='cuda'):
    logits=self.model(row['chunks'],row['positions'],self.device).float()[0]
   if not torch.isfinite(logits).all():raise FloatingPointError('Nonfinite teacher target')
   self.values[key]=logits.detach().cpu().numpy()
  return torch.as_tensor(self.values[key][None],device=self.device,dtype=torch.float32)
 def flush(self):
  if not self.values:return
  self.path.parent.mkdir(parents=True,exist_ok=True);keys=sorted(self.values)
  temp=self.path.with_suffix('.incoming')
  with temp.open('wb') as f:np.savez_compressed(f,ids=np.asarray(keys,dtype=str),logits=np.stack([self.values[k] for k in keys]),fingerprint=np.asarray(json.dumps(self.fingerprint,sort_keys=True)))
  temp.replace(self.path)
  atomic_json(self.path.with_suffix('.json'),{'fingerprint':self.fingerprint,'sha256':sha(self.path),
       'training_cases_only':True,'examples_cached':len(keys),'target_mode':'frozen teacher, lazy per training case'})
