"""Atomic training state, implementation provenance, mixed precision and deadlines."""
import contextlib,json,os,time
from datetime import datetime,timezone
from pathlib import Path
import torch
from lner.data import sha256,atomic_json

def save(path,obj):
 path=Path(path); path.parent.mkdir(parents=True,exist_ok=True); temp=path.with_suffix('.partial')
 with temp.open('wb') as f:
  torch.save(obj,f); f.flush(); os.fsync(f.fileno())
 os.replace(temp,path)

def source_hashes():
 root=Path(__file__).resolve().parents[1]
 return {str(p.relative_to(root)):sha256(p) for folder in ('legal_d','vendor')
         for p in sorted((root/folder).rglob('*.py'))}

def autocast(device):
 if device.type!='cuda': return contextlib.nullcontext()
 return torch.autocast('cuda',dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16)

def deadline(config,reserve_hours=0):
 return datetime.fromisoformat(config['pretrain']['deadline_utc']).timestamp()-reserve_hours*3600

def runtime_evidence(device):
 return {'torch':torch.__version__,'device':str(device),'gpu':torch.cuda.get_device_name(device) if device.type=='cuda' else None,
         'cuda_visible_devices':os.environ.get('CUDA_VISIBLE_DEVICES'),'started_utc':datetime.now(timezone.utc).isoformat()}
