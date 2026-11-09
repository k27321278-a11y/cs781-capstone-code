import hashlib,json,os,random
from pathlib import Path
import numpy as np
import torch

def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()

def atomic_json(path,obj):
 path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
 tmp=path.with_suffix(path.suffix+'.tmp')
 tmp.write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8');os.replace(tmp,path)

def save(path,obj):
 path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
 tmp=path.with_suffix(path.suffix+'.tmp');torch.save(obj,tmp);os.replace(tmp,path)

def seed_all(seed):
 random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
 if torch.cuda.is_available():torch.cuda.manual_seed_all(seed)

def source_hashes():
 root=Path(__file__).parent
 return {p.relative_to(root).as_posix():sha(p) for p in sorted(root.rglob('*')) if p.is_file() and (p.suffix=='.py' or p.name in ('UnicodeData-13.0.0.txt','UnicodeData-14.0.0.txt')) and '__pycache__' not in p.parts}

def hardware():
 return {'torch':torch.__version__,'gpu':torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU','cuda':torch.version.cuda}
