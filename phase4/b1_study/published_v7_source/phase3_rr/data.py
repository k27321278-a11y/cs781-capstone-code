import bisect,hashlib,json,zipfile
from pathlib import Path,PurePosixPath
import numpy as np
from .common import sha,atomic_json

LABELS=['Fact','Issue','ArgumentPetitioner','ArgumentRespondent','Statute','Dissent','PrecedentReliedUpon','PrecedentNotReliedUpon','PrecedentOverruled','RulingByLowerCourt','RatioOfTheDecision','RulingByPresentCourt','None']

def acquire_data(folder,uploaded_zip=None,download=False):
 folder=Path(folder)
 if uploaded_zip:
  with zipfile.ZipFile(uploaded_zip) as z:
   for info in z.infolist():
    p=PurePosixPath(info.filename)
    if p.is_absolute() or '..' in p.parts:raise ValueError('Unsafe data ZIP')
    if info.is_dir():continue
    if p.name in {'CL_train.json','CL_dev.json','IT_train.json','IT_dev.json','label_vocab.json','CL_test.json','IT_test.json'}:
     part='Test' if p.name in ('CL_test.json','IT_test.json') else 'Train'
     dest=folder/part/p.name;dest.parent.mkdir(parents=True,exist_ok=True)
     data=z.read(info)
     if dest.exists():
      if dest.read_bytes()!=data:raise ValueError('Different existing data file: '+str(dest)+'; use a new DATA_ROOT')
     else:dest.write_bytes(data)
 if download:
  from huggingface_hub import hf_hub_download
  for part,names in {'Train':['CL_train.json','CL_dev.json','IT_train.json','IT_dev.json','label_vocab.json'],'Test':['CL_test.json','IT_test.json']}.items():
   for name in names:
    path=folder/part/name
    if not path.exists():
     import shutil
     source=hf_hub_download('Exploration-Lab/CS-781-Capstone',filename=f'Phase_3_RR/{part}/{name}',repo_type='dataset')
     path.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,path)
 return folder

def documents(path,domain,labeled):
 obj=json.loads(Path(path).read_text(encoding='utf-8-sig'))
 if isinstance(obj,dict):
  if 'data' in obj and isinstance(obj['data'],list):rows=obj['data']
  elif 'id' in obj and 'text' in obj:rows=[obj]
  else:rows=[{'id':key,**value} if isinstance(value,dict) else {'id':key,'text':value} for key,value in obj.items()]
 elif isinstance(obj,list):rows=obj
 else:raise ValueError('Unsupported RR file container')
 result=[];seen=set()
 for row in rows:
  if not isinstance(row,dict) or 'id' not in row or 'text' not in row:raise ValueError('Expected document id/text fields; inspect actual supplied data')
  key=str(row['id']);sentences=row['text'];labels=row.get('labels')
  if key in seen:raise ValueError('Duplicate document ID '+key)
  seen.add(key)
  if not isinstance(sentences,list) or not all(isinstance(x,str) for x in sentences):raise ValueError('RR text must be ordered list[str]')
  if labeled and (not isinstance(labels,list) or len(labels)!=len(sentences) or any(x not in LABELS for x in labels)):raise ValueError('Sentence/label mismatch or unknown label')
  result.append({'id':key,'domain':domain,'text':sentences,'labels':labels if labeled else None})
 return result

def load_official(folder):
 folder=Path(folder);vocab=json.loads((folder/'Train/label_vocab.json').read_text(encoding='utf-8-sig'))
 from .vendor.evaluate_phase3 import load_vocab,normalize_id
 if set(load_vocab(vocab))!=set(LABELS):raise ValueError('Official vocabulary differs from documented13 labels')
 ds={}
 for split in ('train','dev','test'):
  ds[split]=sum((documents(folder/('Test' if split=='test' else 'Train')/f'{d}_{split}.json',d,split!='test') for d in ('CL','IT')),[])
  ids=[x['id'] for x in ds[split]]
  if len({normalize_id(x) for x in ids})!=len(ids):raise ValueError('CL/IT document IDs collide after official .txt normalization')
 allids={k:{r['id'] for r in v} for k,v in ds.items()}
 if any(allids[a]&allids[b] for a,b in (('train','dev'),('train','test'),('dev','test'))):raise ValueError('Official splits overlap in document IDs; inspect data before training')
 ds['fingerprint']={p.relative_to(folder).as_posix():sha(p) for p in sorted(folder.rglob('*.json'))}
 return ds

class Tokenizer:
 def __init__(self,path):
  self.path=Path(path);obj=json.loads(self.path.read_text(encoding='utf-8'))
  if obj.get('format')=='cs781-own-byte-bpe-v1':
   from cs781 import own_bpe
   # Pin the saved tokenizer's original L/M/Nd classes across Python versions.
   version=obj.get('unicode_version')
   if version not in ('13.0.0','14.0.0'):raise ValueError('Unsupported original Unicode version: '+str(version))
   table=Path(__file__).parent/('vendor/UnicodeData-'+version+'.txt')
   starts=[];ends=[];cats=[];pending=None
   for line in table.read_text(encoding='utf-8').splitlines():
    fields=line.split(';');code=int(fields[0],16);name,cat=fields[1:3]
    if name.endswith(', First>'):pending=code;continue
    first=pending if name.endswith(', Last>') else code;pending=None
    if cat[0] in 'LM' or cat=='Nd':starts.append(first);ends.append(code);cats.append(cat)
   class UnicodeTable:
    unidata_version=version
    @staticmethod
    def category(c):
     i=bisect.bisect_right(starts,ord(c))-1
     return cats[i] if i>=0 and ord(c)<=ends[i] else 'Cn'
   own_bpe.unicodedata=UnicodeTable
   self.impl=own_bpe.OwnBPE.load(path)
  else:
   from tokenizers import Tokenizer as HFTokenizer
   self.impl=HFTokenizer.from_file(str(path))
 def encode(self,text):return self.impl.encode(text).ids

def prepare_examples(docs,tok,max_tokens=256,context_sentences=1):
 result=[];truncated=0
 for doc in docs:
  for i,sentence in enumerate(doc['text']):
   target=tok.encode(sentence)
   if len(target)>max_tokens-2:truncated+=1
   # Deterministic head+tail coverage for long sentences; report truncation, never omit labels.
   if len(target)>max_tokens-2:
    n=(max_tokens-2)//2;target=target[:n]+target[-(max_tokens-2-n):]
   if not target:target=[3]
   budget=max_tokens-2-len(target)
   left=tok.encode(' '.join(doc['text'][max(0,i-context_sentences):i])) if context_sentences else []
   right=tok.encode(' '.join(doc['text'][i+1:i+1+context_sentences])) if context_sentences else []
   l=min(len(left),budget//2);r=min(len(right),budget-l);l=min(len(left),budget-r)
   before=left[-l:] if l else [];after=right[:r]
   ids=[1]+before+target+after+[2];mask=[0]*(1+len(before))+[1]*len(target)+[0]*(len(after)+1)
   result.append({'doc':doc['id'],'domain':doc['domain'],'index':i,'ids':ids,'mask':mask,
    'position':i/max(1,len(doc['text'])-1),'label':LABELS.index(doc['labels'][i]) if doc['labels'] is not None else -1})
 return result,{'sentences':len(result),'target_sentences_truncated':truncated,'max_tokens':max_tokens,'context_sentences':context_sentences}

def validate_predictions(pred,docs):
 if set(pred)!={d['id'] for d in docs}:raise ValueError('Official ID mismatch')
 for d in docs:
  if len(pred[d['id']])!=len(d['text']) or any(x not in LABELS for x in pred[d['id']]):raise ValueError('Invalid sentence count/label')

def export(path,pred,docs):
 validate_predictions(pred,docs);path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
 tmp=path.with_suffix('.partial')
 with zipfile.ZipFile(tmp,'w',zipfile.ZIP_DEFLATED) as z:z.writestr('prediction.json',json.dumps(pred,ensure_ascii=False))
 with zipfile.ZipFile(tmp) as z:
  if z.namelist()!=['prediction.json'] or z.testzip() is not None:raise ValueError('Invalid ZIP')
 tmp.replace(path);return sha(path)
