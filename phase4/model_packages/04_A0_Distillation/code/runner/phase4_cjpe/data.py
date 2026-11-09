"""Streaming official CJPE data, split audit and bounded text-region caches."""
import hashlib,json,random,re,shutil,sqlite3,zipfile
from collections import Counter
from pathlib import Path,PurePosixPath
import numpy as np
from phase3_rr.data import Tokenizer
from .common import sha,atomic_json,digest,status

FILES={'train':'Train/train.json','dev':'Train/dev.json','test':'Test/test.json'}
INDEX_POLICY='exact_train_dev_duplicates_once_v2'

def acquire(root,uploaded_zip=None,download=False):
 root=Path(root);root.mkdir(parents=True,exist_ok=True)
 if uploaded_zip:
  with zipfile.ZipFile(uploaded_zip) as z:
   selected={}
   for info in z.infolist():
    p=PurePosixPath(info.filename)
    if p.is_absolute() or '..' in p.parts:raise ValueError('Unsafe official data ZIP path')
    for split,relative in FILES.items():
     if not info.is_dir() and p.name==split+'.json':
      if split in selected:raise ValueError('Ambiguous duplicate split file in data ZIP')
      selected[split]=info
   if set(selected)!=set(FILES):raise ValueError('Official ZIP must contain train.json, dev.json and test.json')
   for split,info in selected.items():
    dest=root/FILES[split];dest.parent.mkdir(parents=True,exist_ok=True)
    tmp=dest.with_suffix('.incoming')
    with z.open(info) as a,tmp.open('wb') as b:shutil.copyfileobj(a,b,8*1024*1024)
    if dest.exists() and sha(dest)!=sha(tmp):raise ValueError('Different existing data; choose a new DATA_ROOT')
    tmp.replace(dest)
 if download:
  from huggingface_hub import hf_hub_download,HfApi
  revision=HfApi().dataset_info('Exploration-Lab/CS-781-Capstone').sha
  atomic_json(root/'download_revision.json',{'repository':'Exploration-Lab/CS-781-Capstone','revision':revision})
  for split,relative in FILES.items():
   dest=root/relative
   if not dest.exists():
    # Authentication is provided by the explicit notebook login cell only.
    source=hf_hub_download('Exploration-Lab/CS-781-Capstone',filename='Phase_4_CJPE/'+relative,repo_type='dataset',revision=revision)
    dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,dest)
 for relative in FILES.values():
  if not (root/relative).is_file():raise FileNotFoundError(str(root/relative))
 return root

def stream_rows(path):
 import ijson
 with Path(path).open('rb') as f:
  first=f.read(3);f.seek(0)
  if first==b'\xef\xbb\xbf':f.seek(3)
  firstchar=b''
  while not firstchar or firstchar.isspace():firstchar=f.read(1)
  offset=3 if first==b'\xef\xbb\xbf' else 0;f.seek(offset)
  if firstchar==b'[':
   yield from ijson.items(f,'item',use_float=True);return
  if firstchar!=b'{':raise ValueError('Expected official JSON list or ID-to-case object')
  envelope=None
  for n,(prefix,event,value) in enumerate(ijson.parse(f,use_float=True)):
   if prefix=='' and event=='map_key' and value in ('data','cases','examples'):envelope=value;break
   if n>=120:break
  f.seek(offset)
  if envelope:yield from ijson.items(f,envelope+'.item',use_float=True)
  else:
   for key,value in ijson.kvitems(f,'',use_float=True):
    if not isinstance(value,dict):raise ValueError('Unsupported case container; inspect the displayed raw schema')
    yield {'id':key,**value}

def normal(row,split):
 if not isinstance(row,dict) or 'id' not in row or 'text' not in row:raise ValueError('CJPE requires id/text; actual keys: '+str(list(row)[:20]))
 ident=str(row['id'])
 if not ident.strip():raise ValueError('Empty official case ID')
 text=row['text']
 if isinstance(text,list) and all(isinstance(v,str) for v in text):text='\n'.join(text)
 if not isinstance(text,str):raise ValueError('Case text must be a string or list[str]')
 label=None
 if split!='test':
  candidates=[row[k] for k in ('label','gold','labels') if k in row]
  if not candidates:raise ValueError('No label/gold field; inspect official schema before training')
  labels=[]
  for value in candidates:
   if isinstance(value,bool) or value not in (0,1,'0','1'):raise ValueError('Expected scalar binary training label')
   labels.append(int(value))
  if len(set(labels))!=1:raise ValueError('Conflicting label fields')
  label=labels[0]
 # Deliberately discard all remaining fields, including experts and test labels.
 canonical=' '.join(text.casefold().split())
 text_sha=hashlib.sha256(canonical.encode()).hexdigest() if canonical else None
 return ident,text,label,text_sha

def index_data(folder,root,progress_root=None):
 folder=Path(folder);root=Path(root);root.mkdir(parents=True,exist_ok=True)
 fp={relative:sha(folder/relative) for relative in FILES.values()}
 db=root/'cases.sqlite';record=root/'data_audit.json'
 if db.exists() and record.exists():
  obj=json.loads(record.read_text())
  if obj['files']!=fp:raise ValueError('Official dataset changed: choose a new run root')
  if obj.get('index_policy')!=INDEX_POLICY:raise ValueError('Data indexing policy changed: preserve this run and choose a new run root')
  return db,obj
 tmp=root/'cases.building.sqlite'
 if tmp.exists():tmp.unlink() # Only this interrupted generated index, never input data.
 con=sqlite3.connect(tmp);con.execute('CREATE TABLE cases(split TEXT,id TEXT,text TEXT,label INTEGER,text_sha TEXT,PRIMARY KEY(split,id))')
 counts={};schema={};duplicate_examples={'train':[],'dev':[]}
 try:
  for split,relative in FILES.items():
   labels=Counter();n=0;raw_rows=0;identical_duplicates=0
   for row in stream_rows(folder/relative):
    raw_rows+=1
    if not n:schema[split]={'keys':list(row),'text_type':type(row.get('text')).__name__}
    ident,text,label,h=normal(row,split)
    try:con.execute('INSERT INTO cases VALUES(?,?,?,?,?)',(split,ident,text,label,h))
    except sqlite3.IntegrityError as exc:
     previous=con.execute('SELECT text,label FROM cases WHERE split=? AND id=?',(split,ident)).fetchone()
     if split in ('train','dev') and previous==(text,label):
      identical_duplicates+=1
      if len(duplicate_examples[split])<20:duplicate_examples[split].append({'id':ident,'label':label,'text_sha256':hashlib.sha256(text.encode()).hexdigest()})
      continue
     kind='Conflicting' if split in ('train','dev') else 'Duplicate'
     raise ValueError(kind+' official '+split+' ID '+ident+'; same_text='+str(previous is not None and previous[0]==text)+'; labels='+str((previous[1] if previous else None,label))) from exc
    labels[str(label)]+=1;n+=1
    if n%500==0:con.commit();status(progress_root or root,'indexing',split=split,cases=n)
   con.commit();counts[split]={'cases':n,'raw_rows':raw_rows,'identical_duplicates_removed':identical_duplicates,'labels':dict(labels)}
   if n==0:raise ValueError('Empty official '+split+' split')
   if split!='test' and set(labels)!=set(['0','1']):raise ValueError('Both binary classes required in '+split)
  overlap=con.execute('SELECT a.id,a.split,b.split FROM cases a JOIN cases b ON a.id=b.id AND a.split<b.split LIMIT 5').fetchall()
  con.execute('CREATE INDEX hash_index ON cases(text_sha)')
  duplicates=con.execute('SELECT a.id,b.id,a.split,b.split FROM cases a JOIN cases b ON a.text_sha=b.text_sha AND a.split<b.split WHERE a.text_sha IS NOT NULL LIMIT 5').fetchall()
  if overlap or duplicates:raise ValueError('Split leakage detected; do not train: '+str({'ids':overlap,'texts':duplicates}))
  con.commit()
  if con.execute('PRAGMA quick_check').fetchone()!=('ok',):raise sqlite3.DatabaseError('Generated local case index failed integrity check')
 finally:con.close()
 tmp.replace(db)
 obj={'files':fp,'index_policy':INDEX_POLICY,'counts':counts,'schema':schema,'duplicate_examples':duplicate_examples,'training_duplicate_examples':duplicate_examples['train'],
  'cross_split_id_overlap':0,'cross_split_normalized_text_overlap':0,'test_labels_loaded':False}
 atomic_json(record,obj);return db,obj

def choose_ids(db,seed,max_train_cases=12000,holdout_per_class=256):
 con=sqlite3.connect(db);rng=np.random.default_rng(seed);train=[];holdout=[]
 for label in (0,1):
  ids=[r[0] for r in con.execute('SELECT id FROM cases WHERE split="train" AND label=? ORDER BY id',(label,))]
  order=rng.permutation(len(ids));n=min(holdout_per_class,max(1,len(ids)//10))
  if len(ids)<=n:raise ValueError('Too few training cases for a train-only holdout')
  holdout += [ids[i] for i in order[:n]];train += [ids[i] for i in order[n:]]
 if max_train_cases and len(train)>max_train_cases:
  # Stratified proportional sample; the limit is recorded, not claimed as full-corpus training.
  selected=[]
  labels={r[0]:r[1] for r in con.execute('SELECT id,label FROM cases WHERE split="train"')}
  for label in (0,1):
   group=[i for i in train if labels[i]==label];n=max(1,round(max_train_cases*len(group)/len(train)));selected+=group[:n]
  train=selected
 result={'train':sorted(train),'holdout':sorted(holdout)}
 for split in ('dev','test'):result[split]=[r[0] for r in con.execute('SELECT id FROM cases WHERE split=? ORDER BY id',(split,))]
 con.close();return result

def chunks(text,tok,max_tokens=256,count=4,region_chars=8192):
 if max_tokens<16 or count<1:raise ValueError('Invalid chunk dimensions')
 length=len(text);positions=np.linspace(0,1,count).tolist() if count>1 else [0.]
 result=[];used=set();regions={};out_pos=[];budget=max_tokens-2
 for i,pos in enumerate(positions):
  start=int(max(0,length-region_chars)*pos);end=min(length,start+region_chars)
  if (start,end) not in regions:regions[(start,end)]=tok.encode(text[start:end])
  ids=regions[(start,end)];offset=0
  if len(ids)>budget:
   fraction=pos if length<=region_chars else (0. if i==0 else 1. if i==len(positions)-1 else .5)
   offset=int((len(ids)-budget)*fraction);ids=ids[offset:offset+budget]
  location=(start,end,offset,offset+len(ids))
  if location in used:continue
  result.append([1]+(ids or [3])+[2]);used.add(location);out_pos.append(pos)
 # Region coverage is NOT token coverage: only a bounded part of each region is encoded.
 return result,out_pos,{'document_chars':length,'region_chars_inspected':sum(b-a for a,b in regions),'tokens_retained':sum(len(x)-2 for x in result),'chunks':len(result)}

def cache_tokens(db,ids,bundle,config,root,progress_root=None):
 root=Path(root);dest=root/'tokens.sqlite';meta=root/'token_cache.json'
 fp={'data':config['data_fingerprint'],'tokenizer':sha(Path(bundle)/'tokenizer.json'),'ids':digest(ids),'implementation':sha(__file__),
     'max_tokens':config['max_tokens'],'chunks':config['chunks_per_case'],'region_chars':config['region_chars']}
 if dest.exists() and meta.exists():
  if json.loads(meta.read_text())['fingerprint']!=fp:raise ValueError('Different token cache; choose a new run root')
  return TokenStore(dest,ids)
 tok=Tokenizer(Path(bundle)/'tokenizer.json');con=sqlite3.connect(db)
 tmp=root/'tokens.building.sqlite'
 if tmp.exists():tmp.unlink()
 out=sqlite3.connect(tmp);out.execute('CREATE TABLE tokens(split TEXT,id TEXT,payload TEXT,PRIMARY KEY(split,id))')
 stats={}
 try:
  for split,keys in ids.items():
   stats[split]={'cases':0,'tokens_retained':0,'document_chars':0,'chunks':0}
   original='train' if split=='holdout' else split
   for n,key in enumerate(keys):
    text,label=con.execute('SELECT text,label FROM cases WHERE split=? AND id=?',(original,key)).fetchone()
    seq,pos,coverage=chunks(text,tok,config['max_tokens'],config['chunks_per_case'],config['region_chars'])
    row={'id':key,'label':label,'chunks':seq,'positions':pos}
    out.execute('INSERT INTO tokens VALUES(?,?,?)',(split,key,json.dumps(row)))
    stats[split]['cases']+=1
    for k in ('tokens_retained','document_chars','chunks'):stats[split][k]+=coverage[k]
    if (n+1)%100==0:out.commit();status(progress_root or root,'tokenizing',split=split,cases=n+1,total=len(keys))
   out.commit()
  if out.execute('PRAGMA quick_check').fetchone()!=('ok',):raise sqlite3.DatabaseError('Generated local token cache failed integrity check')
 finally:out.close();con.close()
 tmp.replace(dest);atomic_json(meta,{'fingerprint':fp,'stats':stats});return TokenStore(dest,ids)

class TokenStore:
 def __init__(self,path,ids):self.path=str(path);self.ids=ids;self.con=sqlite3.connect(path)
 def row(self,split,key):
  value=self.con.execute('SELECT payload FROM tokens WHERE split=? AND id=?',(split,key)).fetchone()
  if value is None:raise KeyError((split,key))
  return json.loads(value[0])
 def rows(self,split):
  for key in self.ids[split]:yield self.row(split,key)
 def close(self):self.con.close()
