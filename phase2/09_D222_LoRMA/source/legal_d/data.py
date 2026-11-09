"""Verified official cached tokens; full next-token or continuation supervision."""
import json,math
from pathlib import Path
import numpy as np
import torch
from lner.data import sha256
from cs781.prepare import load_prepared
from legalspan.bpe import OwnBPE

class LMBlocks:
 def __init__(self,root,split,seed,context,tokenizer_path):
  self.root=Path(root); self.split=split; self.seed=seed; self.context=context
  self.offsets=np.load(self.root/(split+'.offsets.npy'),mmap_mode='r',allow_pickle=False)
  self.handle=(self.root/(split+'.jsonl')).open('rb'); self.epoch=None; self.order=None
  self.tokenizer=OwnBPE.load(tokenizer_path)
  self.bos=self.tokenizer.token_to_id('<bos>'); self.eos=self.tokenizer.token_to_id('<eos>')
 def __len__(self): return len(self.offsets)
 def at(self,cursor,allow_long=False):
  epoch,index=divmod(cursor,len(self))
  if epoch!=self.epoch:
   self.order=np.random.default_rng(self.seed+epoch).permutation(len(self)); self.epoch=epoch
  self.handle.seek(int(self.offsets[int(self.order[index])]))
  row=json.loads(self.handle.readline())
  if row['split']!=self.split or row['source'] not in ('acts','high_courts','supreme_court','tribunals'):
   raise ValueError('Corpus source/split differs')
  tokens=[self.bos]+row['prefix_ids']+row['suffix_ids']+[self.eos]
  if (len(tokens)-1>self.context and not allow_long) or len(tokens)<3 or min(tokens)<0 or max(tokens)>=self.tokenizer.get_vocab_size():
   raise ValueError('Cached tokens/context invalid')
  return row,tokens
 def chunks(self,cursor,device):
  _,tokens=self.at(cursor,allow_long=True)
  for start in range(0,len(tokens)-1,self.context):
   part=tokens[start:start+self.context+1]
   yield torch.tensor([part[:-1]],dtype=torch.long,device=device),torch.tensor([part[1:]],dtype=torch.long,device=device)
 def batch(self,cursor,size,device,mode='full'):
  rows=[self.at(cursor+i) for i in range(size)]; width=max(len(ids)-1 for row,ids in rows)
  x=torch.zeros((size,width),dtype=torch.long,device=device)
  y=torch.full_like(x,-100); source_counts={}
  for i,(row,ids) in enumerate(rows):
   n=len(ids)-1; x[i,:n]=torch.tensor(ids[:-1],device=device); y[i,:n]=torch.tensor(ids[1:],device=device)
   if mode=='continuation': y[i,:len(row['prefix_ids'])]=-100
   source_counts[row['source']]=source_counts.get(row['source'],0)+int((y[i]!=-100).sum())
  return x,y,sum(len(ids)-1 for row,ids in rows),int((y!=-100).sum()),source_counts

def verify_prepared(config):
 data=load_prepared(config['prepared_root'])
 if data.tokenizer_sha256!=sha256(config['tokenizer']): raise ValueError('Prepared/tokenizer hashes differ')
 tok=OwnBPE.load(config['tokenizer'])
 if tok.get_vocab_size()!=config['model']['vocab_size']: raise ValueError('Actual tokenizer vocabulary differs')
 return {'manifest_sha256':data.manifest_sha256,'tokenizer_sha256':data.tokenizer_sha256,
  'split_counts':{k:getattr(data,k+'_examples') for k in ('train','dev','confirmation')}}
