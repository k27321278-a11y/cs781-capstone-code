"""Strict result archives and official held-out evaluation."""
import json,subprocess,sys,zipfile
from pathlib import Path
import numpy as np
import torch
from sklearn.metrics import f1_score,precision_score,recall_score,confusion_matrix
from .common import atomic_json,sha,status,now

def autocast(device):return torch.autocast(device_type=device.type,dtype=torch.float16,enabled=device.type=='cuda')

@torch.no_grad()
def predict(model,store,split,device,root):
 model.eval();keys=[];logits=[];truth=[]
 status(root,'evaluating',evaluating_split=split,evaluated_cases=0)
 for n,row in enumerate(store.rows(split)):
  with autocast(device):value=model(row['chunks'],row['positions'],device).float()[0]
  if not torch.isfinite(value).all():raise FloatingPointError('Nonfinite inference')
  keys.append(row['id']);logits.append(value.cpu().numpy());truth.append(row['label'])
  if (n+1)%100==0:status(root,'evaluating',evaluating_split=split,evaluated_cases=n+1,total_cases=len(store.ids[split]))
 return keys,np.stack(logits),truth

def probabilities(logits):
 values=logits-logits.max(1,keepdims=True);values=np.exp(values);return values[:,1]/values.sum(1)

def score(labels,logits,threshold=.5):
 pred=(probabilities(logits)>=threshold).astype(int)
 return {'macro_f1':float(f1_score(labels,pred,labels=[0,1],average='macro',zero_division=0)),
  'precision':float(precision_score(labels,pred,zero_division=0)),
  'recall':float(recall_score(labels,pred,zero_division=0)),
  'confusion_matrix':confusion_matrix(labels,pred,labels=[0,1]).tolist(),
  'class_support':[int(sum(y==k for y in labels)) for k in (0,1)],'threshold':float(threshold),'cases':len(labels)}

def select_threshold(labels,logits):
 values=probabilities(np.asarray(logits));gold=np.asarray(labels,dtype=np.int64)
 if len(values)!=len(gold) or not len(gold) or not np.isfinite(values).all():raise ValueError('Invalid holdout probabilities')
 unique=np.unique(values)
 candidates=np.unique(np.r_[0.,.5,np.nextafter(1.,2.),(unique[:-1]+unique[1:])/2])
 order=np.argsort(values);sorted_values=values[order];positive=np.r_[0,np.cumsum(gold[order])]
 k=np.searchsorted(sorted_values,candidates,side='left');tp=positive[-1]-positive[k]
 fp=len(gold)-k-tp;fn=positive[-1]-tp;tn=k-fn
 fpos=np.divide(2*tp,2*tp+fp+fn,out=np.zeros_like(tp,dtype=float),where=(2*tp+fp+fn)>0)
 fneg=np.divide(2*tn,2*tn+fp+fn,out=np.zeros_like(tn,dtype=float),where=(2*tn+fp+fn)>0)
 objective=(fpos+fneg)/2
 best=max(range(len(candidates)),key=lambda i:(objective[i],-abs(candidates[i]-.5)))
 result=score(labels,logits,float(candidates[best]));result['search']='all distinct holdout probability boundaries'
 return result

def write_predictions(path,keys,logits,threshold):
 values=(probabilities(logits)>=threshold).astype(int)
 rows=[{'id':key,'pred':int(v)} for key,v in zip(keys,values)]
 if len(set(keys))!=len(keys):raise ValueError('Duplicate prediction IDs')
 atomic_json(path,rows);return rows

def validate_zip(path,expected):
 with zipfile.ZipFile(path) as z:
  if z.namelist()!=['prediction.json'] or z.testzip() is not None:raise ValueError('Invalid result ZIP root/CRC')
  rows=json.loads(z.read('prediction.json'))
 if not isinstance(rows,list) or len(rows)!=len(expected):raise ValueError('Wrong prediction count')
 keys=[]
 for row in rows:
  if set(row)!=set(['id','pred']) or type(row['id']) is not str or type(row['pred']) is not int or row['pred'] not in (0,1):raise ValueError('Invalid prediction row')
  keys.append(row['id'])
 if len(keys)!=len(set(keys)) or set(keys)!=set(expected):raise ValueError('Submission IDs differ from official test IDs')
 return {'sha256':sha(path),'bytes':Path(path).stat().st_size,'cases':len(keys),'root_prediction_json_only':True,'crc_verified':True,'exact_ids_verified':True}

def export_result(model,store,device,config,root,stage='final',threshold=.5):
 root=Path(root);folder=root/'exports'/stage;folder.mkdir(parents=True,exist_ok=True)
 devkeys,devlogits,labels=predict(model,store,'dev',device,root)
 metrics=score(labels,devlogits,threshold)
 metrics['at_default_threshold']=score(labels,devlogits,.5)
 metrics['probability_quantiles']=np.quantile(probabilities(devlogits),[0,.1,.25,.5,.75,.9,1]).tolist()
 np.savez_compressed(folder/'dev_probabilities.npz',ids=np.asarray(devkeys,dtype=str),logits=devlogits,labels=np.asarray(labels))
 write_predictions(folder/'dev_prediction.json',devkeys,devlogits,threshold)
 atomic_json(folder/'dev_truth.json',[{'id':key,'gold':int(y)} for key,y in zip(devkeys,labels)])
 command=[sys.executable,str(Path(__file__).with_name('evaluate_phase4.py')),'--truth',str(folder/'dev_truth.json'),'--prediction',str(folder/'dev_prediction.json')]
 evaluation=subprocess.run(command,capture_output=True,text=True)
 (folder/'official_evaluation.txt').write_text(evaluation.stdout+'\n'+evaluation.stderr,encoding='utf-8')
 if evaluation.returncode:raise RuntimeError('Official evaluator rejected dev predictions: '+evaluation.stderr)
 keys,values,_=predict(model,store,'test',device,root)
 np.savez_compressed(folder/'test_probabilities.npz',ids=np.asarray(keys,dtype=str),logits=values)
 status(root,'exporting',export_stage=stage)
 write_predictions(folder/'prediction.json',keys,values,threshold)
 dest=folder/('UPLOAD_Phase4_'+config['name']+'.zip');tmp=dest.with_suffix('.partial')
 with zipfile.ZipFile(tmp,'w',zipfile.ZIP_DEFLATED) as z:z.write(folder/'prediction.json','prediction.json')
 tmp.replace(dest);checks=validate_zip(dest,store.ids['test'])
 result={**checks,'submission':str(dest),'dev_metrics':metrics,'stage':stage,'finished_utc':now(),
  'hidden_leaderboard_score':None,'official_evaluator_passed':True}
 atomic_json(folder/'result.json',result);atomic_json(root/'latest_export.json',result)
 print('VERIFIED SUBMISSION:',dest,'DEV MACRO F1:',metrics['macro_f1'],flush=True)
 return result

def package_records(result):
 root=Path(result['run_root']);dest=root/('MODEL_RECORDS_'+result['name']+'.zip')
 tmp=dest.with_suffix('.partial');manifest={}
 with zipfile.ZipFile(tmp,'w',zipfile.ZIP_DEFLATED,allowZip64=True) as z:
  selected=[p for p in root.rglob('*') if p.is_file() and (p.suffix in ('.json','.txt','.jsonl','.log') or p.name in ('best.pt','tokenizer.json')) and not p.name.endswith('.partial')]
  for p in selected:
   relative='records/'+p.relative_to(root).as_posix();z.write(p,relative);manifest[relative]=sha(p)
  upload=Path(result['submission']);z.write(upload,'submission/'+upload.name);manifest['submission/'+upload.name]=sha(upload)
  source=Path(__file__).resolve().parents[1]
  for package in ('phase4_cjpe','phase3_rr'):
   for p in sorted((source/package).rglob('*')):
    if p.is_file() and '__pycache__' not in p.parts and (p.suffix=='.py' or p.name in ('UnicodeData-13.0.0.txt','UnicodeData-14.0.0.txt')):
     relative='source/'+p.relative_to(source).as_posix();z.write(p,relative);manifest[relative]=sha(p)
  z.writestr('MANIFEST.json',json.dumps(manifest,indent=2))
 tmp.replace(dest);print('MODEL RECORDS (not a submission):',dest,flush=True);return dest
