"""Bounded NER search, strict character spans, and exact prediction ZIPs."""
import gc,json,math,pickle,random,time,subprocess,sys
from collections import Counter
from pathlib import Path
import numpy as np
import torch
from .common import save,source_hashes,autocast,deadline,runtime_evidence
from .ner import load_ner
from lner.data import (load_dataset,prepare_windows,OffsetTokenizer,atomic_json,sha256,
                       strict_score,decode_bio,export_zip,validate_predictions)
from lner.models import tensor_row

@torch.inference_mode()
def document_logits(model,rows,dataset,encoding,device,status=None,time_limit=None):
 model.eval(); texts=dataset['test'] if rows and rows[0]['id'] in dataset['test'] else dataset['texts']
 keys=set(texts) if texts is dataset['test'] else {row['id'] for row in rows}
 outputs={key:np.zeros((len(texts[key]),25),np.float32) for key in keys}
 mass={key:np.zeros(len(texts[key]),np.float32) for key in keys}
 for i,row in enumerate(rows):
  if time_limit is not None and time.time()>=time_limit: raise TimeoutError('NER generation budget exhausted')
  with autocast(device): logits=model(*tensor_row(row,encoding,device))[0].float().cpu().numpy()
  size=row['end']-row['start']; weights=np.minimum(np.arange(1,size+1),np.arange(size,0,-1)).clip(1,240).astype(np.float32)/240
  a,b=row['start'],row['end']; key=row['id']; outputs[key][a:b]+=logits*weights[:,None]; mass[key][a:b]+=weights
  if status and (i%50==0 or i+1==len(rows)): atomic_json(status,{'phase':'ner_generation','completed_windows':i+1,'total_windows':len(rows)})
 for key in keys:
  if (mass[key]<=0).any() or not np.isfinite(outputs[key]).all(): raise ValueError('Uncovered or nonfinite character features')
  if len(texts[key]): outputs[key]/=mass[key][:,None]
 return outputs

def prepare_ner(config):
 root=Path(config['output_root']); root.mkdir(parents=True,exist_ok=True); n=config['ner']
 fingerprint={'data':{str(p.relative_to(config['phase2_data'])):sha256(p) for p in sorted(Path(config['phase2_data']).rglob('*')) if p.is_file()},
  'tokenizer':sha256(config['tokenizer']),'seed':config['seed'],'window':{k:n[k] for k in ('max_chars','overlap','max_tokens')}}
 cache=root/'ner_prepared.pkl'
 if cache.exists():
  with cache.open('rb') as f: obj=pickle.load(f)
  if obj['fingerprint']!=fingerprint: raise ValueError('NER preparation changed')
  return obj['dataset'],obj['windows']
 dataset=load_dataset(config['phase2_data'],config['seed'])
 windows=prepare_windows(dataset,{'own':OffsetTokenizer(config['tokenizer'])},n['max_chars'],n['overlap'],n['max_tokens'])
 with cache.with_suffix('.partial').open('wb') as f: pickle.dump({'fingerprint':fingerprint,'dataset':dataset,'windows':windows},f)
 cache.with_suffix('.partial').replace(cache)
 atomic_json(root/'ner_split.json',{'train_ids':dataset['train_ids'],'validation_ids':dataset['validation_ids'],
  'test_ids':sorted(dataset['test']),'fingerprint':fingerprint,'char_vocab':dataset['char_vocab'],'labels':dataset['labels']})
 return dataset,windows

def decode_crf(model,outputs,labels,bias,time_limit=None):
 transition,start=model.crf.scores(); transition=transition.detach().cpu().numpy(); start=start.detach().cpu().numpy()
 end=model.crf.end.detach().cpu().numpy(); result={}
 for key,values in outputs.items():
  if time_limit is not None and time.time()>=time_limit: raise TimeoutError('NER decoding budget exhausted')
  if len(values)==0: result[key]=[]; continue
  values=values.copy(); values[:,0]+=bias
  score=start+values[0]; back=np.zeros(values.shape,dtype=np.uint8)
  for i in range(1,len(values)):
   possible=score[:,None]+transition; back[i]=possible.argmax(0); score=possible.max(0)+values[i]
  state=int((score+end).argmax()); tags=[state]
  for i in range(len(values)-1,0,-1): state=int(back[i,state]); tags.append(state)
  result[key]=decode_bio(tags[::-1],labels)
 return result

def evaluate(model,dataset,windows,device,time_limit=None):
 model.eval(); outputs=document_logits(model,windows['validation'],dataset,'own',device,time_limit=time_limit)
 truth={key:dataset['gold'][key] for key in dataset['validation_ids']}; selected=None
 for bias in (0.,.5,1.,1.5,2.):
  predictions=decode_crf(model,outputs,dataset['labels'],bias,time_limit); metrics=strict_score(truth,predictions,dataset['labels'])
  if selected is None or metrics['macro_f1']>selected['metrics']['macro_f1']:
   selected={'outside_bias':bias,'metrics':metrics,'predictions':predictions}
 return selected

def train_candidate(config,candidate,checkpoint,dataset,windows,stop_epoch=None,export=False,train_until=None):
 root=Path(config['output_root']); run=root/'ner'/candidate['name']; run.mkdir(parents=True,exist_ok=True); n=config['ner']
 fingerprint={'candidate':candidate,'config':config,'pretrained_sha256':sha256(checkpoint),
              'split_sha256':sha256(root/'ner_split.json'),'source_hashes':source_hashes()}
 done=run/'complete.json'
 if done.exists():
  obj=json.loads(done.read_text())
  if obj['fingerprint']!=fingerprint: raise ValueError('Completed NER lineage differs')
  assert sha256(obj['submission'])==obj['sha256'] and sha256(run/'best.pt')==obj['best_checkpoint_sha256']; return obj
 device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'); seed=config['seed']
 random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.set_num_threads(4)
 model,evidence=load_ner(checkpoint,len(dataset['char_vocab'])+2,25,candidate,n['rank'],n['alpha']); model.to(device)
 base=[p for name,p in model.named_parameters() if name.startswith('backbone.') and p.requires_grad]
 head=[p for name,p in model.named_parameters() if not name.startswith('backbone.') and p.requires_grad]
 optimizer=torch.optim.AdamW([{'params':base,'lr':candidate['backbone_lr']},{'params':head,'lr':candidate['head_lr']}],weight_decay=.01)
 scaler=torch.amp.GradScaler('cuda',enabled=device.type=='cuda' and not torch.cuda.is_bf16_supported())
 counts=Counter(t for row in windows['train'] for t in row['targets'])
 weights=torch.tensor([1.]+[min(12.,math.sqrt(counts[0]/max(counts[k],1))) for k in range(1,25)],device=device)
 rare=np.array([max((1./math.sqrt(max(counts[t],1)) for t in row['targets'] if t>0 and t%2==1),default=.001) for row in windows['train']],dtype=np.float64)
 probability=(1-n['rare_sampling_mix'])/len(rare)+n['rare_sampling_mix']*rare/rare.sum()
 state={'epoch':0,'group':0,'step':0,'best':-1.,'bad':0,'trained_chars':0}; latest=run/'latest.pt'
 if latest.exists():
  saved=torch.load(latest,map_location='cpu',weights_only=False)
  if saved['fingerprint']!=fingerprint: raise ValueError('NER resume lineage differs')
  model.load_state_dict(saved['model'],strict=True); optimizer.load_state_dict(saved['optimizer']); scaler.load_state_dict(saved['scaler']); state=saved['state']
  torch.set_rng_state(saved['rng'])
  if device.type=='cuda': torch.cuda.set_rng_state(saved['cuda_rng'],device)
  del saved
 evidence.update({'candidate':candidate,'fingerprint':fingerprint,**runtime_evidence(device)})
 atomic_json(run/'run.json',evidence)
 def snapshot():
  save(latest,{'model':model.state_dict(),'optimizer':optimizer.state_dict(),'scaler':scaler.state_dict(),
              'state':state,'fingerprint':fingerprint,'rng':torch.get_rng_state(),
              'cuda_rng':torch.cuda.get_rng_state(device) if device.type=='cuda' else None})
 began=time.monotonic(); initial_step=state['step']; groups=math.ceil(len(windows['train'])/n['accumulation'])
 target_epoch=n['max_epochs'] if stop_epoch is None else stop_epoch
 train_until=min(deadline(config)-1200,train_until if train_until is not None else deadline(config)-1200)
 finalize_limit=min(deadline(config)-600,train_until+600)
 while state['epoch']<target_epoch and state['bad']<n['patience'] and time.time()<train_until:
  order=np.random.default_rng(seed+state['epoch']).choice(len(rare),len(rare),replace=True,p=probability).tolist()
  model.train(); completed_epoch=True
  for group in range(state['group'],groups):
   if time.time()>=train_until: completed_epoch=False; break
   indices=order[group*n['accumulation']:(group+1)*n['accumulation']]
   optimizer.zero_grad(set_to_none=True); loss_sum=0
   progress=(state['step']+1)/(groups*n['max_epochs']); schedule=min(1.,(state['step']+1)/max(1,groups*.5))*(.1+.9*.5*(1+math.cos(math.pi*min(progress,1.))))
   for pg,lr in zip(optimizer.param_groups,(candidate['backbone_lr'],candidate['head_lr'])): pg['lr']=lr*schedule
   for idx in indices:
    row=windows['train'][idx]; tags=torch.tensor(row['targets'],dtype=torch.long,device=device)
    with autocast(device): loss=model.loss(tensor_row(row,'own',device),tags,weights,n['boundary_weight'])
    if not torch.isfinite(loss): raise FloatingPointError('Nonfinite NER loss')
    scaler.scale(loss/len(indices)).backward(); loss_sum+=float(loss.detach())/len(indices); state['trained_chars']+=len(tags)
   scaler.unscale_(optimizer); norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
   scaler.step(optimizer); scaler.update(); state['step']+=1; state['group']=group+1
   if state['step']%10==0 or state['step']==1:
    elapsed=time.monotonic()-began; rate=(state['step']-initial_step)/max(elapsed,1e-6)
    status={'phase':'ner_training','candidate':candidate['name'],'epoch':state['epoch']+1,'step':state['step'],'loss':loss_sum,
            'grad_norm':float(norm),'best_macro_f1':state['best'],'steps_per_second':rate,'trained_chars':state['trained_chars']}
    atomic_json(run/'status.json',status)
    with (run/'metrics.jsonl').open('a') as f: f.write(json.dumps(status)+'\n')
    print(json.dumps(status),flush=True)
   if state['step']%50==0: snapshot()
  selection=evaluate(model,dataset,windows,device,finalize_limit); score=selection['metrics']['macro_f1']
  if score>state['best']+1e-5:
   state['best']=score; state['bad']=0
   save(run/'best.pt',{'model':model.state_dict(),'evidence':evidence,'selected_epoch':state['epoch']+1,
                      'partial_epoch':not completed_epoch,'state':dict(state)})
   atomic_json(run/'selection.json',{k:v for k,v in selection.items() if k!='predictions'})
   atomic_json(run/'validation_predictions.json',selection['predictions'])
  else: state['bad']+=1
  if completed_epoch: state['epoch']+=1; state['group']=0
  snapshot(); print(json.dumps({'phase':'ner_validation','candidate':candidate['name'],'macro_f1':score,'best':state['best']}),flush=True)
  if not completed_epoch: break
 if not export:
  record={'phase':'pilot_complete','candidate':candidate['name'],'macro_f1':state['best'],'epochs':state['epoch']}
  atomic_json(run/'pilot.json',record); del model,optimizer; gc.collect(); torch.cuda.empty_cache(); return record
 if not (run/'best.pt').exists(): raise RuntimeError('No selected NER checkpoint; cannot fabricate a submission')
 model.load_state_dict(torch.load(run/'best.pt',map_location='cpu',weights_only=False)['model'],strict=True)
 selection=json.loads((run/'selection.json').read_text())
 outputs=document_logits(model,windows['test'],dataset,'own',device,run/'status.json',finalize_limit)
 predictions=decode_crf(model,outputs,dataset['labels'],selection['outside_bias'],finalize_limit)
 validate_predictions(predictions,dataset['test'],dataset['labels'])
 name='UPLOAD_Phase2_D222_'+('Full' if candidate['method']=='full' else 'LORMA')+'.zip'
 archive=root/'deliverables'/name; digest=export_zip(archive,predictions,dataset['test'],dataset['labels'])
 atomic_json(run/'validation_truth.json',{key:dataset['gold'][key] for key in dataset['validation_ids']})
 scorer=Path(__file__).resolve().parents[1]/'vendor/evaluate_phase2.py'
 scored=subprocess.run([sys.executable,str(scorer),'--truth',str(run/'validation_truth.json'),
  '--prediction',str(run/'validation_predictions.json'),'--labels',str(Path(config['phase2_data'])/'Train/labels.txt')],
  check=True,capture_output=True,text=True)
 (run/'official_validation.txt').write_text(scored.stdout)
 record={'phase':'complete','candidate':candidate['name'],'method':candidate['method'],'submission':str(archive),'sha256':digest,
  'best_checkpoint_sha256':sha256(run/'best.pt'),'pretrained_sha256':fingerprint['pretrained_sha256'],
  'official_ids':len(predictions),'validation':selection['metrics'],'beats_saved_validation_baseline':selection['metrics']['macro_f1']>n['best_existing_macro_f1'],
  'trained_chars':state['trained_chars'],'epochs_completed':state['epoch'],'hardware':runtime_evidence(device),'fingerprint':fingerprint}
 atomic_json(done,record); atomic_json(run/'status.json',record); print(json.dumps(record),flush=True)
 del model,optimizer; gc.collect(); torch.cuda.empty_cache(); return record

def run_ner(config,checkpoint):
 dataset,windows=prepare_ner(config); candidates=config['ner']['candidates']; pilots=[]
 started=time.time(); available=max(0.,deadline(config)-started-2400)
 if available<1200: raise RuntimeError('Insufficient measured NER/export budget; preserve prior submissions')
 fractions=config['ner']['budget_fractions']; full_candidates=[c for c in candidates if c['method']=='full']
 pilot_end=started+available*fractions['pilot_total']
 for candidate in candidates:
  if candidate['method']=='full':
   quota=available*fractions['pilot_total']/len(full_candidates)
   result=train_candidate(config,candidate,checkpoint,dataset,windows,config['ner']['pilot_epochs'],False,train_until=min(pilot_end,time.time()+quota))
   value=result['macro_f1'] if 'macro_f1' in result else result['validation']['macro_f1']
   pilots.append({'candidate':candidate['name'],'macro_f1':value})
 selected=max(pilots,key=lambda x:x['macro_f1'])
 winner=next(x for x in candidates if x['name']==selected['candidate'])
 atomic_json(Path(config['output_root'])/'ner_search.json',{'pilot_scores':pilots,'promoted':winner,'criterion':'heldout_strict_macro_f1'})
 full_end=pilot_end+available*fractions['full_promoted']
 atomic_json(Path(config['output_root'])/'ner_budget.json',{'started':started,'pilot_end':pilot_end,'full_training_end':full_end,
  'lorma_training_end':started+available,'export_reserve_seconds':2400})
 results=[train_candidate(config,winner,checkpoint,dataset,windows,export=True,train_until=full_end)]
 for candidate in candidates:
  if candidate['method']=='lorma': results.append(train_candidate(config,candidate,checkpoint,dataset,windows,export=True,train_until=started+available))
 return results
