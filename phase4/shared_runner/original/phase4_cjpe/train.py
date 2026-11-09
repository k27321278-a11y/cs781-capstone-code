"""T4 training, exact resume and independent own-model distillation."""
import copy,gc,json,math,random,shutil,time,traceback
from datetime import datetime
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from phase3_rr.archive_inputs import resolve_bundle
from .common import atomic_json,save,seed_all,sha,digest,source_hashes,hardware,status,now
from .data import index_data,choose_ids,cache_tokens
from .models import make_model,restore
from .export import predict,score,select_threshold,export_result,autocast

def deadline_timestamp(config):
 return datetime.fromisoformat(config['deadline_utc'].replace('Z','+00:00')).timestamp()

def snapshot(model,optimizer,scaler,state,evidence,fingerprint):
 return {'model':model.state_dict(),'optimizer':optimizer.state_dict(),'scaler':scaler.state_dict(),
  'state':copy.deepcopy(state),'evidence':evidence,'fingerprint':fingerprint,
  'torch_rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state().cpu(),
  'numpy_rng':np.random.get_state(),'python_rng':random.getstate()}

def train_one(config,bundle,store,audit,teacher=None):
 root=Path(config['run_root']);root.mkdir(parents=True,exist_ok=True)
 if not torch.cuda.is_available():raise RuntimeError('Select a T4 GPU runtime')
 for key in ('training_hours','max_epochs','accumulation','eval_every','save_every'):
  if config[key]<=0:raise ValueError('Invalid '+key)
 device=torch.device('cuda');seed_all(config['seed'])
 fp={'config':config,'parent_sha256':sha(bundle/'checkpoint.pt'),'tokenizer_sha256':sha(bundle/'tokenizer.json'),
  'official_data':audit['files'],'split_ids':digest(store.ids),'source':source_hashes(),
  'teacher':teacher['fingerprint'] if teacher else None,
  'warm_start_sha256':sha(config['warm_start']) if config.get('warm_start') else None}
 done=root/'complete.json'
 if done.exists():
  result=json.loads(done.read_text())
  if result['fingerprint']!=fp or sha(result['submission'])!=result['sha256']:raise ValueError('Completed artifact fingerprint changed')
  return result
 if time.time()>=deadline_timestamp(config) and not config.get('allow_after_deadline',False):raise RuntimeError('Official deadline passed; enable allow_after_deadline only for a truthfully labelled later experiment')
 status(root,'loading',step=0,last_optimizer_update_utc=None)
 model,evidence=make_model(bundle,config['family'],config['method'],config['rank'],config['alpha'])
 if teacher and teacher['parameters']<=evidence['parameters']:
  raise ValueError('The defined distillation teacher must be larger than its A0 student')
 warm=config.get('warm_start')
 if warm and not (root/'latest.pt').exists():
  old=torch.load(warm,map_location='cpu',weights_only=False)
  if old['evidence']['family']!=config['family'] or old['evidence']['method']!=config['method']:raise ValueError('Recovery warm-start architecture mismatch')
  model.load_state_dict(old['model'],strict=True);del old
  evidence['recovery_warm_start']={'path':warm,'sha256':sha(warm),'optimizer':'fresh; not an exact resume'}
 model.to(device)
 backbone=[p for p in model.backbone.parameters() if p.requires_grad]
 head=[p for n,p in model.named_parameters() if not n.startswith('backbone.') and p.requires_grad]
 optimizer=torch.optim.AdamW([{'params':backbone,'lr':config['backbone_lr']},{'params':head,'lr':config['head_lr']}],weight_decay=.01)
 scaler=torch.amp.GradScaler('cuda',init_scale=128)
 state={'epoch':0,'position':0,'step':0,'seen_cases':0,'skipped_updates':0,'best_f1':-1.,'bad_evals':0,'elapsed_seconds':0.,'threshold':.5}
 latest=root/'latest.pt'
 if latest.exists():
  obj=torch.load(latest,map_location='cpu',weights_only=False)
  if obj['fingerprint']!=fp:raise ValueError('Resume source/config/data/parent differs; choose a new RUN_ROOT')
  model.load_state_dict(obj['model'],strict=True);optimizer.load_state_dict(obj['optimizer']);scaler.load_state_dict(obj['scaler']);state=obj['state'];evidence=obj['evidence']
  torch.set_rng_state(obj['torch_rng'].cpu());torch.cuda.set_rng_state(obj['cuda_rng'].cpu());np.random.set_state(obj['numpy_rng']);random.setstate(obj['python_rng']);del obj
 evidence.update(fingerprint=fp,hardware=hardware(),initialization='own Phase1 LM; new CJPE head and optimizer on first start')
 atomic_json(root/'run.json',evidence);atomic_json(root/'config.json',config)
 shutil.copy2(bundle/'tokenizer.json',root/'tokenizer.json')
 atomic_json(root/'splits.json',store.ids)
 labels=[store.row('train',key)['label'] for key in store.ids['train']];counts=np.bincount(labels,minlength=2)
 weights=torch.tensor(np.minimum(3.,len(labels)/(2*np.maximum(counts,1))),device=device,dtype=torch.float32)
 atomic_json(root/'class_weights.json',{'train_class_counts':counts.tolist(),'weights':weights.tolist()})
 start=time.monotonic();prior_elapsed=state['elapsed_seconds'];hours_left=max(0.,config['training_hours']*3600-prior_elapsed)
 cutoff=time.time()+hours_left
 if not config.get('allow_after_deadline',False):cutoff=min(cutoff,deadline_timestamp(config)-config['export_reserve_minutes']*60)
 total_steps=max(1,math.ceil(len(labels)/config['accumulation'])*config['max_epochs']);warm_steps=min(50,max(1,total_steps//20))
 best=root/'best.pt';exported=(root/'latest_export.json').exists();updates_since_eval=0;backbone_checked=state['step']>0
 stop=False
 try:
  while state['epoch']<config['max_epochs'] and not stop:
   order=np.random.default_rng(config['seed']+state['epoch']).permutation(len(labels))
   while state['position']<len(order):
    if time.time()>=cutoff:stop=True;break
    indices=order[state['position']:state['position']+config['accumulation']]
    model.train();optimizer.zero_grad(set_to_none=True);loss_sum=0.
    status(root,'training',step=state['step'],epoch=state['epoch'],seen_cases=state['seen_cases'],best_holdout_f1=state['best_f1'])
    for index in indices:
     key=store.ids['train'][int(index)];row=store.row('train',key)
     with autocast(device):logits=model(row['chunks'],row['positions'],device).float()
     target=torch.tensor([row['label']],device=device,dtype=torch.long)
     loss=F.cross_entropy(logits,target)*weights[row['label']]
     if teacher:
      teacher_value=torch.as_tensor(teacher['values'][teacher['index'][key]][None],device=device,dtype=torch.float32)
      temp=config['kd_temperature']
      soft=F.kl_div(F.log_softmax(logits/temp,-1),F.softmax(teacher_value/temp,-1),reduction='batchmean')*temp*temp
      loss=(1-config['kd_weight'])*loss+config['kd_weight']*soft
     if not torch.isfinite(loss):raise FloatingPointError('Nonfinite training loss')
     loss_sum+=float(loss.detach());scaler.scale(loss/len(indices)).backward()
    scaler.unscale_(optimizer)
    params=[p for p in model.parameters() if p.requires_grad and p.grad is not None]
    finite=all(bool(torch.isfinite(p.grad).all()) for p in params)
    if not finite:
     scaler.step(optimizer);scaler.update();optimizer.zero_grad(set_to_none=True)
     state['skipped_updates']+=1
     if state['skipped_updates']>20:raise FloatingPointError('Repeated AMP overflow; inspect failed run')
     continue # Retry the same case group at a reduced scaler value.
    if not backbone_checked:
     grads=[(n,float(p.grad.float().norm())) for n,p in model.backbone.named_parameters() if p.requires_grad and p.grad is not None]
     nonzero=[(n,v) for n,v in grads if v>0]
     if not nonzero:raise RuntimeError('Adaptation/backbone has no nonzero gradients')
     atomic_json(root/'first_backbone_gradients.json',{'step':state['step']+1,'nonzero_examples':nonzero[:20]});backbone_checked=True
    torch.nn.utils.clip_grad_norm_(params,1.,error_if_nonfinite=True)
    step=state['step']+1
    factor=min(1.,step/warm_steps)*(.1+.9*.5*(1+math.cos(math.pi*min(step,total_steps)/total_steps)))
    optimizer.param_groups[0]['lr']=config['backbone_lr']*factor;optimizer.param_groups[1]['lr']=config['head_lr']*factor
    scaler.step(optimizer);scaler.update();optimizer.zero_grad(set_to_none=True)
    state['position']+=len(indices);state['seen_cases']+=len(indices);state['step']+=1;updates_since_eval+=1
    state['elapsed_seconds']=prior_elapsed+time.monotonic()-start
    progress={'step':state['step'],'epoch':state['epoch'],'loss':loss_sum/len(indices),'seen_cases':state['seen_cases'],
     'last_optimizer_update_utc':now(),'elapsed_training_seconds':state['elapsed_seconds'],
     'seconds_per_update':(time.monotonic()-start)/max(1,updates_since_eval)}
    status(root,'training',**progress)
    with (root/'training.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(progress)+'\n')
    if state['step']%config['save_every']==0:save(latest,snapshot(model,optimizer,scaler,state,evidence,fp))
    if state['step']%config['eval_every']==0:
     _,values,gold=predict(model,store,'holdout',device,root);metrics=select_threshold(gold,values)
     if metrics['macro_f1']>state['best_f1']:
      state['best_f1']=metrics['macro_f1'];state['threshold']=metrics['threshold'];state['bad_evals']=0
      save(best,{'model':model.state_dict(),'evidence':evidence,'state':copy.deepcopy(state),'fingerprint':fp})
      atomic_json(root/'selection.json',{'selection_split':'fixed train-only holdout','metrics':metrics,'step':state['step'],'sha256':sha(best)})
     else:state['bad_evals']+=1
     save(latest,snapshot(model,optimizer,scaler,state,evidence,fp))
     if not exported:
      current={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
      chosen=torch.load(best,map_location='cpu',weights_only=False);model.load_state_dict(chosen['model']);del chosen
      export_result(model,store,device,config,root,'early',state['threshold']);exported=True
      model.load_state_dict(current);del current
     if state['bad_evals']>=config['patience']:stop=True;break
   if state['position']>=len(order):state['epoch']+=1;state['position']=0
  if state['step']==0:raise RuntimeError('No optimizer updates completed before the budget; no trained submission fabricated')
  if not best.exists() or (updates_since_eval and state['step']%config['eval_every']):
   _,values,gold=predict(model,store,'holdout',device,root);metrics=select_threshold(gold,values)
   if metrics['macro_f1']>state['best_f1']:
    state['best_f1']=metrics['macro_f1'];state['threshold']=metrics['threshold']
    save(best,{'model':model.state_dict(),'evidence':evidence,'state':copy.deepcopy(state),'fingerprint':fp})
    atomic_json(root/'selection.json',{'selection_split':'fixed train-only holdout','metrics':metrics,'step':state['step'],'sha256':sha(best)})
  state['elapsed_seconds']=prior_elapsed+time.monotonic()-start
  save(latest,snapshot(model,optimizer,scaler,state,evidence,fp))
  selected=torch.load(best,map_location='cpu',weights_only=False);model.load_state_dict(selected['model']);selected_state=selected['state'];del selected
  result=export_result(model,store,device,config,root,'final',selected_state['threshold'])
  result.update(run_root=str(root),name=config['name'],selected_checkpoint=str(best),selected_sha256=sha(best),fingerprint=fp,
   optimizer_updates=state['step'],seen_training_cases=state['seen_cases'],selected_step=selected_state['step'],
   completed_after_deadline=time.time()>deadline_timestamp(config),hardware=hardware(),
   parameters=evidence['parameters'],trainable_parameters=evidence['trainable_parameters'])
  atomic_json(done,result);status(root,'complete',submission=result['submission'],dev_macro_f1=result['dev_metrics']['macro_f1'])
  return result
 finally:
  del model,optimizer,scaler;gc.collect();torch.cuda.empty_cache()

def prepare(config):
 root=Path(config['run_root']);root.mkdir(parents=True,exist_ok=True)
 bundle,manifest=resolve_bundle(config['bundle'],Path(config['work_root'])/'weights')
 if manifest['family']!=config['family']:raise ValueError('Wrong model upload bundle')
 # SQLite must live on the runtime's local disk, never the mounted Drive filesystem.
 cache_root=Path(config['work_root'])/'cjpe_cache_local'/config['name']
 cache_root.mkdir(parents=True,exist_ok=True)
 status(root,'preparing_data',error=None,note='SQLite caches on local runtime disk; checkpoints and results remain on Drive.')
 db,audit=index_data(config['data_root'],cache_root,progress_root=root)
 cfg=copy.deepcopy(config);cfg['data_fingerprint']=audit['files']
 ids=choose_ids(db,cfg['seed'],cfg['max_train_cases'],cfg['holdout_per_class'])
 store=cache_tokens(db,ids,bundle,cfg,cache_root,progress_root=root)
 for name in ('data_audit.json','token_cache.json'):shutil.copy2(cache_root/name,root/name)
 atomic_json(root/'cache_location.json',{'sqlite_on_local_disk':True,'cache_root':str(cache_root),'persistent_run_root':str(root),
  'local_caches_rebuilt_after_runtime_reset':True,'old_drive_cache_preserved':True})
 return cfg,bundle,store,audit

def run(config):
 cfg,bundle,store,audit=prepare(config)
 try:return train_one(cfg,bundle,store,audit)
 finally:store.close()

def distill(config):
 root=Path(config['run_root']);teacher_cfg=copy.deepcopy(config)
 teacher_cfg.update(name=config['name']+'_Teacher',family='d222',method='lorma',bundle=config['teacher_bundle'],
  run_root=str(root/'teacher'),training_hours=config['teacher_hours'],backbone_lr=2e-4,head_lr=3e-4)
 teacher_cfg.pop('warm_start',None)
 teacher_result=run(teacher_cfg)
 cfg,bundle,student_store,audit=prepare(config)
 teacher_cfg,teacher_bundle,teacher_store,_=prepare(teacher_cfg)
 if student_store.ids!=teacher_store.ids:raise ValueError('Teacher/student split ordering differs')
 root.mkdir(parents=True,exist_ok=True);cache=root/'teacher_logits.npz';record=root/'teacher_logits.json'
 teacher_fp={'weights':teacher_result['selected_sha256'],'ids':digest(student_store.ids['train']),'data':audit['files']}
 if cache.exists() and record.exists():
  if json.loads(record.read_text())['fingerprint']!=teacher_fp:raise ValueError('Teacher logit provenance changed')
  saved=np.load(cache,allow_pickle=False);values=saved['logits'];keys=saved['ids'].tolist();saved.close()
 else:
  model,evidence=restore(teacher_result['selected_checkpoint']);device=torch.device('cuda')
  try:keys,values,_=predict(model,teacher_store,'train',device,root)
  finally:del model;gc.collect();torch.cuda.empty_cache()
  with cache.with_suffix('.partial').open('wb') as f:np.savez_compressed(f,ids=np.asarray(keys,dtype=str),logits=values)
  cache.with_suffix('.partial').replace(cache);atomic_json(record,{'fingerprint':teacher_fp,'sha256':sha(cache),'examples':len(keys),'training_cases_only':True})
 teacher_store.close()
 if keys!=student_store.ids['train'] or values.shape!=(len(keys),2):raise ValueError('Teacher logit ordering/shape mismatch')
 teacher={'fingerprint':teacher_fp,'values':values,'index':{key:n for n,key in enumerate(keys)},'parameters':teacher_result['parameters']}
 try:
  result=train_one(cfg,bundle,student_store,audit,teacher)
  atomic_json(root/'distillation.json',{'teacher':teacher_result,'student':{'family':'a0','parameters':result['parameters'],'result':result['dev_metrics']},'temperature':cfg['kd_temperature'],'soft_loss_weight':cfg['kd_weight']})
  return result
 finally:student_store.close()

def worker(config):
 root=Path(config['run_root']);root.mkdir(parents=True,exist_ok=True)
 try:return distill(config) if config['experiment']=='distillation' else run(config)
 except Exception as exc:
  record={'type':type(exc).__name__,'message':str(exc),'traceback':traceback.format_exc(),'failed_utc':now()}
  atomic_json(root/'failure.json',record);status(root,'failed',error=record['message']);raise
