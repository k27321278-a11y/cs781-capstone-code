"""Resumable from-scratch pretraining with measured deadline/token budget."""
import argparse,fcntl,gc,json,math,os,random,time
from dataclasses import asdict
from pathlib import Path
import numpy as np
import torch
from .model import DConfig,EfficientDecoder
from .data import LMBlocks,verify_prepared
from .common import save,source_hashes,autocast,deadline,runtime_evidence
from lner.data import atomic_json,sha256

@torch.inference_mode()
def validation_loss(model,data,device,count=32,time_limit=None):
 model.eval(); weighted=total=0
 for i in range(min(count,len(data))):
  for x,y in data.chunks(i,device):
   if time_limit is not None and time.time()>=time_limit: raise TimeoutError('LM validation budget exhausted')
   n=y.numel()
   with autocast(device): _,loss,_=model(x,y)
   weighted+=float(loss)*n; total+=n
 model.train(); return weighted/total

def pretrain(config,run_dir,resume=True):
 run=Path(run_dir); run.mkdir(parents=True,exist_ok=True)
 lock=(run/'train.lock').open('a+'); fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 provenance=verify_prepared(config); device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
 torch.set_num_threads(4); seed=config['seed']; random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
 cfg=DConfig(**config['model']); model=EfficientDecoder(cfg).to(device)
 if model.count_parameters()!=cfg.parameter_formula() or not 50_000_000<=model.count_parameters()<=300_000_000:
  raise ValueError('Parameter formula or course range fails')
 p=config['pretrain']; optimizer=torch.optim.AdamW(model.parameters(),lr=p['learning_rate'],betas=(.9,.95),weight_decay=p['weight_decay'])
 scaler=torch.amp.GradScaler('cuda',enabled=device.type=='cuda' and not torch.cuda.is_bf16_supported())
 fingerprint={'config':config,'data':provenance,'source_hashes':source_hashes()}
 state={'step':0,'cursor':0,'input_tokens':0,'target_tokens':0,'best_loss':float('inf'),
        'source_tokens':{},'effective_target_tokens':p['target_tokens'],'candidates':[]}
 latest=run/'latest.pt'
 if latest.exists():
  if not resume: raise FileExistsError('Existing checkpoint requires resume')
  saved=torch.load(latest,map_location='cpu',weights_only=False)
  if saved['fingerprint']!=fingerprint: raise ValueError('Resume lineage differs')
  model.load_state_dict(saved['model'],strict=True); optimizer.load_state_dict(saved['optimizer']); scaler.load_state_dict(saved['scaler'])
  state=saved['state']; torch.set_rng_state(saved['rng'])
  if device.type=='cuda': torch.cuda.set_rng_state(saved['cuda_rng'],device)
  del saved
 evidence={**runtime_evidence(device),'parameters':model.count_parameters(),'model_config':asdict(cfg),
           'data':provenance,'fingerprint':fingerprint,'initialization':'random_from_scratch'}
 atomic_json(run/'run.json',evidence)
 train=LMBlocks(config['prepared_root'],'train',seed,cfg.context,config['tokenizer'])
 dev=LMBlocks(config['prepared_root'],'dev',seed,cfg.context,config['tokenizer'])
 began=time.monotonic(); start_tokens=state['target_tokens']; start_step=state['step']
 finalize_limit=deadline(config,p['reserve_ner_hours']); cutoff=finalize_limit-300
 def snapshot():
  save(latest,{'model':model.state_dict(),'optimizer':optimizer.state_dict(),'scaler':scaler.state_dict(),
   'state':state,'fingerprint':fingerprint,'evidence':evidence,'rng':torch.get_rng_state(),
   'cuda_rng':torch.cuda.get_rng_state(device) if device.type=='cuda' else None})
 def select():
  value=validation_loss(model,dev,device,time_limit=finalize_limit)
  path=run/('candidate_%07d.pt'%state['step'])
  if len(state['candidates'])<p['selection_candidates'] or value<max(x['loss'] for x in state['candidates']):
   save(path,{'model':model.state_dict(),'state':dict(state),'evidence':evidence,'validation_loss':value})
   state['candidates'].append({'path':str(path),'loss':value,'step':state['step'],'target_tokens':state['target_tokens']})
   state['candidates']=sorted(state['candidates'],key=lambda x:x['loss'])[:p['selection_candidates']]
  if value<state['best_loss']:
   state['best_loss']=value
   save(run/'best.pt',{'model':model.state_dict(),'state':dict(state),'evidence':evidence,'validation_loss':value})
  atomic_json(run/'candidate_inventory.json',state['candidates'])
  print(json.dumps({'phase':'lm_validation','step':state['step'],'loss':value,'best_loss':state['best_loss']}),flush=True)
 while state['target_tokens']<state['effective_target_tokens'] and time.time()<cutoff:
  optimizer.zero_grad(set_to_none=True); losses=[]
  progress=state['target_tokens']/max(1,state['effective_target_tokens'])
  scale=min(1.,(state['target_tokens']+p['microbatch']*p['accumulation']*cfg.context)/(max(1,p['target_tokens'])*p['warmup_fraction']))
  scale*=.1+.9*.5*(1+math.cos(math.pi*min(1.,progress)))
  for group in optimizer.param_groups: group['lr']=p['learning_rate']*scale
  for _ in range(p['accumulation']):
   x,y,inputs,targets,sources=train.batch(state['cursor'],p['microbatch'],device)
   with autocast(device): _,loss,_=model(x,y)
   if not torch.isfinite(loss): raise FloatingPointError('Nonfinite LM loss')
   scaler.scale(loss/p['accumulation']).backward(); losses.append(float(loss.detach()))
   state['cursor']+=p['microbatch']; state['input_tokens']+=inputs; state['target_tokens']+=targets
   for key,n in sources.items(): state['source_tokens'][key]=state['source_tokens'].get(key,0)+n
  scaler.unscale_(optimizer); norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
  scaler.step(optimizer); scaler.update(); state['step']+=1
  elapsed=time.monotonic()-began; throughput=(state['target_tokens']-start_tokens)/max(elapsed,1e-6)
  if state['step']-start_step==p['calibration_steps'] and not (run/'calibration.json').exists():
   available=max(0.,cutoff-time.time()); budget=min(p['target_tokens'],state['target_tokens']+int(throughput*available*.85))
   state['effective_target_tokens']=max(state['target_tokens'],budget)
   calibration={'tokens_per_second':throughput,'seconds_remaining_for_pretrain':available,
                'requested_target_tokens':p['target_tokens'],'effective_target_tokens':state['effective_target_tokens'],
                'peak_gpu_bytes':torch.cuda.max_memory_allocated(device) if device.type=='cuda' else 0}
   atomic_json(run/'calibration.json',calibration); print(json.dumps({'phase':'calibration',**calibration}),flush=True)
  status={'phase':'pretraining','step':state['step'],'input_tokens':state['input_tokens'],'target_tokens':state['target_tokens'],
          'effective_target_tokens':state['effective_target_tokens'],'loss':sum(losses)/len(losses),'grad_norm':float(norm),
          'tokens_per_second':throughput,'elapsed_seconds':elapsed,'remaining_seconds_estimate':
          min(max(0.,cutoff-time.time()),max(0,state['effective_target_tokens']-state['target_tokens'])/max(throughput,1.)),
          'best_validation_loss':state['best_loss'] if math.isfinite(state['best_loss']) else None,'source_tokens':state['source_tokens']}
  if state['step']%p['log_every']==0 or state['step']==1:
   atomic_json(run/'status.json',status)
   with (run/'metrics.jsonl').open('a') as f: f.write(json.dumps(status)+'\n')
   print(json.dumps(status),flush=True)
  if state['step']%p['evaluate_every']==0: select()
  if state['step']%p['save_every']==0: snapshot()
  if state['target_tokens']>=p['pilot_tokens'] and not (run/'pilot.pt').exists():
   save(run/'pilot.pt',{'model':model.state_dict(),'state':dict(state),'evidence':evidence})
   atomic_json(run/'pilot.json',{'target_tokens':state['target_tokens'],'checkpoint_sha256':sha256(run/'pilot.pt')})
 if state['target_tokens']==0: raise RuntimeError('No pretraining budget remains; no trained model can be exported')
 try: select()
 except TimeoutError:
  if not (run/'best.pt').exists(): snapshot(); raise
  atomic_json(run/'final_validation_deferred.json',{'reason':'Reserved NER time reached; retained earlier measured best checkpoint'})
 snapshot()
 # Selected LM checkpoint carries its own achieved token budget, not the final trainer count.
 best=run/'best.pt'; selected=torch.load(best,map_location='cpu',weights_only=False)
 record={'phase':'complete','checkpoint':str(best),'checkpoint_sha256':sha256(best),
         'trainer_target_tokens':state['target_tokens'],'selected_target_tokens':selected['state']['target_tokens'],
         'selection':'heldout_LM_loss_preliminary_before_generation_metrics','validation_loss':selected['validation_loss'],
         'deadline_budgeted':time.time()>=cutoff,'parameters':model.count_parameters()}
 atomic_json(run/'complete.json',record); atomic_json(run/'status.json',record)
 print(json.dumps(record),flush=True)
 del model,optimizer,selected; gc.collect()
 if device.type=='cuda': torch.cuda.empty_cache()
 return record

if __name__=='__main__':
 parser=argparse.ArgumentParser(); parser.add_argument('--config',required=True); args=parser.parse_args()
 config=json.loads(Path(args.config).read_text(encoding='utf-8-sig')); pretrain(config,Path(config['output_root'])/'pretrain')
