"""T4-safe serial RR training, own-model distillation and exact result archives."""
import copy,gc,json,math,pickle,random,time,zipfile
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from sklearn.metrics import precision_recall_fscore_support,confusion_matrix
from .common import atomic_json,save,seed_all,sha,source_hashes,hardware
from .archive_inputs import resolve_bundle
from .data import Tokenizer,load_official,prepare_examples,LABELS,export
from .models import make_model,tensors

def autocast(device):return torch.autocast(device_type=device.type,dtype=torch.float16,enabled=device.type=='cuda')

@torch.no_grad()
def logits_for(model,rows,device,progress=False):
 model.eval();values=[];began=time.time()
 for i,row in enumerate(rows):
  with autocast(device):value=model(*tensors(row,device)).float()
  if not torch.isfinite(value).all():raise FloatingPointError('Nonfinite inference')
  values.append(value[0].cpu().numpy())
  if progress and (i+1)%500==0:print(f'Inference {i+1}/{len(rows)}; elapsed {(time.time()-began)/60:.1f}min',flush=True)
 return np.stack(values) if values else np.empty((0,len(LABELS)),np.float32)

def score(rows,values):
 truth=[r['label'] for r in rows];pred=values.argmax(1);result={}
 for domain in ('all','CL','IT'):
  indices=[i for i,r in enumerate(rows) if domain=='all' or r['domain']==domain]
  if not indices:raise ValueError('Missing dev domain')
  a=np.array(truth)[indices];b=pred[indices]
  precision,recall,f1,support=precision_recall_fscore_support(a,b,labels=list(range(13)),zero_division=0)
  result[domain]={'macro_f1':float(f1.mean()),'macro_precision':float(precision.mean()),'macro_recall':float(recall.mean()),
   'sentences':len(indices),'per_label':{name:{'f1':float(f1[i]),'precision':float(precision[i]),'recall':float(recall[i]),'support':int(support[i])} for i,name in enumerate(LABELS)},
   'confusion_matrix':confusion_matrix(a,b,labels=list(range(13))).tolist()}
 return result

def prepare(config,bundle,dataset):
 root=Path(config['run_root']);root.mkdir(parents=True,exist_ok=True)
 fingerprint={'data':dataset['fingerprint'],'tokenizer':sha(bundle/'tokenizer.json'),'max_tokens':config['max_tokens'],'context_sentences':config['context_sentences']}
 cache=root/'tokenized.pkl'
 if cache.exists():
  with cache.open('rb') as f:obj=pickle.load(f)
  if obj['fingerprint']!=fingerprint:raise ValueError('Tokenization cache differs; choose a new RUN_ROOT')
  return obj['rows']
 tok=Tokenizer(bundle/'tokenizer.json');rows={};stats={}
 for split in ('train','dev','test'):
  rows[split],stats[split]=prepare_examples(dataset[split],tok,config['max_tokens'],config['context_sentences'])
  print('Tokenized',split,stats[split],flush=True)
 tmp=cache.with_suffix('.partial')
 with tmp.open('wb') as f:pickle.dump({'fingerprint':fingerprint,'rows':rows},f)
 tmp.replace(cache);atomic_json(root/'tokenization.json',{'fingerprint':fingerprint,'stats':stats})
 if not rows['train'] or not rows['dev']:raise ValueError('No train/dev sentences')
 return rows

def train_one(config,bundle,dataset,rows,teacher_logits=None):
 if config['max_tokens']<16 or config['context_sentences']<0 or config['accumulation']<1 or config['max_epochs']<1 or config['training_hours']<=0:raise ValueError('Invalid training budget or input dimensions')
 root=Path(config['run_root']);root.mkdir(parents=True,exist_ok=True)
 fingerprint={'config':config,'parent_sha':sha(bundle/'checkpoint.pt'),'tokenizer_sha':sha(bundle/'tokenizer.json'),
  'data':dataset['fingerprint'],'source_hashes':source_hashes(),'teacher_sha':config.get('teacher_logits_sha')}
 done=root/'complete.json'
 if done.exists():
  obj=json.loads(done.read_text())
  if obj['fingerprint']!=fingerprint or sha(obj['submission'])!=obj['sha256']:raise ValueError('Completed run differs')
  print('Already completed:',obj['submission']);return obj
 seed_all(config['seed']);device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
 if device.type!='cuda' and not config.get('allow_cpu',False):raise RuntimeError('Select a T4 GPU runtime before training')
 model,evidence=make_model(bundle,config['family'],config['method'],config['rank'],config['alpha']);model.to(device)
 base=[p for n,p in model.named_parameters() if n.startswith('backbone.') and p.requires_grad]
 head=[p for n,p in model.named_parameters() if not n.startswith('backbone.') and p.requires_grad]
 optimizer=torch.optim.AdamW([{'params':base,'lr':config['backbone_lr']},{'params':head,'lr':config['head_lr']}],weight_decay=.01)
 scaler=torch.amp.GradScaler('cuda',enabled=device.type=='cuda',init_scale=1024)
 state={'epoch':0,'group':0,'step':0,'best':-1.,'bad':0,'trained_sentences':0,'elapsed_training_seconds':0.}
 latest=root/'latest.pt'
 if latest.exists():
  payload=torch.load(latest,map_location='cpu',weights_only=False)
  if payload['fingerprint']!=fingerprint:raise ValueError('Resume fingerprint differs; do not reuse a RUN_ROOT with changed settings')
  model.load_state_dict(payload['model'],strict=True);optimizer.load_state_dict(payload['optimizer']);scaler.load_state_dict(payload['scaler']);state=payload['state']
  torch.set_rng_state(payload['torch_rng']);random.setstate(payload['python_rng']);np.random.set_state(payload['numpy_rng'])
  if device.type=='cuda':torch.cuda.set_rng_state(payload['cuda_rng'].cpu())
  del payload
 evidence.update({'fingerprint':fingerprint,'hardware':hardware(),'labels':LABELS,'initialization':'own_Phase1_weights; new_RR_head_and_optimizer_at_first_start'})
 atomic_json(root/'run.json',evidence)
 from collections import Counter
 counts=Counter(r['label'] for r in rows['train'])
 weights=torch.tensor([min(5.,math.sqrt(len(rows['train'])/(13*max(1,counts[k])))) for k in range(13)],device=device)
 if teacher_logits is not None and teacher_logits.shape!=(len(rows['train']),13):raise ValueError('Teacher logit ordering/shape mismatch')
 # Measure inference on the actual GPU before budgeting export time.
 probe=rows['dev'][:min(32,len(rows['dev']))];start=time.monotonic();logits_for(model,probe,device)
 seconds_per_sentence=(time.monotonic()-start)/len(probe)
 reserve=max(600.,1.6*seconds_per_sentence*(2*len(rows['dev'])+len(rows['test'])))
 hours=config['training_hours'];remaining=max(0.,hours*3600-state['elapsed_training_seconds'])
 train_until=time.monotonic()+max(0.,remaining-reserve)
 atomic_json(root/'calibration.json',{'inference_seconds_per_sentence':seconds_per_sentence,'export_and_validation_reserve_seconds':reserve,'remaining_total_budget_seconds':remaining,'not_a_convergence_guarantee':True})
 print(f'GPU={hardware()["gpu"]}; trainable={evidence["trainable_parameters"]:,}; reserve={reserve/60:.1f}min; remaining budget={remaining/3600:.2f}h',flush=True)
 groups=math.ceil(len(rows['train'])/config['accumulation']);stop=False
 def snapshot():
  save(latest,{'model':model.state_dict(),'optimizer':optimizer.state_dict(),'scaler':scaler.state_dict(),'state':state,'fingerprint':fingerprint,
   'torch_rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state() if device.type=='cuda' else None,'python_rng':random.getstate(),'numpy_rng':np.random.get_state()})
 while state['epoch']<config['max_epochs'] and state['bad']<config['patience'] and not stop:
  order=np.random.default_rng(config['seed']+state['epoch']).permutation(len(rows['train']))
  model.train();complete_epoch=True
  for group in range(state['group'],groups):
   if time.monotonic()>=train_until:complete_epoch=False;stop=True;break
   began=time.monotonic();optimizer.zero_grad(set_to_none=True)
   indices=order[group*config['accumulation']:(group+1)*config['accumulation']]
   scale=min(1.,(state['step']+1)/max(10,groups*.15))*(.2+.8*.5*(1+math.cos(math.pi*min(1.,state['step']/max(1,groups*config['max_epochs'])))))
   for pg,lr in zip(optimizer.param_groups,(config['backbone_lr'],config['head_lr'])):pg['lr']=lr*scale
   loss_value=0.
   for idx in indices:
    row=rows['train'][int(idx)];target=torch.tensor([row['label']],device=device)
    with autocast(device):values=model(*tensors(row,device))
    # FP32 losses for T4 numerical stability.
    loss=F.cross_entropy(values.float(),target,label_smoothing=.02)*(weights[target].mean()/weights.mean())
    if teacher_logits is not None:
     t=config['temperature'];teacher=torch.tensor(teacher_logits[int(idx)][None],device=device)
     kl=F.kl_div(F.log_softmax(values.float()/t,-1),F.softmax(teacher/t,-1),reduction='batchmean')*t*t
     loss=(1-config['distill_weight'])*loss+config['distill_weight']*kl
    if not torch.isfinite(loss):raise FloatingPointError('Nonfinite training loss; preserve checkpoint')
    scaler.scale(loss/len(indices)).backward();loss_value+=float(loss.detach())/len(indices)
   scaler.unscale_(optimizer)
   if state['step']==0 and config['method']!='full':
    active=[n for n,p in model.backbone.named_parameters() if p.requires_grad and p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum()>0]
    if not active:raise RuntimeError('Adapter gradients absent; no long run started')
   norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
   scaler.step(optimizer);scaler.update();state['step']+=1;state['group']=group+1;state['trained_sentences']+=len(indices)
   state['elapsed_training_seconds']+=time.monotonic()-began
   if state['step']%10==0 or state['step']==1:
    status={'phase':'training',**state,'epoch_display':state['epoch']+1,'loss':loss_value,'grad_norm':float(norm)}
    atomic_json(root/'status.json',status);print(json.dumps(status),flush=True)
    with (root/'metrics.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(status)+'\n')
   if state['step']%config['save_every']==0:snapshot()
  if state['step']==0:raise RuntimeError('Budget does not allow even one update; increase hours, reduce context or use another session')
  began=time.monotonic();values=logits_for(model,rows['dev'],device,True);metrics=score(rows['dev'],values);state['elapsed_training_seconds']+=time.monotonic()-began
  value=metrics['all']['macro_f1']
  if value>state['best']+1e-6:
   state['best']=value;state['bad']=0
   save(root/'best.pt',{'model':model.state_dict(),'evidence':evidence,'state':dict(state),'selected_epoch':state['epoch']+1,'partial_epoch':not complete_epoch})
   atomic_json(root/'selection.json',{'metrics':metrics,'epoch':state['epoch']+1,'partial_epoch':not complete_epoch})
   atomic_json(root/'dev_predictions.json',[{'doc':r['doc'],'index':r['index'],'truth':LABELS[r['label']],'prediction':LABELS[int(v.argmax())]} for r,v in zip(rows['dev'],values)])
  else:state['bad']+=1
  print('DEV macroF1=',value,'best=',state['best'],flush=True)
  if complete_epoch:state['epoch']+=1;state['group']=0
  snapshot()
 model.load_state_dict(torch.load(root/'best.pt',map_location='cpu',weights_only=False)['model'],strict=True)
 values=logits_for(model,rows['test'],device,True);pred={d['id']:['']*len(d['text']) for d in dataset['test']}
 for r,v in zip(rows['test'],values):pred[r['doc']][r['index']]=LABELS[int(v.argmax())]
 archive=root/f'UPLOAD_Phase3_{config["name"]}.zip';digest=export(archive,pred,dataset['test'])
 # Run the unmodified organizer-supplied evaluator on our own held-out development labels.
 import subprocess,sys
 dev_records=json.loads((root/'dev_predictions.json').read_text())
 dev_pred={d['id']:['']*len(d['text']) for d in dataset['dev']}
 for row in dev_records:dev_pred[row['doc']][row['index']]=row['prediction']
 atomic_json(root/'dev_prediction.json',dev_pred)
 for domain in ('CL','IT'):atomic_json(root/f'{domain}_dev_truth.json',{d['id']:d['labels'] for d in dataset['dev'] if d['domain']==domain})
 official=subprocess.run([sys.executable,str(Path(__file__).parent/'vendor/evaluate_phase3.py'),
  '--truth-cl',str(root/'CL_dev_truth.json'),'--truth-it',str(root/'IT_dev_truth.json'),
  '--prediction',str(root/'dev_prediction.json'),'--labels',str(Path(config['data_root'])/'Train/label_vocab.json')],capture_output=True,text=True,check=True)
 (root/'official_validation.txt').write_text(official.stdout,encoding='utf-8');print(official.stdout,flush=True)
 # Context ablation changes inputs only, with fixed selected weights; not a new training-method claim.
 ablation=None
 if config['context_sentences']:
  no_context,_=prepare_examples(dataset['dev'],Tokenizer(bundle/'tokenizer.json'),config['max_tokens'],0)
  ablation=score(no_context,logits_for(model,no_context,device,True));atomic_json(root/'context_ablation.json',ablation)
 record={'phase':'complete','submission':str(archive),'sha256':digest,'best_checkpoint_sha256':sha(root/'best.pt'),
  'metrics':json.loads((root/'selection.json').read_text())['metrics'],'trained_sentences':state['trained_sentences'],'epochs_completed':state['epoch'],
  'fingerprint':fingerprint,'hardware':hardware(),'context_ablation':ablation,'leaderboard_score':None}
 atomic_json(done,record);atomic_json(root/'status.json',record)
 print('READY TO UPLOAD:',archive,'held-out macroF1=',record['metrics']['all']['macro_f1'],flush=True)
 del model,optimizer;gc.collect();torch.cuda.empty_cache();return record

def run(config):
 dataset=load_official(config['data_root']);bundle,manifest=resolve_bundle(config['bundle'],Path(config['work_root'])/'inputs')
 if manifest['family']!=config['family']:raise ValueError('Wrong family input bundle')
 rows=prepare(config,bundle,dataset)
 return train_one(config,bundle,dataset,rows)

def distill(config):
 dataset=load_official(config['data_root']);root=Path(config['run_root']);root.mkdir(parents=True,exist_ok=True)
 teacher=copy.deepcopy(config);teacher.update({'name':'D222_teacher','family':'d222','method':'lorma','bundle':config['teacher_bundle'],
  'run_root':str(root/'teacher'),'training_hours':config['teacher_hours'],'backbone_lr':2e-4})
 tb,tm=resolve_bundle(teacher['bundle'],Path(config['work_root'])/'inputs')
 if config.get('existing_teacher_run'):
  teacher_dir=Path(config['existing_teacher_run']);teacher_record=json.loads((teacher_dir/'complete.json').read_text())
  prior=teacher_record['fingerprint'];priorconfig=prior['config']
  if priorconfig['family']!='d222' or priorconfig['method']!='lorma' or prior['parent_sha']!=sha(tb/'checkpoint.pt') or prior['data']!=dataset['fingerprint']:
   raise ValueError('Provided teacher provenance mismatch')
  if sha(teacher_dir/'best.pt')!=teacher_record['best_checkpoint_sha256']:raise ValueError('Provided teacher weights differ')
  teacher=copy.deepcopy(priorconfig);teacher['run_root']=str(teacher_dir)
 else:teacher_record=None
 tr=prepare(teacher,tb,dataset)
 if teacher_record is None:teacher_record=train_one(teacher,tb,dataset,tr)
 cache=root/'teacher_train_logits.npy';meta=root/'teacher_train_logits.json'
 fingerprint={'teacher_best_sha':teacher_record['best_checkpoint_sha256'],'data':dataset['fingerprint'],'ordered_train_sentences':[[r['doc'],r['index']] for r in tr['train']]}
 if cache.exists():
  if not meta.exists() or json.loads(meta.read_text())['fingerprint']!=fingerprint or sha(cache)!=json.loads(meta.read_text())['sha256']:raise ValueError('Teacher cache lineage differs')
 else:
  device=torch.device('cuda');model,evidence=make_model(tb,'d222','lorma',teacher['rank'],teacher['alpha']);model.load_state_dict(torch.load(Path(teacher['run_root'])/'best.pt',map_location='cpu',weights_only=False)['model'],strict=True);model.to(device)
  values=logits_for(model,tr['train'],device,True)
  with cache.with_suffix('.partial').open('wb') as f:np.save(f,values)
  cache.with_suffix('.partial').replace(cache);atomic_json(meta,{'fingerprint':fingerprint,'sha256':sha(cache)})
  del model;gc.collect();torch.cuda.empty_cache()
 student=copy.deepcopy(config);student['method']='full';student['teacher_logits_sha']=sha(cache)
 sb,sm=resolve_bundle(student['bundle'],Path(config['work_root'])/'inputs');sr=prepare(student,sb,dataset)
 if [[r['doc'],r['index']] for r in sr['train']]!=fingerprint['ordered_train_sentences']:raise ValueError('Teacher/student sentence ordering differs')
 result=train_one(student,sb,dataset,sr,np.load(cache))
 atomic_json(root/'distillation_comparison.json',{'teacher_macro_f1':teacher_record['metrics']['all']['macro_f1'],'student_macro_f1':result['metrics']['all']['macro_f1'],
  'teacher_checkpoint_sha':teacher_record['best_checkpoint_sha256'],'student_checkpoint_sha':result['best_checkpoint_sha256'],'temperature':config['temperature'],'distill_weight':config['distill_weight'],
  'teacher_parameters':json.loads((Path(teacher['run_root'])/'run.json').read_text())['parameters'],'student_parameters':json.loads((root/'run.json').read_text())['parameters']})
 return result

def package_records(result):
 root=Path(result['submission']).parent;target=root/f'MODEL_RECORDS_{root.name}.zip'
 files={}
 for name in ('best.pt','run.json','complete.json','selection.json','tokenization.json','dev_predictions.json','context_ablation.json','distillation_comparison.json','teacher_train_logits.json','calibration.json','metrics.jsonl','official_validation.txt','dev_prediction.json','CL_dev_truth.json','IT_dev_truth.json'):
  if (root/name).exists():files['run/'+name]=root/name
 files['submission/'+Path(result['submission']).name]=Path(result['submission'])
 config=result['fingerprint']['config'];bundle,_=resolve_bundle(config['bundle'],Path(config['work_root'])/'inputs')
 files['tokenizer.json']=bundle/'tokenizer.json';files['INPUT_MANIFEST.json']=bundle/'INPUT_MANIFEST.json'
 for p in Path(__file__).parent.rglob('*'):
  if p.is_file() and '__pycache__' not in p.parts:files['source/phase3_rr/'+p.relative_to(Path(__file__).parent).as_posix()]=p
 with zipfile.ZipFile(target.with_suffix('.partial'),'w',allowZip64=True) as z:
  for name,p in files.items():z.write(p,name,compress_type=zipfile.ZIP_STORED if p.suffix=='.pt' else zipfile.ZIP_DEFLATED)
  z.writestr('MANIFEST.json',json.dumps({'files':{name:sha(p) for name,p in files.items()},'note':'Selected full model weights, not an optimizer resume. Keep latest.pt separately in persistent storage. Upload only the nested UPLOAD zip to Codabench.'}))
 with zipfile.ZipFile(target.with_suffix('.partial')) as z:
  if z.testzip() is not None:raise ValueError('Model records CRC failed')
 target.with_suffix('.partial').replace(target)
 atomic_json(root/'records_archive.json',{'path':str(target),'sha256':sha(target)})
 return target
