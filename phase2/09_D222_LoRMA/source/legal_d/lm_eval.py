"""Phase 1-style validation and export; this does not submit to a closed phase."""
import gc,json,time,zipfile,hashlib
from pathlib import Path
import torch
from .model import DConfig,EfficientDecoder
from .common import autocast
from legalspan.bpe import OwnBPE
from lner.data import atomic_json,sha256
from cs781.evaluate import score_predictions

@torch.inference_mode()
def generate(model,tokenizer,prompt,policy):
 if len(prompt.split())!=512: raise ValueError('Expected an exact 512-word prompt')
 device=next(model.parameters()).device; ids=[tokenizer.token_to_id('<bos>')]+tokenizer.encode(prompt).ids
 if len(ids)+2>model.cfg.context: raise ValueError('Prompt exceeds configured inference context; no truncation')
 x=torch.tensor([ids],dtype=torch.long,device=device); cache=None; generated=[]
 rng=torch.Generator(device=device).manual_seed(policy.get('seed',781)); model.eval()
 for _ in range(min(2048,model.cfg.context-len(ids))):
  with autocast(device): logits,_,cache=model(x,caches=cache,use_cache=True)
  values=logits[0,-1].float().clone()
  if not torch.isfinite(values).all(): raise FloatingPointError('Nonfinite decoding values')
  values[:4]=-torch.inf
  repetition=policy.get('repetition_penalty',1.05)
  if generated:
   index=torch.tensor(sorted(set(generated)),device=device); old=values[index]
   values[index]=torch.where(old<0,old*repetition,old/repetition)
  if policy['strategy']=='greedy': token=int(values.argmax())
  else:
   sorted_values,index=(values/policy.get('temperature',.8)).sort(descending=True)
   probabilities=sorted_values.softmax(-1); excluded=probabilities.cumsum(-1)-probabilities>policy.get('top_p',.9)
   sorted_values[excluded]=-torch.inf; token=int(index[torch.multinomial(sorted_values.softmax(-1),1,generator=rng)])
  generated.append(token); text=tokenizer.decode(generated); words=text.split()
  if len(words)>128 or (len(words)==128 and text[-1:].isspace()): return ' '.join(words[:128])
  x=torch.tensor([[token]],dtype=torch.long,device=device)
 raise RuntimeError('No 128 complete words within generation cap; no filler added')

def predict(model,tok,prompts,policy,path,lineage):
 path=Path(path); result=json.loads(path.read_text()) if path.exists() else {}
 lineage={**lineage,'policy':policy,'prompts_sha256':hashlib.sha256(json.dumps(prompts,sort_keys=True,ensure_ascii=False).encode()).hexdigest()}
 context=path.with_suffix('.lineage.json')
 if context.exists():
  if json.loads(context.read_text())!=lineage: raise ValueError('Generation resume lineage differs')
 elif result: raise ValueError('Generation provenance missing')
 else: atomic_json(context,lineage)
 if set(result)-set(prompts): raise ValueError('Generation resume IDs differ')
 for key,obj in prompts.items():
  if key in result: continue
  text=obj['text'] if isinstance(obj,dict) else obj
  result[key]={'text':generate(model,tok,text,policy)}; atomic_json(path,result)
  print(json.dumps({'phase':'lm_generation','file':path.name,'completed':len(result),'total':len(prompts)}),flush=True)
 if set(result)!=set(prompts) or any(len(v['text'].split())!=128 for v in result.values()): raise ValueError('Invalid continuation result')
 return result

def run_lm_benchmark(config):
 root=Path(config['output_root']); eval_root=root/'phase1_benchmark'; eval_root.mkdir(exist_ok=True)
 inventory=json.loads((root/'pretrain/candidate_inventory.json').read_text())
 tok=OwnBPE.load(config['tokenizer']); device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
 prepared=Path(config['prepared_root']); prompts=json.loads((prepared/'dev.prompts.json').read_text())
 references=prepared/'dev.references.json'; evaluator=Path(__file__).resolve().parents[1]/'vendor/phase1_evaluate.py'
 policies=[{'strategy':'greedy','repetition_penalty':1.05},{'strategy':'nucleus','temperature':.8,'top_p':.9,'repetition_penalty':1.08,'seed':781}]
 records=[]
 for item in inventory:
  payload=torch.load(item['path'],map_location='cpu',weights_only=False); cfg=DConfig(**payload['evidence']['model_config']); cfg.context=8192
  model=EfficientDecoder(cfg); model.load_state_dict(payload['model'],strict=True); del payload; model.to(device).eval()
  for i,policy in enumerate(policies):
   output=eval_root/(Path(item['path']).stem+'_policy'+str(i)+'_dev.json')
   lineage={'checkpoint_sha256':sha256(item['path']),'tokenizer_sha256':sha256(config['tokenizer']),'inference_context':8192}
   predict(model,tok,prompts,policy,output,lineage)
   metrics=score_predictions(references,output,evaluator,config['scorer_cache'])
   records.append({'checkpoint':item['path'],'checkpoint_sha256':sha256(item['path']),'policy':policy,'metrics':metrics})
   atomic_json(eval_root/'candidate_scores.json',records)
  del model; gc.collect(); torch.cuda.empty_cache()
 chosen=max(records,key=lambda x:(x['metrics']['bertscore_f1'],x['metrics']['rouge_l_f1'],x['metrics']['bleu4']))
 atomic_json(eval_root/'selection.json',chosen)
 payload=torch.load(chosen['checkpoint'],map_location='cpu',weights_only=False); cfg=DConfig(**payload['evidence']['model_config']); cfg.context=8192
 model=EfficientDecoder(cfg); model.load_state_dict(payload['model'],strict=True); del payload; model.to(device).eval()
 confirmation=json.loads((prepared/'confirmation.prompts.json').read_text()); output=eval_root/'confirmation.json'
 lineage={'checkpoint_sha256':chosen['checkpoint_sha256'],'tokenizer_sha256':sha256(config['tokenizer']),'inference_context':8192}
 predict(model,tok,confirmation,chosen['policy'],output,lineage)
 confirmation_metrics=score_predictions(prepared/'confirmation.references.json',output,evaluator,config['scorer_cache'])
 atomic_json(eval_root/'confirmation_metrics.json',confirmation_metrics)
 official=json.loads(Path(config['phase1_test']).read_text()); pred=predict(model,tok,official,chosen['policy'],eval_root/'official_predictions.json',lineage)
 if len(pred)!=1000: raise ValueError('Official Phase 1 test must have 1000 IDs')
 archive=root/'deliverables/BENCHMARK_ONLY_Phase1_D222.zip'; archive.parent.mkdir(exist_ok=True)
 with zipfile.ZipFile(archive.with_suffix('.partial'),'w',zipfile.ZIP_DEFLATED) as z: z.writestr('prediction.json',json.dumps(pred,ensure_ascii=False))
 with zipfile.ZipFile(archive.with_suffix('.partial')) as z: assert z.namelist()==['prediction.json'] and z.testzip() is None
 archive.with_suffix('.partial').replace(archive)
 record={'phase':'complete','submission':str(archive),'sha256':sha256(archive),'phase1_closed':True,
         'selection':chosen,'confirmation_metrics':confirmation_metrics,'inference_context':8192}
 atomic_json(eval_root/'complete.json',record); print(json.dumps(record),flush=True); return record
