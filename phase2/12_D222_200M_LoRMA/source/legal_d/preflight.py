"""Runtime launch gates; no score claim follows from these invariants."""
import argparse,gc,json,time
from dataclasses import asdict
from pathlib import Path
import torch
from .model import DConfig,EfficientDecoder
from .data import verify_prepared,LMBlocks
from .common import autocast,runtime_evidence,source_hashes
from lner.data import atomic_json

def preflight(config):
 torch.set_num_threads(4); torch.manual_seed(config['seed'])
 device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
 tiny=DConfig(vocab_size=64,layers=4,width=32,query_heads=4,kv_heads=2,head_dim=8,ff_width=64,
              context=128,local_window=8,global_every=2,query_tile=5,checkpointing=False)
 model=EfficientDecoder(tiny).to(device).eval(); ids=torch.randint(4,64,(1,40),device=device)
 with torch.inference_mode():
  whole=model(ids)[0]; changed=ids.clone(); changed[:,24:]=torch.randint(4,64,(1,16),device=device)
  assert torch.allclose(whole[:,:24],model(changed)[0][:,:24],atol=1e-5,rtol=1e-5),'Future token leakage'
  first,_,cache=model(ids[:,:17],use_cache=True); error=0.
  for pos in range(17,40):
   part,_,cache=model(ids[:,pos:pos+1],caches=cache,use_cache=True)
   error=max(error,float((part[0,0]-whole[0,pos]).abs().max()))
   for i,c in enumerate(cache):
    assert c['next_position']==pos+1 and c['k'].shape[1]==tiny.kv_heads
    if (i+1)%tiny.global_every: assert c['k'].shape[2]<=tiny.local_window
  assert error<2e-5,('Cache parity differs',error)
 del model; gc.collect(); torch.cuda.empty_cache()
 provenance=verify_prepared(config); cfg=DConfig(**config['model']); actual=EfficientDecoder(cfg).to(device).train()
 dev=LMBlocks(config['prepared_root'],'dev',config['seed'],cfg.context,config['tokenizer'])
 dev_lengths=[len(dev.at(i,allow_long=True)[1])-1 for i in range(len(dev))]
 longest_index=int(torch.tensor(dev_lengths).argmax())
 assert sum(y.numel() for x,y in dev.chunks(longest_index,device))==dev_lengths[longest_index]
 assert actual.count_parameters()==222079872==cfg.parameter_formula(),'Real parameter count differs'
 sample=LMBlocks(config['prepared_root'],'train',config['seed'],cfg.context,config['tokenizer'])
 x,y,_,_,_=sample.batch(0,1,device); x,y=x[:,:256],y[:,:256]
 with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA]) as profile:
  with autocast(device): _,loss,_=actual(x,y)
  loss.backward()
 gradients=[p.grad for p in actual.parameters() if p.grad is not None]
 assert torch.isfinite(loss) and gradients and all(torch.isfinite(g).all() for g in gradients)
 kernels=[e.key for e in profile.key_averages() if 'attention' in e.key.lower() or 'flash' in e.key.lower()]
 # NER CRF gradient and boundary/empty-document gates use a tiny random fixture, never a foundation claim.
 from .ner import DNER
 from lner.data import validate_predictions
 small=DNER(EfficientDecoder(tiny).to(device),32,25,.1).to(device)
 args=[torch.randint(4,64,(1,12),device=device),(torch.arange(1600,device=device)%12)[None],torch.randint(1,32,(1,1600),device=device),torch.zeros((1,12),dtype=torch.long,device=device)]
 tags=torch.zeros(1600,dtype=torch.long,device=device); tags[:7]=torch.tensor([0,1,2,0,3,4,4],device=device)
 ner_began=time.monotonic()
 with autocast(device): ner_loss=small.loss(args,tags,torch.ones(25,device=device))
 ner_loss.backward(); assert torch.isfinite(ner_loss) and torch.isfinite(small.crf.transitions.grad).all()
 ner_seconds=time.monotonic()-ner_began
 validate_predictions({'empty':[]},{'empty':''},['APP','RESP','A.COUNSEL','R.COUNSEL','JUDGE','WIT','AUTH','COURT','STAT','PREC','DATE','CASENO'])
 record={**runtime_evidence(device),'parameters':actual.count_parameters(),'nominal_config':asdict(cfg),
         'causality':'PASS','cache_max_error':error,'cropped_cache_absolute_positions':'PASS',
         'finite_real_loss':float(loss),'finite_real_gradients':True,'attention_events':kernels,
         'ner_crf_finite_gradient':True,'ner_1600_character_window_seconds':ner_seconds,
         'empty_document':'PASS','data':provenance,'dev_max_tokens':max(dev_lengths),
         'dev_over_context_documents':sum(n>cfg.context for n in dev_lengths),'dev_full_coverage_windows':'PASS',
         'fingerprint':{'config':config,'data':provenance,'source_hashes':source_hashes()}}
 atomic_json(Path(config['output_root'])/'preflight.json',record); print(json.dumps(record),flush=True)
 del actual,small,gradients; gc.collect(); torch.cuda.empty_cache()
 return record

if __name__=='__main__':
 parser=argparse.ArgumentParser(); parser.add_argument('--config',required=True); a=parser.parse_args()
 preflight(json.loads(Path(a.config).read_text(encoding='utf-8-sig')))
