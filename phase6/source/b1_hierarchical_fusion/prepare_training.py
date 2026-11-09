import json, pathlib
root=pathlib.Path('experiments/hierarchical_fusion')
old=pathlib.Path(r'C:\Users\Iwill\Documents\Codex\2026-10-28\referenced-chatgpt-conversation-this-is-an-3')
runner=(old/'work/phase6_run_20261103/train_phase6.py').read_text(encoding='utf-8')
runner=runner.replace("('b1_fusion','b1_adalora')","('b1_fusion','b1_adalora','evidence_fusion')")
runner=runner.replace("choices=['b1_fusion'","choices=['evidence_fusion','b1_fusion'")
runner=runner.replace("optimizer=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=args.lr,weight_decay=.01)","new_prefixes=('hierarchy','pointer_')\\n    new_params=[p for n,p in model.named_parameters() if p.requires_grad and n.startswith(new_prefixes)]\\n    old_params=[p for n,p in model.named_parameters() if p.requires_grad and not n.startswith(new_prefixes)]\\n    optimizer=torch.optim.AdamW([{'params':old_params,'lr':args.lr},{'params':new_params,'lr':args.lr*3}],weight_decay=.01)".replace('\\n','\n'))
runner=runner.replace("for row,source,_ in rows:\n            ids,mask=pack_source", "for generation_index,(row,source,_) in enumerate(rows):\n            print(json.dumps({'event':'generation_start','index':generation_index+1,'count':len(rows),'id':row['id']}),flush=True)\n            ids,mask=pack_source")
runner=runner.replace("predictions.append({'id':row['id'],'summary':text})", "predictions.append({'id':row['id'],'summary':text})\n            print(json.dumps({'event':'generation_complete','index':generation_index+1,'count':len(rows),'id':row['id'],'words':len(text.split())}),flush=True)")
runner=runner.replace("'train_nll':float(np.mean(running))", "'train_objective_nll_plus_coverage':float(np.mean(running))")
(root/'train_evidence.py').write_text(runner,encoding='utf-8')
setup="""from pathlib import Path
import sys, os, gc, hashlib, json
code_root=next(p.parent for p in Path('/kaggle/input').rglob('train.json') if (p.parent/'b1_fusion.py').exists())
sys.path.insert(0,str(code_root))
models=next(p.parent for p in Path('/kaggle/input').rglob('b1_latest_model.pt'))
adapters=list(Path('/kaggle/input').rglob('best_adapter.pt'))
assert len(adapters)==1
os.environ['PHASE6_WARM_ADAPTER']=str(adapters[0])
print('Code, parent and trained B1 adapter found',flush=True)
"""
cells=[{'cell_type':'code','metadata':{},'source':setup,'outputs':[],'execution_count':None}]
for name in ['evidence_fusion.py','train_evidence.py']:
 cells.append({'cell_type':'code','metadata':{},'source':'%%writefile '+name+'\n'+(root/name).read_text(),'outputs':[],'execution_count':None})
probe="""import torch, evidence_fusion
assert torch.cuda.is_available()
probe_model,probe_tok,probe_meta=evidence_fusion.build(models,'cuda')
s=torch.tensor([[probe_tok.encode('The court allowed the appeal.')]],device='cuda')
t=torch.tensor([[probe_tok.bos_id]+probe_tok.encode('The appeal was allowed.')+[probe_tok.eos_id]],device='cuda')
with torch.autocast('cuda',dtype=torch.float16):
    probe_loss=evidence_fusion.training_loss(probe_model,s,s.ne(0),t,t.ne(0))
probe_loss.backward()
assert torch.isfinite(probe_loss)
assert probe_model.pointer_gate.weight.grad is not None
assert all(p.grad is None for p in probe_model.parameters() if not p.requires_grad)
print(json.dumps({'event':'cuda_backward_probe_passed','loss':float(probe_loss.detach()),'parent':probe_meta['parent_checkpoint_sha256'],'warm_start_epoch':probe_meta['warm_start_epoch'],'trainable':probe_meta['trainable_parameters']}),flush=True)
del probe_loss,probe_model,s,t
# Backbone hook and temporary model references may form cycles.
gc.collect();torch.cuda.empty_cache()
"""
cells.append({'cell_type':'code','metadata':{},'source':probe,'outputs':[],'execution_count':None})
run="""import train_evidence
sys.argv=['train_evidence.py','--input',str(code_root),'--models-input',str(models),'--output','/kaggle/working/evidence_fusion','--method','evidence_fusion','--epochs','6','--accum','8','--lr','0.00005','--passages','16','--passage-tokens','256','--target-tokens','768','--validation-limit','16','--confirmation-limit','32','--max-train-hours','3','--skip-bertscore']
train_evidence.main()
"""
cells.append({'cell_type':'code','metadata':{},'source':run,'outputs':[],'execution_count':None})
for i,c in enumerate(cells):c['id']='evidence-cell-'+str(i)
stage=root/'kaggle';stage.mkdir(exist_ok=True)
(stage/'run.ipynb').write_text(json.dumps({'nbformat':4,'nbformat_minor':5,'metadata':{'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'}},'cells':cells}),encoding='utf-8')
meta={'id':'bprabha2828/cs781-phase6-hierarchical-evidence-fusion','title':'CS781 Phase6 Hierarchical Evidence Fusion','code_file':'run.ipynb','language':'python','kernel_type':'notebook','is_private':True,'enable_gpu':True,'enable_internet':False,'dataset_sources':['bprabha2828/cs781-phase6-models-runners-20261103','bprabha2828/cs781-phase6-official-data-20261103'],'kernel_sources':['bprabha2828/cs781-phase6-b1-passage-fusion-lorma'],'competition_sources':[]}
(stage/'kernel-metadata.json').write_text(json.dumps(meta,indent=2),encoding='utf-8')
print('Prepared private training notebook and resumable runner')

