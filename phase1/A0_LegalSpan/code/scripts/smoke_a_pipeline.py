"""Actual tiny random model training→generation→published scoring→ZIP evidence.

Entire fixture is explicitly synthetic and cannot become an official submission.
"""
import argparse
import gc
import json
from pathlib import Path
from dataclasses import asdict
import torch
from cs781.contracts import TrainConfig
from cs781.io import write_json_atomic, sha256_file
from cs781.evaluate import score_predictions
from legalspan.bpe import OwnBPE
from legalspan.data import prepare
from legalspan.model import SpanConfig
from legalspan.train import train
from legalspan.pipeline import generate_records, package

p=argparse.ArgumentParser(); p.add_argument('--run',required=True,type=Path); p.add_argument('--scorer-cache',required=True,type=Path); a=p.parse_args()
a.run.mkdir(parents=True,exist_ok=True)
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tests'))
from test_legalspan_data import fixture
work=a.run/'fixture'; work.mkdir(exist_ok=True)
if not (work/'catalog.sqlite').exists():
    db,parent,tok=fixture(work)
    # Synthetic held-out prompts deliberately share the simple grammatical
    # pattern. This verifies mechanics and carries no quality generalization claim.
    text=' '.join((['Section','123','ABC','ordinary.']*128))
    for split in ('dev','confirmation'):
        for suffix,n in (('prompts',512),('references',128)):
            path=parent/f'{split}.{suffix}.json'; records=json.loads(path.read_text())
            for key in records: records[key]['text']=' '.join(text.split()[:n])
            write_json_atomic(path,records)
else: db,parent,tok=work/'catalog.sqlite',work/'parent',work/'tokenizer.json'
data=a.run/'data'
if not data.exists(): prepare(db,parent,tok,data)
cfg=SpanConfig(vocab_size=OwnBPE.load(tok).get_vocab_size(),max_seq_len=8192,
                n_layers=2,d_model=32,n_heads=4,d_head=8,d_ff=64,route_layers=1,route_dim=16)
training=TrainConfig(target_tokens=150000,grad_accum_steps=2,learning_rate=.003,device='cuda',save_every=25,log_every=25)
run=a.run/'training'
train(data,cfg,training,run,resume=(run/'checkpoints/latest.pt').exists(),smoke=True)
gc.collect(); torch.cuda.empty_cache()
records=json.loads((data/'dev.prompts.json').read_text()); key=next(iter(records))
prompts={key:records[key]['text']}; output=a.run/'prediction.json'
failures=[]
for policy in range(3):
    try:
        predictions,fingerprint=generate_records(run/'candidates/final.pt',data,prompts,policy,a.run/f'policy{policy}.json',allow_smoke=True)
        write_json_atomic(output,{k:{'text':v} for k,v in predictions.items()}); break
    except ValueError as exc: failures.append(str(exc))
else: raise RuntimeError('Tiny fixture generation failed all policies: '+str(failures))
reference=a.run/'reference.json'; refs=json.loads((data/'dev.references.json').read_text())
write_json_atomic(reference,{key:refs[key]})
metrics=score_predictions(reference,output,Path(__file__).resolve().parents[1]/'vendor/phase1_evaluate.py',a.scorer_cache)
digest=package(predictions,prompts,a.run/'ENGINEERING_ONLY.zip')
write_json_atomic(a.run/'smoke_evidence.json',{'engineering_only':True,'training_checkpoint_sha256':sha256_file(run/'candidates/final.pt'),
                   'trained_target_tokens':training.target_tokens,'actual_generated_words':len(predictions[key].split()),
                   'published_metrics':metrics,'zip_sha256':digest,'generation':fingerprint,'policy_failures':failures})
print(json.dumps({'smoke_complete':True,'metrics':metrics,'zip_sha256':digest}),flush=True)
