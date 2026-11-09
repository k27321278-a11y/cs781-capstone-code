"""Explicit, audited deadline extension of deferred milestone NER after submission."""
import copy,json,os,sys,time,traceback
from datetime import datetime,timezone
from pathlib import Path
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from legal_d.ner_train import prepare_ner,train_candidate
from legal_d.common import source_hashes,save
from lner.data import sha256,atomic_json
code=Path(__file__).resolve().parents[1];original=json.loads((code/'configs/d222.json').read_text());base=Path(original['output_root'])
for million in (100,200):
 try:
  milestone=json.loads((base/f'milestones/{million}M.json').read_text());assert milestone['phase']=='captured'
  checkpoint=milestone['checkpoint'];assert sha256(checkpoint)==milestone['sha256']
  assert source_hashes()==milestone['original_fingerprint']['source_hashes']
  root=base/f'milestones/ner_{million}M_afterdeadline';root.mkdir(exist_ok=True)
  configpath=root/'config.json'
  if configpath.exists():config=json.loads(configpath.read_text())
  else:
   config=copy.deepcopy(original);config['output_root']=str(root)
   config['official_phase2_deadline_utc']=original['pretrain']['deadline_utc']
   config['pretrain']['deadline_utc']=datetime.fromtimestamp(time.time()+3*3600,timezone.utc).isoformat()
   config['milestone']={'actual_tokens':milestone['actual_target_tokens'],'sha256':milestone['sha256']}
   atomic_json(configpath,config)
  dataset,windows=prepare_ner(config)
  candidate=dict(next(c for c in config['ner']['candidates'] if c['method']=='lorma'));candidate['name']=f'lorma_{million}M'
  run=root/'ner'/candidate['name'];run.mkdir(parents=True,exist_ok=True)
  old=base/f'milestones/ner_{million}M/ner'/candidate['name']/'latest.pt'
  if old.exists() and not (run/'latest.pt').exists():
   payload=torch.load(old,map_location='cpu',weights_only=False)
   prior=configpath.parent.parent/f'ner_{million}M/config.json'
   oldconfig=json.loads(prior.read_text())
   expected={'candidate':candidate,'config':oldconfig,'pretrained_sha256':sha256(checkpoint),'split_sha256':sha256(old.parent.parent.parent/'ner_split.json'),'source_hashes':source_hashes()}
   assert payload['fingerprint']==expected,'Original full resume lineage differs'
   fingerprint={**expected,'config':config,'split_sha256':sha256(root/'ner_split.json')}
   assert expected['split_sha256']==fingerprint['split_sha256']
   atomic_json(root/'resume_migration.json',{'reason':'Authorized deferred adaptation after original deadline','parent_path':str(old),'parent_sha256':sha256(old),'original_fingerprint':expected,'new_fingerprint':fingerprint,'changes':'Output directory and documented runtime deadline only; model/optimizer/RNG/state retained','time':time.time()})
   payload['fingerprint']=fingerprint;save(run/'latest.pt',payload);del payload
   import shutil
   for name in ('best.pt','selection.json','validation_predictions.json'):
    if (old.parent/name).exists():shutil.copy2(old.parent/name,run/name)
  result=train_candidate(config,candidate,checkpoint,dataset,windows,export=True,train_until=datetime.fromisoformat(config['pretrain']['deadline_utc']).timestamp()-3600)
  import shutil
  dest=base/f'milestones/deliverables/UPLOAD_Phase2_D222_{million}M_LORMA.zip';dest.parent.mkdir(exist_ok=True)
  shutil.copy2(result['submission'],dest);assert sha256(dest)==result['sha256']
  result.update({'submission':str(dest),'actual_lm_tokens':milestone['actual_target_tokens'],'ready_before_official_deadline':False,'official_phase2_deadline_utc':original['pretrain']['deadline_utc']})
  atomic_json(base/f'milestones/{million}M_phase2_complete.json',result)
 except BaseException as error:
  atomic_json(base/f'milestones/{million}M_afterdeadline_failure.json',{'error':str(error),'traceback':traceback.format_exc()});raise
