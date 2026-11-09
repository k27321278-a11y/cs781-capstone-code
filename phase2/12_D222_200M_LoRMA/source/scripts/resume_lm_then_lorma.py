"""Resume original full LM state, then export an isolated second LoRMA result."""
import copy,fcntl,json,os,signal,sys,time,traceback
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from legal_d.pretrain import pretrain
from legal_d.ner_train import prepare_ner,train_candidate
from legal_d.common import deadline
from lner.data import atomic_json,sha256

code=Path(__file__).resolve().parents[1]
original=json.loads((code/'configs/d222.json').read_text())
root=Path(original['output_root']); out=root/'continuation_20261011'
out.mkdir(exist_ok=True)
lock=(out/'queue.lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
assert os.environ.get('CUDA_VISIBLE_DEVICES')=='1'
try:
 # Defer only the identified own CPU/GPU milestone queue. Keep all saved states.
 for pid,fragment in ((189136,b'milestone_watch.py'),(200224,b'milestone_ner.py')):
  proc=Path(f'/proc/{pid}/cmdline')
  if proc.exists():
   assert fragment in proc.read_bytes()
   os.kill(pid,signal.SIGTERM)
 for _ in range(100):
  if not Path('/proc/200224').exists():break
  time.sleep(.2)
 else:raise RuntimeError('Own milestone trainer did not exit; no overlapping trainer launched')
 import subprocess
 gpu=subprocess.check_output(['nvidia-smi','-i','1','--query-gpu=uuid','--format=csv,noheader'],text=True).strip()
 apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader'],text=True)
 if gpu in apps:raise RuntimeError('GPU1 occupied; preserve unrelated process')
 history=out/'previous_records';history.mkdir(exist_ok=True)
 for relative in ('pretrain/complete.json','pretrain/status.json','queue_status.json','milestones/watch_status.json','milestones/200M.json'):
  path=root/relative
  if path.exists():
   destination=history/relative;destination.parent.mkdir(parents=True,exist_ok=True)
   if not destination.exists():destination.write_bytes(path.read_bytes())
 atomic_json(out/'status.json',{'phase':'resumed_pretraining','pid':os.getpid(),'original_config':original,'old_submissions_preserved':True,'planned_cutoff_ist':'20:54 (around user requested 20:50)'})
 base=pretrain(original,root/'pretrain',resume=True)
 config=copy.deepcopy(original);config['output_root']=str(out)
 config['continuation_parent']={'checkpoint':base['checkpoint'],'sha256':base['checkpoint_sha256'],'selected_tokens':base['selected_target_tokens']}
 atomic_json(out/'config.json',config)
 atomic_json(out/'status.json',{'phase':'lorma','pretrained':base})
 data,windows=prepare_ner(config)
 candidate=dict(next(c for c in config['ner']['candidates'] if c['method']=='lorma'))
 candidate['name']='lorma_continued'
 result=train_candidate(config,candidate,base['checkpoint'],data,windows,export=True,train_until=deadline(config)-2400)
 import shutil
 archive=out/'deliverables/UPLOAD_Phase2_D222_Continued_LORMA.zip'
 shutil.copy2(result['submission'],archive)
 assert sha256(archive)==result['sha256']
 result.update({'submission':str(archive),'selected_lm_tokens':base['selected_target_tokens'],'pretraining_record':base})
 atomic_json(out/'complete.json',result)
 atomic_json(out/'status.json',{'phase':'complete','result':result})
 print(json.dumps(result),flush=True)
except BaseException as error:
 atomic_json(out/'failure.json',{'error':str(error),'traceback':traceback.format_exc()})
 raise
