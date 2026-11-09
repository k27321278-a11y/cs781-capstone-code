"""CPU-only retention and milestone capture for the authorized resumed LM."""
import json,os,sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
from milestone_watch import prune,capture
from lner.data import atomic_json
root=Path('/media/hdd/SAMBA4080_left_Dataset/stud1/cs781_d222_20261011')
out=root/'continuation_20261011'
old=root/'milestones/200M.json'
if old.exists() and json.loads(old.read_text()).get('phase')=='not_reached':
 archive=out/'previous_records/milestones/200M_before_continuation.json'
 archive.parent.mkdir(parents=True,exist_ok=True)
 if not archive.exists():archive.write_bytes(old.read_bytes())
 old.unlink()
identity=Path('/proc/201701/stat').read_text().split()[21]
try:
 while Path('/proc/201701/stat').exists() and Path('/proc/201701/stat').read_text().split()[21]==identity:
  phase=json.loads((out/'status.json').read_text()).get('phase')
  if phase!='resumed_pretraining':break
  prune(root);capture(root)
  atomic_json(out/'watch_status.json',{'phase':'watching','pid':os.getpid(),'trainer_pid':201701,'time':time.time()})
  time.sleep(60)
 atomic_json(out/'watch_status.json',{'phase':'finished_lm_monitor','time':time.time()})
except BaseException as error:
 atomic_json(out/'watch_failure.json',{'error':str(error),'time':time.time()})
 raise
