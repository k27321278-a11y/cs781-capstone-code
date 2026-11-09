"""Authorized early LM handoff; preserve best/latest and use unchanged training source."""
import fcntl,json,os,signal,subprocess,sys,time
from pathlib import Path
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from legal_d.common import source_hashes
from lner.data import atomic_json,sha256

code=Path(__file__).resolve().parents[1]
config=json.loads((code/'configs/d222.json').read_text())
root=Path(config['output_root'])
assert str(root)=='/media/hdd/SAMBA4080_left_Dataset/stud1/cs781_d222_20261011'
pid=153218
command=Path(f'/proc/{pid}/cmdline').read_bytes()
assert b'legal_d.pipeline' in command and b'configs/d222.json' in command
assert not (root/'pretrain/complete.json').exists()
handoff=root/'early_phase2_handoff'; handoff.mkdir(exist_ok=True)
assert not (handoff/'complete.json').exists()
identity=Path(f'/proc/{pid}/stat').read_text().split()[21]
atomic_json(handoff/'request.json',{'reason':'User authorized best checkpoint now for Phase 2','pid':pid,'identity':identity,'requested_at':time.time(),'original_config':config})
# Freeze only the identified own queue while reading immutable saved checkpoints.
os.kill(pid,signal.SIGSTOP)
try:
 torch.set_num_threads(1)
 original=json.loads((root/'pretrain/run.json').read_text())
 best=torch.load(root/'pretrain/best.pt',map_location='cpu',weights_only=False)
 latest=torch.load(root/'pretrain/latest.pt',map_location='cpu',weights_only=False)
 assert latest['fingerprint']==original['fingerprint']
 assert best['evidence']['fingerprint']==original['fingerprint']
 assert source_hashes()==original['fingerprint']['source_hashes']
 assert all(k in latest for k in ('optimizer','scaler','rng','state','model'))
 frozen=handoff/'selected_best.pt'
 with (root/'pretrain/best.pt').open('rb') as src,frozen.open('xb') as dst:
  import shutil
  shutil.copyfileobj(src,dst)
 digest=sha256(frozen)
 record={'phase':'complete','checkpoint':str(frozen),'checkpoint_sha256':digest,
  'trainer_target_tokens':latest['state']['target_tokens'],
  'selected_target_tokens':best['state']['target_tokens'],
  'selection':'best_measured_heldout_LM_loss_at_user_requested_early_handoff',
  'validation_loss':best['validation_loss'],'deadline_budgeted':False,'parameters':222079872,
  'early_handoff':True,'latest_full_resume':str(root/'pretrain/latest.pt'),
  'latest_step':latest['state']['step'],'latest_saved_tokens':latest['state']['target_tokens'],
  'live_status_at_stop':json.loads((root/'pretrain/status.json').read_text()),
  'stopped_at':time.time()}
 atomic_json(handoff/'selected.json',record)
 del best,latest
except BaseException:
 os.kill(pid,signal.SIGCONT)
 raise
# TERM is restricted to our verified process identity. Full resume state remains intact.
assert Path(f'/proc/{pid}/stat').read_text().split()[21]==identity
os.kill(pid,signal.SIGTERM); os.kill(pid,signal.SIGCONT)
for _ in range(120):
 if not Path(f'/proc/{pid}').exists(): break
 time.sleep(.5)
else: raise RuntimeError('Original queue still alive; no second GPU trainer launched')
# The old CPU watcher would treat the intentional switch as an incomplete queue.
watch_pid=int((root/'milestones/watch.pid').read_text())
if Path(f'/proc/{watch_pid}/cmdline').exists():
 assert b'milestone_watch.py' in Path(f'/proc/{watch_pid}/cmdline').read_bytes()
 os.kill(watch_pid,signal.SIGTERM)
 for _ in range(40):
  if not Path(f'/proc/{watch_pid}').exists(): break
  time.sleep(.25)
queue_lock=(root/'queue.lock').open('a+')
fcntl.flock(queue_lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
atomic_json(root/'pretrain/complete.json',record)
atomic_json(root/'pretrain/status.json',record)
atomic_json(root/'milestones/200M.json',{'phase':'not_reached','requested_target_tokens':200000000,'actual_target_tokens':record['trainer_target_tokens'],'reason':'User requested early switch to Phase 2'})
atomic_json(handoff/'complete.json',record)
fcntl.flock(queue_lock,fcntl.LOCK_UN);queue_lock.close()
env=dict(os.environ,CUDA_VISIBLE_DEVICES='1',OMP_NUM_THREADS='4')
log=root/('queue.phase2_early.%d.log'%time.time())
with log.open('ab') as output:
 child=subprocess.Popen([sys.executable,'-u','-m','legal_d.pipeline','--config','configs/d222.json'],cwd=code,env=env,stdout=output,stderr=subprocess.STDOUT,start_new_session=True)
(root/'queue.pid').write_text(str(child.pid))
(root/'queue_log_path.txt').write_text(str(log))
with (root/'milestones/watch.log').open('ab') as output:
 watcher=subprocess.Popen([sys.executable,'-u',str(code/'scripts/milestone_watch.py'),'--config',str(code/'configs/d222.json')],cwd=code,env=env,stdout=output,stderr=subprocess.STDOUT,start_new_session=True)
(root/'milestones/watch.pid').write_text(str(watcher.pid))
atomic_json(handoff/'launch.json',{'trainer_pid':child.pid,'watcher_pid':watcher.pid,'log':str(log),'checkpoint':record,'time':time.time()})
print(json.dumps({'trainer_pid':child.pid,'watcher_pid':watcher.pid,'log':str(log),'checkpoint':record}),flush=True)
