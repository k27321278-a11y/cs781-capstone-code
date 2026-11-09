"""One identified GPU1 queue; NER delivery precedes the closed Phase 1 benchmark."""
import argparse,fcntl,json,time,traceback
from pathlib import Path
from .common import deadline,source_hashes
from .data import verify_prepared
from .pretrain import pretrain
from .ner_train import run_ner
from .lm_eval import run_lm_benchmark
from lner.data import atomic_json,sha256

def main(config):
 root=Path(config['output_root']); root.mkdir(parents=True,exist_ok=True)
 lock=(root/'queue.lock').open('a+'); fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 status=root/'queue_status.json'
 try:
  trained=root/'pretrain/complete.json'
  if trained.exists():
   base=json.loads(trained.read_text()); assert sha256(base['checkpoint'])==base['checkpoint_sha256']
   evidence=json.loads((root/'pretrain/run.json').read_text())
   expected={'config':config,'data':verify_prepared(config),'source_hashes':source_hashes()}
   if evidence['fingerprint']!=expected: raise ValueError('Completed pretraining lineage differs')
  else:
   atomic_json(status,{'phase':'pretraining'}); base=pretrain(config,root/'pretrain')
  atomic_json(status,{'phase':'ner','pretrained':base})
  results=run_ner(config,base['checkpoint'])
  atomic_json(status,{'phase':'phase2_complete','submissions':results,'pretrained':base})
 except Exception as error:
  atomic_json(root/'failure.json',{'error':str(error),'traceback':traceback.format_exc()})
  raise
 # This is independent of the time-sensitive Phase 2 delivery; failures never erase it.
 if time.time()>=deadline(config) and not config.get('allow_postdeadline_benchmark',False):
  atomic_json(status,{'phase':'phase2_complete_benchmark_deferred','submissions':results,'pretrained':base})
  return
 try:
  atomic_json(status,{'phase':'phase1_benchmark','submissions':results,'pretrained':base})
  benchmark=run_lm_benchmark(config)
  atomic_json(status,{'phase':'complete','submissions':results,'phase1_benchmark':benchmark,'pretrained':base})
 except Exception as error:
  atomic_json(root/'benchmark_failure.json',{'error':str(error),'traceback':traceback.format_exc()})
  atomic_json(status,{'phase':'phase2_complete_benchmark_failed','submissions':results,'pretrained':base,'benchmark_error':str(error)})
  raise

if __name__=='__main__':
 p=argparse.ArgumentParser(); p.add_argument('--config',required=True); a=p.parse_args()
 main(json.loads(Path(a.config).read_text(encoding='utf-8-sig')))
