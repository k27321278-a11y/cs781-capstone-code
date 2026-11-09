"""Download and validate each completed submission; safe to restart."""
import hashlib
import json
import shutil
import subprocess
import sys
import time
import zipfile
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from lner.data import atomic_json,sha256,validate_predictions

CODE=Path(__file__).resolve().parents[1]
CONFIG=json.loads((CODE/'configs/queue.json').read_text())
OUTPUT=CODE.parent/'outputs/phase2_20261011'
OUTPUT.mkdir(parents=True,exist_ok=True)
TEXTS=json.loads((CODE/'data/Phase_2_LNER/Test/fold3-text.json').read_text(encoding='utf-8'))
LABELS=(CODE/'data/Phase_2_LNER/Train/labels.txt').read_text().split()
completed={}

while len(completed)<len(CONFIG['experiments']):
    for spec in CONFIG['experiments']:
        if spec['name'] in completed: continue
        try:
            remote=CONFIG['output_root']+'/'+spec['name']+'/complete.json'
            r=subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=15','stud1','cat',remote],
                             capture_output=True,text=True,timeout=30)
            if r.returncode: continue
            record=json.loads(r.stdout)
            destination=OUTPUT/spec['submission_name']
            if not destination.exists() or sha256(destination)!=record['sha256']:
                partial=destination.with_suffix('.partial')
                subprocess.run(['scp','-B','-q','stud1:'+record['submission'],str(partial)],check=True,timeout=120)
                if sha256(partial)!=record['sha256']: raise ValueError('Downloaded hash mismatch')
                partial.replace(destination)
            with zipfile.ZipFile(destination) as archive:
                if archive.namelist()!=['prediction.json'] or archive.testzip() is not None:
                    raise ValueError('Wrong Codabench ZIP structure')
                predictions=json.loads(archive.read('prediction.json'))
            validate_predictions(predictions,TEXTS,LABELS)
            downloads=Path.home()/'Downloads'/spec['submission_name']
            shutil.copy2(destination,downloads)
            if sha256(downloads)!=record['sha256']: raise ValueError('Downloads copy hash mismatch')
            completed[spec['name']]={**record,'local_path':str(destination),'downloads_path':str(downloads)}
            atomic_json(OUTPUT/'download_inventory.json',completed)
            print('DOWNLOADED_AND_VERIFIED '+spec['submission_name']+' '+record['sha256'],flush=True)
        except Exception as error:
            print(spec['name']+': '+type(error).__name__+': '+str(error),flush=True)
    if len(completed)<len(CONFIG['experiments']): time.sleep(60)
print('ALL_FOUR_SUBMISSIONS_DOWNLOADED_AND_VERIFIED',flush=True)
