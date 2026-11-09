"""Download each verified D prediction independently of later benchmark stages."""
import json,shlex,shutil,subprocess,sys,time,zipfile
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import legal_d
from lner.data import sha256,atomic_json,validate_predictions
root=Path(__file__).resolve().parents[1]; config=json.loads((root/'configs/d222.json').read_text(encoding='utf-8-sig'))
out=root.parent/'outputs/d222_20261011'; out.mkdir(parents=True,exist_ok=True)
texts=json.loads((root/'data/Phase_2_LNER/Test/fold3-text.json').read_text(encoding='utf-8'))
labels=(root/'data/Phase_2_LNER/Train/labels.txt').read_text().split()
manifest=out/'download_inventory.json'; completed=json.loads(manifest.read_text()) if manifest.exists() else {}
print('D222_DOWNLOADER_STARTED',flush=True)
while True:
 try:
  code="import json,pathlib; r=pathlib.Path("+repr(config['output_root'])+"); print(json.dumps([json.loads(p.read_text()) for p in list(r.glob('ner/*/complete.json'))+list(r.glob('phase1_benchmark/complete.json'))]))"
  result=subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=15','stud1','python3 -c '+shlex.quote(code)],capture_output=True,text=True,timeout=30)
  if result.returncode: raise ConnectionError(result.stderr.strip())
  for record in json.loads(result.stdout):
   name=Path(record['submission']).name; destination=out/name
   if name in completed and destination.exists() and sha256(destination)==record['sha256']: continue
   partial=destination.with_suffix('.partial')
   subprocess.run(['scp','-B','-q','stud1:'+record['submission'],str(partial)],check=True,timeout=120)
   assert sha256(partial)==record['sha256']
   with zipfile.ZipFile(partial) as z:
    assert z.namelist()==['prediction.json'] and z.testzip() is None
    pred=json.loads(z.read('prediction.json'))
    if record.get('phase1_closed'):
     assert len(pred)==1000 and all(len(v['text'].split())==128 for v in pred.values())
    else: validate_predictions(pred,texts,labels); assert len(pred)==35
   partial.replace(destination); downloads=Path.home()/'Downloads'/name; shutil.copy2(destination,downloads)
   assert sha256(downloads)==record['sha256']
   completed[name]={**record,'local_path':str(destination),'downloads_path':str(downloads)}; atomic_json(manifest,completed)
   print('DOWNLOADED_AND_VERIFIED '+name+' '+record['sha256'],flush=True)
  if len(completed)>=3: print('ALL_D222_OUTPUTS_DOWNLOADED',flush=True); break
 except Exception as error: print(type(error).__name__+': '+str(error),flush=True)
 time.sleep(60)
