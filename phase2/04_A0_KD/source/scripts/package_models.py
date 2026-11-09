"""Package completed Phase 2 models with checkpoint, exact sources and submission."""
import json,zipfile
from pathlib import Path
from lner.data import sha256,atomic_json
code=Path.cwd(); config=json.loads((code/'configs/queue.json').read_text()); root=Path(config['output_root'])
out=root/'archives'; out.mkdir(exist_ok=True)
for spec in config['experiments']:
 run=root/spec['name']; done=run/'complete.json'
 if not done.exists(): continue
 record=json.loads(done.read_text()); target=out/('CS781_Phase2_'+spec['name']+'_Complete.zip')

 assert sha256(run/'best.pt')==record['best_checkpoint_sha256']
 assert sha256(record['submission'])==record['sha256']
 paths=[]
 for directory in ('lner','backbones','vendor','scripts','configs','docs','data'):
  for p in (code/directory).rglob('*'):
   if p.is_file() and '__pycache__' not in p.parts and not p.name.endswith('.before_empty_fix'):
    paths.append((p,'source/'+p.relative_to(code).as_posix()))
 paths.append((code/'README.md','source/README.md'))
 for p in run.iterdir():
  if p.is_file() and p.name!='latest.pt': paths.append((p,'run/'+p.name))
 paths.append((root/'split_manifest.json','run/split_manifest.json'))
 for name,path in config['tokenizers'].items(): paths.append((Path(path),'tokenizers/'+name+'.json'))
 paths.append((Path(record['submission']),'submission/'+spec['submission_name']))
 if spec['method']=='kd':
  teacher=root/spec['teacher']
  evidence=json.loads((run/'run.json').read_text())
  assert sha256(teacher/'best.pt')==evidence['teacher_sha256']
  for name in ('best.pt','run.json','complete.json','selection.json'):
   paths.append((teacher/name,'teacher/'+name))
 if record.get('recovery'):
  for name in ('train.py','data.py'):
   paths.append((code/'lner'/(name+'.before_empty_fix'),'training_source/lner/'+name))
 manifest={'experiment':spec,'completion':record,'files':{name:{'sha256':sha256(p),'bytes':p.stat().st_size} for p,name in paths},
  'contents':'Full selected NER state (including base model weights), source, configs, original tokenizers, official data, run and selection records, exact submission ZIP. Original Phase 1 parent checkpoint identified by run.json SHA; its optimizer state is not included.'}
 temporary=target.with_suffix('.partial')
 with zipfile.ZipFile(temporary,'w',allowZip64=True) as z:
  for p,name in paths: z.write(p,name,compress_type=zipfile.ZIP_STORED if p.suffix=='.pt' else zipfile.ZIP_DEFLATED)
  z.writestr('MANIFEST.json',json.dumps(manifest,indent=2),compress_type=zipfile.ZIP_DEFLATED)
  z.writestr('README_ARCHIVE.md','This is a complete model archive for your records. Upload only the ZIP inside submission/ to Codabench. run/best.pt contains the complete selected adapted NER weights, not an optimizer. source/configs/queue.json records original server paths. Tokenizers are included in tokenizers/. Official data and split IDs are included. Hardware records are actual observations. Training provenance and corrected export provenance are retained separately when applicable.\n',compress_type=zipfile.ZIP_DEFLATED)
 with zipfile.ZipFile(temporary) as z: assert z.testzip() is None
 temporary.replace(target)
 info={'archive':str(target),'sha256':sha256(target),'bytes':target.stat().st_size,'experiment':spec['name'],'files':len(paths)+2}
 atomic_json(target.with_suffix('.manifest.json'),info)
 print(json.dumps(info),flush=True)

