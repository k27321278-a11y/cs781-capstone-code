import json,shutil
from pathlib import Path
root=Path(__file__).resolve().parents[1]
src=Path('C:/Users/Iwill/Downloads/Phase3_D222_Input.zip')
out=root/'outputs/phase3_t4/upload_parts';out.mkdir(exist_ok=True)
with src.open('rb') as source:
 index=1
 while True:
  block=source.read(90_000_000)
  if not block:break
  p=out/(src.name+f'.part{index:03d}');p.write_bytes(block)
  print(p,len(block),flush=True);index+=1
p=root/'outputs/phase3_t4/01_D222_LoRMA.ipynb'
nb=json.loads(p.read_text());s=''.join(nb['cells'][3]['source'])
if 'BUNDLE_PATHS={}' not in s:
 start=s.index('missing=[str(INPUT_DIR/n)');end=s.index("print('Persistent run:'",start)
 replacement="""BUNDLE_PATHS={}
for name in required:
    original=INPUT_DIR/name
    if original.exists():
        BUNDLE_PATHS[name]=str(original)
        continue
    parts=sorted(INPUT_DIR.glob(name+'.part[0-9][0-9][0-9]'))
    assert parts, 'Upload '+name+' or its numbered parts to '+str(INPUT_DIR)
    assert [p.name[-3:] for p in parts]==[f'{i:03d}' for i in range(1,len(parts)+1)], 'Missing input part'
    combined=WORK_ROOT/name
    if not combined.exists() or combined.stat().st_size!=sum(p.stat().st_size for p in parts):
        import shutil
        temporary=combined.with_suffix('.partial')
        with temporary.open('wb') as target:
            for part in parts:
                print('Joining',part.name,flush=True)
                with part.open('rb') as source:shutil.copyfileobj(source,target,8*1024*1024)
        temporary.replace(combined)
    BUNDLE_PATHS[name]=str(combined)
"""
 s=s[:start]+replacement+s[end:];nb['cells'][3]['source']=s.splitlines(keepends=True)
 s=''.join(nb['cells'][5]['source'])
 s=s.replace("bundle=str(INPUT_DIR/'Phase3_D222_Input.zip')","bundle=BUNDLE_PATHS['Phase3_D222_Input.zip']")
 nb['cells'][5]['source']=s.splitlines(keepends=True);p.write_text(json.dumps(nb,indent=1))
print('Notebook supports split input upload',flush=True)
