"""Read-only inspection of our own checkpoints and dataset access."""
import json
from pathlib import Path
import torch

ROOT = Path('/media/hdd/SAMBA4080_left_Dataset/stud1/cs781_phase1_v2')
paths = {
    'legalspan': ROOT/'a_legalspan_v1/experiment/full/checkpoints/latest.pt',
    'b1': ROOT/'own_bpe_20261006/b1_improved/checkpoints/latest.pt',
    'c1': ROOT/'own_bpe_20261006/c1_scratch/checkpoints/latest.pt',
    'a0': ROOT/'a_phase1_5pass_20261003/checkpoints/latest.pt',
}
for name, path in paths.items():
    obj = torch.load(path, map_location='cpu', weights_only=False)
    details = {'name': name, 'keys': list(obj), 'path': str(path)}
    for key in ('model_config', 'config', 'cfg', 'state', 'tokenizer_sha256', 'tokenizer_path'):
        value = obj.get(key)
        if value is not None:
            details[key] = value if isinstance(value, (dict,list,str,int,float)) else str(value)
    model = obj.get('model', obj.get('model_state_dict', {}))
    details['model_keys'] = list(model)[:8]
    print(json.dumps(details, default=str), flush=True)
    del obj
from huggingface_hub import get_token
print(json.dumps({'huggingface_credential_cached': bool(get_token())}))
for p in ROOT.rglob('tokenizer.json'):
    if 'hf_datasets' not in str(p):
        print(json.dumps({'tokenizer': str(p)}))
