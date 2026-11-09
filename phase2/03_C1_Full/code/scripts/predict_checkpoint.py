"""Reproduce predictions from an extracted complete archive; no parent checkpoint needed."""
import argparse,json,sys
from pathlib import Path
import torch
from lner.data import load_dataset,OffsetTokenizer,prepare_windows,sha256,export_zip
from lner.models import CharacterNER,inject_adapters
from lner.train import document_logits,decode

p=argparse.ArgumentParser(); p.add_argument('--model-dir',required=True); p.add_argument('--output',required=True)
args=p.parse_args(); root=Path(args.model_dir).resolve()
manifest=json.loads((root/'MANIFEST.json').read_text()); spec=manifest['experiment']
checkpoint=root/'run/best.pt'
assert sha256(checkpoint)==manifest['completion']['best_checkpoint_sha256']
state=torch.load(checkpoint,map_location='cpu',weights_only=False)
cfg=state['evidence']['pretrained_model_config']; family=spec['family']
if family=='a0':
 from cs781.contracts import ModelConfig
 from cs781.models.a import LegalGPT
 base=LegalGPT(ModelConfig(**cfg))
elif family=='legalspan':
 from legalspan.model import SpanConfig,LegalSpanGPT
 base=LegalSpanGPT(SpanConfig(**cfg))
elif family=='b1':
 from cs781.models.b import EncoderDecoderConfig,EncoderDecoderLegalModel
 base=EncoderDecoderLegalModel(EncoderDecoderConfig(**cfg))
elif family=='c1':
 from cs781.models.c import RecurrentConfig,RecurrentLegalModel
 base=RecurrentLegalModel(RecurrentConfig(**cfg))
else: raise ValueError('Unknown family')
if spec['method'] in ('lora','lorma'):
 inject_adapters(base,spec['method'],spec.get('rank',8),spec.get('alpha',32),.05)
config=json.loads((root/'source/configs/queue.json').read_text())
dataset=load_dataset(root/'source/data/Phase_2_LNER',config['seed'])
split=json.loads((root/'run/split_manifest.json').read_text())
assert dataset['train_ids']==split['train_ids'] and dataset['validation_ids']==split['validation_ids']
assert dataset['char_vocab']==split['char_vocab']
model=CharacterNER(base,family,len(dataset['char_vocab'])+2,1+2*len(dataset['labels']))
model.load_state_dict(state['model'],strict=True); del state
# Tokenize the official test using the same shared window limits and both original tokenizers.
tokenizers={k:OffsetTokenizer(root/'tokenizers'/(k+'.json')) for k in config['tokenizers']}
windows=prepare_windows(dataset,tokenizers,config['max_chars'],config['overlap'],config['max_tokens'])
device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'); torch.set_num_threads(4)
model.to(device).eval()
selection=json.loads((root/'run/selection.json').read_text())
logits=document_logits(model,windows['test'],dataset,spec['encoding'],device)
predictions=decode(logits,dataset['labels'],selection['outside_bias'])
print(json.dumps({'output':args.output,'sha256':export_zip(args.output,predictions,dataset['test'],dataset['labels']),'ids':len(predictions)}))
