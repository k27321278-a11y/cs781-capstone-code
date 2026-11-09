"""Export finished NER checkpoints after fixing the empty-document edge case."""
import gc,json,subprocess,sys,time
from pathlib import Path
import torch
from lner.data import atomic_json,sha256,export_zip,validate_predictions
from lner.models import build_model
from lner.train import prepare,document_logits,decode,implementation_hashes

config=json.loads(Path('configs/queue.json').read_text())
root=Path(config['output_root'])
dataset,windows=prepare(config)
device=torch.device('cuda')
torch.set_num_threads(4)
for spec in config['experiments'][:3]:
 run=root/spec['name']
 if (run/'complete.json').exists(): continue
 saved=torch.load(run/'latest.pt',map_location='cpu',weights_only=False)
 fp=saved['fingerprint']
 if fp['experiment']!=spec or fp['split_sha256']!=sha256(root/'split_manifest.json'):
  raise ValueError('Saved training provenance differs')
 if not (saved['epoch']>=config['max_epochs'] or saved['bad']>=config['patience']):
  raise ValueError('Training is not finished; export-only recovery refused')
 epochs=saved['epoch']; del saved
 state=torch.load(run/'best.pt',map_location='cpu',weights_only=False)
 if state['evidence']['fingerprint']!=fp: raise ValueError('Selected checkpoint lineage differs')
 model,evidence=build_model(spec,len(dataset['char_vocab'])+2,1+2*len(dataset['labels']))
 if evidence['pretrained_checkpoint_sha256']!=fp['pretrained_sha256']:
  raise ValueError('Parent checkpoint changed')
 model.load_state_dict(state['model'],strict=True); selected_epoch=state['epoch']; del state
 model.to(device).eval()
 selection=json.loads((run/'selection.json').read_text())
 logits=document_logits(model,windows['test'],dataset,spec['encoding'],device,run/'status.json')
 predictions=decode(logits,dataset['labels'],selection['outside_bias'])
 validate_predictions(predictions,dataset['test'],dataset['labels'])
 empty=[k for k,v in dataset['test'].items() if not v]
 assert all(predictions[k]==[] for k in empty)
 archive=root/'deliverables'/spec['submission_name']
 digest=export_zip(archive,predictions,dataset['test'],dataset['labels'])
 atomic_json(run/'validation_truth.json',{k:dataset['gold'][k] for k in dataset['validation_ids']})
 result=subprocess.run([sys.executable,'vendor/evaluate_phase2.py','--truth',str(run/'validation_truth.json'),
  '--prediction',str(run/'validation_predictions.json'),'--labels',str(Path(config['data_root'])/'Train/labels.txt')],
  check=True,capture_output=True,text=True)
 (run/'official_validation.txt').write_text(result.stdout)
 record={'phase':'complete','experiment':spec['name'],'submission':str(archive),'sha256':digest,
  'best_checkpoint_sha256':sha256(run/'best.pt'),'official_ids':len(predictions),
  'entity_spans':sum(map(len,predictions.values())),'validation':selection['metrics'],
  'epochs_completed':epochs,'selected_epoch':selected_epoch,'recovery':'export_only_empty_document_fix',
  'empty_document_ids':empty,'training_implementation_hashes':fp['implementation_hashes'],
  'export_implementation_hashes':implementation_hashes()}
 atomic_json(run/'complete.json',record); atomic_json(run/'status.json',record)
 print(json.dumps(record),flush=True)
 del model,logits; gc.collect(); torch.cuda.empty_cache()
print('THREE_EXISTING_CHECKPOINTS_EXPORTED',flush=True)
