import argparse
import contextlib
import fcntl
import gc
import json
import math
import os
import pickle
import random
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from .data import (OffsetTokenizer,atomic_json,constrained_decode,export_zip,
                   load_dataset,prepare_windows,sha256,strict_score,validate_predictions)
from .models import build_model,tensor_row


def save_torch(path,value):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix(path.suffix+'.tmp')
    with open(temp,'wb') as handle:
        torch.save(value,handle); handle.flush(); os.fsync(handle.fileno())
    os.replace(temp,path)


def implementation_hashes():
    root=Path(__file__).resolve().parents[1]
    paths=[p for directory in ('lner','backbones','vendor')
           for p in (root/directory).rglob('*.py')]
    return {str(p.relative_to(root)):sha256(p) for p in sorted(paths)}


def autocast(device):
    if device.type!='cuda': return contextlib.nullcontext()
    return torch.autocast('cuda',dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16)


def prepare(config):
    root=Path(config['output_root']); root.mkdir(parents=True,exist_ok=True)
    for spec in config['experiments']:
        if sha256(spec['tokenizer'])!=sha256(config['tokenizers'][spec['encoding']]):
            raise ValueError('Configured encoding differs from the checkpoint tokenizer')
    files=sorted(Path(config['data_root']).rglob('*.json'))+[Path(config['data_root'])/'Train/labels.txt']
    fingerprint={'data':{str(p.relative_to(config['data_root'])):sha256(p) for p in files},
                 'tokenizers':{k:sha256(v) for k,v in config['tokenizers'].items()},
                 'max_chars':config['max_chars'],'overlap':config['overlap'],
                 'max_tokens':config['max_tokens'],'seed':config['seed']}
    cache=root/'prepared.pkl'
    if cache.exists():
        with cache.open('rb') as handle: record=pickle.load(handle)
        if record['fingerprint']!=fingerprint: raise ValueError('Prepared data fingerprint changed')
        return record['dataset'],record['windows']
    dataset=load_dataset(config['data_root'],config['seed'])
    tokenizers={k:OffsetTokenizer(v) for k,v in config['tokenizers'].items()}
    windows=prepare_windows(dataset,tokenizers,config['max_chars'],config['overlap'],config['max_tokens'])
    with cache.with_suffix('.tmp').open('wb') as handle:
        pickle.dump({'fingerprint':fingerprint,'dataset':dataset,'windows':windows},handle)
    os.replace(cache.with_suffix('.tmp'),cache)
    atomic_json(root/'split_manifest.json',{'train_ids':dataset['train_ids'],
        'validation_ids':dataset['validation_ids'],'official_test_ids':sorted(dataset['test']),
        'window_counts':{k:len(v) for k,v in windows.items()},'fingerprint':fingerprint,
        'char_vocab':dataset['char_vocab'],'labels':dataset['labels']})
    print(json.dumps({'phase':'prepared','windows':{k:len(v) for k,v in windows.items()}}),flush=True)
    return dataset,windows


@torch.inference_mode()
def document_logits(model,rows,dataset,encoding,device,status=None):
    model.eval()
    texts=dataset['test'] if rows and rows[0]['id'] in dataset['test'] else dataset['texts']
    classes=1+2*len(dataset['labels'])
    keys=set(texts) if texts is dataset['test'] else {r['id'] for r in rows}
    outputs={key:np.zeros((len(texts[key]),classes),np.float32) for key in keys}
    mass={key:np.zeros(len(texts[key]),np.float32) for key in outputs}
    for i,row in enumerate(rows):
        args=tensor_row(row,encoding,device)
        with autocast(device): logits=model(*args)[0].float().cpu().numpy()
        size=row['end']-row['start']
        # Lower weight near context edges; each original character remains covered.
        weights=np.minimum(np.arange(1,size+1),np.arange(size,0,-1)).clip(1,240).astype(np.float32)/240
        a,b=row['start'],row['end']; key=row['id']
        outputs[key][a:b]+=logits*weights[:,None]; mass[key][a:b]+=weights
        if status and (i%50==0 or i+1==len(rows)):
            atomic_json(status,{'phase':'generation','completed_windows':i+1,'total_windows':len(rows)})
    for key in outputs:
        if not np.isfinite(outputs[key]).all() or (mass[key]<=0).any():
            raise ValueError('Nonfinite or uncovered document logits')
        if len(texts[key]): outputs[key]/=mass[key][:,None]
    return outputs


def decode(outputs,labels,bias):
    return {key:constrained_decode(value,labels,bias) for key,value in outputs.items()}


def evaluate(model,spec,dataset,windows,device):
    outputs=document_logits(model,windows['validation'],dataset,spec['encoding'],device)
    truth={key:dataset['gold'][key] for key in dataset['validation_ids']}
    best=None
    for bias in (0.,.5,1.,1.5,2.):
        pred=decode(outputs,dataset['labels'],bias)
        metrics=strict_score(truth,pred,dataset['labels'])
        if best is None or metrics['macro_f1']>best['metrics']['macro_f1']:
            best={'outside_bias':bias,'metrics':metrics,'predictions':pred}
    return best


def teacher_cache(config,teacher_spec,dataset,windows,device):
    root=Path(config['output_root']); source=root/teacher_spec['name']/'best.pt'
    done=source.parent/'complete.json'
    if not source.exists() or not done.exists():
        raise FileNotFoundError('Teacher must finish verified NER training and export before distillation')
    completion=json.loads(done.read_text())
    digest=sha256(source); cache=root/'teacher_train_probs.pt'
    if completion.get('best_checkpoint_sha256')!=digest:
        raise ValueError('Completed teacher checkpoint changed')
    if sha256(completion['submission'])!=completion['sha256']:
        raise ValueError('Completed teacher submission changed')
    teacher_state=torch.load(source,map_location='cpu',weights_only=False)
    split_digest=sha256(root/'split_manifest.json')
    provenance=teacher_state['evidence']['fingerprint']
    if (provenance['split_sha256']!=split_digest or provenance['experiment']!=teacher_spec
        or teacher_state['epoch']!=completion['selected_epoch']):
        raise ValueError('Teacher training provenance differs')
    lineage={'teacher_sha256':digest,'split_sha256':split_digest,
             'implementation_hashes':implementation_hashes(),'temperature':2.,'split':'train_only'}
    if cache.exists():
        obj=torch.load(cache,map_location='cpu',weights_only=False)
        if obj.get('lineage')==lineage and len(obj['probabilities'])==len(windows['train']):
            return obj['probabilities'],digest
        raise ValueError('Teacher cache lineage changed')
    teacher,_=build_model(teacher_spec,len(dataset['char_vocab'])+2,1+2*len(dataset['labels']))
    teacher.load_state_dict(teacher_state['model'],strict=True)
    del teacher_state
    teacher.to(device).eval(); result=[]
    with torch.inference_mode():
        for i,row in enumerate(windows['train']):
            with autocast(device):
                logits=teacher(*tensor_row(row,teacher_spec['encoding'],device))[0].float()
            result.append((logits/2.).softmax(-1).cpu().half())
            if i%100==0: print(json.dumps({'phase':'teacher_cache','completed':i,'total':len(windows['train'])}),flush=True)
    del teacher; gc.collect()
    if device.type=='cuda': torch.cuda.empty_cache()
    save_torch(cache,{'lineage':lineage,'probabilities':result})
    return result,digest


def run_stage(config,spec,dataset,windows,device):
    root=Path(config['output_root']); run=root/spec['name']; run.mkdir(parents=True,exist_ok=True)
    done=run/'complete.json'
    if done.exists():
        record=json.loads(done.read_text())
        if sha256(record['submission'])!=record['sha256']: raise ValueError('Completed submission changed')
        return record
    random.seed(config['seed']); np.random.seed(config['seed']); torch.manual_seed(config['seed'])
    probabilities=teacher_digest=None
    if spec['method']=='kd':
        teacher=next(x for x in config['experiments'] if x['name']==spec['teacher'])
        probabilities,teacher_digest=teacher_cache(config,teacher,dataset,windows,device)
    model,evidence=build_model(spec,len(dataset['char_vocab'])+2,1+2*len(dataset['labels']))
    model.to(device)
    if device.type=='cuda': torch.cuda.reset_peak_memory_stats(device)
    base=[p for n,p in model.named_parameters() if n.startswith('backbone.') and p.requires_grad]
    head=[p for n,p in model.named_parameters() if not n.startswith('backbone.') and p.requires_grad]
    optimizer=torch.optim.AdamW([{'params':base,'lr':spec['base_lr']},
                                {'params':head,'lr':spec['head_lr']}],weight_decay=.01)
    fp16=device.type=='cuda' and not torch.cuda.is_bf16_supported()
    scaler=torch.amp.GradScaler('cuda',enabled=fp16)
    label_counts=Counter(tag for row in windows['train'] for tag in row['targets'])
    classes=1+2*len(dataset['labels'])
    weights=torch.tensor([min(12.,math.sqrt(label_counts[0]/max(label_counts[k],1)))
                          for k in range(classes)],device=device)
    weights[0]=1.
    epoch,start_group,step,best,bad=0,0,0,-1.,0
    latest=run/'latest.pt'
    fingerprint={'experiment':spec,'split_sha256':sha256(root/'split_manifest.json'),
                 'implementation_hashes':implementation_hashes(),
                 'teacher_sha256':teacher_digest,'pretrained_sha256':evidence['pretrained_checkpoint_sha256']}
    if latest.exists():
        state=torch.load(latest,map_location='cpu',weights_only=False)
        if state['fingerprint']!=fingerprint: raise ValueError('Resume provenance differs')
        model.load_state_dict(state['model'],strict=True); optimizer.load_state_dict(state['optimizer'])
        scaler.load_state_dict(state['scaler'])
        epoch,start_group,step,best,bad=[state[k] for k in ('epoch','next_group','step','best','bad')]
        torch.set_rng_state(state['rng']);
        if device.type=='cuda': torch.cuda.set_rng_state(state['cuda_rng'],device)
        del state
    evidence.update({'experiment':spec,'hardware':str(torch.cuda.get_device_name(device)) if device.type=='cuda' else 'CPU',
                     'torch':torch.__version__,'teacher_sha256':teacher_digest,'fingerprint':fingerprint})
    atomic_json(run/'run.json',evidence)
    def snapshot(epoch,next_group):
        save_torch(latest,{'model':model.state_dict(),'optimizer':optimizer.state_dict(),
          'scaler':scaler.state_dict(),'epoch':epoch,'next_group':next_group,'step':step,
          'best':best,'bad':bad,'fingerprint':fingerprint,'rng':torch.get_rng_state(),
          'cuda_rng':torch.cuda.get_rng_state(device) if device.type=='cuda' else None})
    began=time.monotonic(); processed=0; trained_chars=0
    groups=math.ceil(len(windows['train'])/config['grad_accum'])
    total_steps=groups*config['max_epochs']
    while epoch<config['max_epochs'] and bad<config['patience']:
        order=list(range(len(windows['train']))); random.Random(config['seed']+epoch).shuffle(order)
        model.train()
        for group in range(start_group,groups):
            indices=order[group*config['grad_accum']:(group+1)*config['grad_accum']]
            optimizer.zero_grad(set_to_none=True); loss_sum=0.
            fraction=(step+1)/max(total_steps,1)
            schedule=min(1.,fraction/.05)*(.1+.9*.5*(1+math.cos(math.pi*min(fraction,1.))))
            for pg,rate in zip(optimizer.param_groups,(spec['base_lr'],spec['head_lr'])): pg['lr']=rate*schedule
            for idx in indices:
                row=windows['train'][idx]
                target=torch.tensor(row['targets'],dtype=torch.long,device=device)
                with autocast(device): logits=model(*tensor_row(row,spec['encoding'],device))[0].float()
                loss=F.cross_entropy(logits,target,weight=weights,label_smoothing=.01)
                if probabilities is not None:
                    teacher_prob=probabilities[idx].to(device).float()
                    kd=F.kl_div((logits/2.).log_softmax(-1),teacher_prob,reduction='batchmean')*4.
                    loss=.5*loss+.5*kd
                if not torch.isfinite(loss): raise FloatingPointError('Nonfinite NER loss')
                scaler.scale(loss/len(indices)).backward(); loss_sum+=float(loss.detach())/len(indices)
                trained_chars+=len(row['chars'])
            scaler.unscale_(optimizer)
            norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
            scaler.step(optimizer); scaler.update(); step+=1; processed+=1
            if step%10==0 or processed==1:
                elapsed=time.monotonic()-began
                status={'phase':'training','experiment':spec['name'],'epoch':epoch+1,
                    'step':step,'groups_per_epoch':groups,'loss':loss_sum,'grad_norm':float(norm),
                    'chars_per_second':trained_chars/elapsed,'elapsed_seconds':elapsed,
                    'maximum_remaining_seconds':max(0,total_steps-step)*elapsed/processed,
                    'best_validation_macro_f1':best,
                    'peak_memory_bytes':torch.cuda.max_memory_allocated(device) if device.type=='cuda' else 0}
                atomic_json(run/'status.json',status)
                with (run/'metrics.jsonl').open('a') as handle: handle.write(json.dumps(status)+'\n')
                print(json.dumps(status),flush=True)
            if step%100==0: snapshot(epoch,group+1)
        selection=evaluate(model,spec,dataset,windows,device)
        score=selection['metrics']['macro_f1']
        print(json.dumps({'phase':'validation','experiment':spec['name'],'epoch':epoch+1,
                          'macro_f1':score,'outside_bias':selection['outside_bias']}),flush=True)
        if score>best+1e-5:
            best,bad=score,0
            save_torch(run/'best.pt',{'model':model.state_dict(),'evidence':evidence,'epoch':epoch+1})
            atomic_json(run/'selection.json',{k:v for k,v in selection.items() if k!='predictions'})
            atomic_json(run/'validation_predictions.json',selection['predictions'])
        else: bad+=1
        epoch+=1; start_group=0; snapshot(epoch,0)
    model.load_state_dict(torch.load(run/'best.pt',map_location='cpu',weights_only=False)['model'],strict=True)
    selection=json.loads((run/'selection.json').read_text())
    outputs=document_logits(model,windows['test'],dataset,spec['encoding'],device,run/'status.json')
    predictions=decode(outputs,dataset['labels'],selection['outside_bias'])
    validate_predictions(predictions,dataset['test'],dataset['labels'])
    archive=root/'deliverables'/spec['submission_name']
    digest=export_zip(archive,predictions,dataset['test'],dataset['labels'])
    atomic_json(run/'validation_truth.json',{key:dataset['gold'][key] for key in dataset['validation_ids']})
    result=subprocess.run([sys.executable,str(Path(__file__).resolve().parents[1]/'vendor/evaluate_phase2.py'),
         '--truth',str(run/'validation_truth.json'),'--prediction',str(run/'validation_predictions.json'),
         '--labels',str(Path(config['data_root'])/'Train/labels.txt')],capture_output=True,text=True,check=True)
    (run/'official_validation.txt').write_text(result.stdout)
    record={'phase':'complete','experiment':spec['name'],'submission':str(archive),'sha256':digest,
            'best_checkpoint_sha256':sha256(run/'best.pt'),
            'official_ids':len(predictions),'entity_spans':sum(map(len,predictions.values())),
            'validation':selection['metrics'],'epochs_completed':epoch,
            'selected_epoch':torch.load(run/'best.pt',map_location='cpu',weights_only=False)['epoch']}
    atomic_json(done,record); atomic_json(run/'status.json',record)
    print(json.dumps(record),flush=True)
    del model,optimizer; gc.collect()
    if device.type=='cuda': torch.cuda.empty_cache()
    return record


def preflight(config,dataset,windows,device):
    results=[]
    row=windows['train'][0]
    for spec in config['experiments']:
        torch.manual_seed(config['seed'])
        model,evidence=build_model(spec,len(dataset['char_vocab'])+2,1+2*len(dataset['labels']))
        model.to(device).train()
        with autocast(device): logits=model(*tensor_row(row,spec['encoding'],device))[0].float()
        loss=F.cross_entropy(logits,torch.tensor(row['targets'],device=device))
        loss.backward()
        grads=[p.grad for p in model.parameters() if p.requires_grad and p.grad is not None]
        if not torch.isfinite(loss) or not grads or not all(torch.isfinite(g).all() for g in grads):
            raise FloatingPointError('Preflight failed')
        results.append({'name':spec['name'],'loss':float(loss),'finite_gradients':True,
                        'trainable_parameters':evidence['trainable_parameters'],
                        'pretrained_sha256':evidence['pretrained_checkpoint_sha256']})
        print(json.dumps(results[-1]),flush=True)
        del model,grads; gc.collect()
        if device.type=='cuda': torch.cuda.empty_cache()
    atomic_json(Path(config['output_root'])/'preflight.json',results)


def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--config',required=True)
    parser.add_argument('--preflight',action='store_true'); parser.add_argument('--prepare-only',action='store_true')
    args=parser.parse_args(); config=json.loads(Path(args.config).read_text())
    torch.set_num_threads(4)
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dataset,windows=prepare(config)
    if args.prepare_only: return
    if args.preflight: preflight(config,dataset,windows,device); return
    root=Path(config['output_root']); lock=open(root/'queue.lock','a+')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    lock.seek(0); lock.truncate(); lock.write(str(os.getpid())); lock.flush()
    records=[]
    for spec in config['experiments']:
        atomic_json(root/'queue_status.json',{'phase':'running','active':spec['name'],'completed':[x['experiment'] for x in records]})
        for attempt in range(2):
            try:
                records.append(run_stage(config,spec,dataset,windows,device)); break
            except Exception as error:
                import traceback
                traceback.print_exc()
                atomic_json(root/spec['name']/'failure.json',{'attempt':attempt+1,'error':str(error)})
                gc.collect()
                if device.type=='cuda': torch.cuda.empty_cache()
                if attempt==1: print(json.dumps({'phase':'failed','experiment':spec['name'],'error':str(error)}),flush=True)
                else: time.sleep(5)
    phase='complete' if len(records)==len(config['experiments']) else 'incomplete'
    atomic_json(root/'queue_status.json',{'phase':phase,'completed':records})
    if phase!='complete': raise SystemExit(1)

if __name__=='__main__': main()
