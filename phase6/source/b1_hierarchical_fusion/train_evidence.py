"""Resumable Phase 6 generation training from the student's Phase 1 weights."""
import argparse
import contextlib
import hashlib
import importlib
import json
import math
import random
import sys
import time
import zipfile
from pathlib import Path

import numpy as np
import torch
from phase6_data import normalize_rows, split_rows, select_passages, validate_predictions, document_hash


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream,'sha256').hexdigest()


def save_json(path,value):
    path=Path(path)
    tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(value,indent=2),encoding='utf-8')
    tmp.replace(path)


def trainable_state(model):
    return {name:p.detach().cpu() for name,p in model.named_parameters() if p.requires_grad}


def restore_trainable(model,state):
    params=dict(model.named_parameters())
    with torch.no_grad():
        for name,value in state.items():
            if name not in params or not params[name].requires_grad:
                raise ValueError('Adapter state mismatch: '+name)
            params[name].copy_(value.to(params[name].device))


def pack_source(tokens,method,tokenizer,passages,width,device):
    parts=select_passages(tokens,passages,width)
    if method in ('b1_fusion','b1_adalora','evidence_fusion'):
        ids=torch.full((1,len(parts),max(map(len,parts))),tokenizer.pad_id,dtype=torch.long,device=device)
        mask=torch.zeros_like(ids,dtype=torch.bool)
        for j,part in enumerate(parts):
            ids[0,j,:len(part)]=torch.tensor(part,device=device)
            mask[0,j,:len(part)]=True
    else:
        flat=[token for part in parts for token in part]
        ids=torch.tensor([flat],dtype=torch.long,device=device)
        mask=torch.ones_like(ids,dtype=torch.bool)
    return ids,mask


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--input',required=True)
    parser.add_argument('--models-input',default=None)
    parser.add_argument('--output',required=True)
    parser.add_argument('--method',choices=['evidence_fusion','b1_fusion','legalspan_lora','b1_adalora','c1_houlsby','d222_prefix','a0_kd'],required=True)
    parser.add_argument('--epochs',type=int,default=6)
    parser.add_argument('--accum',type=int,default=8)
    parser.add_argument('--lr',type=float,default=0.00015)
    parser.add_argument('--passages',type=int,default=8)
    parser.add_argument('--passage-tokens',type=int,default=384)
    parser.add_argument('--target-tokens',type=int,default=1536)
    parser.add_argument('--validation-limit',type=int,default=32)
    parser.add_argument('--confirmation-limit',type=int,default=64)
    parser.add_argument('--max-train-hours',type=float,default=5.0)
    parser.add_argument('--resume',default=None)
    parser.add_argument('--teacher-predictions',default=None)
    parser.add_argument('--skip-bertscore',action='store_true')
    args=parser.parse_args()
    if min(args.epochs,args.accum,args.passages,args.passage_tokens,args.target_tokens)<1:
        raise ValueError('Training dimensions must be positive')
    random.seed(781)
    np.random.seed(781)
    torch.manual_seed(781)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(781)
    device='cuda' if torch.cuda.is_available() else 'cpu'
    inp,out=Path(args.input),Path(args.output)
    out.mkdir(parents=True,exist_ok=True)
    train_path=next(inp.rglob('train.json'))
    test_path=next(inp.rglob('test.json'))
    raw_train=json.loads(train_path.read_text(encoding='utf-8-sig'))
    train=normalize_rows(raw_train,True,drop_empty_summaries=True)
    excluded_empty_summaries=len(raw_train)-len(train)
    test=normalize_rows(json.loads(test_path.read_text(encoding='utf-8-sig')),False)
    test_hashes={document_hash(row['document']) for row in test}
    eligible=[row for row in train if document_hash(row['document']) not in test_hashes]
    splits=split_rows(eligible)
    if not all(splits.values()): raise ValueError('One grouped split is empty')
    teacher_rows=[]
    teacher_sha256=None
    if args.method=='a0_kd':
        if not args.teacher_predictions:
            raise ValueError('A0 knowledge distillation requires teacher predictions')
        teacher_path=Path(args.teacher_predictions)
        teacher_sha256=sha(teacher_path)
        teacher_map=json.loads(teacher_path.read_text(encoding='utf-8'))
        fit_by_id={row['id']:row for row in splits['fit']}
        if (not isinstance(teacher_map,dict) or len(teacher_map)<64
                or not set(teacher_map).issubset(fit_by_id)
                or any(not isinstance(summary,str) or not summary.strip()
                       for summary in teacher_map.values())):
            raise ValueError('Teacher targets must be nonempty and fitting-only')
        # Three teacher passes give the 256 generated targets a meaningful
        # fraction of updates while retaining every gold fitting example.
        teacher_rows=[dict(fit_by_id[item],summary=summary)
                      for item,summary in sorted(teacher_map.items()) for _ in range(3)]
    elif args.teacher_predictions:
        raise ValueError('Teacher predictions are reserved for the KD experiment')
    split_manifest={'data_sha256':{'train':sha(train_path),'test':sha(test_path)},'ids':{k:[r['id'] for r in v] for k,v in splits.items()},'excluded_empty_summaries':excluded_empty_summaries,'excluded_train_test_exact_duplicates':len(train)-len(eligible)}
    identity_args={k:v for k,v in vars(args).items() if k not in ('input','models_input','output','resume','max_train_hours','teacher_predictions')}
    run_identity={'split':split_manifest,'settings':identity_args,'teacher_predictions_sha256':teacher_sha256}
    save_json(out/'split_manifest.json',split_manifest)
    module=importlib.import_module(args.method)
    models_input=Path(args.models_input) if args.models_input else inp
    source_archive=models_input/'legalspan_source.zip'
    if args.method=='legalspan_lora' and source_archive.exists():
        runtime_source=out/'runtime_legalspan_source'
        with zipfile.ZipFile(source_archive) as archive:
            for name in archive.namelist():
                resolved=(runtime_source/name).resolve()
                if not resolved.is_relative_to(runtime_source.resolve()):
                    raise ValueError('Unsafe source archive path')
            archive.extractall(runtime_source)
        sys.path.insert(0,str(runtime_source))
    model,tokenizer,metadata=module.build(models_input,device)
    if teacher_sha256 is not None:
        metadata['teacher_predictions_sha256']=teacher_sha256
        metadata['teacher_fit_documents']=len(teacher_map)
    print(json.dumps({'event':'model_loaded','device':device,'metadata':metadata}),flush=True)
    if device!='cuda': raise RuntimeError('GPU training requested, but CUDA is unavailable')
    scorer=None
    if not args.skip_bertscore:
        from bert_score import BERTScorer
        scorer=BERTScorer(lang='en',device=device,batch_size=8)
    def encode_rows(rows):
        result=[]
        for i,row in enumerate(rows):
            source=tokenizer.encode(row['document'])
            target=tokenizer.encode(row['summary'])[:args.target_tokens-1] if 'summary' in row else None
            result.append((row,source,target))
            if (i+1)%500==0: print(json.dumps({'event':'tokenized','rows':i+1}),flush=True)
        return result
    encoded={k:encode_rows(v) for k,v in splits.items()}
    if teacher_rows:
        encoded['fit'].extend(encode_rows(teacher_rows))
    encoded_test=encode_rows(test)
    new_prefixes=('hierarchy','pointer_')
    new_params=[p for n,p in model.named_parameters() if p.requires_grad and n.startswith(new_prefixes)]
    old_params=[p for n,p in model.named_parameters() if p.requires_grad and not n.startswith(new_prefixes)]
    optimizer=torch.optim.AdamW([{'params':old_params,'lr':args.lr},{'params':new_params,'lr':args.lr*3}],weight_decay=.01)
    scaler=torch.amp.GradScaler('cuda')
    history=[]
    best_score=-math.inf
    start_epoch=start_offset=updates=0
    if args.resume:
        record=torch.load(args.resume,map_location='cpu',weights_only=False)
        if record['run_identity']!=run_identity: raise ValueError('Resume data, split or training settings differ')
        old_hashes={k:v for k,v in record['parent_metadata'].items() if 'sha256' in k}
        new_hashes={k:v for k,v in metadata.items() if 'sha256' in k}
        if old_hashes!=new_hashes: raise ValueError('Resume parent provenance differs')
        restore_trainable(model,record['adapter'])
        optimizer.load_state_dict(record['optimizer'])
        scaler.load_state_dict(record['scaler'])
        history=record['history']
        best_score=record['best_score']
        start_epoch,start_offset,updates=record['epoch'],record['next_offset'],record['updates']
        random.setstate(record['random_state'])
        np.random.set_state(record['numpy_state'])
        torch.set_rng_state(record['torch_state'])
        torch.cuda.set_rng_state_all(record['cuda_states'])
        if record.get('best_record') is not None:
            torch.save(record['best_record'],out/'best_adapter.pt')
    def checkpoint(epoch,next_offset):
        best_record=torch.load(out/'best_adapter.pt',map_location='cpu',weights_only=False) if (out/'best_adapter.pt').exists() else None
        record={'adapter':trainable_state(model),'optimizer':optimizer.state_dict(),'scaler':scaler.state_dict(),'parent_metadata':metadata,'run_identity':run_identity,'best_record':best_record,'epoch':epoch,'next_offset':next_offset,'updates':updates,'history':history,'best_score':best_score,'random_state':random.getstate(),'numpy_state':np.random.get_state(),'torch_state':torch.get_rng_state(),'cuda_states':torch.cuda.get_rng_state_all(),'args':vars(args)}
        tmp=out/'latest.tmp'
        torch.save(record,tmp)
        tmp.replace(out/'latest.pt')
    @torch.no_grad()
    def generate_rows(rows):
        model.eval()
        predictions=[]
        for generation_index,(row,source,_) in enumerate(rows):
            print(json.dumps({'event':'generation_start','index':generation_index+1,'count':len(rows),'id':row['id']}),flush=True)
            ids,mask=pack_source(source,args.method,tokenizer,args.passages,args.passage_tokens,device)
            with torch.autocast('cuda',dtype=torch.float16):
                result=module.generate(model,ids,mask,args.target_tokens,tokenizer.eos_id)[0]
            cleaned=[]
            for token in result:
                if token==tokenizer.eos_id: break
                if token not in (tokenizer.pad_id,tokenizer.bos_id): cleaned.append(int(token))
            text=tokenizer.decode(cleaned).strip()
            if not text: raise RuntimeError('Empty generated summary: '+row['id'])
            predictions.append({'id':row['id'],'summary':text})
            print(json.dumps({'event':'generation_complete','index':generation_index+1,'count':len(rows),'id':row['id'],'words':len(text.split())}),flush=True)
        return predictions
    def evaluate(name,limit):
        subset=sorted(encoded[name],key=lambda item:hashlib.sha256(item[0]['id'].encode()).hexdigest())[:limit]
        predictions=generate_rows(subset)
        refs=[row['summary'] for row,_,_ in subset]
        save_json(out/(name+'_predictions.json'),predictions)
        if scorer:
            _,_,scores=scorer.score([r['summary'] for r in predictions],refs,batch_size=8)
            result={'bertscore_f1':float(scores.mean()),'count':len(subset),'scorer_hash':scorer.hash}
        else:
            result={'bertscore_f1':None,'count':len(subset),'selection':'negative_summary_nll'}
            losses=[]
            model.eval()
            with torch.no_grad():
                for row,source,target in subset:
                    ids,mask=pack_source(source,args.method,tokenizer,args.passages,args.passage_tokens,device)
                    targets=torch.tensor([[tokenizer.bos_id]+target+[tokenizer.eos_id]],device=device)
                    with torch.autocast('cuda',dtype=torch.float16):
                        loss=module.training_loss(model,ids,mask,targets,torch.ones_like(targets,dtype=torch.bool))
                    losses.append(float(loss))
            result['negative_nll']=-float(np.mean(losses))
        return result
    began=time.monotonic()
    stop=False
    for epoch in range(start_epoch,args.epochs):
        order=list(range(len(encoded['fit'])))
        random.Random(781+epoch).shuffle(order)
        model.train()
        optimizer.zero_grad(set_to_none=True)
        running=[]
        offset=start_offset if epoch==start_epoch else 0
        for pos in range(offset,len(order)):
            row,source,target=encoded['fit'][order[pos]]
            ids,mask=pack_source(source,args.method,tokenizer,args.passages,args.passage_tokens,device)
            targets=torch.tensor([[tokenizer.bos_id]+target+[tokenizer.eos_id]],device=device)
            group_start=(pos//args.accum)*args.accum
            group_size=min(args.accum,len(order)-group_start)
            with torch.autocast('cuda',dtype=torch.float16):
                loss=module.training_loss(model,ids,mask,targets,torch.ones_like(targets,dtype=torch.bool))
            if not torch.isfinite(loss): raise RuntimeError('Nonfinite training loss')
            scaler.scale(loss/group_size).backward()
            running.append(float(loss.detach()))
            if (pos+1)%args.accum==0 or pos+1==len(order):
                scaler.unscale_(optimizer)
                trainable=[p for p in model.parameters() if p.requires_grad]
                gradients_finite=all(torch.isfinite(p.grad).all() for p in trainable if p.grad is not None)
                if gradients_finite:
                    torch.nn.utils.clip_grad_norm_(trainable,1.0)
                else:
                    print(json.dumps({'event':'amp_overflow_skipped','epoch':epoch+1,'row':pos+1,'scale':scaler.get_scale()}),flush=True)
                scaler.step(optimizer)
                scaler.update()
                if gradients_finite and hasattr(module,'on_optimizer_step'):
                    module.on_optimizer_step(model,updates+1)
                optimizer.zero_grad(set_to_none=True)
                updates+=int(gradients_finite)
                if updates<=3 or updates%20==0:
                    print(json.dumps({'event':'training','epoch':epoch+1,'row':pos+1,'rows':len(order),'updates':updates,'mean_loss':float(np.mean(running[-160:])),'elapsed_minutes':(time.monotonic()-began)/60}),flush=True)
                if updates%100==0: checkpoint(epoch,pos+1)
                if time.monotonic()-began>args.max_train_hours*3600:
                    checkpoint(epoch,pos+1)
                    stop=True
                    break
        val=evaluate('validation',args.validation_limit)
        metric=val['bertscore_f1'] if val['bertscore_f1'] is not None else val['negative_nll']
        history.append({'epoch':epoch+1,'updates':updates,'train_objective_nll_plus_coverage':float(np.mean(running)) if running else None,'validation':val,'partial_epoch':stop})
        print(json.dumps(history[-1]),flush=True)
        save_json(out/'training_history.json',history)
        epoch_record={'adapter':trainable_state(model),'parent_metadata':metadata,'epoch':epoch+1,'updates':updates,'validation':val,'args':vars(args)}
        torch.save(epoch_record,out/('epoch_'+str(epoch+1)+'_adapter.pt'))
        save_json(out/('epoch_'+str(epoch+1)+'_validation_predictions.json'),json.loads((out/'validation_predictions.json').read_text()))
        if metric>best_score:
            best_score=metric
            torch.save({'adapter':trainable_state(model),'parent_metadata':metadata,'epoch':epoch+1,'updates':updates,'validation':val,'args':vars(args)},out/'best_adapter.pt')
            save_json(out/'best_validation_predictions.json',json.loads((out/'validation_predictions.json').read_text()))
        if not stop: checkpoint(epoch+1,0)
        if stop: break
    best=torch.load(out/'best_adapter.pt',map_location='cpu',weights_only=False)
    restore_trainable(model,best['adapter'])
    confirmation=evaluate('confirmation',args.confirmation_limit)
    predictions=generate_rows(encoded_test)
    validate_predictions(predictions,[r['id'] for r in test])
    zip_path=out/('phase6_'+args.method+'_prediction.zip')
    with zipfile.ZipFile(zip_path,'w',compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr('prediction.json',json.dumps(predictions,ensure_ascii=False,separators=(',',':')))
    report={'method':args.method,'phase1_parent':metadata,'counts':{k:len(v) for k,v in splits.items()},'test_count':len(test),'selected_epoch':best['epoch'],'validation':best['validation'],'confirmation':confirmation,'passage_strategy':'document-only uniform coverage including first and last; no reference-dependent passage selection','args':vars(args),'zip_sha256':sha(zip_path),'elapsed_minutes':(time.monotonic()-began)/60,'codabench_score':None}
    save_json(out/'report.json',report)
    print(json.dumps({'event':'complete','report':report}),flush=True)


if __name__=='__main__': main()
