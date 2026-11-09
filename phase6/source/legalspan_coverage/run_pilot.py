"""Paired validation-only decoding pilot. No test predictions or submission."""
import argparse
import collections
import hashlib
import json
import re
import time
from pathlib import Path
import torch
from coverage_decode import install_coverage


def words(text): return re.findall(r'\w+', text.lower())


def metrics(prediction,reference):
    tokens=words(prediction); gold=words(reference)
    pc,gc=collections.Counter(tokens),collections.Counter(gold)
    overlap=sum((pc&gc).values())
    grams=[tuple(tokens[i:i+3]) for i in range(max(0,len(tokens)-2))]
    return {'word_count':len(tokens),'unigram_overlap_f1_proxy':2*overlap/max(1,len(tokens)+len(gold)),
            'repeated_trigram_fraction':1-len(set(grams))/len(grams) if grams else 0.}


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--input',required=True)
    parser.add_argument('--models',required=True)
    parser.add_argument('--teacher',required=True)
    parser.add_argument('--output',default='pilot_output')
    parser.add_argument('--limit',type=int,default=8)
    parser.add_argument('--max-new-tokens',type=int,default=384)
    args=parser.parse_args()
    from phase6_data import normalize_rows,split_rows,select_passages
    import legalspan_lora as native
    out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    if not torch.cuda.is_available():raise RuntimeError('Pilot requires GPU')
    model,tok,meta=native.build(args.models,'cuda');model.eval()
    adapter_path=Path(args.teacher)
    record=torch.load(adapter_path,map_location='cpu',weights_only=False)
    parent=record['parent_metadata']
    if parent['checkpoint_sha256']!=meta['checkpoint_sha256'] or parent['tokenizer_sha256']!=meta['tokenizer_sha256']:
        raise ValueError('Phase 6 adapter parent/tokenizer mismatch')
    params=dict(model.named_parameters())
    expected={n for n,p in params.items() if p.requires_grad}
    if set(record['adapter'])!=expected:raise ValueError('Incomplete or unexpected adapter state')
    with torch.no_grad():
        for n,v in record['adapter'].items():params[n].copy_(v.to(params[n].device))
    rows=normalize_rows(json.loads((Path(args.input)/'train.json').read_text()),True,drop_empty_summaries=True)
    validation=split_rows(rows)['validation'][:args.limit]
    report={'settings':vars(args),'parent':meta,'adapter_sha256':hashlib.sha256(adapter_path.read_bytes()).hexdigest(),
            'selected_epoch':record['epoch'],'validation_ids':[r['id'] for r in validation],
            'metric_note':'Unigram overlap proxy is not official ROUGE or BERTScore. Validation only; no confirmation/test selection.',
            'runs':[]}
    print(json.dumps({'event':'model_loaded','device':'cuda','parent_sha256':meta['checkpoint_sha256'],'epoch':record['epoch']}),flush=True)
    for strength in (0.,.5):
        install_coverage(model,strength)
        for row in validation:
            path=out/f"strength_{strength}_{row['id']}.json"
            if path.exists():
                result=json.loads(path.read_text());report['runs'].append(result);continue
            tokens=tok.encode(row['document'])
            source=[t for part in select_passages(tokens,8,384) for t in part]
            ids=torch.tensor([source],device='cuda');mask=torch.ones_like(ids,dtype=torch.bool)
            began=time.monotonic()
            print(json.dumps({'event':'generation_start','id':row['id'],'strength':strength,'source_tokens':len(source)}),flush=True)
            with torch.inference_mode(),torch.autocast('cuda',dtype=torch.float16):
                generated=native.generate(model,ids,mask,args.max_new_tokens,tok.eos_id)[0]
            prediction=tok.decode([x for x in generated if x not in (tok.bos_id,tok.eos_id,tok.pad_id)])
            result={'id':row['id'],'strength':strength,'summary':prediction,'reference':row['summary'],
                    'metrics':metrics(prediction,row['summary']),'elapsed_seconds':time.monotonic()-began}
            path.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
            report['runs'].append(result)
            (out/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
            print(json.dumps({'event':'generation_complete','id':row['id'],'strength':strength,**result['metrics'],'elapsed_seconds':result['elapsed_seconds']}),flush=True)
    print(json.dumps({'event':'pilot_complete','documents':len(validation),'comparisons':len(report['runs'])}),flush=True)

if __name__=='__main__':main()
