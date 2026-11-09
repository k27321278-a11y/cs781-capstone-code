"""Generate fitting-only sequence targets from the completed own B1 teacher."""
import argparse
import hashlib
import json
from pathlib import Path

import torch

import b1_fusion
from phase6_data import normalize_rows, split_rows, document_hash
from train_phase6 import pack_source, restore_trainable, sha, save_json


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--models-input',required=True)
    parser.add_argument('--data-input',required=True)
    parser.add_argument('--teacher-adapter',required=True)
    parser.add_argument('--output',required=True)
    parser.add_argument('--count',type=int,default=256)
    parser.add_argument('--max-new-tokens',type=int,default=512)
    args=parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError('Teacher generation requires Kaggle CUDA')
    data=Path(args.data_input)
    train=normalize_rows(json.loads(next(data.rglob('train.json')).read_text(encoding='utf-8-sig')),
                         True,drop_empty_summaries=True)
    test=normalize_rows(json.loads(next(data.rglob('test.json')).read_text(encoding='utf-8-sig')),
                        False)
    test_hashes={document_hash(row['document']) for row in test}
    fit=split_rows([row for row in train
                    if document_hash(row['document']) not in test_hashes])['fit']
    chosen=sorted(fit,key=lambda row:hashlib.sha256(row['id'].encode()).hexdigest())[:args.count]
    if len(chosen)<64:
        raise ValueError('Not enough fitting documents for meaningful KD')
    model,tokenizer,metadata=b1_fusion.build(args.models_input,'cuda')
    adapter_path=Path(args.teacher_adapter)
    saved=torch.load(adapter_path,map_location='cpu',weights_only=False)
    if saved['parent_metadata']['phase1_checkpoint_sha256'] != b1_fusion.PARENT_SHA256:
        raise ValueError('Teacher adapter has the wrong Phase1 parent')
    restore_trainable(model,saved['adapter'])
    if metadata['total_parameters']<=110119680:
        raise ValueError('Teacher must be larger than own Phase1 A0 student')
    model.eval()
    predictions={}
    for index,row in enumerate(chosen,1):
        tokens=tokenizer.encode(row['document'])
        ids,mask=pack_source(tokens,'b1_fusion',tokenizer,8,384,'cuda')
        with torch.inference_mode(),torch.autocast('cuda',dtype=torch.float16):
            output=b1_fusion.generate(model,ids,mask,args.max_new_tokens,tokenizer.eos_id)[0]
        cleaned=[]
        for token in output:
            if token==tokenizer.eos_id:
                break
            if token not in (tokenizer.pad_id,tokenizer.bos_id):
                cleaned.append(int(token))
        summary=tokenizer.decode(cleaned).strip()
        if not summary:
            raise RuntimeError('Empty teacher summary for '+row['id'])
        predictions[row['id']]=summary
        if index%16==0:
            print(json.dumps({'event':'teacher_generated','count':index,'target':len(chosen)}),flush=True)
    output_path=Path(args.output)
    output_path.parent.mkdir(parents=True,exist_ok=True)
    save_json(output_path,predictions)
    save_json(output_path.with_suffix('.manifest.json'),{
        'teacher_parent_sha256':b1_fusion.PARENT_SHA256,
        'teacher_adapter_sha256':sha(adapter_path),
        'teacher_selected_epoch':saved['epoch'],
        'teacher_model_parameters':metadata['total_parameters'],
        'student_phase1_parameters':110119680,
        'fit_only':True,
        'document_count':len(predictions),
        'output_sha256':sha(output_path),
    })
    print(json.dumps({'event':'teacher_complete','count':len(predictions),
                      'teacher_adapter_sha256':sha(adapter_path)}),flush=True)


if __name__=='__main__':
    main()
