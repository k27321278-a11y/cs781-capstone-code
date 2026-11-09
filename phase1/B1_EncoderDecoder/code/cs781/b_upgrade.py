"""Verified own-weight B continuation, full held-out evaluation and exports."""
import hashlib
import json
import math
import os
import shutil
import uuid
from collections import Counter
from dataclasses import asdict, replace
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import torch

from .contracts import ContinuationExample, GenerationConfig, SOURCES
from .cli import _download_official_test, _official_dataset_revision
from .evaluate import score_predictions
from .generate_b import generate_continuation_b
from .io import sha256_file, write_json_atomic
from .models.b import EncoderDecoderConfig, EncoderDecoderLegalModel
from .prepare import load_prepared
from .splits import iter_catalog
from .submission import load_json_strict, validate_predictions, write_submission, extract_official_test_prompts
from .tokenizer import load_tokenizer
from .train_b import _code_digest as legacy_digest
from .prediction_progress import PredictionProgress

POLICIES = [GenerationConfig(repetition_penalty=p,max_new_tokens=2048) for p in (1.,1.10,1.20)]
POLICIES.append(GenerationConfig(strategy='nucleus',temperature=.8,top_p=.9,
                               repetition_penalty=1.10,max_new_tokens=2048))


def validate_policies(config):
    declared=[]
    for value in config.get('decoding',[]):
        value=dict(value);value.setdefault('max_new_tokens',2048)
        declared.append(GenerationConfig(**value))
    if declared!=POLICIES:raise ValueError('Declared decoding policies differ from implemented experiment')


def upgrade_digest():
    root=Path(__file__).parents[1]
    digest=hashlib.sha256(legacy_digest().encode())
    for name in ('cs781/b_upgrade.py','cs781/train_b_improved.py',
                 'cs781/prediction_progress.py','scripts/improve_b.py'):
        digest.update(name.encode()); digest.update((root/name).read_bytes())
    return digest.hexdigest()


def word_windows(words,document_id,seed,stride=256):
    if stride<1: raise ValueError('Positive word stride required')
    if len(words)<640:return
    available=min(stride-1,len(words)-640)
    offset=int(hashlib.sha256(f'{seed}:{document_id}:word-window'.encode()).hexdigest()[:16],16)%(available+1)
    for start in range(offset,len(words)-639,stride):
        yield start,words[start:start+512],words[start+512:start+640]


def prepare_task_aligned(base,catalog,output,max_seq_len=8192,seed=781):
    base=load_prepared(base.root if hasattr(base,'root') else base)
    original=load_json_strict(base.root/'manifest.json')
    if sha256_file(catalog)!=original['catalog_sha256']:
        raise ValueError('Use the exact original split catalog; no held-out reassignment')
    output=Path(output)
    if output.exists():raise FileExistsError(output)
    staging=output.with_name(output.name+'.staging-'+uuid.uuid4().hex); staging.mkdir(parents=True)
    for name in original['files']:
        if not name.startswith('train.'):
            shutil.copyfile(base.root/name,staging/name)
    tokenizer=load_tokenizer(base.root/'tokenizer.json')
    offsets=[]; sources=Counter(); targets=0; max_prompt=0; max_suffix=0; short=0
    with (staging/'train.jsonl').open('wb') as handle:
        for doc in iter_catalog(catalog,'train'):
            words=doc.raw.text.split()
            if len(words)<640:short+=1
            for start,prompt,suffix in word_windows(words,doc.raw.id,seed):
                prefix=tokenizer.encode(' '.join(prompt)).ids
                tail=tokenizer.encode(' '.join(suffix)).ids
                if not prefix or not tail or len(prefix)+len(tail)+2>max_seq_len+1:
                    raise ValueError('Task-aligned example exceeds context; never truncate or filter')
                row=ContinuationExample(f'{doc.raw.id}:word{start}',doc.raw.id,doc.raw.source,
                                        'train',prefix,tail)
                offsets.append(handle.tell()); handle.write((json.dumps(asdict(row),ensure_ascii=False)+'\n').encode())
                sources[doc.raw.source]+=1; targets+=len(tail)+1
                max_prompt=max(max_prompt,len(prefix)+1); max_suffix=max(max_suffix,len(tail)+1)
                if len(offsets)%10000==0:print(json.dumps({'phase':'prepare_B','examples':len(offsets)}),flush=True)
        handle.flush();os.fsync(handle.fileno())
    if not offsets or set(sources)!=set(SOURCES):raise ValueError('Missing official training source')
    np.save(staging/'train.offsets.npy',np.asarray(offsets,dtype=np.int64))
    manifest={**original,'max_seq_len':max_seq_len,
        'counts':{**original['counts'],'train':len(offsets)},
        'source_counts':{**original['source_counts'],'train':dict(sources)},
        'excluded_short':{**original['excluded_short'],'train':short},
        'overlength':{**original['overlength'],'train':0},
        'parent_manifest_sha256':base.manifest_sha256,
        'training_window_policy':{'prompt_words':512,'target_words':128,'stride_words':256,
                                 'seed':seed,'train_documents_only':True},
        'training_target_tokens_one_pass':targets,
        'max_training_prompt_tokens':max_prompt,'max_training_suffix_tokens':max_suffix,
        'files':{p.name:{'sha256':sha256_file(p),'bytes':p.stat().st_size}
                 for p in staging.iterdir() if p.is_file()}}
    write_json_atomic(staging/'manifest.json',manifest);os.rename(staging,output)
    return load_prepared(output)


def snapshot(path):
    with Path(path).open('rb') as handle:
        payload=torch.load(handle,map_location='cpu',weights_only=False);handle.seek(0)
        digest=hashlib.sha256()
        for chunk in iter(lambda:handle.read(1024*1024),b''):digest.update(chunk)
    return payload,digest.hexdigest()


def load_parent(path,original_data):
    path=Path(path).resolve();run=path.parent.parent
    if path!=run/'checkpoints/latest.pt':raise ValueError('Use the completed own B latest checkpoint')
    payload,digest=snapshot(path);metadata=load_json_strict(run/'run.json')
    status=load_json_strict(run/'training_status.json')
    manifest=load_json_strict(original_data.root/'manifest.json')
    if (payload.get('metadata')!=metadata or metadata.get('architecture')!='B'
        or metadata.get('synthetic_smoke') is not False or metadata.get('variant')
        or metadata.get('code_sha256')!=legacy_digest()
        or metadata.get('data_manifest_sha256')!=original_data.manifest_sha256
        or metadata.get('tokenizer_sha256')!=original_data.tokenizer_sha256
        or manifest.get('synthetic') is not False or _official_dataset_revision(manifest) is None
        or not 50_000_000<=metadata.get('unique_parameters',0)<=300_000_000
        or status.get('status')!='complete' or status.get('architecture')!='B'
        or status.get('state')!=payload.get('state') or status.get('checkpoint_sha256')!=digest
        or status.get('code_sha256')!=metadata['code_sha256']
        or status.get('target_tokens')!=metadata['train_config']['target_tokens']
        or payload['state']['target_tokens']<metadata['train_config']['target_tokens']):
        raise ValueError('Parent B checkpoint/data/completion/source provenance does not match')
    return payload,digest


def initialization(parent_path,original_data,model_cfg):
    payload,digest=load_parent(parent_path,original_data)
    expected={**payload['metadata']['model_config'],'max_seq_len':model_cfg.max_seq_len}
    if asdict(model_cfg)!=expected:raise ValueError('Own B initialization may extend only nonlearned context')
    lineage={'checkpoint_sha256':digest,'checkpoint':str(Path(parent_path).resolve()),
             'state':payload['state'],'source_sha256':payload['metadata']['code_sha256'],
             'data_manifest_sha256':original_data.manifest_sha256,
             'learned_parameter_changes':False}
    return payload['model'],lineage


def load_model(checkpoint,original_data,aligned_data=None,device='cuda'):
    checkpoint=Path(checkpoint).resolve()
    payload,digest=snapshot(checkpoint);metadata=payload.get('metadata',{})
    if metadata.get('variant')=='B_taskaligned':
        run=checkpoint.parent.parent
        status=load_json_strict(run/'training_status.json')
        run_meta=load_json_strict(run/'run.json')
        if (aligned_data is None or metadata!=run_meta or metadata.get('architecture')!='B'
            or metadata.get('synthetic_smoke') is not False
            or metadata.get('code_sha256')!=upgrade_digest()
            or metadata.get('data_manifest_sha256')!=aligned_data.manifest_sha256
            or metadata.get('tokenizer_sha256')!=original_data.tokenizer_sha256
            or status.get('status')!='complete' or status.get('architecture')!='B'
            or status.get('code_sha256')!=metadata['code_sha256']
            or status.get('target_tokens')!=metadata['train_config']['target_tokens']
            or status.get('state',{}).get('target_tokens',-1)<metadata['train_config']['target_tokens']
            or status.get('checkpoint_sha256')!=sha256_file(run/'checkpoints/latest.pt')):
            raise ValueError('Continued B requires a matching complete parent training run')
        if checkpoint==run/'checkpoints/latest.pt' and status.get('state')!=payload.get('state'):
            raise ValueError('Latest B state differs from completion record')
        parent,parent_hash=load_parent(metadata['initialization']['checkpoint'],original_data)
        if parent_hash!=metadata['initialization']['checkpoint_sha256']:
            raise ValueError('Initial own B checkpoint changed')
        cfg=EncoderDecoderConfig(**metadata['model_config'])
        if cfg.max_seq_len!=load_json_strict(aligned_data.root/'manifest.json')['max_seq_len']:
            raise ValueError('Continued B context/data mismatch')
        allowed={**parent['metadata']['model_config'],'max_seq_len':cfg.max_seq_len}
        if asdict(cfg)!=allowed:raise ValueError('Continued B graph differs from documented own foundation')
    else:
        payload,digest=load_parent(checkpoint,original_data);metadata=payload['metadata']
        cfg=replace(EncoderDecoderConfig(**metadata['model_config']),max_seq_len=8192)
    cfg.validate(); model=EncoderDecoderLegalModel(cfg).to(device)
    if model.count_parameters()!=metadata['unique_parameters']:
        raise ValueError('Checkpoint parameter count mismatch')
    model.load_state_dict(payload['model'],strict=True);model.eval()
    payload['_loaded_checkpoint_sha256']=digest
    return model,payload


def generate_records(output,prompts,model,tokenizer,decode,fingerprint):
    progress=PredictionProgress(Path(output).with_suffix('.progress.json'),prompts,fingerprint)
    for index,key in enumerate(list(progress.pending()),1):
        try:progress.record(key,generate_continuation_b(model,tokenizer,prompts[key],decode))
        except (RuntimeError,ValueError) as exc:progress.failure(key,type(exc).__name__+': '+str(exc))
        if index%10==0 or len(progress.predictions)==len(prompts):
            print(json.dumps({'phase':'B_generation','completed':len(progress.predictions),
                              'total':len(prompts),'failures':len(progress.failures)}),flush=True)
    return progress.complete()


def evaluate(checkpoint,original_data,aligned_data,split,policy,output,cache):
    output=Path(output)
    prompts={k:v['text'] for k,v in load_json_strict(original_data.root/f'{split}.prompts.json').items()}
    refs=load_json_strict(original_data.root/f'{split}.references.json')
    if set(prompts)!=set(refs):raise ValueError('Held-out IDs differ')
    model,payload=load_model(checkpoint,original_data,aligned_data)
    decode=POLICIES[policy]
    fingerprint={'checkpoint':payload['_loaded_checkpoint_sha256'],
        'code':upgrade_digest(),'policy':asdict(decode),'tokenizer':original_data.tokenizer_sha256,
        'split':split,'prompt_file':sha256_file(original_data.root/f'{split}.prompts.json'),
        'encoder_context':model.cfg.max_seq_len}
    predictions=generate_records(output,prompts,model,load_tokenizer(original_data.root/'tokenizer.json'),decode,fingerprint)
    del model;torch.cuda.empty_cache()
    expected={k:{'text':v} for k,v in predictions.items()}
    if output.exists() and load_json_strict(output)!=expected:raise ValueError('Saved predictions differ')
    write_json_atomic(output,expected)
    reference=output.with_suffix('.references.json')
    if reference.exists() and load_json_strict(reference)!=refs:raise ValueError('Saved reference differs')
    write_json_atomic(reference,refs)
    metrics_path=output.with_suffix('.metrics.json')
    if metrics_path.exists():
        record=load_json_strict(metrics_path)
        if record.get('prediction_sha256')!=sha256_file(output) or record.get('reference_sha256')!=sha256_file(reference):
            raise ValueError('Saved metrics do not match predictions/references')
        metrics=record['metrics']
    else:
        metrics=score_predictions(reference,output,Path(__file__).parents[1]/'vendor/phase1_evaluate.py',cache)
    if any(not math.isfinite(metrics[k]) for k in ('bertscore_f1','rouge_l_f1','bleu4')):
        raise ValueError('Nonfinite official metrics')
    report={'architecture':'B','variant':payload['metadata'].get('variant','original_B'),
        'checkpoint_sha256':payload['_loaded_checkpoint_sha256'],'checkpoint_state':payload['state'],
        'initialization':payload['metadata'].get('initialization'),'split':split,
        'declared_ids':list(prompts),'metrics':metrics,'generation':fingerprint,
        'limited_sample':False,'synthetic_smoke':False,
        'mean_distinct_words':sum(len(set(t.split())) for t in predictions.values())/len(predictions)}
    write_json_atomic(output.with_suffix('.context.json'),report)
    return report


def select_improved(results,ids,baseline):
    valid=[]
    for item in results:
        report=item.get('report',{}); metrics=report.get('metrics',{})
        if (set(report.get('declared_ids',[]))==set(ids) and len(report['declared_ids'])==len(ids)
            and all(type(metrics.get(k)) in (float,int) and math.isfinite(metrics[k])
                    for k in ('bertscore_f1','rouge_l_f1','bleu4'))):valid.append(item)
    if not valid:raise ValueError('No complete finite dev result')
    best=max(valid,key=lambda i:tuple(i['report']['metrics'][k] for k in ('bertscore_f1','rouge_l_f1','bleu4')))
    # Preserve a valid Codabench candidate even when the measured continuation
    # does not beat the scratch parent; experiment_result.json reports the gain.
    return best


def finalize_export(output,predictions,prompts,report):
    output=Path(output);validate_predictions(predictions,prompts)
    if output.exists():
        with ZipFile(output) as archive:
            if archive.namelist()!=['prediction.json']:raise ValueError('Invalid ZIP root entries')
            saved=json.loads(archive.read('prediction.json'))
        if saved!={k:{'text':v.strip()} for k,v in predictions.items()}:raise ValueError('ZIP differs from predictions')
        digest=sha256_file(output)
    else:digest=write_submission(predictions,prompts,output)
    report={**report,'prediction_zip_sha256':digest}
    context=output.with_suffix('.context.json')
    if context.exists() and load_json_strict(context)!=report:raise ValueError('Export context differs')
    write_json_atomic(context,report);return report


def export(checkpoint,original_data,aligned_data,policy,output,selection,test_file=None):
    model,payload=load_model(checkpoint,original_data,aligned_data)
    manifest=load_json_strict(original_data.root/'manifest.json');revision=_official_dataset_revision(manifest)
    test=(Path(test_file) if test_file is not None else
          _download_official_test(revision,original_data.root.parent/'hf_hub'))
    if (not test.is_file() or sha256_file(test) !=
            '614f348b3dcbb2a9f43706cd239438386dc396246cd5cc685a5ccb6eb06b7218'):
        raise ValueError('Official test file does not match the pinned course revision')
    prompts=extract_official_test_prompts(load_json_strict(test));decode=POLICIES[policy]
    fingerprint={'checkpoint':payload['_loaded_checkpoint_sha256'],'code':upgrade_digest(),
        'tokenizer':original_data.tokenizer_sha256,'policy':asdict(decode),'test':sha256_file(test),
        'selection':sha256_file(selection),'encoder_context':model.cfg.max_seq_len}
    predictions=generate_records(Path(output),prompts,model,load_tokenizer(original_data.root/'tokenizer.json'),decode,fingerprint)
    write_json_atomic(Path(output).with_suffix('.predictions.json'),{k:{'text':v} for k,v in predictions.items()})
    return finalize_export(output,predictions,prompts,{'architecture':'B','variant':'B_taskaligned',
        'checkpoint_sha256':payload['_loaded_checkpoint_sha256'],'checkpoint_state':payload['state'],
        'initialization':payload['metadata']['initialization'],'fingerprint':fingerprint,
        'dataset_revision':revision,'prediction_count':len(predictions),'synthetic_smoke':False})
