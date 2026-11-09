"""Complete B continuation/selection/export in an isolated, locked GPU1 job."""
import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path

from cs781.b_upgrade import (POLICIES,evaluate,export,initialization,load_parent,
                            prepare_task_aligned,select_improved,upgrade_digest,validate_policies)
from cs781.checkpoints import run_lock
from cs781.contracts import TrainConfig
from cs781.io import sha256_file,write_json_atomic
from cs781.models.b import EncoderDecoderConfig
from cs781.prepare import load_prepared
from cs781.submission import load_json_strict
from cs781.train_b_improved import train_b_improved


def commit_selection(path,value):
    path=Path(path)
    if path.exists():
        old=load_json_strict(path)
        if old!=value:raise ValueError('Selection is immutable; inputs or winner changed')
        return old
    write_json_atomic(path,value);return value


def main(argv=None):
    parser=argparse.ArgumentParser()
    for name in ('parent-checkpoint','data','catalog','run-dir','aligned-data','scorer-cache','output'):
        parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--config',type=Path,default=Path('configs/b_improved.json'))
    parser.add_argument('--test-file',type=Path)
    parser.add_argument('--resume',action='store_true')
    args=parser.parse_args(argv)
    run=args.run_dir;phase=run/'pipeline_status.json';work=run/'evaluation'
    with run_lock(run/'pipeline'):
        work.mkdir(parents=True,exist_ok=True)
        try:
            config=load_json_strict(args.config);validate_policies(config);original=load_prepared(args.data)
            parent,parent_hash=load_parent(args.parent_checkpoint,original)
            cfg=replace(EncoderDecoderConfig(**parent['metadata']['model_config']),max_seq_len=config['max_seq_len'])
            train=TrainConfig(target_tokens=config['additional_target_tokens'],**config['train'])
            fingerprint={'config_sha256':sha256_file(args.config),'parent_checkpoint_sha256':parent_hash,
                'original_data_sha256':original.manifest_sha256,'code_sha256':upgrade_digest(),
                'policies':[vars(p) for p in POLICIES]}
            previous=run/'experiment.json'
            if previous.exists() and load_json_strict(previous)!=fingerprint:
                raise ValueError('Experiment provenance changed; use a new run')
            write_json_atomic(previous,fingerprint)
            del parent
            write_json_atomic(phase,{'phase':'preparing_task_aligned_data'})
            if args.aligned_data.exists():
                aligned=load_prepared(args.aligned_data)
                manifest=load_json_strict(aligned.root/'manifest.json')
                if (manifest.get('parent_manifest_sha256')!=original.manifest_sha256
                    or manifest.get('max_seq_len')!=cfg.max_seq_len
                    or manifest.get('catalog_sha256')!=sha256_file(args.catalog)
                    or manifest.get('training_window_policy')!={'prompt_words':512,'target_words':128,
                        'stride_words':256,'seed':781,'train_documents_only':True}):
                    raise ValueError('Existing aligned data provenance differs')
            else:aligned=prepare_task_aligned(original,args.catalog,args.aligned_data,cfg.max_seq_len)
            write_json_atomic(phase,{'phase':'baseline_dev'})
            baseline=evaluate(args.parent_checkpoint,original,None,'dev',0,work/'baseline_dev.json',args.scorer_cache)
            write_json_atomic(phase,{'phase':'training','additional_target_tokens':train.target_tokens})
            latest=run/'checkpoints/latest.pt';status=run/'training_status.json'
            complete=status.exists() and load_json_strict(status).get('status')=='complete'
            if not complete:
                if latest.exists() and not args.resume:
                    raise ValueError('Interrupted training requires explicit --resume; no automatic restart')
                train_b_improved(aligned,cfg,train,run,args.parent_checkpoint,original,resume=latest.exists())
            selection_path=work/'selection.json'
            if selection_path.exists():
                selection=load_json_strict(selection_path)
                if selection.get('experiment')!=fingerprint or selection.get('aligned_data_sha256')!=aligned.manifest_sha256:
                    raise ValueError('Committed selection experiment differs')
            else:
                results=[];failures=[]
                checkpoints=[run/'candidates'/f'milestone_{train.target_tokens//2}.pt',latest]
                for checkpoint in checkpoints:
                    if not checkpoint.is_file():raise FileNotFoundError(checkpoint)
                    for policy in range(len(POLICIES)):
                        write_json_atomic(phase,{'phase':'dev_evaluation','checkpoint':str(checkpoint),'policy':policy})
                        try:
                            report=evaluate(checkpoint,original,aligned,'dev',policy,
                                            work/f'{checkpoint.stem}_policy{policy}_dev.json',args.scorer_cache)
                            results.append({'checkpoint':str(checkpoint),'policy':policy,'report':report})
                        except (RuntimeError,ValueError,OSError) as exc:
                            failures.append({'checkpoint':str(checkpoint),'policy':policy,'error':str(exc)})
                        write_json_atomic(work/'dev_results.json',{'results':results,'failures':failures})
                selected=select_improved(results,set(baseline['declared_ids']),baseline['metrics']['bertscore_f1'])
                selection=commit_selection(selection_path,{'experiment':fingerprint,
                    'aligned_data_sha256':aligned.manifest_sha256,'baseline':baseline,'selected':selected,
                    'rule':'full dev BERTScore, then ROUGE-L, then BLEU-4','failures':failures,
                    'confirmation_and_test_not_used_for_selection':True})
            selected=selection['selected'];checkpoint=Path(selected['checkpoint']);policy=selected['policy']
            verified=evaluate(checkpoint,original,aligned,'dev',policy,
                              work/f'{checkpoint.stem}_policy{policy}_dev.json',args.scorer_cache)
            if verified!=selected['report']:raise ValueError('Selected dev evidence changed')
            write_json_atomic(phase,{'phase':'confirmation','selected_policy':policy})
            base_confirm=evaluate(args.parent_checkpoint,original,None,'confirmation',0,
                                  work/'baseline_confirmation.json',args.scorer_cache)
            confirm=evaluate(checkpoint,original,aligned,'confirmation',policy,
                             work/'selected_confirmation.json',args.scorer_cache)
            results={'baseline_B_leaderboard':{'bertscore_f1':.7972,'rouge_l_f1':.1595,'bleu4':.0125},
                'baseline_dev':baseline['metrics'],'selected_dev':verified['metrics'],
                'baseline_confirmation':base_confirm['metrics'],'selected_confirmation':confirm['metrics'],
                'confirmation_bertscore_gain':confirm['metrics']['bertscore_f1']-base_confirm['metrics']['bertscore_f1'],
                'improved_on_confirmation':confirm['metrics']['bertscore_f1']>base_confirm['metrics']['bertscore_f1'],
                'target_met_on_confirmation':confirm['metrics']['bertscore_f1']>=config['target_bertscore_f1'],
                'leaderboard_score':None,'selected':selected,
                'score_scope':'same held-out IDs/evaluator; leaderboard result still requires Codabench'}
            write_json_atomic(work/'experiment_result.json',results)
            write_json_atomic(phase,{'phase':'official_generation','selected_policy':policy,
                                     'improved_on_confirmation':results['improved_on_confirmation']})
            with run_lock(work/'official-generation'):
                submission=export(checkpoint,original,aligned,policy,args.output,selection_path,args.test_file)
            write_json_atomic(work/'experiment_result.json',{**results,'submission':submission})
            write_json_atomic(phase,{'phase':'complete','submission_zip':str(args.output),
                                     'zip_sha256':submission['prediction_zip_sha256'],
                                     'improved_on_confirmation':results['improved_on_confirmation'],
                                     'target_met_on_confirmation':results['target_met_on_confirmation']})
            return 0
        except Exception as exc:
            write_json_atomic(phase,{'phase':'failed','error':type(exc).__name__+': '+str(exc)})
            print(type(exc).__name__+': '+str(exc),file=sys.stderr,flush=True);return 2


if __name__=='__main__':raise SystemExit(main())
