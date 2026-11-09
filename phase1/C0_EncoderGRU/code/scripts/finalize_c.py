"""Wait for the exact C trainer, measure dev policies, confirm and export.

This is experiment orchestration, not a modification of frozen training code.
It never resumes/restarts training or selects using official test/confirmation.
"""
import argparse
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

from cs781.checkpoints import run_lock
from cs781.cli_c import (
    _config, _decode_config, _download_official_test, _finalize_official_export,
    _generate_records, _model_from_checkpoint, _official_dataset_revision,
    _require_complete_training,
)
from cs781.io import sha256_file, write_json_atomic
from cs781.prepare import load_prepared
from cs781.submission import extract_official_test_prompts, load_json_strict
from cs781.tokenizer import load_tokenizer


MATCHED_BUDGET = 209914685
TEN_PASS_BUDGET = 419829370


def select_dev(reports, expected_ids):
    if not reports:
        raise ValueError('No successful full dev results')
    for item in reports:
        report = item['report']
        metrics = report['metrics']
        if (report.get('split') != 'dev' or report.get('limited_sample') is not False
                or set(report.get('declared_ids', [])) != set(expected_ids)
                or any(not math.isfinite(metrics[k]) or not 0 <= metrics[k] <= 1
                       for k in ('bertscore_f1', 'rouge_l_f1', 'bleu4'))):
            raise ValueError('Selection requires complete full dev results only')
    return max(reports, key=lambda item: tuple(item['report']['metrics'][key]
               for key in ('bertscore_f1', 'rouge_l_f1', 'bleu4')))


def require_selected_candidate(checkpoint, payload, run_dir, metadata):
    checkpoint, run_dir = Path(checkpoint).resolve(), Path(run_dir).resolve()
    latest = run_dir/'checkpoints/latest.pt'
    budget = metadata.get('train_config', {}).get('target_tokens')
    if type(budget) is not int or budget < TEN_PASS_BUDGET:
        raise ValueError('Selected export requires the declared ten-pass parent budget')
    if checkpoint == latest:
        _require_complete_training(checkpoint, payload, run_dir, metadata)
        return load_json_strict(run_dir/'training_status.json')
    status = load_json_strict(run_dir/'training_status.json')
    final_state = status.get('state', {})
    if (checkpoint.parent != run_dir/'candidates' or metadata.get('synthetic_smoke') is not False
            or status.get('architecture') != 'C' or status.get('status') != 'complete'
            or budget < TEN_PASS_BUDGET
            or status.get('target_tokens') != budget
            or type(final_state.get('target_tokens')) is not int
            or final_state['target_tokens'] < budget
            or status.get('code_sha256') != metadata.get('code_sha256')
            or status.get('checkpoint_sha256') != sha256_file(latest)):
        raise ValueError('Selected candidate requires a verified completed parent run')
    state = payload.get('state', {})
    if (type(state.get('target_tokens')) is not int or state['target_tokens'] < MATCHED_BUDGET
            or state['target_tokens'] > final_state['target_tokens']
            or type(state.get('step')) is not int or state['step'] < 1
            or state['step'] > final_state.get('step', -1)
            or payload.get('_loaded_checkpoint_sha256') != sha256_file(checkpoint)):
        raise ValueError('Selected checkpoint has not reached the matched budget or changed')
    return status


def commit_selection(path, selection, expected_ids):
    path = Path(path)
    if path.exists():
        previous = load_json_strict(path)
        if any(previous.get(key) != selection.get(key)
               for key in ('config_sha256', 'data_manifest_sha256')):
            raise ValueError('Committed selection belongs to a different config or dataset')
        select_dev([previous['selected']], expected_ids)
        return previous
    select_dev([selection['selected']], expected_ids)
    write_json_atomic(path, selection)
    return selection


def trainer_alive(pid, run_dir):
    command = Path(f'/proc/{pid}/cmdline')
    if not command.exists():
        return False
    try:
        args = command.read_bytes().replace(b'\0', b' ').decode()
    except FileNotFoundError:
        return False
    # A reused PID or a terminated zombie must not prolong the wait.
    return 'cs781.cli_c train' in args and str(run_dir) in args


def run_evaluation(data, checkpoint, config, policy, split, output, scorer_cache):
    context = output.with_suffix('.context.json')
    if not context.exists():
        args = [sys.executable, '-u', '-m', 'cs781.cli_c', 'evaluate', '--data', str(data.root),
                '--checkpoint', str(checkpoint), '--output', str(output), '--config', str(config),
                '--policy', str(policy), '--split', split, '--scorer-cache', str(scorer_cache),
                '--device', 'cuda']
        with output.with_suffix('.log').open('a') as log:
            result = subprocess.run(args, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT)
        if result.returncode:
            raise RuntimeError(f'Evaluation failed: {output.with_suffix(".log")}')
    report = load_json_strict(context)
    metric_record = load_json_strict(output.with_suffix('.metrics.json'))
    expected_decode = _decode_config(_config(config), policy)
    from dataclasses import asdict
    if (report.get('checkpoint_sha256') != sha256_file(checkpoint)
            or report.get('decoding') != asdict(expected_decode)
            or report.get('split') != split or report.get('limited_sample') is not False
            or report.get('synthetic_smoke') is not False
            or metric_record.get('prediction_sha256') != sha256_file(output)
            or metric_record.get('reference_sha256') != sha256_file(output.with_suffix('.references.json'))
            or metric_record.get('metrics') != report.get('metrics')):
        raise ValueError('Saved evaluation context does not match this experiment')
    return report


def export_selected(data, checkpoint, config, policy, output, selection):
    from dataclasses import asdict
    model, payload, run_dir, metadata, device = _model_from_checkpoint(checkpoint, data, 'cuda')
    parent = require_selected_candidate(checkpoint, payload, run_dir, metadata)
    manifest = load_json_strict(data.root/'manifest.json')
    revision = _official_dataset_revision(manifest)
    test = _download_official_test(revision, data.root.parent/'hf_hub')
    prompts = extract_official_test_prompts(load_json_strict(test))
    decode = _decode_config(_config(config), policy)
    fingerprint = {'checkpoint': payload['_loaded_checkpoint_sha256'],
                   'tokenizer': data.tokenizer_sha256, 'code': metadata['code_sha256'],
                   'policy': asdict(decode), 'test': sha256_file(test), 'dataset_revision': revision,
                   'selection_sha256': sha256_file(selection), 'orchestration_sha256': sha256_file(__file__)}
    predictions = _generate_records(output, prompts, model,
                                     load_tokenizer(data.root/'tokenizer.json'), decode, fingerprint)
    write_json_atomic(output.with_suffix('.predictions.json'),
                      {key: {'text': text} for key, text in predictions.items()})
    report = {'architecture': 'C', 'checkpoint_sha256': payload['_loaded_checkpoint_sha256'],
              'checkpoint_state': payload['state'], 'checkpoint_run_dir': str(run_dir),
              'completed_parent_state': parent['state'], 'code_sha256': metadata['code_sha256'],
              'data_manifest_sha256': data.manifest_sha256, 'tokenizer_sha256': data.tokenizer_sha256,
              'dataset_revision': revision, 'test_sha256': sha256_file(test),
              'generation_config': asdict(decode), 'synthetic_smoke': False,
              'prediction_count': len(predictions), 'selection_sha256': sha256_file(selection),
              'orchestration_sha256': sha256_file(__file__)}
    return _finalize_official_export(output, predictions, prompts, report)


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--training-pid', type=int, required=True)
    for name in ('run-dir', 'data', 'config', 'scorer-cache'):
        parser.add_argument('--'+name, type=Path, required=True)
    args = parser.parse_args(argv)
    run, status_path = args.run_dir.resolve(), args.run_dir/'finalizer_status.json'
    work = run/'evaluation'
    work.mkdir(parents=True, exist_ok=True)
    with run_lock(run/'finalizer'):
        try:
            while trainer_alive(args.training_pid, run):
                write_json_atomic(status_path, {'phase': 'waiting_for_training',
                                                'training_pid': args.training_pid})
                time.sleep(60)
            complete = load_json_strict(run/'training_status.json')
            if complete.get('status') != 'complete':
                raise RuntimeError('C trainer ended without a completed budget; no automatic restart')
            metadata = load_json_strict(run/'run.json')
            budget = metadata.get('train_config', {}).get('target_tokens')
            if (metadata.get('architecture') != 'C' or metadata.get('synthetic_smoke') is not False
                    or type(budget) is not int or budget < TEN_PASS_BUDGET
                    or complete.get('target_tokens') != budget
                    or complete.get('state', {}).get('target_tokens', -1) < budget
                    or complete.get('code_sha256') != metadata.get('code_sha256')
                    or complete.get('checkpoint_sha256') != sha256_file(run/'checkpoints/latest.pt')):
                raise ValueError('Dev selection requires the verified completed ten-pass parent')
            data = load_prepared(args.data)
            prompts = load_json_strict(data.root/'dev.prompts.json')
            selection_path = work/'selection.json'
            checkpoints = [run/'candidates'/f'milestone_{MATCHED_BUDGET}.pt',
                           run/'checkpoints/latest.pt']
            results, failures = [], []
            # Once committed, selection is immutable. Retrying failed candidates
            # must never invalidate confirmation or partially generated test IDs.
            if selection_path.exists():
                selection = load_json_strict(selection_path)
                selection = commit_selection(selection_path,
                    {**selection, 'config_sha256': sha256_file(args.config),
                     'data_manifest_sha256': data.manifest_sha256}, set(prompts))
                selected = selection['selected']
            for checkpoint in (() if selection_path.exists() else checkpoints):
                if not checkpoint.is_file():
                    raise FileNotFoundError(f'Required C checkpoint missing: {checkpoint}')
                for policy in range(len(_config(args.config)['decoding'])):
                    write_json_atomic(status_path, {'phase': 'dev_evaluation',
                                                    'checkpoint': str(checkpoint), 'policy': policy})
                    output = work/f'{checkpoint.stem}_policy{policy}_dev.json'
                    try:
                        report = run_evaluation(data, checkpoint, args.config, policy,
                                                'dev', output, args.scorer_cache)
                        results.append({'checkpoint': str(checkpoint), 'policy': policy, 'report': report})
                    except (RuntimeError, ValueError, OSError) as exc:
                        failures.append({'checkpoint': str(checkpoint), 'policy': policy, 'error': str(exc)})
                    write_json_atomic(work/'dev_results.json', {'results': results, 'failures': failures})
            if not selection_path.exists():
                selected = select_dev(results, set(prompts))
                selection = {'rule': 'full dev BERTScore, then ROUGE-L, then BLEU-4',
                         'baseline_B_leaderboard': {'bertscore_f1': 0.7972, 'rouge_l_f1': 0.1595, 'bleu4': 0.0125},
                         'target_bertscore_f1': 0.84, 'selected': selected,
                         'attempted_results': results, 'failed_combinations': failures,
                         'test_not_used_for_selection': True,
                         'config_sha256': sha256_file(args.config),
                         'data_manifest_sha256': data.manifest_sha256}
                selection = commit_selection(selection_path, selection, set(prompts))
            checkpoint, policy = Path(selected['checkpoint']), selected['policy']
            verified_dev = run_evaluation(data, checkpoint, args.config, policy, 'dev',
                work/f'{checkpoint.stem}_policy{policy}_dev.json', args.scorer_cache)
            if verified_dev != selected['report']:
                raise ValueError('Committed dev selection no longer matches saved evidence')
            write_json_atomic(status_path, {'phase': 'confirmation', 'selected_policy': policy})
            confirmation = run_evaluation(data, checkpoint, args.config, policy, 'confirmation',
                                           work/'selected_confirmation.json', args.scorer_cache)
            write_json_atomic(status_path, {'phase': 'official_generation', 'selected_policy': policy})
            with run_lock(work/'official-generation'):
                export = export_selected(data, checkpoint, args.config, policy,
                                          run.parent/'CS781_Model_C_Codabench_Submission_20261003.zip',
                                          selection_path)
            write_json_atomic(work/'experiment_result.json', {
                'dev': selected['report']['metrics'], 'confirmation': confirmation['metrics'],
                'target_met_on_confirmation': confirmation['metrics']['bertscore_f1'] >= 0.84,
                'leaderboard_score': None, 'submission': export,
                'baseline_scope': 'B is leaderboard; C scores here are held-out local evaluation'})
            write_json_atomic(status_path, {'phase': 'complete', 'submission_zip':
                                            str(run.parent/'CS781_Model_C_Codabench_Submission_20261003.zip')})
            return 0
        except Exception as exc:
            write_json_atomic(status_path, {'phase': 'failed', 'error': type(exc).__name__+': '+str(exc)})
            print(type(exc).__name__+': '+str(exc), file=sys.stderr)
            return 2


if __name__ == '__main__':
    raise SystemExit(main())
