"""Locked pilot→foundation→dev selection→confirmation→official ZIP controller.

All generated text comes from the participant model. Scoring encoder use is confined
to evaluation. Failure preserves progress; it never manufactures text or scores.
"""
import argparse
import gc
import hashlib
import json
import math
import os
import re
import subprocess
import sys
import time
from collections import Counter
from dataclasses import asdict, replace
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import torch
from cs781.checkpoints import run_lock
from cs781.contracts import GenerationConfig, TrainConfig
from cs781.evaluate import score_predictions, load_scorer_provenance
from cs781.io import sha256_file, write_json_atomic
from cs781.prediction_progress import PredictionProgress
from cs781.submission import load_json_strict, extract_official_test_prompts, validate_predictions, write_submission
from .bpe import OwnBPE
from .data import load_data
from .generate import generate
from .model import SpanConfig, LegalSpanGPT
from .train import code_digest

ROOT = Path(__file__).resolve().parents[1]
POLICIES = [GenerationConfig(max_new_tokens=2048, repetition_penalty=p) for p in (1., 1.1)]
POLICIES.append(GenerationConfig(strategy='nucleus', temperature=.8, top_p=.9, repetition_penalty=1.05, max_new_tokens=2048))


def inference_digest():
    digest = hashlib.sha256(code_digest().encode())
    for name in ('legalspan/generate.py', 'legalspan/pipeline.py', 'cs781/submission.py', 'cs781/prediction_progress.py', 'cs781/evaluate.py'):
        digest.update(name.encode()); digest.update((ROOT / name).read_bytes())
    return digest.hexdigest()


def choose(results, expected):
    eligible = [r for r in results if set(r.get('ids', [])) == set(expected)
                and all(math.isfinite(r['metrics'][k]) for k in r['metrics'])]
    if not eligible: raise ValueError('No complete finite dev result; no filtered selection')
    return max(eligible, key=lambda r: r['metrics']['bertscore_f1'])


def screening_gate(reports, expected):
    if set(reports) != {'control', 'generic', 'full'}:
        raise ValueError('All three matched pilot dev reports required')
    for report in reports.values():
        if set(report['ids']) != set(expected) or any(not math.isfinite(v) for v in report['metrics'].values()):
            raise ValueError('Pilot screening needs complete finite dev coverage')
    control = reports['control']['diagnostics']['per_id']
    repetition_rate = sum(r['repeated_4gram_fraction'] > .1 for r in control.values()) / len(control)
    return {'pilot_engineering_only': True, 'all_dev_ids': len(expected),
            'control_fraction_with_over_10pct_repeated_4grams': repetition_rate,
            'screen_control_repetition_intervention': repetition_rate >= .25,
            'decision': 'Test full-route hypothesis substantively; pilot scores are not convergence or an improvement claim'}


def context_preflight(prompts, tokenizer, max_seq_len, max_new_tokens):
    lengths = []
    if len(prompts) != 1000: raise ValueError('Official preflight requires all 1000 IDs')
    for key, text in prompts.items():
        if len(text.split()) != 512: raise ValueError('Official prompt word contract differs')
        encoded = tokenizer.encode(text)
        if tokenizer.decode(encoded.ids) != text: raise ValueError('Prompt own-BPE round trip failed')
        length = len(encoded.ids) + 1
        if length + max_new_tokens > max_seq_len:
            raise ValueError(f'Full prompt and declared generation cap exceed context: {key}')
        lengths.append(length)
    return {'count': len(prompts), 'min_prompt_tokens': min(lengths), 'max_prompt_tokens': max(lengths),
            'max_new_tokens': max_new_tokens, 'max_seq_len': max_seq_len, 'truncated_ids': []}


def commit_selection(path, value):
    path = Path(path)
    if path.exists():
        if load_json_strict(path) != value: raise ValueError('Selection is immutable')
    else: write_json_atomic(path, value)
    return value


def package(predictions, prompts, output):
    validate_predictions(predictions, prompts)
    output = Path(output)
    if output.exists():
        with ZipFile(output) as z:
            if z.namelist() != ['prediction.json'] or z.testzip() is not None:
                raise ValueError('Existing ZIP root or CRC invalid')
            pairs = []
            actual = json.loads(z.read('prediction.json'), object_pairs_hook=lambda x: _strict_pairs(x))
            if actual != {k: {'text': v.strip()} for k, v in predictions.items()}:
                raise ValueError('Existing ZIP predictions differ')
        return sha256_file(output)
    return write_submission(predictions, prompts, output)


def _strict_pairs(items):
    result = {}
    for k, v in items:
        if k in result: raise ValueError('Duplicate ZIP JSON key')
        result[k] = v
    return result


def load_model(checkpoint, data, allow_smoke=False):
    checkpoint, data = Path(checkpoint), Path(data)
    payload = torch.load(checkpoint, map_location='cpu', weights_only=False)
    meta = payload['metadata']
    if meta != load_json_strict(checkpoint.parent.parent / 'run.json'):
        raise ValueError('Checkpoint/run metadata differs')
    if meta['data_manifest_sha256'] != sha256_file(data / 'manifest.json') or meta['tokenizer_sha256'] != sha256_file(data / 'tokenizer.json') or meta['code_sha256'] != code_digest():
        raise ValueError('Checkpoint data/tokenizer/code lineage differs')
    if meta['synthetic_smoke'] and not allow_smoke: raise ValueError('Engineering checkpoint is not a submission foundation')
    model = LegalSpanGPT(SpanConfig(**meta['model_config'])).cuda().eval()
    if model.count_parameters() != meta['unique_parameters']: raise ValueError('Unique parameter mismatch')
    model.load_state_dict(payload['model'], strict=True)
    del payload['model']
    return model, payload


def generate_records(checkpoint, data, prompts, policy, output, allow_smoke=False):
    model, payload = load_model(checkpoint, data, allow_smoke)
    cfg = POLICIES[policy]
    fingerprint = {'checkpoint_sha256': sha256_file(checkpoint), 'code_sha256': inference_digest(),
                   'tokenizer_sha256': payload['metadata']['tokenizer_sha256'], 'policy': asdict(cfg)}
    progress = PredictionProgress(Path(output).with_suffix('.progress.json'), prompts, fingerprint)
    tok = OwnBPE.load(Path(data) / 'tokenizer.json')
    began = time.monotonic()
    for i, key in enumerate(list(progress.pending())):
        try: progress.record(key, generate(model, tok, prompts[key], cfg))
        except (RuntimeError, ValueError) as exc: progress.failure(key, type(exc).__name__ + ': ' + str(exc))
        if i % 10 == 0: print(json.dumps({'phase': 'generation', 'output': str(output), 'done': len(progress.predictions), 'total': len(prompts), 'seconds': time.monotonic()-began}), flush=True)
    del model; gc.collect(); torch.cuda.empty_cache()
    predictions = progress.complete()
    write_json_atomic(output, {key: {'text': text} for key, text in predictions.items()})
    return predictions, fingerprint


def diagnostics(predictions, references, prompts):
    results = {}
    for key, text in predictions.items():
        words = text.split(); grams = [tuple(words[i:i+4]) for i in range(len(words)-3)]
        reference_digits = Counter(re.findall(r'\d+', references[key]['text']))
        output_digits = Counter(re.findall(r'\d+', text))
        results[key] = {'repeated_4gram_fraction': 1-len(set(grams))/max(1, len(grams)),
                        'distinct_word_count': len(set(words)),
                        'digit_multiset_precision': sum((reference_digits & output_digits).values())/max(1, sum(output_digits.values())),
                        'digit_multiset_recall': sum((reference_digits & output_digits).values())/max(1, sum(reference_digits.values()))}
    return {'per_id': results, 'digit_metric_note': 'Surface strings only; neither citation accuracy nor factual correctness',
            'mean_repeated_4gram_fraction': float(np.mean([r['repeated_4gram_fraction'] for r in results.values()]))}


def evaluate(checkpoint, data, split, policy, output, scorer, allow_smoke=False):
    data, output = Path(data), Path(output)
    prompts = {k: v['text'] for k, v in load_json_strict(data / f'{split}.prompts.json').items()}
    refs = load_json_strict(data / f'{split}.references.json')
    if set(prompts) != set(refs): raise ValueError('Evaluation ID mismatch')
    predictions, fingerprint = generate_records(checkpoint, data, prompts, policy, output, allow_smoke)
    context_path = output.with_suffix('.context.json')
    if context_path.exists():
        old = load_json_strict(context_path)
        if old['generation'] != fingerprint: raise ValueError('Evaluation provenance mismatch')
        return old
    metrics = score_predictions(data / f'{split}.references.json', output, ROOT / 'vendor/phase1_evaluate.py', scorer)
    report = {'checkpoint': str(checkpoint), 'checkpoint_sha256': sha256_file(checkpoint), 'policy': policy,
              'ids': list(prompts), 'split': split, 'metrics': metrics, 'generation': fingerprint,
              'diagnostics': diagnostics(predictions, refs, prompts), 'limited_sample': False,
              'engineering_only': bool(allow_smoke)}
    write_json_atomic(context_path, report)
    return report


def export(checkpoint, data, policy, output, test_path):
    manifest = load_data(data)
    if manifest['synthetic']: raise ValueError('Official export requires official training data')
    prompts = extract_official_test_prompts(load_json_strict(test_path))
    predictions, fingerprint = generate_records(checkpoint, data, prompts, policy, Path(output).with_suffix('.json'))
    digest = package(predictions, prompts, output)
    write_json_atomic(Path(output).with_suffix('.context.json'), {
        'architecture': 'A legal-span', 'prediction_count': len(predictions), 'zip_sha256': digest,
        'generation': fingerprint, 'data_manifest_sha256': sha256_file(Path(data)/'manifest.json'),
        'official_test_sha256': sha256_file(test_path), 'dataset_revision': manifest['catalog']['corpus_config']['revision'],
        'continuation_words': 128, 'prompt_words': 512, 'score_claim': 'No private leaderboard score inferred'})
    return digest


def run_trainer(data, run, settings, smoke=False):
    config = run / 'config.json'; run.mkdir(parents=True, exist_ok=True)
    if config.exists() and load_json_strict(config) != settings: raise ValueError('Training configuration changed')
    write_json_atomic(config, settings)
    status = run / 'training_status.json'
    if status.exists() and load_json_strict(status).get('status') == 'complete':
        payload = torch.load(run/'checkpoints/latest.pt', map_location='cpu', weights_only=False)
        if payload['state']['target_tokens'] < settings['train']['target_tokens'] or payload['metadata']['code_sha256'] != code_digest():
            raise ValueError('Claimed completion is not backed by the checkpoint')
        return
    command = [sys.executable, '-u', '-m', 'legalspan.train', '--data', str(data), '--run', str(run), '--config', str(config)]
    if (run / 'checkpoints/latest.pt').exists(): command.append('--resume')
    if smoke: command.append('--smoke')
    with (run / 'trainer.log').open('a') as log:
        process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
        write_json_atomic(run / 'trainer_handle.json', {'pid': process.pid, 'argv': command, 'started': time.time()})
        code = process.wait()
    if code: raise RuntimeError(f'Trainer exited {code}: {run}; last checkpoint preserved')
    if load_json_strict(status).get('status') != 'complete': raise ValueError('Trainer exited without completed token budget')


def per_item_results(selection, data, scorer, output):
    """Published BERTScore settings, plus per-ID values for paired uncertainty.

    Uses scoring weights only, after dev selection is frozen. Published evaluator
    remains authoritative; verify that the supplemental mean matches its result.
    """
    output = Path(output)
    if output.exists(): return load_json_strict(output)
    cache = Path(scorer).resolve(); load_scorer_provenance(cache)
    os.environ.update(HF_HUB_CACHE=str(cache), HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1')
    import bert_score
    pred_path = Path(selection['prediction_file'])
    predictions = load_json_strict(pred_path)
    refs = load_json_strict(Path(data) / f"{selection['split']}.references.json")
    keys = list(refs)
    _, _, f = bert_score.score([predictions[k]['text'] for k in keys], [refs[k]['text'] for k in keys],
                              lang='en', batch_size=8, verbose=False, device='cuda')
    values = f.cpu().tolist()
    if abs(float(np.mean(values)) - selection['metrics']['bertscore_f1']) > 1e-6:
        raise ValueError('Supplemental per-ID scoring does not match published evaluator')
    source = {r['id']: r['source'] for r in map(json.loads, (Path(data)/f"{selection['split']}.jsonl").read_text().splitlines())}
    record = {'prediction_sha256': sha256_file(pred_path), 'per_id_bertscore_f1': dict(zip(keys, values)),
              'per_source_bertscore_f1': {s: float(np.mean([v for k, v in zip(keys, values) if source[k] == s])) for s in set(source.values())}}
    write_json_atomic(output, record); return record


def paired_interval(candidate, control):
    if set(candidate) != set(control): raise ValueError('Paired comparison requires identical IDs')
    differences = np.asarray([candidate[k]-control[k] for k in sorted(candidate)])
    rng = np.random.default_rng(781)
    bootstrap = differences[rng.integers(0, len(differences), (2000, len(differences)))].mean(1)
    return {'mean_difference': float(differences.mean()), 'paired_bootstrap_95pct': np.quantile(bootstrap, [.025, .975]).tolist(), 'examples': len(differences)}


def main(argv=None):
    p = argparse.ArgumentParser()
    for name in ('data', 'run', 'scorer-cache', 'test', 'config'): p.add_argument('--'+name, type=Path, required=True)
    a = p.parse_args(argv); phase = a.run/'pipeline_status.json'
    with run_lock(a.run/'supervisor'):
        manifest = load_data(a.data); settings = load_json_strict(a.config)
        fingerprint = {'data': sha256_file(a.data/'manifest.json'), 'config': sha256_file(a.config),
                       'training_code': code_digest(), 'inference_code': inference_digest(), 'test': sha256_file(a.test)}
        commit_selection(a.run/'experiment.json', fingerprint)
        load_scorer_provenance(a.scorer_cache)
        try:
            prompts = extract_official_test_prompts(load_json_strict(a.test))
            coverage = context_preflight(prompts, OwnBPE.load(a.data/'tokenizer.json'), settings['model']['max_seq_len'], max(c.max_new_tokens for c in POLICIES))
            write_json_atomic(a.run/'public_context_preflight.json', {**coverage, 'test_sha256': sha256_file(a.test)})
            # Pilot snapshots are deliberately separate from foundation lineage.
            for mode in ('full', 'control', 'generic'):
                cfg = {'model': {**settings['model'], 'route_mode': mode},
                       'train': {**settings['train'], 'target_tokens': settings['pilot_target_tokens'], 'log_every': 1}}
                write_json_atomic(phase, {'phase': 'matched_pilot', 'mode': mode, 'engineering_only': True})
                run_trainer(a.data, a.run/'pilots'/mode, cfg, smoke=True)
            # Full-dev screening precedes any substantive candidate. Predeclare
            # one shared nucleus policy for these small, nonconverged pilots;
            # evaluate a repetition intervention on control only if needed.
            dev_ids = set(load_json_strict(a.data/'dev.prompts.json'))
            pilot_reports = {}
            for mode in ('control', 'generic', 'full'):
                write_json_atomic(phase, {'phase': 'pilot_dev_screening', 'mode': mode})
                pilot_reports[mode] = evaluate(a.run/'pilots'/mode/'candidates/final.pt', a.data, 'dev', 2,
                         a.run/'pilots'/mode/'evaluation/dev.json', a.scorer_cache, allow_smoke=True)
            gate = screening_gate(pilot_reports, dev_ids)
            if gate['screen_control_repetition_intervention']:
                write_json_atomic(phase, {'phase': 'pilot_control_repetition_screen'})
                try:
                    gate['separate_control_decoding_intervention'] = evaluate(a.run/'pilots/control/candidates/final.pt', a.data, 'dev', 1,
                                 a.run/'pilots/control/evaluation/repetition_screen.json', a.scorer_cache, allow_smoke=True)
                except (RuntimeError, ValueError) as exc:
                    gate['separate_control_decoding_intervention_failure'] = str(exc)
            write_json_atomic(a.run/'pilot_screening.json', {'reports': pilot_reports, 'gate': gate,
                              'later_decoding_policies_shared_across_architectures': [asdict(c) for c in POLICIES]})
            # Candidate first produces the requested deliverable promptly. The
            # matched control/generic follow; no architecture-gain claim precedes
            # those measurements. This ordering does not select on confirmation.
            selections, confirmations, comparisons = {}, {}, {}
            for mode in ('full', 'control', 'generic'):
                run = a.run/mode
                cfg = {'model': {**settings['model'], 'route_mode': mode}, 'train': settings['train']}
                write_json_atomic(phase, {'phase': 'substantive_training', 'mode': mode, 'target_tokens': settings['train']['target_tokens']})
                run_trainer(a.data, run, cfg)
                selection_path = run/'selection.json'
                if selection_path.exists(): selected = load_json_strict(selection_path)
                else:
                    results, failures = [], []
                    for checkpoint in (run/f"candidates/targets_{settings['train']['target_tokens']//2}.pt", run/'candidates/final.pt'):
                        for policy in range(len(POLICIES)):
                            output = run/'evaluation'/f'{checkpoint.stem}_policy{policy}_dev.json'
                            write_json_atomic(phase, {'phase': 'dev_selection', 'mode': mode, 'checkpoint': str(checkpoint), 'policy': policy})
                            try:
                                result = evaluate(checkpoint, a.data, 'dev', policy, output, a.scorer_cache)
                                result['prediction_file'] = str(output); results.append(result)
                            except (RuntimeError, ValueError) as exc:
                                failures.append({'checkpoint': str(checkpoint), 'policy': policy, 'error': str(exc)})
                            write_json_atomic(run/'dev_results.json', {'results': results, 'failures': failures})
                    selected = choose(results, dev_ids); commit_selection(selection_path, selected)
                selections[mode] = selected
                write_json_atomic(phase, {'phase': 'confirmation', 'mode': mode})
                confirmation = evaluate(Path(selected['checkpoint']), a.data, 'confirmation', selected['policy'],
                            run/'evaluation/confirmation.json', a.scorer_cache)
                confirmation['prediction_file'] = str(run/'evaluation/confirmation.json')
                confirmations[mode] = confirmation
                if mode == 'full':
                    write_json_atomic(phase, {'phase': 'official_generation', 'mode': mode})
                    zip_path = a.run/'deliverables/CS781_A_LegalSpan_Codabench.zip'
                    digest = export(Path(selected['checkpoint']), a.data, selected['policy'], zip_path, a.test)
                    write_json_atomic(a.run/'deliverable_ready.json', {'zip': str(zip_path), 'sha256': digest,
                                      'selected': selected, 'confirmation': confirmation, 'controlled_comparison': 'pending'})
                comparisons[mode] = {'dev': per_item_results(selected, a.data, a.scorer_cache, run/'evaluation/dev_per_item.json'),
                                     'confirmation': per_item_results(confirmation, a.data, a.scorer_cache, run/'evaluation/confirmation_per_item.json')}
                write_json_atomic(a.run/'measured_results.json', {'selections': selections, 'confirmations': confirmations, 'per_item': comparisons,
                                  'confirmation_limitation': 'Earlier aggregate reference statistics were inspected; not a pristine untouched holdout'})
            paired = {split: {mode: paired_interval(comparisons['full'][split]['per_id_bertscore_f1'], comparisons[mode][split]['per_id_bertscore_f1'])
                      for mode in ('control', 'generic')} for split in ('dev', 'confirmation')}
            write_json_atomic(a.run/'controlled_comparison.json', paired)
            write_json_atomic(phase, {'phase': 'complete', 'zip': str(a.run/'deliverables/CS781_A_LegalSpan_Codabench.zip'),
                                    'controlled_comparison': paired, 'leaderboard_score': None})
        except Exception as exc:
            write_json_atomic(phase, {'phase': 'failed', 'error': type(exc).__name__ + ': ' + str(exc), 'last_checkpoint_preserved': True})
            raise


if __name__ == '__main__': main()
