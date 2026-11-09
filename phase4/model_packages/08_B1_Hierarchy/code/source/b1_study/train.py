import copy
import gc
import json
import math
import random
import subprocess
import sys
import time
import zipfile
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F
from sklearn.metrics import average_precision_score, classification_report, roc_auc_score

from phase4_cjpe.common import atomic_json, digest, hardware, now, save, seed_all, sha, status
from phase4_cjpe.export import autocast, probabilities, score, select_threshold, validate_zip, write_predictions
from .data import prepare_data
from .model import build_model, optimizer_groups, set_train_stage, GatedEncoderBlock


def source_digest():
    root = Path(__file__).parents[1]
    return {p.relative_to(root).as_posix(): sha(p) for package in ('b1_study', 'phase3_rr', 'phase4_cjpe')
            for p in sorted((root / package).rglob('*'))
            if p.is_file() and '__pycache__' not in p.parts and p.suffix in ('.py', '.txt')}


def capture_rng():
    return dict(python=random.getstate(), numpy=np.random.get_state(), torch=torch.get_rng_state(),
                cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [])


def restore_rng(state):
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch'].cpu())
    if torch.cuda.is_available():
        torch.cuda.set_rng_state_all([s.cpu() for s in state['cuda']])


def metrics(labels, logits, threshold=.5):
    result = score(labels, logits, threshold)
    probs = probabilities(logits)
    pred = (probs >= threshold).astype(int)
    result['classwise'] = classification_report(labels, pred, labels=[0, 1], output_dict=True, zero_division=0)
    result['roc_auc'] = float(roc_auc_score(labels, probs))
    result['pr_auc'] = float(average_precision_score(labels, probs))
    return result


@torch.no_grad()
def predict(model, store, split, device, root):
    model.eval()
    keys, values, labels = [], [], []
    started = time.monotonic()
    for i, row in enumerate(store.rows(split)):
        with autocast(device):
            logits = model(row['chunks'], row['positions'], device).float()[0]
        if not torch.isfinite(logits).all():
            raise FloatingPointError('Nonfinite inference')
        keys.append(row['id'])
        values.append(logits.cpu().numpy())
        labels.append(row['label'])
        if (i + 1) % 100 == 0:
            status(root, 'evaluating', evaluating_split=split, evaluated_cases=i + 1, total_cases=len(store.ids[split]))
    return keys, np.stack(values), labels, (time.monotonic() - started) / max(1, len(keys))


def operational_preflight(model, store, device, root):
    """Approved launch acceptance checks; no synthetic optimizer update."""
    model.eval()
    rows = [store.row('train', k) for k in store.ids['train'][:3]]
    short = torch.tensor([[1, 4, 5, 6, 2]], device=device)
    valid = torch.ones_like(short, dtype=torch.bool)
    padded = F.pad(short, (0, 3))
    padded_valid = padded.ne(0)
    with torch.no_grad():
        # FP32 here separates padding correctness from low-precision kernel rounding.
        a, b = model.feature_batch(short, valid), model.feature_batch(padded, padded_valid)
        if not torch.allclose(a, b, atol=2e-4, rtol=2e-4):
            raise RuntimeError('Padded and unpadded chunk features disagree')
        if model.depth_mapping:
            memory = model.backbone.tok_embeddings(short)
            for block in model.backbone.encoder:
                if not isinstance(block, GatedEncoderBlock):
                    memory = block(memory, valid)
            memory = model.backbone.encoder_norm(memory)
            actual = model.backbone.encode(short, valid)[0]
            if not torch.allclose(actual, memory, atol=1e-6, rtol=1e-6):
                raise RuntimeError('Zero-gate expansion changed pretrained encoder computation')
    # Check a real maximum-profile case with backward, not a tiny unrelated token sequence.
    row = max(rows, key=lambda r: sum(map(len, r['chunks'])))
    model.train()
    with autocast(device):
        output = model(row['chunks'], row['positions'], device).float()
        loss = F.cross_entropy(output, torch.tensor([row['label']], device=device))
    if output.shape != (1, 2) or not torch.isfinite(loss):
        raise RuntimeError('Invalid classifier output')
    loss.backward()
    grads = [p.grad for p in model.parameters() if p.requires_grad and p.grad is not None]
    if not grads or not all(bool(torch.isfinite(g).all()) for g in grads) or not any(bool(g.abs().sum() > 0) for g in grads):
        raise RuntimeError('Missing or nonfinite task gradients')
    if any(p.grad is not None for p in model.backbone.parameters() if not p.requires_grad):
        raise RuntimeError('Frozen parameter unexpectedly received gradients')
    model.zero_grad(set_to_none=True)
    del grads, output, loss
    model.eval()
    started = time.monotonic()
    with torch.no_grad():
        for item in rows:
            with autocast(device):
                model(item['chunks'], item['positions'], device)
        torch.cuda.synchronize()
    seconds = (time.monotonic() - started) / len(rows)
    record = dict(padding_invariant=True, finite_task_gradient=True, frozen_parameters_respected=True,
                  zero_gate_identity_verified=bool(model.depth_mapping), inference_seconds_per_case=seconds,
                  peak_gpu_bytes=torch.cuda.max_memory_allocated(), device=hardware(),
                  profile_tokens=max(sum(map(len, r['chunks'])) for r in rows), profile_cases=len(rows))
    atomic_json(Path(root) / 'GPU_PREFLIGHT.json', record)
    print('GPU_PREFLIGHT ' + json.dumps(record), flush=True)
    return record


def export(model, store, device, config, root, state, selected_step, checkpoint_hash, stage):
    folder = root / 'exports' / ('segment_' + str(config['segment']))
    folder.mkdir(parents=True, exist_ok=True)
    keys, values, gold, _ = predict(model, store, 'calibration', device, root)
    threshold_result = select_threshold(gold, values)
    threshold = threshold_result['threshold']
    # Fold variation describes threshold stability; it does not choose another threshold.
    fold_thresholds = []
    for fold in range(4):
        indices = np.arange(len(gold)) % 4 == fold
        fold_thresholds.append(select_threshold(np.asarray(gold)[indices], values[indices])['threshold'])
    atomic_json(folder / 'calibration.json', dict(selection_split='disjoint train calibration',
                metrics=threshold_result, descriptive_fold_thresholds=fold_thresholds))
    np.savez_compressed(folder / 'calibration_probabilities.npz', ids=np.asarray(keys), logits=values, labels=np.asarray(gold))
    devkeys, devvalues, devgold, latency = predict(model, store, 'dev', device, root)
    dev_metrics = metrics(devgold, devvalues, threshold)
    dev_metrics['at_default_threshold'] = metrics(devgold, devvalues, .5)
    write_predictions(folder / 'dev_prediction.json', devkeys, devvalues, threshold)
    atomic_json(folder / 'dev_truth.json', [dict(id=k, gold=int(y)) for k, y in zip(devkeys, devgold)])
    evaluator = Path(__file__).parents[1] / 'phase4_cjpe' / 'evaluate_phase4.py'
    result = subprocess.run([sys.executable, str(evaluator), '--truth', str(folder / 'dev_truth.json'),
                             '--prediction', str(folder / 'dev_prediction.json')], capture_output=True, text=True)
    (folder / 'official_evaluation.txt').write_text(result.stdout + '\n' + result.stderr, encoding='utf-8')
    if result.returncode:
        raise RuntimeError('Official evaluator rejected dev predictions')
    np.savez_compressed(folder / 'dev_probabilities.npz', ids=np.asarray(devkeys), logits=devvalues, labels=np.asarray(devgold))
    keys, values, _, test_latency = predict(model, store, 'test', device, root)
    np.savez_compressed(folder / 'test_probabilities.npz', ids=np.asarray(keys), logits=values)
    write_predictions(folder / 'prediction.json', keys, values, threshold)
    dest = folder / ('UPLOAD_Phase4_' + config['name'] + '_' + stage + '.zip')
    temporary = dest.with_suffix('.part')
    with zipfile.ZipFile(temporary, 'w', zipfile.ZIP_DEFLATED) as archive:
        archive.write(folder / 'prediction.json', 'prediction.json')
    temporary.replace(dest)
    checks = validate_zip(dest, store.ids['test'])
    record = dict(checks, name=config['name'], submission=str(dest), dev_metrics=dev_metrics,
                  stage=stage, selected_step=selected_step, selected_checkpoint_sha256=checkpoint_hash,
                  optimizer_updates=state['step'], completed_epochs=state['epoch'], seen_training_cases=state['seen_cases'],
                  inference_seconds_per_case=dict(dev=latency, test=test_latency),
                  peak_gpu_bytes=torch.cuda.max_memory_allocated(), finished_utc=now(), hidden_leaderboard_score=None)
    atomic_json(folder / 'result.json', record)
    atomic_json(root / 'latest_export.json', record)
    print('ZIP_READY ' + json.dumps(record), flush=True)
    return record


def train_segment(config):
    root = Path(config['run_root'])
    root.mkdir(parents=True, exist_ok=True)
    seed_all(config['seed'])
    device = torch.device('cuda')
    if not torch.cuda.is_available():
        raise RuntimeError('GPU required')
    torch.set_num_threads(2)
    bundle = Path(config['bundle'])
    store, audit = prepare_data(config, bundle)
    model, evidence = build_model(bundle, config)
    model.to(device)
    torch.cuda.reset_peak_memory_stats()
    immutable = {k: v for k, v in config.items() if k not in ('segment', 'resume_root', 'run_root', 'work_root',
                     'bundle', 'data_root', 'fixed_holdout', 'segment_hours', 'session_end_unix')}
    fp = dict(config=immutable, data=audit['data'], split_ids=audit['split_ids'], sources=source_digest(),
              parent=evidence['parent_sha256'], tokenizer=evidence['tokenizer_sha256'])
    atomic_json(root / 'config.json', config)
    atomic_json(root / 'lineage.json', dict(fingerprint=fp, evidence=evidence, audit=audit))
    optimizer = torch.optim.AdamW(optimizer_groups(model, config), weight_decay=.01)
    scaler = torch.amp.GradScaler('cuda')
    state = dict(step=0, epoch=0, position=0, seen_cases=0, best_f1=-1., bad_evals=0,
                 skipped_updates=0, training_seconds=0., selected_step=None)
    latest, best = root / 'latest.pt', root / 'best.pt'
    resume = Path(config['resume_root']) if config.get('resume_root') else None
    if resume:
        candidate = resume / 'latest.pt'
        payload = torch.load(candidate, map_location='cpu', weights_only=False)
        if payload['fingerprint'] != fp:
            raise ValueError('Resume lineage/config/source mismatch')
        model.load_state_dict(payload['model'], strict=True)
        optimizer.load_state_dict(payload['optimizer'])
        scaler.load_state_dict(payload['scaler'])
        state = payload['state']
        expected_total = math.ceil(len(store.ids['train']) / config['accumulation']) * config['max_epochs']
        if payload['scheduler'] != dict(step=state['step'], total_steps=expected_total):
            raise ValueError('Resume scheduler state mismatch')
        rng = payload['rng']
        del payload
        import shutil
        shutil.copy2(resume / 'best.pt', best)
    else:
        rng = None
    preflight = operational_preflight(model, store, device, root)
    if rng:
        restore_rng(rng)
    elif state['step'] == 0:
        # Preflight consumes RNG but must not change the experiment's training randomness.
        seed_all(config['seed'])
    reserve = max(900., preflight['inference_seconds_per_case'] *
                  (len(store.ids['holdout']) + len(store.ids['calibration']) + len(store.ids['dev']) + len(store.ids['test'])) * 1.8 + 300.)
    atomic_json(root / 'budget.json', dict(export_reserve_seconds=reserve, segment_hours=config['segment_hours'],
                inference_seconds_per_case=preflight['inference_seconds_per_case']))
    started = time.monotonic()
    cutoff = min(time.time() + config['segment_hours'] * 3600, config['session_end_unix'] - reserve)
    if cutoff <= time.time():
        raise RuntimeError('Insufficient session time for training plus measured export reserve')
    last_save, previous_stage = time.monotonic(), None
    total_steps = math.ceil(len(store.ids['train']) / config['accumulation']) * config['max_epochs']

    def snapshot():
        return dict(model=model.state_dict(), optimizer=optimizer.state_dict(), scaler=scaler.state_dict(),
                    scheduler=dict(step=state['step'], total_steps=total_steps), rng=capture_rng(),
                    state=copy.deepcopy(state), fingerprint=fp, evidence=evidence)

    def persist():
        save(latest, snapshot())
        atomic_json(root / 'checkpoint.json', dict(path=str(latest), step=state['step'], epoch=state['epoch'],
                    position=state['position'], bytes=latest.stat().st_size, saved_utc=now()))
        print('CHECKPOINT_SAVED ' + json.dumps(dict(name=config['name'], step=state['step'], bytes=latest.stat().st_size)), flush=True)

    def validate():
        if state.get('last_validation_step') == state['step']:
            return
        _, logits, gold, _ = predict(model, store, 'holdout', device, root)
        measured = metrics(gold, logits, .5)
        if measured['macro_f1'] > state['best_f1']:
            state.update(best_f1=measured['macro_f1'], bad_evals=0, selected_step=state['step'])
            save(best, dict(model=model.state_dict(), state=copy.deepcopy(state), fingerprint=fp, evidence=evidence))
            atomic_json(root / 'selection.json', dict(split='fixed train holdout', metrics=measured,
                        step=state['step'], sha256=sha(best)))
        else:
            state['bad_evals'] += 1
        state['last_validation_step'] = state['step']
        print('VALIDATION ' + json.dumps(dict(name=config['name'], step=state['step'], epoch=state['epoch'], metrics=measured)), flush=True)

    try:
        while state['epoch'] < config['max_epochs']:
            if time.time() >= cutoff:
                break
            order = np.random.default_rng(config['seed'] + state['epoch']).permutation(len(store.ids['train']))
            while state['position'] < len(order) and time.time() < cutoff:
                stage = 0 if state['step'] < config['warmup_updates'] else min(3, 1 + state['epoch'])
                if stage != previous_stage:
                    names = set_train_stage(model, stage)
                    atomic_json(root / ('train_stage_' + str(stage) + '.json'), dict(step=state['step'], names=names,
                                trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad)))
                    previous_stage = stage
                indices = order[state['position']:state['position'] + config['accumulation']]
                model.train()
                optimizer.zero_grad(set_to_none=True)
                loss_sum = 0.
                tick = time.monotonic()
                for index in indices:
                    row = store.row('train', store.ids['train'][int(index)])
                    with autocast(device):
                        logits = model(row['chunks'], row['positions'], device).float()
                    loss = F.cross_entropy(logits, torch.tensor([row['label']], device=device))
                    if not torch.isfinite(loss):
                        raise FloatingPointError('Nonfinite loss')
                    loss_sum += float(loss.detach())
                    scaler.scale(loss / len(indices)).backward()
                scaler.unscale_(optimizer)
                grads = [p for p in model.parameters() if p.requires_grad and p.grad is not None]
                finite = all(bool(torch.isfinite(p.grad).all()) for p in grads)
                if not finite:
                    scaler.step(optimizer)
                    scaler.update()
                    state['skipped_updates'] += 1
                    if state['skipped_updates'] > 20:
                        raise FloatingPointError('Repeated mixed precision overflow')
                    continue
                torch.nn.utils.clip_grad_norm_(grads, 1., error_if_nonfinite=True)
                factor = min(1., (state['step'] + 1) / 100) * (.1 + .9 * .5 * (1 + math.cos(math.pi * min(state['step'] + 1, total_steps) / total_steps)))
                for group in optimizer.param_groups:
                    group['lr'] = group['initial_lr'] * factor
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                state['step'] += 1
                state['position'] += len(indices)
                state['seen_cases'] += len(indices)
                state['training_seconds'] += time.monotonic() - tick
                if state['step'] % 10 == 0 or state['step'] == 1:
                    progress = dict(name=config['name'], step=state['step'], epoch=state['epoch'], position=state['position'],
                                    seen_cases=state['seen_cases'], loss=loss_sum / len(indices), stage=stage,
                                    seconds_per_update=time.monotonic() - tick, peak_gpu_bytes=torch.cuda.max_memory_allocated())
                    status(root, 'training', **progress)
                    with (root / 'training.jsonl').open('a') as f:
                        f.write(json.dumps(progress) + '\n')
                if state['step'] % 10 == 0 or time.monotonic() - last_save >= 1800:
                    persist()
                    last_save = time.monotonic()
                if state['step'] % 500 == 0:
                    validate()
                    persist()
                    if state['epoch'] >= 2 and state['bad_evals'] >= config['patience']:
                        break
            if state['position'] >= len(order):
                state['epoch'] += 1
                state['position'] = 0
                validate()
                persist()
                print('EPOCH_COMPLETE ' + json.dumps(dict(name=config['name'], epoch=state['epoch'], step=state['step'])), flush=True)
            if state['epoch'] >= 2 and state['bad_evals'] >= config['patience']:
                break
        if state['step'] == 0:
            raise RuntimeError('No actual optimizer update completed; no submission produced')
        validate()
        persist()
        trained = state['epoch'] >= config['max_epochs'] or (state['epoch'] >= 2 and state['bad_evals'] >= config['patience'])
        chosen = torch.load(best, map_location='cpu', weights_only=False)
        selected_step = chosen['state']['step']
        model.load_state_dict(chosen['model'])
        del chosen
        record = export(model, store, device, config, root, state, selected_step, sha(best), 'final' if trained else 'paused')
        record.update(state='complete' if trained else 'paused', elapsed_segment_seconds=time.monotonic() - started,
                      fingerprint=fp, latest_checkpoint_sha256=sha(latest), latest_checkpoint_bytes=latest.stat().st_size)
        atomic_json(root / 'segment_result.json', record)
        status(root, record['state'], step=state['step'], dev_macro_f1=record['dev_metrics']['macro_f1'])
        return record
    finally:
        store.close()
        del model, optimizer, scaler
        gc.collect()
        torch.cuda.empty_cache()
