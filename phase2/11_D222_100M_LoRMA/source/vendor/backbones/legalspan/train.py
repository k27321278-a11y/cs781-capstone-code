"""Fresh foundation training, normalized copy NLL and strict deterministic resume."""
import contextlib
import hashlib
import json
import math
import os
import random
import time
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import torch
from cs781.checkpoints import run_lock, save_checkpoint, restore_checkpoint
from cs781.contracts import TrainConfig, TrainState
from cs781.io import write_json_atomic, sha256_file
from .bpe import OwnBPE
from .data import load_data, MixedExamples
from .model import LegalSpanGPT, SpanConfig


def code_digest():
    root = Path(__file__).resolve().parents[1]
    names = ['legalspan/' + n for n in ('bpe.py', 'data.py', 'model.py', 'train.py')]
    names += ['cs781/' + n for n in ('checkpoints.py', 'contracts.py', 'models/a.py', 'io.py')]
    digest = hashlib.sha256()
    for name in names:
        digest.update(name.encode()); digest.update((root / name).read_bytes())
    return digest.hexdigest()


def batch(row, device):
    prefix, suffix = row['prefix_ids'], row['suffix_ids']
    m = len(prefix) + 1
    x = torch.tensor([[1] + prefix + suffix], dtype=torch.long, device=device)
    y = torch.tensor([[-100] * (m-1) + suffix + [2]], dtype=torch.long, device=device)
    roles = torch.tensor([[0] + row['prompt_roles']], dtype=torch.long, device=device)
    return x, y, roles, m


def atomic_model(path, model, state, metadata):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix('.tmp')
    with temp.open('wb') as f:
        torch.save({'model': model.state_dict(), 'state': asdict(state), 'metadata': metadata}, f)
        f.flush(); os.fsync(f.fileno())
    os.replace(temp, path)


def train(data_root, model_cfg, cfg, run, resume=False, smoke=False, stop_after_steps=None):
    data_root, run = Path(data_root), Path(run)
    if min(cfg.target_tokens, cfg.grad_accum_steps, cfg.save_every, cfg.log_every) <= 0 or cfg.micro_batch_size != 1:
        raise ValueError('Positive token/batch/interval budgets and microbatch=1 required')
    if not math.isfinite(cfg.learning_rate) or cfg.learning_rate <= 0:
        raise ValueError('Positive finite learning rate required')
    manifest = load_data(data_root)
    if manifest['synthetic'] and not smoke:
        raise ValueError('Synthetic data is engineering-only')
    tok = OwnBPE.load(data_root / 'tokenizer.json')
    if model_cfg.vocab_size != tok.get_vocab_size() or model_cfg.max_seq_len != manifest['max_seq_len']:
        raise ValueError('Model vocabulary/context must match prepared own data')
    with run_lock(run):
        path = run / 'checkpoints/latest.pt'
        if path.exists() and not resume: raise FileExistsError('Explicit resume required')
        if resume and not path.exists(): raise FileNotFoundError('Resume checkpoint missing')
        random.seed(cfg.seed); np.random.seed(cfg.seed); torch.manual_seed(cfg.seed)
        device = torch.device(cfg.device)
        if device.type == 'cuda':
            torch.cuda.manual_seed_all(cfg.seed); torch.cuda.reset_peak_memory_stats()
        model = LegalSpanGPT(model_cfg).to(device)
        if not smoke and not 50000000 <= model.count_parameters() <= 300000000:
            raise ValueError('Foundation parameter count outside course bounds')
        optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate,
                    betas=(cfg.beta1, cfg.beta2), weight_decay=cfg.weight_decay, eps=cfg.eps)
        metadata = {'architecture': 'A-legalspan-' + model_cfg.route_mode,
                    'model_config': asdict(model_cfg), 'train_config': asdict(cfg),
                    'data_manifest_sha256': sha256_file(data_root / 'manifest.json'),
                    'tokenizer_sha256': sha256_file(data_root / 'tokenizer.json'),
                    'code_sha256': code_digest(), 'torch': str(torch.__version__),
                    'numpy': np.__version__, 'synthetic_smoke': bool(smoke),
                    'unique_parameters': model.count_parameters(), 'initialization': 'random_seed_781_no_pretrained_participant'}
        state = TrainState()
        if resume:
            state, sampler = restore_checkpoint(path, model, optimizer, metadata)
            if sampler['cursor'] != state.example_cursor: raise ValueError('Checkpoint cursor mismatch')
        write_json_atomic(run / 'run.json', metadata)
        sampler = MixedExamples(data_root, cfg.seed)
        began, initial = time.monotonic(), state.target_tokens
        milestones = (cfg.target_tokens // 4, cfg.target_tokens // 2)
        model.train()
        try:
            with (run / 'metrics.jsonl').open('a') as log:
                while state.target_tokens < cfg.target_tokens and (stop_after_steps is None or state.step < stop_after_steps):
                    rows = [sampler.at(state.example_cursor+i) for i in range(cfg.grad_accum_steps)]
                    target_count = sum(len(r['suffix_ids'])+1 for r in rows)
                    fraction = state.target_tokens / cfg.target_tokens
                    if fraction < cfg.warmup_ratio:
                        scale = max(1/cfg.target_tokens, fraction / max(cfg.warmup_ratio, 1e-12))
                    else:
                        progress = min(1., (fraction-cfg.warmup_ratio) / max(1-cfg.warmup_ratio, 1e-12))
                        scale = cfg.min_lr_ratio + (1-cfg.min_lr_ratio) * .5 * (1+math.cos(math.pi*progress))
                    for group in optimizer.param_groups: group['lr'] = cfg.learning_rate * scale
                    optimizer.zero_grad(set_to_none=True)
                    step_loss, inputs = 0., 0
                    for row in rows:
                        x, y, roles, m = batch(row, device)
                        if x.shape[1] > model_cfg.max_seq_len: raise ValueError('Training context exhausted')
                        context = torch.autocast('cuda', dtype=torch.bfloat16) if device.type == 'cuda' else contextlib.nullcontext()
                        with context:
                            _, loss, _ = model(x, targets=y, prompt_length=m, prompt_roles=roles, return_probs=False)
                        if not torch.isfinite(loss): raise FloatingPointError('Nonfinite NLL; last checkpoint preserved')
                        weight = (len(row['suffix_ids'])+1) / target_count
                        (loss * weight).backward(); step_loss += loss.item() * weight
                        inputs += x.numel()
                    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip, error_if_nonfinite=True)
                    optimizer.step()
                    previous = state.target_tokens
                    state = replace(state, step=state.step+1, input_tokens=state.input_tokens+inputs,
                                    target_tokens=state.target_tokens+target_count,
                                    example_cursor=state.example_cursor+len(rows))
                    elapsed = time.monotonic() - began
                    status = {**asdict(state), 'status': 'running', 'loss': step_loss,
                              'lr': optimizer.param_groups[0]['lr'], 'grad_norm': float(norm),
                              'target_budget': cfg.target_tokens, 'target_tokens_per_second': (state.target_tokens-initial)/elapsed,
                              'session_seconds': elapsed, 'peak_memory_bytes': torch.cuda.max_memory_allocated() if device.type == 'cuda' else 0,
                              'source_examples': dict(__import__('collections').Counter(r['source'] for r in rows)),
                              'window_kinds': dict(__import__('collections').Counter(r['kind'] for r in rows))}
                    if state.step % cfg.log_every == 0 or state.step == 1:
                        log.write(json.dumps(status, allow_nan=False)+'\n'); log.flush()
                        print(json.dumps(status, allow_nan=False), flush=True)
                    write_json_atomic(run / 'training_status.json', status)
                    for milestone in milestones:
                        if previous < milestone <= state.target_tokens:
                            atomic_model(run / f'candidates/targets_{milestone}.pt', model, state, metadata)
                    if state.step % cfg.save_every == 0:
                        save_checkpoint(path, model, optimizer, state, {'cursor': state.example_cursor}, metadata)
                save_checkpoint(path, model, optimizer, state, {'cursor': state.example_cursor}, metadata)
                atomic_model(run / 'candidates/final.pt', model, state, metadata)
                write_json_atomic(run / 'training_status.json', {**asdict(state), 'status': 'complete' if state.target_tokens >= cfg.target_tokens else 'engineering_stop', 'target_budget': cfg.target_tokens,
                                  'session_seconds': time.monotonic()-began, 'checkpoint_sha256': sha256_file(path)})
        except Exception as exc:
            write_json_atomic(run / 'training_status.json', {**asdict(state), 'status': 'failed', 'error': str(exc), 'target_budget': cfg.target_tokens})
            raise
        finally:
            sampler.handle.close()
        return state


if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser()
    for a in ('data', 'run', 'config'): p.add_argument('--'+a, type=Path, required=True)
    p.add_argument('--resume', action='store_true'); p.add_argument('--smoke', action='store_true')
    p.add_argument('--stop-after-steps', type=int)
    a = p.parse_args(); settings = json.loads(a.config.read_text())
    train(a.data, SpanConfig(**settings['model']), TrainConfig(**settings['train']), a.run,
          resume=a.resume, smoke=a.smoke, stop_after_steps=a.stop_after_steps)
