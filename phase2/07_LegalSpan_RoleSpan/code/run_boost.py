"""Time-bounded LegalSpan full fine-tuning with honest warm-start provenance."""
import argparse
import fcntl
import json
import os
import sys
import time
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'vendor'))
sys.path.insert(0, str(ROOT / 'vendor/backbones'))
import torch
from lner.data import atomic_json, sha256, validate_predictions
from lner.models import tensor_row as original_tensor_row
from legal_d.common import autocast
import legal_d.ner_train as trainer
from boost_model import load_warm_start


def source_fingerprint():
    return {str(p.relative_to(ROOT)): sha256(p) for p in sorted(ROOT.rglob('*.py'))}


def main(path, preflight_only=False):
    config = json.loads(Path(path).read_text())
    root = Path(config['output_root'])
    root.mkdir(parents=True, exist_ok=True)
    lock = (root / 'boost.lock').open('a+')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '1':
        raise ValueError('Physical GPU1 required')
    torch.set_num_threads(4)
    torch.cuda.set_per_process_memory_fraction(config['gpu_memory_fraction'], 0)
    torch.manual_seed(config['seed'])
    candidate = config['ner']['candidates'][0]
    trainer.source_hashes = source_fingerprint
    trainer.load_ner = lambda checkpoint, char_vocab, classes, current, rank, alpha: load_warm_start(config, checkpoint, char_vocab, current)

    def tensor_row(row, encoding, device):
        return original_tensor_row(row, encoding, device) + [torch.tensor(row['header_chars'], dtype=torch.long, device=device)[None]]

    trainer.tensor_row = tensor_row
    dataset, windows = trainer.prepare_ner(config)
    for split, rows in windows.items():
        texts = dataset['test'] if split == 'test' else dataset['texts']
        for row in rows:
            header = texts[row['id']][:config['header_chars']]
            row['header_chars'] = [dataset['char_vocab'].get(c, 1) for c in header] or [1]
    model, evidence = load_warm_start(config, config['warm_start_checkpoint'], len(dataset['char_vocab'])+2, candidate)
    model.cuda().train()
    # A real training window exercises full LegalSpan, header context, CRF and boundary gradients.
    row = windows['train'][0]
    args = tensor_row(row, 'own', torch.device('cuda'))
    tags = torch.tensor(row['targets'], dtype=torch.long, device='cuda')
    with autocast(torch.device('cuda')):
        loss = model.loss(args, tags, torch.ones(25, device='cuda'), config['ner']['boundary_weight'])
    if not torch.isfinite(loss):
        raise FloatingPointError('Nonfinite boosted NER preflight loss')
    loss.backward()
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
    record = {**evidence, 'phase': 'preflight_complete', 'loss': float(loss.detach()), 'grad_norm': float(norm),
              'peak_allocated_bytes': torch.cuda.max_memory_allocated(), 'hardware': torch.cuda.get_device_name(),
              'source_hashes': source_fingerprint(), 'config': config}
    atomic_json(root / 'preflight.json', record)
    print(json.dumps({k: v for k, v in record.items() if k not in ('source_hashes', 'config', 'pretrained_model_config', 'pretrained_state', 'adapted_layers')}), flush=True)
    del model, args, tags, loss
    import gc
    gc.collect(); torch.cuda.empty_cache()
    if preflight_only:
        return
    allowed = time.time() + config['training_hours']*3600
    result = trainer.train_candidate(config, candidate, config['warm_start_checkpoint'], dataset, windows, export=True, train_until=allowed)
    destination = root / 'deliverables/UPLOAD_Phase2_06_LegalSpan_Full_HeaderCRF.zip'
    source = Path(result['submission'])
    source.rename(destination)
    with zipfile.ZipFile(destination) as archive:
        if archive.namelist() != ['prediction.json'] or archive.testzip() is not None:
            raise ValueError('Boosted submission ZIP differs')
        prediction = json.loads(archive.read('prediction.json'))
        validate_predictions(prediction, dataset['test'], dataset['labels'])
    result.update({'submission': str(destination), 'sha256': sha256(destination), 'method': 'full_legalspan_header_context_crf_boundary',
                   'warm_start_ner_sha256': config['warm_start_sha256'], 'ensemble_validation_baseline': .4630667646052263,
                   'beats_ensemble_validation': result['validation']['macro_f1'] > .4630667646052263})
    atomic_json(root / 'ner' / candidate['name'] / 'complete.json', result)
    atomic_json(root / 'complete.json', result)
    print(json.dumps({k: v for k, v in result.items() if k != 'fingerprint'}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--preflight-only', action='store_true')
    args = parser.parse_args()
    try:
        main(args.config, args.preflight_only)
    except Exception as error:
        import traceback
        cfg = json.loads(Path(args.config).read_text())
        atomic_json(Path(cfg['output_root']) / 'failure.json', {'error': str(error), 'traceback': traceback.format_exc()})
        raise
