"""C1 recurrent summarizer with bottleneck adapters and recurrent readout tuning."""
import importlib
import json
import sys
import hashlib
import zipfile
from pathlib import Path

import torch
from torch import nn

import legalspan_lora

PARENT_SHA256 = 'f1f440afb15d8a3b6ffa7eb0d7299628ac9279a5170f8079f3ac6e505ff09399'
TOKENIZER_SHA256 = '5c0bd30290e0bd6919253b1ff864f52053924808aabd33482df92cc4bd471c8e'
SOURCE_C_SHA256 = 'd9c16e1b53c23db7fd2a2be3e83f09c30642012fd31a63dc8282b7e39dc98147'


class BottleneckAdapter(nn.Module):
    def __init__(self, width, bottleneck=64):
        super().__init__()
        self.down = nn.Linear(width, bottleneck)
        self.up = nn.Linear(bottleneck, width)
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def forward(self, hidden):
        return hidden + self.up(torch.nn.functional.gelu(self.down(hidden)))


class EncoderWithAdapter(nn.Module):
    def __init__(self, base, width, bottleneck=64):
        super().__init__()
        self.base = base
        self.adapter = BottleneckAdapter(width, bottleneck)

    def forward(self, hidden, mask):
        return self.adapter(self.base(hidden, mask))


class FusionWithAdapter(nn.Module):
    def __init__(self, base, width, bottleneck=64):
        super().__init__()
        self.base = base
        self.adapter = BottleneckAdapter(width, bottleneck)

    def forward(self, hidden):
        return self.adapter(self.base(hidden))


def adapt_model(model, bottleneck=64):
    model.requires_grad_(False)
    width = model.cfg.d_model
    model.encoder = nn.ModuleList(
        EncoderWithAdapter(block, width, bottleneck) for block in model.encoder)
    model.fusion = FusionWithAdapter(model.fusion, width, bottleneck)
    # The decoder-facing path is small relative to the frozen backbone and
    # gives the recurrent generator capacity to learn summary style.
    model.initial_projection.requires_grad_(True)
    model.gru.requires_grad_(True)
    model.cross_attn.requires_grad_(True)
    return model


def build(input_dir, device):
    root = Path(input_dir)
    expanded = sorted(root.rglob('legalspan_source/cs781/models/c.py'))
    if expanded:
        legalspan_lora._import_source(input_dir)
        source_sha = legalspan_lora._sha256(expanded[0])
    else:
        archives = sorted(root.rglob('legalspan_source.zip'))
        if len(archives) != 1:
            raise FileNotFoundError('C1 original source is required')
        with zipfile.ZipFile(archives[0]) as archive:
            source_sha = hashlib.sha256(archive.read('cs781/models/c.py')).hexdigest()
        for name in list(sys.modules):
            if name in ('cs781', 'legalspan') or name.startswith(('cs781.', 'legalspan.')):
                del sys.modules[name]
        sys.path.insert(0, str(archives[0]))
        importlib.invalidate_caches()
    original = importlib.import_module('cs781.models.c')
    checkpoint = legalspan_lora._unique_file(input_dir, 'c1_phase1_model.pt')
    tokenizer_path = legalspan_lora._unique_file(input_dir, 'c1_tokenizer.json')
    if (legalspan_lora._sha256(checkpoint) != PARENT_SHA256
            or legalspan_lora._sha256(tokenizer_path) != TOKENIZER_SHA256
            or source_sha != SOURCE_C_SHA256):
        raise ValueError('C1 Phase 1 parent, tokenizer or source hash mismatch')
    record = torch.load(checkpoint, map_location='cpu', weights_only=True)
    if record['metadata'].get('architecture') != 'C':
        raise ValueError('Expected original Phase 1 C recurrent architecture')
    cfg = original.RecurrentConfig(**record['metadata']['model_config'])
    model = original.RecurrentLegalModel(cfg)
    model.load_state_dict(record['model'], strict=True)
    tokenizer = legalspan_lora.TokenizerAdapter(tokenizer_path)
    if tokenizer.raw.get_vocab_size() != cfg.vocab_size:
        raise ValueError('C1 tokenizer vocabulary mismatch')
    model = adapt_model(model).to(device)
    metadata = {
        'method': 'C1 Houlsby-style bottleneck adapters with recurrent readout tuning',
        'phase1_checkpoint_sha256': PARENT_SHA256,
        'tokenizer_sha256': TOKENIZER_SHA256,
        'source_model_sha256': SOURCE_C_SHA256,
        'phase1_source_checkpoint_sha256': record['transfer_provenance']['phase1_source_checkpoint_sha256'],
        'model_config': record['metadata']['model_config'],
        'adaptation': {'type': 'Houlsby bottleneck', 'bottleneck': 64,
                       'positions': 'every encoder block and recurrent fusion output',
                       'additional_tuned_modules': ['initial_projection', 'gru', 'cross_attn']},
        'trainable_parameters': sum(p.numel() for p in model.parameters() if p.requires_grad),
        'total_parameters': sum(p.numel() for p in model.parameters()),
    }
    return model, tokenizer, metadata


def training_loss(model, source_ids, source_mask, target_ids, target_mask):
    if source_ids.ndim != 2 or source_ids.shape != source_mask.shape:
        raise ValueError('C1 source must be [batch,prompt] with matching mask')
    if target_ids.ndim != 2 or target_ids.shape != target_mask.shape or target_ids.shape[1] < 2:
        raise ValueError('C1 targets require BOS and summary tokens')
    labels = target_ids[:, 1:].masked_fill(~target_mask[:, 1:].bool(), -100)
    _, loss, _ = model(source_ids, target_ids[:, :-1], targets=labels,
                       prompt_mask=source_mask.bool())
    return loss


@torch.inference_mode()
def generate(model, source_ids, source_mask, max_new_tokens, eos_id):
    if max_new_tokens + 1 > model.cfg.max_seq_len:
        raise ValueError('C1 generation exceeds decoder context')
    was_training = model.training
    model.eval()
    try:
        current = torch.full((source_ids.size(0), 1), 1, dtype=torch.long, device=source_ids.device)
        cache = None
        output = [[] for _ in range(source_ids.size(0))]
        finished = [False] * source_ids.size(0)
        for _ in range(max_new_tokens):
            logits, _, cache = model(source_ids if cache is None else None, current,
                                     prompt_mask=source_mask.bool() if cache is None else None,
                                     past_kvs=cache, use_cache=True)
            current = logits[:, -1].argmax(dim=-1, keepdim=True)
            for index, token in enumerate(current[:, 0].tolist()):
                if not finished[index]:
                    output[index].append(int(token))
                    finished[index] = token == eos_id
            if all(finished):
                break
        return output
    finally:
        model.train(was_training)
