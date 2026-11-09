"""Phase 6 B1: independent passage encoding, fused memory, LoRMA+ adaptation.

All pretrained parameters come from the selected Phase 1 own-BPE checkpoint.
LoRMA+ here applies (I + alpha/rank * B A) to a frozen projection's output;
this is a multiplicative weight adaptation, not additive input-side LoRA.
"""
from __future__ import annotations

import hashlib
import importlib
import json
import math
import sys
import unicodedata
import zipfile
from dataclasses import asdict
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

import legalspan_lora as unicode13_compat

PARENT_SHA256 = '111296fb36a23dac1a11c07ede927cfc89dc86d51253e441d59afb82bd6924eb'
TOKENIZER_SHA256 = '5c0bd30290e0bd6919253b1ff864f52053924808aabd33482df92cc4bd471c8e'
# Required import closure, hashed directly from the original phase1_source.zip.
SOURCE_FILES_SHA256 = {
    'cs781/__init__.py': '4a3613ae45d376f074874f3924962127b635a12fc6a03c5cf2c2d2e9cc3c1c49',
    'cs781/contracts.py': 'b4d17bf29617b79bff3c3507cf01245dbe16ad550c2c394a15aaddedd0733689',
    'cs781/models/__init__.py': '0da8971507f5266ea3b85759054329394979d08d7f8e4896472fd471f78d3d49',
    'cs781/models/a.py': 'b5eb47496093388acf0798fa44fe2155116bd30a3fd1a9fbec3f6a71b5dca268',
    'cs781/models/b.py': '9919cc6a3ac87fb080acf6ac2be2df84bfb0d1d6d314e53fcf6c977bde14c057',
    'cs781/own_bpe.py': '2cac66fc09ee09136174f3837676a7578c71fa7d59a23bada83a18203029e07e',
}


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(2**20), b''):
            digest.update(block)
    return digest.hexdigest()


def _find(root, name):
    root = Path(root)
    direct = root / name
    if direct.is_file():
        return direct
    matches = sorted(root.rglob(name))
    if not matches:
        raise FileNotFoundError(f'{name} missing under {root}')
    if len(matches) > 1 and len({_sha256(p) for p in matches}) != 1:
        raise ValueError(f'Ambiguous {name}: different files under {root}')
    return matches[0]


def _verify_source(path):
    path = Path(path)
    is_archive = path.is_file()
    actual = {}
    if is_archive:
        with zipfile.ZipFile(path) as archive:
            for name in SOURCE_FILES_SHA256:
                actual[name] = hashlib.sha256(archive.read(name)).hexdigest()
    else:
        actual = {name: _sha256(path / name) for name in SOURCE_FILES_SHA256}
    for name, expected in SOURCE_FILES_SHA256.items():
        if actual[name] != expected:
            raise ValueError(f'Original B1 source file hash mismatch: {name}')
    # Independent of archive expansion, timestamps and filesystem ordering.
    manifest = json.dumps(actual, sort_keys=True, separators=(',', ':')).encode('utf-8')
    return {
        'phase1_source_format': 'zip' if is_archive else 'directory',
        'phase1_source_sha256': hashlib.sha256(manifest).hexdigest(),
        'phase1_source_hash_kind': 'sha256 of canonical JSON required-file SHA256 manifest',
        'phase1_source_files_sha256': actual,
        'phase1_source_archive_sha256': _sha256(path) if is_archive else None,
    }


def _discover_source(input_root):
    root = Path(input_root)
    # The model bundle includes other Phase 1 families with their own cs781
    # package trees. Kaggle expands the named B1 archive to phase1_source/.
    candidates = sorted(root.rglob('phase1_source.zip'))
    candidates.extend(sorted(p for p in root.rglob('phase1_source')
                             if p.is_dir() and (p / 'cs781/models/b.py').is_file()))
    if not candidates:
        # Older bundles may use another folder name. Only a complete B1
        # import closure is a candidate; unrelated vendor trees are ignored.
        candidates = sorted({p.parent.parent.parent for p in root.rglob('cs781/models/b.py')
                             if all((p.parent.parent.parent / name).is_file()
                                    for name in SOURCE_FILES_SHA256)})
    if not candidates:
        here = Path(__file__).resolve().parent
        candidates = [p for folder in (here, here.parent)
                      for p in (folder / 'phase1_source.zip', folder / 'phase1_source')
                      if p.exists()]
    if not candidates:
        raise FileNotFoundError('Original B1 source ZIP or expanded cs781 source is required')
    # Verify every supplied candidate instead of silently accepting conflicting
    # copies. Each required module must equal the archived original bytes.
    verified = [(path, _verify_source(path)) for path in candidates]
    return verified[0]


def _activate_source(source_path):
    prefix = str(Path(source_path).resolve()).replace('\\', '/').casefold() + '/'
    package = sys.modules.get('cs781')
    loaded_from = str(getattr(package, '__file__', '')).replace('\\', '/').casefold()
    if package is not None and not loaded_from.startswith(prefix):
        # Import from the verified expanded source even if an earlier local
        # build imported the ZIP (or another source tree) in this interpreter.
        for name in list(sys.modules):
            if name == 'cs781' or name.startswith('cs781.'):
                del sys.modules[name]
    sys.path.insert(0, str(source_path))
    importlib.invalidate_caches()


class TokenizerAdapter:
    """Original own-BPE merges with the archived Unicode13 piece boundaries."""
    pad_id, bos_id, eos_id = 0, 1, 2

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def encode(self, text):
        # Keep the original own-BPE merge implementation; only supply its
        # saved segmentation boundaries rather than the runtime Unicode DB.
        return [token for piece in unicode13_compat._pieces(text)
                for token in self.tokenizer._encode_piece(piece.encode('utf-8'))]

    def decode(self, ids):
        return self.tokenizer.decode([int(i) for i in ids])


class LoRMAOutputLinear(nn.Module):
    """Frozen linear y=W x followed by identity-initialized y+s B A y."""

    def __init__(self, base, rank=8, alpha=16):
        super().__init__()
        if not isinstance(base, nn.Linear) or rank < 1:
            raise ValueError('LoRMA requires a linear projection and positive rank')
        self.base = base
        self.base.requires_grad_(False)
        self.rank, self.scale = rank, alpha / rank
        self.a = nn.Parameter(base.weight.new_empty(rank, base.out_features))
        self.b = nn.Parameter(base.weight.new_zeros(base.out_features, rank))
        nn.init.kaiming_uniform_(self.a, a=math.sqrt(5))

    def forward(self, x):
        y = self.base(x)
        return y + F.linear(F.linear(y, self.a), self.b) * self.scale


class B1PassageFusion(nn.Module):
    """Use the actual B1 encoder and decoder, sharing encoder across passages."""

    def __init__(self, backbone, rank=8, alpha=16, max_passages=32, bos_id=1):
        super().__init__()
        self.backbone = backbone
        self.cfg = backbone.cfg
        self.bos_id = bos_id
        self.rank, self.alpha = rank, alpha
        backbone.requires_grad_(False)
        self.adapter_targets = []
        # Leave embeddings, norms and tied vocabulary head frozen.
        for name, module in list(backbone.named_modules()):
            if not isinstance(module, nn.Linear) or name == 'lm_head':
                continue
            if name.rsplit('.', 1)[-1] not in {'wqkv', 'wq', 'wk', 'wv', 'wo', 'w1', 'w2', 'w3'}:
                continue
            parent_name, child_name = name.rsplit('.', 1)
            parent = backbone.get_submodule(parent_name)
            setattr(parent, child_name, LoRMAOutputLinear(module, rank, alpha))
            self.adapter_targets.append(name)
        # Position 0 for every passage retains the parent behavior at startup.
        self.passage_positions = nn.Parameter(torch.zeros(max_passages, self.cfg.d_model))

    def encode_passages(self, source_ids, source_mask):
        if source_ids.ndim != 3 or source_ids.shape != source_mask.shape or source_mask.dtype != torch.bool:
            raise ValueError('B1 expects IDs and boolean mask of shape [batch,passages,length]')
        batch, passages, width = source_ids.shape
        if min(batch, passages, width) < 1 or passages > self.passage_positions.shape[0]:
            raise ValueError('Invalid passage shape or passage count exceeds configured positions')
        flat_mask = source_mask.reshape(batch * passages, width)
        nonempty = flat_mask.any(dim=1)
        if not source_mask.reshape(batch, -1).any(dim=1).all():
            raise ValueError('Every document must have at least one valid source token')
        # Original B1 rejects entirely padded prompts. Omit empty passages from
        # encoding and scatter zero memories back, keeping their mask all false.
        indices = nonempty.nonzero(as_tuple=False).flatten()
        encoded, _ = self.backbone.encode(
            source_ids.reshape(batch * passages, width).index_select(0, indices),
            flat_mask.index_select(0, indices))
        memory = encoded.new_zeros(batch * passages, width, self.cfg.d_model)
        memory = memory.index_copy(0, indices, encoded).reshape(batch, passages, width, -1)
        memory = memory + self.passage_positions[:passages][None, :, None, :].to(memory.dtype)
        memory = memory.masked_fill(~source_mask.unsqueeze(-1), 0)
        return memory.reshape(batch, passages * width, -1), source_mask.reshape(batch, passages * width)

    def forward(self, source_ids, source_mask, decoder_ids, past_kvs=None, use_cache=False):
        if decoder_ids.ndim != 2 or decoder_ids.shape[1] < 1:
            raise ValueError('Decoder IDs must be nonempty [batch,length]')
        if past_kvs is None:
            memory, prompt_mask = self.encode_passages(source_ids, source_mask)
            self_pasts = cross_pasts = [None] * len(self.backbone.decoder)
            offset = 0
        else:
            if not use_cache or source_ids is not None or source_mask is not None:
                raise ValueError('Cached decoding requires use_cache and no repeated source')
            if set(past_kvs) != {'self', 'cross', 'prompt_mask'}:
                raise ValueError('Malformed decoder cache')
            self_pasts, cross_pasts = past_kvs['self'], past_kvs['cross']
            prompt_mask, memory = past_kvs['prompt_mask'], None
            if len(self_pasts) != len(self.backbone.decoder) or len(cross_pasts) != len(self.backbone.decoder):
                raise ValueError('Cache layer count mismatch')
            if decoder_ids.shape[1] != 1:
                raise ValueError('Cached decoding requires exactly one new token')
            offset = self_pasts[0][0].shape[-2]
        if decoder_ids.shape[0] != prompt_mask.shape[0] or offset + decoder_ids.shape[1] > self.cfg.max_seq_len:
            raise ValueError('Decoder batch mismatch or context exceeded')
        x = self.backbone.tok_embeddings(decoder_ids)
        new_self, new_cross = [], []
        for index, block in enumerate(self.backbone.decoder):
            if self.training and self.cfg.gradient_checkpointing and not use_cache:
                x = checkpoint(lambda inp, mem, blk=block: blk(inp, mem, prompt_mask)[0],
                               x, memory, use_reentrant=False)
            else:
                x, caches = block(x, memory, prompt_mask, self_pasts[index], cross_pasts[index], use_cache)
                if use_cache:
                    new_self.append(caches[0])
                    new_cross.append(caches[1])
        logits = self.backbone.lm_head(self.backbone.decoder_norm(x))
        cache = {'self': new_self, 'cross': new_cross, 'prompt_mask': prompt_mask} if use_cache else None
        return logits, cache


def build(input_dir, device):
    root = Path(input_dir)
    parent_path = _find(root, 'b1_latest_model.pt')
    tokenizer_path = _find(root, 'tokenizer.json')
    source_path, source_metadata = _discover_source(root)
    parent_sha, tokenizer_sha = _sha256(parent_path), _sha256(tokenizer_path)
    if parent_sha != PARENT_SHA256 or tokenizer_sha != TOKENIZER_SHA256:
        raise ValueError('Selected Phase 1 B1 checkpoint/tokenizer provenance mismatch')
    _activate_source(source_path)
    from cs781.models.b import EncoderDecoderConfig, EncoderDecoderLegalModel
    from cs781.own_bpe import OwnBPE, POLICY
    record = torch.load(parent_path, map_location='cpu', weights_only=False)
    if record['metadata']['variant'] != 'B_taskaligned':
        raise ValueError('Expected selected Phase 1 B1 own-BPE generator')
    cfg = EncoderDecoderConfig(**record['metadata']['model_config'])
    backbone = EncoderDecoderLegalModel(cfg)
    backbone.load_state_dict(record['model'], strict=True)
    tokenizer_record = json.loads(tokenizer_path.read_text(encoding='utf-8'))
    if (tokenizer_record.get('format') != 'cs781-own-byte-bpe-v1'
            or tokenizer_record.get('policy') != POLICY
            or tokenizer_record.get('unicode_version') != '13.0.0'):
        raise ValueError('Unexpected own-BPE tokenizer format/segmentation')
    # Replay exact own-BPE merges with the original Unicode13 L/M/Nd/whitespace
    # boundaries even when Kaggle supplies a newer Unicode database.
    tokenizer = TokenizerAdapter(OwnBPE(tokenizer_record['merges']))
    if tokenizer.tokenizer.get_vocab_size() != cfg.vocab_size or cfg.pad_token_id != tokenizer.pad_id:
        raise ValueError('Checkpoint and own-BPE vocabulary mismatch')
    model = B1PassageFusion(backbone).to(device)
    metadata = {
        'method': 'B1 independent passage encoder fusion with LoRMA+',
        'parent_checkpoint_sha256': parent_sha,
        'phase1_checkpoint_sha256': parent_sha,
        'tokenizer_sha256': tokenizer_sha,
        **source_metadata,
        'parent_variant': record['metadata']['variant'],
        'model_config': asdict(cfg),
        'adaptation': {'type': 'LoRMA+', 'formula': '(I + alpha/rank * B @ A) @ W',
            'rank': model.rank, 'alpha': model.alpha, 'initialization': 'B=zeros, identity transform',
            'targets': model.adapter_targets, 'max_passages': len(model.passage_positions),
            'additional_parameters': 'zero-initialized passage position embeddings'},
        'trainable_parameters': sum(p.numel() for p in model.parameters() if p.requires_grad),
        'total_parameters': sum(p.numel() for p in model.parameters()),
        'tokenizer_unicode_version': tokenizer_record.get('unicode_version'),
        'runtime_unicode_version': unicodedata.unidata_version,
        'tokenizer_segmentation': {
            'unicode_version': '13.0.0',
            'policy': POLICY,
            'compatibility_helper': Path(unicode13_compat.__file__).name,
            'compatibility_helper_sha256': _sha256(unicode13_compat.__file__),
            'unicode_data_sha256': 'bdbffbbfc8ad4d3a6d01b5891510458f3d36f7170422af4ea2bed3211a73e8bb',
        },
    }
    metadata['frozen_parameters'] = metadata['total_parameters'] - metadata['trainable_parameters']
    return model, tokenizer, metadata


def training_loss(model, source_ids, source_mask, target_ids, target_mask):
    if target_ids.ndim != 2 or target_ids.shape != target_mask.shape or target_ids.shape[1] < 2:
        raise ValueError('Targets must be matching [batch,BOS+tokens+EOS] IDs/mask')
    labels = target_ids[:, 1:].masked_fill(~target_mask[:, 1:].bool(), -100)
    if not labels.ne(-100).any():
        raise ValueError('Target batch contains no supervised tokens')
    logits, _ = model(source_ids, source_mask.bool(), target_ids[:, :-1])
    return F.cross_entropy(logits.float().reshape(-1, logits.shape[-1]), labels.reshape(-1), ignore_index=-100)


@torch.inference_mode()
def generate(model, source_ids, source_mask, max_new_tokens, eos_id):
    if max_new_tokens < 0 or max_new_tokens > model.cfg.max_seq_len:
        raise ValueError('Generation length outside decoder context')
    was_training = model.training
    model.eval()
    try:
        batch = source_ids.shape[0]
        token = torch.full((batch, 1), model.bos_id, device=source_ids.device, dtype=torch.long)
        finished = torch.zeros(batch, device=source_ids.device, dtype=torch.bool)
        rows, cache = [[] for _ in range(batch)], None
        for _ in range(max_new_tokens):
            logits, cache = model(source_ids if cache is None else None,
                source_mask.bool() if cache is None else None, token, past_kvs=cache, use_cache=True)
            chosen = logits[:, -1].argmax(dim=-1)
            for index, value in enumerate(chosen.tolist()):
                if not finished[index].item():
                    rows[index].append(value)
            finished |= chosen.eq(eos_id)
            if finished.all().item():
                break
            token = chosen.masked_fill(finished, eos_id).unsqueeze(1)
        return rows
    finally:
        model.train(was_training)
