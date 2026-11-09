"""D222 grouped-query decoder with learned layerwise attention K/V prefixes."""
import hashlib
import importlib
import json
import sys
import zipfile
from pathlib import Path

import torch
from torch import nn

import b1_fusion

PARENT_SHA256 = '0d0e3d47b2e4254d3662f8f5c03e7436207bd822cb9298218f0d7a02558e2c9b'
TOKENIZER_SHA256 = b1_fusion.TOKENIZER_SHA256
SOURCE_D_SHA256 = '35c3ebc178de3a5797884542bb739d225a3ea7fd32af9774b759187c73c272fc'


class GlobalPrefixAttention(nn.Module):
    """Add virtual keys/values to a global GQA layer without shifting RoPE time."""
    def __init__(self, base, prefix_len=16):
        super().__init__()
        if not base.global_layer:
            raise ValueError('D222 prefix injection requires a global layer')
        self.base = base
        cfg = base.cfg
        self.prefix_len = prefix_len
        self.prefix_k = nn.Parameter(torch.empty(1, cfg.kv_heads, prefix_len, cfg.head_dim))
        self.prefix_v = nn.Parameter(torch.zeros(1, cfg.kv_heads, prefix_len, cfg.head_dim))
        nn.init.normal_(self.prefix_k, std=0.02)

    def forward(self, hidden, cache=None, use_cache=False):
        if cache is not None:
            return self.base(hidden, cache, use_cache)
        cfg = self.base.cfg
        batch, length, _ = hidden.shape
        if length > cfg.context:
            raise ValueError('D222 context exhausted')
        position = torch.arange(length, device=hidden.device)
        q = self.base.wq(hidden).view(batch, length, cfg.query_heads, cfg.head_dim).transpose(1, 2)
        k = self.base.wk(hidden).view(batch, length, cfg.kv_heads, cfg.head_dim).transpose(1, 2)
        v = self.base.wv(hidden).view(batch, length, cfg.kv_heads, cfg.head_dim).transpose(1, 2)
        native = importlib.import_module('legal_d.model')
        q = native.rotary(q, position, cfg.rope_theta)
        k = native.rotary(k, position, cfg.rope_theta)
        prefix_position = torch.arange(-self.prefix_len, 0, device=hidden.device)
        key_position = torch.cat((prefix_position, position))
        k = torch.cat((self.prefix_k.to(k.dtype).expand(batch, -1, -1, -1), k), dim=2)
        v = torch.cat((self.prefix_v.to(v.dtype).expand(batch, -1, -1, -1), v), dim=2)
        attended = self.base.attend(q, k, v, position, key_position)
        output = self.base.wo(attended.transpose(1, 2).contiguous().view(batch, length, cfg.width))
        saved = ({'k': k, 'v': v, 'positions': key_position,
                  'next_position': length} if use_cache else None)
        return output, saved


def adapt_model(model, prefix_len=16):
    model.requires_grad_(False)
    for block in model.blocks:
        if block.attn.global_layer:
            block.attn = GlobalPrefixAttention(block.attn, prefix_len)
    # Layer norms are a tiny learned readout calibration for the new prefixes.
    for name, parameter in model.named_parameters():
        if name.endswith('norm.weight'):
            parameter.requires_grad_(True)
    return model


def _activate_source(root):
    root = Path(root)
    expanded = sorted(root.rglob('d222_source/source/legal_d/model.py'))
    if expanded:
        source_sha = b1_fusion._sha256(expanded[0])
        source_root = expanded[0].parent.parent
        path = str(source_root)
    else:
        archives = sorted(root.rglob('d222_source.zip'))
        if len(archives) != 1:
            raise FileNotFoundError('D222 native Phase 1 source is required')
        with zipfile.ZipFile(archives[0]) as archive:
            source_sha = hashlib.sha256(archive.read('source/legal_d/model.py')).hexdigest()
        path = str(archives[0]) + '/source'
    if source_sha != SOURCE_D_SHA256:
        raise ValueError('D222 original source hash mismatch')
    for name in list(sys.modules):
        if name == 'legal_d' or name.startswith('legal_d.'):
            del sys.modules[name]
    sys.path.insert(0, path)
    importlib.invalidate_caches()
    return importlib.import_module('legal_d.model')


def build(input_dir, device):
    root = Path(input_dir)
    original = _activate_source(root)
    parent_path = b1_fusion._find(root, 'd222_phase1_model.pt')
    tokenizer_path = b1_fusion._find(root, 'd222_tokenizer.json')
    if b1_fusion._sha256(parent_path) != PARENT_SHA256 or b1_fusion._sha256(tokenizer_path) != TOKENIZER_SHA256:
        raise ValueError('D222 parent or tokenizer hash mismatch')
    source_path, _ = b1_fusion._discover_source(root)
    b1_fusion._activate_source(source_path)
    from cs781.own_bpe import OwnBPE, POLICY
    record = torch.load(parent_path, map_location='cpu', weights_only=True)
    cfg = original.DConfig()
    model = original.EfficientDecoder(cfg)
    model.load_state_dict(record['model'], strict=True)
    tokenizer_record = json.loads(tokenizer_path.read_text(encoding='utf-8'))
    if (tokenizer_record.get('format') != 'cs781-own-byte-bpe-v1'
            or tokenizer_record.get('policy') != POLICY
            or tokenizer_record.get('unicode_version') != '13.0.0'):
        raise ValueError('D222 own-BPE tokenizer policy mismatch')
    tokenizer = b1_fusion.TokenizerAdapter(OwnBPE(tokenizer_record['merges']))
    if tokenizer.tokenizer.get_vocab_size() != cfg.vocab_size:
        raise ValueError('D222 tokenizer vocabulary mismatch')
    model = adapt_model(model).to(device)
    metadata = {
        'method': 'D222 global grouped-query attention prefix tuning',
        'phase1_checkpoint_sha256': PARENT_SHA256,
        'tokenizer_sha256': TOKENIZER_SHA256,
        'source_model_sha256': SOURCE_D_SHA256,
        'phase1_source_checkpoint_sha256': record['transfer_provenance']['phase1_source_checkpoint_sha256'],
        'model_config': vars(cfg),
        'adaptation': {'type': 'prefix tuning', 'prefix_length': 16,
                       'targets': 'global attention layers only',
                       'additional_trainable': 'RMSNorm scales'},
        'trainable_parameters': sum(p.numel() for p in model.parameters() if p.requires_grad),
        'total_parameters': sum(p.numel() for p in model.parameters()),
    }
    return model, tokenizer, metadata


def _causal_batch(model, source_ids, source_mask, target_ids, target_mask):
    if source_ids.ndim != 2 or source_ids.shape != source_mask.shape or not source_mask.all():
        raise ValueError('D222 requires unpadded source IDs')
    if target_ids.ndim != 2 or target_ids.shape != target_mask.shape or target_ids.shape[1] < 2:
        raise ValueError('D222 target needs BOS and summary')
    combined = torch.cat((source_ids, target_ids[:, :-1]), dim=1)
    if combined.shape[1] > model.cfg.context:
        raise ValueError('D222 source plus target exceeds context')
    labels = torch.full_like(combined, -100)
    labels[:, source_ids.shape[1]:] = target_ids[:, 1:].masked_fill(~target_mask[:, 1:].bool(), -100)
    return combined, labels


def training_loss(model, source_ids, source_mask, target_ids, target_mask):
    combined, labels = _causal_batch(model, source_ids, source_mask, target_ids, target_mask)
    _, loss, _ = model(combined, targets=labels)
    return loss


@torch.inference_mode()
def generate(model, source_ids, source_mask, max_new_tokens, eos_id):
    if source_ids.size(0) != 1 or not source_mask.all():
        raise ValueError('D222 generation requires one unpadded document')
    prompt = torch.cat((source_ids, torch.ones((1, 1), dtype=torch.long, device=source_ids.device)), dim=1)
    if prompt.shape[1] + max_new_tokens > model.cfg.context:
        raise ValueError('D222 generation exceeds configured context')
    was_training = model.training
    model.eval()
    try:
        output = []
        cache = None
        current = prompt
        for _ in range(max_new_tokens):
            logits, _, cache = model(current, caches=cache, use_cache=True)
            token = int(logits[0, -1].argmax())
            output.append(token)
            if token == eos_id:
                break
            current = torch.tensor([[token]], device=source_ids.device)
        return [output]
    finally:
        model.train(was_training)
