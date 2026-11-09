"""Smaller A0 causal student for sequence-level knowledge distillation."""
import json
from pathlib import Path

import torch

import b1_fusion

PARENT_SHA256 = 'e4f2659533a019534cca0b629ad5106246788b45e3bede94c85af581fbc06873'
TOKENIZER_SHA256 = '9b32abd6b09b9c30fcd6cccac236d5815a8740f35d4b70993cd4390a18a72e0c'


class TokenizerAdapter:
    pad_id, bos_id, eos_id = 0, 1, 2

    def __init__(self, path):
        from tokenizers import Tokenizer
        self.raw = Tokenizer.from_file(str(path))

    def encode(self, text):
        return self.raw.encode(text, add_special_tokens=False).ids

    def decode(self, ids):
        return self.raw.decode([int(item) for item in ids], skip_special_tokens=True)


def build(input_dir, device):
    root = Path(input_dir)
    checkpoint = b1_fusion._find(root, 'a0_phase1_model.pt')
    tokenizer_path = b1_fusion._find(root, 'a0_tokenizer.json')
    if b1_fusion._sha256(checkpoint) != PARENT_SHA256 or b1_fusion._sha256(tokenizer_path) != TOKENIZER_SHA256:
        raise ValueError('A0 own Phase 1 checkpoint or tokenizer hash mismatch')
    source_path, source_meta = b1_fusion._discover_source(root)
    b1_fusion._activate_source(source_path)
    from cs781.contracts import ModelConfig
    from cs781.models.a import LegalGPT
    record = torch.load(checkpoint, map_location='cpu', weights_only=True)
    cfg = ModelConfig(**record['metadata']['model_config'])
    model = LegalGPT(cfg)
    model.load_state_dict(record['model'], strict=True)
    tokenizer = TokenizerAdapter(tokenizer_path)
    if tokenizer.raw.get_vocab_size() != cfg.vocab_size:
        raise ValueError('A0 tokenizer vocabulary mismatch')
    model.requires_grad_(False)
    for block in model.blocks[-4:]:
        block.requires_grad_(True)
    model.norm.requires_grad_(True)
    model = model.to(device)
    metadata = {
        'method': 'A0 sequence-level KD student with last-four-block tuning',
        'phase1_checkpoint_sha256': PARENT_SHA256,
        'tokenizer_sha256': TOKENIZER_SHA256,
        **source_meta,
        'phase1_source_checkpoint_sha256': record['transfer_provenance']['phase1_source_checkpoint_sha256'],
        'model_config': record['metadata']['model_config'],
        'adaptation': {'type': 'knowledge distillation',
                       'student_tuning': 'last four causal blocks and output norm',
                       'teacher': 'larger own Phase1 B1 with Phase6 LoRMA+ adapter',
                       'signal': 'teacher-generated summaries on fitting documents only'},
        'trainable_parameters': sum(p.numel() for p in model.parameters() if p.requires_grad),
        'total_parameters': sum(p.numel() for p in model.parameters()),
    }
    return model, tokenizer, metadata


def _causal_batch(model, source_ids, source_mask, target_ids, target_mask):
    if source_ids.ndim != 2 or source_ids.shape != source_mask.shape or not source_mask.all():
        raise ValueError('A0 requires unpadded source IDs')
    if target_ids.ndim != 2 or target_ids.shape != target_mask.shape or target_ids.shape[1] < 2:
        raise ValueError('A0 target needs BOS and summary')
    combined = torch.cat((source_ids, target_ids[:, :-1]), dim=1)
    if combined.shape[1] > model.cfg.max_seq_len:
        raise ValueError('A0 source and target exceed context')
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
        raise ValueError('A0 generation expects one unpadded document')
    prompt = torch.cat((source_ids, torch.ones((1, 1), dtype=torch.long, device=source_ids.device)), dim=1)
    if prompt.shape[1] + max_new_tokens > model.cfg.max_seq_len:
        raise ValueError('A0 generation exceeds context')
    was_training = model.training
    model.eval()
    try:
        current = prompt
        cache = None
        result = []
        for _ in range(max_new_tokens):
            logits, _, cache = model(current, past_kvs=cache, use_cache=True)
            token = int(logits[0, -1].argmax())
            result.append(token)
            if token == eos_id:
                break
            current = torch.tensor([[token]], device=source_ids.device)
        return [result]
    finally:
        model.train(was_training)
