"""Own LegalSpan NER warm start, learned document context and constrained CRF."""
import json
import sys
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'vendor'))
sys.path.insert(0, str(ROOT / 'vendor/backbones'))
from lner.models import build_model
from lner.data import sha256
from legal_d.ner import LinearCRF


class BoostNER(nn.Module):
    def __init__(self, previous):
        super().__init__()
        self.previous = previous
        self.backbone = previous.backbone
        # Avoid registering the same backbone twice in optimizer parameter groups.
        del self.previous._modules['backbone']
        self.previous.__dict__['backbone'] = self.backbone
        self.header_gru = nn.GRU(32, 64, batch_first=True, bidirectional=True)
        self.header_query = nn.Linear(256, 128)
        self.header_attention = nn.MultiheadAttention(128, 4, batch_first=True)
        self.header_output = nn.Linear(128, 256)
        nn.init.zeros_(self.header_output.weight)
        nn.init.zeros_(self.header_output.bias)
        self.boundaries = nn.Linear(256, 2)
        self.crf = LinearCRF(25)

    def emissions(self, ids, mapping, chars, roles, header_chars):
        hidden = self.previous.representations(ids, roles)
        values = hidden.gather(1, mapping[:, :, None].expand(-1, -1, hidden.shape[-1]))
        values = torch.cat((values, self.previous.char_embedding(chars)), -1)
        with torch.autocast(device_type=ids.device.type, enabled=False):
            values, _ = self.previous.char_context(values.float())
            header, _ = self.header_gru(self.previous.char_embedding(header_chars).float())
        context, _ = self.header_attention(self.header_query(values), header, header, need_weights=False)
        values = values + self.header_output(context)
        values = self.previous.dropout(values)
        return self.previous.classifier(values), self.boundaries(values)

    def forward(self, ids, mapping, chars, roles, header_chars):
        return self.emissions(ids, mapping, chars, roles, header_chars)[0]

    def loss(self, args, tags, class_weights, boundary_weight=.15):
        emissions, boundaries = self.emissions(*args)
        emissions, boundaries = emissions[0].float(), boundaries[0].float()
        ce = F.cross_entropy(emissions, tags, weight=class_weights, label_smoothing=.01)
        sequence = self.crf.nll(emissions, tags)
        starts = (tags > 0) & (tags % 2 == 1)
        after = torch.cat((tags[1:], torch.zeros(1, dtype=tags.dtype, device=tags.device)))
        ends = (tags > 0) & (((tags+1)//2 != (after+1)//2) | ((after > 0) & (after % 2 == 1)))
        truth = torch.stack((starts, ends), 1).float()
        boundary = F.binary_cross_entropy_with_logits(boundaries, truth, pos_weight=torch.full((2,), 8., device=tags.device))
        return .5*ce + .5*sequence + boundary_weight*boundary


def load_warm_start(config, checkpoint, char_vocab, candidate):
    spec = dict(config['warm_start_spec'], method='full')
    previous, evidence = build_model(spec, char_vocab, 25)
    old = torch.load(checkpoint, map_location='cpu', weights_only=False)
    if sha256(checkpoint) != config['warm_start_sha256']:
        raise ValueError('Warm-start checkpoint SHA differs')
    if old['evidence']['tokenizer_sha256'] != sha256(config['tokenizer']):
        raise ValueError('Warm-start tokenizer differs')
    state = dict(old['model'])
    maximum_error = 0.
    for key in list(state):
        if not key.endswith('.base.weight'):
            continue
        prefix = key[:-len('.base.weight')]
        weight = state.pop(key).float()
        down = state.pop(prefix + '.down.weight').float()
        up = state.pop(prefix + '.up.weight').float()
        scale = config['warm_start_spec']['alpha'] / config['warm_start_spec']['rank']
        merged = weight + scale * (weight @ up) @ down
        probe = torch.linspace(-.2, .2, weight.shape[1])[None]
        expected = F.linear(probe + scale*F.linear(F.linear(probe, down), up), weight)
        actual = F.linear(probe, merged)
        maximum_error = max(maximum_error, float((actual-expected).abs().max()))
        state[prefix + '.weight'] = merged
    if maximum_error > 1e-4:
        raise ValueError('LoRMA merge parity differs')
    previous.load_state_dict(state, strict=True)
    model = BoostNER(previous)
    evidence.update({'warm_start_ner_sha256': sha256(checkpoint), 'warm_start_selected_epoch': old['epoch'],
                     'adapter_merge_max_absolute_error': maximum_error, 'method': candidate['method'],
                     'header_characters': config['header_chars'], 'parameters': sum(p.numel() for p in model.parameters()),
                     'trainable_parameters': sum(p.numel() for p in model.parameters() if p.requires_grad)})
    return model, evidence
