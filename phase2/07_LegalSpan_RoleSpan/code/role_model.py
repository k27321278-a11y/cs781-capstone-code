"""Train coarse spans, classify pooled legal roles, then decode fine BIO labels."""
import math
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from boost_model import load_warm_start
from legal_d.ner import LinearCRF
from lner.data import decode_bio

GENERIC = ['ROLE_ENTITY', 'AUTH', 'COURT', 'STAT', 'PREC', 'DATE', 'CASENO']
FINE_TO_COARSE = [0] + [v for _ in range(6) for v in (1, 2)] + list(range(3, 15))


class RoleSpanNER(nn.Module):
    def __init__(self, context):
        super().__init__()
        self.context = context
        self.backbone = context.backbone
        del context._modules['backbone']
        context.__dict__['backbone'] = self.backbone
        self.coarse_classifier = nn.Linear(256, 15)
        self.role_classifier = nn.Linear(256*3 + 128 + 16, 6)
        self.length_embedding = nn.Embedding(16, 16)
        self.generic_crf = LinearCRF(15)
        self.crf = LinearCRF(25)
        old = context.previous.classifier
        with torch.no_grad():
            for coarse in range(15):
                indices = [fine for fine, value in enumerate(FINE_TO_COARSE) if value == coarse]
                self.coarse_classifier.weight[coarse].copy_(old.weight[indices].mean(0))
                self.coarse_classifier.bias[coarse].copy_(old.bias[indices].mean())
            self.role_classifier.weight.zero_()
            for role in range(6):
                pair = [1+2*role, 2+2*role]
                self.role_classifier.weight[role, :256].copy_(old.weight[pair].mean(0))
                self.role_classifier.bias[role].copy_(old.bias[pair].mean())
        old.requires_grad_(False)

    def features(self, args):
        ids, mapping, chars, roles, header_chars = args
        previous = self.context.previous
        hidden = previous.representations(ids, roles)
        values = hidden.gather(1, mapping[:, :, None].expand(-1, -1, hidden.shape[-1]))
        values = torch.cat((values, previous.char_embedding(chars)), -1)
        with torch.autocast(device_type=ids.device.type, enabled=False):
            values, _ = previous.char_context(values.float())
            header, _ = self.context.header_gru(previous.char_embedding(header_chars).float())
        attended, _ = self.context.header_attention(self.context.header_query(values), header, header, need_weights=False)
        values = previous.dropout(values + self.context.header_output(attended))
        return values[0], header[0].mean(0)

    def classify_span(self, values, header, start, end):
        empty = values.new_zeros(256)
        left = values[max(0, start-32):start].mean(0) if start else empty
        right = values[end:min(len(values), end+32)].mean(0) if end < len(values) else empty
        bucket = min(15, int(math.log2(max(1, end-start))))
        length = self.length_embedding.weight[bucket]
        feature = torch.cat((values[start:end].mean(0), left, right, header.to(values.dtype), length.to(values.dtype)))
        return self.role_classifier(feature)

    def fine_emissions(self, coarse, values, header, spans):
        # Non-role types retain their own B/I logits. All six role types share
        # a span detector; role posterior is constant across each proposed span.
        fine = torch.stack([coarse[:, c] for c in FINE_TO_COARSE], -1)
        fine[:, 1:13] = fine[:, 1:13] - math.log(6)
        for start, end, *_ in spans:
            posterior = self.classify_span(values, header, start, end).float().log_softmax(-1)
            for role in range(6):
                fine[start:end, 1+2*role] = coarse[start:end, 1] + posterior[role]
                fine[start:end, 2+2*role] = coarse[start:end, 2] + posterior[role]
        return fine

    @torch.no_grad()
    def propose(self, coarse):
        transition, start = self.generic_crf.scores()
        transition = transition.float().cpu().numpy()
        score = start.float().cpu().numpy() + coarse[0].float().cpu().numpy()
        values = coarse.float().cpu().numpy()
        back = np.zeros(values.shape, dtype=np.uint8)
        for i in range(1, len(values)):
            possible = score[:, None] + transition
            back[i] = possible.argmax(0)
            score = possible.max(0) + values[i]
        tag = int((score + self.generic_crf.end.float().cpu().numpy()).argmax())
        tags = [tag]
        for i in range(len(values)-1, 0, -1):
            tag = int(back[i, tag]); tags.append(tag)
        return [s for s in decode_bio(tags[::-1], GENERIC) if s[2] == 'ROLE_ENTITY']

    def forward(self, *args):
        values, header = self.features(args)
        coarse = self.coarse_classifier(values).float()
        return self.fine_emissions(coarse, values, header, self.propose(coarse))[None]

    def loss(self, args, tags, class_weights, boundary_weight=.15):
        values, header = self.features(args)
        coarse = self.coarse_classifier(values).float()
        mapping = torch.tensor(FINE_TO_COARSE, device=tags.device)
        generic_tags = mapping[tags]
        generic_weights = torch.stack([class_weights[[i for i, c in enumerate(FINE_TO_COARSE) if c == value]].mean() for value in range(15)])
        spans = []
        start = role = None
        for position, tag in enumerate(tags.detach().cpu().tolist() + [0]):
            current = (tag-1)//2 if 0 < tag <= 12 else None
            if start is not None and (current != role or tag % 2 == 1):
                spans.append((start, position, role)); start = role = None
            if current is not None and start is None:
                start, role = position, current
        fine = self.fine_emissions(coarse, values, header, spans)
        if spans:
            role_logits = torch.stack([self.classify_span(values, header, a, b) for a, b, _ in spans]).float()
            targets = torch.tensor([r for _, _, r in spans], device=tags.device)
            role_loss = F.cross_entropy(role_logits, targets, weight=class_weights[1:13:2], label_smoothing=.02)
        else:
            role_loss = values.sum()*0
        starts = (tags > 0) & (tags % 2 == 1)
        after = torch.cat((tags[1:], tags.new_zeros(1)))
        ends = (tags > 0) & (((tags+1)//2 != (after+1)//2) | ((after > 0) & (after % 2 == 1)))
        boundary = F.binary_cross_entropy_with_logits(self.context.boundaries(values).float(), torch.stack((starts, ends), 1).float(), pos_weight=values.new_full((2,), 8.))
        return (.4*F.cross_entropy(coarse, generic_tags, weight=generic_weights, label_smoothing=.01)
                + .25*self.generic_crf.nll(coarse, generic_tags) + .75*role_loss
                + .2*F.cross_entropy(fine, tags, weight=class_weights, label_smoothing=.01)
                + .1*self.crf.nll(fine, tags) + boundary_weight*boundary)


def load_role_start(config, checkpoint, char_vocab, candidate):
    context, evidence = load_warm_start(config, checkpoint, char_vocab, candidate)
    model = RoleSpanNER(context)
    evidence.update({'method': 'coarse_span_then_contextual_role', 'coarse_bio_classes': 15,
                     'legal_role_classes': 6, 'parameters': sum(p.numel() for p in model.parameters()),
                     'trainable_parameters': sum(p.numel() for p in model.parameters() if p.requires_grad)})
    return model, evidence
