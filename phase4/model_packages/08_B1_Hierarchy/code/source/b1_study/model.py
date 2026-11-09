import copy
from dataclasses import replace

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from phase3_rr.models import load_backbone
from phase4_cjpe.models import EvidenceCaseModel


class GatedEncoderBlock(nn.Module):
    def __init__(self, block):
        super().__init__()
        self.block = copy.deepcopy(block)
        self.attention_gate = nn.Parameter(torch.zeros(()))
        self.ffn_gate = nn.Parameter(torch.zeros(()))

    def forward(self, x, valid_mask):
        x = x + self.attention_gate * self.block.attn(self.block.attn_norm(x), valid_mask)
        return x + self.ffn_gate * self.block.ffn(self.block.ffn_norm(x))


def grow_encoder(backbone):
    if len(backbone.encoder) != 6:
        raise ValueError('Depth expansion requires the original six encoder blocks')
    layers, mapping = [], []
    for i, block in enumerate(backbone.encoder):
        layers.append(block)
        mapping.append({'new_index': len(layers) - 1, 'original_index': i, 'gated_copy': False})
        if i in (2, 5):
            layers.append(GatedEncoderBlock(block))
            mapping.append({'new_index': len(layers) - 1, 'original_index': i, 'gated_copy': True})
    backbone.encoder = nn.ModuleList(layers)
    backbone.cfg = replace(backbone.cfg, n_encoder_layers=8)
    return mapping


def expand_upper_ffn(backbone, new_inner=3072):
    names = []
    targets = [('encoder', i, b) for i, b in enumerate(backbone.encoder)][-2:]
    targets += [('decoder', i, b) for i, b in enumerate(backbone.decoder)][-2:]
    for family, index, block in targets:
        block = block.block if isinstance(block, GatedEncoderBlock) else block
        ffn = block.ffn
        old, width = ffn.w1.weight.shape
        if new_inner <= old:
            raise ValueError('FFN expansion must increase inner width')
        device, dtype = ffn.w1.weight.device, ffn.w1.weight.dtype
        first = nn.Linear(width, new_inner, bias=False, device=device, dtype=dtype)
        third = nn.Linear(width, new_inner, bias=False, device=device, dtype=dtype)
        second = nn.Linear(new_inner, width, bias=False, device=device, dtype=dtype)
        with torch.no_grad():
            first.weight[:old].copy_(ffn.w1.weight)
            third.weight[:old].copy_(ffn.w3.weight)
            second.weight.zero_()
            second.weight[:, :old].copy_(ffn.w2.weight)
        ffn.w1, ffn.w2, ffn.w3 = first, second, third
        names.append(f'{family}.{index}.ffn:{old}->{new_inner}')
    return names


class DocumentBlock(nn.Module):
    def __init__(self, width=256, heads=4, inner=768, dropout=.1):
        super().__init__()
        self.heads, self.dropout = heads, dropout
        self.attention_norm, self.ffn_norm = nn.LayerNorm(width), nn.LayerNorm(width)
        self.qkv = nn.Linear(width, 3 * width)
        self.output = nn.Linear(width, width)
        self.up, self.gate = nn.Linear(width, inner), nn.Linear(width, inner)
        self.down = nn.Linear(inner, width)

    def forward(self, x):
        b, length, width = x.shape
        q, k, v = [a.reshape(b, length, self.heads, width // self.heads).transpose(1, 2)
                   for a in self.qkv(self.attention_norm(x)).chunk(3, -1)]
        out = F.scaled_dot_product_attention(q, k, v, dropout_p=self.dropout if self.training else 0.)
        x = x + F.dropout(self.output(out.transpose(1, 2).reshape(b, length, width)), self.dropout, self.training)
        z = self.ffn_norm(x)
        return x + F.dropout(self.down(F.silu(self.gate(z)) * self.up(z)), self.dropout, self.training)


class B1Study(EvidenceCaseModel):
    def __init__(self, backbone, config):
        width = backbone.cfg.d_model
        super().__init__(backbone, 'b1', width)
        self.study_config = config
        self.mode = config['head_mode']
        self.chunk_batch = config['chunk_batch']
        self.depth_mapping = grow_encoder(backbone) if config.get('encoder_depth', 6) == 8 else []
        self.ffn_mapping = expand_upper_ffn(backbone) if config.get('expand_ffn', False) else []
        if self.mode != 'reference':
            for name in ('pool', 'head', 'position', 'chunk_context', 'context_projection',
                         'context_scale', 'context_norm', 'document_mix'):
                delattr(self, name)
            self.projection = nn.Linear(width, 256)
            self.document_position = nn.Linear(2, 256, bias=False)
            self.document_blocks = nn.ModuleList([DocumentBlock() for _ in range(2)]) if self.mode != 'gru_only' else nn.ModuleList()
            self.document_gru = nn.GRU(256, 128, bidirectional=True, batch_first=True) if self.mode != 'transformer_only' else None
            self.document_norm = nn.LayerNorm(256)
            self.document_attention = nn.Sequential(nn.Linear(256, 64), nn.Tanh(), nn.Linear(64, 1))
            self.classifier = nn.Sequential(nn.LayerNorm(256), nn.Dropout(.15), nn.Linear(256, 2))

    def feature_batch(self, ids, valid):
        m = self.backbone
        memory, valid = m.encode(ids, valid)
        x = m.tok_embeddings(ids)
        # Right padding cannot influence valid decoder positions through causal self-attention.
        for block in m.decoder:
            if self.training and m.cfg.gradient_checkpointing:
                x = checkpoint(lambda a, b, blk=block: blk(a, b, valid)[0], x, memory, use_reentrant=False)
            else:
                x = block(x, memory, valid)[0]
        values = self.token_norm(m.decoder_norm(x).float())
        content = valid.clone()
        content[:, 0] = False
        lengths = valid.sum(-1)
        content.scatter_(1, (lengths - 1).reshape(-1, 1), False)
        if not content.any(-1).all():
            raise ValueError('Each chunk needs a content token')
        attention = self.token_score(values).squeeze(-1).float().masked_fill(~content, -torch.inf).softmax(-1)
        mean = (values * content[:, :, None]).sum(1) / content.sum(1)[:, None]
        last = values[torch.arange(len(ids), device=ids.device), lengths - 2]
        summaries = torch.stack((mean, (values * attention[:, :, None]).sum(1), last), 0)
        return (summaries * self.token_mix.softmax(0)[:, None, None]).sum(0)

    def forward(self, chunks, positions, device):
        if not chunks or len(chunks) != len(positions):
            raise ValueError('Chunk/position mismatch')
        # Buckets contain chunks from this case, restored to document order before aggregation.
        order = sorted(range(len(chunks)), key=lambda i: len(chunks[i]))
        results = [None] * len(chunks)
        for offset in range(0, len(order), self.chunk_batch):
            selected = order[offset:offset + self.chunk_batch]
            length = max(len(chunks[i]) for i in selected)
            ids = torch.zeros((len(selected), length), dtype=torch.long, device=device)
            valid = torch.zeros_like(ids, dtype=torch.bool)
            for j, i in enumerate(selected):
                seq = torch.as_tensor(chunks[i], dtype=torch.long, device=device)
                ids[j, :len(seq)] = seq
                valid[j, :len(seq)] = True
            # Encoder and decoder blocks already checkpoint independently. A second
            # checkpoint around the entire feature path would repeat recomputation.
            features = self.feature_batch(ids, valid)
            for j, i in enumerate(selected):
                results[i] = features[j]
        h = torch.stack(results).float()
        pos = torch.as_tensor(positions, dtype=h.dtype, device=device).reshape(-1, 1)
        pos = torch.cat((pos, 1 - pos), -1)
        if self.mode == 'reference':
            h = h + self.position(pos)
            with torch.autocast(device_type=device.type, enabled=False):
                context, _ = self.chunk_context(h[None].float())
                h = self.context_norm(h + self.context_scale * self.context_projection(context[0]))
            attention = self.pool(torch.cat((h, pos), -1)).float().squeeze(-1).softmax(0)
            summaries = torch.stack(((h * attention[:, None]).sum(0), h.mean(0)), 0)
            document = (summaries * self.document_mix.softmax(0)[:, None]).sum(0, keepdim=True)
            return self.head(document)
        h = self.projection(h) + self.document_position(pos)
        for block in self.document_blocks:
            h = block(h[None])[0]
        if self.document_gru is not None:
            with torch.autocast(device_type=device.type, enabled=False):
                h = self.document_gru(h[None].float())[0][0]
        h = self.document_norm(h)
        weights = self.document_attention(h).float().squeeze(-1).softmax(0)
        return self.classifier((h * weights[:, None]).sum(0, keepdim=True))


def set_train_stage(model, stage):
    """All optimizer groups exist from start; freeze controls only gradient creation."""
    model.requires_grad_(True)
    model.backbone.requires_grad_(False)
    for block in model.backbone.encoder:
        if isinstance(block, GatedEncoderBlock):
            block.requires_grad_(True)
    if stage >= 1:
        encoder = [b for b in model.backbone.encoder if not isinstance(b, GatedEncoderBlock)]
        count = 2 if stage == 1 else 4 if stage == 2 else len(encoder)
        for block in encoder[-count:] + list(model.backbone.decoder)[-count:]:
            block.requires_grad_(True)
        model.backbone.encoder_norm.requires_grad_(True)
        model.backbone.decoder_norm.requires_grad_(True)
    return [name for name, p in model.named_parameters() if p.requires_grad]


def build_model(bundle, config):
    backbone, width, evidence = load_backbone(bundle, 'b1')
    model = B1Study(backbone, config)
    evidence.update(task_head=config['head_mode'], width=width, depth_mapping=model.depth_mapping,
                    ffn_mapping=model.ffn_mapping, original_model_config=evidence['model_config'],
                    expanded_model_config=vars(backbone.cfg), parameters=sum(p.numel() for p in model.parameters()))
    set_train_stage(model, 0)
    return model, evidence


def optimizer_groups(model, config):
    original, new = [], []
    for name, p in model.named_parameters():
        (original if name.startswith('backbone.') and '.block.' not in name
         and not name.endswith(('_gate',)) else new).append(p)
    return [dict(params=original, lr=config['backbone_lr'], initial_lr=config['backbone_lr']),
            dict(params=new, lr=config['head_lr'], initial_lr=config['head_lr'])]
