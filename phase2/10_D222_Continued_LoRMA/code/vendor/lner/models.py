"""NER adapters for our own saved models.

LoRA: Hu et al., 2021, https://arxiv.org/abs/2106.09685.
LoRMA+: Bihany, Patel, Modi, ACL 2025, identity rank inflation;
https://github.com/Exploration-Lab/LoRMA. For nn.Linear's transposed
weight storage this is W_effective = W @ (I + alpha/r * B @ A).
"""
import math
import sys
from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint
from .data import sha256

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'backbones'))


class AdaptedLinear(nn.Module):
    def __init__(self, base, method, rank=8, alpha=32, dropout=.05):
        super().__init__()
        if method not in ('lora','lorma') or rank < 1:
            raise ValueError('Invalid adaptation method/rank')
        self.base, self.method, self.scale = base,method,alpha/rank
        self.dropout = nn.Dropout(dropout)
        self.down = nn.Linear(base.in_features,rank,bias=False)
        self.up = nn.Linear(rank,base.in_features if method == 'lorma' else base.out_features,bias=False)
        self.base.requires_grad_(False)
        if method == 'lorma':
            nn.init.zeros_(self.down.weight)
            nn.init.kaiming_uniform_(self.up.weight,a=math.sqrt(5))
        else:
            nn.init.kaiming_uniform_(self.down.weight,a=math.sqrt(5))
            nn.init.zeros_(self.up.weight)

    def forward(self,x):
        update = self.up(self.down(self.dropout(x)))*self.scale
        return self.base(x+update) if self.method == 'lorma' else self.base(x)+update


def inject_adapters(model,method,rank,alpha,dropout):
    model.requires_grad_(False)
    selected=[]
    targets={'wqkv','wo','wq','wk','wv','w1','w2','w3','q','k','v','out','role_q'}
    def visit(module,prefix=''):
        for name,child in list(module.named_children()):
            path=prefix+name
            if isinstance(child,nn.Linear) and name in targets:
                setattr(module,name,AdaptedLinear(child,method,rank,alpha,dropout))
                selected.append(path)
            else:
                visit(child,path+'.')
    visit(model)
    if not selected: raise ValueError('No projection layers adapted')
    return selected


class CharacterNER(nn.Module):
    def __init__(self,backbone,family,char_vocab_size,classes):
        super().__init__()
        self.backbone,self.family=backbone,family
        self.char_embedding=nn.Embedding(char_vocab_size,32,padding_idx=0)
        self.char_context=nn.GRU(backbone.cfg.d_model+32,128,batch_first=True,bidirectional=True)
        self.dropout=nn.Dropout(.1)
        self.classifier=nn.Linear(256,classes)

    def representations(self,ids,roles):
        m=self.backbone
        if self.family in ('a0','legalspan'):
            x=m.tok_embeddings(ids)
            if self.family=='legalspan':
                from legalspan.model import continuity
                counts=continuity(ids,ids,0)
            for block in m.blocks:
                def run(inp,blk=block):
                    update,_=blk.attn(blk.attn_norm(inp))
                    hidden=inp+update
                    if hasattr(blk,'route'):
                        hidden,*_=blk.route(hidden,0,ids.shape[1],roles,ids>=4,counts)
                    return hidden+blk.ffn(blk.ffn_norm(hidden))
                x=checkpoint(run,x,use_reentrant=False) if self.training and m.cfg.gradient_checkpointing else run(x)
            return m.norm(x)
        memory,mask=m.encode(ids,torch.ones_like(ids,dtype=torch.bool))
        if self.family=='b1':
            x=m.tok_embeddings(ids)
            for block in m.decoder:
                def run(inp,blk=block):
                    return blk(inp,memory,mask)[0]
                x=checkpoint(run,x,use_reentrant=False) if self.training and m.cfg.gradient_checkpointing else run(x)
            return m.decoder_norm(x)
        if self.family=='c1':
            mean=memory.mean(1)
            initial=torch.tanh(m.initial_projection(mean)).float()
            hidden=initial[None].expand(m.cfg.n_gru_layers,-1,-1).contiguous()
            with torch.autocast(device_type=ids.device.type,enabled=False):
                recurrent,_=m.gru(m.tok_embeddings(ids).float(),hidden)
            update,_=m.cross_attn(m.cross_norm(recurrent),mask,memory=memory)
            x=recurrent+update
            return m.output_norm(x+m.fusion(m.fusion_norm(x)))
        raise ValueError('Unknown backbone family')

    def forward(self,ids,mapping,chars,roles):
        hidden=self.representations(ids,roles)
        indices=mapping[:,:,None].expand(-1,-1,hidden.shape[-1])
        h=torch.cat((hidden.gather(1,indices),self.char_embedding(chars)),dim=-1)
        with torch.autocast(device_type=ids.device.type,enabled=False):
            h,_=self.char_context(h.float())
        return self.classifier(self.dropout(h))


def build_model(spec,char_vocab_size,classes):
    from cs781.contracts import ModelConfig
    from cs781.models.a import LegalGPT
    from cs781.models.b import EncoderDecoderConfig,EncoderDecoderLegalModel
    from cs781.models.c import RecurrentConfig,RecurrentLegalModel
    from legalspan.model import SpanConfig,LegalSpanGPT
    payload=torch.load(spec['checkpoint'],map_location='cpu',weights_only=False)
    metadata=payload['metadata']
    if metadata.get('tokenizer_sha256') != sha256(spec['tokenizer']):
        raise ValueError('Pretrained checkpoint/tokenizer SHA-256 mismatch')
    config=dict(metadata['model_config'])
    family=spec['family']
    if family=='a0': model=LegalGPT(ModelConfig(**config))
    elif family=='legalspan': model=LegalSpanGPT(SpanConfig(**config))
    elif family=='b1': model=EncoderDecoderLegalModel(EncoderDecoderConfig(**config))
    elif family=='c1': model=RecurrentLegalModel(RecurrentConfig(**config))
    else: raise ValueError('Unknown checkpoint family')
    model.load_state_dict(payload['model'],strict=True)
    state=payload['state']; del payload
    adapted=[]
    if spec['method'] in ('lora','lorma'):
        adapted=inject_adapters(model,spec['method'],spec.get('rank',8),spec.get('alpha',32),.05)
    output=CharacterNER(model,family,char_vocab_size,classes)
    evidence={'pretrained_model_config':config,'pretrained_state':state,
              'pretrained_checkpoint_sha256':sha256(spec['checkpoint']),
              'tokenizer_sha256':metadata['tokenizer_sha256'],
              'adapted_layers':adapted,
              'parameters':sum(p.numel() for p in output.parameters()),
              'trainable_parameters':sum(p.numel() for p in output.parameters() if p.requires_grad)}
    return output,evidence


def tensor_row(row,encoding,device):
    record=row['encodings'][encoding]
    return [torch.tensor(x,dtype=torch.long,device=device)[None] for x in
            (record['ids'],record['mapping'],row['chars'],record['roles'])]
