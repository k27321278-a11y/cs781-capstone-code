"""Own-B1 hierarchy + source pointer, adapted from our verified passage fusion.

Original-token memory is preserved. Coverage is a training auxiliary objective;
four-gram blocking is a separate decoding heuristic. Neither ensures factuality.
"""
import hashlib
import math
import os
from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F
import b1_fusion as base


class EvidenceFusion(base.B1PassageFusion):
    def __init__(self,backbone,rank=8,alpha=16):
        super().__init__(backbone,rank,alpha)
        self.add_evidence_modules()

    def add_evidence_modules(self):
        d=self.cfg.d_model
        self.hierarchy=nn.GRU(d,d//2,batch_first=True,bidirectional=True)
        self.hierarchy_projection=nn.Linear(d,d,bias=False)
        self.hierarchy_gate=nn.Parameter(torch.zeros(()))
        self.pointer_q=nn.Linear(d,min(d,128),bias=False)
        self.pointer_k=nn.Linear(d,min(d,128),bias=False)
        self.pointer_gate=nn.Linear(d,1)
        nn.init.zeros_(self.pointer_gate.weight)
        nn.init.constant_(self.pointer_gate.bias,-4.)
        self.coverage_weight=.05
        self.backbone.decoder_norm.register_forward_hook(self._capture_hidden)

    def _capture_hidden(self,module,inputs,output):self._decoder_hidden=output

    def encode_passages(self,source_ids,source_mask):
        memory,mask=super().encode_passages(source_ids,source_mask)
        b,p,w=source_ids.shape;d=memory.shape[-1]
        tokens=memory.reshape(b,p,w,d)
        valid=source_mask.any(-1)
        pooled=tokens.sum(2)/source_mask.sum(2).clamp_min(1)[...,None]
        # Packing excludes trailing empty passages. Source packer generates only
        # trailing empty padding; reject holes instead of silently mixing them.
        lengths=valid.sum(-1)
        if not torch.equal(valid,torch.arange(p,device=valid.device)[None,:]<lengths[:,None]):
            raise ValueError('Only trailing empty passages supported')
        packed=nn.utils.rnn.pack_padded_sequence(pooled,lengths.cpu(),batch_first=True,enforce_sorted=False)
        contextual,_=self.hierarchy(packed)
        contextual,_=nn.utils.rnn.pad_packed_sequence(contextual,batch_first=True,total_length=p)
        enriched=tokens+torch.tanh(self.hierarchy_gate)*self.hierarchy_projection(contextual)[:,:,None,:]
        enriched=enriched.masked_fill(~source_mask[...,None],0).reshape(b,p*w,d)
        self._encoded_source=(enriched,source_ids.reshape(b,p*w),mask)
        return enriched,mask

    def features(self,source_ids,source_mask,decoder_ids,past_kvs=None,use_cache=False):
        native_cache=None if past_kvs is None else past_kvs['native']
        logits,native=super().forward(source_ids,source_mask,decoder_ids,native_cache,use_cache)
        if past_kvs is None:
            memory,ids,mask=self._encoded_source
            eligible=mask & ids.ge(4)
            keys=self.pointer_k(memory)
        else:
            keys,ids,eligible=past_kvs['keys'],past_kvs['ids'],past_kvs['eligible']
        hidden=self._decoder_hidden
        scores=torch.bmm(self.pointer_q(hidden).float(),keys.float().transpose(1,2))/math.sqrt(keys.shape[-1])
        has=eligible.any(-1)
        safe=eligible | ~has[:,None]
        attention=scores.masked_fill(~safe[:,None],-torch.inf).softmax(-1)*eligible[:,None]
        gate=self.pointer_gate(hidden).float()
        cache={'native':native,'keys':keys,'ids':ids,'eligible':eligible} if use_cache else None
        return logits,attention,gate,ids,has,cache

    def forward(self,source_ids,source_mask,decoder_ids,past_kvs=None,use_cache=False):
        logits,a,g,ids,has,cache=self.features(source_ids,source_mask,decoder_ids,past_kvs,use_cache)
        log_vocab=logits.float().log_softmax(-1)
        copy=torch.zeros_like(log_vocab).scatter_add(-1,ids[:,None].expand(-1,a.shape[1],-1),a)
        mix=torch.logaddexp(F.logsigmoid(-g)+log_vocab,F.logsigmoid(g)+copy.clamp_min(1e-30).log())
        return torch.where(has[:,None,None],mix,log_vocab),cache


def training_loss(model,source_ids,source_mask,target_ids,target_mask):
    labels=target_ids[:,1:]
    valid=target_mask[:,1:].bool()
    if not valid.any():raise ValueError('No supervised target tokens')
    logits,a,g,ids,has,_=model.features(source_ids,source_mask,target_ids[:,:-1])
    # Exact gold-token probability avoids allocating the full pointer vocabulary
    # distribution while training.
    vocab=logits.float().log_softmax(-1).gather(-1,labels[...,None]).squeeze(-1)
    copy=(a*ids[:,None].eq(labels[...,None])).sum(-1)
    mix=torch.logaddexp(F.logsigmoid(-g.squeeze(-1))+vocab,F.logsigmoid(g.squeeze(-1))+copy.clamp_min(1e-30).log())
    mix=torch.where(has[:,None],mix,vocab)
    loss=-mix[valid].mean()
    if model.training:
        masked=a*valid[...,None]
        previous=masked.cumsum(1)-masked
        coverage=torch.minimum(masked,previous).sum(-1)
        loss=loss+model.coverage_weight*(coverage*torch.sigmoid(g.squeeze(-1)))[valid].mean()
    return loss


@torch.inference_mode()
def generate(model,source_ids,source_mask,max_new_tokens,eos_id):
    was_training=model.training;model.eval()
    try:
        rows=[[] for _ in range(source_ids.shape[0])]
        token=torch.full((len(rows),1),model.bos_id,device=source_ids.device,dtype=torch.long)
        done=torch.zeros(len(rows),dtype=torch.bool,device=source_ids.device);cache=None
        for step in range(max_new_tokens):
            logp,cache=model(source_ids if cache is None else None,source_mask if cache is None else None,token,past_kvs=cache,use_cache=True)
            scores=logp[:,-1].clone()
            for i,row in enumerate(rows):
                if done[i]:continue
                if step<min(64,max_new_tokens//2):scores[i,eos_id]=-torch.inf
                if len(row)>=3:
                    prefix=tuple(row[-3:])
                    blocked={row[j+3] for j in range(len(row)-3) if tuple(row[j:j+3])==prefix}
                    if blocked:scores[i,list(blocked)]=-torch.inf
                if not torch.isfinite(scores[i]).any():scores[i,eos_id]=0.
            chosen=scores.argmax(-1)
            for i,value in enumerate(chosen.tolist()):
                if not done[i]:rows[i].append(value)
            done |= chosen.eq(eos_id)
            if done.all():break
            token=chosen.masked_fill(done,eos_id)[:,None]
        return rows
    finally:model.train(was_training)


def build(input_dir,device):
    model,tokenizer,metadata=base.build(input_dir,'cpu')
    warm=Path('/kaggle/input') if Path('/kaggle/input').exists() else Path(input_dir)
    candidates=[Path(os.environ['PHASE6_WARM_ADAPTER'])] if os.environ.get('PHASE6_WARM_ADAPTER') else list(warm.rglob('best_adapter.pt'))
    if len(candidates)!=1:raise ValueError('Exactly one B1 Phase 6 adapter source required')
    path=candidates[0];record=torch.load(path,map_location='cpu',weights_only=False)
    if record['parent_metadata']['parent_checkpoint_sha256']!=metadata['parent_checkpoint_sha256']:
        raise ValueError('Warm-start parent mismatch')
    if record['parent_metadata']['tokenizer_sha256']!=metadata['tokenizer_sha256']:
        raise ValueError('Warm-start tokenizer mismatch')
    params=dict(model.named_parameters())
    expected={n for n,p in params.items() if p.requires_grad}
    if set(record['adapter'])!=expected:raise ValueError('Warm-start adapter parameters mismatch')
    with torch.no_grad():
        for n,value in record['adapter'].items():params[n].copy_(value)
    model.__class__=EvidenceFusion
    model.add_evidence_modules()
    model=model.to(device)
    metadata.update({'method':'B1 hierarchical token evidence fusion with pointer generator and trained coverage',
        'warm_start_adapter_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'warm_start_epoch':record['epoch'],
        'hierarchy':'masked passage means, bidirectional GRU, zero-gated residual into original token memory',
        'coverage_weight':model.coverage_weight,'decoding':'greedy, four-gram blocking, minimum64 tokens subject to budget',
        'trainable_parameters':sum(p.numel() for p in model.parameters() if p.requires_grad),
        'total_parameters':sum(p.numel() for p in model.parameters())})
    return model,tokenizer,metadata
