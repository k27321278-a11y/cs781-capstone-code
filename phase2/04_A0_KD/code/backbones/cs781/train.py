import contextlib
import hashlib
import json
import math
import random
import time
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import torch

from .checkpoints import restore_checkpoint, run_lock, save_checkpoint
from .contracts import ContinuationExample, TrainState
from .examples import collate_a
from .io import sha256_file, write_json_atomic
from .models.a import LegalGPT
from .prepare import load_prepared
from .tokenizer import load_tokenizer


class IndexedExamples:
    def __init__(self,root,split="train"):
        self.path = Path(root)/f"{split}.jsonl"
        self.offsets = np.load(Path(root)/f"{split}.offsets.npy",mmap_mode="r",allow_pickle=False)
        self.epoch,self.order = None,None

    def at(self,cursor,seed):
        epoch,index = divmod(cursor,len(self.offsets))
        if epoch != self.epoch:
            self.order = list(range(len(self.offsets)))
            random.Random(seed+epoch).shuffle(self.order)
            self.epoch = epoch
        with self.path.open("rb") as handle:
            handle.seek(int(self.offsets[self.order[index]]))
            return ContinuationExample(**json.loads(handle.readline()))


def _code_digest():
    root = Path(__file__).parent
    digest = hashlib.sha256()
    for name in ("train.py","checkpoints.py","examples.py","models/a.py","contracts.py"):
        digest.update(name.encode())
        digest.update((root/name).read_bytes())
    return digest.hexdigest()


def train_a(data,model_cfg,train_cfg,run_dir,resume,*,stop_after_steps=None,smoke_test=False):
    if min(train_cfg.target_tokens,train_cfg.micro_batch_size,train_cfg.grad_accum_steps,train_cfg.save_every,train_cfg.log_every) < 1:
        raise ValueError("Positive training budgets, batches and intervals required")
    if not math.isfinite(train_cfg.learning_rate) or train_cfg.learning_rate <= 0:
        raise ValueError("Invalid learning rate")
    data = load_prepared(data.root)
    manifest = json.loads((data.root/"manifest.json").read_text())
    if manifest["synthetic"] and not smoke_test:
        raise ValueError("Synthetic data is for explicitly labeled smoke tests only")
    if model_cfg.max_seq_len != manifest["max_seq_len"]:
        raise ValueError("Model and prepared training context differ")
    run_dir = Path(run_dir)
    with run_lock(run_dir):
        checkpoint = run_dir/"checkpoints/latest.pt"
        if not resume and checkpoint.exists():
            raise FileExistsError("Existing checkpoint: explicitly resume or use a new run")
        if resume and not checkpoint.is_file():
            raise FileNotFoundError("Requested resume without a checkpoint")
        random.seed(train_cfg.seed)
        np.random.seed(train_cfg.seed)
        torch.manual_seed(train_cfg.seed)
        device = torch.device(train_cfg.device)
        if device.type=="cuda":
            torch.cuda.manual_seed_all(train_cfg.seed)
            torch.cuda.reset_peak_memory_stats(device)
        model = LegalGPT(model_cfg).to(device)
        if not smoke_test and not 50_000_000 <= model.count_parameters() <= 300_000_000:
            raise ValueError("Substantive foundation must have 50M–300M unique parameters")
        tokenizer = load_tokenizer(data.root/"tokenizer.json")
        if tokenizer.get_vocab_size() > model_cfg.vocab_size:
            raise ValueError("Tokenizer vocabulary exceeds model vocabulary")
        ids = {key:tokenizer.token_to_id(token) for key,token in (("pad_id","<pad>"),("bos_id","<bos>"),("eos_id","<eos>"))}
        optimizer = torch.optim.AdamW(model.parameters(),lr=train_cfg.learning_rate,betas=(train_cfg.beta1,train_cfg.beta2),eps=train_cfg.eps,weight_decay=train_cfg.weight_decay)
        metadata = {"model_config":asdict(model_cfg),"train_config":asdict(train_cfg),"data_manifest_sha256":data.manifest_sha256,
                    "tokenizer_sha256":data.tokenizer_sha256,"code_sha256":_code_digest(),"torch":str(torch.__version__),"numpy":np.__version__,
                    "synthetic_smoke":bool(smoke_test),"unique_parameters":model.count_parameters()}
        state = TrainState()
        if resume:
            state,sampler = restore_checkpoint(checkpoint,model,optimizer,metadata)
            if sampler["cursor"] != state.example_cursor:
                raise ValueError("Sampler/checkpoint cursor mismatch")
        write_json_atomic(run_dir/"run.json",metadata)
        dataset = IndexedExamples(data.root)
        began,initial_target = time.monotonic(),state.target_tokens
        model.train()
        with (run_dir/"metrics.jsonl").open("a",encoding="utf-8") as log:
            while state.target_tokens < train_cfg.target_tokens and (stop_after_steps is None or state.step < stop_after_steps):
                batches,step_input,step_target = [],0,0
                cursor = state.example_cursor
                source_counts = {}
                for _ in range(train_cfg.grad_accum_steps):
                    examples = [dataset.at(cursor+i,train_cfg.seed) for i in range(train_cfg.micro_batch_size)]
                    cursor += len(examples)
                    for e in examples:
                        source_counts[e.source] = source_counts.get(e.source,0)+1
                    batch = collate_a(examples,**ids,max_seq_len=model_cfg.max_seq_len)
                    batches.append(batch)
                    step_input += batch.input_tokens
                    step_target += batch.target_tokens
                fraction = state.target_tokens/train_cfg.target_tokens
                if fraction < train_cfg.warmup_ratio:
                    scale = max(1/train_cfg.target_tokens,fraction/max(train_cfg.warmup_ratio,1e-12))
                else:
                    progress = (fraction-train_cfg.warmup_ratio)/max(1-train_cfg.warmup_ratio,1e-12)
                    scale = train_cfg.min_lr_ratio + (1-train_cfg.min_lr_ratio)*.5*(1+math.cos(math.pi*min(1,progress)))
                for group in optimizer.param_groups:
                    group["lr"] = train_cfg.learning_rate*scale
                optimizer.zero_grad(set_to_none=True)
                step_loss = 0.0
                for batch in batches:
                    ctx = torch.autocast("cuda",dtype=torch.bfloat16) if device.type=="cuda" else contextlib.nullcontext()
                    with ctx:
                        _,loss,_ = model(batch.input_ids.to(device),batch.targets.to(device))
                    if not torch.isfinite(loss):
                        raise FloatingPointError("Nonfinite loss; previous checkpoint preserved")
                    (loss*batch.target_tokens/step_target).backward()
                    step_loss += loss.item()*batch.target_tokens/step_target
                norm = torch.nn.utils.clip_grad_norm_(model.parameters(),train_cfg.grad_clip,error_if_nonfinite=True)
                optimizer.step()
                state = replace(state,step=state.step+1,input_tokens=state.input_tokens+step_input,target_tokens=state.target_tokens+step_target,example_cursor=cursor)
                if state.step % train_cfg.log_every == 0:
                    elapsed = time.monotonic()-began
                    row = {**asdict(state),"loss":step_loss,"lr":optimizer.param_groups[0]["lr"],"grad_norm":float(norm),
                           "gpu_hours_this_session":elapsed/3600 if device.type=="cuda" else 0,
                           "target_tokens_per_second":(state.target_tokens-initial_target)/elapsed,
                           "peak_memory_bytes":torch.cuda.max_memory_allocated(device) if device.type=="cuda" else 0,"source_examples_this_step":source_counts}
                    log.write(json.dumps(row,allow_nan=False)+"\n")
                    log.flush()
                if state.step % train_cfg.save_every == 0:
                    save_checkpoint(checkpoint,model,optimizer,state,{"cursor":cursor},metadata)
                    candidate = run_dir/f"candidates/step_{state.step:07d}.pt"
                    candidate.parent.mkdir(exist_ok=True)
                    torch.save({"model":model.state_dict(),"metadata":metadata,"state":asdict(state)},candidate)
            save_checkpoint(checkpoint,model,optimizer,state,{"cursor":state.example_cursor},metadata)
        return state
