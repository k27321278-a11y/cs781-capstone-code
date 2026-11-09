#!/usr/bin/env python3
"""
CS781 Capstone — Phase 1 — Architecture A: Dense Flash LegalGPT

End-to-end pipeline:
  official Phase-1 corpus -> legal Byte-Level BPE -> document-safe split ->
  ~110M decoder-only Transformer -> checkpoint/resume -> local eval ->
  official test generation -> prediction.json -> submission.zip

IMPORTANT COMPETITION ASSUMPTIONS (from the capstone document):
  * Train from scratch on official Phase-1 data only.
  * 50M <= model parameters <= 300M.
  * Test input is a 512-word legal prompt.
  * Output is the generated 128-word continuation only.
  * Codabench submission is submission.zip with prediction.json at ZIP root.

Architecture A:
  vocab=32768, layers=12, d_model=768, heads=12, head_dim=64,
  SwiGLU hidden=2048, RoPE, Pre-RMSNorm, tied embedding/LM head,
  causal MHA implemented with torch.scaled_dot_product_attention (Flash SDPA
  is used automatically on supported CUDA hardware).

The default local evaluation does NOT load any external pretrained model.
It reports validation NLL/perplexity, ROUGE-L, and BLEU-4. BERTScore is left
for Codabench by default because Phase-1 rules prohibit pretrained model
weights; an optional flag exists only if the course staff explicitly permits
using the official BERTScore scorer locally.
"""

from __future__ import annotations

import argparse
import contextlib
import dataclasses
import hashlib
import json
import math
import os
import random
import re
import shutil
import time
import zipfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Sequence, Tuple
from urllib.parse import quote

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.checkpoint import checkpoint

# Optional/runtime dependencies are imported lazily where possible.

REPO_ID = "Exploration-Lab/CS-781-Capstone"
CONFIG_NAME = "phase_1"
TRAIN_FILES = {
    "acts": "Phase_1_Pretraining/Train/train_acts.json",
    "high_courts": "Phase_1_Pretraining/Train/train_high_courts.json",
    "supreme_court": "Phase_1_Pretraining/Train/train_supreme_court.json",
    "tribunals": "Phase_1_Pretraining/Train/train_tribunals.json",
}
TEST_FILE = "Phase_1_Pretraining/Test/Phase1- Test.json"
SPECIAL_TOKENS = ["<pad>", "<bos>", "<eos>", "<unk>"]


# ----------------------------- configuration ----------------------------- #

@dataclass
class ModelConfig:
    vocab_size: int = 32768
    max_seq_len: int = 2048
    n_layers: int = 12
    d_model: int = 768
    n_heads: int = 12
    d_head: int = 64
    d_ff: int = 2048
    rope_theta: float = 10000.0
    dropout: float = 0.0
    gradient_checkpointing: bool = True

    def validate(self) -> None:
        assert self.n_heads * self.d_head == self.d_model
        assert self.vocab_size < 65535, "uint16 token cache assumes vocab < 65535"


@dataclass
class DataConfig:
    repo_id: str = REPO_ID
    config_name: str = CONFIG_NAME
    vocab_size: int = 32768
    tokenizer_train_chars: int = 100_000_000
    train_tokens: int = 200_000_000
    val_tokens: int = 5_000_000
    val_fraction: float = 0.02
    val_generation_examples: int = 128
    shuffle_buffer: int = 4096
    mix_mode: str = "equal"  # equal | natural
    seed: int = 781


@dataclass
class TrainConfig:
    micro_batch_size: int = 1
    grad_accum_steps: int = 32
    learning_rate: float = 3e-4
    min_lr_ratio: float = 0.10
    warmup_ratio: float = 0.02
    weight_decay: float = 0.10
    beta1: float = 0.9
    beta2: float = 0.95
    eps: float = 1e-8
    grad_clip: float = 1.0
    target_train_tokens: int = 200_000_000
    eval_every: int = 100
    eval_batches: int = 32
    save_every: int = 100
    log_every: int = 10
    seed: int = 781


@dataclass
class GenerationConfig:
    strategy: str = "greedy"  # greedy | nucleus | topk
    temperature: float = 0.8
    top_p: float = 0.90
    top_k: int = 50
    repetition_penalty: float = 1.05
    target_words: int = 128
    max_new_tokens: int = 384
    seed: int = 781


# ----------------------------- reproducibility ---------------------------- #

def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def sha01(s: str) -> float:
    h = hashlib.sha1(s.encode("utf-8", errors="ignore")).digest()[:8]
    return int.from_bytes(h, "big") / float(2**64)


def is_validation_doc(key: str, fraction: float) -> bool:
    return sha01("VAL::" + key) < fraction


def distributed_info() -> Tuple[bool, int, int, int]:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    return world_size > 1, rank, local_rank, world_size


def setup_distributed() -> Tuple[bool, int, int, int, torch.device]:
    distributed, rank, local_rank, world_size = distributed_info()
    if distributed:
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl")
        device = torch.device("cuda", local_rank)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return distributed, rank, local_rank, world_size, device


def cleanup_distributed() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


def rank0() -> bool:
    return int(os.environ.get("RANK", "0")) == 0


# ------------------------------ data helpers ------------------------------ #

def extract_text(row: object) -> str:
    """Robustly extract a document text field from common JSON/HF row layouts."""
    if isinstance(row, str):
        return row
    if not isinstance(row, dict):
        return ""

    preferred = ["text", "document", "content", "body", "judgment", "case_text"]
    for k in preferred:
        v = row.get(k)
        if isinstance(v, str) and v.strip():
            return v
        if isinstance(v, dict):
            vv = extract_text(v)
            if vv:
                return vv

    # Some JSON formats wrap a single object under an ID key.
    if len(row) == 1:
        only = next(iter(row.values()))
        if isinstance(only, (dict, str)):
            vv = extract_text(only)
            if vv:
                return vv

    # Conservative fallback: choose the longest string-valued field.
    strings = [v for v in row.values() if isinstance(v, str) and v.strip()]
    if strings:
        return max(strings, key=len)
    return ""


def extract_id(row: object, fallback: str) -> str:
    if isinstance(row, dict):
        for k in ["id", "ID", "doc_id", "document_id", "case_id", "uid"]:
            v = row.get(k)
            if v is not None and str(v).strip():
                return str(v)
    return fallback


def legal_source_url(repo_id: str, path: str) -> str:
    encoded = quote(path, safe="/")
    return f"https://huggingface.co/datasets/{repo_id}/resolve/main/{encoded}"


def stream_one_source(repo_id: str, path: str, source: str, seed: int, shuffle_buffer: int) -> Iterator[Dict[str, str]]:
    from datasets import load_dataset

    url = legal_source_url(repo_id, path)
    ds = load_dataset("json", data_files={"train": url}, split="train", streaming=True)
    ds = ds.shuffle(seed=seed, buffer_size=shuffle_buffer)
    for i, row in enumerate(ds):
        text = extract_text(row)
        if not text or len(text) < 64:
            continue
        rid = extract_id(row, f"{source}:{i}:{hashlib.sha1(text[:512].encode('utf-8', errors='ignore')).hexdigest()[:12]}")
        yield {"id": rid, "text": text, "source": source}


def stream_natural(repo_id: str, config_name: str, seed: int, shuffle_buffer: int) -> Iterator[Dict[str, str]]:
    from datasets import load_dataset

    ds = load_dataset(repo_id, config_name, split="train", streaming=True)
    ds = ds.shuffle(seed=seed, buffer_size=shuffle_buffer)
    for i, row in enumerate(ds):
        text = extract_text(row)
        if not text or len(text) < 64:
            continue
        rid = extract_id(row, f"natural:{i}:{hashlib.sha1(text[:512].encode('utf-8', errors='ignore')).hexdigest()[:12]}")
        yield {"id": rid, "text": text, "source": str(row.get("source", "unknown")) if isinstance(row, dict) else "unknown"}


def stream_equal_mix(cfg: DataConfig) -> Iterator[Dict[str, str]]:
    """Equal-probability online interleave across the four official Phase-1 sources."""
    names = list(TRAIN_FILES)
    iters = {
        name: iter(stream_one_source(cfg.repo_id, TRAIN_FILES[name], name, cfg.seed + j * 997, cfg.shuffle_buffer))
        for j, name in enumerate(names)
    }
    rng = random.Random(cfg.seed)
    alive = set(names)
    while alive:
        name = rng.choice(sorted(alive))
        try:
            yield next(iters[name])
        except StopIteration:
            alive.remove(name)


def iter_documents(cfg: DataConfig) -> Iterator[Dict[str, str]]:
    if cfg.mix_mode == "natural":
        yield from stream_natural(cfg.repo_id, cfg.config_name, cfg.seed, cfg.shuffle_buffer)
        return
    if cfg.mix_mode == "equal":
        try:
            yield from stream_equal_mix(cfg)
            return
        except Exception as e:
            print(f"[WARN] Equal-source streaming failed ({type(e).__name__}: {e}). Falling back to official combined phase_1 stream.")
            yield from stream_natural(cfg.repo_id, cfg.config_name, cfg.seed, cfg.shuffle_buffer)
            return
    raise ValueError(f"Unknown mix_mode={cfg.mix_mode}")


def tokenizer_paths(workdir: Path) -> Tuple[Path, Path]:
    return workdir / "tokenizer" / "tokenizer.json", workdir / "tokenizer" / "tokenizer_meta.json"


def train_tokenizer(workdir: Path, cfg: DataConfig, force: bool = False) -> None:
    from tokenizers import Tokenizer
    from tokenizers.decoders import ByteLevel as ByteLevelDecoder
    from tokenizers.models import BPE
    from tokenizers.pre_tokenizers import ByteLevel
    from tokenizers.trainers import BpeTrainer

    tok_path, meta_path = tokenizer_paths(workdir)
    if tok_path.exists() and not force:
        print(f"[tokenizer] Using existing {tok_path}")
        return

    (workdir / "tokenizer").mkdir(parents=True, exist_ok=True)
    model = BPE(unk_token="<unk>")
    tokenizer = Tokenizer(model)
    tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=False)
    tokenizer.decoder = ByteLevelDecoder()
    trainer = BpeTrainer(
        vocab_size=cfg.vocab_size,
        min_frequency=2,
        special_tokens=SPECIAL_TOKENS,
        show_progress=True,
    )

    chars_seen = 0
    docs_seen = 0

    def text_iter() -> Iterator[str]:
        nonlocal chars_seen, docs_seen
        for doc in iter_documents(cfg):
            key = f"{doc['source']}::{doc['id']}::{hashlib.sha1(doc['text'][:512].encode('utf-8', errors='ignore')).hexdigest()}"
            if is_validation_doc(key, cfg.val_fraction):
                continue
            text = doc["text"]
            remain = cfg.tokenizer_train_chars - chars_seen
            if remain <= 0:
                break
            piece = text[:remain]
            chars_seen += len(piece)
            docs_seen += 1
            yield piece
            if chars_seen >= cfg.tokenizer_train_chars:
                break

    print(f"[tokenizer] Training legal Byte-Level BPE on up to {cfg.tokenizer_train_chars:,} official-corpus characters...")
    tokenizer.train_from_iterator(text_iter(), trainer=trainer)
    tokenizer.save(str(tok_path))
    meta = {
        "repo_id": cfg.repo_id,
        "vocab_size_requested": cfg.vocab_size,
        "vocab_size_actual": tokenizer.get_vocab_size(),
        "chars_seen": chars_seen,
        "docs_seen": docs_seen,
        "special_tokens": SPECIAL_TOKENS,
        "mix_mode": cfg.mix_mode,
        "val_fraction_excluded": cfg.val_fraction,
    }
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"[tokenizer] Saved {tok_path}; actual vocab={tokenizer.get_vocab_size():,}")


def load_tokenizer(workdir: Path):
    from tokenizers import Tokenizer

    tok_path, _ = tokenizer_paths(workdir)
    if not tok_path.exists():
        raise FileNotFoundError(f"Tokenizer missing: {tok_path}. Run --mode prepare first.")
    return Tokenizer.from_file(str(tok_path))


def special_id(tokenizer, token: str) -> int:
    tid = tokenizer.token_to_id(token)
    if tid is None:
        raise ValueError(f"Special token {token} is missing from tokenizer")
    return int(tid)


def deterministic_window(words: List[str], key: str, total_words: int = 640) -> Optional[List[str]]:
    if len(words) < total_words:
        return None
    max_start = len(words) - total_words
    if max_start <= 0:
        start = 0
    else:
        h = int(hashlib.sha1(("WINDOW::" + key).encode("utf-8", errors="ignore")).hexdigest()[:16], 16)
        start = h % (max_start + 1)
    return words[start : start + total_words]


def prepare_token_cache(workdir: Path, cfg: DataConfig, force: bool = False) -> None:
    tokenizer = load_tokenizer(workdir)
    vocab_actual = tokenizer.get_vocab_size()
    if vocab_actual >= 65535:
        raise ValueError("Tokenizer vocab too large for uint16 token cache")

    data_dir = workdir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    train_bin = data_dir / "train_tokens.bin"
    val_bin = data_dir / "val_tokens.bin"
    val_gen_path = data_dir / "val_continuations.json"
    meta_path = data_dir / "data_meta.json"

    if train_bin.exists() and val_bin.exists() and val_gen_path.exists() and meta_path.exists() and not force:
        print(f"[data] Using existing token cache under {data_dir}")
        return

    for p in [train_bin, val_bin, val_gen_path, meta_path]:
        if p.exists():
            p.unlink()

    bos = special_id(tokenizer, "<bos>")
    eos = special_id(tokenizer, "<eos>")

    train_count = 0
    val_count = 0
    docs_train = 0
    docs_val = 0
    source_train = {k: 0 for k in TRAIN_FILES}
    source_val = {k: 0 for k in TRAIN_FILES}
    val_examples: List[Dict[str, str]] = []

    print(f"[data] Building uint16 token cache: train={cfg.train_tokens:,}, val={cfg.val_tokens:,}")
    with train_bin.open("wb") as ftr, val_bin.open("wb") as fva:
        for doc in iter_documents(cfg):
            text = doc["text"]
            source = doc.get("source", "unknown")
            key = f"{source}::{doc['id']}::{hashlib.sha1(text[:512].encode('utf-8', errors='ignore')).hexdigest()}"
            is_val = is_validation_doc(key, cfg.val_fraction)

            if is_val:
                docs_val += 1
                source_val[source] = source_val.get(source, 0) + 1
                if val_count < cfg.val_tokens:
                    ids = [bos] + tokenizer.encode(text).ids + [eos]
                    ids = ids[: max(0, cfg.val_tokens - val_count)]
                    if ids:
                        np.asarray(ids, dtype=np.uint16).tofile(fva)
                        val_count += len(ids)
                if len(val_examples) < cfg.val_generation_examples:
                    words = re.findall(r"\S+", text)
                    window = deterministic_window(words, key, 640)
                    if window is not None:
                        val_examples.append({
                            "id": key,
                            "source": source,
                            "prompt": " ".join(window[:512]),
                            "reference": " ".join(window[512:640]),
                        })
            else:
                docs_train += 1
                source_train[source] = source_train.get(source, 0) + 1
                if train_count < cfg.train_tokens:
                    ids = [bos] + tokenizer.encode(text).ids + [eos]
                    ids = ids[: max(0, cfg.train_tokens - train_count)]
                    if ids:
                        np.asarray(ids, dtype=np.uint16).tofile(ftr)
                        train_count += len(ids)

            if (docs_train + docs_val) % 1000 == 0:
                print(
                    f"[data] docs={docs_train+docs_val:,} train_tokens={train_count:,}/{cfg.train_tokens:,} "
                    f"val_tokens={val_count:,}/{cfg.val_tokens:,} val_gen={len(val_examples)}/{cfg.val_generation_examples}"
                )

            if (
                train_count >= cfg.train_tokens
                and val_count >= cfg.val_tokens
                and len(val_examples) >= cfg.val_generation_examples
            ):
                break

    val_gen_path.write_text(json.dumps(val_examples, ensure_ascii=False, indent=2), encoding="utf-8")
    meta = {
        "repo_id": cfg.repo_id,
        "mix_mode": cfg.mix_mode,
        "val_fraction": cfg.val_fraction,
        "train_tokens": train_count,
        "val_tokens": val_count,
        "train_docs_seen": docs_train,
        "val_docs_seen": docs_val,
        "source_train_docs": source_train,
        "source_val_docs": source_val,
        "val_generation_examples": len(val_examples),
        "dtype": "uint16",
    }
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"[data] Saved cache under {data_dir}")
    print(json.dumps(meta, indent=2))


def resolve_hf_revision(repo_id: str) -> Optional[str]:
    try:
        from huggingface_hub import HfApi
        return HfApi().dataset_info(repo_id).sha
    except Exception:
        return None


# ------------------------------- model code ------------------------------- #

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        x_float = x.float()
        x_norm = x_float * torch.rsqrt(x_float.pow(2).mean(-1, keepdim=True) + self.eps)
        return (x_norm.to(dtype) * self.weight)


class RotaryEmbedding(nn.Module):
    def __init__(self, head_dim: int, max_seq_len: int, theta: float = 10000.0):
        super().__init__()
        assert head_dim % 2 == 0
        inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))
        positions = torch.arange(max_seq_len, dtype=torch.float32)
        freqs = torch.outer(positions, inv_freq)
        self.register_buffer("cos", freqs.cos(), persistent=False)
        self.register_buffer("sin", freqs.sin(), persistent=False)

    def apply_rotary(self, x: torch.Tensor, offset: int = 0) -> torch.Tensor:
        # x: [B, H, T, Dh]
        t = x.size(-2)
        if offset + t > self.cos.size(0):
            raise ValueError(f"RoPE position {offset+t} exceeds max_seq_len={self.cos.size(0)}")
        cos = self.cos[offset : offset + t].to(device=x.device, dtype=x.dtype)[None, None, :, :]
        sin = self.sin[offset : offset + t].to(device=x.device, dtype=x.dtype)[None, None, :, :]
        xe = x[..., 0::2]
        xo = x[..., 1::2]
        ye = xe * cos - xo * sin
        yo = xe * sin + xo * cos
        return torch.stack((ye, yo), dim=-1).flatten(-2)


class CausalMHA(nn.Module):
    def __init__(self, cfg: ModelConfig, rope: RotaryEmbedding):
        super().__init__()
        self.cfg = cfg
        self.rope = rope
        self.wqkv = nn.Linear(cfg.d_model, 3 * cfg.d_model, bias=False)
        self.wo = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.dropout = cfg.dropout

    def forward(
        self,
        x: torch.Tensor,
        past_kv: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        b, t, d = x.shape
        qkv = self.wqkv(x)
        q, k, v = qkv.chunk(3, dim=-1)
        q = q.view(b, t, self.cfg.n_heads, self.cfg.d_head).transpose(1, 2)
        k = k.view(b, t, self.cfg.n_heads, self.cfg.d_head).transpose(1, 2)
        v = v.view(b, t, self.cfg.n_heads, self.cfg.d_head).transpose(1, 2)

        past_len = 0 if past_kv is None else past_kv[0].size(-2)
        q = self.rope.apply_rotary(q, offset=past_len)
        k = self.rope.apply_rotary(k, offset=past_len)

        if past_kv is not None:
            if t != 1:
                raise ValueError("Cached decoding currently expects one new token at a time")
            pk, pv = past_kv
            k = torch.cat([pk, k], dim=-2)
            v = torch.cat([pv, v], dim=-2)
            causal = False  # one query at newest position may attend to all cached keys
        else:
            causal = True

        y = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=None,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=causal,
        )
        y = y.transpose(1, 2).contiguous().view(b, t, d)
        y = self.wo(y)
        cache = (k, v) if use_cache else None
        return y, cache


class SwiGLU(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.w1 = nn.Linear(cfg.d_model, cfg.d_ff, bias=False)
        self.w3 = nn.Linear(cfg.d_model, cfg.d_ff, bias=False)
        self.w2 = nn.Linear(cfg.d_ff, cfg.d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class TransformerBlock(nn.Module):
    def __init__(self, cfg: ModelConfig, rope: RotaryEmbedding):
        super().__init__()
        self.attn_norm = RMSNorm(cfg.d_model)
        self.attn = CausalMHA(cfg, rope)
        self.ffn_norm = RMSNorm(cfg.d_model)
        self.ffn = SwiGLU(cfg)

    def forward(
        self,
        x: torch.Tensor,
        past_kv: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        a, cache = self.attn(self.attn_norm(x), past_kv=past_kv, use_cache=use_cache)
        x = x + a
        x = x + self.ffn(self.ffn_norm(x))
        return x, cache


class LegalGPT(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        cfg.validate()
        self.cfg = cfg
        self.tok_embeddings = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.rope = RotaryEmbedding(cfg.d_head, cfg.max_seq_len, cfg.rope_theta)
        self.blocks = nn.ModuleList([TransformerBlock(cfg, self.rope) for _ in range(cfg.n_layers)])
        self.norm = RMSNorm(cfg.d_model)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self.lm_head.weight = self.tok_embeddings.weight
        self.gradient_checkpointing = cfg.gradient_checkpointing
        self.apply(self._init_weights)
        # GPT-style residual scaling for output projections.
        residual_std = 0.02 / math.sqrt(2 * cfg.n_layers)
        for block in self.blocks:
            nn.init.normal_(block.attn.wo.weight, mean=0.0, std=residual_std)
            nn.init.normal_(block.ffn.w2.weight, mean=0.0, std=residual_std)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self,
        input_ids: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
        past_kvs: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[List[Tuple[torch.Tensor, torch.Tensor]]]]:
        b, t = input_ids.shape
        past_len = 0 if not past_kvs else past_kvs[0][0].size(-2)
        if past_len + t > self.cfg.max_seq_len:
            raise ValueError(f"Sequence length {past_len+t} exceeds max_seq_len={self.cfg.max_seq_len}")

        x = self.tok_embeddings(input_ids)
        new_caches: List[Tuple[torch.Tensor, torch.Tensor]] = []

        for i, block in enumerate(self.blocks):
            pkv = None if past_kvs is None else past_kvs[i]
            if self.training and self.gradient_checkpointing and not use_cache and pkv is None:
                def block_only(inp, blk=block):
                    return blk(inp, past_kv=None, use_cache=False)[0]
                x = checkpoint(block_only, x, use_reentrant=False)
                cache = None
            else:
                x, cache = block(x, past_kv=pkv, use_cache=use_cache)
            if use_cache:
                assert cache is not None
                new_caches.append(cache)

        x = self.norm(x)
        logits = self.lm_head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
        return logits, loss, (new_caches if use_cache else None)

    def count_parameters(self) -> int:
        # Deduplicate tied weights.
        return sum(p.numel() for p in {id(p): p for p in self.parameters()}.values())


# ------------------------------ train helpers ----------------------------- #

class RandomMemmapBatcher:
    def __init__(self, path: Path, block_size: int, batch_size: int, seed: int):
        self.path = path
        self.block_size = block_size
        self.batch_size = batch_size
        self.mm = np.memmap(path, dtype=np.uint16, mode="r")
        if len(self.mm) < block_size + 2:
            raise ValueError(f"Token cache {path} is too small ({len(self.mm)} tokens)")
        self.rng = np.random.default_rng(seed)

    def batch(self, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        starts = self.rng.integers(0, len(self.mm) - self.block_size - 1, size=self.batch_size)
        arr = np.stack([np.asarray(self.mm[s : s + self.block_size + 1], dtype=np.int64) for s in starts])
        x = torch.from_numpy(arr[:, :-1]).to(device=device, dtype=torch.long, non_blocking=True)
        y = torch.from_numpy(arr[:, 1:]).to(device=device, dtype=torch.long, non_blocking=True)
        return x, y


def build_optimizer(model: nn.Module, cfg: TrainConfig):
    decay, no_decay = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        (decay if p.ndim >= 2 else no_decay).append(p)
    groups = [
        {"params": decay, "weight_decay": cfg.weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]
    kwargs = dict(lr=cfg.learning_rate, betas=(cfg.beta1, cfg.beta2), eps=cfg.eps)
    if torch.cuda.is_available():
        try:
            return torch.optim.AdamW(groups, fused=True, **kwargs)
        except Exception:
            pass
    return torch.optim.AdamW(groups, **kwargs)


def lr_for_step(step: int, max_steps: int, cfg: TrainConfig) -> float:
    warmup = max(1, int(max_steps * cfg.warmup_ratio))
    min_lr = cfg.learning_rate * cfg.min_lr_ratio
    if step < warmup:
        return cfg.learning_rate * (step + 1) / warmup
    progress = (step - warmup) / max(1, max_steps - warmup)
    cosine = 0.5 * (1.0 + math.cos(math.pi * min(max(progress, 0.0), 1.0)))
    return min_lr + (cfg.learning_rate - min_lr) * cosine


def set_lr(optimizer, lr: float) -> None:
    for g in optimizer.param_groups:
        g["lr"] = lr


def amp_dtype(device: torch.device) -> torch.dtype:
    if device.type != "cuda":
        return torch.float32
    if torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float16


def autocast_context(device: torch.device, dtype: torch.dtype):
    if device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=dtype)
    return contextlib.nullcontext()


def evaluate_loss(model: nn.Module, batcher: RandomMemmapBatcher, batches: int, device: torch.device, dtype: torch.dtype) -> float:
    model.eval()
    losses = []
    with torch.no_grad():
        for _ in range(batches):
            x, y = batcher.batch(device)
            with autocast_context(device, dtype):
                _, loss, _ = model(x, targets=y)
            losses.append(float(loss.detach().cpu()))
    model.train()
    return float(np.mean(losses))


def model_config_path(workdir: Path) -> Path:
    return workdir / "checkpoints" / "model_config.json"


def save_checkpoint(
    workdir: Path,
    model: nn.Module,
    optimizer,
    scaler,
    step: int,
    best_val: float,
    tokens_seen: int,
    model_cfg: ModelConfig,
    train_cfg: TrainConfig,
    is_best: bool,
) -> None:
    ckpt_dir = workdir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    raw = model.module if isinstance(model, DDP) else model
    state = {
        "model": raw.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scaler": scaler.state_dict() if scaler is not None else None,
        "step": step,
        "best_val": best_val,
        "tokens_seen": tokens_seen,
        "model_config": asdict(model_cfg),
        "train_config": asdict(train_cfg),
    }
    tmp = ckpt_dir / "latest.tmp.pt"
    torch.save(state, tmp)
    os.replace(tmp, ckpt_dir / "latest.pt")
    model_config_path(workdir).write_text(json.dumps(asdict(model_cfg), indent=2), encoding="utf-8")
    if is_best:
        shutil.copy2(ckpt_dir / "latest.pt", ckpt_dir / "best.pt")


def load_training_checkpoint(workdir: Path, model: nn.Module, optimizer, scaler, device: torch.device):
    path = workdir / "checkpoints" / "latest.pt"
    if not path.exists():
        return 0, float("inf"), 0
    print(f"[resume] Loading {path}")
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["model"], strict=True)
    optimizer.load_state_dict(ckpt["optimizer"])
    if scaler is not None and ckpt.get("scaler") is not None:
        scaler.load_state_dict(ckpt["scaler"])
    return int(ckpt.get("step", 0)), float(ckpt.get("best_val", float("inf"))), int(ckpt.get("tokens_seen", 0))


def train(workdir: Path, model_cfg: ModelConfig, train_cfg: TrainConfig, resume: bool = True) -> None:
    distributed, rank, local_rank, world_size, device = setup_distributed()
    try:
        seed_everything(train_cfg.seed + rank)
        torch.set_float32_matmul_precision("high")
        if device.type == "cuda":
            torch.backends.cuda.enable_flash_sdp(True)
            torch.backends.cuda.enable_mem_efficient_sdp(True)
            torch.backends.cuda.enable_math_sdp(True)

        tokenizer = load_tokenizer(workdir)
        actual_vocab = tokenizer.get_vocab_size()
        if actual_vocab != model_cfg.vocab_size:
            print(f"[model] Updating vocab_size {model_cfg.vocab_size} -> tokenizer actual {actual_vocab}")
            model_cfg.vocab_size = actual_vocab

        train_bin = workdir / "data" / "train_tokens.bin"
        val_bin = workdir / "data" / "val_tokens.bin"
        if not train_bin.exists() or not val_bin.exists():
            raise FileNotFoundError("Token caches missing. Run --mode prepare first.")

        train_batcher = RandomMemmapBatcher(train_bin, model_cfg.max_seq_len, train_cfg.micro_batch_size, train_cfg.seed + rank * 10007)
        val_batcher = RandomMemmapBatcher(val_bin, model_cfg.max_seq_len, train_cfg.micro_batch_size, train_cfg.seed + 99991)

        model = LegalGPT(model_cfg).to(device)
        n_params = model.count_parameters()
        if rank0():
            print(f"[model] parameters = {n_params:,} ({n_params/1e6:.2f}M)")
            if not (50_000_000 <= n_params <= 300_000_000):
                raise ValueError("Model violates capstone 50M–300M parameter requirement")
            print(f"[device] {device}; AMP={amp_dtype(device)}")
            if device.type == "cuda":
                print(f"[cuda] GPU={torch.cuda.get_device_name(device)} flash_sdp_enabled={torch.backends.cuda.flash_sdp_enabled()}")

        optimizer = build_optimizer(model, train_cfg)
        dtype = amp_dtype(device)
        use_scaler = device.type == "cuda" and dtype == torch.float16
        scaler = torch.cuda.amp.GradScaler(enabled=use_scaler) if device.type == "cuda" else None

        start_step, best_val, tokens_seen = (0, float("inf"), 0)
        if resume:
            start_step, best_val, tokens_seen = load_training_checkpoint(workdir, model, optimizer, scaler, device)

        if distributed:
            model = DDP(model, device_ids=[local_rank], output_device=local_rank, broadcast_buffers=False)

        tokens_per_step = train_cfg.micro_batch_size * model_cfg.max_seq_len * train_cfg.grad_accum_steps * world_size
        max_steps = math.ceil(train_cfg.target_train_tokens / tokens_per_step)
        if rank0():
            print(f"[train] tokens/optimizer-step={tokens_per_step:,}; target={train_cfg.target_train_tokens:,}; max_steps={max_steps:,}")
            print(f"[train] starting at step={start_step:,}; tokens_seen≈{tokens_seen:,}")

        model.train()
        optimizer.zero_grad(set_to_none=True)
        t0 = time.time()
        last_log = time.time()

        for step in range(start_step, max_steps):
            lr = lr_for_step(step, max_steps, train_cfg)
            set_lr(optimizer, lr)
            step_loss = 0.0

            for micro in range(train_cfg.grad_accum_steps):
                x, y = train_batcher.batch(device)
                sync_ctx = (
                    model.no_sync()
                    if distributed and micro < train_cfg.grad_accum_steps - 1
                    else contextlib.nullcontext()
                )
                with sync_ctx:
                    with autocast_context(device, dtype):
                        _, loss, _ = model(x, targets=y)
                        loss_scaled = loss / train_cfg.grad_accum_steps
                    if scaler is not None and use_scaler:
                        scaler.scale(loss_scaled).backward()
                    else:
                        loss_scaled.backward()
                step_loss += float(loss.detach().cpu())

            if scaler is not None and use_scaler:
                scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), train_cfg.grad_clip)
            if scaler is not None and use_scaler:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad(set_to_none=True)

            tokens_seen += tokens_per_step
            mean_loss = step_loss / train_cfg.grad_accum_steps

            if rank0() and ((step + 1) % train_cfg.log_every == 0 or step == start_step):
                now = time.time()
                dt = now - last_log
                steps_since = train_cfg.log_every if step + 1 >= train_cfg.log_every else 1
                tok_s = (tokens_per_step * steps_since) / max(dt, 1e-9)
                print(
                    f"[train] step={step+1}/{max_steps} loss={mean_loss:.4f} ppl={math.exp(min(mean_loss, 20)):.2f} "
                    f"lr={lr:.3e} tokens={tokens_seen/1e6:.1f}M tok/s≈{tok_s:,.0f}"
                )
                last_log = now

            do_eval = (step + 1) % train_cfg.eval_every == 0 or (step + 1) == max_steps
            if do_eval:
                if distributed:
                    dist.barrier()
                if rank0():
                    raw = model.module if isinstance(model, DDP) else model
                    val_loss = evaluate_loss(raw, val_batcher, train_cfg.eval_batches, device, dtype)
                    val_ppl = math.exp(min(val_loss, 20))
                    improved = val_loss < best_val
                    best_val = min(best_val, val_loss)
                    print(f"[val] step={step+1} nll={val_loss:.4f} ppl={val_ppl:.2f} best={best_val:.4f}")
                    save_checkpoint(workdir, model, optimizer, scaler, step + 1, best_val, tokens_seen, model_cfg, train_cfg, is_best=improved)
                if distributed:
                    dist.barrier()

            elif rank0() and (step + 1) % train_cfg.save_every == 0:
                save_checkpoint(workdir, model, optimizer, scaler, step + 1, best_val, tokens_seen, model_cfg, train_cfg, is_best=False)

        if rank0():
            elapsed = time.time() - t0
            print(f"[train] complete in {elapsed/3600:.2f} h")
    finally:
        cleanup_distributed()


# ------------------------------ generation ------------------------------- #

def load_model_for_inference(workdir: Path, device: torch.device, checkpoint_name: str = "best.pt") -> Tuple[LegalGPT, object, ModelConfig]:
    ckpt_path = workdir / "checkpoints" / checkpoint_name
    if not ckpt_path.exists():
        ckpt_path = workdir / "checkpoints" / "latest.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError("No checkpoint found")
    ckpt = torch.load(ckpt_path, map_location=device)
    cfg = ModelConfig(**ckpt["model_config"])
    cfg.gradient_checkpointing = False
    model = LegalGPT(cfg).to(device)
    model.load_state_dict(ckpt["model"], strict=True)
    model.eval()
    tokenizer = load_tokenizer(workdir)
    print(f"[inference] loaded {ckpt_path}; params={model.count_parameters()/1e6:.2f}M")
    return model, tokenizer, cfg


def apply_repetition_penalty(logits: torch.Tensor, token_ids: Sequence[int], penalty: float) -> torch.Tensor:
    if penalty is None or penalty <= 1.0 or not token_ids:
        return logits
    logits = logits.clone()
    unique = set(int(t) for t in token_ids)
    for tid in unique:
        val = logits[..., tid]
        logits[..., tid] = torch.where(val < 0, val * penalty, val / penalty)
    return logits


def sample_token(logits: torch.Tensor, cfg: GenerationConfig, generated: Sequence[int], generator: torch.Generator) -> int:
    logits = apply_repetition_penalty(logits, generated, cfg.repetition_penalty)
    if cfg.strategy == "greedy":
        return int(torch.argmax(logits, dim=-1).item())

    temperature = max(cfg.temperature, 1e-5)
    logits = logits / temperature

    if cfg.strategy == "topk":
        k = min(max(1, cfg.top_k), logits.size(-1))
        vals, idx = torch.topk(logits, k)
        probs = F.softmax(vals, dim=-1)
        choice = torch.multinomial(probs, 1, generator=generator)
        return int(idx.gather(-1, choice).item())

    if cfg.strategy == "nucleus":
        sorted_logits, sorted_idx = torch.sort(logits, descending=True)
        probs = F.softmax(sorted_logits, dim=-1)
        cum = torch.cumsum(probs, dim=-1)
        remove = cum > cfg.top_p
        remove[..., 1:] = remove[..., :-1].clone()
        remove[..., 0] = False
        sorted_logits = sorted_logits.masked_fill(remove, float("-inf"))
        probs = F.softmax(sorted_logits, dim=-1)
        choice = torch.multinomial(probs, 1, generator=generator)
        return int(sorted_idx.gather(-1, choice).item())

    raise ValueError(f"Unknown generation strategy={cfg.strategy}")


def trim_to_words(text: str, n: int) -> str:
    matches = list(re.finditer(r"\S+", text))
    if len(matches) < n:
        return text.strip()
    end = matches[n - 1].end()
    return text[:end].strip()


def word_count(text: str) -> int:
    return len(re.findall(r"\S+", text))


@torch.no_grad()
def generate_continuation(
    model: LegalGPT,
    tokenizer,
    prompt: str,
    cfg: GenerationConfig,
    device: torch.device,
    seed_offset: int = 0,
) -> str:
    pad = special_id(tokenizer, "<pad>")
    bos = special_id(tokenizer, "<bos>")
    eos = special_id(tokenizer, "<eos>")
    unk = special_id(tokenizer, "<unk>")
    special_forbidden = [pad, bos, eos, unk]
    prompt_ids = [bos] + tokenizer.encode(prompt).ids
    if len(prompt_ids) >= model.cfg.max_seq_len - 16:
        raise ValueError(f"Prompt tokenized to {len(prompt_ids)} tokens, too long for max_seq_len={model.cfg.max_seq_len}")

    x = torch.tensor(prompt_ids, dtype=torch.long, device=device)[None, :]
    dtype = amp_dtype(device)
    with autocast_context(device, dtype):
        logits, _, caches = model(x, use_cache=True)
    next_logits = logits[:, -1, :]

    gen_ids: List[int] = []
    g = torch.Generator(device=device if device.type == "cuda" else "cpu")
    g.manual_seed(cfg.seed + seed_offset)

    hard_limit = min(cfg.max_new_tokens, model.cfg.max_seq_len - len(prompt_ids))
    for _ in range(hard_limit):
        # Fixed-length continuation task: never emit tokenizer control symbols.
        clean_logits = next_logits.clone()
        clean_logits[..., special_forbidden] = float("-inf")
        tid = sample_token(clean_logits, cfg, gen_ids, g)
        gen_ids.append(tid)

        text = tokenizer.decode(gen_ids, skip_special_tokens=True)
        if word_count(text) >= cfg.target_words:
            return trim_to_words(text, cfg.target_words)

        token = torch.tensor([[tid]], dtype=torch.long, device=device)
        with autocast_context(device, dtype):
            logits, _, caches = model(token, past_kvs=caches, use_cache=True)
        next_logits = logits[:, -1, :]

    text = tokenizer.decode(gen_ids, skip_special_tokens=True)
    if word_count(text) < cfg.target_words:
        raise RuntimeError(
            f"Generated only {word_count(text)} words after {hard_limit} tokens. "
            f"Increase max_seq_len/max_new_tokens or inspect tokenizer behavior."
        )
    return trim_to_words(text, cfg.target_words)


# ------------------------------- evaluation ------------------------------- #

def load_val_examples(workdir: Path) -> List[Dict[str, str]]:
    path = workdir / "data" / "val_continuations.json"
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}; run --mode prepare")
    return json.loads(path.read_text(encoding="utf-8"))


def compute_generation_metrics(hyps: List[str], refs: List[str], allow_local_bertscore: bool = False) -> Dict[str, float]:
    from rouge_score import rouge_scorer
    import sacrebleu

    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    rouge_l = float(np.mean([scorer.score(r, h)["rougeL"].fmeasure for h, r in zip(hyps, refs)]))
    bleu4 = float(sacrebleu.corpus_bleu(hyps, [refs], smooth_method="exp").score / 100.0)
    out = {"rougeL_f1": rouge_l, "bleu4": bleu4}

    if allow_local_bertscore:
        # WARNING: this loads a pretrained scorer model. Enable only if course staff
        # explicitly says local use of the official BERTScore scorer is permitted.
        from bert_score import score as bert_score
        _, _, f1 = bert_score(hyps, refs, lang="en", verbose=False)
        out["bertscore_f1"] = float(f1.mean().item())
    return out


def evaluate_generation(
    workdir: Path,
    gen_cfg: GenerationConfig,
    limit: int,
    allow_local_bertscore: bool = False,
) -> Dict[str, float]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, tokenizer, _ = load_model_for_inference(workdir, device)
    examples = load_val_examples(workdir)[:limit]
    hyps, refs = [], []
    per_example = []
    for i, ex in enumerate(examples):
        hyp = generate_continuation(model, tokenizer, ex["prompt"], gen_cfg, device, seed_offset=i)
        hyps.append(hyp)
        refs.append(ex["reference"])
        per_example.append({"id": ex["id"], "source": ex.get("source"), "prediction": hyp, "reference": ex["reference"]})
        print(f"[gen-eval] {i+1}/{len(examples)} words={word_count(hyp)}")

    metrics = compute_generation_metrics(hyps, refs, allow_local_bertscore=allow_local_bertscore)
    metrics.update({"n_examples": len(examples), "strategy": gen_cfg.strategy})
    out_dir = workdir / "evaluation"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"generation_{gen_cfg.strategy}.json").write_text(json.dumps(per_example, ensure_ascii=False, indent=2), encoding="utf-8")
    (out_dir / f"metrics_{gen_cfg.strategy}.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print("[gen-eval]", json.dumps(metrics, indent=2))
    return metrics


# ------------------------------ submission -------------------------------- #

def download_test_json(workdir: Path, repo_id: str) -> Path:
    from huggingface_hub import hf_hub_download

    local = hf_hub_download(repo_id=repo_id, filename=TEST_FILE, repo_type="dataset")
    dst = workdir / "data" / "Phase1-Test.json"
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(local, dst)
    return dst


def normalize_test_examples(obj: object) -> List[Tuple[str, str]]:
    examples: List[Tuple[str, str]] = []
    if isinstance(obj, dict):
        # Exact capstone format: {"1": {"text": "..."}, ...}
        if all(isinstance(v, (dict, str)) for v in obj.values()):
            for k, v in obj.items():
                text = extract_text(v)
                if not text:
                    raise ValueError(f"Could not extract text for official test ID {k}")
                examples.append((str(k), text))
            return examples
        # Possible wrapper around a list.
        for key in ["test", "data", "examples"]:
            if key in obj:
                return normalize_test_examples(obj[key])
    elif isinstance(obj, list):
        for i, row in enumerate(obj):
            if not isinstance(row, dict):
                raise ValueError("Unexpected list entry in test JSON")
            rid = extract_id(row, str(i + 1))
            text = extract_text(row)
            if not text:
                raise ValueError(f"Could not extract text for test row {i}")
            examples.append((rid, text))
        return examples
    raise ValueError("Unsupported official test JSON schema")


def make_submission(workdir: Path, gen_cfg: GenerationConfig, repo_id: str = REPO_ID) -> Path:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, tokenizer, _ = load_model_for_inference(workdir, device)
    test_path = download_test_json(workdir, repo_id)
    test_obj = json.loads(test_path.read_text(encoding="utf-8"))
    tests = normalize_test_examples(test_obj)
    print(f"[submit] official test examples={len(tests)}")

    predictions: Dict[str, Dict[str, str]] = {}
    for i, (rid, prompt) in enumerate(tests):
        if word_count(prompt) != 512:
            print(f"[WARN] Official prompt ID={rid} has {word_count(prompt)} whitespace words, expected 512 per capstone.")
        cont = generate_continuation(model, tokenizer, prompt, gen_cfg, device, seed_offset=i)
        if word_count(cont) != gen_cfg.target_words:
            raise AssertionError(f"ID {rid}: continuation has {word_count(cont)} words, expected {gen_cfg.target_words}")
        predictions[rid] = {"text": cont}
        print(f"[submit] {i+1}/{len(tests)} id={rid} words={word_count(cont)}")

    out_dir = workdir / "submission" / gen_cfg.strategy
    out_dir.mkdir(parents=True, exist_ok=True)
    pred_path = out_dir / "prediction.json"
    pred_path.write_text(json.dumps(predictions, ensure_ascii=False, indent=2), encoding="utf-8")

    # Strict integrity checks before zipping.
    expected_ids = [rid for rid, _ in tests]
    if set(predictions) != set(expected_ids) or len(predictions) != len(expected_ids):
        raise AssertionError("Prediction IDs do not exactly match official test IDs")
    for rid in expected_ids:
        if list(predictions[rid].keys()) != ["text"]:
            raise AssertionError(f"ID {rid}: output object must contain only text")
        if word_count(predictions[rid]["text"]) != gen_cfg.target_words:
            raise AssertionError(f"ID {rid}: output must contain exactly {gen_cfg.target_words} words")

    zip_path = out_dir / "submission.zip"
    if zip_path.exists():
        zip_path.unlink()
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.write(pred_path, arcname="prediction.json")
    with zipfile.ZipFile(zip_path, "r") as zf:
        names = zf.namelist()
        if names != ["prediction.json"]:
            raise AssertionError(f"Bad ZIP structure: {names}")

    run_meta = {
        "architecture": "A_Dense_Flash_LegalGPT",
        "generation": asdict(gen_cfg),
        "repo_id": repo_id,
        "repo_revision": resolve_hf_revision(repo_id),
        "test_file": TEST_FILE,
        "n_test_examples": len(tests),
        "zip_contents": ["prediction.json"],
    }
    (out_dir / "submission_meta.json").write_text(json.dumps(run_meta, indent=2), encoding="utf-8")
    print(f"[submit] READY: {zip_path}")
    return zip_path


# ------------------------------ self test --------------------------------- #

def self_test() -> None:
    print("[selftest] testing model forward/cache/generation primitives")
    cfg = ModelConfig(vocab_size=256, max_seq_len=64, n_layers=2, d_model=64, n_heads=4, d_head=16, d_ff=128, gradient_checkpointing=False)
    model = LegalGPT(cfg)
    x = torch.randint(0, 256, (2, 16))
    y = torch.randint(0, 256, (2, 16))
    logits, loss, _ = model(x, targets=y)
    assert logits.shape == (2, 16, 256)
    assert loss is not None and torch.isfinite(loss)
    model.eval()
    p = torch.randint(0, 256, (1, 10))
    logits, _, cache = model(p, use_cache=True)
    assert cache is not None and len(cache) == 2
    t = torch.randint(0, 256, (1, 1))
    logits2, _, cache2 = model(t, past_kvs=cache, use_cache=True)
    assert logits2.shape == (1, 1, 256)
    assert cache2[0][0].shape[-2] == 11
    print(f"[selftest] PASS; tiny params={model.count_parameters():,}")


# ------------------------------- CLI -------------------------------------- #

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--mode", choices=["prepare", "train", "eval", "submit", "all", "selftest"], required=True)
    p.add_argument("--workdir", type=Path, default=Path("./runs/A_legalgpt"))
    p.add_argument("--force_prepare", action="store_true")

    # Data
    p.add_argument("--vocab_size", type=int, default=32768)
    p.add_argument("--tokenizer_train_chars", type=int, default=100_000_000)
    p.add_argument("--prepare_train_tokens", type=int, default=200_000_000)
    p.add_argument("--prepare_val_tokens", type=int, default=5_000_000)
    p.add_argument("--val_fraction", type=float, default=0.02)
    p.add_argument("--val_generation_examples", type=int, default=128)
    p.add_argument("--mix_mode", choices=["equal", "natural"], default="equal")

    # Model
    p.add_argument("--max_seq_len", type=int, default=2048)
    p.add_argument("--n_layers", type=int, default=12)
    p.add_argument("--d_model", type=int, default=768)
    p.add_argument("--n_heads", type=int, default=12)
    p.add_argument("--d_head", type=int, default=64)
    p.add_argument("--d_ff", type=int, default=2048)
    p.add_argument("--dropout", type=float, default=0.0)
    p.add_argument("--no_gradient_checkpointing", action="store_true")

    # Train
    p.add_argument("--micro_batch_size", type=int, default=1)
    p.add_argument("--grad_accum_steps", type=int, default=32)
    p.add_argument("--learning_rate", type=float, default=3e-4)
    p.add_argument("--weight_decay", type=float, default=0.1)
    p.add_argument("--target_train_tokens", type=int, default=200_000_000)
    p.add_argument("--eval_every", type=int, default=100)
    p.add_argument("--eval_batches", type=int, default=32)
    p.add_argument("--save_every", type=int, default=100)
    p.add_argument("--log_every", type=int, default=10)
    p.add_argument("--no_resume", action="store_true")

    # Generation
    p.add_argument("--strategy", choices=["greedy", "nucleus", "topk"], default="greedy")
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--top_p", type=float, default=0.90)
    p.add_argument("--top_k", type=int, default=50)
    p.add_argument("--repetition_penalty", type=float, default=1.05)
    p.add_argument("--target_words", type=int, default=128)
    p.add_argument("--max_new_tokens", type=int, default=384)
    p.add_argument("--eval_generation_limit", type=int, default=32)
    p.add_argument("--allow_local_bertscore", action="store_true")
    p.add_argument("--seed", type=int, default=781)
    return p


def configs_from_args(a):
    data_cfg = DataConfig(
        vocab_size=a.vocab_size,
        tokenizer_train_chars=a.tokenizer_train_chars,
        train_tokens=a.prepare_train_tokens,
        val_tokens=a.prepare_val_tokens,
        val_fraction=a.val_fraction,
        val_generation_examples=a.val_generation_examples,
        mix_mode=a.mix_mode,
        seed=a.seed,
    )
    model_cfg = ModelConfig(
        vocab_size=a.vocab_size,
        max_seq_len=a.max_seq_len,
        n_layers=a.n_layers,
        d_model=a.d_model,
        n_heads=a.n_heads,
        d_head=a.d_head,
        d_ff=a.d_ff,
        dropout=a.dropout,
        gradient_checkpointing=not a.no_gradient_checkpointing,
    )
    train_cfg = TrainConfig(
        micro_batch_size=a.micro_batch_size,
        grad_accum_steps=a.grad_accum_steps,
        learning_rate=a.learning_rate,
        weight_decay=a.weight_decay,
        target_train_tokens=a.target_train_tokens,
        eval_every=a.eval_every,
        eval_batches=a.eval_batches,
        save_every=a.save_every,
        log_every=a.log_every,
        seed=a.seed,
    )
    gen_cfg = GenerationConfig(
        strategy=a.strategy,
        temperature=a.temperature,
        top_p=a.top_p,
        top_k=a.top_k,
        repetition_penalty=a.repetition_penalty,
        target_words=a.target_words,
        max_new_tokens=a.max_new_tokens,
        seed=a.seed,
    )
    return data_cfg, model_cfg, train_cfg, gen_cfg


def main() -> None:
    args = build_parser().parse_args()
    args.workdir.mkdir(parents=True, exist_ok=True)
    seed_everything(args.seed)
    if args.mode == "selftest":
        self_test()
        return

    data_cfg, model_cfg, train_cfg, gen_cfg = configs_from_args(args)
    (args.workdir / "run_config.json").write_text(
        json.dumps({
            "data": asdict(data_cfg),
            "model": asdict(model_cfg),
            "train": asdict(train_cfg),
            "generation": asdict(gen_cfg),
            "repo_revision_at_start": resolve_hf_revision(data_cfg.repo_id),
        }, indent=2),
        encoding="utf-8",
    )

    if args.mode in ["prepare", "all"]:
        train_tokenizer(args.workdir, data_cfg, force=args.force_prepare)
        prepare_token_cache(args.workdir, data_cfg, force=args.force_prepare)

    if args.mode in ["train", "all"]:
        train(args.workdir, model_cfg, train_cfg, resume=not args.no_resume)

    if args.mode in ["eval", "all"]:
        evaluate_generation(
            args.workdir,
            gen_cfg,
            limit=args.eval_generation_limit,
            allow_local_bertscore=args.allow_local_bertscore,
        )

    if args.mode in ["submit", "all"]:
        make_submission(args.workdir, gen_cfg, data_cfg.repo_id)


if __name__ == "__main__":
    main()
