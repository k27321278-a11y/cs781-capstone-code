import gc
import hashlib
import json
import math
import os
import random
import time
import zipfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.utils.rnn import pad_sequence
from torch.utils.checkpoint import checkpoint
from tokenizers import Tokenizer
from sklearn.feature_extraction.text import TfidfVectorizer


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

    def validate(self):
        if self.n_heads * self.d_head != self.d_model or self.d_head % 2:
            raise ValueError("Invalid Phase 1 attention dimensions")
        if min(self.vocab_size, self.max_seq_len, self.n_layers, self.d_ff) < 1:
            raise ValueError("Invalid Phase 1 model dimensions")


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        values = x.float()
        return (values * torch.rsqrt(values.square().mean(-1, keepdim=True) + self.eps)).to(x.dtype) * self.weight


class RotaryEmbedding(nn.Module):
    def __init__(self, head_dim, max_seq_len, theta=10000.0):
        super().__init__()
        inv = 1.0 / theta ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim)
        frequencies = torch.outer(torch.arange(max_seq_len, dtype=torch.float32), inv)
        self.register_buffer("cos", frequencies.cos(), persistent=False)
        self.register_buffer("sin", frequencies.sin(), persistent=False)

    def apply_rotary(self, x, offset=0):
        t = x.size(-2)
        if offset + t > self.cos.size(0):
            raise ValueError("Phase 1 RoPE context exhausted")
        cos = self.cos[offset:offset + t].to(dtype=x.dtype)[None, None]
        sin = self.sin[offset:offset + t].to(dtype=x.dtype)[None, None]
        even, odd = x[..., 0::2], x[..., 1::2]
        return torch.stack((even * cos - odd * sin, even * sin + odd * cos), dim=-1).flatten(-2)


class CausalMHA(nn.Module):
    def __init__(self, cfg, rope):
        super().__init__()
        self.cfg, self.rope = cfg, rope
        self.wqkv = nn.Linear(cfg.d_model, 3 * cfg.d_model, bias=False)
        self.wo = nn.Linear(cfg.d_model, cfg.d_model, bias=False)

    def forward(self, x, past_kv=None, use_cache=False):
        b, t, d = x.shape
        q, k, v = [a.view(b, t, self.cfg.n_heads, self.cfg.d_head).transpose(1, 2)
                   for a in self.wqkv(x).chunk(3, dim=-1)]
        offset = 0 if past_kv is None else past_kv[0].size(-2)
        q, k = self.rope.apply_rotary(q, offset), self.rope.apply_rotary(k, offset)
        if past_kv is not None:
            if t != 1:
                raise ValueError("Cached Phase 1 decoding accepts one token")
            k, v = torch.cat((past_kv[0], k), dim=-2), torch.cat((past_kv[1], v), dim=-2)
        output = F.scaled_dot_product_attention(q, k, v, is_causal=past_kv is None,
                                                dropout_p=self.cfg.dropout if self.training else 0.0)
        output = self.wo(output.transpose(1, 2).contiguous().view(b, t, d))
        return output, (k, v) if use_cache else None


class SwiGLU(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.w1 = nn.Linear(cfg.d_model, cfg.d_ff, bias=False)
        self.w3 = nn.Linear(cfg.d_model, cfg.d_ff, bias=False)
        self.w2 = nn.Linear(cfg.d_ff, cfg.d_model, bias=False)

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class TransformerBlock(nn.Module):
    def __init__(self, cfg, rope):
        super().__init__()
        self.attn_norm, self.ffn_norm = RMSNorm(cfg.d_model), RMSNorm(cfg.d_model)
        self.attn, self.ffn = CausalMHA(cfg, rope), SwiGLU(cfg)

    def forward(self, x, past_kv=None, use_cache=False):
        output, cache = self.attn(self.attn_norm(x), past_kv, use_cache)
        x = x + output
        return x + self.ffn(self.ffn_norm(x)), cache


class LegalGPT(nn.Module):
    """State-dict-compatible A0 LegalGPT definition, embedded for standalone Kaggle execution."""
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
        std = 0.02 / math.sqrt(2 * cfg.n_layers)
        for block in self.blocks:
            nn.init.normal_(block.attn.wo.weight, std=std)
            nn.init.normal_(block.ffn.w2.weight, std=std)

    @staticmethod
    def _init_weights(module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, std=0.02)


SEED = 781
WINDOW_TOKENS = 256
WINDOWS_PER_DOC = 4
PROJ_DIM = 256
TRAIN_BATCH_QUERIES = 2
TRAIN_EPOCHS = 2
TRAIN_LR = 2e-5
LORMA_RANK = 8
LORMA_ALPHA = 16
KD_TEMPERATURE = 2.0

DATA_ROOT = None
PARENT_ROOT = Path("/kaggle/input/datasets/krshnakarthikeyagn/cs781-phase4-private-inputs-20261022/Phase3_A0_Input")
OUT = Path("/kaggle/working/phase7_output")
CKPT_DIR = OUT / "checkpoints"
OUT.mkdir(parents=True, exist_ok=True)
CKPT_DIR.mkdir(parents=True, exist_ok=True)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
AMP = DEVICE.type == "cuda"
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if AMP:
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cuda.matmul.allow_tf32 = True


def log(*args):
    print(time.strftime("[%H:%M:%S]"), *args, flush=True)


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def record_text(row):
    return " ".join(str(x) for x in row.get("text", []) if x is not None)


def load_split(name):
    folder = {"train": "Train", "dev": "Dev", "test": "Test"}[name]
    qf = {"train": "train.json", "dev": "dev.json", "test": "test.json"}[name]
    cf = {"train": "train_candidates.json", "dev": "dev_candidates.json", "test": "test_candidates.json"}[name]
    return (read_json(DATA_ROOT / folder / qf), read_json(DATA_ROOT / folder / cf))


def locate_data_root():
    input_root = Path("/kaggle/input")
    def complete_roots(base):
        roots = []
        for train_file in base.rglob("train.json"):
            if train_file.parent.name != "Train":
                continue
            root = train_file.parents[1]
            required = [root / "Train/train_candidates.json", root / "Dev/dev.json",
                        root / "Dev/dev_candidates.json", root / "Test/test.json",
                        root / "Test/test_candidates.json"]
            if all(p.is_file() for p in required):
                roots.append(root)
        return roots

    roots = complete_roots(input_root)
    if not roots:
        archives = list(input_root.rglob("Phase_7_PCR.zip"))
        for archive in archives:
            with zipfile.ZipFile(archive) as zf:
                unsafe = [name for name in zf.namelist()
                          if Path(name).is_absolute() or ".." in Path(name).parts]
                if unsafe:
                    raise ValueError(f"Unsafe paths in official Phase 7 archive: {unsafe[:3]}")
                zf.extractall(OUT / "input_unpacked")
        roots = complete_roots(OUT / "input_unpacked")
    if len(roots) != 1:
        visible = [str(p) for p in input_root.iterdir()] if input_root.exists() else []
        raise FileNotFoundError(f"Expected one complete Phase 7 data root; found {len(roots)}. Input roots: {visible}")
    root = roots[0]
    required = [root / "Train/train.json", root / "Train/train_candidates.json",
                root / "Dev/dev.json", root / "Dev/dev_candidates.json",
                root / "Test/test.json", root / "Test/test_candidates.json"]
    missing = [str(p) for p in required if not p.is_file()]
    if missing:
        raise FileNotFoundError(f"Phase 7 bundle is incomplete: {missing}")
    return root


def metric_at7(pred, queries):
    tp = sum(len(set(pred[qid]) & set(map(str, row["relevant_candidates"]))) for qid, row in queries.items())
    n_pred = sum(len(pred[qid]) for qid in queries)
    n_gold = sum(len(row["relevant_candidates"]) for row in queries.values())
    fp, fn = n_pred - tp, n_gold - tp
    p = tp / max(1, tp + fp)
    r = tp / max(1, tp + fn)
    f1 = 2 * p * r / max(1e-12, p + r)
    return {"micro_f1_at_7": f1, "micro_precision_at_7": p, "micro_recall_at_7": r,
            "tp": int(tp), "fp": int(fp), "fn": int(fn), "queries": len(queries)}


def top7_predictions(scores, qids, cids):
    out = {}
    for i, qid in enumerate(qids):
        order = np.argsort(-scores[i], kind="stable")[:7]
        out[qid] = [str(cids[j]) for j in order]
    return out


def row_zscore(a):
    a = np.asarray(a, dtype=np.float32)
    mean = a.mean(axis=1, keepdims=True)
    std = a.std(axis=1, keepdims=True)
    return (a - mean) / np.maximum(std, 1e-6)


def tune_blend(dense, lexical, qids, cids, queries):
    dz, lz = row_zscore(dense), row_zscore(lexical)
    best = None
    for weight in np.linspace(0.0, 1.0, 11):
        pred = top7_predictions(weight * dz + (1.0 - weight) * lz, qids, cids)
        m = metric_at7(pred, queries)
        item = {"dense_weight": float(weight), **m}
        if best is None or item["micro_f1_at_7"] > best["metrics"]["micro_f1_at_7"]:
            best = {"metrics": item, "predictions": pred, "scores": weight * dz + (1.0 - weight) * lz}
    return best


def build_lexical(queries, candidates, label):
    qids, cids = list(queries), list(candidates)
    log("TF-IDF", label, "queries", len(qids), "candidates", len(cids))
    vectorizer = TfidfVectorizer(
        max_features=300_000, ngram_range=(1, 2), min_df=2, max_df=0.98,
        dtype=np.float32, sublinear_tf=True, strip_accents="unicode",
        token_pattern=r"(?u)\b\w+\b",
    )
    dmat = vectorizer.fit_transform([record_text(candidates[cid]) for cid in cids])
    qmat = vectorizer.transform([record_text(queries[qid]) for qid in qids])
    scores = (qmat @ dmat.T).toarray().astype(np.float32, copy=False)
    log("TF-IDF ready", label, "features", dmat.shape[1], "nnz", dmat.nnz)
    return {"qids": qids, "cids": cids, "scores": scores}


def make_hard_negatives(train_q, train_c, lexical):
    hard = {}
    for i, qid in enumerate(lexical["qids"]):
        gold = set(map(str, train_q[qid]["relevant_candidates"]))
        ranked = np.argsort(-lexical["scores"][i], kind="stable")
        hard[qid] = [lexical["cids"][j] for j in ranked if lexical["cids"][j] not in gold][:64]
    return hard


def prepare_token_cache(records, tokenizer, label):
    bos = tokenizer.token_to_id("<bos>")
    eos = tokenizer.token_to_id("<eos>")
    unk = tokenizer.token_to_id("<unk>")
    max_content = WINDOW_TOKENS - (int(bos is not None) + int(eos is not None))
    cache, tok_lengths = {}, []
    start_time = time.time()
    for idx, (doc_id, row) in enumerate(records.items()):
        raw = tokenizer.encode(record_text(row), add_special_tokens=False).ids
        if not raw:
            raw = [unk if unk is not None else 0]
        tok_lengths.append(len(raw))
        if len(raw) <= max_content:
            starts = [0]
        else:
            final = len(raw) - max_content
            starts = sorted(set(int(round(x)) for x in np.linspace(0, final, WINDOWS_PER_DOC)))
        windows = []
        for start in starts:
            content = raw[start:start + max_content]
            seq = ([bos] if bos is not None else []) + content + ([eos] if eos is not None else [])
            windows.append(seq)
        cache[str(doc_id)] = windows
        if (idx + 1) % 1000 == 0:
            log("tokenized", label, idx + 1, "elapsed_s", round(time.time() - start_time))
    log("tokenized", label, len(cache), "mean_tokens", round(float(np.mean(tok_lengths)), 1),
        "p95_tokens", int(np.percentile(tok_lengths, 95)), "max_tokens", int(max(tok_lengths)))
    return cache


class LoRMALinear(nn.Module):
    """LoRMA+ post-mode: W(I + BA), with the identity residual at initialization."""
    def __init__(self, base, rank=LORMA_RANK, alpha=LORMA_ALPHA):
        super().__init__()
        self.base = base
        self.scale = alpha / rank
        self.drop = nn.Dropout(0.05)
        self.A = nn.Parameter(base.weight.new_empty(rank, base.in_features))
        self.B = nn.Parameter(base.weight.new_zeros(base.in_features, rank))
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))
        self.base.requires_grad_(False)

    def forward(self, x):
        update = F.linear(F.linear(self.drop(x), self.A), self.B) * self.scale
        return self.base(x + update)


class RetrievalModel(nn.Module):
    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone
        width = backbone.cfg.d_model
        self.projection = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, PROJ_DIM, bias=False))

    def forward_windows(self, input_ids, mask):
        x = self.backbone.tok_embeddings(input_ids)
        for block in self.backbone.blocks:
            if self.training and torch.is_grad_enabled():
                x = checkpoint(lambda value, b=block: b(value)[0], x, use_reentrant=False)
            else:
                x = block(x)[0]
        x = self.backbone.norm(x)
        weights = mask.to(x.dtype).unsqueeze(-1)
        pooled = (x * weights).sum(1) / weights.sum(1).clamp_min(1.0)
        return F.normalize(self.projection(pooled).float(), dim=-1)


def inject_lorma(backbone):
    count = 0
    for block in backbone.blocks:
        for parent, names in ((block.attn, ("wqkv", "wo")), (block.ffn, ("w1", "w2", "w3"))):
            for name in names:
                setattr(parent, name, LoRMALinear(getattr(parent, name)))
                count += 1
    return count


def build_model(method, base_state, base_cfg, device, student=False):
    cfg = dict(base_cfg)
    if student:
        cfg["n_layers"] = max(1, int(base_cfg["n_layers"]) // 2)
    backbone = LegalGPT(ModelConfig(**cfg))
    if student:
        sub = {}
        for key, value in base_state.items():
            if key.startswith("blocks."):
                pieces = key.split(".", 2)
                old_i = int(pieces[1])
                if old_i % 2 == 0:
                    sub[f"blocks.{old_i // 2}.{pieces[2]}"] = value
            else:
                sub[key] = value
        backbone.load_state_dict(sub, strict=True)
    else:
        backbone.load_state_dict(base_state, strict=True)

    if method in ("lorma", "partial"):
        backbone.requires_grad_(False)
    if method == "lorma":
        n = inject_lorma(backbone)
        log("LoRMA+ post-mode modules", n, "rank", LORMA_RANK)
    elif method == "partial":
        for block in backbone.blocks[-4:]:
            block.requires_grad_(True)
        backbone.norm.requires_grad_(True)
    elif method == "full" or method == "kd":
        backbone.requires_grad_(True)
    else:
        raise ValueError(f"unknown method {method}")
    model = RetrievalModel(backbone).to(device)
    return model


def pack_sequences(sequences, pad_id, device):
    max_len = max(len(x) for x in sequences)
    ids = torch.full((len(sequences), max_len), pad_id, dtype=torch.long, device=device)
    mask = torch.zeros((len(sequences), max_len), dtype=torch.bool, device=device)
    for i, seq in enumerate(sequences):
        ids[i, :len(seq)] = torch.as_tensor(seq, dtype=torch.long, device=device)
        mask[i, :len(seq)] = True
    return ids, mask


def encode_doc_batch(model, doc_ids, token_cache, pad_id, device, window_batch=12):
    by_doc = [token_cache[str(doc_id)] for doc_id in doc_ids]
    seqs = [seq for windows in by_doc for seq in windows]
    bounds, cursor = [], 0
    for windows in by_doc:
        bounds.append((cursor, cursor + len(windows)))
        cursor += len(windows)
    embs = []
    for start in range(0, len(seqs), window_batch):
        ids, mask = pack_sequences(seqs[start:start + window_batch], pad_id, device)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=AMP):
            z = model.forward_windows(ids, mask)
        embs.append(z)
    flat = torch.cat(embs, dim=0)
    per_doc = [flat[a:b] for a, b in bounds]
    lengths = torch.tensor([x.shape[0] for x in per_doc], device=device)
    padded = pad_sequence(per_doc, batch_first=True)
    win_mask = torch.arange(padded.shape[1], device=device)[None, :] < lengths[:, None]
    return padded, win_mask


@torch.no_grad()
def encode_all(model, ids, token_cache, pad_id, device, batch_docs=12):
    model.eval()
    all_emb, all_mask = [], []
    start = time.time()
    for n, a in enumerate(range(0, len(ids), batch_docs)):
        batch = ids[a:a + batch_docs]
        z, m = encode_doc_batch(model, batch, token_cache, pad_id, device, window_batch=12)
        all_emb.append(z.detach().float().cpu().numpy())
        all_mask.append(m.detach().cpu().numpy())
        if (n + 1) % 100 == 0:
            log("embedded docs", a + len(batch), "of", len(ids), "elapsed_s", round(time.time() - start))
    max_windows = max(x.shape[1] for x in all_emb)
    emb = np.concatenate(
        [np.pad(x, ((0, 0), (0, max_windows - x.shape[1]), (0, 0))) for x in all_emb],
        axis=0)
    mask = np.concatenate(
        [np.pad(x, ((0, 0), (0, max_windows - x.shape[1])), constant_values=False)
         for x in all_mask], axis=0)
    return emb, mask


def late_scores_torch(q, qm, d, dm):
    sims = torch.einsum("qwd,cvd->qcwv", q, d)
    sims = sims.masked_fill(~dm[None, :, None, :], -1e4)
    per_q_window = sims.max(-1).values
    per_q_window = per_q_window.masked_fill(~qm[:, None, :], -1e4)
    k = min(2, per_q_window.shape[-1])
    return per_q_window.topk(k, dim=-1).values.mean(-1)


@torch.no_grad()
def all_pair_scores(qemb, qmask, demb, dmask, device, q_batch=8, c_batch=256):
    out = np.empty((len(qemb), len(demb)), dtype=np.float32)
    for qa in range(0, len(qemb), q_batch):
        qb = min(len(qemb), qa + q_batch)
        q = torch.as_tensor(qemb[qa:qb], device=device)
        qm = torch.as_tensor(qmask[qa:qb], device=device)
        for ca in range(0, len(demb), c_batch):
            cb = min(len(demb), ca + c_batch)
            d = torch.as_tensor(demb[ca:cb], device=device)
            dm = torch.as_tensor(dmask[ca:cb], device=device)
            out[qa:qb, ca:cb] = late_scores_torch(q, qm, d, dm).float().cpu().numpy()
    return out


def evaluate_model(model, queries, candidates, qcache, ccache, lexical, pad_id, device, label):
    qids, cids = list(queries), list(candidates)
    log("dev encode start", label)
    qemb, qmask = encode_all(model, qids, qcache, pad_id, device)
    demb, dmask = encode_all(model, cids, ccache, pad_id, device)
    dense = all_pair_scores(qemb, qmask, demb, dmask, device)
    selection = tune_blend(dense, lexical["scores"], qids, cids, queries)
    log("dev result", label, json.dumps(selection["metrics"]))
    return selection, dense


def trainable_state(model, method):
    full = model.state_dict()
    if method in ("full", "kd"):
        return {k: v.detach().cpu() for k, v in full.items()}
    names = {name for name, p in model.named_parameters() if p.requires_grad}
    return {k: v.detach().cpu() for k, v in full.items() if k in names}


def save_checkpoint(model, method, path, epoch, score, base_hash, cfg):
    payload = {"method": method, "epoch": int(epoch), "dev_f1_at_7": float(score),
               "base_sha256": base_hash, "model_config": cfg, "state": trainable_state(model, method)}
    tmp = str(path) + ".tmp"
    torch.save(payload, tmp)
    os.replace(tmp, path)


def load_experiment_model(method, path, base_state, base_cfg, device, student=False):
    model = build_model(method, base_state, base_cfg, device, student=student)
    data = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(data["state"], strict=(method in ("full", "kd")))
    return model


def train_supervised(method, train_q, train_c, hard, train_qcache, train_ccache,
                     dev_q, dev_c, dev_qcache, dev_ccache, dev_lex, base_state,
                     base_cfg, base_hash, pad_id, device, epochs=TRAIN_EPOCHS):
    model = build_model(method, base_state, base_cfg, device)
    trainable = [p for p in model.parameters() if p.requires_grad]
    trainable_count = sum(p.numel() for p in trainable)
    total_count = sum(p.numel() for p in model.parameters())
    log(method, "parameters trainable/total", trainable_count, total_count)
    optimizer = torch.optim.AdamW(trainable,
                                  lr=TRAIN_LR, weight_decay=0.01)
    scaler = torch.cuda.amp.GradScaler(enabled=AMP)
    qids = [qid for qid, row in train_q.items()
            if row.get("relevant_candidates") and qid in train_qcache]
    ckpt = CKPT_DIR / f"{method}_best.pt"
    best = {"metrics": {"micro_f1_at_7": -1.0}, "predictions": None, "scores": None}
    started = time.time()
    for epoch in range(1, epochs + 1):
        model.train()
        random.shuffle(qids)
        losses = []
        for step0 in range(0, len(qids), TRAIN_BATCH_QUERIES):
            batch_q = qids[step0:step0 + TRAIN_BATCH_QUERIES]
            q_doc_ids, d_doc_ids, targets = [], [], []
            for bi, qid in enumerate(batch_q):
                golds = [str(x) for x in train_q[qid]["relevant_candidates"] if str(x) in train_c]
                pos = random.choice(golds)
                negatives = [x for x in hard.get(qid, []) if x not in set(golds)]
                if len(negatives) >= 2:
                    sampled = random.sample(negatives[:32], 2)
                else:
                    pool = [x for x in train_c if x not in set(golds)]
                    sampled = random.sample(pool, 2)
                q_doc_ids.append(qid)
                targets.append(len(d_doc_ids))
                d_doc_ids.extend([pos] + sampled)
            optimizer.zero_grad(set_to_none=True)
            qemb, qmask = encode_doc_batch(model, q_doc_ids, train_qcache, pad_id, device)
            demb, dmask = encode_doc_batch(model, d_doc_ids, train_ccache, pad_id, device)
            scores = late_scores_torch(qemb, qmask, demb, dmask) / 0.07
            target = torch.as_tensor(targets, device=device)
            loss = F.cross_entropy(scores, target)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            scaler.step(optimizer)
            scaler.update()
            losses.append(float(loss.detach().cpu()))
            if (step0 // TRAIN_BATCH_QUERIES + 1) % 100 == 0:
                log(method, "epoch", epoch, "step", step0 + len(batch_q), "loss", round(float(np.mean(losses[-100:])), 4),
                    "elapsed_min", round((time.time() - started) / 60, 1))
        selection, dense = evaluate_model(model, dev_q, dev_c, dev_qcache, dev_ccache,
                                          dev_lex, pad_id, device, f"{method}_epoch{epoch}")
        if selection["metrics"]["micro_f1_at_7"] > best["metrics"]["micro_f1_at_7"]:
            best = selection
            save_checkpoint(model, method, ckpt, epoch, best["metrics"]["micro_f1_at_7"], base_hash, base_cfg)
            best_dense = dense.copy()
            best_epoch = epoch
        log(method, "epoch", epoch, "mean_loss", round(float(np.mean(losses)), 5),
            "best_dev_f1", round(float(best["metrics"]["micro_f1_at_7"]), 5))
    del model, optimizer, scaler
    gc.collect()
    if AMP:
        torch.cuda.empty_cache()
    return {"method": method, "checkpoint": str(ckpt), "best_epoch": best_epoch,
            "dev_metrics": best["metrics"], "dev_predictions": best["predictions"],
            "dev_scores": best["scores"], "dev_dense_scores": best_dense,
            "trainable_parameters": trainable_count, "total_parameters": total_count,
            "elapsed_seconds": time.time() - started}


def build_student_base(base_state, base_cfg, device):
    model = build_model("kd", base_state, base_cfg, device, student=True)
    return model


def teacher_targets(teacher_model, train_q, train_c, train_qcache, train_ccache,
                    pad_id, device, train_lex):
    qids, cids = list(train_q), list(train_c)
    log("KD teacher embeddings", len(qids), "queries", len(cids), "candidate cases")
    qemb, qmask = encode_all(teacher_model, qids, train_qcache, pad_id, device)
    demb, dmask = encode_all(teacher_model, cids, train_ccache, pad_id, device)
    scores = all_pair_scores(qemb, qmask, demb, dmask, device, q_batch=8, c_batch=256)
    cindex = {cid: i for i, cid in enumerate(cids)}
    target_sets = {}
    for i, qid in enumerate(qids):
        golds = [str(x) for x in train_q[qid]["relevant_candidates"] if str(x) in cindex]
        sorted_golds = sorted(golds, key=lambda cid: scores[i, cindex[cid]], reverse=True)[:7]
        top = [cids[j] for j in np.argsort(-scores[i], kind="stable")[:12]]
        target_ids = list(dict.fromkeys(top + sorted_golds))
        logit = np.asarray([scores[i, cindex[cid]] for cid in target_ids], dtype=np.float32)
        prob = np.exp((logit - logit.max()) / KD_TEMPERATURE)
        prob /= prob.sum()
        target_sets[qid] = {"candidates": target_ids, "teacher_prob": prob.tolist(),
                            "positive_ids": sorted_golds}
    return target_sets


def train_student_kd(target_sets, train_q, train_c, train_qcache, train_ccache,
                     dev_q, dev_c, dev_qcache, dev_ccache, dev_lex, base_state,
                     base_cfg, base_hash, pad_id, device, epochs=1):
    model = build_student_base(base_state, base_cfg, device)
    trainable = list(model.parameters())
    trainable_count = sum(p.numel() for p in trainable)
    total_count = trainable_count
    log("kd", "parameters trainable/total", trainable_count, total_count)
    optimizer = torch.optim.AdamW(trainable, lr=3e-5, weight_decay=0.01)
    scaler = torch.cuda.amp.GradScaler(enabled=AMP)
    qids = list(target_sets)
    ckpt = CKPT_DIR / "kd_student_best.pt"
    best = {"metrics": {"micro_f1_at_7": -1.0}, "predictions": None, "scores": None}
    started = time.time()
    for epoch in range(1, epochs + 1):
        model.train()
        random.shuffle(qids)
        losses = []
        for step, qid in enumerate(qids, 1):
            row = target_sets[qid]
            doc_ids = row["candidates"]
            positives = set(row["positive_ids"])
            qemb, qmask = encode_doc_batch(model, [qid], train_qcache, pad_id, device)
            demb, dmask = encode_doc_batch(model, doc_ids, train_ccache, pad_id, device, window_batch=12)
            scores = late_scores_torch(qemb, qmask, demb, dmask)[0]
            probs = torch.as_tensor(row["teacher_prob"], dtype=torch.float32, device=device)
            logp = F.log_softmax(scores / KD_TEMPERATURE, dim=-1)
            kd = F.kl_div(logp, probs, reduction="sum") * KD_TEMPERATURE ** 2
            pos_idx = torch.as_tensor([i for i, cid in enumerate(doc_ids) if cid in positives],
                                      dtype=torch.long, device=device)
            sup = -(torch.logsumexp(scores.index_select(0, pos_idx), 0) -
                    torch.logsumexp(scores, 0))
            loss = 0.75 * kd + 0.25 * sup
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            scaler.step(optimizer)
            scaler.update()
            losses.append(float(loss.detach().cpu()))
            if step % 100 == 0:
                log("kd", "epoch", epoch, "step", step, "loss", round(float(np.mean(losses[-100:])), 4),
                    "elapsed_min", round((time.time() - started) / 60, 1))
        selection, dense = evaluate_model(model, dev_q, dev_c, dev_qcache, dev_ccache,
                                          dev_lex, pad_id, device, f"kd_epoch{epoch}")
        if selection["metrics"]["micro_f1_at_7"] > best["metrics"]["micro_f1_at_7"]:
            best = selection
            cfg_student = dict(base_cfg)
            cfg_student["n_layers"] = max(1, int(base_cfg["n_layers"]) // 2)
            save_checkpoint(model, "kd", ckpt, epoch, best["metrics"]["micro_f1_at_7"], base_hash, cfg_student)
            best_dense = dense.copy()
            best_epoch = epoch
        log("kd", "epoch", epoch, "mean_loss", round(float(np.mean(losses)), 5),
            "best_dev_f1", round(float(best["metrics"]["micro_f1_at_7"]), 5))
    del model, optimizer, scaler
    gc.collect()
    if AMP:
        torch.cuda.empty_cache()
    return {"method": "kd", "checkpoint": str(ckpt), "best_epoch": best_epoch,
            "dev_metrics": best["metrics"], "dev_predictions": best["predictions"],
            "dev_scores": best["scores"], "dev_dense_scores": best_dense,
            "trainable_parameters": trainable_count, "total_parameters": total_count,
            "elapsed_seconds": time.time() - started}


def main():
    global DATA_ROOT
    DATA_ROOT = locate_data_root()
    if not DATA_ROOT.is_dir() or not PARENT_ROOT.is_dir():
        raise FileNotFoundError(f"Missing Kaggle input. DATA={DATA_ROOT.exists()} PARENT={PARENT_ROOT.exists()}")
    log("device", str(DEVICE), "gpu", torch.cuda.get_device_name(0) if AMP else "none",
        "cuda_mem_gb", round(torch.cuda.get_device_properties(0).total_memory / 1e9, 2) if AMP else 0)
    train_q, train_c = load_split("train")
    dev_q, dev_c = load_split("dev")
    test_q, test_c = load_split("test")
    log("split counts", json.dumps({"train_queries": len(train_q), "train_candidates": len(train_c),
        "dev_queries": len(dev_q), "dev_candidates": len(dev_c), "test_queries": len(test_q),
        "test_candidates": len(test_c), "train_relevant": sum(len(x["relevant_candidates"]) for x in train_q.values()),
        "dev_relevant": sum(len(x["relevant_candidates"]) for x in dev_q.values())}))

    parent = PARENT_ROOT / "checkpoint.pt"
    tokenizer_path = PARENT_ROOT / "tokenizer.json"
    parent_sha = sha256_file(parent)
    expected_sha = "1e8aca9800b87e952ed82e54307dd874220845d6a8545d2568bb743993cfb9dc"
    if parent_sha != expected_sha:
        raise ValueError(f"Phase1 A0 parent checkpoint hash mismatch: {parent_sha}")
    payload = torch.load(parent, map_location="cpu", weights_only=False)
    if "model" not in payload or "metadata" not in payload:
        raise ValueError("Phase1 parent checkpoint lacks model/metadata")
    base_state = payload["model"]
    parent_meta = dict(payload["metadata"])
    base_cfg = dict(parent_meta["model_config"])
    base_step = payload.get("state", {}).get("step")
    if parent_meta.get("tokenizer_sha256") and parent_meta["tokenizer_sha256"] != sha256_file(tokenizer_path):
        raise ValueError("Phase1 tokenizer hash mismatch")
    log("verified Phase1 parent", parent_sha, "config", json.dumps(base_cfg),
        "steps", parent_meta.get("parent_state", {}).get("step", base_step))
    del payload
    gc.collect()
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    pad_id = tokenizer.token_to_id("<pad>")
    if pad_id is None:
        raise ValueError("Own Legal LM tokenizer has no <pad> token")
    bos_id, eos_id = tokenizer.token_to_id("<bos>"), tokenizer.token_to_id("<eos>")
    log("token IDs", json.dumps({"pad": pad_id, "bos": bos_id, "eos": eos_id,
                                  "vocab": tokenizer.get_vocab_size()}))

    t0 = time.time()
    train_qcache = prepare_token_cache(train_q, tokenizer, "train_queries")
    train_ccache = prepare_token_cache(train_c, tokenizer, "train_candidates")
    dev_qcache = prepare_token_cache(dev_q, tokenizer, "dev_queries")
    dev_ccache = prepare_token_cache(dev_c, tokenizer, "dev_candidates")
    test_qcache = prepare_token_cache(test_q, tokenizer, "test_queries")
    test_ccache = prepare_token_cache(test_c, tokenizer, "test_candidates")
    log("all tokenization seconds", round(time.time() - t0, 1))
    del tokenizer

    train_lex = build_lexical(train_q, train_c, "train")
    dev_lex = build_lexical(dev_q, dev_c, "dev")
    test_lex = build_lexical(test_q, test_c, "test")
    train_hard = make_hard_negatives(train_q, train_c, train_lex)
    lexical_dev = top7_predictions(dev_lex["scores"], dev_lex["qids"], dev_lex["cids"])
    lexical_metrics = metric_at7(lexical_dev, dev_q)
    log("lexical baseline dev", json.dumps(lexical_metrics))
    del train_lex
    gc.collect()

    results = {}
    teacher_candidates = []
    for method in ("full", "lorma", "partial"):
        result = train_supervised(method, train_q, train_c, train_hard,
            train_qcache, train_ccache, dev_q, dev_c, dev_qcache, dev_ccache,
            dev_lex, base_state, base_cfg, parent_sha, pad_id, DEVICE)
        results[method] = result
        teacher_candidates.append(method)

    teacher_method = max(teacher_candidates, key=lambda name: results[name]["dev_metrics"]["micro_f1_at_7"])
    log("KD teacher selected by dev", teacher_method,
        "dev_f1", results[teacher_method]["dev_metrics"]["micro_f1_at_7"])
    teacher = load_experiment_model(teacher_method, results[teacher_method]["checkpoint"],
                                    base_state, base_cfg, DEVICE)
    targets = teacher_targets(teacher, train_q, train_c, train_qcache, train_ccache,
                             pad_id, DEVICE, train_lex=None)
    del teacher
    gc.collect()
    if AMP:
        torch.cuda.empty_cache()
    kd_result = train_student_kd(targets, train_q, train_c, train_qcache, train_ccache,
        dev_q, dev_c, dev_qcache, dev_ccache, dev_lex, base_state, base_cfg,
        parent_sha, pad_id, DEVICE, epochs=1)
    results["kd"] = kd_result
    del targets

    # Compare all adapted systems and one two-model ensemble on dev.
    best_single = max(results, key=lambda name: results[name]["dev_metrics"]["micro_f1_at_7"])
    single_metric = results[best_single]["dev_metrics"]["micro_f1_at_7"]
    second = max((name for name in results if name != best_single),
                 key=lambda name: results[name]["dev_metrics"]["micro_f1_at_7"])
    ensemble = None
    ens_dz_a = row_zscore(results[best_single]["dev_dense_scores"])
    ens_dz_b = row_zscore(results[second]["dev_dense_scores"])
    lex_z = row_zscore(dev_lex["scores"])
    for ea in np.linspace(0.0, 1.0, 6):
        dense_combo = ea * ens_dz_a + (1.0 - ea) * ens_dz_b
        for lexical_weight in np.linspace(0.0, 1.0, 6):
            scores = (1.0 - lexical_weight) * dense_combo + lexical_weight * lex_z
            pred = top7_predictions(scores, dev_lex["qids"], dev_lex["cids"])
            metric = metric_at7(pred, dev_q)
            item = {"metrics": {**metric, "method_a": best_single, "method_b": second,
                                "method_a_weight": float(ea), "lexical_weight": float(lexical_weight)},
                    "predictions": pred, "method_a": best_single, "method_b": second,
                    "method_a_weight": float(ea), "lexical_weight": float(lexical_weight)}
            if ensemble is None or item["metrics"]["micro_f1_at_7"] > ensemble["metrics"]["micro_f1_at_7"]:
                ensemble = item
    if ensemble["metrics"]["micro_f1_at_7"] > single_metric:
        chosen = {"type": "ensemble", **ensemble}
    else:
        chosen = {"type": "single", "method": best_single,
                  "metrics": results[best_single]["dev_metrics"],
                  "predictions": results[best_single]["dev_predictions"]}
    log("validation-selected system", json.dumps({k: v for k, v in chosen.items() if k != "predictions"}))

    # Use the selected dev configuration for the official unlabeled test pool.
    if chosen["type"] == "single":
        test_model = load_experiment_model(chosen["method"], results[chosen["method"]]["checkpoint"],
                                           base_state, base_cfg, DEVICE, student=chosen["method"] == "kd")
        tqemb, tqmask = encode_all(test_model, list(test_q), test_qcache, pad_id, DEVICE)
        tdemb, tdmask = encode_all(test_model, list(test_c), test_ccache, pad_id, DEVICE)
        test_dense = all_pair_scores(tqemb, tqmask, tdemb, tdmask, DEVICE)
        alpha = float(results[chosen["method"]]["dev_metrics"]["dense_weight"])
        test_scores = alpha * row_zscore(test_dense) + (1.0 - alpha) * row_zscore(test_lex["scores"])
        del test_model
    else:
        test_a = load_experiment_model(chosen["method_a"], results[chosen["method_a"]]["checkpoint"],
                                       base_state, base_cfg, DEVICE, student=chosen["method_a"] == "kd")
        test_b = load_experiment_model(chosen["method_b"], results[chosen["method_b"]]["checkpoint"],
                                       base_state, base_cfg, DEVICE, student=chosen["method_b"] == "kd")
        qa, qam = encode_all(test_a, list(test_q), test_qcache, pad_id, DEVICE)
        da, dam = encode_all(test_a, list(test_c), test_ccache, pad_id, DEVICE)
        qb, qbm = encode_all(test_b, list(test_q), test_qcache, pad_id, DEVICE)
        db, dbm = encode_all(test_b, list(test_c), test_ccache, pad_id, DEVICE)
        sa, sb = all_pair_scores(qa, qam, da, dam, DEVICE), all_pair_scores(qb, qbm, db, dbm, DEVICE)
        dense_combo = chosen["method_a_weight"] * row_zscore(sa) + (1.0 - chosen["method_a_weight"]) * row_zscore(sb)
        lexical_weight = chosen["lexical_weight"]
        test_scores = (1.0 - lexical_weight) * dense_combo + lexical_weight * row_zscore(test_lex["scores"])
        del test_a, test_b

    prediction = top7_predictions(test_scores, list(test_q), list(test_c))
    if set(prediction) != set(test_q):
        raise ValueError("Test query coverage mismatch")
    test_candidate_ids = set(map(str, test_c))
    for qid, ids in prediction.items():
        if len(ids) != 7 or len(set(ids)) != 7 or not set(ids) <= test_candidate_ids:
            raise ValueError(f"Invalid ranked candidate list for {qid}")
    with open(OUT / "prediction.json", "w", encoding="utf-8") as f:
        json.dump(prediction, f, ensure_ascii=False, separators=(",", ":"))
    with zipfile.ZipFile(OUT / "CS781_Phase7_PCR_Submission.zip", "w", compression=zipfile.ZIP_DEFLATED) as z:
        z.write(OUT / "prediction.json", arcname="prediction.json")

    method_summary = {}
    for name, row in results.items():
        method_summary[name] = {"best_epoch": row["best_epoch"], "dev_metrics": row["dev_metrics"],
                                "elapsed_seconds": row["elapsed_seconds"], "checkpoint": Path(row["checkpoint"]).name,
                                "trainable_parameters": row["trainable_parameters"],
                                "total_parameters": row["total_parameters"]}
    method_descriptions = {
        "full": "Full fine-tuning of the 12-layer Phase 1 A0 backbone plus retrieval projection.",
        "lorma": "LoRMA+ post-mode multiplicative adapters (rank 8, alpha 16) in attention and feed-forward linear layers; backbone weights frozen.",
        "partial": "Upper-layer fine-tuning of the final four of twelve Phase 1 transformer blocks plus final norm and retrieval projection.",
        "kd": "Knowledge distillation from the best validation single-model teacher into a six-layer student initialized from alternating Phase 1 blocks; teacher score distribution plus gold labels.",
    }
    report = {
        "task": "CS781 Phase 7 Prior Case Retrieval",
        "metric": "Micro-F1@7",
        "phase1_parent_sha256": parent_sha,
        "phase1_parent_tokenizer_sha256": sha256_file(tokenizer_path),
        "phase1_model_config": base_cfg,
        "device": torch.cuda.get_device_name(0) if AMP else "cpu",
        "data_counts": {"train_queries": len(train_q), "train_candidates": len(train_c),
                        "dev_queries": len(dev_q), "dev_candidates": len(dev_c),
                        "test_queries": len(test_q), "test_candidates": len(test_c)},
        "lexical_tfidf_dev": lexical_metrics,
        "experiments": {name: {"description": method_descriptions[name], **row} for name, row in method_summary.items()},
        "kd_teacher": teacher_method,
        "ensemble_dev": ensemble["metrics"],
        "selected_system": {k: v for k, v in chosen.items() if k != "predictions"},
        "test_predictions": {"query_count": len(prediction), "ranked_candidates_per_query": 7,
                             "unique_ids_verified": True, "candidate_pool_membership_verified": True,
                             "test_score": None},
        "protocol": {"test_labels_used": False, "split_candidate_pools_respected": True,
                     "encoding": f"Phase 1 Legal LM tokenizer; {WINDOWS_PER_DOC} evenly spaced {WINDOW_TOKENS}-token windows per long query/case; mean hidden pooling per window and max interaction across case windows.",
                     "long_documents": f"Up to {WINDOWS_PER_DOC} evenly spaced {WINDOW_TOKENS}-token windows; mean hidden pooling per window; max interaction across candidate windows."},
    }
    with open(OUT / "experiment_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    with open(OUT / "dev_predictions_by_experiment.json", "w", encoding="utf-8") as f:
        json.dump({name: row["dev_predictions"] for name, row in results.items()}, f, ensure_ascii=False)
    log("COMPLETE", json.dumps({k: v for k, v in report.items() if k in ("lexical_tfidf_dev", "experiments", "kd_teacher", "ensemble_dev", "selected_system", "test_predictions")}))
    log("artifacts", sorted(x.name for x in OUT.iterdir()))


if __name__ == "__main__":
    main()
