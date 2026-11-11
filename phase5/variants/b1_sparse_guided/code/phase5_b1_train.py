"""CS781 Phase 5 LSI: B1-own-BPE encoder adaptation from Phase 1 weights.

Kaggle command: python phase5_b1_train.py --input /kaggle/input/cs781-phase5-lsi-private-20261028 --output /kaggle/working
Only the official Phase 5 train split is optimized. Calibration is a disjoint
grouped subset of train; the official dev is evaluated once after selection.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys
import time
import zipfile
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).with_name("phase1_source.zip")))
from phase5_core import choose_thresholds, f1_scores, pack_windows
from cs781.models.b import EncoderDecoderConfig, EncoderDecoderLegalModel
from cs781.own_bpe import OwnBPE


LABELS = 100
SEED = 781


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(2**20), b""):
            digest.update(block)
    return digest.hexdigest()


class LoRALinear(nn.Module):
    def __init__(self, base, rank=8, alpha=16):
        super().__init__()
        self.base = base
        self.rank = rank
        self.scale = alpha / rank
        self.a = nn.Parameter(torch.empty(rank, base.in_features))
        self.b = nn.Parameter(torch.zeros(base.out_features, rank))
        nn.init.kaiming_uniform_(self.a, a=5**.5)

    def forward(self, x):
        update = torch.nn.functional.linear(torch.nn.functional.linear(x, self.a), self.b)
        return self.base(x) + update * self.scale


class B1Classifier(nn.Module):
    def __init__(self, checkpoint):
        super().__init__()
        record = torch.load(checkpoint, map_location="cpu", weights_only=False)
        if record["metadata"]["variant"] != "B_taskaligned":
            raise ValueError("Expected selected B1 own-BPE checkpoint")
        cfg = EncoderDecoderConfig(**record["metadata"]["model_config"])
        self.backbone = EncoderDecoderLegalModel(cfg)
        self.backbone.load_state_dict(record["model"], strict=True)
        for param in self.backbone.parameters():
            param.requires_grad_(False)
        for block in self.backbone.encoder:
            block.attn.wqkv = LoRALinear(block.attn.wqkv)
            block.attn.wo = LoRALinear(block.attn.wo)
            for name in ("w1", "w2", "w3"):
                setattr(block.ffn, name, LoRALinear(getattr(block.ffn, name)))
        self.dropout = nn.Dropout(.1)
        self.head = nn.Linear(4 * cfg.d_model, LABELS)

    def forward(self, ids, mask):
        batch, windows, width = ids.shape
        encoded, _ = self.backbone.encode(ids.reshape(batch * windows, width), mask.reshape(batch * windows, width))
        valid = mask.reshape(batch * windows, width)
        means = (encoded * valid.unsqueeze(-1)).sum(1) / valid.sum(1).clamp(min=1).unsqueeze(-1)
        maxima = encoded.masked_fill(~valid.unsqueeze(-1), torch.finfo(encoded.dtype).min).max(1).values
        features = torch.cat((means.reshape(batch, -1), maxima.reshape(batch, -1)), dim=1)
        return self.head(self.dropout(features))

    def trainable_state(self):
        return {name: param.detach().cpu() for name, param in self.named_parameters() if param.requires_grad}


class Rows(Dataset):
    def __init__(self, ids, masks, targets, selected):
        self.ids, self.masks, self.targets = ids, masks, targets
        self.selected = np.asarray(selected, dtype=np.int64)

    def __len__(self):
        return len(self.selected)

    def __getitem__(self, index):
        i = self.selected[index]
        return torch.from_numpy(self.ids[i].astype(np.int64)), torch.from_numpy(self.masks[i]), torch.from_numpy(self.targets[i])


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def tokenize_split(name, rows, tokenizer, cache):
    ids_file = cache / f"{name}_ids.npy"
    masks_file = cache / f"{name}_masks.npy"
    if ids_file.exists() and masks_file.exists():
        ids, masks = np.load(ids_file, mmap_mode="r"), np.load(masks_file, mmap_mode="r")
        if ids.shape == (len(rows), 2, 384) and masks.shape == ids.shape:
            return ids, masks
    ids = np.zeros((len(rows), 2, 384), dtype=np.int32)
    masks = np.zeros_like(ids, dtype=np.bool_)
    for i, row in enumerate(rows):
        ids[i], masks[i] = pack_windows(row["text"], tokenizer)
        if i % 2000 == 0:
            print(json.dumps({"tokenizing": name, "done": i, "total": len(rows)}), flush=True)
    np.save(ids_file, ids)
    np.save(masks_file, masks)
    return ids, masks


def label_matrix(rows):
    y = np.zeros((len(rows), LABELS), dtype=np.float32)
    for i, row in enumerate(rows):
        for label in set(row.get("labels", [])):
            if type(label) is not int or not 0 <= label < LABELS:
                raise ValueError(f"Invalid statute ID in {row['id']}")
            y[i, label] = 1
    return y


@torch.no_grad()
def predict(model, loader, device):
    model.eval()
    out = []
    for i, (ids, masks, _) in enumerate(loader):
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            logits = model(ids.to(device), masks.to(device))
        out.append(torch.sigmoid(logits.float()).cpu().numpy())
        if i % 100 == 0:
            print(json.dumps({"predict_batches": i, "total": len(loader)}), flush=True)
    return np.concatenate(out, axis=0)


def train(args):
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    if not torch.cuda.is_available():
        raise RuntimeError("A CUDA GPU is required for this model run")
    torch.backends.cuda.matmul.allow_tf32 = True
    inp, out = Path(args.input), Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    cache = out / "token_cache"
    cache.mkdir(exist_ok=True)
    expected = {
        "train": "f3601a7fce994616daf99fec251ca7936e98675a8fbf8af20a782cd55054c8a7",
        "dev": "ab2660ac23abf38e902d63ab6871df740dc38e3fd973f441745f42564dadbd15",
        "test": "c699d50bcc2f3e369955268d6195a80b23322ed91cb1fadc8d58b26ca8e9aa77",
    }
    for split, digest in expected.items():
        if sha256(inp / f"{split}.json") != digest:
            raise ValueError(f"Official {split} file hash mismatch")
    checkpoint = inp / "b1_latest_model.pt"
    tokenizer_path = inp / "tokenizer.json"
    if sha256(checkpoint) != "111296fb36a23dac1a11c07ede927cfc89dc86d51253e441d59afb82bd6924eb":
        raise ValueError("Phase 1 model-only checkpoint hash mismatch")
    if sha256(tokenizer_path) != "5c0bd30290e0bd6919253b1ff864f52053924808aabd33482df92cc4bd471c8e":
        raise ValueError("Phase 1 tokenizer hash mismatch")
    splits = {s: read_json(inp / f"{s}.json") for s in expected}
    manifest = read_json(inp / "phase5_split_manifest.json")
    if manifest["source_sha256"] != expected:
        raise ValueError("Split manifest hashes mismatch")
    train_ids = [str(row["id"]) for row in splits["train"]]
    lookup = {key: i for i, key in enumerate(train_ids)}
    fit = np.array([lookup[key] for key in manifest["fit_ids"]], dtype=np.int64)
    cal = np.array([lookup[key] for key in manifest["calibration_ids"]], dtype=np.int64)
    excluded = set(manifest["excluded_train_ids"])
    if (len(set(train_ids)) != len(train_ids) or set(manifest["fit_ids"]) & set(manifest["calibration_ids"])
            or set(manifest["fit_ids"]) & excluded or set(manifest["calibration_ids"]) & excluded
            or len(fit) != 38113 or len(cal) != 4313 or any("labels" in row for row in splits["test"])):
        raise ValueError("Split or test contract mismatch")
    # The saved BPE was trained under Unicode 13. The exact tokenizer file is
    # hash-checked above. Kaggle's newer Python has a newer Unicode database;
    # the merge table itself can be replayed without retraining or replacing it.
    tokenizer = OwnBPE(read_json(tokenizer_path)["merges"])
    arrays = {s: tokenize_split(s, rows, tokenizer, cache) for s, rows in splits.items()}
    targets = {s: label_matrix(rows) for s, rows in splits.items()}
    train_rows = Rows(*arrays["train"], targets["train"], fit)
    cal_rows = Rows(*arrays["train"], targets["train"], cal)
    dev_rows = Rows(*arrays["dev"], targets["dev"], range(len(splits["dev"])))
    test_rows = Rows(*arrays["test"], targets["test"], range(len(splits["test"])))
    loader = DataLoader(train_rows, batch_size=args.batch, shuffle=True, num_workers=0, pin_memory=True)
    cal_loader = DataLoader(cal_rows, batch_size=args.eval_batch, num_workers=0, pin_memory=True)
    dev_loader = DataLoader(dev_rows, batch_size=args.eval_batch, num_workers=0, pin_memory=True)
    test_loader = DataLoader(test_rows, batch_size=args.eval_batch, num_workers=0, pin_memory=True)
    device = "cuda"
    model = B1Classifier(checkpoint).to(device)
    trainable = [p for p in model.parameters() if p.requires_grad]
    print(json.dumps({"device": torch.cuda.get_device_name(), "trainable_parameters": sum(p.numel() for p in trainable), "all_parameters": sum(p.numel() for p in model.parameters()), "fit": len(fit), "calibration": len(cal)}), flush=True)
    supports = targets["train"][fit].sum(axis=0)
    pos_weights = torch.tensor(np.minimum(np.sqrt((len(fit) - supports) / np.maximum(supports, 1)), 12), device=device, dtype=torch.float32)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weights)
    optimizer = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=.01)
    scaler = torch.amp.GradScaler("cuda")
    best_score, best_path, history = -1, out / "best_adapter.pt", []
    started = time.monotonic()
    for epoch in range(args.epochs):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        running = 0
        for step, (ids, masks, y) in enumerate(loader):
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                logits = model(ids.to(device), masks.to(device))
                loss = criterion(logits.float(), y.to(device)) / args.accum
            scaler.scale(loss).backward()
            if (step + 1) % args.accum == 0 or step + 1 == len(loader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(trainable, 1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
            running += float(loss.detach()) * args.accum
            if step % 100 == 0:
                print(json.dumps({"epoch": epoch + 1, "step": step, "steps": len(loader), "mean_loss": running / (step + 1), "elapsed_minutes": round((time.monotonic() - started) / 60, 2)}), flush=True)
        cal_probs = predict(model, cal_loader, device)
        cal_y = targets["train"][cal]
        thresholds = choose_thresholds(cal_y, cal_probs)
        metrics = f1_scores(cal_y, cal_probs >= thresholds)
        history.append({"epoch": epoch + 1, "calibration": metrics, "mean_loss": running / len(loader)})
        print(json.dumps({"epoch": epoch + 1, "calibration_macro_f1": metrics["macro_f1"]}), flush=True)
        if metrics["macro_f1"] > best_score:
            best_score = metrics["macro_f1"]
            torch.save({"trainable_state": model.trainable_state(), "thresholds": thresholds, "epoch": epoch + 1, "calibration_metrics": metrics, "phase1_checkpoint_sha256": sha256(checkpoint)}, best_path)
        (out / "training_history.json").write_text(json.dumps(history, indent=2))
    selected = torch.load(best_path, map_location="cpu", weights_only=False)
    own = dict(model.named_parameters())
    for name, value in selected["trainable_state"].items():
        own[name].data.copy_(value.to(device))
    dev_probs = predict(model, dev_loader, device)
    dev_y = targets["dev"]
    dev_metrics = f1_scores(dev_y, dev_probs >= selected["thresholds"])
    test_probs = predict(model, test_loader, device)
    np.save(out / "dev_probabilities.npy", dev_probs)
    np.save(out / "test_probabilities.npy", test_probs)
    predictions = [{"id": str(row["id"]), "pred": np.flatnonzero(prob >= selected["thresholds"]).astype(int).tolist()} for row, prob in zip(splits["test"], test_probs)]
    with zipfile.ZipFile(out / "phase5_b1_lora_prediction.zip", "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("prediction.json", json.dumps(predictions, separators=(",", ":")))
    report = {"phase1_checkpoint_sha256": sha256(checkpoint), "tokenizer_sha256": sha256(tokenizer_path), "data_sha256": expected, "selected_epoch": selected["epoch"], "calibration": selected["calibration_metrics"], "dev": dev_metrics, "thresholds": selected["thresholds"].tolist(), "fit_count": len(fit), "calibration_count": len(cal), "dev_count": len(splits["dev"]), "test_count": len(splits["test"]), "elapsed_minutes": (time.monotonic() - started) / 60, "submission_sha256": sha256(out / "phase5_b1_lora_prediction.zip")}
    (out / "phase5_b1_lora_report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps({"complete": True, "dev_macro_f1": dev_metrics["macro_f1"], "dev_micro_f1": dev_metrics["micro_f1"], "output": str(out)}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--eval-batch", type=int, default=16)
    parser.add_argument("--accum", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--lr", type=float, default=2e-4)
    train(parser.parse_args())
