"""Phase 5 LSI knowledge distillation: trained B1 teacher -> 4-layer B1 student.

The teacher's task adapter must be a completed, validation-selected Phase 5
checkpoint. Both networks initialize from the user's own selected Phase 1 B1
checkpoint. No private labels or outside pretrained weights are used.
"""
from __future__ import annotations

import argparse
import json
import random
import time
import zipfile
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader

import phase5_b1_train as base
from phase5_core import choose_thresholds, f1_scores
from cs781.models.b import EncoderDecoderConfig, EncoderDecoderLegalModel
from cs781.own_bpe import OwnBPE


class StudentB1Classifier(nn.Module):
    """Retain Phase 1 embeddings and first 4 encoder blocks; omit decoder."""

    def __init__(self, checkpoint):
        super().__init__()
        record = torch.load(checkpoint, map_location="cpu", weights_only=False)
        if record["metadata"]["variant"] != "B_taskaligned":
            raise ValueError("Expected selected B1 own-BPE Phase 1 checkpoint")
        cfg = EncoderDecoderConfig(**record["metadata"]["model_config"])
        self.backbone = EncoderDecoderLegalModel(cfg)
        self.backbone.load_state_dict(record["model"], strict=True)
        self.backbone.encoder = nn.ModuleList(list(self.backbone.encoder[:4]))
        self.backbone.decoder = nn.ModuleList()
        self.backbone.decoder_norm = nn.Identity()
        self.backbone.lm_head = nn.Identity()
        self.dropout = nn.Dropout(.1)
        self.head = nn.Linear(4 * cfg.d_model, base.LABELS)

    def forward(self, ids, mask):
        batch, windows, width = ids.shape
        encoded, _ = self.backbone.encode(ids.reshape(batch * windows, width), mask.reshape(batch * windows, width))
        valid = mask.reshape(batch * windows, width)
        means = (encoded * valid.unsqueeze(-1)).sum(1) / valid.sum(1).clamp(min=1).unsqueeze(-1)
        maxima = encoded.masked_fill(~valid.unsqueeze(-1), torch.finfo(encoded.dtype).min).max(1).values
        features = torch.cat((means.reshape(batch, -1), maxima.reshape(batch, -1)), dim=1)
        return self.head(self.dropout(features))


def distillation_loss(student_logits, teacher_logits, labels, pos_weight, temperature=2.0):
    supervised = F.binary_cross_entropy_with_logits(student_logits.float(), labels.float(), pos_weight=pos_weight)
    teacher_soft = torch.sigmoid(teacher_logits.float() / temperature)
    soft = F.binary_cross_entropy_with_logits(student_logits.float() / temperature, teacher_soft)
    return .7 * supervised + .3 * temperature**2 * soft


def load_teacher(checkpoint, adapter_path, method):
    if method == "lorma":
        from phase5_lorma import LoRMAPlusLinear
        base.LoRALinear = LoRMAPlusLinear
    elif method != "lora":
        raise ValueError("Teacher method must be lora or lorma")
    teacher = base.B1Classifier(checkpoint)
    artifact = torch.load(adapter_path, map_location="cpu", weights_only=False)
    if artifact.get("phase1_checkpoint_sha256") != base.sha256(checkpoint):
        raise ValueError("Teacher adapter parent checkpoint mismatch")
    named = dict(teacher.named_parameters())
    if set(artifact["trainable_state"]) != {name for name, p in named.items() if p.requires_grad}:
        raise ValueError("Teacher adapter parameter names mismatch")
    for name, value in artifact["trainable_state"].items():
        if tuple(named[name].shape) != tuple(value.shape):
            raise ValueError(f"Teacher adapter shape mismatch: {name}")
        named[name].data.copy_(value)
    teacher.requires_grad_(False)
    teacher.eval()
    return teacher, artifact


def run(args):
    random.seed(base.SEED + 1)
    np.random.seed(base.SEED + 1)
    torch.manual_seed(base.SEED + 1)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU required")
    inp, out = Path(args.input), Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    checkpoint = inp / "b1_latest_model.pt"
    tokenizer_path = inp / "tokenizer.json"
    expected = {
        "train": "f3601a7fce994616daf99fec251ca7936e98675a8fbf8af20a782cd55054c8a7",
        "dev": "ab2660ac23abf38e902d63ab6871df740dc38e3fd973f441745f42564dadbd15",
        "test": "c699d50bcc2f3e369955268d6195a80b23322ed91cb1fadc8d58b26ca8e9aa77",
    }
    for name, digest in expected.items():
        if base.sha256(inp / f"{name}.json") != digest:
            raise ValueError(f"Official {name} hash mismatch")
    if base.sha256(checkpoint) != "111296fb36a23dac1a11c07ede927cfc89dc86d51253e441d59afb82bd6924eb":
        raise ValueError("Own Phase 1 B1 checkpoint mismatch")
    if base.sha256(tokenizer_path) != "5c0bd30290e0bd6919253b1ff864f52053924808aabd33482df92cc4bd471c8e":
        raise ValueError("Own BPE tokenizer mismatch")
    splits = {name: base.read_json(inp / f"{name}.json") for name in expected}
    manifest = base.read_json(inp / "phase5_split_manifest.json")
    if manifest["source_sha256"] != expected:
        raise ValueError("Split manifest mismatch")
    lookup = {str(row["id"]): i for i, row in enumerate(splits["train"])}
    fit = np.asarray([lookup[x] for x in manifest["fit_ids"]], dtype=np.int64)
    cal = np.asarray([lookup[x] for x in manifest["calibration_ids"]], dtype=np.int64)
    if (len(lookup) != len(splits["train"]) or len(fit) != 38113 or len(cal) != 4313
            or set(manifest["fit_ids"]) & set(manifest["calibration_ids"])
            or any("labels" in row for row in splits["test"])):
        raise ValueError("Fit/calibration/test integrity failure")
    tokenizer = OwnBPE(base.read_json(tokenizer_path)["merges"])
    cache = out / "token_cache"
    cache.mkdir(exist_ok=True)
    arrays = {name: base.tokenize_split(name, rows, tokenizer, cache) for name, rows in splits.items()}
    targets = {name: base.label_matrix(rows) for name, rows in splits.items()}
    train_rows = base.Rows(*arrays["train"], targets["train"], fit)
    cal_rows = base.Rows(*arrays["train"], targets["train"], cal)
    dev_rows = base.Rows(*arrays["dev"], targets["dev"], range(len(splits["dev"])))
    test_rows = base.Rows(*arrays["test"], targets["test"], range(len(splits["test"])))
    train_loader = DataLoader(train_rows, batch_size=args.batch, shuffle=True, pin_memory=True)
    cal_loader = DataLoader(cal_rows, batch_size=args.eval_batch, pin_memory=True)
    dev_loader = DataLoader(dev_rows, batch_size=args.eval_batch, pin_memory=True)
    test_loader = DataLoader(test_rows, batch_size=args.eval_batch, pin_memory=True)
    teacher, teacher_artifact = load_teacher(checkpoint, args.teacher_adapter, args.teacher_method)
    teacher = teacher.cuda()
    student = StudentB1Classifier(checkpoint).cuda()
    # Transfer the teacher's task head, then distill its probabilities while
    # updating the smaller student on official labeled training cases.
    student.head.load_state_dict(teacher.head.state_dict())
    print(json.dumps({"teacher_method": args.teacher_method, "teacher_epoch": teacher_artifact["epoch"], "teacher_parameters": sum(p.numel() for p in teacher.parameters()), "student_parameters": sum(p.numel() for p in student.parameters()), "fit": len(fit), "calibration": len(cal)}), flush=True)
    support = targets["train"][fit].sum(axis=0)
    weight = torch.tensor(np.minimum(np.sqrt((len(fit) - support) / np.maximum(support, 1)), 12), device="cuda", dtype=torch.float32)
    optimizer = torch.optim.AdamW(student.parameters(), lr=args.lr, weight_decay=.01)
    scaler = torch.amp.GradScaler("cuda")
    best_score, history, start = -1.0, [], time.monotonic()
    for epoch in range(args.epochs):
        student.train()
        optimizer.zero_grad(set_to_none=True)
        running = 0.0
        for step, (ids, mask, labels) in enumerate(train_loader):
            ids, mask, labels = ids.cuda(non_blocking=True), mask.cuda(non_blocking=True), labels.cuda(non_blocking=True)
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
                teacher_logits = teacher(ids, mask)
            with torch.autocast("cuda", dtype=torch.float16):
                student_logits = student(ids, mask)
                loss = distillation_loss(student_logits, teacher_logits, labels, weight, args.temperature) / args.accum
            scaler.scale(loss).backward()
            if (step + 1) % args.accum == 0 or step + 1 == len(train_loader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
            running += float(loss.detach()) * args.accum
            if step % 100 == 0:
                print(json.dumps({"epoch": epoch + 1, "step": step, "steps": len(train_loader), "mean_loss": running / (step + 1), "elapsed_minutes": round((time.monotonic() - start) / 60, 2)}), flush=True)
        cal_probs = base.predict(student, cal_loader, "cuda")
        thresholds = choose_thresholds(targets["train"][cal], cal_probs)
        scores = f1_scores(targets["train"][cal], cal_probs >= thresholds)
        history.append({"epoch": epoch + 1, "calibration_macro_f1": scores["macro_f1"], "train_loss": running / len(train_loader)})
        print(json.dumps(history[-1]), flush=True)
        if scores["macro_f1"] > best_score:
            best_score = scores["macro_f1"]
            torch.save({"student_state": student.state_dict(), "thresholds": thresholds, "epoch": epoch + 1, "calibration_metrics": scores, "phase1_checkpoint_sha256": base.sha256(checkpoint), "teacher_adapter_sha256": base.sha256(args.teacher_adapter), "teacher_method": args.teacher_method}, out / "best_student.pt")
        (out / "training_history.json").write_text(json.dumps(history, indent=2))
    selected = torch.load(out / "best_student.pt", map_location="cpu", weights_only=False)
    student.load_state_dict(selected["student_state"], strict=True)
    student.cuda()
    dev_probs = base.predict(student, dev_loader, "cuda")
    dev_scores = f1_scores(targets["dev"], dev_probs >= selected["thresholds"])
    test_probs = base.predict(student, test_loader, "cuda")
    np.save(out / "dev_probabilities.npy", dev_probs)
    np.save(out / "test_probabilities.npy", test_probs)
    predictions = [{"id": str(row["id"]), "pred": np.flatnonzero(prob >= selected["thresholds"]).astype(int).tolist()} for row, prob in zip(splits["test"], test_probs)]
    submission = out / "phase5_b1_kd_prediction.zip"
    with zipfile.ZipFile(submission, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("prediction.json", json.dumps(predictions, separators=(",", ":")))
    report = {"method": "supervised multi-label BCE plus B1-teacher logit distillation", "teacher_method": args.teacher_method, "teacher_adapter_sha256": base.sha256(args.teacher_adapter), "phase1_checkpoint_sha256": base.sha256(checkpoint), "student_encoder_layers": 4, "student_parameters": sum(p.numel() for p in student.parameters()), "selected_epoch": selected["epoch"], "calibration": selected["calibration_metrics"], "dev": dev_scores, "thresholds": selected["thresholds"].tolist(), "fit_count": len(fit), "calibration_count": len(cal), "dev_count": len(splits["dev"]), "test_count": len(splits["test"]), "submission_sha256": base.sha256(submission), "elapsed_minutes": (time.monotonic() - start) / 60}
    (out / "phase5_b1_kd_report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps({"complete": True, "dev_macro_f1": dev_scores["macro_f1"], "submission": str(submission)}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--teacher-adapter", required=True)
    parser.add_argument("--teacher-method", choices=["lora", "lorma"], required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--eval-batch", type=int, default=16)
    parser.add_argument("--accum", type=int, default=2)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--temperature", type=float, default=2.0)
    run(parser.parse_args())
