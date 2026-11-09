"""Causal pathway probes for the completed LegalSpan Phase 6 adapter.

Run inside a private Kaggle kernel with the official-data/model datasets and the
completed LegalSpan kernel output attached. Uses fitting labels only as held-out
validation targets; it never loads private test references.
"""
import hashlib
import json
import sys
from pathlib import Path

import torch


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(4 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    input_root = Path("/kaggle/input")
    data_manifests = list(input_root.rglob("phase6_data_manifest.json"))
    model_checkpoints = list(input_root.rglob("legalspan_phase1.pt"))
    adapters = list(input_root.rglob("best_adapter.pt"))
    if len(data_manifests) != 1 or len(model_checkpoints) != 1 or len(adapters) != 1:
        raise RuntimeError(
            f"Need one data manifest, LegalSpan parent, and trained adapter; got "
            f"{len(data_manifests)}, {len(model_checkpoints)}, {len(adapters)}"
        )
    if not torch.cuda.is_available():
        raise RuntimeError("Causal pathway analysis requires the attached GPU")
    data_root = data_manifests[0].parent
    models_root = model_checkpoints[0].parent
    sys.path.insert(0, str(data_root))
    import legalspan_lora as method
    from phase6_data import normalize_rows, split_rows, document_hash
    from train_phase6 import pack_source, restore_trainable, save_json

    model, tokenizer, metadata = method.build(models_root, "cuda")
    checkpoint = torch.load(adapters[0], map_location="cpu", weights_only=False)
    if checkpoint["parent_metadata"]["checkpoint_sha256"] != metadata["checkpoint_sha256"]:
        raise ValueError("Adapter and Phase 1 LegalSpan parent differ")
    restore_trainable(model, checkpoint["adapter"])
    model.eval()

    train = normalize_rows(
        json.loads(next(data_root.rglob("train.json")).read_text(encoding="utf-8-sig")),
        True,
        drop_empty_summaries=True,
    )
    test = normalize_rows(
        json.loads(next(data_root.rglob("test.json")).read_text(encoding="utf-8-sig")), False
    )
    test_hashes = {document_hash(row["document"]) for row in test}
    splits = split_rows([row for row in train if document_hash(row["document"]) not in test_hashes])
    # Use a deterministic length-stratified slice so the causal comparison
    # includes both short and long judgments. Within each length quartile,
    # select by ID hash to keep the sample reproducible and label-independent.
    validation_by_length = sorted(
        splits["validation"], key=lambda row: (len(row["document"]), row["id"])
    )
    validation = []
    length_stratum = {}
    for quartile in range(4):
        start = quartile * len(validation_by_length) // 4
        end = (quartile + 1) * len(validation_by_length) // 4
        bucket = sorted(
            validation_by_length[start:end],
            key=lambda row: hashlib.sha256(row["id"].encode()).hexdigest(),
        )[:8]
        for row in bucket:
            validation.append(row)
            length_stratum[row["id"]] = f"q{quartile + 1}"
    if len(validation) != 32:
        raise RuntimeError(f"Expected 32 validation documents, got {len(validation)}")

    # Match the submitted LegalSpan run's document-only passage selection.
    # Passing the entire tokenized document directly can exceed the parent's
    # fixed context and is not how the training kernel represented the source.
    examples = []
    excluded = []
    for row in validation:
        full_source = tokenizer.encode(row["document"])
        source_ids, source_mask = pack_source(
            full_source, "legalspan_lora", tokenizer, 8, 384, "cuda"
        )
        source_length = int(source_mask.sum().item())
        # Keep the gold-summary prefix identical across interventions while
        # respecting the parent's source+decoder context limit.
        target_budget = min(1534, model.cfg.max_seq_len - source_length - 1)
        if target_budget < 32:
            excluded.append({
                "id": row["id"],
                "reason": "selected source leaves fewer than 32 target tokens",
                "document_chars": len(row["document"]),
                "selected_source_tokens": source_length,
            })
            continue
        target = [tokenizer.bos_id] + tokenizer.encode(row["summary"])[:target_budget] + [tokenizer.eos_id]
        target_ids = torch.tensor([target], dtype=torch.long, device="cuda")
        target_mask = torch.ones_like(target_ids, dtype=torch.bool)
        examples.append((row, source_ids, source_mask, target_ids, target_mask, len(full_source)))
    if not examples:
        raise RuntimeError("No validation example fits the LegalSpan training context")

    config = model.cfg
    original_flags = {
        "use_roles": config.use_roles,
        "use_continuity": config.use_continuity,
        "use_fusion": config.use_fusion,
        "use_copy": config.use_copy,
    }
    original_copy_gate = {k: v.detach().clone() for k, v in model.copy_gate.state_dict().items()}
    # SpanConfig is a frozen dataclass. These are temporary serial interventions
    # on the already-loaded model; object.__setattr__ lets us switch a feature
    # without reconstructing modules or changing any learned tensor.
    named_routes = [(f"block_{i}", block.route) for i, block in enumerate(model.blocks) if hasattr(block, "route")]
    if len(named_routes) != 4:
        raise RuntimeError(f"Expected four trained prompt routes; found {len(named_routes)}")

    def restore_flags():
        for name, value in original_flags.items():
            object.__setattr__(config, name, value)
        model.copy_gate.load_state_dict(original_copy_gate)

    def score_nll():
        totals = 0.0
        tokens = 0
        by_document = []
        with torch.inference_mode():
            for row, source_ids, source_mask, target_ids, target_mask, full_source_tokens in examples:
                loss = method.training_loss(model, source_ids, source_mask, target_ids, target_mask)
                count = int(target_mask.sum().item()) - 1
                totals += float(loss) * count
                tokens += count
                by_document.append({
                    "id": row["id"],
                    "nll": float(loss),
                    "target_tokens": count,
                    "document_chars": len(row["document"]),
                    "selected_source_tokens": int(source_mask.sum().item()),
                    "full_source_tokens": full_source_tokens,
                    "length_quartile": length_stratum[row["id"]],
                })
        quartiles = {}
        for quartile in range(4):
            key = f"q{quartile + 1}"
            docs = [d for d in by_document if d["length_quartile"] == key]
            q_tokens = sum(d["target_tokens"] for d in docs)
            quartiles[key] = {
                "documents": len(docs),
                "token_weighted_nll": (
                    sum(d["nll"] * d["target_tokens"] for d in docs) / q_tokens
                    if q_tokens else None
                ),
                "mean_document_chars": sum(d["document_chars"] for d in docs) / max(1, len(docs)),
            }
        return {
            "token_weighted_nll": totals / tokens,
            "target_tokens": tokens,
            "nll_by_length_quartile": quartiles,
            "documents": by_document,
        }

    variants = ["baseline", "no_roles", "no_continuity", "no_fusion", "no_copy", "vocab_only", "copy_only"]
    variants += [f"route_update_zero_{name}" for name, _ in named_routes]
    ablations = {}
    for variant in variants:
        restore_flags()
        temporary_handles = []
        if variant == "no_roles":
            object.__setattr__(config, "use_roles", False)
        elif variant == "no_continuity":
            object.__setattr__(config, "use_continuity", False)
        elif variant == "no_fusion":
            object.__setattr__(config, "use_fusion", False)
        elif variant == "no_copy":
            object.__setattr__(config, "use_copy", False)
        elif variant in ("vocab_only", "copy_only"):
            with torch.no_grad():
                model.copy_gate.weight.zero_()
                model.copy_gate.bias.fill_(-20.0 if variant == "vocab_only" else 20.0)
        elif variant.startswith("route_update_zero_"):
            target = variant.removeprefix("route_update_zero_")
            route = dict(named_routes)[target]
            temporary_handles.append(
                route.out.register_forward_hook(lambda _m, _i, out: torch.zeros_like(out))
            )
        ablations[variant] = score_nll()
        print(json.dumps({"event": "ablation_complete", "variant": variant,
                          "nll": ablations[variant]["token_weighted_nll"]}), flush=True)
        for handle in temporary_handles:
            handle.remove()

    baseline_nll = ablations["baseline"]["token_weighted_nll"]
    for result in ablations.values():
        result["delta_nll_vs_baseline"] = result["token_weighted_nll"] - baseline_nll

    restore_flags()
    generated = []
    pathway = {name: {"gate": [], "route_entropy": [], "role_mass": []} for name, _ in named_routes}
    copy_gate_values = []
    gate_handle = model.copy_gate.register_forward_hook(
        lambda _m, _i, out: copy_gate_values.extend(out.detach().float().sigmoid().flatten().cpu().tolist())
    )
    route_handles = []
    for name, route in named_routes:
        def collect(_module, _inputs, output, route_name=name):
            alpha = output[2].detach().float()
            if not alpha.numel():
                return
            pathway[route_name]["gate"].append(float(output[4].detach().float().mean().cpu()))
            entropy = -(alpha * alpha.clamp_min(1e-30).log()).sum(-1).mean()
            pathway[route_name]["route_entropy"].append(float(entropy.cpu()))
        route_handles.append(route.register_forward_hook(collect))

    # A few fixed validation documents permit readable causal examples without
    # spending the full generation budget on every intervention.
    # Generate one short, one middle-length, and one long validation example.
    ordered_examples = sorted(examples, key=lambda ex: len(ex[0]["document"]))
    generation_examples = [ordered_examples[0], ordered_examples[len(ordered_examples) // 2], ordered_examples[-1]]
    generation_arms = ["baseline", "no_fusion", "no_copy"]
    for arm in generation_arms:
        restore_flags()
        if arm == "no_fusion":
            object.__setattr__(config, "use_fusion", False)
        elif arm == "no_copy":
            object.__setattr__(config, "use_copy", False)
        arm_outputs = []
        for row, source_ids, source_mask, *_ in generation_examples:
            copy_gate_values.clear()
            for values in pathway.values():
                values["gate"].clear(); values["route_entropy"].clear()
            generation_budget = min(
                768, model.cfg.max_seq_len - int(source_mask.sum().item())
            )
            if generation_budget < 1:
                raise RuntimeError(f"No context left to generate for validation ID {row['id']}")
            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
                ids = method.generate(model, source_ids, source_mask, generation_budget, tokenizer.eos_id)[0]
            clean = []
            for token in ids:
                if token == tokenizer.eos_id:
                    break
                if token not in (tokenizer.pad_id, tokenizer.bos_id):
                    clean.append(int(token))
            source_set = set(source_ids[0].tolist())
            text = tokenizer.decode(clean).strip()
            arm_outputs.append({
                "id": row["id"],
                "summary": text,
                "reference_summary": row["summary"],
                "generation_budget": generation_budget,
                "selected_source_tokens": int(source_mask.sum().item()),
                "tokens": len(clean),
                "generated_tokens_found_in_source_fraction": (
                    sum(token in source_set for token in clean) / max(1, len(clean))
                ),
                "mean_copy_gate": sum(copy_gate_values) / max(1, len(copy_gate_values)),
                "route_pathway": {
                    name: {
                        "mean_gate_strength": sum(values["gate"]) / max(1, len(values["gate"])),
                        "mean_attention_entropy": sum(values["route_entropy"]) / max(1, len(values["route_entropy"])),
                    }
                    for name, values in pathway.items()
                },
            })
        generated.append({"arm": arm, "outputs": arm_outputs})

    gate_handle.remove()
    for handle in route_handles:
        handle.remove()
    report = {
        "method": "LegalSpan causal pathway ablation",
        "phase1_parent_sha256": metadata["checkpoint_sha256"],
        "tokenizer_sha256": metadata["tokenizer_sha256"],
        "adapter_sha256": sha(adapters[0]),
        "teacher_selected_epoch": checkpoint["epoch"],
        "validation_ids": [row["id"] for row, *_ in examples],
        "validation_length_strata": length_stratum,
        "validation_documents": len(examples),
        "validation_documents_requested": len(validation),
        "context_exclusions": excluded,
        "generation_documents": [row["id"] for row, *_ in generation_examples],
        "parameters": metadata["total_parameters"],
        "base_validation_token_weighted_nll": ablations["baseline"]["token_weighted_nll"],
        "ablation_nll": ablations,
        "pathway_generation_probes": generated,
        "bertscore_f1": None,
        "notes": "Causal interventions on a small fixed validation slice; effects do not prove semantic feature labels or leaderboard gains.",
    }
    out = Path("/kaggle/working/legalspan_interpretability")
    out.mkdir(parents=True, exist_ok=True)
    (out / "interpretability_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (out / "generated_interventions.json").write_text(json.dumps(generated, indent=2), encoding="utf-8")
    print(json.dumps({"event": "interpretability_complete", "output": str(out),
                      "baseline_nll": report["base_validation_token_weighted_nll"]}), flush=True)


if __name__ == "__main__":
    main()
