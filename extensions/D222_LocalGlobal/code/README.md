# D222: efficient legal decoder and NER

D222 is an independent from-scratch decoder: 222,079,872 parameters, 24 layers, width 896, 14 query heads / 2 KV heads, 512-token local windows and six periodic global layers. SWA/GQA mechanisms follow Mistral 7B; their use does not establish leaderboard improvement. Tiled local attention computes a bounded query/KV halo; global layers may use CUDA Flash SDPA, confirmed only by profiler evidence. Optional flash-attn uses a real window kernel. Actual throughput and hardware are recorded.

Start with CS781_D222_Pretrain_and_NER.ipynb. Run from this directory with PYTHONPATH=.:vendor:vendor/backbones on Linux. Configure paths in configs/d222.json. Runtime is the existing Python 3.10/PyTorch 2.5.1+cu121 environment. Own BPE Unicode compatibility is checked by its loader; copying it into a different Python Unicode version may fail correctly.

1. python -m legal_d.preflight --config configs/d222.json
2. python -m legal_d.pretrain --config configs/d222.json (manual pretraining only), or bash scripts/run_gpu1.sh for the complete detached queue.
3. The pipeline pretrains, performs bounded NER candidate selection and full/LoRMA exports, then benchmarks closed Phase 1. The scorer's pinned cache/provenance are required; missing evaluation assets produce an explicit blocker.

Pretraining supervises all eligible cached train tokens and preserves dev/confirmation isolation. The requested 300M training-token budget is separate from the 222M parameter count. Calibration may reduce achieved tokens to preserve three hours for NER before the configured deadline. A larger model with too little data may score worse. NER has an original-character BiGRU head, learned BIO CRF, weighted CE and auxiliary start/end loss. Rare-label sampling uses training counts only. New methods keep the same held-out documents, and the existing best score/ZIP is retained if D fails to improve it.

Only files named UPLOAD_Phase2_D222_Full.zip and UPLOAD_Phase2_D222_LORMA.zip are Phase 2 result submissions. They contain root prediction.json, all 35 official IDs and valid original character spans. BENCHMARK_ONLY_Phase1_D222.zip is a local Phase 1-format artifact; Phase 1 ended on October 6 and this code does not upload it. Source ZIPs and checkpoints are not Codabench result ZIPs.

The notebook/config do not fabricate measured scores, achieved token counts, T4 usage or completed predictions. Corpus default is the four official sources; Wikipedia is not enabled because the live data-only rule conflicts with the supplied FAQ. The current brief says six Phase 2+ submissions in Important Instructions but four in Section 8.2; plan for six meaningful experiments and clarify the discrepancy.

Pretraining checkpoint selection is initially held-out LM loss to avoid consuming the NER deadline. After Phase 2 exports, candidate checkpoints/decoding policies are scored by official BERTScore F1, ROUGE-L and BLEU, followed by untouched confirmation. Do not present preliminary CE selection as measured BERTScore selection. Any optional longer continuation or extra data variant needs a new config/run/provenance record.

References: https://arxiv.org/abs/2310.06825 ; https://arxiv.org/abs/2203.15556 ; https://aclanthology.org/2025.findings-acl.527/ . Own source/tokenizer modules are retained in vendor with their original documentation and attributed references. Evaluation-only RoBERTa is the course scorer, not a participant model or training teacher.

Review fixes: dev documents longer than context are covered by deterministic windows, not dropped; cached completions/generations require matching fingerprints; preflight is bound to current config/source/data/GPU; NER candidates have separate time allocations and validation/export margins. CRF dynamic programming runs on CPU with differentiable copies to avoid thousands of tiny GPU kernels. The authorized closed Phase 1 benchmark may continue after today's deadline (allow_postdeadline_benchmark=true); it does not submit to that phase.
