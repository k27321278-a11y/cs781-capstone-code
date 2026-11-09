# CS781 Phase 2 Legal NER

Four sequential experiments: LegalSpan LoRMA+, B1 LoRA, C1 full fine-tuning, and A0 knowledge distillation from B1. All use our own Phase 1 checkpoints. A0 and LegalSpan are related architectures; this package does not certify four independent families.

The NER head is a bidirectional character GRU and BIO classifier over the pretrained contextual token representations plus character embeddings. Exact original character offsets are maintained through both tokenizer types and overlapping windows. Gold entity overlaps are checked; this dataset has none. The fixed held-out document split is shared by all models. No test annotations or external pretrained models are used.

Run: `PYTHONPATH=backbones:. python -m lner.train --config configs/queue.json --preflight`, then run without `--preflight`. Configure checkpoint/tokenizer/data/output paths for your environment. Physical-GPU1 deployment is in scripts/launch_gpu1.sh. Check with scripts/monitor_gpu1.sh. Re-running the queue resumes interrupted stages and validates already completed ZIP hashes.

Each `UPLOAD_Phase2_*.zip` contains only root prediction.json for all 35 official test IDs. These are the files to upload to Codabench. Source packages and checkpoints are not Codabench submissions. Automatic downloads go to the chat's outputs/phase2_20261011 directory and Windows Downloads. The local downloader catches up when the computer is online; remote training survives an SSH/VS Code disconnect.

Run records contain actual hardware, source/config/checkpoint/data hashes, training curves, trainable parameter counts, selected epochs, strict validation metrics, and completed ZIP hashes. The T4 configuration is a portability example, not evidence of a T4 run. No private leaderboard score is inferred from local validation.

References: Hu et al. (2021), LoRA, https://arxiv.org/abs/2106.09685; Bihany, Patel and Modi (2025), LoRMA, https://aclanthology.org/2025.findings-acl.527/ and https://github.com/Exploration-Lab/LoRMA; Hinton, Vinyals and Dean (2015), Distilling the Knowledge in a Neural Network, https://arxiv.org/abs/1503.02531. The retained LoRMA reference is CC-BY-NC and is used only for mathematical attribution/reference. Official evaluator: course Evaluation Script Drive folder, evaluate_phase2.py. The adaptation wrappers here are our implementation.

An extracted complete model archive can reproduce predictions without the original Phase 1 checkpoint: from its source directory run PYTHONPATH=backbones:. python scripts/predict_checkpoint.py --model-dir .. --output reproduced.zip. PyTorch, NumPy and tokenizers are required. The archived own BPE tokenizer expects the original Python 3.10 Unicode version; use the recorded runtime for exact reproduction.
