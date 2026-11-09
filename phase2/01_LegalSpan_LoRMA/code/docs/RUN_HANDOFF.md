# Phase 2 running queue

Started 11 October 2026 around 01:51 IST. Remote PID: 125448. Physical GPU1: NVIDIA RTX 4080. Local downloader PID: 14608. These are launch observations; use live status for current state.

Order: LegalSpan LoRMA+, B1 LoRA, C1 full fine-tuning, A0 knowledge distillation from the completed B1 NER model. Maximum 12 epochs each, validation early stopping after 3 epochs without improvement. Four versions; A0 and LegalSpan are related architectures.

Submission ZIPs are named UPLOAD_Phase2_*.zip and contain only prediction.json. The source archive is for your records, not for Codabench upload.

## Locations

Code: /home/stud1/cs781_phase2_20261011
Runs: /media/hdd/SAMBA4080_left_Dataset/stud1/cs781_phase2_20261011/runs
Local verified downloads: outputs/phase2_20261011 and Windows Downloads.
Downloader logs: outputs/phase2_downloader.log and outputs/phase2_downloader.err.

## Monitor

`ssh stud1 "bash /home/stud1/cs781_phase2_20261011/scripts/monitor_gpu1.sh"`

Each stage records run.json, metrics.jsonl, latest.pt, best.pt, selection.json, official_validation.txt, and complete.json with verified prediction ZIP and selected checkpoint hashes. Distillation requires verified B1 completion. Resume checks the config, checkpoint, split, teacher, and all implementation hashes.

Remote training survives VS Code and SSH disconnection. Keep this PC awake and online for immediate automatic downloads. The downloader catches up after reconnection.

Actual hardware is recorded honestly. configs/t4_portability.json is an example configuration, not a measured T4 run.

