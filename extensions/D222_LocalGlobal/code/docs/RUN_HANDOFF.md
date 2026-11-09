# D222 live queue handoff

Started 11 October 2026 09:50:53 India time. Physical GPU1, NVIDIA RTX 4080; PID 153218 is the launch observation, not a permanent identity. Local downloader PID 10192. Code /home/stud1/cs781_d222_20261011; artifacts /media/hdd/SAMBA4080_left_Dataset/stud1/cs781_d222_20261011. queue_log_path.txt names the actual current log; queue_status.json and stage status files report progress. Verify PIDs/commands and GPU ownership before any restart.

222,079,872 unique parameters. Requested 300M supervised LM tokens, 50M pilot snapshot. Real calibration around 9.2k tokens/s, ~5GB peak allocated GPU memory; measured rates can change. Reserve NER time and retain the existing best predictions. Pretraining initially selects by held-out LM loss. NER performs bounded full-fine-tuning candidate search, exports the promoted model, then separately trains/exports LoRMA. After Phase 2 delivery, official Phase 1 benchmark generation/scoring can continue beyond today's deadline because that phase is already closed; it is not uploaded.

Downloads: outputs/d222_20261011 and Windows Downloads. Each submission verifies SHA, CRC, root prediction.json, exact IDs and character/word contracts. Keep the PC awake/online for immediate downloads; remote training survives SSH disconnection.

Do not edit legal_d or vendor code beneath the live trainer. Source/config/data/tokenizer fingerprints gate restarts. On errors, preserve checkpoints and diagnose only this queue; never stop unrelated shared processes.

The old four-model Phase 2 queue is complete, verified and its heartbeat paused. D222 is separate. No D222 prediction ZIP or accuracy score exists at launch. Never report target tokens as achieved tokens, or a runtime gate as a measured score improvement.
