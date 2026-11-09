# CS781 Phase 2 L-NER design

The user approved four experiments and a sequential physical-GPU1 queue with automatic prediction ZIP creation and local downloads. The fourth experiment uses the original A0 checkpoint; the count of A0 and LegalSpan as distinct architecture families is not established. This work cannot certify four architectural families.

## Experiments

1. LegalSpan pretrained backbone, LORMA adaptation, contextual NER classification head.
2. B1 pretrained encoder-decoder backbone, LoRA adaptation, NER head on token representations.
3. C1 pretrained hybrid backbone, full fine-tuning, NER head.
4. A0 pretrained backbone as smaller student, supervised NER plus teacher-distribution distillation. Use B1 as the larger teacher if its validation predictions pass checks; teacher data excludes held-out validation documents.

## Data and predictions

Only official Phase_2_LNER files. A deterministic document-level validation split from the two labelled folds is shared across experiments. Token offsets refer to the original unchanged Unicode string. Chunking includes overlap; gold alignment and predictions preserve exact character offsets. Output is all official fold3 IDs mapped to [start,end,label] lists. Every submission ZIP contains only root prediction.json, passes official evaluator format checks, and receives SHA-256 provenance.

## Reliability

Validate checkpoint architecture/tokenizer pairing and load weights strictly. Fine-tuning starts from the saved pretrained weights. Validate forward/backward, adapter initialization and gradients, gold character-offset round-trip, strict span scoring, ZIP structure, restart behavior, and full test-ID coverage before launch. Physical GPU1 only, no modification or signals to unrelated processes. Keep completed stage artifacts and resumable checkpoints. Each stage exports its ZIP before the next stage starts. Local downloader retries network outages and verifies remote and local hashes; an offline/asleep computer catches up when it reconnects.

## Provenance

Record actual hardware, configurations, source and checkpoint hashes, split IDs, data hashes, training curves, local metrics, and status. A T4 portability configuration does not constitute a measured T4 run. Never fabricate hardware or Codabench scores.
