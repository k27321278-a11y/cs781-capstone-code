# LegalSpan Phase 2 improvement

Authorized after a measured two-of-four exact-span ensemble improved held-out macro F1 from 0.411641 to 0.463067. No hidden-score guarantee.

Starts from the selected own LegalSpan LoRMA NER checkpoint (epoch 7), merges its rank-8 multiplicative adapters into the backbone with a checked functional equivalence, and fully fine-tunes at 1e-5. The existing trained character BiGRU/BIO classifier is retained. Adds attention to a learned BiGRU representation of the document's first 1024 original characters, a learned constrained BIO CRF, and auxiliary start/end losses. Header residual is initially zero, preserving the warmed classifier before new training. Rare-entity windows receive a 50% sampling mixture. New head learning rate 2e-4. Original 58/12 document split and exact 35 test IDs retained.

Training is bounded to three hours with at most twelve epochs and patience three, followed by validation/offset-correct export. Physical GPU1 allocation is capped to 44% of device memory; concurrent own D222 training can slow both jobs. No unrelated process is stopped. Preflight exercises a real full-backbone forward/backward and adapter merge parity before training. Actual hardware, checkpoint/tokenizer/source hashes, learned parameters, validation selection and exact prediction ZIP are recorded.

Source primitives are isolated copies of the existing CS781 own code. No external pretrained weights or official test labels are loaded. LoRMA: Bihany, Patel and Modi (2025); CRF sequence tagging follows established BIO linear-chain modelling. These changes require new training. The ensemble ZIP is independent and requires none.
