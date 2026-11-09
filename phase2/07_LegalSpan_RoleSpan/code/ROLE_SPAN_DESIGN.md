# LegalSpan role-aware span experiment

User approved implementation and training on 11 October 2026. Own selected LegalSpan NER epoch-7 weights are merged from LoRMA with strict parity and tokenizer SHA checks. Existing document split retained.

Stage 1 predicts 15 coarse BIO classes: O plus B/I for role-bearing entities, authority, court, statute, precedent, date and case number. Role-bearing entities include organisations as parties and human names. A learned generic CRF and explicit boundaries train span detection.

Stage 2 pools each proposed role-bearing span, 32 characters of left/right context, learned document-header context and a length embedding, then classifies APP, RESP, A.COUNSEL, R.COUNSEL, JUDGE or WIT. Training uses gold training spans; validation/test uses spans proposed by the coarse CRF. Role logits initialize from the trained fine classifier. Fine BIO emissions combine spans and role posteriors and merge across windows for a final 25-class CRF. Auxiliary fine-label training retains the own NER warm start.

Repeated normalized names within a document contribute to role consistency through averaged learned scores, requiring a joint margin and updating only low-margin individual decisions. Observed text and model outputs only; no test annotations. Original offsets retained.

Backbone LR 5e-6, new head 3e-4, 65% rare-window sampling, up to sixteen epochs/patience four, three-hour cap followed by deadline-bounded export. GPU1 cap 44%, alongside identified own D222. Real forward/backward and merge parity gate training. Correct evaluator path vendor/vendor/evaluate_phase2.py is provided before deployment.

Output UPLOAD_Phase2_07_LegalSpan_RoleSpan.zip: only root prediction.json, 35 exact IDs, original offsets, 12 labels. Automatic local/Downloads verification. Goal 0.60+, not a predicted score. Compare with previous 0.411641 and ensemble 0.463067 validation F1. No external upload automated.
