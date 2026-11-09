# Published Phase-1 evaluator

Unmodified source fetched and read during the 2 October 2026 requirements audit.
Source: https://drive.google.com/file/d/1RgRgm_-u91IBkZVLszaEGRlRzLEwAv2f/view
Drive metadata: modified 17 September 2026; 4,341 bytes. The exact saved source hash is recorded in each evaluation report.
Saved SHA-256: `3f3a9e62be718b478e3f52b378fcc535350ab90b9cb84e3a4c49fec247abf1db` (4,341 bytes, verified locally).

Primary metric: bert_score.score(lang='en', batch_size=8), default English model/settings. Secondary metrics: raw-whitespace LCS ROUGE-L and unsmoothed corpus BLEU-4. Private Codabench scoring-image equivalence has not been independently established.

Unit tests stub only the unavailable bert_score import to inspect pure secondary-metric functions. They do not produce real BERTScore values. Integration evaluation runs the untouched script as a subprocess with its real dependencies and scoring weights.
