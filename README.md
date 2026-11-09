# CS781 capstone: recovered source and evidence

This repository is **private while being organized**. It currently covers Phase 1 legal language modelling, Phase 2 legal named entity recognition, and Phase 4 court judgment prediction. Every source archive, result, and notebook is labeled by provenance. The large original checkpoint/data archives remain in separate local backups; the compact code and evidence packages here are intended for inspection and report preparation.

## Where to start

- [Phase 1 package index](packages/phase1/README_FIRST.md): eight lineages, original source/config, A0 historical Colab notebook, later illustrated notebooks, reports, diagrams, and compact logs/metrics. The D222 extension is marked post-deadline.
- [Phase 2 package index](packages/phase2/README_FIRST.md): twelve prediction exports across five backbone families, including one ensemble and two post-deadline checkpoint studies.
- [Phase 4 package index](phase4/model_packages/README.md): six initial Kaggle adaptations, four B1 head variants, and three later probability stacks, each with source, evidence, an illustrated companion notebook, PDF/LaTeX methodology, and a verified prediction ZIP.
- [Phase 1 best-model report](report/phase1_best_legalspan/Phase1_LegalSpan_Report.pdf): LegalSpan technical report with Markdown and editable LaTeX source.

Each lineage directory contains `source/` recovered from its original complete archive and a `code/` source snapshot inside the reproducibility package structure. Its README, technical reference, line-level code map, illustrated notebook, figure, report PDF/Markdown/LaTeX, and small evidence records live beside the code. The same materials are downloadable as per-model and all-model ZIPs under `packages/`.

## Evidence rules

Historical Codabench screenshot numbers, locally evaluated development scores, and hidden-test scores are different evidence classes. A valid prediction ZIP is not proof of platform submission. No notebook assembled after training is shown as an originally executed notebook. The recovered run logs and metric files are preserved separately in each `evidence/` folder. The seven October 6 Phase 1 complete archives contained no `.ipynb`; the A0 shared-chat package separately contained a Colab code notebook. D222 has an original code notebook in its own source archive.

## Reproduction and disclosure

Original scripts may contain paths from the former GPU server. Reproduction requires the official course data, the exact tokenizer, selected weights, environment dependencies, and path changes. Multi-gigabyte checkpoints and official corpus copies are not in GitHub. The final capstone report and public release must follow the course's submission and LLM-use declaration requirements; this repository is not yet the complete all-phase final submission.
