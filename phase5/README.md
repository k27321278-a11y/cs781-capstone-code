# Phase 5: legal statute identification

The code under `source/` contains the archived B1 adaptation, sparse text classifier, hybrid blending/calibration, and other Phase 5 experiment scripts. The `notebooks/` directory holds available Kaggle source notebooks; `evidence/` holds compact JSON reports. Original checkpoints, course data, and large outputs are stored separately.

The strongest visible finished Codabench submission was `phase5_b1_sparse_hybrid_prediction.zip`, **0.3949 macro-F1**, on 28 October 2026. The saved run report describes a B1 LoRA branch blended with a full-document sparse classifier; the sparse weight was 0.9, selected on training-only grouped calibration. Its official-dev macro-F1 was 0.3909 and micro-F1 0.5000. These local values are separate from the hidden score. The related CV-calibrated variant scored 0.3944 hidden. The live task page called macro-F1 the primary metric even though a prompt in the supplied report template referred to micro-F1.

The source scripts may require the original B1 checkpoint/tokenizer, saved sparse model, environment packages and mounted course datasets. The compact repository snapshot is code and evidence, not a standalone trained model release.
