# Architecture and assistance disclosure

This experiment is implemented with our own byte-BPE learner/encoder/decoder and manually coded participant model components. It starts from random weights on official CS781 data. The trained BERTScore encoder is used only by the published evaluator, never as participant initialization or a training teacher.

Established ideas studied: byte BPE (Sennrich et al., ACL2016, https://aclanthology.org/P16-1162/); RMSNorm/RoPE/SwiGLU causal backbone as used by LLaMA (https://arxiv.org/abs/2302.13971); pointer/generation mixture (https://aclanthology.org/P17-1099/ and CopyNet https://aclanthology.org/P16-1154/); query-conditioned attention gating (https://arxiv.org/abs/2505.06708); legal copy confidence as a related precedent (CoCoLex https://arxiv.org/abs/2508.05534). Educational minbpe was studied, not imported as our codec.

Our tested modification: prompt-only route in upper blocks jointly conditions on text-derived role cues and observed-history span continuity, gates an internal residual update, then supplies optional output copy mass. Full vs generic route and dense control share own BPE, data stream and target budgets. Roles are heuristics, not entity annotations. This is a course architectural hypothesis, not a claim of world-first invention or guaranteed quality gain.

LLM assistance: OpenAI Codex helped design, implement, review and debug this project under the user's instructor clarification permitting full coding assistance. The student must understand the model, mathematical contracts, citations, experimental limitations and implementation for the viva. The separate research/proposal files record studied sources and motivation.
