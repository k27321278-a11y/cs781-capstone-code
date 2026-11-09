from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from torch import Tensor

SOURCES = ("acts", "high_courts", "supreme_court", "tribunals")
SOURCE_PATHS = {s: f"Phase_1_Pretraining/Train/train_{s}.json" for s in SOURCES}
SPECIAL_TOKENS = ["<pad>", "<bos>", "<eos>", "<unk>"]


class SourceReadError(RuntimeError): pass
class ProvenanceError(RuntimeError): pass
class IncompleteDataError(RuntimeError): pass
class CheckpointMismatch(RuntimeError): pass
class GenerationIncomplete(RuntimeError): pass
class PromptTooLong(RuntimeError): pass
class RunAlreadyActive(RuntimeError): pass


@dataclass(frozen=True)
class RawDocument:
    id: str
    source: str
    text: str
    fingerprint: str


@dataclass(frozen=True)
class Document:
    raw: RawDocument
    cluster_id: str
    split: str


@dataclass(frozen=True)
class ContinuationExample:
    id: str
    document_id: str
    source: str
    split: str
    prefix_ids: list[int]
    suffix_ids: list[int]


@dataclass(frozen=True)
class CorpusConfig:
    revision: str
    repo_id: str = "Exploration-Lab/CS-781-Capstone"
    source_paths: dict[str, str] = field(default_factory=lambda: dict(SOURCE_PATHS))
    seed: int = 781
    max_chars_per_source: int | None = None

    def __post_init__(self):
        if self.repo_id != "Exploration-Lab/CS-781-Capstone" or not re.fullmatch(r"[0-9a-f]{40}", self.revision):
            raise ValueError("Use the official repository with a pinned commit SHA")
        if self.source_paths != SOURCE_PATHS:
            raise ValueError("Only the four official training source paths are allowed")
        if self.max_chars_per_source is not None and self.max_chars_per_source < 1:
            raise ValueError("Positive source character limit required")


@dataclass(frozen=True)
class PreparedData:
    root: Path
    manifest_sha256: str
    train_examples: int
    dev_examples: int
    confirmation_examples: int
    tokenizer_sha256: str


@dataclass(frozen=True)
class Batch:
    input_ids: Tensor
    targets: Tensor
    input_tokens: int
    target_tokens: int


@dataclass(frozen=True)
class TrainState:
    step: int = 0
    input_tokens: int = 0
    target_tokens: int = 0
    example_cursor: int = 0
    best_generation_f1: float | None = None


@dataclass(frozen=True)
class ModelConfig:
    vocab_size: int = 32768
    max_seq_len: int = 2048
    n_layers: int = 12
    d_model: int = 768
    n_heads: int = 12
    d_head: int = 64
    d_ff: int = 2048
    rope_theta: float = 10000.0
    dropout: float = 0.0
    gradient_checkpointing: bool = True

    def validate(self):
        if self.n_heads * self.d_head != self.d_model or self.d_head % 2:
            raise ValueError("Head dimensions must match model width and be even")
        if min(self.vocab_size, self.max_seq_len, self.n_layers, self.d_ff) < 1 or not 0 <= self.dropout < 1:
            raise ValueError("Invalid model configuration")


@dataclass(frozen=True)
class TrainConfig:
    target_tokens: int
    micro_batch_size: int = 1
    grad_accum_steps: int = 32
    learning_rate: float = 3e-4
    min_lr_ratio: float = .10
    warmup_ratio: float = .02
    weight_decay: float = .10
    beta1: float = .9
    beta2: float = .95
    eps: float = 1e-8
    grad_clip: float = 1.0
    save_every: int = 100
    log_every: int = 10
    seed: int = 781
    device: str = "cuda"


@dataclass(frozen=True)
class GenerationConfig:
    strategy: str = "greedy"
    temperature: float = .8
    top_p: float = .9
    repetition_penalty: float = 1.0
    target_words: int = 128
    max_new_tokens: int = 1024
    seed: int = 781
