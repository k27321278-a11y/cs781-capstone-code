import torch
import pytest

from cs781.cli_b import _require_complete_training
from cs781.contracts import ContinuationExample
from cs781.examples_alt import collate_b
from cs781.io import sha256_file, write_json_atomic
from cs781.models.b import EncoderDecoderConfig, EncoderDecoderLegalModel


def tiny_model():
    cfg = EncoderDecoderConfig(
        vocab_size=64, max_seq_len=32, n_encoder_layers=2, n_decoder_layers=2,
        d_model=32, n_heads=4, d_head=8, d_ff=64,
        dropout=0.0, gradient_checkpointing=False, pad_token_id=0,
    )
    return EncoderDecoderLegalModel(cfg).eval()


def test_b_collator_separates_prompt_and_shifted_suffix():
    example = ContinuationExample(
        id="x", document_id="d", source="acts", split="train",
        prefix_ids=[11, 12, 13], suffix_ids=[21, 22],
    )
    batch = collate_b([example], pad_id=0, bos_id=1, eos_id=2, max_seq_len=16)
    assert batch.prompt_ids.tolist() == [[1, 11, 12, 13]]
    assert batch.decoder_ids.tolist() == [[1, 21, 22]]
    assert batch.targets.tolist() == [[21, 22, 2]]
    assert batch.prompt_mask.tolist() == [[True, True, True, True]]
    assert batch.target_tokens == 3


def test_b_decoder_future_tokens_do_not_change_earlier_logits():
    torch.manual_seed(781)
    model = tiny_model()
    prompt = torch.tensor([[1, 5, 6, 7]])
    left = torch.tensor([[1, 8, 9, 10]])
    changed_future = torch.tensor([[1, 8, 20, 21]])
    first, _, _ = model(prompt, left)
    second, _, _ = model(prompt, changed_future)
    torch.testing.assert_close(first[:, :2], second[:, :2], atol=1e-6, rtol=1e-6)


def test_b_cached_generation_matches_full_decoder_pass():
    torch.manual_seed(782)
    model = tiny_model()
    prompt = torch.tensor([[1, 5, 6, 7]])
    decoder = torch.tensor([[1, 8, 9, 10]])
    full, _, _ = model(prompt, decoder)

    pieces, cache = [], None
    for index in range(decoder.shape[1]):
        if cache is None:
            logits, _, cache = model(prompt, decoder[:, index:index + 1], use_cache=True)
        else:
            logits, _, cache = model(None, decoder[:, index:index + 1], past_kvs=cache, use_cache=True)
        pieces.append(logits)
    torch.testing.assert_close(torch.cat(pieces, dim=1), full, atol=2e-5, rtol=2e-5)


def test_b_prompt_padding_is_masked_from_decoder_memory():
    torch.manual_seed(783)
    model = tiny_model()
    mask = torch.tensor([[True, True, False, False]])
    prompt_a = torch.tensor([[1, 5, 0, 0]])
    prompt_b = torch.tensor([[1, 5, 31, 42]])
    decoder = torch.tensor([[1, 8, 9]])
    a, _, _ = model(prompt_a, decoder, prompt_mask=mask)
    b, _, _ = model(prompt_b, decoder, prompt_mask=mask)
    torch.testing.assert_close(a, b, atol=1e-6, rtol=1e-6)


def test_b_target_loss_ignores_unsupervised_positions():
    torch.manual_seed(784)
    model = tiny_model()
    prompt = torch.tensor([[1, 5, 6]])
    decoder = torch.tensor([[1, 8, 9]])
    targets = torch.tensor([[-100, 9, 2]])
    _, loss, _ = model(prompt, decoder, targets)
    logits, _, _ = model(prompt, decoder)
    expected = torch.nn.functional.cross_entropy(
        logits[:, 1:].float().reshape(-1, model.cfg.vocab_size),
        targets[:, 1:].reshape(-1),
    )
    torch.testing.assert_close(loss, expected)


def test_b_config_rejects_invalid_rope_theta():
    with pytest.raises(ValueError, match="theta"):
        EncoderDecoderConfig(rope_theta=0).validate()


def test_b_official_submission_gate_requires_completed_checkpoint(tmp_path):
    run_dir = tmp_path / "run"
    checkpoint = run_dir / "checkpoints" / "latest.pt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"trusted test fixture")
    metadata = {
        "synthetic_smoke": False,
        "train_config": {"target_tokens": 100},
        "code_sha256": "test-code-digest",
    }
    payload = {"state": {"target_tokens": 99}}
    with pytest.raises(ValueError, match="has not reached"):
        _require_complete_training(checkpoint, payload, run_dir, metadata)


def test_b_official_submission_gate_accepts_matching_complete_record(tmp_path):
    run_dir = tmp_path / "run"
    checkpoint = run_dir / "checkpoints" / "latest.pt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"trusted test fixture")
    metadata = {
        "synthetic_smoke": False,
        "train_config": {"target_tokens": 100},
        "code_sha256": "test-code-digest",
    }
    state = {"target_tokens": 100, "step": 1}
    write_json_atomic(run_dir / "training_status.json", {
        "architecture": "B", "status": "complete", "target_tokens": 100,
        "state": state, "code_sha256": metadata["code_sha256"],
        "checkpoint_sha256": sha256_file(checkpoint),
    })
    _require_complete_training(checkpoint, {"state": state}, run_dir, metadata)


def test_b_default_size_is_in_allowed_foundation_range():
    # Meta tensors count the tied embedding once without allocating ~500 MB on CPU.
    with torch.device("meta"):
        model = EncoderDecoderLegalModel(EncoderDecoderConfig())
    assert model.count_parameters() == 124_280_832
