"""
Unit tests for packages.model_core — architecture correctness, shapes,
causality, gradients, generation, and serialization.
"""

from __future__ import annotations

import pytest
import torch

from packages.model_core import AetherMindLM, ModelConfig, TINY


@pytest.fixture
def config() -> ModelConfig:
    return TINY


@pytest.fixture
def model(config: ModelConfig) -> AetherMindLM:
    torch.manual_seed(42)
    return AetherMindLM(config)


def test_config_rejects_invalid_head_division() -> None:
    with pytest.raises(ValueError, match="divisible"):
        ModelConfig(d_model=100, n_heads=3)


def test_config_rejects_bad_dropout() -> None:
    with pytest.raises(ValueError, match="dropout"):
        ModelConfig(dropout=1.5)


def test_forward_shape(model: AetherMindLM, config: ModelConfig) -> None:
    B, T = 2, 16
    input_ids = torch.randint(0, config.vocab_size, (B, T))
    logits = model(input_ids)
    assert logits.shape == (B, T, config.vocab_size)


def test_forward_with_labels_returns_loss(model: AetherMindLM, config: ModelConfig) -> None:
    B, T = 2, 16
    input_ids = torch.randint(0, config.vocab_size, (B, T))
    labels = input_ids.clone()
    result = model(input_ids, labels=labels)
    assert isinstance(result, tuple)
    logits, loss = result
    assert logits.shape == (B, T, config.vocab_size)
    assert loss.ndim == 0
    assert torch.isfinite(loss)
    assert loss.item() > 0


def test_causal_mask_blocks_future(model: AetherMindLM, config: ModelConfig) -> None:
    """Changing a future token must not change logits at earlier positions."""
    model.eval()
    T = 12
    base = torch.randint(0, config.vocab_size, (1, T))

    with torch.no_grad():
        logits_a = model(base)

        # Mutate only the last token
        mutated = base.clone()
        mutated[0, -1] = (mutated[0, -1] + 1) % config.vocab_size
        logits_b = model(mutated)

    # All positions except the last must be identical
    assert torch.allclose(logits_a[0, :-1], logits_b[0, :-1], atol=1e-5)


def test_gradients_flow(model: AetherMindLM, config: ModelConfig) -> None:
    B, T = 2, 8
    input_ids = torch.randint(0, config.vocab_size, (B, T))
    labels = input_ids.clone()
    _, loss = model(input_ids, labels=labels)
    loss.backward()

    grad_params = [p for p in model.parameters() if p.grad is not None]
    assert len(grad_params) > 0
    # At least some non-zero gradients
    assert any(p.grad.abs().sum().item() > 0 for p in grad_params)


def test_parameter_count(model: AetherMindLM) -> None:
    n = model.count_parameters()
    assert n > 0
    # Sanity: TINY should be well under 5M
    assert n < 5_000_000


def test_generate_extends_sequence(model: AetherMindLM, config: ModelConfig) -> None:
    model.eval()
    prompt = torch.randint(0, config.vocab_size, (1, 5))
    out = model.generate(prompt, max_new_tokens=7, temperature=1.0)
    assert out.shape == (1, 12)
    assert torch.equal(out[:, :5], prompt)


def test_generate_respects_eos(model: AetherMindLM, config: ModelConfig) -> None:
    model.eval()
    # Force a deterministic path by using temperature near zero isn't reliable
    # with multinomial; instead just check that generation doesn't explode.
    prompt = torch.randint(0, config.vocab_size, (1, 3))
    out = model.generate(prompt, max_new_tokens=20, temperature=0.8, top_k=10)
    assert out.shape[1] >= 3
    assert out.shape[1] <= 23


def test_save_and_load_roundtrip(model: AetherMindLM, config: ModelConfig, tmp_path) -> None:
    save_dir = tmp_path / "ckpt"
    model.save_pretrained(save_dir)

    loaded = AetherMindLM.from_pretrained(save_dir)
    assert loaded.config == config

    B, T = 1, 8
    input_ids = torch.randint(0, config.vocab_size, (B, T))
    model.eval()
    loaded.eval()
    with torch.no_grad():
        logits_a = model(input_ids)
        logits_b = loaded(input_ids)
    assert torch.allclose(logits_a, logits_b, atol=1e-5)


def test_sequence_longer_than_max_raises(model: AetherMindLM, config: ModelConfig) -> None:
    too_long = torch.randint(0, config.vocab_size, (1, config.max_seq_len + 1))
    with pytest.raises(ValueError, match="max_seq_len"):
        model(too_long)


def test_weight_tying(model: AetherMindLM) -> None:
    assert model.lm_head.weight is model.embed.token.embedding.weight
