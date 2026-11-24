"""
Unit tests for the pure helpers and error paths of training.scripts.train_lm.

The end-to-end resume round trip lives in tests/integration; these tests
cover the schedule, data splitting, dataset windows, metrics log, trainer
state persistence and the user-facing failure modes.
"""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path

import pytest
import torch
from packages.tokenizer import BPETokenizer
from training.scripts import train_lm


# ----------------------------------------------------------------------
# cosine_lr
# ----------------------------------------------------------------------
def test_cosine_lr_warms_up_linearly() -> None:
    assert train_lm.cosine_lr(0, 100, 1.0, warmup_steps=10) == pytest.approx(0.1)
    assert train_lm.cosine_lr(4, 100, 1.0, warmup_steps=10) == pytest.approx(0.5)
    assert train_lm.cosine_lr(9, 100, 1.0, warmup_steps=10) == pytest.approx(1.0)


def test_cosine_lr_peaks_after_warmup_and_decays_to_floor() -> None:
    assert train_lm.cosine_lr(10, 110, 1.0, warmup_steps=10) == pytest.approx(1.0)
    assert train_lm.cosine_lr(110, 110, 1.0, warmup_steps=10) == pytest.approx(0.1)
    assert train_lm.cosine_lr(110, 110, 2.0, warmup_steps=10, min_lr_ratio=0.5) == pytest.approx(
        1.0
    )


def test_cosine_lr_never_increases_after_warmup() -> None:
    values = [train_lm.cosine_lr(s, 200, 1.0, warmup_steps=20) for s in range(20, 201)]
    assert all(a >= b - 1e-12 for a, b in zip(values, values[1:], strict=False))


def test_cosine_lr_handles_zero_warmup() -> None:
    assert train_lm.cosine_lr(0, 10, 1.0, warmup_steps=0) == pytest.approx(1.0)


# ----------------------------------------------------------------------
# split_train_val_test
# ----------------------------------------------------------------------
def test_split_is_chronological_and_lossless() -> None:
    ids = list(range(1000))
    train, val, test = train_lm.split_train_val_test(ids, 0.1, 0.2)
    assert train + val + test == ids
    assert len(train) == pytest.approx(700, abs=1)
    assert len(val) == pytest.approx(100, abs=1)
    assert len(test) == pytest.approx(200, abs=1)


def test_split_rejects_ratios_that_leave_no_training_data() -> None:
    with pytest.raises(SystemExit):
        train_lm.split_train_val_test(list(range(100)), 0.5, 0.5)


# ----------------------------------------------------------------------
# TextDataset / has_windows
# ----------------------------------------------------------------------
def test_text_dataset_windows_are_shifted_by_one() -> None:
    ds = train_lm.TextDataset(list(range(10)), seq_len=4)
    assert len(ds) == 6
    first = ds[0]
    assert first["input_ids"].tolist() == [0, 1, 2, 3]
    assert first["labels"].tolist() == [1, 2, 3, 4]
    last = ds[len(ds) - 1]
    assert last["input_ids"].tolist() == [5, 6, 7, 8]
    assert last["labels"].tolist() == [6, 7, 8, 9]


def test_text_dataset_too_short_is_empty() -> None:
    assert len(train_lm.TextDataset([1, 2, 3], seq_len=3)) == 0
    assert len(train_lm.TextDataset([], seq_len=3)) == 0


def test_has_windows() -> None:
    assert train_lm.has_windows(list(range(10)), 4) is True
    assert train_lm.has_windows(list(range(4)), 4) is False


# ----------------------------------------------------------------------
# load_token_ids
# ----------------------------------------------------------------------
def _make_tokenizer(tmp_path: Path, text: str = "hello world. hello there.") -> Path:
    path = tmp_path / "tokenizer.json"
    BPETokenizer.train([text * 5], vocab_size=300).save(path)
    return path


def test_load_token_ids_reads_files_in_sorted_order(tmp_path: Path) -> None:
    tokenizer_path = _make_tokenizer(tmp_path)
    corpus = tmp_path / "corpus"
    (corpus / "nested").mkdir(parents=True)
    (corpus / "b.txt").write_text("hello there.", encoding="utf-8")
    (corpus / "a.txt").write_text("hello world.", encoding="utf-8")
    (corpus / "nested" / "c.txt").write_text("hello", encoding="utf-8")
    (corpus / "ignored.md").write_text("not a txt file", encoding="utf-8")

    tokenizer = BPETokenizer.load(tokenizer_path)
    expected = (
        tokenizer.encode("hello world.")
        + tokenizer.encode("hello there.")
        + tokenizer.encode("hello")
    )
    # rglob + sorted orders by full path: a.txt, b.txt, nested/c.txt
    assert train_lm.load_token_ids(tokenizer_path, corpus) == expected


def test_load_token_ids_without_txt_files_exits(tmp_path: Path) -> None:
    tokenizer_path = _make_tokenizer(tmp_path)
    corpus = tmp_path / "empty_corpus"
    corpus.mkdir()
    with pytest.raises(SystemExit):
        train_lm.load_token_ids(tokenizer_path, corpus)


# ----------------------------------------------------------------------
# MetricsLogger
# ----------------------------------------------------------------------
def test_metrics_logger_writes_header_then_steps(tmp_path: Path) -> None:
    path = tmp_path / "logs" / "metrics.jsonl"
    logger = train_lm.MetricsLogger(path, {"seed": 7})
    logger.log(step=1, train_loss=2.5)
    logger.log(step=2, train_loss=2.0)

    lines = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert lines[0] == {"event": "run_start", "seed": 7}
    assert lines[1] == {"event": "step", "step": 1, "train_loss": 2.5}
    assert lines[2]["step"] == 2


def test_metrics_logger_appends_across_runs(tmp_path: Path) -> None:
    path = tmp_path / "metrics.jsonl"
    train_lm.MetricsLogger(path, {"run": 1})
    train_lm.MetricsLogger(path, {"run": 2})
    lines = path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2


# ----------------------------------------------------------------------
# trainer state persistence
# ----------------------------------------------------------------------
def test_training_state_round_trip_restores_step_optimizer_and_rng(tmp_path: Path) -> None:
    model = torch.nn.Linear(2, 2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    model(torch.ones(1, 2)).sum().backward()
    optimizer.step()

    train_lm.save_training_state(tmp_path, step=7, best_val=1.5, optimizer=optimizer, seed=42)
    expected_torch = torch.rand(1)
    expected_python = random.random()

    fresh = torch.optim.AdamW(torch.nn.Linear(2, 2).parameters(), lr=1e-3)
    step, best_val = train_lm.load_training_state(tmp_path, fresh)

    assert (step, best_val) == (7, 1.5)
    assert len(fresh.state_dict()["state"]) > 0
    assert torch.equal(torch.rand(1), expected_torch)
    assert random.random() == expected_python


def test_set_seed_is_deterministic() -> None:
    train_lm.set_seed(1)
    first = (torch.rand(3), random.random())
    train_lm.set_seed(1)
    second = (torch.rand(3), random.random())
    assert torch.equal(first[0], second[0])
    assert first[1] == second[1]


# ----------------------------------------------------------------------
# main(): user-facing failure and fallback paths
# ----------------------------------------------------------------------
def _base_args(tmp_path: Path, tokenizer_path: Path, corpus: Path) -> list[str]:
    return [
        "--config",
        "tiny",
        "--seq-len",
        "8",
        "--batch-size",
        "2",
        "--eval-every",
        "100",
        "--tokenizer",
        str(tokenizer_path),
        "--corpus-dir",
        str(corpus),
        "--output",
        str(tmp_path / "ckpt" / "lm"),
        "--device",
        "cpu",
    ]


def test_main_exits_when_training_split_is_too_short(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    tokenizer_path = _make_tokenizer(tmp_path)
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "tiny.txt").write_text("hi", encoding="utf-8")

    argv = ["train_lm.py", "--steps", "1", *_base_args(tmp_path, tokenizer_path, corpus)]
    monkeypatch.setattr(sys, "argv", argv)
    with pytest.raises(SystemExit):
        train_lm.main()


def test_main_resume_without_checkpoint_starts_fresh(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    tokenizer_path = _make_tokenizer(tmp_path)
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "sample.txt").write_text("hello world. hello there. " * 100, encoding="utf-8")

    argv = [
        "train_lm.py",
        "--steps",
        "2",
        "--resume",
        *_base_args(tmp_path, tokenizer_path, corpus),
    ]
    monkeypatch.setattr(sys, "argv", argv)
    train_lm.main()

    log_path = tmp_path / "ckpt" / "lm_logs" / "metrics.jsonl"
    header = json.loads(log_path.read_text(encoding="utf-8").splitlines()[0])
    assert header["event"] == "run_start"
    assert header["resumed"] is False
    assert header["start_step"] == 0
    assert (tmp_path / "ckpt" / "lm_final" / "pytorch_model.bin").exists()
