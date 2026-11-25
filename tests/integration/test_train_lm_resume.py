"""Integration smoke test for the training pipeline's resume behaviour.

Runs a couple of steps of real (tiny) training end-to-end against a
synthetic corpus + freshly trained tokenizer, entirely under tmp_path, to
guard against regressions in checkpointing / resumable training state
(packages are fast enough here that this stays a sub-second test).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch
from packages.tokenizer import BPETokenizer
from training.scripts import train_lm


def _write_corpus(corpus_dir: Path) -> None:
    corpus_dir.mkdir(parents=True, exist_ok=True)
    text = "the quick brown fox jumps over the lazy dog. " * 200
    (corpus_dir / "sample.txt").write_text(text, encoding="utf-8")


def _train_tiny_tokenizer(corpus_dir: Path, tokenizer_path: Path) -> None:
    text = (corpus_dir / "sample.txt").read_text(encoding="utf-8")
    tokenizer = BPETokenizer.train([text], vocab_size=300)
    tokenizer.save(tokenizer_path)


def _run(monkeypatch, argv: list[str]) -> None:
    monkeypatch.setattr(sys, "argv", ["train_lm.py", *argv])
    train_lm.main()


def test_train_lm_resume_round_trip(tmp_path: Path, monkeypatch) -> None:
    corpus_dir = tmp_path / "corpus"
    tokenizer_path = tmp_path / "tokenizer.json"
    output = tmp_path / "ckpt" / "lm_test"

    _write_corpus(corpus_dir)
    _train_tiny_tokenizer(corpus_dir, tokenizer_path)

    common = [
        "--config",
        "tiny",
        "--seq-len",
        "8",
        "--batch-size",
        "2",
        "--val-ratio",
        "0.2",
        "--test-ratio",
        "0.2",
        "--eval-every",
        "3",
        "--save-every",
        "3",
        "--warmup-steps",
        "2",
        "--tokenizer",
        str(tokenizer_path),
        "--corpus-dir",
        str(corpus_dir),
        "--output",
        str(output),
        "--device",
        "cpu",
    ]

    # First run: 6 fresh steps.
    _run(monkeypatch, ["--steps", "6", *common])

    last_dir = tmp_path / "ckpt" / "lm_test_last"
    state_dir = tmp_path / "ckpt" / "lm_test_state"
    final_dir = tmp_path / "ckpt" / "lm_test_final"
    log_path = tmp_path / "ckpt" / "lm_test_logs" / "metrics.jsonl"

    assert (last_dir / "pytorch_model.bin").exists()
    assert (final_dir / "pytorch_model.bin").exists()
    assert (state_dir / "trainer_state.pt").exists()
    assert log_path.exists()

    state_after_first = torch.load(state_dir / "trainer_state.pt", weights_only=False)
    assert state_after_first["step"] == 6

    lines = [json.loads(line) for line in log_path.read_text(encoding="utf-8").splitlines()]
    assert lines[0]["event"] == "run_start"
    assert lines[0]["resumed"] is False
    assert any(line.get("event_final") for line in lines)
    test_events = [line for line in lines if line.get("event_test")]
    assert len(test_events) == 1
    assert test_events[0]["test_loss"] > 0

    # Second run: resume and train 6 more steps (to step 12).
    _run(monkeypatch, ["--steps", "12", "--resume", *common])

    state_after_second = torch.load(state_dir / "trainer_state.pt", weights_only=False)
    assert state_after_second["step"] == 12

    lines_after_resume = [
        json.loads(line) for line in log_path.read_text(encoding="utf-8").splitlines()
    ]
    # A second run_start line was appended, recording that it resumed from step 6.
    resume_headers = [line for line in lines_after_resume if line["event"] == "run_start"]
    assert len(resume_headers) == 2
    assert resume_headers[1]["resumed"] is True
    assert resume_headers[1]["start_step"] == 6
