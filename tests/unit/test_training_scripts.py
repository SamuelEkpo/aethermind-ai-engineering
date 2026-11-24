"""
End-to-end tests for the small training CLI scripts.

Each test calls the script's main() with a patched sys.argv, exactly as a
user would from the command line, against real tiny artifacts in tmp_path.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch
from packages.model_core import AetherMindLM, ModelConfig
from packages.tokenizer import BPETokenizer
from training.scripts import evaluate_lm, evaluate_tokenizer, generate, train_tokenizer

TEXT = "hello world. hello there. def add(a, b): return a + b\n"


def _run(monkeypatch: pytest.MonkeyPatch, module_name: str, argv: list[str]) -> None:
    monkeypatch.setattr(sys, "argv", [module_name, *argv])
    {
        "train_tokenizer": train_tokenizer,
        "evaluate_tokenizer": evaluate_tokenizer,
        "generate": generate,
        "evaluate_lm": evaluate_lm,
    }[module_name].main()


@pytest.fixture(scope="module")
def artifacts(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Path]:
    """Tiny tokenizer, untrained model checkpoint and text corpus on disk."""
    root = tmp_path_factory.mktemp("cli_artifacts")
    tokenizer = BPETokenizer.train([TEXT * 20], vocab_size=300)
    tokenizer_path = root / "tokenizer.json"
    tokenizer.save(tokenizer_path)

    torch.manual_seed(0)
    config = ModelConfig(
        vocab_size=tokenizer.vocab_size,
        max_seq_len=32,
        d_model=32,
        n_layers=1,
        n_heads=2,
        d_ff=64,
        dropout=0.0,
    )
    checkpoint = root / "checkpoint"
    AetherMindLM(config).save_pretrained(checkpoint)

    corpus = root / "corpus"
    corpus.mkdir()
    (corpus / "sample.txt").write_text(TEXT * 100, encoding="utf-8")
    return {"tokenizer": tokenizer_path, "checkpoint": checkpoint, "corpus": corpus}


# ----------------------------------------------------------------------
# train_tokenizer
# ----------------------------------------------------------------------
def test_iter_corpus_reads_txt_files_in_sorted_order(tmp_path: Path) -> None:
    (tmp_path / "b.txt").write_text("second", encoding="utf-8")
    (tmp_path / "a.txt").write_text("first", encoding="utf-8")
    (tmp_path / "skip.md").write_text("ignored", encoding="utf-8")
    assert train_tokenizer.iter_corpus(tmp_path) == ["first", "second"]


def test_train_tokenizer_cli_writes_loadable_tokenizer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "a.txt").write_text(TEXT * 20, encoding="utf-8")
    output = tmp_path / "out" / "tokenizer.json"

    _run(
        monkeypatch,
        "train_tokenizer",
        ["--input-dir", str(corpus), "--vocab-size", "300", "--output", str(output)],
    )

    assert "Trained tokenizer" in capsys.readouterr().out
    tokenizer = BPETokenizer.load(output)
    assert tokenizer.vocab_size <= 300
    assert tokenizer.decode(tokenizer.encode("hello world.")) == "hello world."


def test_train_tokenizer_cli_exits_without_corpus(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(SystemExit):
        _run(
            monkeypatch,
            "train_tokenizer",
            ["--input-dir", str(empty), "--output", str(tmp_path / "t.json")],
        )


# ----------------------------------------------------------------------
# evaluate_tokenizer
# ----------------------------------------------------------------------
def test_evaluate_tokenizer_cli_reports_lossless_round_trip(
    artifacts: dict[str, Path],
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _run(monkeypatch, "evaluate_tokenizer", ["--tokenizer", str(artifacts["tokenizer"])])
    out = capsys.readouterr().out
    assert "Vocab size:" in out
    assert "Lossless round-trip: OK" in out
    assert "bytes/token" in out


# ----------------------------------------------------------------------
# generate
# ----------------------------------------------------------------------
def test_generate_cli_prints_a_completion(
    artifacts: dict[str, Path],
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    torch.manual_seed(0)
    _run(
        monkeypatch,
        "generate",
        [
            "--checkpoint",
            str(artifacts["checkpoint"]),
            "--tokenizer",
            str(artifacts["tokenizer"]),
            "--prompt",
            "hello",
            "--max-new-tokens",
            "4",
        ],
    )
    out = capsys.readouterr().out
    assert out.count("─" * 60) == 2


def test_generate_cli_handles_empty_prompt(
    artifacts: dict[str, Path],
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    torch.manual_seed(0)
    _run(
        monkeypatch,
        "generate",
        [
            "--checkpoint",
            str(artifacts["checkpoint"]),
            "--tokenizer",
            str(artifacts["tokenizer"]),
            "--prompt",
            "",
            "--max-new-tokens",
            "3",
        ],
    )
    assert capsys.readouterr().out.count("─" * 60) == 2


# ----------------------------------------------------------------------
# evaluate_lm
# ----------------------------------------------------------------------
def test_evaluate_lm_cli_reports_loss_and_perplexity(
    artifacts: dict[str, Path],
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _run(
        monkeypatch,
        "evaluate_lm",
        [
            "--checkpoint",
            str(artifacts["checkpoint"]),
            "--tokenizer",
            str(artifacts["tokenizer"]),
            "--corpus-dir",
            str(artifacts["corpus"]),
            "--seq-len",
            "8",
            "--batch-size",
            "2",
            "--max-batches",
            "2",
        ],
    )
    out = capsys.readouterr().out
    assert "tokens_evaluated:" in out
    assert "loss:" in out
    assert "perplexity:" in out


def test_evaluate_lm_cli_exits_without_corpus(
    artifacts: dict[str, Path], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(SystemExit):
        _run(
            monkeypatch,
            "evaluate_lm",
            [
                "--checkpoint",
                str(artifacts["checkpoint"]),
                "--tokenizer",
                str(artifacts["tokenizer"]),
                "--corpus-dir",
                str(empty),
            ],
        )
