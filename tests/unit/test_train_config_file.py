"""
Tests for the --config-file feature of training.scripts.train_lm.

Covers TOML loading, precedence (command line over file over defaults),
error handling, and validity of the shipped preset files.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pytest
from packages.tokenizer import BPETokenizer
from training.scripts import train_lm

PRESET_DIR = Path(__file__).resolve().parents[2] / "training" / "configs"


def _mini_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="tiny")
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--output", type=Path, default=Path("default_out"))
    parser.add_argument("--config-file", type=Path, default=None)
    return parser


def _write(tmp_path: Path, text: str) -> Path:
    path = tmp_path / "run.toml"
    path.write_text(text, encoding="utf-8")
    return path


# ----------------------------------------------------------------------
# load_config_file
# ----------------------------------------------------------------------
def test_load_config_file_accepts_dashes_and_underscores(tmp_path: Path) -> None:
    path = _write(tmp_path, 'steps = 10\nbatch-size = 4\neval_every = 3\n')
    assert train_lm.load_config_file(path) == {"steps": 10, "batch_size": 4, "eval_every": 3}


def test_load_config_file_missing_file_exits(tmp_path: Path) -> None:
    with pytest.raises(SystemExit):
        train_lm.load_config_file(tmp_path / "nope.toml")


def test_load_config_file_invalid_toml_exits(tmp_path: Path) -> None:
    with pytest.raises(SystemExit):
        train_lm.load_config_file(_write(tmp_path, "steps = = 3"))


# ----------------------------------------------------------------------
# parse_args_with_config_file
# ----------------------------------------------------------------------
def test_without_config_file_uses_builtin_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "argv", ["train_lm.py"])
    args = train_lm.parse_args_with_config_file(_mini_parser())
    assert args.steps == 5
    assert args.config_file is None


def test_file_values_replace_defaults(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = _write(tmp_path, 'steps = 99\noutput = "from_file"\n')
    monkeypatch.setattr(sys, "argv", ["train_lm.py", "--config-file", str(path)])
    args = train_lm.parse_args_with_config_file(_mini_parser())
    assert args.steps == 99
    assert args.output == Path("from_file")
    assert args.batch_size == 2  # untouched default
    assert args.config_file == path


def test_command_line_overrides_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = _write(tmp_path, "steps = 99\nbatch_size = 8\n")
    monkeypatch.setattr(
        sys, "argv", ["train_lm.py", "--config-file", str(path), "--steps", "7"]
    )
    args = train_lm.parse_args_with_config_file(_mini_parser())
    assert args.steps == 7  # command line wins
    assert args.batch_size == 8  # file still applies


def test_unknown_key_in_file_exits(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = _write(tmp_path, "stepz = 10\n")
    monkeypatch.setattr(sys, "argv", ["train_lm.py", "--config-file", str(path)])
    with pytest.raises(SystemExit) as excinfo:
        train_lm.parse_args_with_config_file(_mini_parser())
    assert "stepz" in str(excinfo.value)


def test_config_file_key_inside_file_is_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = _write(tmp_path, 'config_file = "other.toml"\n')
    monkeypatch.setattr(sys, "argv", ["train_lm.py", "--config-file", str(path)])
    with pytest.raises(SystemExit):
        train_lm.parse_args_with_config_file(_mini_parser())


def test_invalid_model_preset_name_in_file_exits(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = _write(tmp_path, 'config = "gigantic"\n')
    monkeypatch.setattr(sys, "argv", ["train_lm.py", "--config-file", str(path)])
    with pytest.raises(SystemExit):
        train_lm.parse_args_with_config_file(_mini_parser())


def test_model_preset_flag_is_not_mistaken_for_config_file(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Regression: argparse prefix matching once read "--config" as "--config-file".
    monkeypatch.setattr(sys, "argv", ["train_lm.py", "--config", "tiny"])
    args = train_lm.parse_args_with_config_file(train_lm.build_parser())
    assert args.config == "tiny"
    assert args.config_file is None


# ----------------------------------------------------------------------
# Shipped presets
# ----------------------------------------------------------------------
@pytest.mark.parametrize("name", ["tiny", "small", "base"])
def test_shipped_presets_are_valid(name: str) -> None:
    values = train_lm.load_config_file(PRESET_DIR / f"{name}.toml")
    valid_keys = set(vars(train_lm.build_parser().parse_args([]))) - {"config_file"}

    assert set(values) <= valid_keys
    assert values["config"] == name
    assert values["config"] in train_lm.CONFIGS
    assert values["seq_len"] <= train_lm.CONFIGS[name].max_seq_len
    assert values["val_ratio"] + values["test_ratio"] < 1.0
    assert values["warmup_steps"] < values["steps"]


def test_training_run_with_config_file_records_it_and_honours_overrides(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    text = "hello world. hello there. " * 100
    (corpus / "sample.txt").write_text(text, encoding="utf-8")
    tokenizer_path = tmp_path / "tokenizer.json"
    BPETokenizer.train([text], vocab_size=300).save(tokenizer_path)
    output = tmp_path / "ckpt" / "lm"

    argv = [
        "train_lm.py",
        "--config-file",
        str(PRESET_DIR / "tiny.toml"),
        "--steps",
        "2",
        "--warmup-steps",
        "1",
        "--eval-every",
        "100",
        "--batch-size",
        "2",
        "--seq-len",
        "8",
        "--tokenizer",
        str(tokenizer_path),
        "--corpus-dir",
        str(corpus),
        "--output",
        str(output),
    ]
    monkeypatch.setattr(sys, "argv", argv)
    train_lm.main()

    log_path = tmp_path / "ckpt" / "lm_logs" / "metrics.jsonl"
    header = json.loads(log_path.read_text(encoding="utf-8").splitlines()[0])
    assert header["config_file"].endswith("tiny.toml")
    assert header["steps"] == 2  # command line overrides the file's 500
    assert header["val_ratio"] == 0.1  # taken from the file
    assert header["seed"] == 42
