"""Unit tests for repository indexing and retrieval (pure Python, no torch)."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
from packages.code_intelligence import (
    BM25Index,
    Chunk,
    RepoIndex,
    chunk_text,
    split_identifier,
    tokenize,
)

# Assembled at runtime so this source file never contains a literal credential.
FAKE_AWS_KEY = "AKIA" + "A" * 16


# ----------------------------------------------------------------------
# Tokenizer
# ----------------------------------------------------------------------


@pytest.mark.parametrize(
    ("identifier", "expected"),
    [
        ("parseHttpRequest", ["parse", "http", "request"]),
        ("HTTPServer", ["http", "server"]),
        ("snake_case_name", ["snake", "case", "name"]),
        ("__init__", ["init"]),
        ("PascalCase", ["pascal", "case"]),
        ("plain", ["plain"]),
    ],
)
def test_split_identifier(identifier: str, expected: list[str]) -> None:
    assert split_identifier(identifier) == expected


def test_tokenize_emits_full_identifier_and_subwords() -> None:
    assert tokenize("parseHttpRequest") == ["parsehttprequest", "parse", "http", "request"]
    assert tokenize("parse_request") == ["parserequest", "parse", "request"]


def test_tokenize_plain_word_is_not_duplicated() -> None:
    assert tokenize("Foo") == ["foo"]


def test_tokenize_drops_single_chars_and_numbers() -> None:
    assert tokenize("x = 42 + y") == []
    assert tokenize("getX") == ["getx", "get"]


# ----------------------------------------------------------------------
# Chunker
# ----------------------------------------------------------------------


def test_chunk_text_overlaps_and_covers_all_lines() -> None:
    text = "\n".join(f"line{i}" for i in range(1, 11))
    chunks = chunk_text("a.py", text, max_lines=4, overlap=1)
    assert [(c.start_line, c.end_line) for c in chunks] == [(1, 4), (4, 7), (7, 10)]
    assert chunks[0].text.splitlines() == ["line1", "line2", "line3", "line4"]
    assert all(c.path == "a.py" for c in chunks)


def test_chunk_text_single_chunk_when_short() -> None:
    (only,) = chunk_text("a.py", "x\ny\nz", max_lines=40, overlap=8)
    assert (only.start_line, only.end_line) == (1, 3)


def test_chunk_text_drops_whitespace_only_chunks() -> None:
    chunks = chunk_text("a.py", "a\n\n\n\n\n\nb", max_lines=2, overlap=0)
    assert [(c.start_line, c.end_line) for c in chunks] == [(1, 2), (7, 7)]


def test_chunk_text_empty_input() -> None:
    assert chunk_text("a.py", "") == []


@pytest.mark.parametrize(("max_lines", "overlap"), [(0, 0), (4, 4), (4, -1), (4, 5)])
def test_chunk_text_rejects_bad_parameters(max_lines: int, overlap: int) -> None:
    with pytest.raises(ValueError):
        chunk_text("a.py", "x", max_lines=max_lines, overlap=overlap)


# ----------------------------------------------------------------------
# BM25
# ----------------------------------------------------------------------


def test_bm25_rare_term_ranks_first() -> None:
    index = BM25Index([["common", "rare"], ["common", "x"], ["common", "y"]])
    ranked = index.search(["rare", "common"])
    assert ranked[0][0] == 0
    assert [doc for doc, _ in ranked] == [0, 1, 2]


def test_bm25_higher_term_frequency_wins() -> None:
    index = BM25Index([["cat", "sat"], ["dog", "barked", "loudly"], ["cat", "cat", "dog"]])
    assert [doc for doc, _ in index.search(["cat"])] == [2, 0]


def test_bm25_omits_zero_scores_and_unknown_terms() -> None:
    index = BM25Index([["a"], ["b"]])
    assert index.search(["zzz"]) == []
    assert [doc for doc, _ in index.search(["a"])] == [0]


def test_bm25_ties_break_by_document_order() -> None:
    index = BM25Index([["a"], ["a"]])
    assert [doc for doc, _ in index.search(["a"])] == [0, 1]


def test_bm25_term_in_every_document_still_scores_positive() -> None:
    index = BM25Index([["a"], ["a"]])
    assert all(score > 0 for _, score in index.search(["a"]))


def test_bm25_respects_k_and_ignores_duplicate_query_terms() -> None:
    index = BM25Index([["c", "1"], ["c", "2"], ["c", "3"]])
    assert len(index.search(["c"], k=2)) == 2
    assert index.search(["c", "c"]) == index.search(["c"])


def test_bm25_empty_index_and_invalid_k() -> None:
    assert BM25Index([]).search(["a"]) == []
    assert BM25Index([["a"]]).search(["a"], k=0) == []


@pytest.mark.parametrize(("k1", "b"), [(-1.0, 0.5), (1.5, -0.1), (1.5, 1.1)])
def test_bm25_rejects_bad_parameters(k1: float, b: float) -> None:
    with pytest.raises(ValueError):
        BM25Index([["a"]], k1=k1, b=b)


# ----------------------------------------------------------------------
# RepoIndex.build
# ----------------------------------------------------------------------


def _write(root: Path, rel: str, content: str | bytes) -> Path:
    path = root / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(content, bytes):
        path.write_bytes(content)
    else:
        path.write_text(content, encoding="utf-8")
    return path


@pytest.fixture()
def repo(tmp_path: Path) -> Path:
    _write(tmp_path, "pkg/router.py", "def route_request(path):\n    return dispatch(path)\n")
    _write(tmp_path, "pkg/db.py", "def connect_database(url):\n    return open_pool(url)\n")
    _write(tmp_path, "pkg/http.py", "def parseHttpRequest(raw):\n    return raw.split()\n")
    _write(tmp_path, "README.md", "# Demo\nA tiny project.\n")
    # Things that must NOT be indexed:
    _write(tmp_path, "node_modules/lib/x.js", "function leaked() {}\n")
    _write(tmp_path, "demo.egg-info/PKG.json", '{"name": "demo"}\n')
    _write(tmp_path, "notes.txt", "not a code extension\n")
    _write(tmp_path, "creds.py", f'AWS = "{FAKE_AWS_KEY}"\n')
    _write(tmp_path, "blob.py", b"\x00\x01\x02binary")
    _write(tmp_path, "big.py", "x = 1\n" * 400)  # > 1 KB
    _write(tmp_path, "gen.py", "# auto-generated, do not edit\nVALUE = 1\n")
    _write(tmp_path, "empty.py", "   \n")
    return tmp_path


def test_build_indexes_only_eligible_files(repo: Path) -> None:
    index = RepoIndex.build(repo, max_file_kb=1)
    assert {c.path for c in index.chunks} == {
        "pkg/router.py",
        "pkg/db.py",
        "pkg/http.py",
        "README.md",
    }
    assert index.stats.files_indexed == 4
    assert index.stats.chunks == len(index.chunks)
    assert index.stats.skipped == {
        "secret": 1,
        "binary": 1,
        "too-large": 1,
        "generated": 1,
        "empty": 1,
    }


def test_secret_bearing_files_are_never_retrievable(repo: Path) -> None:
    index = RepoIndex.build(repo, max_file_kb=1)
    assert not any(FAKE_AWS_KEY in c.text for c in index.chunks)
    assert index.search("AWS") == []


def test_search_finds_relevant_file(repo: Path) -> None:
    index = RepoIndex.build(repo, max_file_kb=1)
    assert index.search("route request")[0].chunk.path == "pkg/router.py"
    assert index.search("connect database")[0].chunk.path == "pkg/db.py"


def test_search_matches_camel_case_via_subwords(repo: Path) -> None:
    index = RepoIndex.build(repo, max_file_kb=1)
    hits = index.search("parse http request")
    assert hits[0].chunk.path == "pkg/http.py"
    assert hits[0].score > 0


def test_search_matches_file_names(repo: Path) -> None:
    index = RepoIndex.build(repo, max_file_kb=1)
    assert index.search("readme")[0].chunk.path == "README.md"


def test_search_empty_or_noise_query_returns_nothing(repo: Path) -> None:
    index = RepoIndex.build(repo, max_file_kb=1)
    assert index.search("") == []
    assert index.search("42 + 7") == []


def test_search_respects_k(repo: Path) -> None:
    index = RepoIndex.build(repo, max_file_kb=1)
    assert len(index.search("def", k=2)) <= 2


def test_build_honours_extensions_and_skip_dirs(repo: Path) -> None:
    only_md = RepoIndex.build(repo, extensions=(".md",))
    assert {c.path for c in only_md.chunks} == {"README.md"}
    no_pkg = RepoIndex.build(repo, max_file_kb=1, skip_dirs=frozenset({"pkg"}))
    assert not any(c.path.startswith("pkg/") for c in no_pkg.chunks)


def test_build_requires_a_directory(tmp_path: Path) -> None:
    with pytest.raises(NotADirectoryError):
        RepoIndex.build(tmp_path / "missing")


def test_build_counts_unreadable_files(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _write(tmp_path, "ok.py", "def fine():\n    pass\n")
    _write(tmp_path, "bad.py", "def broken():\n    pass\n")
    original = Path.read_bytes

    def flaky(self: Path) -> bytes:
        if self.name == "bad.py":
            raise PermissionError("denied")
        return original(self)

    monkeypatch.setattr(Path, "read_bytes", flaky)
    index = RepoIndex.build(tmp_path)
    assert {c.path for c in index.chunks} == {"ok.py"}
    assert index.stats.skipped == {"unreadable": 1}


def test_build_does_not_follow_symlinks(tmp_path: Path) -> None:
    outside = tmp_path / "outside"
    root = tmp_path / "root"
    _write(outside, "secret_notes.py", "def outside_only():\n    pass\n")
    _write(root, "real.py", "def inside():\n    pass\n")
    try:
        os.symlink(outside / "secret_notes.py", root / "link.py")
    except (OSError, NotImplementedError):
        pytest.skip("symlinks not supported on this platform/privilege level")
    index = RepoIndex.build(root)
    assert {c.path for c in index.chunks} == {"real.py"}
    assert index.stats.skipped == {"symlink": 1}
    assert index.search("outside only") == []


# ----------------------------------------------------------------------
# Persistence
# ----------------------------------------------------------------------


def test_save_and_load_roundtrip(repo: Path, tmp_path: Path) -> None:
    index = RepoIndex.build(repo, max_file_kb=1)
    target = tmp_path / "nested" / "dir" / "index.json"
    index.save(target)
    loaded = RepoIndex.load(target)
    assert loaded.chunks == index.chunks
    assert [(h.chunk, h.score) for h in loaded.search("route request")] == [
        (h.chunk, h.score) for h in index.search("route request")
    ]


def test_constructor_derives_stats_from_chunks() -> None:
    chunks = [Chunk("a.py", 1, 1, "x"), Chunk("a.py", 2, 2, "y"), Chunk("b.py", 1, 1, "z")]
    index = RepoIndex(chunks)
    assert index.stats.files_indexed == 2
    assert index.stats.chunks == 3


@pytest.mark.parametrize(
    "payload",
    [
        "not json at all",
        json.dumps({"version": 999, "chunks": []}),
        json.dumps({"chunks": []}),
        json.dumps({"version": 1}),
        json.dumps({"version": 1, "chunks": [{"path": "a.py"}]}),
        json.dumps({"version": 1, "chunks": [{"path": "a", "start_line": "x"}]}),
        json.dumps([1, 2, 3]),
    ],
)
def test_load_rejects_malformed_index(tmp_path: Path, payload: str) -> None:
    target = tmp_path / "index.json"
    target.write_text(payload, encoding="utf-8")
    with pytest.raises(ValueError, match="malformed index file"):
        RepoIndex.load(target)
