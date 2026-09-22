"""Integration smoke test for the `data.dialect_signal` CLI.

Per docs/testing.md, confirms the pipeline wiring (read a corpus TSV ->
analyze -> render) connects end-to-end against a tiny, hand-written
fixture -- never the real private corpus (ADR 0002) -- and, since this
tool's whole purpose is to be safe to paste into a public issue, that its
printed output never contains the raw sentence text it read.
"""

from __future__ import annotations

from data.dialect_signal import main

_GROUP_A_WORDS = ["k'o", "chuwäch", "ronojel", "q'ij", "na'oj", "winaqi'"]
_GROUP_B_WORDS = ["ko", "chuwach", "ronohel", "qih", "naoh", "winaqih"]


def _rotations(words: list[str]) -> list[str]:
    return [" ".join(words[i:] + words[:i]) + "." for i in range(len(words))]


def test_cli_prints_aggregate_stats_and_never_raw_sentence_text(tmp_path, capsys):
    kaqchikel_sentences = _rotations(_GROUP_A_WORDS) + _rotations(_GROUP_B_WORDS)
    corpus_path = tmp_path / "corpus.tsv"
    lines = [f"es sentence {i}\t{cak}" for i, cak in enumerate(kaqchikel_sentences)]
    corpus_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    exit_code = main([str(corpus_path)])

    assert exit_code == 0
    output = capsys.readouterr().out

    assert "Cluster 0" in output
    assert "Cluster 1" in output
    assert f"{len(kaqchikel_sentences)}" in output

    for sentence in kaqchikel_sentences:
        assert sentence not in output
    for i in range(len(kaqchikel_sentences)):
        assert f"es sentence {i}" not in output
