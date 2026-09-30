"""Integration smoke test for the `data.glued_word_signal` CLI.

Per docs/testing.md, confirms the pipeline wiring (read a corpus TSV ->
scan -> render) connects end-to-end against a tiny, hand-written fixture --
never the real private corpus (ADR 0002) -- and that its printed output
never contains raw sentence text.
"""

from __future__ import annotations

from data.glued_word_signal import main


def test_cli_prints_aggregate_stats_and_never_raw_sentence_text(tmp_path, capsys):
    lines = [
        "Esto es dedios verdad.\tRi tzij.",
        "Una frase normal sin patrones.\tJun tzij chik.",
    ]
    corpus_path = tmp_path / "corpus.tsv"
    corpus_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    exit_code = main([str(corpus_path)])

    assert exit_code == 0
    output = capsys.readouterr().out

    assert "Sentences scanned: 2" in output
    assert "'dedios': 1 occurrence" in output
    assert "Esto es dedios verdad." not in output
    assert "Ri tzij." not in output
