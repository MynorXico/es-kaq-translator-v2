"""Integration smoke test for the `data.alignment_signal` CLI.

Per docs/testing.md, confirms the pipeline wiring (read a corpus TSV ->
sample -> embed -> analyze -> render) connects end-to-end against a tiny,
hand-written fixture -- never the real private corpus (ADR 0002), and
never a real embedding model (`_embed_sentences` is monkeypatched to a
fast, deterministic fake, the same "duck-type and fixture" convention
`tests/integration/test_train_pipeline.py` uses for heavy model calls).
"""

from __future__ import annotations

import numpy as np

import data.alignment_signal as alignment_signal_module
from data.alignment_signal import main


def test_cli_prints_aggregate_stats_and_never_raw_sentence_text(tmp_path, capsys, monkeypatch):
    lines = [
        "hola mundo\tutz sq'ij",
        "buenos dias\tla utz awäch",
        "como estas\tla utz awäch rat",
        "hasta luego\tchabe",
    ]
    corpus_path = tmp_path / "corpus.tsv"
    corpus_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    def fake_embed_sentences(texts, model_name, batch_size=32):
        # Deterministic fake: embed by (length, first-char-ordinal) so
        # real/shuffled comparisons are exercised without any real model.
        return np.array([[float(len(t)), float(ord(t[0]))] for t in texts])

    monkeypatch.setattr(alignment_signal_module, "_embed_sentences", fake_embed_sentences)

    exit_code = main([str(corpus_path), "--sample-size", "4", "--seed", "1"])

    assert exit_code == 0
    output = capsys.readouterr().out

    assert "Pairs analyzed: 4" in output
    assert "AUC" in output
    for line in lines:
        es, cak = line.split("\t")
        assert es not in output
        assert cak not in output
