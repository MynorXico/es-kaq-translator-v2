"""Fast smoke tests for the cleaning pipeline wiring, against tiny fixture
files -- never the real corpus, never a real SageMaker job. These confirm
the pieces (I/O, normalize, dedup, length filter, split validation) connect
correctly; corpus quality (BLEU/chrF) is a separate concern in ml/evaluation.
"""

from pathlib import Path

import pytest

from data.corpus_io import read_tsv_pairs
from data.pipeline import clean_corpus_file, validate_split_files
from data.split_integrity import SplitIntegrityError

FIXTURES = Path(__file__).parent.parent / "fixtures"


def test_clean_corpus_file_dedupes_normalizes_and_filters(tmp_path):
    output_path = tmp_path / "cleaned.tsv"

    result = clean_corpus_file(str(FIXTURES / "sample_raw_pairs.tsv"), str(output_path))

    # "Buenos días / Utz awäch" appears twice in the fixture -> deduped.
    # "Gracias" has extra whitespace -> normalized.
    # "Solo" has an empty target -> dropped by length filtering.
    # "Sí / Una oración ..." has an extreme length ratio -> dropped.
    # "Adiós / Chabäl" is a normal pair -> kept.
    assert result == [
        ("Buenos días", "Utz awäch"),
        ("Gracias", "Matyox"),
        ("Adiós", "Chabäl"),
    ]

    written = read_tsv_pairs(str(output_path))
    assert written == result


def test_validate_split_files_raises_on_leaked_pair():
    with pytest.raises(SplitIntegrityError):
        validate_split_files(
            str(FIXTURES / "sample_train.tsv"),
            str(FIXTURES / "sample_val_with_leak.tsv"),
        )


def test_validate_split_files_passes_on_clean_split():
    report = validate_split_files(
        str(FIXTURES / "sample_train.tsv"),
        str(FIXTURES / "sample_val_clean.tsv"),
    )

    assert report.exact_pair_overlap == []
