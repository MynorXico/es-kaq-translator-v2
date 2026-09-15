import pytest

from data.split_integrity import SplitIntegrityError, validate_split_integrity


def test_passes_when_no_overlap_at_all():
    train = [("Buenos días", "Utz awäch"), ("Gracias", "Matyox")]
    val = [("Adiós", "Chabäl")]

    report = validate_split_integrity(train, val)

    assert report.exact_pair_overlap == []
    assert report.source_overlap == []
    assert report.target_overlap == []


def test_raises_on_exact_pair_overlap_by_default():
    train = [("Buenos días", "Utz awäch")]
    val = [("Buenos días", "Utz awäch")]

    with pytest.raises(SplitIntegrityError):
        validate_split_integrity(train, val)


def test_report_lists_the_overlapping_pair():
    train = [("Buenos días", "Utz awäch"), ("Gracias", "Matyox")]
    val = [("Buenos días", "Utz awäch")]

    with pytest.raises(SplitIntegrityError) as exc_info:
        validate_split_integrity(train, val)

    assert exc_info.value.report.exact_pair_overlap == [("Buenos días", "Utz awäch")]


def test_can_check_without_raising():
    train = [("Buenos días", "Utz awäch")]
    val = [("Buenos días", "Utz awäch")]

    report = validate_split_integrity(train, val, raise_on_overlap=False)

    assert report.exact_pair_overlap == [("Buenos días", "Utz awäch")]


def test_flags_source_side_overlap_as_a_separate_warning_from_exact_pairs():
    # Same Spanish sentence in both splits with a *different* Kaqchikel
    # translation isn't an exact-pair duplicate, but it's still partial
    # leakage worth flagging.
    train = [("Gracias", "Matyox")]
    val = [("Gracias", "Tyox")]

    report = validate_split_integrity(train, val, raise_on_overlap=False)

    assert report.exact_pair_overlap == []
    assert report.source_overlap == ["Gracias"]


def test_flags_target_side_overlap_as_a_separate_warning_from_exact_pairs():
    train = [("Hola", "Utz awäch")]
    val = [("Buenos días", "Utz awäch")]

    report = validate_split_integrity(train, val, raise_on_overlap=False)

    assert report.exact_pair_overlap == []
    assert report.target_overlap == ["Utz awäch"]
