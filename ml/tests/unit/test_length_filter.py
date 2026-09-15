from data.length_filter import LengthFilterConfig, filter_pairs_by_length, is_valid_length


def test_default_config_accepts_a_normal_pair():
    assert is_valid_length("Buenos días", "Utz awäch") is True


def test_rejects_pair_with_empty_source():
    assert is_valid_length("", "Utz awäch") is False


def test_rejects_pair_with_empty_target():
    assert is_valid_length("Buenos días", "") is False


def test_rejects_pair_with_only_whitespace():
    assert is_valid_length("   ", "Utz awäch") is False


def test_rejects_pair_shorter_than_min_length():
    config = LengthFilterConfig(min_length=3)
    assert is_valid_length("Hi", "Ab", config=config) is False


def test_rejects_pair_longer_than_max_length():
    config = LengthFilterConfig(max_length=20)
    long_sentence = "a" * 21
    assert is_valid_length(long_sentence, "short", config=config) is False


def test_rejects_pair_with_extreme_length_ratio():
    # One side wildly longer than the other suggests misalignment, not a
    # genuinely long/short pair of the same content.
    config = LengthFilterConfig(max_ratio=3.0)
    assert is_valid_length("short", "a very very very much longer sentence than that", config=config) is False


def test_accepts_pair_within_ratio_threshold():
    config = LengthFilterConfig(max_ratio=3.0)
    assert is_valid_length("short one", "a bit longer one here", config=config) is True


def test_filter_pairs_by_length_drops_only_invalid_pairs():
    pairs = [
        ("Buenos días", "Utz awäch"),
        ("", "Matyox"),
        ("Hola", ""),
    ]

    result = filter_pairs_by_length(pairs)

    assert result == [("Buenos días", "Utz awäch")]


def test_filter_pairs_by_length_uses_provided_config():
    config = LengthFilterConfig(min_length=3)
    pairs = [
        ("Buenos días", "Utz awäch"),
        ("Hi", "Ab"),
    ]

    result = filter_pairs_by_length(pairs, config=config)

    assert result == [("Buenos días", "Utz awäch")]
