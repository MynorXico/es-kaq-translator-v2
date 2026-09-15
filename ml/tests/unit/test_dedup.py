from data.dedup import deduplicate_pairs


def test_removes_exact_duplicate_pairs():
    pairs = [
        ("Buenos días", "Utz awäch"),
        ("Buenos días", "Utz awäch"),
        ("Gracias", "Matyox"),
    ]

    result = deduplicate_pairs(pairs)

    assert result == [
        ("Buenos días", "Utz awäch"),
        ("Gracias", "Matyox"),
    ]


def test_preserves_first_occurrence_order():
    pairs = [
        ("Hola", "Utz awäch"),
        ("Gracias", "Matyox"),
        ("Hola", "Utz awäch"),
        ("Adiós", "Chabäl"),
    ]

    result = deduplicate_pairs(pairs)

    assert result == [
        ("Hola", "Utz awäch"),
        ("Gracias", "Matyox"),
        ("Adiós", "Chabäl"),
    ]


def test_does_not_treat_different_pairs_with_same_source_as_duplicates():
    # Same Spanish source with two different Kaqchikel translations is not
    # a duplicate to drop -- only exact (source, target) matches are.
    pairs = [
        ("Gracias", "Matyox"),
        ("Gracias", "Tyox"),
    ]

    result = deduplicate_pairs(pairs)

    assert result == pairs


def test_empty_input_returns_empty_list():
    assert deduplicate_pairs([]) == []
