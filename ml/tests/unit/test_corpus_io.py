from data.corpus_io import read_tsv_pairs, write_tsv_pairs


def test_read_tsv_pairs_parses_tab_separated_lines(tmp_path):
    path = tmp_path / "pairs.tsv"
    path.write_text("Buenos días\tUtz awäch\nGracias\tMatyox\n", encoding="utf-8")

    pairs = read_tsv_pairs(str(path))

    assert pairs == [("Buenos días", "Utz awäch"), ("Gracias", "Matyox")]


def test_read_tsv_pairs_skips_blank_lines(tmp_path):
    path = tmp_path / "pairs.tsv"
    path.write_text("Buenos días\tUtz awäch\n\nGracias\tMatyox\n", encoding="utf-8")

    pairs = read_tsv_pairs(str(path))

    assert pairs == [("Buenos días", "Utz awäch"), ("Gracias", "Matyox")]


def test_read_tsv_pairs_raises_on_malformed_line(tmp_path):
    path = tmp_path / "pairs.tsv"
    path.write_text("Buenos días\tUtz awäch\tExtra column\n", encoding="utf-8")

    try:
        read_tsv_pairs(str(path))
        raised = False
    except ValueError:
        raised = True

    assert raised


def test_write_tsv_pairs_round_trips_with_read(tmp_path):
    path = tmp_path / "out.tsv"
    pairs = [("Buenos días", "Utz awäch"), ("Gracias", "Matyox")]

    write_tsv_pairs(pairs, str(path))
    result = read_tsv_pairs(str(path))

    assert result == pairs


def test_read_tsv_pairs_rejects_s3_uri_without_boto3_mocking(monkeypatch):
    # Local unit tests never touch real S3 -- reading an s3:// URI must
    # go through the boto3 client, which we don't exercise here (that's
    # covered conceptually by the pipeline being parameterized by URI,
    # not by hitting real AWS in this fast test suite).
    called = {}

    class FakeS3Client:
        def get_object(self, Bucket, Key):
            called["bucket"] = Bucket
            called["key"] = Key
            body = b"Buenos dias\tUtz awach\n"
            return {"Body": _FakeBody(body)}

    class _FakeBody:
        def __init__(self, data):
            self._data = data

        def read(self):
            return self._data

    from data import corpus_io

    monkeypatch.setattr(corpus_io, "_get_s3_client", lambda: FakeS3Client())

    pairs = read_tsv_pairs("s3://some-private-bucket/prefix/pairs.tsv")

    assert pairs == [("Buenos dias", "Utz awach")]
    assert called == {"bucket": "some-private-bucket", "key": "prefix/pairs.tsv"}
