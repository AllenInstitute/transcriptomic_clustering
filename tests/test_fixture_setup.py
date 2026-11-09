import pytest

from .conftest import check_fixture_payloads


def test_unresolved_lfs_pointer_has_download_instructions(tmp_path):
    (tmp_path / "fixture.h5ad").write_text(
        "version https://git-lfs.github.com/spec/v1\noid sha256:abc\nsize 10\n"
    )
    with pytest.raises(pytest.UsageError, match="git lfs pull"):
        check_fixture_payloads(tmp_path)


def test_missing_fixture_directory_has_download_instructions(tmp_path):
    with pytest.raises(pytest.UsageError, match="git lfs pull"):
        check_fixture_payloads(tmp_path / "missing")


def test_fixture_payload_is_accepted(tmp_path):
    (tmp_path / "fixture.h5ad").write_bytes(b"\x89HDF\r\n\x1a\n")
    check_fixture_payloads(tmp_path)
