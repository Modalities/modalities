from pathlib import Path

from modalities.utils.file_ops import get_file_md5sum


def test_md5sum_identical(tmp_path: Path):
    file1 = tmp_path / "file1.txt"
    file2 = tmp_path / "file2.txt"

    content = b"Hello, world!\n"
    file1.write_bytes(content)
    file2.write_bytes(content)

    assert get_file_md5sum(file1) == get_file_md5sum(file2)


def test_md5sum_different(tmp_path: Path):
    file1 = tmp_path / "file1.txt"
    file2 = tmp_path / "file2.txt"

    file1.write_bytes(b"Hello, world!\n")
    file2.write_bytes(b"Goodbye, world!\n")

    assert get_file_md5sum(file1) != get_file_md5sum(file2)
