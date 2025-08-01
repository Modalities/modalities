import hashlib
from pathlib import Path


def get_file_md5sum(path: Path, chunk_size: int = 8192) -> str:
    hash_md5 = hashlib.md5()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()
