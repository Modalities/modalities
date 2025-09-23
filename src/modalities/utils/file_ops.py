import hashlib
from pathlib import Path


def get_file_md5sum(path: Path, chunk_size: int = 8192) -> str:
    """Calculate the MD5 checksum of a file.
    Args:
        path (Path): Path to the file.
        chunk_size (int): Size of chunks to read the file. Default is 8192 bytes.
    Returns:
        str: The MD5 checksum of the file as a hexadecimal string.
    """
    hash_md5 = hashlib.md5()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()
