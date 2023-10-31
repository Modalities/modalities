import pickle
from pathlib import Path

from .create_index import IndexGenerator


def test_index_creation(tmpdir):
    larger_test_data_path = Path("/home/shared/openwebtext/head20000_openwebtext2_en.jsonl")
    indexer = IndexGenerator(larger_test_data_path)
    dummy_dst_path = Path(tmpdir, "index.pkl")
    indexer.run(dummy_dst_path)

    index = pickle.loads(dummy_dst_path.read_bytes())
    assert index[:5] == [(0, 477), (477, 3798), (4275, 1731), (6006, 11181), (17187, 4887)]
